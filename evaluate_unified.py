"""
evaluate_unified.py — Fair ADE/FDE comparison between S&F and CPA-GRN.

Both models are evaluated on the SAME test set using a CORRECT geodesic
denormalisation (per-axis: LAT uses lat bounds, LON uses lon bounds),
avoiding the scale_values bug in geographic_utils.py (which collapses
dlon == dlat because the 1°x1° bounding box made the bug numerically
invisible during training but still corrupts the *reported* ADE/FDE).

Reports ADE and FDE in:
  - degrees
  - nautical miles (1° ≈ 60 nm, valid near 32°N)

Run:
    python evaluate_unified.py --cpagrn_tag CPA_GRN_v4 \
        --sf_hidden_size 6 --sf_criterion dist_error --split test
"""

from __future__ import annotations
import os
import argparse
import math

import torch
import numpy as np

import models
from data import load_data, collate_function
from train_utils import get_dirs, predict
from utils import seed_everything
from geographic_utils import min_lat, max_lat, min_lon, max_lon  # radians

from model_cpagrn import CPAGRN, sample_trajectories

seed_everything(100)

NM_PER_DEGREE = 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Correct per-axis denormalisation (NOT the buggy scale_values)
# ─────────────────────────────────────────────────────────────────────────────

def denorm_to_degrees(lat_norm, lon_norm):
    lat_rad = lat_norm * (max_lat - min_lat) + min_lat
    lon_rad = lon_norm * (max_lon - min_lon) + min_lon
    return lat_rad * (180.0 / math.pi), lon_rad * (180.0 / math.pi)


def l2_degrees(plat, plon, tlat, tlon):
    return np.sqrt((plat - tlat) ** 2 + (plon - tlon) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split_data', action='store_true', default=False)
    p.add_argument('--zone', type=int, default=11)
    p.add_argument('--sequence_length', type=int, default=5)
    p.add_argument('--prediction_length', type=int, default=5)
    p.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--gpu_num', type=int, default=0)

    # S&F model args
    p.add_argument('--sf_model', type=str, default='spatial_temporal_model')
    p.add_argument('--sf_feature_size', type=int, default=2)
    p.add_argument('--sf_hidden_size', type=int, default=6)
    p.add_argument('--sf_delta_bearing', type=float, default=60)
    p.add_argument('--sf_delta_heading', type=float, default=60)
    p.add_argument('--sf_param_domain', type=float, default=2)
    p.add_argument('--sf_criterion', type=str, default='dist_error')

    # CPA-GRN args
    p.add_argument('--cpagrn_tag', type=str, default='CPA_GRN_v4')
    p.add_argument('--n_samples', type=int, default=20)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# S&F evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sf(args, test_ds, device):
    print('\n--- Evaluating S&F (spatial_temporal_model) ---')

    net = getattr(models, args.sf_model)(
        args.sequence_length, args.prediction_length, args.sf_feature_size,
        args.sf_hidden_size, args.sf_delta_bearing, args.sf_delta_heading,
        args.sf_param_domain
    ).float().to(device)

    # Reconstruct checkpoint path (same convention as main.py)
    net_dir = args.sf_model + '/'
    netfile = net_dir + 'hsz_' + str(args.sf_hidden_size)
    if hasattr(net, 'spatialAttention'):
        netfile += '_' + args.sf_criterion
    netfile += '.pt'

    assert os.path.exists(netfile), f'S&F checkpoint not found: {netfile}'
    net.load_state_dict(torch.load(netfile, map_location=device, weights_only=False))
    net.eval()

    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_function()
    )

    T = args.prediction_length
    ade_per_horizon = [[] for _ in range(T)]
    fde_list = []

    with torch.no_grad():
        for batch in loader:
            pred, target, sequence, _, vessels = predict(batch, net)
            # pred, target: [B, N, T, 2]  (normalised LAT, LON)
            pred   = pred.cpu().numpy()
            target = target.cpu().numpy()

            if pred.ndim == 3:   # single-sample batch may squeeze B dim
                pred   = pred[np.newaxis, ...]
                target = target[np.newaxis, ...]

            plat, plon = denorm_to_degrees(pred[..., 0],   pred[..., 1])
            tlat, tlon = denorm_to_degrees(target[..., 0], target[..., 1])

            err = l2_degrees(plat, plon, tlat, tlon)  # [B, N, T]

            B, N, _ = err.shape
            for b in range(B):
                for n in range(N):
                    if np.allclose(target[b, n], 0):
                        continue  # padded vessel
                    for t in range(T):
                        ade_per_horizon[t].append(err[b, n, t])
                    fde_list.append(err[b, n, -1])

    return ade_per_horizon, fde_list


# ─────────────────────────────────────────────────────────────────────────────
# CPA-GRN evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cpagrn(args, test_ds, device):
    print('\n--- Evaluating CPA-GRN ---')

    ckpt_path = os.path.join('checkpoints', args.cpagrn_tag, 'val_best.pth')
    assert os.path.exists(ckpt_path), f'CPA-GRN checkpoint not found: {ckpt_path}'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})

    model = CPAGRN(
        feature_size = saved.get('feature_size', 4),
        d_model      = saved.get('d_model', 64),
        gru_layers   = saved.get('gru_layers', 2),
        K            = saved.get('K', 3),
        pred_len     = saved.get('prediction_length', 5),
        dropout      = 0.0,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    K = saved.get('K', 3)

    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_function()
    )

    T = args.prediction_length
    ade_per_horizon = [[] for _ in range(T)]
    fde_list = []

    with torch.no_grad():
        for batch in loader:
            (ip, op, _d, _b, _h, _ipm, _opm, vessel_count) = [t.to(device) for t in batch]

            N = ip.shape[1]
            v_idx = torch.arange(N, device=device).unsqueeze(0)
            vessel_mask = v_idx < vessel_count.long().unsqueeze(1)

            obs_seq     = ip.permute(0, 2, 1, 3)
            last_obs    = ip[:, :, -1, :2]
            target_disp = op[..., :2] - last_obs.unsqueeze(2)

            gmm_params, _ = model(obs_seq, ip_mask=vessel_mask)
            samples = sample_trajectories(gmm_params, n_samples=args.n_samples, K=K)

            last_obs_np = last_obs.cpu().numpy()
            target_np   = target_disp.cpu().numpy()
            samples_np  = samples.cpu().numpy()
            mask_np     = vessel_mask.cpu().numpy()

            B = mask_np.shape[0]
            pred_abs   = samples_np + last_obs_np[np.newaxis, :, :, np.newaxis, :]
            target_abs = target_np  + last_obs_np[:, :, np.newaxis, :]

            plat, plon = denorm_to_degrees(pred_abs[..., 0],   pred_abs[..., 1])
            tlat, tlon = denorm_to_degrees(target_abs[..., 0], target_abs[..., 1])

            for b in range(B):
                for n in range(N):
                    if not mask_np[b, n]:
                        continue
                    err = l2_degrees(plat[:, b, n, :], plon[:, b, n, :],
                                      tlat[b, n, :],    tlon[b, n, :])
                    best = err.mean(axis=1).argmin()
                    for t in range(T):
                        ade_per_horizon[t].append(err[best, t])
                    fde_list.append(err[best, -1])

    return ade_per_horizon, fde_list


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def report(name, ade_per_horizon, fde_list):
    T = len(ade_per_horizon)
    ade_h = [np.mean(h) for h in ade_per_horizon]
    overall = np.mean(ade_h)
    fde = np.mean(fde_list)

    print(f'\n{"="*55}')
    print(f'  {name}')
    print('='*55)
    for t, a in enumerate(ade_h, 1):
        print(f'  ADE {t}min : {a:.6f}°  ({a*NM_PER_DEGREE:.5f} nm)')
    print('-'*55)
    print(f'  Overall ADE : {overall:.6f}°  ({overall*NM_PER_DEGREE:.5f} nm)')
    print(f'  FDE         : {fde:.6f}°  ({fde*NM_PER_DEGREE:.5f} nm)')
    print('='*55)
    return overall, fde


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data_dir = f'data/{args.zone:02d}/'

    # S&F uses feature_size=2 datasets; CPA-GRN uses feature_size=4 datasets.
    # Load both separately with the correct feature_size.
    args_sf = argparse.Namespace(**vars(args))
    args_sf.feature_size = args.sf_feature_size
    _, val_sf, test_sf = load_data(data_dir, args_sf)

    args_cpa = argparse.Namespace(**vars(args))
    args_cpa.feature_size = 4
    _, val_cpa, test_cpa = load_data(data_dir, args_cpa)

    eval_sf  = test_sf  if args.split == 'test' else val_sf
    eval_cpa = test_cpa if args.split == 'test' else val_cpa

    sf_ade, sf_fde   = evaluate_sf(args, eval_sf, device)
    cpa_ade, cpa_fde = evaluate_cpagrn(args, eval_cpa, device)

    sf_overall,  sf_f   = report('S&F (spatial_temporal_model)', sf_ade, sf_fde)
    cpa_overall, cpa_f  = report(f'CPA-GRN ({args.cpagrn_tag})', cpa_ade, cpa_fde)

    print(f'\n{"="*55}')
    print('  COMPARISON (same dataset, same evaluation protocol)')
    print('='*55)
    print(f'  S&F      ADE : {sf_overall:.6f}°  ({sf_overall*NM_PER_DEGREE:.5f} nm)')
    print(f'  CPA-GRN  ADE : {cpa_overall:.6f}°  ({cpa_overall*NM_PER_DEGREE:.5f} nm)')
    if sf_overall > 0:
        improvement = (sf_overall - cpa_overall) / sf_overall * 100
        print(f'  CPA-GRN vs S&F: {improvement:+.1f}%')
    print('='*55)


if __name__ == '__main__':
    main()
