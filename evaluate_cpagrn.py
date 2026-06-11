"""
evaluate_cpagrn.py — Evaluation for CPA-GRN.

Reports:
  - minADE-20 and FDE in degrees
  - ADE per prediction horizon (1min … pred_len min)
  - ADE in nautical miles (for comparison with S&F)

Run:
    python evaluate_cpagrn.py --tag CPA_GRN_v1 --split test --gpu_num 0
"""

from __future__ import annotations
import os
import argparse
import math

import torch
import numpy as np

from data import load_data, collate_function
from utils import seed_everything
from geographic_utils import min_lat, max_lat, min_lon, max_lon  # radians
from model_cpagrn import CPAGRN, sample_trajectories

seed_everything(100)

NM_PER_DEGREE = 60.0  # approximation valid near 32°N


# ─────────────────────────────────────────────────────────────────────────────
# Denormalisation
# ─────────────────────────────────────────────────────────────────────────────

def denorm_to_degrees(lat_norm: np.ndarray, lon_norm: np.ndarray):
    """
    Reverse data.py normalisation.
    data.py: converts degrees→radians, then min-max normalises.
    min_lat/max_lat/min_lon/max_lon from geographic_utils are in radians.
    """
    lat_rad = lat_norm * (max_lat - min_lat) + min_lat
    lon_rad = lon_norm * (max_lon - min_lon) + min_lon
    return lat_rad * (180.0 / math.pi), lon_rad * (180.0 / math.pi)


def l2_degrees(pred_lat, pred_lon, true_lat, true_lon) -> np.ndarray:
    return np.sqrt((pred_lat - true_lat) ** 2 + (pred_lon - true_lon) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Evaluate CPA-GRN')
    p.add_argument('--split_data',       action='store_true', default=False)
    p.add_argument('--tag',              type=str, default='CPA_GRN_v1')
    p.add_argument('--split',            type=str, default='test',
                   choices=['val', 'test'])
    p.add_argument('--n_samples',        type=int, default=20)
    p.add_argument('--zone',             type=int, default=11)
    p.add_argument('--sequence_length',  type=int, default=5)
    p.add_argument('--prediction_length',type=int, default=5)
    p.add_argument('--feature_size',     type=int, default=4)
    p.add_argument('--batch_size',       type=int, default=32)
    p.add_argument('--gpu_num',          type=int, default=0)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Load checkpoint ───────────────────────────────────────────────
    ckpt_path = os.path.join('checkpoints', args.tag, 'val_best.pth')
    assert os.path.exists(ckpt_path), f'Checkpoint not found: {ckpt_path}'
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved     = ckpt.get('args', {})
    print(f'Loaded epoch {ckpt["epoch"]}  val_loss={ckpt.get("val_loss", "?"):.6f}')

    # ── Rebuild model ─────────────────────────────────────────────────
    model = CPAGRN(
        feature_size = saved.get('feature_size',       4),
        d_model      = saved.get('d_model',            64),
        gru_layers   = saved.get('gru_layers',         2),
        K            = saved.get('K',                  3),
        pred_len     = saved.get('prediction_length',  5),
        dropout      = 0.0,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    K = saved.get('K', 3)

    # ── Data ──────────────────────────────────────────────────────────
    data_dir = f'data/{args.zone:02d}/'
    assert os.path.isdir(data_dir), f'Missing: {data_dir}'

    train_ds, val_ds, test_ds = load_data(data_dir, args)
    eval_ds = test_ds if args.split == 'test' else val_ds

    loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_function(), num_workers=0
    )
    print(f'Evaluating on {args.split} set  ({len(eval_ds)} samples)')

    # ── Evaluation loop ───────────────────────────────────────────────
    T = args.prediction_length
    ade_per_horizon = [[] for _ in range(T)]
    fde_list        = []

    with torch.no_grad():
        for batch in loader:
            # Unpack — same approach as training
            (ip, op, dist_matrix, bear_matrix, hdg_matrix,
             _ip_mask, _op_mask, vessel_count) = [t.to(device) for t in batch]

            # Reliable vessel mask from vessel_count
            N     = ip.shape[1]
            v_idx = torch.arange(N, device=device).unsqueeze(0)
            vessel_mask = v_idx < vessel_count.long().unsqueeze(1)  # [B, N]

            # Model input
            obs_seq     = ip.permute(0, 2, 1, 3)          # [B, T_obs, N, F]
            last_obs    = ip[:, :, -1, :2]                 # [B, N, 2]
            target_disp = op[..., :2] - last_obs.unsqueeze(2)  # [B, N, T, 2]

            # Forward + sample
            gmm_params, _ = model(obs_seq, ip_mask=vessel_mask)
            traj_samples  = sample_trajectories(
                gmm_params, n_samples=args.n_samples, K=K
            )  # [n_samples, B, N, T, 2]

            # To numpy
            last_obs_np = last_obs.cpu().numpy()          # [B, N, 2]
            target_np   = target_disp.cpu().numpy()        # [B, N, T, 2]
            samples_np  = traj_samples.cpu().numpy()       # [S, B, N, T, 2]
            mask_np     = vessel_mask.cpu().numpy()        # [B, N]

            B = mask_np.shape[0]

            # Absolute positions
            pred_abs   = samples_np + last_obs_np[np.newaxis, :, :, np.newaxis, :]
            target_abs = target_np  + last_obs_np[:, :, np.newaxis, :]

            # Denormalise to degrees
            pred_lat,  pred_lon  = denorm_to_degrees(pred_abs[..., 0],   pred_abs[..., 1])
            true_lat,  true_lon  = denorm_to_degrees(target_abs[..., 0], target_abs[..., 1])

            # minADE-20 per vessel per horizon
            for b in range(B):
                for n in range(N):
                    if not mask_np[b, n]:
                        continue

                    err  = l2_degrees(
                        pred_lat[:, b, n, :], pred_lon[:, b, n, :],
                        true_lat[b, n, :],    true_lon[b, n, :],
                    )  # [n_samples, T]

                    best = err.mean(axis=1).argmin()
                    for t in range(T):
                        ade_per_horizon[t].append(err[best, t])
                    fde_list.append(err[best, -1])

    # ── Report ────────────────────────────────────────────────────────
    ade_horizons = [np.mean(h) for h in ade_per_horizon]
    overall_ade  = np.mean(ade_horizons)
    fde          = np.mean(fde_list)

    print('\n' + '=' * 55)
    print(f'  CPA-GRN | Tag: {args.tag} | Split: {args.split}')
    print('=' * 55)
    for t, ade_t in enumerate(ade_horizons, 1):
        print(f'  ADE {t}min : {ade_t:.6f}°  ({ade_t*NM_PER_DEGREE:.5f} nm)')
    print('-' * 55)
    print(f'  minADE-20 : {overall_ade:.6f}°  ({overall_ade*NM_PER_DEGREE:.5f} nm)')
    print(f'  FDE       : {fde:.6f}°  ({fde*NM_PER_DEGREE:.5f} nm)')
    print('=' * 55)
    print(f'\n  S&F paper ADE : 0.03314 nm')
    print(f'  Ours      ADE : {overall_ade*NM_PER_DEGREE:.5f} nm')
    delta = (0.03314 - overall_ade * NM_PER_DEGREE) / 0.03314 * 100
    print(f'  vs S&F        : {delta:+.1f}%')
    print('=' * 55)


if __name__ == '__main__':
    main()