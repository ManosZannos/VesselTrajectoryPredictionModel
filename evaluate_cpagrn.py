"""
evaluate_cpagrn.py — Evaluation for CPA-GRN.

Reports:
  - minADE-20 and FDE in degrees  (our primary metric)
  - ADE per prediction horizon    (1min, 2min, …, 5min)
  - ADE in nautical miles         (for direct comparison with S&F's published numbers)

Conversion: 1° latitude ≈ 60 nautical miles
  → ADE_nm = ADE_degrees * 60  (approximation valid for San Diego lat ~32°)

Run:
    python evaluate_cpagrn.py --tag CPA_GRN_v1 --split test
"""

from __future__ import annotations
import os
import sys
import argparse
import math

import torch
import numpy as np
from torch.utils.data import DataLoader

from data import load_data
from train_utils import get_batch
from utils import seed_everything
from geographic_utils import min_lat, max_lat, min_lon, max_lon  # radians bounds

from model_cpagrn import CPAGRN, sample_trajectories

seed_everything(100)

NM_PER_DEGREE = 60.0   # approximate (valid near 32°N)


# ─────────────────────────────────────────────────────────────────────────────
# Denormalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def denorm_to_degrees(lat_norm: np.ndarray, lon_norm: np.ndarray):
    """
    Convert normalised [0,1] positions back to degrees.
    data.py normalises via radians:  norm = (rad - min_rad) / (max_rad - min_rad)
    So: rad = norm*(max_rad - min_rad) + min_rad  → degrees = rad*(180/pi)
    """
    lat_rad = lat_norm * (max_lat - min_lat) + min_lat   # min_lat/max_lat in radians
    lon_rad = lon_norm * (max_lon - min_lon) + min_lon
    lat_deg = lat_rad * (180.0 / math.pi)
    lon_deg = lon_rad * (180.0 / math.pi)
    return lat_deg, lon_deg


def l2_degrees(pred_lat, pred_lon, true_lat, true_lon) -> np.ndarray:
    """Euclidean distance in degree space. Shape: [...] → [...]"""
    return np.sqrt((pred_lat - true_lat) ** 2 + (pred_lon - true_lon) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Evaluate CPA-GRN')

    p.add_argument('--tag',              type=str, default='CPA_GRN_v1')
    p.add_argument('--split',            type=str, default='test',
                   choices=['val', 'test'])
    p.add_argument('--n_samples',        type=int, default=20)

    # Must match training args
    p.add_argument('--zone',             type=int, default=11)
    p.add_argument('--sequence_length',  type=int, default=5)
    p.add_argument('--prediction_length',type=int, default=5)
    p.add_argument('--feature_size',     type=int, default=4)
    p.add_argument('--batch_size',       type=int, default=32)
    p.add_argument('--gpu_num',          type=int, default=0)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Load checkpoint ───────────────────────────────────────────────
    ckpt_path = os.path.join('checkpoints', args.tag, 'val_best.pth')
    assert os.path.exists(ckpt_path), f'Checkpoint not found: {ckpt_path}'
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get('args', {})
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]} '
          f'(val loss: {ckpt.get("val_loss", "n/a"):.6f})')

    # ── Rebuild model from saved args ─────────────────────────────────
    model = CPAGRN(
        feature_size = saved_args.get('feature_size',  4),
        d_model      = saved_args.get('d_model',       64),
        gru_layers   = saved_args.get('gru_layers',    2),
        K            = saved_args.get('K',             3),
        pred_len     = saved_args.get('prediction_length', 5),
        dropout      = 0.0,   # no dropout at evaluation
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # ── Data ──────────────────────────────────────────────────────────
    data_dir = f'data/{args.zone:02d}/'
    assert os.path.isdir(data_dir), f'Data directory not found: {data_dir}. Run grid.py first.'
    train_ds, val_ds, test_ds = load_data(args, data_dir)
    eval_ds = test_ds if args.split == 'test' else val_ds
    loader  = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=2)
    print(f'Evaluating on {args.split} set ({len(eval_ds)} samples)')

    # ── Accumulate per-horizon errors ─────────────────────────────────
    T = args.prediction_length
    ade_per_horizon = [[] for _ in range(T)]   # list of scalars per horizon
    fde_list        = []

    with torch.no_grad():
        for batch in loader:
            (sequence, target,
             _d, _b, _h,
             ip_mask, op_mask, _v) = get_batch(batch)

            sequence = sequence.to(device)
            target   = target.to(device)
            ip_mask  = ip_mask.to(device)
            op_mask  = op_mask.to(device)

            # ── Displacement target ───────────────────────────────────
            last_obs    = sequence[:, -1, :, :2].unsqueeze(2)   # [B, N, 1, 2]
            target_BNTF = target.permute(0, 2, 1, 3)            # [B, N, T, 2]
            target_disp = target_BNTF - last_obs                 # [B, N, T, 2]

            # ── Sample 20 trajectories ────────────────────────────────
            gmm_params, _ = model(sequence, ip_mask=ip_mask, op_mask=op_mask)
            # gmm_params: [B, N, T, K*6]

            K = saved_args.get('K', 3)
            traj_samples = sample_trajectories(
                gmm_params, n_samples=args.n_samples, K=K
            )
            # traj_samples: [n_samples, B, N, T, 2]  — displacement predictions

            # ── Convert to absolute normalised positions ───────────────
            last_obs_np = last_obs.squeeze(2).cpu().numpy()     # [B, N, 2]
            target_np   = target_disp.cpu().numpy()             # [B, N, T, 2]
            samples_np  = traj_samples.cpu().numpy()            # [S, B, N, T, 2]
            mask_np     = op_mask.cpu().numpy()                 # [B, N]

            B, N = mask_np.shape

            # Absolute predictions: pred_abs = disp + last_obs
            pred_abs = samples_np + last_obs_np[np.newaxis, :, :, np.newaxis, :]
            # [S, B, N, T, 2]

            target_abs = target_np + last_obs_np[:, :, np.newaxis, :]
            # [B, N, T, 2]

            # ── Denormalise to degrees ────────────────────────────────
            # pred_abs[..., 0] = LAT_norm, pred_abs[..., 1] = LON_norm
            pred_lat_deg, pred_lon_deg = denorm_to_degrees(
                pred_abs[..., 0], pred_abs[..., 1]
            )
            true_lat_deg, true_lon_deg = denorm_to_degrees(
                target_abs[..., 0], target_abs[..., 1]
            )

            # ── Per-vessel, per-horizon ADE → minADE-20 ───────────────
            for b in range(B):
                for n in range(N):
                    if not mask_np[b, n]:
                        continue   # skip padded vessels

                    # errors: [n_samples, T]
                    err = l2_degrees(
                        pred_lat_deg[:, b, n, :],
                        pred_lon_deg[:, b, n, :],
                        true_lat_deg[b, n, :],
                        true_lon_deg[b, n, :],
                    )

                    # minADE-20 per horizon
                    best_sample = err.mean(axis=1).argmin()  # sample with lowest avg ADE
                    for t in range(T):
                        ade_per_horizon[t].append(err[best_sample, t])

                    fde_list.append(err[best_sample, -1])    # FDE = last horizon

    # ── Report ────────────────────────────────────────────────────────
    ade_horizons = [np.mean(h) for h in ade_per_horizon]
    overall_ade  = np.mean(ade_horizons)
    fde          = np.mean(fde_list)

    print('\n' + '=' * 55)
    print(f'  CPA-GRN  |  Tag: {args.tag}  |  Split: {args.split}')
    print('=' * 55)

    for t, ade_t in enumerate(ade_horizons, 1):
        nm = ade_t * NM_PER_DEGREE
        print(f'  ADE {t:>2}min : {ade_t:.6f}°  ({nm:.5f} nm)')

    print('-' * 55)
    print(f'  minADE-20 : {overall_ade:.6f}°  ({overall_ade*NM_PER_DEGREE:.5f} nm)')
    print(f'  FDE       : {fde:.6f}°  ({fde*NM_PER_DEGREE:.5f} nm)')
    print('=' * 55)
    print()
    print('  S&F paper  ADE: 0.03314 nm  (5-min prediction, Jan 2017)')
    print(f'  Ours (nm)  ADE: {overall_ade*NM_PER_DEGREE:.5f} nm')
    improvement = (0.03314 - overall_ade * NM_PER_DEGREE) / 0.03314 * 100
    print(f'  Improvement vs S&F: {improvement:+.1f}%')
    print('=' * 55)


if __name__ == '__main__':
    main()
