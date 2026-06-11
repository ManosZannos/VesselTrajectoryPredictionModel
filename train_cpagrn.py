"""
train_cpagrn.py — Training script for CPA-GRN.

Uses S&F's data.py pipeline unchanged (feature_size=4 to include SOG+Heading).
Key differences from S&F training:
  - Probabilistic output (GMM, not deterministic)
  - Displacement targets (not absolute positions) → prevents overfitting
  - NLL loss (not ADE loss)
  - Warmup + cosine LR schedule

Run from project root:
    python train_cpagrn.py --zone 11 --epochs 200 --gpu_num 0 --tag CPA_GRN_v1
"""

from __future__ import annotations
import os
import sys
import math
import time
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# S&F pipeline (data loading, identical to their setup)
from data import load_data
from train_utils import get_batch
from utils import seed_everything

from model_cpagrn import CPAGRN, gmm_nll

seed_everything(100)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Train CPA-GRN')

    # Data (must match S&F preprocessing)
    p.add_argument('--zone',             type=int,   default=11)
    p.add_argument('--sequence_length',  type=int,   default=5,
                   help='Observation length (minutes)')
    p.add_argument('--prediction_length',type=int,   default=5,
                   help='Prediction length (minutes)')
    p.add_argument('--feature_size',     type=int,   default=4,
                   help='4 = LAT,LON,SOG,Heading')

    # Model
    p.add_argument('--d_model',    type=int,   default=64)
    p.add_argument('--gru_layers', type=int,   default=2)
    p.add_argument('--K',          type=int,   default=3,
                   help='Number of GMM components')
    p.add_argument('--dropout',    type=float, default=0.1)

    # Training
    p.add_argument('--epochs',      type=int,   default=200)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--weight_decay',type=float, default=1e-4)
    p.add_argument('--clip_grad',   type=float, default=1.0)
    p.add_argument('--warmup',      type=int,   default=10,
                   help='Linear warmup epochs')

    # Misc
    p.add_argument('--gpu_num',  type=int,   default=0)
    p.add_argument('--tag',      type=str,   default='CPA_GRN_v1')
    p.add_argument('--log_every',type=int,   default=10)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(epoch: int, args) -> float:
    if epoch < args.warmup:
        return args.lr * (epoch + 1) / args.warmup
    progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(loader, model, optimizer, device, args, train: bool):
    model.train(train)
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        # ── Unpack S&F batch ──────────────────────────────────────────
        # sequence:      [B, T_obs, N, feature_size]
        # target:        [B, T_pred, N, 2]          (absolute LAT/LON normalised)
        # dist/bear/hdg: [B, T_obs, N, N]           (pre-computed by data.py)
        # ip_mask:       [B, N] bool
        # op_mask:       [B, N] bool
        (sequence, target,
         _dist, _bearing, _heading,
         ip_mask, op_mask, _vessels) = get_batch(batch)

        sequence = sequence.to(device)
        target   = target.to(device)
        ip_mask  = ip_mask.to(device)
        op_mask  = op_mask.to(device)

        # ── Displacement target ───────────────────────────────────────
        # last_obs: [B, N, 2] → [B, N, 1, 2] for broadcasting
        last_obs    = sequence[:, -1, :, :2].unsqueeze(2)   # [B, N, 1, 2]
        # target shape from data.py: [B, T_pred, N, 2] — rearrange to [B, N, T_pred, 2]
        target_BNTF = target.permute(0, 2, 1, 3)            # [B, N, T_pred, 2]
        target_disp = target_BNTF - last_obs                 # [B, N, T_pred, 2]

        # ── Forward ──────────────────────────────────────────────────
        # obs_seq expected as [B, T_obs, N, F]  (sequence is already in that shape)
        gmm_params, _ = model(sequence, ip_mask=ip_mask, op_mask=op_mask)
        # gmm_params: [B, N, pred_len, K*6]

        loss = gmm_nll(gmm_params, target_disp, op_mask, K=args.K)

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # ── Device ───────────────────────────────────────────────────────
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Directories & logging ─────────────────────────────────────────
    ckpt_dir = os.path.join('checkpoints', args.tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, 'train.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger()
    log.info(f'Tag: {args.tag}')
    log.info(f'Args: {vars(args)}')

    # ── Data (reuses S&F pipeline exactly) ────────────────────────────
    data_dir = f'data/{args.zone:02d}/'
    assert os.path.isdir(data_dir), f'Data directory not found: {data_dir}. Run grid.py first.'
    train_ds, val_ds, _ = load_data(data_dir, args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    log.info(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

    # ── Model ─────────────────────────────────────────────────────────
    model = CPAGRN(
        feature_size = args.feature_size,
        d_model      = args.d_model,
        gru_layers   = args.gru_layers,
        K            = args.K,
        pred_len     = args.prediction_length,
        dropout      = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Model parameters: {n_params:,}')

    # ── Optimiser ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_epoch    = 0

    for epoch in range(args.epochs):
        # Update LR
        new_lr = get_lr(epoch, args)
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr

        t0 = time.time()
        train_loss = run_epoch(train_loader, model, optimizer, device, args, train=True)
        val_loss   = run_epoch(val_loader,   model, optimizer, device, args, train=False)
        elapsed    = time.time() - t0

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            log.info(
                f'Epoch {epoch+1:>3}/{args.epochs} | '
                f'lr={new_lr:.2e} | '
                f'train={train_loss:.6f} | '
                f'val={val_loss:.6f} | '
                f't={elapsed:.1f}s'
            )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch + 1
            torch.save({
                'epoch':      epoch + 1,
                'model':      model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'val_loss':   val_loss,
                'args':       vars(args),
            }, os.path.join(ckpt_dir, 'val_best.pth'))

        # Always save latest
        torch.save({
            'epoch':     epoch + 1,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(ckpt_dir, 'latest.pth'))

    log.info(f'Training complete. Best val loss: {best_val_loss:.6f} at epoch {best_epoch}')


if __name__ == '__main__':
    main()
