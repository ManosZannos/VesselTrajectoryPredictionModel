"""
train_cpagrn.py — Training script for CPA-GRN.

Key fixes from previous version:
  - Uses S&F's collate_function() for DataLoader (handles variable vessel counts)
  - Correct batch shape: S&F returns [B, N, T, F], model expects [B, T, N, F]
  - Correct displacement target computation from S&F batch format
  - Removed get_batch dependency

Run:
    python train_cpagrn.py --zone 11 --gpu_num 0 --tag CPA_GRN_v1 \
        --sequence_length 5 --prediction_length 5 --feature_size 4
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

from data import load_data, collate_function
from utils import seed_everything
from model_cpagrn import CPAGRN, gmm_nll

seed_everything(100)


def get_args():
    p = argparse.ArgumentParser(description='Train CPA-GRN')
    p.add_argument('--split_data',       action='store_true', default=False)
    p.add_argument('--zone',             type=int,   default=11)
    p.add_argument('--sequence_length',  type=int,   default=5)
    p.add_argument('--prediction_length',type=int,   default=5)
    p.add_argument('--feature_size',     type=int,   default=4)
    p.add_argument('--d_model',    type=int,   default=64)
    p.add_argument('--gru_layers', type=int,   default=2)
    p.add_argument('--K',          type=int,   default=3)
    p.add_argument('--dropout',    type=float, default=0.1)
    p.add_argument('--epochs',      type=int,   default=200)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--weight_decay',type=float, default=1e-4)
    p.add_argument('--clip_grad',   type=float, default=1.0)
    p.add_argument('--warmup',      type=int,   default=10)
    p.add_argument('--gpu_num',  type=int, default=0)
    p.add_argument('--tag',      type=str, default='CPA_GRN_v1')
    p.add_argument('--log_every',type=int, default=10)
    return p.parse_args()


def get_lr(epoch: int, args) -> float:
    if epoch < args.warmup:
        return args.lr * (epoch + 1) / args.warmup
    progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_epoch(loader, model, optimizer, device, args, train: bool):
    model.train(train)
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        # Unpack batch from collate_function
        (ip, op, dist_matrix, bear_matrix, hdg_matrix,
         _ip_mask, _op_mask, vessel_count) = [t.to(device) for t in batch]

        # Build reliable vessel mask from vessel_count
        # (ip_mask/op_mask from data.py may have wrong shape)
        N = ip.shape[1]
        v_idx = torch.arange(N, device=device).unsqueeze(0)    # [1, N]
        vessel_mask = v_idx < vessel_count.long().unsqueeze(1)  # [B, N] bool

        # Model expects [B, T_obs, N, F]
        obs_seq = ip.permute(0, 2, 1, 3)                        # [B, T_obs, N, F]

        # Displacement target (LAT/LON only)
        last_obs    = ip[:, :, -1, :2]                          # [B, N, 2]
        target_disp = op[..., :2] - last_obs.unsqueeze(2)       # [B, N, T_pred, 2]

        gmm_params, _ = model(obs_seq, ip_mask=vessel_mask, op_mask=vessel_mask)
        loss = gmm_nll(gmm_params, target_disp, vessel_mask, K=args.K)

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def main():
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    ckpt_dir = os.path.join('checkpoints', args.tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(ckpt_dir, 'train.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger()
    log.info(f'Tag: {args.tag}')
    log.info(f'Args: {vars(args)}')

    data_dir = f'data/{args.zone:02d}/'
    assert os.path.isdir(data_dir), f'Not found: {data_dir}. Run grid.py first.'

    train_ds, val_ds, _ = load_data(data_dir, args)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_function(), num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_function(), num_workers=0
    )
    log.info(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

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

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_loss = float('inf')
    best_epoch    = 0

    for epoch in range(args.epochs):
        new_lr = get_lr(epoch, args)
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr

        t0 = time.time()
        train_loss = run_epoch(train_loader, model, optimizer, device, args, train=True)
        val_loss   = run_epoch(val_loader,   model, optimizer, device, args, train=False)
        elapsed    = time.time() - t0

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            log.info(
                f'Epoch {epoch+1:>3}/{args.epochs} | lr={new_lr:.2e} | '
                f'train={train_loss:.6f} | val={val_loss:.6f} | t={elapsed:.1f}s'
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch + 1
            torch.save({
                'epoch': epoch + 1, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss, 'args': vars(args),
            }, os.path.join(ckpt_dir, 'val_best.pth'))

        torch.save({
            'epoch': epoch + 1, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(ckpt_dir, 'latest.pth'))

    log.info(f'Training complete. Best val: {best_val_loss:.6f} at epoch {best_epoch}')


if __name__ == '__main__':
    main()