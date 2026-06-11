"""
model_cpagrn.py — CPA-Aware Graph Recurrent Network

Philosophy: vessels interact through encounters. Instead of soft global
attention (VeST) or a discretised domain matrix (S&F), we model each
encounter explicitly using TCPA (Time to Closest Point of Approach) and
DCPA (Distance at Closest Point of Approach) — the exact quantities every
navigator checks to assess collision risk.

Architecture:
  1. Feature Embedding          Linear(4 → d_model) + LayerNorm
  2. GRU Encoder                per-vessel recurrence over obs_len steps
  3. CPA-Aware Spatial Layer    message-passing with TCPA/DCPA edge features
  4. Non-autoregressive Decoder MLP: d_model → pred_len × d_model
  5. GMM Head                   K bivariate Gaussians (displacement targets)

All computations are in the normalised [0,1] geographic space used by
data.py. TCPA and DCPA are computed from consecutive normalised positions.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CPA Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

class CPAFeatures(nn.Module):
    """
    Computes 7 pairwise edge features from the observed trajectory.

    For each vessel pair (i → j):
        TCPA   — time until closest approach (negative = already past)
        DCPA   — distance at closest approach
        dist   — current distance
        sin/cos(bearing)   — direction from i to j
        sin/cos(delta_hdg) — relative heading difference

    All values are in normalised geographic space [0, 1].
    """
    EDGE_DIM = 7

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        obs_seq: [B, T_obs, N, 4]  (LAT_norm, LON_norm, SOG_norm, Hdg_norm)
        Returns: [B, N, N, 7]
        """
        B, T, N, _ = obs_seq.shape

        pos = obs_seq[:, -1, :, :2]                        # [B, N, 2]
        vel = (obs_seq[:, -1, :, :2] - obs_seq[:, -2, :, :2]
               if T >= 2 else torch.zeros_like(pos))       # [B, N, 2]
        hdg = obs_seq[:, -1, :, 3]                         # [B, N]  (normalised 0-1)

        # Expand to [B, N, N, *] for pairwise computation
        pos_i = pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = pos.unsqueeze(1).expand(B, N, N, 2)
        vel_i = vel.unsqueeze(2).expand(B, N, N, 2)
        vel_j = vel.unsqueeze(1).expand(B, N, N, 2)
        hdg_i = hdg.unsqueeze(2).expand(B, N, N)
        hdg_j = hdg.unsqueeze(1).expand(B, N, N)

        r = pos_j - pos_i                                   # relative position [B,N,N,2]
        v = vel_j - vel_i                                   # relative velocity  [B,N,N,2]

        dist    = r.norm(dim=-1)                            # [B, N, N]
        bearing = torch.atan2(r[..., 0], r[..., 1])        # [B, N, N]

        # Heading difference in radians (normalised hdg → [0, 2π])
        delta_hdg = (hdg_j - hdg_i) * 2.0 * math.pi       # [B, N, N]

        # TCPA = -(r · v) / (|v|² + ε)
        v_sq = (v * v).sum(dim=-1) + self.eps
        tcpa = (-(r * v).sum(dim=-1) / v_sq).clamp(-5.0, 5.0)

        # DCPA = |r + TCPA * v|
        dcpa = (r + tcpa.unsqueeze(-1) * v).norm(dim=-1).clamp(0.0, 10.0)

        return torch.stack([
            tcpa,
            dcpa,
            dist,
            torch.sin(bearing),
            torch.cos(bearing),
            torch.sin(delta_hdg),
            torch.cos(delta_hdg),
        ], dim=-1)                                          # [B, N, N, 7]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CPA-Aware Spatial Attention Layer
# ─────────────────────────────────────────────────────────────────────────────

class CPAAwareSpatialLayer(nn.Module):
    """
    One round of CPA-aware message passing.

    For vessel i, the attention weight toward vessel j is learned from:
        [h_i || h_j || edge_features_ij]
    where edge_features_ij contains TCPA and DCPA — encoding collision risk.

    A vessel with low DCPA (high collision risk) receives more attention,
    but the exact weighting is learned end-to-end from data.
    """
    EDGE_DIM = 7

    def __init__(self, d_model: int):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * d_model + self.EDGE_DIM, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.norm       = nn.LayerNorm(d_model)

    def forward(
        self,
        h:             torch.Tensor,   # [B, N, d_model]
        edge_features: torch.Tensor,   # [B, N, N, 7]
        mask:          torch.Tensor | None,  # [B, N] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: updated h [B, N, d_model], attention weights [B, N, N]."""
        B, N, D = h.shape

        h_i = h.unsqueeze(2).expand(B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)

        scores = self.attn_mlp(
            torch.cat([h_i, h_j, edge_features], dim=-1)
        ).squeeze(-1)                                       # [B, N, N]

        # Mask absent/padded vessels
        if mask is not None:
            mask_j = mask.unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(~mask_j, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)       # fully-masked rows → 0

        values_j = self.value_proj(h).unsqueeze(1).expand(B, N, N, D)
        agg      = (weights.unsqueeze(-1) * values_j).sum(dim=2)  # [B, N, D]

        return self.norm(h + agg), weights


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Main Model
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRN(nn.Module):
    """CPA-Aware Graph Recurrent Network."""

    def __init__(
        self,
        feature_size: int   = 4,
        d_model:      int   = 64,
        gru_layers:   int   = 2,
        K:            int   = 3,
        pred_len:     int   = 5,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.d_model  = d_model
        self.pred_len = pred_len
        self.K        = K

        # ── Layers ────────────────────────────────────────────────────
        self.embed = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.gru = nn.GRU(
            d_model, d_model,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        self.cpa_features = CPAFeatures()
        self.spatial      = CPAAwareSpatialLayer(d_model)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * d_model),
        )

        self.gmm_head = nn.Linear(d_model, K * 6)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        obs_seq: torch.Tensor,            # [B, T_obs, N, feature_size]
        ip_mask: torch.Tensor | None = None,  # [B, N] bool
        op_mask: torch.Tensor | None = None,  # [B, N] bool  (unused here, used in loss)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gmm_params:   [B, N, pred_len, K*6]
            attn_weights: [B, N, N]
        """
        B, T, N, _ = obs_seq.shape

        # 1. Embed
        x = self.embed(obs_seq)           # [B, T, N, d_model]

        # 2. GRU — process each vessel independently over time
        x_in  = x.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        h_out, _ = self.gru(x_in)         # [B*N, T, d_model]
        h = h_out[:, -1].reshape(B, N, self.d_model)  # [B, N, d_model]

        # Zero out padded vessels so they don't pollute attention
        if ip_mask is not None:
            h = h * ip_mask.float().unsqueeze(-1)

        # 3. CPA-aware spatial attention
        edge_feat = self.cpa_features(obs_seq)          # [B, N, N, 7]
        h, attn_weights = self.spatial(h, edge_feat, ip_mask)

        # 4. Decode (non-autoregressive)
        dec = self.decoder(h).reshape(B, N, self.pred_len, self.d_model)

        # 5. GMM params
        gmm_params = self.gmm_head(dec)                 # [B, N, pred_len, K*6]

        return gmm_params, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Loss Function
# ─────────────────────────────────────────────────────────────────────────────

def gmm_nll(
    gmm_params:  torch.Tensor,   # [B, N, T, K*6]
    target_disp: torch.Tensor,   # [B, N, T, 2]  displacement targets
    op_mask:     torch.Tensor,   # [B, N]  True = vessel present
    K:           int = 3,
    eps:         float = 1e-6,
) -> torch.Tensor:
    """
    Negative log-likelihood of target displacement under a mixture of K
    bivariate Gaussians.  Only active (unmasked) vessels contribute to loss.

    GMM params per component: (pi_logit, mu_x, mu_y, log_sx, log_sy, atanh_rho)
    """
    B, N, T, _ = gmm_params.shape
    params = gmm_params.reshape(B, N, T, K, 6)

    pi      = F.softmax(params[..., 0], dim=-1)          # [B,N,T,K]
    mu_x    = params[..., 1]
    mu_y    = params[..., 2]
    log_sx  = params[..., 3]
    log_sy  = params[..., 4]
    rho     = torch.tanh(params[..., 5])

    sx  = torch.exp(log_sx).clamp(min=1e-4)
    sy  = torch.exp(log_sy).clamp(min=1e-4)
    rho = rho.clamp(-0.99, 0.99)

    dx = target_disp[..., 0:1] - mu_x                   # [B,N,T,K]
    dy = target_disp[..., 1:2] - mu_y

    z = (dx / sx) ** 2 - 2 * rho * (dx / sx) * (dy / sy) + (dy / sy) ** 2

    log_norm = (torch.log(2 * math.pi * sx * sy)
                + 0.5 * torch.log((1 - rho ** 2).clamp(min=eps)))
    log_p    = -0.5 * z / (1 - rho ** 2).clamp(min=eps) - log_norm

    # Log-sum-exp over K components for numerical stability
    log_mix = torch.log(pi + eps) + log_p               # [B,N,T,K]
    nll     = -torch.logsumexp(log_mix, dim=-1)          # [B,N,T]

    # Mask padded vessels
    if op_mask is not None:
        mask = op_mask.unsqueeze(-1).expand(B, N, T)     # [B,N,T]
        nll  = nll[mask]

    return nll.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Trajectory Sampling  (for minADE-20 evaluation)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_trajectories(
    gmm_params:  torch.Tensor,   # [B, N, T, K*6]
    n_samples:   int = 20,
    K:           int = 3,
    eps:         float = 1e-6,
) -> torch.Tensor:
    """
    Coherent sampling: one GMM component is chosen per vessel (averaged over T),
    then held fixed across all pred_len timesteps.

    Returns: [n_samples, B, N, T, 2]  displacement samples
    """
    B, N, T, _ = gmm_params.shape
    params = gmm_params.reshape(B, N, T, K, 6)

    # Average mixture weights over time → one distribution per vessel
    pi_avg = F.softmax(params[..., 0], dim=-1).mean(dim=2)  # [B, N, K]

    samples = []
    for _ in range(n_samples):
        # Pick one component coherently for each vessel
        comp = torch.multinomial(
            pi_avg.reshape(B * N, K), num_samples=1
        ).reshape(B, N)                                      # [B, N]

        # Gather params for chosen component
        c = comp.unsqueeze(-1).unsqueeze(-1).expand(B, N, T, 1)
        mu_x   = params[..., 1].gather(-1, c).squeeze(-1)   # [B,N,T]
        mu_y   = params[..., 2].gather(-1, c).squeeze(-1)
        log_sx = params[..., 3].gather(-1, c).squeeze(-1)
        log_sy = params[..., 4].gather(-1, c).squeeze(-1)
        rho    = torch.tanh(params[..., 5].gather(-1, c).squeeze(-1))

        sx  = torch.exp(log_sx).clamp(min=1e-4)
        sy  = torch.exp(log_sy).clamp(min=1e-4)
        rho = rho.clamp(-0.99, 0.99)

        # Sample bivariate Gaussian
        eps1 = torch.randn_like(mu_x)
        eps2 = torch.randn_like(mu_y)
        sx_val = mu_x + sx * eps1
        sy_val = mu_y + sy * (rho * eps1 + torch.sqrt((1 - rho ** 2).clamp(min=eps)) * eps2)

        samples.append(torch.stack([sx_val, sy_val], dim=-1))  # [B,N,T,2]

    return torch.stack(samples, dim=0)   # [n_samples, B, N, T, 2]
