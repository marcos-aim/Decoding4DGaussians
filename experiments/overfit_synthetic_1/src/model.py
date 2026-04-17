"""Gaussian decoder heads: Canonical + Deformation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: float) -> float:
    """Inverse of sigmoid function."""
    return math.log(x / (1.0 - x))


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions (wxyz convention).
    q1, q2: [..., 4]
    Returns: [..., 4]
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


class CanonicalGaussianHead(nn.Module):
    """
    Produces canonical (rest-pose) Gaussian parameters from aggregated tokens.
    Supports K Gaussians per patch token for higher spatial resolution.

    Input:  tokens_mean [P, 2048]
    Output: xyz [P*K, 3], scale [P*K, 3], rot [P*K, 4], opacity [P*K, 1], sh [P*K, C, 3]
    """

    def __init__(self, dim_in: int = 2048, dim_hidden: int = 512, sh_degree: int = 1,
                 init_xyz: torch.Tensor = None, num_gaussians_per_patch: int = 1,
                 init_xyz_per_gaussian: torch.Tensor = None):
        super().__init__()
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.K = num_gaussians_per_patch

        self.norm = nn.LayerNorm(dim_in)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.GELU(),
            nn.Linear(1024, dim_hidden),
            nn.GELU(),
        )

        # Each head outputs K sets of parameters per patch
        self.xyz_head = nn.Linear(dim_hidden, self.K * 3)
        self.scale_head = nn.Linear(dim_hidden, self.K * 3)
        self.rot_head = nn.Linear(dim_hidden, self.K * 4)
        self.opacity_head = nn.Linear(dim_hidden, self.K * 1)
        self.sh_head = nn.Linear(dim_hidden, self.K * self.num_sh_coeffs * 3)

        # Position anchors from actual 3D point cloud
        if init_xyz_per_gaussian is not None:
            # Pre-computed [P*K, 3] anchors from full point cloud
            self.register_buffer("xyz_anchor", init_xyz_per_gaussian)
        elif init_xyz is not None:
            if self.K > 1:
                P = init_xyz.shape[0]
                spread = 0.05
                offsets = torch.randn(P, self.K, 3) * spread
                offsets[:, 0, :] = 0.0
                anchor_expanded = init_xyz.unsqueeze(1).expand(-1, self.K, -1) + offsets
                self.register_buffer("xyz_anchor", anchor_expanded.reshape(-1, 3))
            else:
                self.register_buffer("xyz_anchor", init_xyz)
        else:
            self.xyz_anchor = None

        self._init_weights()

    def _init_weights(self):
        # Identity quaternion for each of K rotations
        nn.init.zeros_(self.rot_head.weight)
        with torch.no_grad():
            bias = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.K)
            self.rot_head.bias.copy_(bias)

        # Moderate initial opacity
        nn.init.zeros_(self.opacity_head.weight)
        nn.init.constant_(self.opacity_head.bias, inverse_sigmoid(0.5))

        # Scale init: softplus(log(0.5)) ≈ 0.4 — visible enough for SSIM warmup
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, math.log(0.5))

    def forward(self, tokens_mean: torch.Tensor):
        """
        Args:
            tokens_mean: [P, 2048]
        Returns:
            dict with Gaussian params, all shaped [P*K, ...]
        """
        P = tokens_mean.shape[0]
        K = self.K
        x = self.mlp(self.norm(tokens_mean))  # [P, dim_hidden]

        # Predict K Gaussians per patch, then flatten to [P*K, ...]
        xyz_residual = self.xyz_head(x).view(P * K, 3)
        xyz = xyz_residual + self.xyz_anchor if self.xyz_anchor is not None else xyz_residual

        log_scale = self.scale_head(x).view(P * K, 3)
        rot = F.normalize(self.rot_head(x).view(P * K, 4), dim=-1)
        logit_opacity = self.opacity_head(x).view(P * K, 1)
        sh = self.sh_head(x).view(P * K, self.num_sh_coeffs, 3)

        return {
            "xyz": xyz,
            "log_scale": log_scale,
            "logit_opacity": logit_opacity,
            "rot": rot,
            "sh": sh,
            "hidden": x,
        }


class DeformationHead(nn.Module):
    """
    Produces per-frame deformation deltas from per-frame tokens.
    Supports K Gaussians per patch — attention runs at patch level,
    then each patch independently predicts K sets of deltas.
    """

    def __init__(self, dim_in: int = 2048, dim_hidden: int = 256,
                 n_attn_heads: int = 8, n_attn_layers: int = 2,
                 sh_degree: int = 1, num_gaussians_per_patch: int = 1):
        super().__init__()
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.K = num_gaussians_per_patch

        self.norm = nn.LayerNorm(dim_in)
        self.proj = nn.Linear(dim_in, dim_hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden,
            nhead=n_attn_heads,
            dim_feedforward=dim_hidden * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers)

        # Each head outputs K sets of deltas per patch
        self.dxyz_head = nn.Linear(dim_hidden, self.K * 3)
        self.dscale_head = nn.Linear(dim_hidden, self.K * 3)
        self.drot_head = nn.Linear(dim_hidden, self.K * 4)
        self.dopacity_head = nn.Linear(dim_hidden, self.K * 1)
        self.dsh_head = nn.Linear(dim_hidden, self.K * self.num_sh_coeffs * 3)

        self._init_weights()

    def _init_weights(self):
        for head in [self.dxyz_head, self.dscale_head, self.drot_head,
                     self.dopacity_head, self.dsh_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            bias = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.K)
            self.drot_head.bias.copy_(bias)

    def forward(self, tokens_t: torch.Tensor):
        """
        Args:
            tokens_t: [P, 2048]
        Returns:
            dict with delta params, all shaped [P*K, ...]
        """
        P = tokens_t.shape[0]
        K = self.K

        x = self.proj(self.norm(tokens_t))  # [P, dim_hidden]
        x = self.attn(x.unsqueeze(0)).squeeze(0)  # [P, dim_hidden]

        dxyz = torch.tanh(self.dxyz_head(x).view(P * K, 3)) * 0.1
        dscale = torch.tanh(self.dscale_head(x).view(P * K, 3)) * 0.5
        drot = F.normalize(self.drot_head(x).view(P * K, 4), dim=-1)
        dopacity = torch.tanh(self.dopacity_head(x).view(P * K, 1)) * 0.3
        dsh_flat = self.dsh_head(x).view(P * K, self.num_sh_coeffs * 3)
        dsh = dsh_flat.view(P * K, self.num_sh_coeffs, 3)

        all_deltas = torch.cat([dxyz, dscale, drot, dopacity, dsh_flat], dim=-1)

        return {
            "dxyz": dxyz,
            "dscale": dscale,
            "drot": drot,
            "dopacity": dopacity,
            "dsh": dsh,
            "all_deltas": all_deltas,
        }


def compose_gaussians(canonical: dict, deltas: dict = None, scale_factor: float = 1.0):
    """
    Compose final Gaussian parameters from canonical + optional deformation deltas.

    Args:
        canonical: dict from CanonicalGaussianHead.forward()
        deltas:    dict from DeformationHead.forward(), or None for static
        scale_factor: multiplier for Gaussian scales (1.0 = no change, <1.0 = smaller)
                      Used for progressive scale annealing during training.

    Returns:
        means3D [P, 3], scales [P, 3], rotations [P, 4], opacity [P, 1], shs [P, C, 3]
    """
    if deltas is None:
        # Static: canonical only
        means3D = canonical["xyz"]
        scales = scale_factor * F.softplus(canonical["log_scale"])
        rotations = canonical["rot"]
        opacity = torch.sigmoid(canonical["logit_opacity"])
        shs = canonical["sh"]
    else:
        means3D = canonical["xyz"] + deltas["dxyz"]
        scales = scale_factor * F.softplus(canonical["log_scale"] + deltas["dscale"])
        rotations = quaternion_multiply(canonical["rot"], deltas["drot"])
        rotations = F.normalize(rotations, dim=-1)
        opacity = torch.sigmoid(canonical["logit_opacity"] + deltas["dopacity"])
        shs = canonical["sh"] + deltas["dsh"]

    # Clamp scales: min prevents collapse, max prevents extreme billboard Gaussians
    scales = torch.clamp(scales, min=1e-6, max=5.0)

    return means3D, scales, rotations, opacity, shs
