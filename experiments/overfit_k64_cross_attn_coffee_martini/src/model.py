"""Gaussian decoder heads: Canonical + Cross-Attention Deformation.

Key change from original: DeformationHead uses cross-attention between
canonical Gaussian features (queries) and per-frame STV2 tokens (keys/values).
This lets each Gaussian dynamically attend to relevant frame-specific features.
Opacity is predicted as absolute (0-1) per frame, allowing Gaussians to
fully appear/disappear for topology changes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: float) -> float:
    return math.log(x / (1.0 - x))


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


class CanonicalGaussianHead(nn.Module):
    """Same as original — produces canonical Gaussian params from time-averaged tokens."""

    def __init__(self, dim_in: int = 2048, dim_hidden: int = 512, sh_degree: int = 1,
                 init_xyz: torch.Tensor = None, num_gaussians_per_patch: int = 1,
                 init_xyz_per_gaussian: torch.Tensor = None,
                 init_log_scale: float = math.log(0.5),
                 spread: float = 0.05):
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

        self.xyz_head = nn.Linear(dim_hidden, self.K * 3)
        self.scale_head = nn.Linear(dim_hidden, self.K * 3)
        self.rot_head = nn.Linear(dim_hidden, self.K * 4)
        self.opacity_head = nn.Linear(dim_hidden, self.K * 1)
        self.sh_head = nn.Linear(dim_hidden, self.K * self.num_sh_coeffs * 3)

        if init_xyz_per_gaussian is not None:
            self.register_buffer("xyz_anchor", init_xyz_per_gaussian)
        elif init_xyz is not None:
            if self.K > 1:
                P = init_xyz.shape[0]
                offsets = torch.randn(P, self.K, 3) * spread
                offsets[:, 0, :] = 0.0
                anchor_expanded = init_xyz.unsqueeze(1).expand(-1, self.K, -1) + offsets
                self.register_buffer("xyz_anchor", anchor_expanded.reshape(-1, 3))
            else:
                self.register_buffer("xyz_anchor", init_xyz)
        else:
            self.xyz_anchor = None

        self._init_log_scale = init_log_scale
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.rot_head.weight)
        with torch.no_grad():
            bias = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.K)
            self.rot_head.bias.copy_(bias)
        nn.init.zeros_(self.opacity_head.weight)
        nn.init.constant_(self.opacity_head.bias, inverse_sigmoid(0.5))
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, self._init_log_scale)

    def forward(self, tokens_mean: torch.Tensor):
        P = tokens_mean.shape[0]
        K = self.K
        x = self.mlp(self.norm(tokens_mean))  # [P, dim_hidden]

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
            "hidden": x,  # [P, dim_hidden] — used as queries for cross-attention
        }


class CrossAttentionDeformationHead(nn.Module):
    """
    Cross-attention deformation head.

    Queries: canonical Gaussian hidden features [P, dim_canonical]
    Keys/Values: per-frame STV2 tokens [P, dim_tokens]

    Each canonical Gaussian attends to all per-frame tokens to gather
    frame-specific information. This allows Gaussians to:
    - Track moving objects by attending to the right spatial tokens
    - Predict absolute opacity (0-1) for appear/disappear effects
    - Make large position adjustments (not clamped to tiny deltas)
    """

    def __init__(self, dim_canonical: int = 1024, dim_tokens: int = 2048,
                 dim_hidden: int = 256, n_heads: int = 8, n_layers: int = 2,
                 sh_degree: int = 0, num_gaussians_per_patch: int = 1,
                 max_displacement: float = 2.0):
        super().__init__()
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.K = num_gaussians_per_patch
        self.max_displacement = max_displacement

        # Project queries and keys/values to same dimension
        self.q_proj = nn.Sequential(
            nn.LayerNorm(dim_canonical),
            nn.Linear(dim_canonical, dim_hidden),
        )
        self.kv_proj = nn.Sequential(
            nn.LayerNorm(dim_tokens),
            nn.Linear(dim_tokens, dim_hidden),
        )

        # Cross-attention layers with residual connections
        self.cross_attn_layers = nn.ModuleList()
        self.cross_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(n_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(dim_hidden, n_heads, batch_first=True, dropout=0.0)
            )
            self.cross_norms.append(nn.LayerNorm(dim_hidden))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden * 4),
                nn.GELU(),
                nn.Linear(dim_hidden * 4, dim_hidden),
            ))
            self.ffn_norms.append(nn.LayerNorm(dim_hidden))

        # Output heads — predict per-frame modifications for K Gaussians per patch
        self.dxyz_head = nn.Linear(dim_hidden, self.K * 3)
        self.dscale_head = nn.Linear(dim_hidden, self.K * 3)
        self.drot_head = nn.Linear(dim_hidden, self.K * 4)
        # ABSOLUTE opacity per frame (not delta) — allows full appear/disappear
        self.opacity_head = nn.Linear(dim_hidden, self.K * 1)
        # Per-frame color (additive delta on canonical SH)
        self.dsh_head = nn.Linear(dim_hidden, self.K * self.num_sh_coeffs * 3)

        self._init_weights()

    def _init_weights(self):
        # Start with near-zero outputs so initial behavior matches canonical
        for head in [self.dxyz_head, self.dscale_head, self.drot_head, self.dsh_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # Identity quaternion delta
        with torch.no_grad():
            bias = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.K)
            self.drot_head.bias.copy_(bias)
        # Opacity: init bias to 0 so sigmoid(0)=0.5, canonical opacity dominates early
        nn.init.zeros_(self.opacity_head.weight)
        nn.init.zeros_(self.opacity_head.bias)

    def forward(self, canonical_hidden: torch.Tensor, frame_tokens: torch.Tensor):
        """
        Args:
            canonical_hidden: [P, dim_canonical] from CanonicalGaussianHead
            frame_tokens: [P, dim_tokens] per-frame STV2 tokens
        Returns:
            dict with per-frame Gaussian modifications
        """
        P = canonical_hidden.shape[0]
        K = self.K

        Q = self.q_proj(canonical_hidden)  # [P, dim_hidden]
        KV = self.kv_proj(frame_tokens)    # [P, dim_hidden]

        # Add batch dim for attention
        Q = Q.unsqueeze(0)   # [1, P, dim_hidden]
        KV = KV.unsqueeze(0)  # [1, P, dim_hidden]

        for cross_attn, cross_norm, ffn, ffn_norm in zip(
            self.cross_attn_layers, self.cross_norms,
            self.ffn_layers, self.ffn_norms
        ):
            # Cross-attention: Gaussians attend to frame tokens
            attn_out, _ = cross_attn(cross_norm(Q), cross_norm(KV), KV)
            Q = Q + attn_out
            # FFN
            Q = Q + ffn(ffn_norm(Q))

        Q = Q.squeeze(0)  # [P, dim_hidden]

        # Position: larger displacement range
        dxyz = torch.tanh(self.dxyz_head(Q).view(P * K, 3)) * self.max_displacement

        # Scale: multiplicative delta
        dscale = torch.tanh(self.dscale_head(Q).view(P * K, 3)) * 0.5

        # Rotation: delta quaternion
        drot = F.normalize(self.drot_head(Q).view(P * K, 4), dim=-1)

        # ABSOLUTE per-frame opacity (0-1) — key change!
        # This is a per-frame opacity MODIFIER that gets combined with canonical
        opacity_logit = self.opacity_head(Q).view(P * K, 1)

        # Color delta
        dsh = self.dsh_head(Q).view(P * K, self.num_sh_coeffs, 3)

        return {
            "dxyz": dxyz,
            "dscale": dscale,
            "drot": drot,
            "opacity_logit": opacity_logit,  # raw logit, composed in compose_gaussians
            "dsh": dsh,
        }


def compose_gaussians(canonical: dict, deltas: dict = None, scale_factor: float = 1.0):
    """
    Compose final Gaussian parameters from canonical + cross-attention deformation.

    Key difference from original: opacity is composed as
    sigmoid(canonical_logit + frame_logit) instead of sigmoid(canonical_logit) + delta.
    This allows full 0-1 range per frame.
    """
    if deltas is None:
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
        # Opacity: canonical logit + frame logit → sigmoid → full 0-1 range
        opacity = torch.sigmoid(canonical["logit_opacity"] + deltas["opacity_logit"])
        shs = canonical["sh"] + deltas["dsh"]

    scales = torch.clamp(scales, min=1e-6, max=5.0)

    return means3D, scales, rotations, opacity, shs
