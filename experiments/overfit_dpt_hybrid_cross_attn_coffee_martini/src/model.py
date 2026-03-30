"""Gaussian decoder heads: Hybrid DPT+MLP Canonical + Cross-Attention Deformation.

H2: Lightweight 2-level DPT for spatial fusion, then per-patch MLP for
Gaussian parameter prediction. Keeps the proven MLP parameterization
from baseline but adds spatial context via DPT feature refinement.

The CrossAttentionDeformationHead and compose_gaussians remain unchanged.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dpt_modules import LightDPTFusion


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


class HybridDPTCanonicalGaussianHead(nn.Module):
    """Hybrid DPT + MLP canonical Gaussian head (H2).

    1. Reshape tokens to 2D spatial grid
    2. Project to intermediate dim
    3. 2-level lightweight DPT fusion (adds spatial context)
    4. Flatten back to per-patch features
    5. Per-patch MLP predicts K Gaussians (like baseline)

    Uses our proven log_scale parameterization (no camera-aware scaling).
    Output interface matches CanonicalGaussianHead exactly.
    """

    def __init__(self, dim_in: int = 2048, dim_hidden: int = 512,
                 grid_h: int = 28, grid_w: int = 38,
                 sh_degree: int = 0, num_gaussians_per_patch: int = 128,
                 init_xyz: torch.Tensor = None,
                 init_xyz_per_gaussian: torch.Tensor = None,
                 init_log_scale: float = math.log(0.5),
                 spread: float = 0.05,
                 dpt_dim: int = 512):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.K = num_gaussians_per_patch

        # 1. Project tokens for 2D conv processing
        self.input_proj = nn.Conv2d(dim_in, dpt_dim, 1)

        # 2. Lightweight 2-level DPT fusion
        self.dpt_fusion = LightDPTFusion(dpt_dim, dpt_dim)

        # 3. Project back and combine with original tokens
        self.output_proj = nn.Linear(dpt_dim, dim_in)
        self.gate = nn.Parameter(torch.tensor(0.1))  # learnable residual gate

        # 4. Per-patch MLP (same as baseline CanonicalGaussianHead)
        self.norm = nn.LayerNorm(dim_in)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.GELU(),
            nn.Linear(1024, dim_hidden),
            nn.GELU(),
        )

        # 5. Gaussian parameter heads (same as baseline)
        self.xyz_head = nn.Linear(dim_hidden, self.K * 3)
        self.scale_head = nn.Linear(dim_hidden, self.K * 3)
        self.rot_head = nn.Linear(dim_hidden, self.K * 4)
        self.opacity_head = nn.Linear(dim_hidden, self.K * 1)
        self.sh_head = nn.Linear(dim_hidden, self.K * self.num_sh_coeffs * 3)

        # Point cloud anchors
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
        """
        Args:
            tokens_mean: [P, dim_in] time-averaged STV2 tokens

        Returns:
            dict matching CanonicalGaussianHead output interface
        """
        P = tokens_mean.shape[0]
        K = self.K

        # Reshape to 2D for DPT: [P, C] -> [1, C, H, W]
        x_2d = tokens_mean.view(self.grid_h, self.grid_w, -1)
        x_2d = x_2d.permute(2, 0, 1).unsqueeze(0)  # [1, dim_in, H, W]

        # Project and fuse spatially
        proj = self.input_proj(x_2d)          # [1, dpt_dim, H, W]
        fused = self.dpt_fusion(proj)         # [1, dpt_dim, H, W]

        # Back to sequence: [1, dpt_dim, H, W] -> [P, dpt_dim]
        fused_seq = fused.squeeze(0).permute(1, 2, 0).reshape(P, -1)  # [P, dpt_dim]
        spatial_ctx = self.output_proj(fused_seq)  # [P, dim_in]

        # Gated residual: tokens + gate * spatial_context
        tokens_enriched = tokens_mean + self.gate * spatial_ctx

        # Standard MLP head (same as baseline from here)
        x = self.mlp(self.norm(tokens_enriched))  # [P, dim_hidden]

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
            "hidden": x,  # [P, dim_hidden] for cross-attention queries
        }


class CrossAttentionDeformationHead(nn.Module):
    """Cross-attention deformation head — UNCHANGED from baseline."""

    def __init__(self, dim_canonical: int = 1024, dim_tokens: int = 2048,
                 dim_hidden: int = 256, n_heads: int = 8, n_layers: int = 2,
                 sh_degree: int = 0, num_gaussians_per_patch: int = 1,
                 max_displacement: float = 2.0):
        super().__init__()
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.K = num_gaussians_per_patch
        self.max_displacement = max_displacement

        self.q_proj = nn.Sequential(
            nn.LayerNorm(dim_canonical),
            nn.Linear(dim_canonical, dim_hidden),
        )
        self.kv_proj = nn.Sequential(
            nn.LayerNorm(dim_tokens),
            nn.Linear(dim_tokens, dim_hidden),
        )

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

        self.dxyz_head = nn.Linear(dim_hidden, self.K * 3)
        self.dscale_head = nn.Linear(dim_hidden, self.K * 3)
        self.drot_head = nn.Linear(dim_hidden, self.K * 4)
        self.opacity_head = nn.Linear(dim_hidden, self.K * 1)
        self.dsh_head = nn.Linear(dim_hidden, self.K * self.num_sh_coeffs * 3)

        self._init_weights()

    def _init_weights(self):
        for head in [self.dxyz_head, self.dscale_head, self.drot_head, self.dsh_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            bias = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.K)
            self.drot_head.bias.copy_(bias)
        nn.init.zeros_(self.opacity_head.weight)
        nn.init.zeros_(self.opacity_head.bias)

    def forward(self, canonical_hidden: torch.Tensor, frame_tokens: torch.Tensor):
        P = canonical_hidden.shape[0]
        K = self.K

        Q = self.q_proj(canonical_hidden)
        KV = self.kv_proj(frame_tokens)

        Q = Q.unsqueeze(0)
        KV = KV.unsqueeze(0)

        for cross_attn, cross_norm, ffn, ffn_norm in zip(
            self.cross_attn_layers, self.cross_norms,
            self.ffn_layers, self.ffn_norms
        ):
            attn_out, _ = cross_attn(cross_norm(Q), cross_norm(KV), KV)
            Q = Q + attn_out
            Q = Q + ffn(ffn_norm(Q))

        Q = Q.squeeze(0)

        dxyz = torch.tanh(self.dxyz_head(Q).view(P * K, 3)) * self.max_displacement
        dscale = torch.tanh(self.dscale_head(Q).view(P * K, 3)) * 0.5
        drot = F.normalize(self.drot_head(Q).view(P * K, 4), dim=-1)
        opacity_logit = self.opacity_head(Q).view(P * K, 1)
        dsh = self.dsh_head(Q).view(P * K, self.num_sh_coeffs, 3)

        return {
            "dxyz": dxyz,
            "dscale": dscale,
            "drot": drot,
            "opacity_logit": opacity_logit,
            "dsh": dsh,
        }


def compose_gaussians(canonical: dict, deltas: dict = None, scale_factor: float = 1.0):
    """Compose final Gaussian parameters from canonical + deformation."""
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
        opacity = torch.sigmoid(canonical["logit_opacity"] + deltas["opacity_logit"])
        shs = canonical["sh"] + deltas["dsh"]

    scales = torch.clamp(scales, min=1e-6, max=5.0)

    return means3D, scales, rotations, opacity, shs
