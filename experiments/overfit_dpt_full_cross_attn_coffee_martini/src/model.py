"""Gaussian decoder heads: DPT Canonical + Cross-Attention Deformation.

H1: Full Synthetic Multi-Scale DPT head replaces MLP-based CanonicalGaussianHead.
Uses 4-level feature pyramid with top-down DPT fusion, optional image injection,
and DA3-style camera-aware Gaussian parameter prediction.

The CrossAttentionDeformationHead and compose_gaussians remain unchanged.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dpt_modules import (
    SyntheticPyramid, DPTFusion, ImageInjector
)


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


class DPTCanonicalGaussianHead(nn.Module):
    """Full DPT-based canonical Gaussian head (H1).

    Replaces MLP-based CanonicalGaussianHead with:
    1. Reshape tokens to 2D spatial grid
    2. 4-level synthetic pyramid
    3. Top-down DPT fusion
    4. Optional RGB image injection
    5. Conv-based Gaussian parameter prediction
    6. DA3-style camera-aware scale normalization

    Output interface matches CanonicalGaussianHead exactly.
    """

    def __init__(self, dim_in: int = 2048, grid_h: int = 28, grid_w: int = 38,
                 sh_degree: int = 0, num_gaussians_per_patch: int = 128,
                 init_xyz: torch.Tensor = None,
                 init_xyz_per_gaussian: torch.Tensor = None,
                 spread: float = 0.05,
                 pyramid_dim: int = 256, fused_dim: int = 128,
                 use_image_injection: bool = True):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.K = num_gaussians_per_patch
        self.use_image_injection = use_image_injection

        # 1. Synthetic multi-scale pyramid
        self.pyramid = SyntheticPyramid(dim_in, pyramid_dim)

        # 2. Top-down DPT fusion
        self.fusion = DPTFusion(pyramid_dim, fused_dim)

        # 3. Optional image injection
        if use_image_injection:
            self.image_injector = ImageInjector(fused_dim)

        # 4. Per-patch feature refinement (fused_dim -> hidden for param heads)
        hidden_dim = 256
        self.patch_refine = nn.Sequential(
            nn.Conv2d(fused_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        # 5. Gaussian parameter conv heads (per-pixel, K Gaussians each)
        self.xyz_head = nn.Conv2d(hidden_dim, self.K * 3, 1)
        self.scale_head = nn.Conv2d(hidden_dim, self.K * 3, 1)
        self.rot_head = nn.Conv2d(hidden_dim, self.K * 4, 1)
        self.opacity_head = nn.Conv2d(hidden_dim, self.K * 1, 1)
        self.sh_head = nn.Conv2d(hidden_dim, self.K * self.num_sh_coeffs * 3, 1)

        # Hidden feature projection for cross-attention deformation queries
        self.hidden_proj = nn.Conv2d(hidden_dim, 1024, 1)

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

        self._init_weights()

    def _init_weights(self):
        # Rotation: identity quaternion
        nn.init.zeros_(self.rot_head.weight)
        with torch.no_grad():
            bias = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.K)
            self.rot_head.bias.copy_(bias)
        # Opacity: sigmoid(0) = 0.5
        nn.init.zeros_(self.opacity_head.weight)
        nn.init.constant_(self.opacity_head.bias, inverse_sigmoid(0.5))
        # Scale: small initial values (will be multiplied by depth-aware factor)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, 0.0)  # sigmoid(0)=0.5, scaled by depth

    def forward(self, tokens_mean: torch.Tensor, image: torch.Tensor = None):
        """
        Args:
            tokens_mean: [P, dim_in] time-averaged STV2 tokens (P = grid_h * grid_w)
            image: [3, H_img, W_img] optional RGB image for injection

        Returns:
            dict matching CanonicalGaussianHead output interface
        """
        P = tokens_mean.shape[0]
        K = self.K

        # Reshape to 2D spatial grid: [P, C] -> [1, C, H, W]
        x = tokens_mean.view(self.grid_h, self.grid_w, -1)
        x = x.permute(2, 0, 1).unsqueeze(0)  # [1, dim_in, grid_h, grid_w]

        # Multi-scale pyramid
        levels = self.pyramid(x)

        # Top-down DPT fusion
        fused = self.fusion(levels)  # [1, fused_dim, grid_h, grid_w]

        # Image injection (optional)
        if self.use_image_injection and image is not None:
            # Resize image to grid resolution
            img = image.unsqueeze(0)  # [1, 3, H_img, W_img]
            img_resized = F.interpolate(img, size=(self.grid_h, self.grid_w),
                                        mode="bilinear", align_corners=True)
            fused = fused + self.image_injector(img_resized)

        # Refine
        features = self.patch_refine(fused)  # [1, hidden_dim, grid_h, grid_w]

        # Predict Gaussian parameters
        xyz_raw = self.xyz_head(features)       # [1, K*3, H, W]
        scale_raw = self.scale_head(features)   # [1, K*3, H, W]
        rot_raw = self.rot_head(features)       # [1, K*4, H, W]
        opa_raw = self.opacity_head(features)   # [1, K*1, H, W]
        sh_raw = self.sh_head(features)         # [1, K*sh_c*3, H, W]

        # Hidden features for deformation cross-attention
        hidden = self.hidden_proj(features)     # [1, 1024, H, W]

        # Reshape all outputs from [1, C, H, W] -> [P*K, params]
        def _reshape(t, param_dim):
            # [1, K*param_dim, H, W] -> [H*W, K*param_dim] -> [H*W*K, param_dim]
            t = t.squeeze(0).permute(1, 2, 0)  # [H, W, K*param_dim]
            t = t.reshape(P, K * param_dim)
            return t.reshape(P * K, param_dim)

        xyz_residual = _reshape(xyz_raw, 3)
        xyz = xyz_residual + self.xyz_anchor if self.xyz_anchor is not None else xyz_residual

        log_scale = _reshape(scale_raw, 3)
        rot = F.normalize(_reshape(rot_raw, 4), dim=-1)
        logit_opacity = _reshape(opa_raw, 1)
        sh = _reshape(sh_raw, self.num_sh_coeffs * 3).view(P * K, self.num_sh_coeffs, 3)

        # Hidden: [1, 1024, H, W] -> [P, 1024]
        hidden_out = hidden.squeeze(0).permute(1, 2, 0).reshape(P, 1024)

        return {
            "xyz": xyz,
            "log_scale": log_scale,
            "logit_opacity": logit_opacity,
            "rot": rot,
            "sh": sh,
            "hidden": hidden_out,
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
