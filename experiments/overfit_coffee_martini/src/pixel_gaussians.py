"""Pixel-aligned Gaussian head: upsamples patch tokens to per-pixel Gaussians.

Instead of predicting K Gaussians per patch (which cluster around patch centers),
this head upsamples patch features to pixel resolution and predicts one Gaussian
per pixel, using STV2's per-pixel 3D points as positions.

Architecture:
  1. Project patch tokens [P, 2048] → [P, dim_feat] (reduce before upsample)
  2. Reshape to spatial grid [1, dim_feat, grid_h, grid_w]
  3. Bilinear upsample to [1, dim_feat, H, W]
  4. Per-pixel MLP → Gaussian attributes (scale, rot, opacity, color)
  5. Positions from points_map (fixed, true 3D coverage)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: float) -> float:
    return math.log(x / (1.0 - x))


def fourier_encode(x: torch.Tensor, num_freqs: int = 4) -> torch.Tensor:
    """
    Fourier positional encoding of coordinates.
    x: [..., D] → [..., D + D*2*num_freqs]
    """
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)  # [L]
    # x: [..., D], freqs: [L] → x_expanded: [..., D, L]
    x_expanded = x.unsqueeze(-1) * freqs  # [..., D, L]
    # Interleave sin and cos
    encoded = torch.cat([x_expanded.sin(), x_expanded.cos()], dim=-1)  # [..., D, 2L]
    encoded = encoded.reshape(*x.shape[:-1], -1)  # [..., D*2L]
    return torch.cat([x, encoded], dim=-1)  # [..., D + D*2L]


class PixelGaussianHead(nn.Module):
    """
    Produces one Gaussian per pixel from upsampled patch features.
    Simple architecture: project → bilinear upsample → 1x1 conv MLP.
    The bilinear upsampling enforces spatial smoothness which acts as
    a strong multi-view consistency prior.

    Input:  tokens [P, 2048], points_map [H, W, 3]
    Output: means3D [N, 3], scales [N, 3], rot [N, 4], opacity [N, 1], sh [N, 1, 3]
    """

    def __init__(self, dim_in: int = 2048, dim_feat: int = 256, dim_hidden: int = 256,
                 grid_h: int = 27, grid_w: int = 39, img_h: int = 384, img_w: int = 512,
                 num_freqs: int = 4):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.img_h = img_h
        self.img_w = img_w
        self.num_freqs = num_freqs
        # Fourier encoding: 3 raw + 3*2*num_freqs = 3 + 24 = 27 channels
        self.xyz_dim = 3 + 3 * 2 * num_freqs

        # Project high-dim tokens to compact features (at patch level)
        self.proj = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_feat),
            nn.GELU(),
        )

        # XYZ branch: Fourier-encoded coords → per-pixel features (added, not concatenated)
        self.xyz_branch = nn.Sequential(
            nn.Conv2d(self.xyz_dim, dim_feat, 1),
            nn.GELU(),
        )

        # Per-pixel MLP (1x1 conv — same input size as without XYZ, fast!)
        # Input: dim_feat, Output: 3(scale) + 4(rot) + 1(opacity) + 3(color) = 11
        self.pixel_mlp = nn.Sequential(
            nn.Conv2d(dim_feat, dim_hidden, 1),
            nn.GELU(),
            nn.Conv2d(dim_hidden, dim_hidden, 1),
            nn.GELU(),
            nn.Conv2d(dim_hidden, dim_hidden // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim_hidden // 2, 11, 1),
        )

        self._init_weights()

    def _init_weights(self):
        final_conv = self.pixel_mlp[-1]
        nn.init.zeros_(final_conv.weight)
        with torch.no_grad():
            bias = torch.zeros(11)
            bias[0:3] = math.log(0.3)       # softplus(log(0.3)) ≈ 0.26
            bias[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            bias[7] = inverse_sigmoid(0.5)
            bias[8:11] = 0.0
            final_conv.bias.copy_(bias)

    def forward(self, tokens: torch.Tensor, points_map: torch.Tensor,
                xyz_feat_precomputed: torch.Tensor = None):
        """
        Args:
            tokens:     [P, 2048] patch tokens
            points_map: [H, W, 3] per-pixel 3D positions from STV2
            xyz_feat_precomputed: [1, xyz_dim, H, W] pre-computed Fourier features (optional)
        Returns:
            dict with Gaussian params, all shaped [N, ...] where N = H*W
        """
        P = tokens.shape[0]
        H, W = self.img_h, self.img_w

        feat = self.proj(tokens)  # [P, dim_feat]

        # Reshape to spatial grid, padding if needed
        grid_n = self.grid_h * self.grid_w
        if P < grid_n:
            pad = torch.zeros(grid_n - P, feat.shape[1], device=feat.device)
            feat_grid = torch.cat([feat, pad], dim=0)
        else:
            feat_grid = feat[:grid_n]

        feat_grid = feat_grid.view(1, self.grid_h, self.grid_w, -1).permute(0, 3, 1, 2)

        # Bilinear upsample to pixel resolution (strong smoothness prior)
        feat_pixel = F.interpolate(feat_grid, size=(H, W), mode='bilinear', align_corners=False)

        # Add Fourier-encoded 3D coordinates for per-pixel spatial detail
        # Additive branch: avoids channel explosion, keeps feature map same size
        if xyz_feat_precomputed is not None:
            feat_pixel = feat_pixel + self.xyz_branch(xyz_feat_precomputed)
        else:
            xyz_encoded = fourier_encode(points_map, self.num_freqs)
            xyz_feat = xyz_encoded.permute(2, 0, 1).unsqueeze(0)
            feat_pixel = feat_pixel + self.xyz_branch(xyz_feat)

        # Per-pixel MLP
        attrs = self.pixel_mlp(feat_pixel)
        attrs = attrs.squeeze(0).permute(1, 2, 0).reshape(-1, 11)

        log_scale = attrs[:, 0:3]
        rot = F.normalize(attrs[:, 3:7], dim=-1)
        logit_opacity = attrs[:, 7:8]
        sh_dc = attrs[:, 8:11].unsqueeze(1)

        xyz = points_map.reshape(-1, 3)

        return {
            "xyz": xyz,
            "log_scale": log_scale,
            "logit_opacity": logit_opacity,
            "rot": rot,
            "sh": sh_dc,
        }


class PixelDeformationHead(nn.Module):
    """
    Predicts per-pixel deformation deltas from per-frame tokens.
    Same simple bilinear + 1x1 conv architecture.
    """

    def __init__(self, dim_in: int = 2048, dim_feat: int = 128, dim_hidden: int = 128,
                 grid_h: int = 27, grid_w: int = 39, img_h: int = 384, img_w: int = 512):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.img_h = img_h
        self.img_w = img_w

        self.proj = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_feat),
            nn.GELU(),
        )

        # dx(3) + dscale(3) + drot(4) + dopacity(1) + dcolor(3) = 14
        self.pixel_mlp = nn.Sequential(
            nn.Conv2d(dim_feat, dim_hidden, 1),
            nn.GELU(),
            nn.Conv2d(dim_hidden, dim_hidden, 1),
            nn.GELU(),
            nn.Conv2d(dim_hidden, 14, 1),
        )

        self._init_weights()

    def _init_weights(self):
        final_conv = self.pixel_mlp[-1]
        nn.init.zeros_(final_conv.weight)
        with torch.no_grad():
            bias = torch.zeros(14)
            bias[6:10] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            final_conv.bias.copy_(bias)

    def forward(self, tokens_t: torch.Tensor):
        """
        Args:
            tokens_t: [P, 2048] per-frame tokens
        Returns:
            dict with delta params, all shaped [H*W, ...]
        """
        P = tokens_t.shape[0]
        H, W = self.img_h, self.img_w

        feat = self.proj(tokens_t)
        grid_n = self.grid_h * self.grid_w
        if P < grid_n:
            pad = torch.zeros(grid_n - P, feat.shape[1], device=feat.device)
            feat_grid = torch.cat([feat, pad], dim=0)
        else:
            feat_grid = feat[:grid_n]

        feat_grid = feat_grid.view(1, self.grid_h, self.grid_w, -1).permute(0, 3, 1, 2)
        feat_pixel = F.interpolate(feat_grid, size=(H, W), mode='bilinear', align_corners=False)

        attrs = self.pixel_mlp(feat_pixel).squeeze(0).permute(1, 2, 0).reshape(-1, 14)

        dxyz = torch.tanh(attrs[:, 0:3]) * 0.1
        dscale = torch.tanh(attrs[:, 3:6]) * 0.5
        drot = F.normalize(attrs[:, 6:10], dim=-1)
        dopacity = torch.tanh(attrs[:, 10:11]) * 0.3
        dsh = attrs[:, 11:14].unsqueeze(1)

        all_deltas = torch.cat([dxyz, dscale, attrs[:, 6:10], dopacity, attrs[:, 11:14]], dim=-1)

        return {
            "dxyz": dxyz,
            "dscale": dscale,
            "drot": drot,
            "dopacity": dopacity,
            "dsh": dsh,
            "all_deltas": all_deltas,
        }


def compose_pixel_gaussians(canonical: dict, deltas: dict = None, scale_factor: float = 1.0):
    """Compose final Gaussian params from pixel-aligned canonical + optional deltas."""
    from model import quaternion_multiply

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
        opacity = torch.sigmoid(canonical["logit_opacity"] + deltas["dopacity"])
        shs = canonical["sh"] + deltas["dsh"]

    scales = torch.clamp(scales, min=1e-6, max=1.0)
    return means3D, scales, rotations, opacity, shs
