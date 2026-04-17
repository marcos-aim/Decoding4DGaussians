"""DPT (Dense Prediction Transformer) building blocks.

Inspired by Depth Anything V3's DPT/GSDPT architecture.
Adapted for single-level SpaTrackerV2 token inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    """Residual convolutional unit: ReLU -> Conv3x3 -> ReLU -> Conv3x3 + skip."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Top-down feature fusion block from DPT.

    Merges lateral (same-level) features with top-down features,
    applies residual conv processing, upsamples, and projects.
    """

    def __init__(self, features: int, out_features: int = None):
        super().__init__()
        out_features = out_features or features

        self.rcu1 = ResidualConvUnit(features)  # for lateral input
        self.rcu2 = ResidualConvUnit(features)  # for merged features
        self.out_conv = nn.Conv2d(features, out_features, 1)

    def forward(self, top: torch.Tensor, lateral: torch.Tensor = None,
                target_size: tuple = None) -> torch.Tensor:
        if lateral is not None:
            top = top + self.rcu1(lateral)

        out = self.rcu2(top)

        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=True)

        return self.out_conv(out)


class SyntheticPyramid(nn.Module):
    """Creates a 4-level feature pyramid from single-level input.

    Takes 2D spatial features at native resolution and produces
    4 scales via strided convolutions.
    """

    def __init__(self, dim_in: int, dim_out: int = 256):
        super().__init__()
        # Level 1: native resolution, project to dim_out
        self.proj1 = nn.Conv2d(dim_in, dim_out, 1)

        # Level 2: stride-2 downsample
        self.down2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
        )

        # Level 3: stride-2 downsample
        self.down3 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
        )

        # Level 4: stride-2 downsample
        self.down4 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, dim_in, H, W] spatial features at native resolution

        Returns:
            List of 4 feature maps at decreasing resolutions:
            [L1: HxW, L2: H/2xW/2, L3: H/4xW/4, L4: H/8xW/8]
        """
        l1 = self.proj1(x)       # [B, dim_out, H, W]
        l2 = self.down2(l1)      # [B, dim_out, H/2, W/2]
        l3 = self.down3(l2)      # [B, dim_out, H/4, W/4]
        l4 = self.down4(l3)      # [B, dim_out, H/8, W/8]
        return [l1, l2, l3, l4]


class DPTFusion(nn.Module):
    """Full 4-level top-down DPT feature fusion.

    Takes 4 feature levels and produces a single fused output
    at the finest resolution via top-down refinement.
    """

    def __init__(self, features: int = 256, out_features: int = 128):
        super().__init__()
        # Adapter layers (1x1 conv for each level)
        self.adapt1 = nn.Conv2d(features, features, 1)
        self.adapt2 = nn.Conv2d(features, features, 1)
        self.adapt3 = nn.Conv2d(features, features, 1)
        self.adapt4 = nn.Conv2d(features, features, 1)

        # Top-down fusion blocks
        self.refine4 = FeatureFusionBlock(features)
        self.refine3 = FeatureFusionBlock(features)
        self.refine2 = FeatureFusionBlock(features)
        self.refine1 = FeatureFusionBlock(features, out_features)

    def forward(self, levels: list) -> torch.Tensor:
        """
        Args:
            levels: [L1, L2, L3, L4] from SyntheticPyramid

        Returns:
            Fused features at L1 resolution: [B, out_features, H, W]
        """
        l1, l2, l3, l4 = levels

        # Adapt
        l1 = self.adapt1(l1)
        l2 = self.adapt2(l2)
        l3 = self.adapt3(l3)
        l4 = self.adapt4(l4)

        # Top-down fusion: L4 -> L3 -> L2 -> L1
        out = self.refine4(l4, target_size=l3.shape[2:])
        out = self.refine3(out, lateral=l3, target_size=l2.shape[2:])
        out = self.refine2(out, lateral=l2, target_size=l1.shape[2:])
        out = self.refine1(out, lateral=l1)

        return out


class LightDPTFusion(nn.Module):
    """Lightweight 2-level DPT fusion for the hybrid approach.

    Only uses 2 scales (native + half resolution).
    """

    def __init__(self, features: int = 512, out_features: int = 512):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(features, features, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
        )

        self.adapt1 = nn.Conv2d(features, features, 1)
        self.adapt2 = nn.Conv2d(features, features, 1)

        self.refine2 = FeatureFusionBlock(features)
        self.refine1 = FeatureFusionBlock(features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, features, H, W]

        Returns:
            Fused features: [B, out_features, H, W]
        """
        l1 = self.adapt1(x)
        l2 = self.adapt2(self.down(x))

        out = self.refine2(l2, target_size=l1.shape[2:])
        out = self.refine1(out, lateral=l1)

        return out


class ImageInjector(nn.Module):
    """Injects RGB image features into DPT-fused features.

    3-layer GELU CNN that projects RGB to match DPT feature dimension,
    then added residually (DA3 GSDPT pattern).
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_dim // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim // 4, out_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim // 2, out_dim, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H_img, W_img] RGB image

        Returns:
            Features: [B, out_dim, H_img, W_img]
        """
        return self.net(image)
