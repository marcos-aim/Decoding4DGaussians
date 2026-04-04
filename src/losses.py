"""Loss registry for shared training engine.

Wraps all loss functions with per-loss config: weight, warmup, compute frequency.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as compute_ssim
import lpips as lpips_lib


# ---------------------------------------------------------------------------
# Raw loss functions (direct ports from experiment losses.py)
# ---------------------------------------------------------------------------

def photometric_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    lambda_ssim: float = 0.85,
) -> torch.Tensor:
    """Combined L1 + SSIM photometric loss.

    Default: 0.85 SSIM + 0.15 L1 (standard 3DGS convention).
    """
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    l1 = F.l1_loss(rendered, target)
    ssim_val = compute_ssim(rendered, target, data_range=1.0, size_average=True)
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim_val)


def scale_regularization(scales: torch.Tensor) -> torch.Tensor:
    """Penalize large Gaussian scales to prevent unbounded growth."""
    return torch.mean(torch.abs(scales))


def opacity_regularization(opacity: torch.Tensor) -> torch.Tensor:
    """Entropy regularization on opacity — push toward 0 or 1.

    Uses binary entropy H(p) = -p*log(p) - (1-p)*log(1-p), which is maximised
    at p=0.5 (loss=log2) and minimised at p=0 or p=1 (loss=0).  This encourages
    hard, binary opacity decisions rather than soft mid-range values.
    """
    p = opacity.clamp(1e-6, 1.0 - 1e-6)
    return torch.mean(-p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p))


def geometric_loss(
    gaussian_means: torch.Tensor,
    points_map: torch.Tensor,
    max_pts: int = 8192,
) -> torch.Tensor:
    """L1 loss between Gaussian centers and nearest STV2 point map points.

    Subsamples both sides to avoid OOM with large Gaussian counts.

    Args:
        gaussian_means: [N, 3] Gaussian positions (N can be P*K)
        points_map:     [H, W, 3] STV2 predicted 3D point cloud for one frame
        max_pts:        max points to use from each side for cdist

    Returns:
        scalar loss
    """
    pts = points_map.reshape(-1, 3)

    N = gaussian_means.shape[0]
    if N > max_pts:
        idx = torch.randperm(N, device=gaussian_means.device)[:max_pts]
        g_sub = gaussian_means[idx]
    else:
        g_sub = gaussian_means

    M = pts.shape[0]
    if M > max_pts:
        idx = torch.randperm(M, device=pts.device)[:max_pts]
        p_sub = pts[idx]
    else:
        p_sub = pts

    dists = torch.cdist(g_sub.unsqueeze(0), p_sub.unsqueeze(0)).squeeze(0)
    min_dists, _ = dists.min(dim=1)
    return min_dists.mean()


def depth_loss(
    rendered_depth: torch.Tensor,
    points_map: torch.Tensor,
    camera,
) -> torch.Tensor:
    """Compare rendered depth map with GT depth derived from STV2 point cloud.

    Args:
        rendered_depth: [1, H, W] from rasterizer
        points_map:     [H, W, 3] world-space 3D points from STV2
        camera:         MiniCam with world_view_transform

    Returns:
        scalar loss
    """
    H, W = rendered_depth.shape[1], rendered_depth.shape[2]
    pts = points_map.reshape(-1, 3)

    ones = torch.ones(pts.shape[0], 1, device=pts.device)
    pts_homo = torch.cat([pts, ones], dim=-1)
    pts_cam = pts_homo @ camera.world_view_transform
    gt_depth = pts_cam[:, 2].reshape(H, W)

    rd = rendered_depth.squeeze(0)

    mask = (gt_depth > 0.01) & (rd > 0.01)
    if mask.sum() < 100:
        return torch.tensor(0.0, device=rendered_depth.device)

    gt_median = gt_depth[mask].median()
    if gt_median < 1e-6:
        return torch.tensor(0.0, device=rendered_depth.device)

    return F.smooth_l1_loss(rd[mask] / gt_median, gt_depth[mask] / gt_median)


def tv_loss(
    deltas_t: torch.Tensor,
    deltas_t_prev: torch.Tensor,
) -> torch.Tensor:
    """Temporal variation loss on deformation deltas.

    Args:
        deltas_t:      [P, D] deltas at frame t
        deltas_t_prev: [P, D] deltas at frame t-1

    Returns:
        scalar loss
    """
    return F.l1_loss(deltas_t, deltas_t_prev)


# ---------------------------------------------------------------------------
# Lazy LPIPS singleton
# ---------------------------------------------------------------------------

_lpips_net = None

def _get_lpips_net(device="cuda"):
    """Lazy-load LPIPS AlexNet (shared singleton)."""
    global _lpips_net
    if _lpips_net is None:
        _lpips_net = lpips_lib.LPIPS(net="alex").to(device)
        _lpips_net.eval()
        for p in _lpips_net.parameters():
            p.requires_grad = False
    return _lpips_net


# ---------------------------------------------------------------------------
# Adapter functions — bridge raw losses to the registry interface
# fn(predictions, targets, **kwargs) -> tensor
# ---------------------------------------------------------------------------

def _rgb_adapter(predictions: dict, targets: dict, **kwargs) -> torch.Tensor:
    lambda_ssim = kwargs.get("lambda_ssim", 0.85)
    return photometric_loss(predictions["rendered"], targets["gt_image"], lambda_ssim=lambda_ssim)


def _ssim_adapter(predictions: dict, targets: dict, **kwargs) -> torch.Tensor:
    rendered = predictions["rendered"]
    target = targets["gt_image"]
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    ssim_val = compute_ssim(rendered, target, data_range=1.0, size_average=True)
    return 1.0 - ssim_val


def _geometric_adapter(predictions: dict, targets: dict, **kwargs) -> torch.Tensor:
    max_pts = kwargs.get("max_pts", 8192)
    return geometric_loss(predictions["gaussian_means"], targets["points_map"], max_pts=max_pts)


def _depth_adapter(predictions: dict, targets: dict, **kwargs) -> torch.Tensor:
    return depth_loss(predictions["rendered_depth"], targets["points_map"], targets["camera"])


def _tv_adapter(predictions: dict, targets: dict, **kwargs) -> torch.Tensor:
    return tv_loss(predictions["all_deltas"], predictions["all_deltas_prev"])


def _scale_reg_adapter(predictions: dict, targets: dict, **kwargs) -> torch.Tensor:
    return scale_regularization(predictions["scales"])


def _opacity_reg_adapter(predictions: dict, targets: dict, **kwargs) -> torch.Tensor:
    return opacity_regularization(predictions["opacity"])


def _lpips_adapter(predictions: dict, targets: dict, **kwargs) -> torch.Tensor:
    """LPIPS perceptual loss with optional downsampling."""
    rendered = predictions["rendered"]
    target = targets["gt_image"]
    downsample = kwargs.get("downsample", 1)

    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    if downsample > 1:
        rendered = F.interpolate(rendered, scale_factor=1.0/downsample, mode="bilinear",
                                 align_corners=False)
        target = F.interpolate(target, scale_factor=1.0/downsample, mode="bilinear",
                               align_corners=False)

    net = _get_lpips_net(rendered.device)
    return net(rendered * 2 - 1, target * 2 - 1).mean()


# ---------------------------------------------------------------------------
# LossRegistry
# ---------------------------------------------------------------------------

class LossRegistry:
    """Registry of named loss functions with per-loss config support.

    Each loss in ``phase_losses`` is a dict with optional keys:
      - ``weight`` (float, default 1.0): multiplicative weight
      - ``warmup_steps`` (int, default 0): linearly ramp weight from 0 to ``weight``
        over this many steps
      - ``compute_every_n`` (int, default 1): only recompute on steps divisible by
        this value; use cached value otherwise
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._losses: dict[str, callable] = {}
        self._cache: dict[str, torch.Tensor] = {}

        # Register all built-ins
        self.register("rgb", _rgb_adapter)
        self.register("ssim", _ssim_adapter)
        self.register("geometric", _geometric_adapter)
        self.register("depth", _depth_adapter)
        self.register("tv", _tv_adapter)
        self.register("scale_reg", _scale_reg_adapter)
        self.register("opacity_reg", _opacity_reg_adapter)
        self.register("lpips", _lpips_adapter)

    def register(self, name: str, loss_fn: callable) -> None:
        """Register a loss function under ``name``.

        Args:
            name:    unique identifier
            loss_fn: callable(predictions, targets, **kwargs) -> scalar tensor
        """
        self._losses[name] = loss_fn

    def list_losses(self) -> list[str]:
        """Return names of all registered losses."""
        return list(self._losses.keys())

    def compute(
        self,
        predictions: dict,
        targets: dict,
        step: int,
        phase_losses: dict[str, dict],
    ) -> torch.Tensor:
        """Compute total weighted loss for the current step.

        Args:
            predictions:  dict of model outputs (keys depend on active losses)
            targets:      dict of ground-truth data
            step:         current training step (0-indexed)
            phase_losses: mapping from loss name to config dict

        Returns:
            scalar total loss tensor
        """
        zero = torch.tensor(0.0, device=self.device)
        total = zero.clone()

        for name, cfg in phase_losses.items():
            if name not in self._losses:
                raise KeyError(f"Loss '{name}' is not registered. Available: {self.list_losses()}")

            weight: float = cfg.get("weight", 1.0)
            warmup_steps: int = cfg.get("warmup_steps", 0)
            compute_every_n: int = cfg.get("compute_every_n", 1)

            # Warmup: linearly scale weight from 0 at step 0 to full weight at warmup_steps
            if warmup_steps > 0:
                weight = weight * min(1.0, step / warmup_steps)

            if weight == 0.0:
                total = total + zero
                continue

            # compute_every_n: reuse cached value on non-compute steps
            if compute_every_n > 1 and step % compute_every_n != 0:
                cached = self._cache.get(name)
                if cached is not None:
                    total = total + weight * cached
                # If no cache yet (e.g. step=1, compute_every_n=5), skip this loss
                continue

            # Compute the loss
            try:
                loss_val = self._losses[name](predictions, targets, **{k: v for k, v in cfg.items()
                                                                        if k not in ("weight", "warmup_steps", "compute_every_n")})
            except KeyError:
                # Missing prediction key — skip gracefully (caller responsible for providing inputs)
                continue

            # Cache for future compute_every_n skips
            self._cache[name] = loss_val.detach()

            total = total + weight * loss_val

        return total
