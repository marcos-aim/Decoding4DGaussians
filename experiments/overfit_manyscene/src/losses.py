"""Loss functions for Gaussian training."""

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as compute_ssim


def photometric_loss(rendered: torch.Tensor, target: torch.Tensor,
                     lambda_ssim: float = 0.85) -> torch.Tensor:
    """
    Combined L1 + SSIM photometric loss.
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
    """Entropy regularization on opacity — push toward 0 or 1."""
    return torch.mean(-torch.log(opacity + 1e-6) - torch.log(1.0 - opacity + 1e-6))


def geometric_loss(gaussian_means: torch.Tensor,
                   points_map: torch.Tensor,
                   max_pts: int = 8192) -> torch.Tensor:
    """
    L1 loss between Gaussian centers and nearest STV2 point map points.
    Subsamples both sides to avoid OOM with large Gaussian counts.

    Args:
        gaussian_means: [N, 3] Gaussian positions (N can be P*K)
        points_map:     [H, W, 3] STV2 predicted 3D point cloud for one frame
        max_pts:        max points to use from each side for cdist

    Returns:
        scalar loss
    """
    pts = points_map.reshape(-1, 3)

    # Subsample Gaussians if too many
    N = gaussian_means.shape[0]
    if N > max_pts:
        idx = torch.randperm(N, device=gaussian_means.device)[:max_pts]
        g_sub = gaussian_means[idx]
    else:
        g_sub = gaussian_means

    # Subsample point map if too many
    M = pts.shape[0]
    if M > max_pts:
        idx = torch.randperm(M, device=pts.device)[:max_pts]
        p_sub = pts[idx]
    else:
        p_sub = pts

    dists = torch.cdist(g_sub.unsqueeze(0), p_sub.unsqueeze(0)).squeeze(0)
    min_dists, _ = dists.min(dim=1)

    return min_dists.mean()


def depth_loss(rendered_depth: torch.Tensor, points_map: torch.Tensor,
               camera) -> torch.Tensor:
    """
    Compare rendered depth map with GT depth derived from STV2 point cloud.

    Args:
        rendered_depth: [1, H, W] from rasterizer
        points_map:     [H, W, 3] world-space 3D points from STV2
        camera:         MiniCam with world_view_transform

    Returns:
        scalar loss
    """
    H, W = rendered_depth.shape[1], rendered_depth.shape[2]
    pts = points_map.reshape(-1, 3)  # [HW, 3]

    # Transform world points to camera space to get GT depth
    ones = torch.ones(pts.shape[0], 1, device=pts.device)
    pts_homo = torch.cat([pts, ones], dim=-1)  # [HW, 4]
    pts_cam = pts_homo @ camera.world_view_transform  # [HW, 4]
    gt_depth = pts_cam[:, 2].reshape(H, W)  # z-depth in camera space

    rd = rendered_depth.squeeze(0)  # [H, W]

    # Only supervise pixels where both depths are valid (positive, non-tiny)
    mask = (gt_depth > 0.01) & (rd > 0.01)
    if mask.sum() < 100:
        return torch.tensor(0.0, device=rendered_depth.device)

    # Normalize both depths by GT median to handle scale ambiguity
    gt_median = gt_depth[mask].median()
    if gt_median < 1e-6:
        return torch.tensor(0.0, device=rendered_depth.device)

    return F.smooth_l1_loss(rd[mask] / gt_median, gt_depth[mask] / gt_median)


def tv_loss(deltas_t: torch.Tensor, deltas_t_prev: torch.Tensor) -> torch.Tensor:
    """
    Temporal variation loss on deformation deltas.

    Args:
        deltas_t:      [P, D] deltas at frame t
        deltas_t_prev: [P, D] deltas at frame t-1

    Returns:
        scalar loss
    """
    return F.l1_loss(deltas_t, deltas_t_prev)


def compute_psnr(rendered: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute PSNR between rendered and target images.

    Args:
        rendered: [3, H, W] in [0, 1]
        target:   [3, H, W] in [0, 1]

    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(rendered, target).item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * torch.log10(torch.tensor(mse)).item()
