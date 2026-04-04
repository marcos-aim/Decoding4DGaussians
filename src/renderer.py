# src/renderer.py
"""Thin wrapper around diff-gaussian-rasterization CUDA kernel.

Provides single-camera and multi-camera rendering.
torch.compile is NOT applied here — the CUDA extension is already optimized.
"""

import math
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render_gaussians(
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacity: torch.Tensor,
    shs: torch.Tensor,
    camera,
    bg_color: torch.Tensor,
    sh_degree: int = 0,
):
    """Render Gaussians from a given camera viewpoint.

    Args:
        means3D:   [N, 3] Gaussian centers in world space
        scales:    [N, 3] Gaussian scales (positive)
        rotations: [N, 4] unit quaternions (wxyz)
        opacity:   [N, 1] opacity values in [0, 1]
        shs:       [N, C, 3] spherical harmonics (C = (sh_degree+1)^2)
        camera:    MiniCam with world_view_transform, full_proj_transform, etc.
        bg_color:  [3] background color tensor on GPU
        sh_degree: SH degree (default 0 = DC only)

    Returns:
        rendered: [3, H, W] RGB image
        radii:    [N] screen-space radii
        depth:    [1, H, W] depth map
        screenspace_points: [N, 3] for gradient flow
    """
    device = means3D.device
    screenspace_points = torch.zeros_like(means3D, requires_grad=True, device=device)

    raster_settings = GaussianRasterizationSettings(
        image_height=camera.image_height,
        image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx / 2.0),
        tanfovy=math.tan(camera.FoVy / 2.0),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered, radii, depth = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    return rendered, radii, depth, screenspace_points


def render_multi_camera(
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacity: torch.Tensor,
    shs: torch.Tensor,
    cameras: list,
    bg_color: torch.Tensor,
    sh_degree: int = 0,
) -> list:
    """Render Gaussians from multiple cameras. Shares Gaussian tensors across renders.

    Returns:
        List of dicts, each with keys: rendered, radii, depth, screenspace_points
    """
    results = []
    for cam in cameras:
        rendered, radii, depth, ssp = render_gaussians(
            means3D, scales, rotations, opacity, shs, cam, bg_color, sh_degree
        )
        results.append({
            "rendered": rendered,
            "radii": radii,
            "depth": depth,
            "screenspace_points": ssp,
        })
    return results
