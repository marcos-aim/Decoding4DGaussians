"""Thin wrapper around diff-gaussian-rasterization.

Note: Pre-rasterization opacity culling is intentionally NOT applied during training —
culling near-transparent Gaussians blocks photometric gradients to the deformation head,
creating permanently dead Gaussians that can't recover. Culling is only used at eval time.
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
    sh_degree: int = 1,
    cull_opacity: float = 0.0,
):
    """
    Render Gaussians from a given camera viewpoint.

    Args:
        means3D:       [N, 3] Gaussian centers in world space
        scales:        [N, 3] Gaussian scales (positive)
        rotations:     [N, 4] unit quaternions (wxyz)
        opacity:       [N, 1] opacity values in [0, 1]
        shs:           [N, C, 3] spherical harmonics
        camera:        MiniCam object
        bg_color:      [3] background color tensor on GPU
        sh_degree:     SH degree
        cull_opacity:  if > 0, cull Gaussians below this opacity (eval only)

    Returns:
        rendered: [3, H, W] rendered RGB image
        radii:   [N] screen-space radii per Gaussian
        depth:   [1, H, W] rendered depth map
        screenspace_points: means2D placeholder for gradient flow
    """
    device = means3D.device

    # Optional culling — only use at eval time (cull_opacity > 0)
    if cull_opacity > 0:
        alive = opacity.squeeze(-1) >= cull_opacity
        if alive.any():
            means3D = means3D[alive]
            scales = scales[alive]
            rotations = rotations[alive]
            opacity = opacity[alive]
            shs = shs[alive]

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
