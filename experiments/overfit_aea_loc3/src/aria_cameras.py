"""Convert Aria (OpenCV c2w, 4x4) camera poses to MiniCam objects compatible
with our diff_gaussian_rasterization renderer.

Our renderer expects the 3DGS convention:
  - world_view_transform: 4x4 row-major-transposed W2V
  - projection_matrix: 4x4 row-major-transposed (znear, zfar, FoVx, FoVy)
  - full_proj_transform: W2V @ Proj
  - camera_center: W2V.inverse()[3, :3]

Aria gives OpenCV-style c2w (X-right, Y-down, Z-forward). That matches the
3DGS convention directly — no axis swap like LLFF needs.
"""

import math
from typing import List

import numpy as np
import torch


def focal_to_fov(focal: float, length: float) -> float:
    return 2.0 * math.atan(length / (2.0 * focal))


def get_projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float) -> torch.Tensor:
    """Standard 3DGS projection matrix (column-major before the caller transposes)."""
    tan_half_fov_y = math.tan(fov_y / 2.0)
    tan_half_fov_x = math.tan(fov_x / 2.0)
    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right

    P = torch.zeros(4, 4, dtype=torch.float32)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class AriaMiniCam:
    def __init__(self, c2w: torch.Tensor, fx: float, fy: float, cx: float, cy: float,
                 width: int, height: int, znear: float = 0.1, zfar: float = 100.0):
        self.image_width = width
        self.image_height = height
        self.znear = znear
        self.zfar = zfar

        # FoV from focal (note: Aria rectified pinhole has principal point at image center)
        self.FoVx = focal_to_fov(fx, width)
        self.FoVy = focal_to_fov(fy, height)

        # W2V from C2W
        c2w_np = c2w.detach().cpu().numpy().astype(np.float64)
        w2v_np = np.linalg.inv(c2w_np)
        w2v = torch.from_numpy(w2v_np).float()

        # 3DGS convention: transpose for row-major shader layout
        self.world_view_transform = w2v.transpose(0, 1).cuda()

        proj = get_projection_matrix(znear, zfar, self.FoVx, self.FoVy)
        self.projection_matrix = proj.transpose(0, 1).cuda()

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)

        self.camera_center = self.world_view_transform.inverse()[3, :3]


def build_cameras_from_aria(dataset, frame_indices: List[int]) -> List[AriaMiniCam]:
    """Build one MiniCam per listed frame index in the AriaSequenceDataset."""
    cams = []
    for i in frame_indices:
        c2w = dataset.get_c2w(i)
        f = dataset.frames[i]
        sx = dataset.target_w / f.w
        sy = dataset.target_h / f.h
        cams.append(AriaMiniCam(
            c2w=c2w,
            fx=f.fx * sx,
            fy=f.fy * sy,
            cx=f.cx * sx,
            cy=f.cy * sy,
            width=dataset.target_w,
            height=dataset.target_h,
        ))
    return cams
