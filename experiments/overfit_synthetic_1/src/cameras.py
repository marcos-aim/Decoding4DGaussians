"""Camera utilities: LLFF parsing, projection matrices, MiniCam for rasterizer.

Matches the 4DGaussians camera loading pipeline exactly:
  1. Load poses_bounds.npy → (N, 3, 5)
  2. Axis swap: [col1, -col0, col2, col3] (LLFF → OpenCV)
  3. R/T transform: R = -R; R[:,0] = -R[:,0]; T = -t @ R
  4. getWorld2View: [[R^T, T], [0, 1]]
  5. znear=0.01, zfar=100.0 (hardcoded, matching 3DGS/4DGS)
"""

import math
import numpy as np
import torch


def parse_llff_poses(poses_bounds_path: str):
    """
    Parse LLFF poses_bounds.npy into camera parameters.
    Returns raw axis-swapped poses (N, 3, 4) plus H, W, focal.
    """
    poses_arr = np.load(poses_bounds_path)  # (N, 17)
    poses_raw = poses_arr[:, :-2].reshape(-1, 3, 5)  # (N, 3, 5)
    near_fars = poses_arr[:, -2:]  # (N, 2)

    H = poses_raw[0, 0, 4]
    W = poses_raw[0, 1, 4]
    focal = poses_raw[0, 2, 4]

    # Axis swap: LLFF [down, right, backwards, t] → [right, -down, backwards, t]
    # Matches 4DGS: poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    poses = np.concatenate([
        poses_raw[..., 1:2],   # right
        -poses_raw[..., :1],   # -down
        poses_raw[..., 2:4],   # backwards + translation
    ], axis=-1)  # (N, 3, 4)

    return {
        "poses": poses,  # (N, 3, 4) — axis-swapped [R | t]
        "H": float(H),
        "W": float(W),
        "focal": float(focal),
        "near_fars": near_fars,
    }


def focal_to_fov(focal: float, pixels: float) -> float:
    """Convert focal length in pixels to field of view in radians."""
    return 2.0 * math.atan(pixels / (2.0 * focal))


def llff_pose_to_R_T(pose_3x4: np.ndarray):
    """
    Convert an axis-swapped LLFF pose (3x4) to R, T for getWorld2View.
    Matches 4DGS multipleview_dataset.py exactly:
        R = -R; R[:,0] = -R[:,0]; T = -t @ R
    """
    R = pose_3x4[:3, :3].copy()
    t = pose_3x4[:3, 3].copy()

    # 4DGS convention: negate R, then un-negate first column
    R = -R
    R[:, 0] = -R[:, 0]

    # T = -t @ R  (camera-space translation)
    T = -t.dot(R)

    return R, T


def get_world2view(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Build 4x4 world-to-view matrix.
    Follows 3DGS/4DGS convention: W2V = [[R^T, T], [0, 1]]
    R and T must already be in the 4DGS convention (from llff_pose_to_R_T).
    """
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    return Rt


def get_projection_matrix(znear: float, zfar: float, fovX: float, fovY: float) -> torch.Tensor:
    """OpenGL-style projection matrix matching 3DGS convention."""
    tanHalfFovY = math.tan(fovY / 2.0)
    tanHalfFovX = math.tan(fovX / 2.0)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P


class MiniCam:
    """Lightweight camera object compatible with diff-gaussian-rasterization."""

    def __init__(self, R: np.ndarray, T: np.ndarray, focal: float,
                 width: int, height: int, orig_W: float, orig_H: float,
                 znear: float = 0.01, zfar: float = 100.0, time: float = 0.0,
                 focal_xy: tuple = None):
        """
        Args:
            R: (3,3) rotation matrix (4DGS convention — already processed by llff_pose_to_R_T)
            T: (3,) translation vector (4DGS convention — camera-space)
            focal: focal length at ORIGINAL resolution
            width: target render width
            height: target render height
            orig_W: original image width
            orig_H: original image height
            focal_xy: optional (fx, fy) already at target resolution — overrides focal scaling
        """
        self.image_width = width
        self.image_height = height
        self.time = time
        self.znear = znear
        self.zfar = zfar

        # Scale focal length to target resolution (or use override directly)
        if focal_xy is not None:
            focal_x, focal_y = focal_xy
        else:
            focal_x = focal * (width / orig_W)
            focal_y = focal * (height / orig_H)

        self.FoVx = focal_to_fov(focal_x, width)
        self.FoVy = focal_to_fov(focal_y, height)

        # World-to-view: 4x4, transposed for row-major (3DGS convention)
        w2v = get_world2view(R, T)
        self.world_view_transform = torch.tensor(w2v).transpose(0, 1).cuda().float()

        # Projection: 4x4, transposed
        proj = get_projection_matrix(znear, zfar, self.FoVx, self.FoVy)
        self.projection_matrix = proj.transpose(0, 1).cuda().float()

        # Full transform = W2V @ Proj
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

        # Camera center from inverse of W2V
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def build_cameras_from_llff(poses_bounds_path: str, target_w: int, target_h: int,
                             cam_files: list, focal_xy_override: tuple = None) -> dict:
    """
    Parse LLFF file and build MiniCam objects for each camera.
    Follows 4DGS pipeline: axis swap → R/T transform → getWorld2View.

    Args:
        focal_xy_override: optional (fx, fy) at target resolution to use instead of
                           the LLFF focal from poses_bounds.npy. Use when VGGT's
                           predicted intrinsics differ significantly from the LLFF focal.
    """
    parsed = parse_llff_poses(poses_bounds_path)
    poses = parsed["poses"]  # (N, 3, 4)

    cameras = {}
    for i, cam_file in enumerate(cam_files):
        cam_name = cam_file.replace(".mp4", "")
        R, T = llff_pose_to_R_T(poses[i])
        cameras[cam_name] = MiniCam(
            R=R,
            T=T,
            focal=parsed["focal"],
            width=target_w,
            height=target_h,
            orig_W=parsed["W"],
            orig_H=parsed["H"],
            focal_xy=focal_xy_override,
        )

    return cameras
