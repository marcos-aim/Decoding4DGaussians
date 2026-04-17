"""Dataset for loading precomputed STV2 features and camera frames."""

import os
import math
import random
import cv2
import torch
import numpy as np


class CachedSceneDataset:
    """
    Loads precomputed features from cache and serves training batches.
    Token and point-map tensors are moved to CUDA at init if available,
    eliminating per-step CPU→GPU transfers. The training loop .cuda() calls
    become no-ops on already-resident tensors.
    """

    def __init__(self, cache_dir: str, cameras: dict, input_camera: str = "cam01",
                 eval_camera: str = "cam00", scene_dir: str = None):
        self.cache_dir = cache_dir
        self.cameras = cameras
        self.input_camera = input_camera
        self.eval_camera = eval_camera

        self.train_cam_names = [name for name in cameras.keys() if name != eval_camera]

        print("Loading cached tensors...")
        self.tokens = torch.load(os.path.join(cache_dir, "tokens.pt"),
                                  weights_only=True).float()
        self.points_map = torch.load(os.path.join(cache_dir, "points_map.pt"),
                                      weights_only=True).float()
        self.unc_metric = torch.load(os.path.join(cache_dir, "unc_metric.pt"),
                                      weights_only=True).float()

        # Align STV2 points from VGGT coordinate system to LLFF world space
        self._align_points_to_llff(cache_dir, cameras, input_camera, scene_dir)

        self.num_frames = self.tokens.shape[0]
        self.num_patches = self.tokens.shape[1]
        self.token_dim = self.tokens.shape[2]

        self.tokens_mean = self.tokens.mean(dim=0)

        print(f"  tokens: {self.tokens.shape}")
        print(f"  points_map: {self.points_map.shape}")
        print(f"  num_patches (P): {self.num_patches}")
        print(f"  train cameras: {len(self.train_cam_names)}")

        # Move entire cache to CUDA — ~2.5GB, well within available VRAM.
        # Eliminates per-step CPU→GPU transfers for tokens and points_map.
        if torch.cuda.is_available():
            self.tokens = self.tokens.cuda()
            self.tokens_mean = self.tokens_mean.cuda()
            self.points_map = self.points_map.cuda()
            print(f"  Cache pinned to CUDA ({self.tokens.element_size() * self.tokens.nelement() / 1e9:.2f}GB tokens)")

    def _align_points_to_llff(self, cache_dir: str, cameras: dict, input_camera: str,
                               scene_dir: str = None):
        """
        Transform STV2 points from VGGT's coordinate system to LLFF world space.

        VGGT places the first frame's camera at the origin with the scene at
        depth ~0.3-2.0. LLFF cameras expect the scene at depth ~7-100+.
        We apply a similarity transform: p_llff = R @ (s * p_stv2) + t
        where s is the scale factor matching STV2 near depth to LLFF near depth.
        """
        poses_path = os.path.join(cache_dir, "poses.pt")
        if not os.path.exists(poses_path):
            print("  [WARN] No STV2 poses found, skipping alignment")
            return

        stv2_poses = torch.load(poses_path, weights_only=True).float()
        stv2_c2w_0 = stv2_poses[0]  # 4x4, cam01 frame 0 in VGGT coords (≈ identity)

        # LLFF cam01: world_view_transform is row-major transposed W2V
        cam01 = cameras[input_camera]
        w2v_llff = cam01.world_view_transform.T.cpu()  # standard column-major W2V
        c2w_llff = torch.inverse(w2v_llff)  # camera-to-world in LLFF frame

        R_c2w = c2w_llff[:3, :3]
        t_c2w = c2w_llff[:3, 3]

        # Compute scale factor: match STV2 min depth to LLFF near bound
        # STV2 points in cam01 camera space (cam01 ≈ origin, Z = depth)
        stv2_inv = torch.inverse(stv2_c2w_0)
        pts_flat = self.points_map.reshape(-1, 3)
        pts_cam = pts_flat @ stv2_inv[:3, :3].T + stv2_inv[:3, 3]
        z_stv2 = pts_cam[:, 2]
        z_positive = z_stv2[z_stv2 > 0.01]
        z_sorted = z_positive.sort().values
        z_near_stv2 = z_sorted[len(z_sorted) // 20]  # 5th percentile as robust near

        # Derive scale from LLFF cam01 camera-center distance from world origin.
        # This is robust against unreliable near/far in poses_bounds.npy (e.g.
        # clamped synthetic exports). STV2 places cam01 near origin with scene
        # at depth ~0.3-2.0; LLFF cameras are typically several units from origin.
        cam_center_llff = torch.inverse(w2v_llff)[:3, 3]
        near_llff = float(cam_center_llff.norm())

        scale = near_llff / float(z_near_stv2)

        print(f"  [Align] STV2 near depth (5th pct): {z_near_stv2:.3f}")
        print(f"  [Align] LLFF cam01 dist from origin: {near_llff:.3f}")
        print(f"  [Align] Scale factor: {scale:.2f}")

        # Print diagnostics before alignment
        valid = pts_flat.abs().sum(-1) > 1e-6
        print(f"  [Align] Before: Z_cam=[{z_stv2[valid].min():.2f}, {z_stv2[valid].max():.2f}]")

        # Lateral correction: VGGT's predicted focal may differ from the true capture
        # focal (LLFF). This compresses X,Y by fx_vggt/fx_llff. We expand them back
        # in STV2 camera space before the similarity transform.
        lateral_x = lateral_y = 1.0
        intrs_path = os.path.join(cache_dir, "intrs.pt")
        if os.path.exists(intrs_path):
            intrs = torch.load(intrs_path, weights_only=True).float()
            fx_vggt = intrs[0, 0, 0].item()
            fy_vggt = intrs[0, 1, 1].item()
            fx_llff = cam01.image_width / (2.0 * math.tan(cam01.FoVx / 2.0))
            fy_llff = cam01.image_height / (2.0 * math.tan(cam01.FoVy / 2.0))
            lateral_x = fx_vggt / fx_llff
            lateral_y = fy_vggt / fy_llff
            print(f"  [Align] VGGT focal: fx={fx_vggt:.2f}, fy={fy_vggt:.2f}  "
                  f"LLFF focal: fx={fx_llff:.2f}, fy={fy_llff:.2f}")
            print(f"  [Align] Lateral correction: x*={lateral_x:.3f}, y*={lateral_y:.3f}")

        # Apply similarity transform: p_llff = R_c2w @ (s * R_stv2_inv @ p_stv2) + t_c2w
        # Since stv2_c2w ≈ I, this simplifies to: p_llff = R_c2w @ (s * p_stv2) + t_c2w
        R_stv2_inv = stv2_inv[:3, :3]
        original_shape = self.points_map.shape
        pts = self.points_map.reshape(-1, 3)
        pts_cam_space = pts @ R_stv2_inv.T + stv2_inv[:3, 3]
        # Expand X, Y to match true FOV before scaling + rotation
        pts_cam_space = pts_cam_space * torch.tensor([lateral_x, lateral_y, 1.0],
                                                      dtype=pts_cam_space.dtype)
        pts_scaled = scale * pts_cam_space
        pts_aligned = pts_scaled @ R_c2w.T + t_c2w
        self.points_map = pts_aligned.reshape(original_shape)

        # Verify: project through cam01 W2V to check depth
        pts_verify = pts_aligned @ w2v_llff[:3, :3].T + w2v_llff[:3, 3]
        z_verify = pts_verify[:, 2]
        valid2 = z_verify > 0
        print(f"  [Align] After (cam01 depth): Z=[{z_verify[valid2].min():.2f}, {z_verify[valid2].max():.2f}]")
        print(f"  [Align] Expected: near={near_llff:.2f}")

        cam_center = cam01.camera_center.cpu()
        print(f"  [Align] LLFF cam01 center: [{cam_center[0]:.2f}, {cam_center[1]:.2f}, {cam_center[2]:.2f}]")

    def get_tokens_mean(self) -> torch.Tensor:
        return self.tokens_mean

    def get_tokens_frame(self, frame_idx: int) -> torch.Tensor:
        return self.tokens[frame_idx]

    def get_points_map_frame(self, frame_idx: int) -> torch.Tensor:
        return self.points_map[frame_idx]

    def load_frame_image(self, cam_name: str, frame_idx: int) -> torch.Tensor:
        path = os.path.join(self.cache_dir, "frames", cam_name, f"{frame_idx:06d}.jpg")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def sample_training_batch(self, batch_frames: int, supervision_cams: int):
        frame_indices = random.sample(range(self.num_frames), batch_frames)
        frame_indices.sort()

        other_cams = [c for c in self.train_cam_names if c != self.input_camera]
        selected_cams = [self.input_camera] + random.sample(
            other_cams, min(supervision_cams - 1, len(other_cams))
        )

        tokens_frames = torch.stack([self.tokens[i] for i in frame_indices])
        points_map_frames = torch.stack([self.points_map[i] for i in frame_indices])

        gt_images = {}
        for cam_name in selected_cams:
            imgs = torch.stack([self.load_frame_image(cam_name, i) for i in frame_indices])
            gt_images[cam_name] = imgs

        return {
            "frame_indices": frame_indices,
            "tokens_mean": self.tokens_mean,
            "tokens_frames": tokens_frames,
            "points_map_frames": points_map_frames,
            "gt_images": gt_images,
            "cam_names": selected_cams,
        }
