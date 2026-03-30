"""Dataset for loading precomputed STV2 features and camera frames."""

import os
import random
import cv2
import torch
import numpy as np


class CachedSceneDataset:
    """
    Loads precomputed features from cache and serves training batches.
    All data stays on CPU; moved to GPU per-batch by the training loop.
    """

    def __init__(self, cache_dir: str, cameras: dict, input_camera: str = "cam01",
                 eval_camera: str = "cam00"):
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
        self._align_points_to_llff(cache_dir, cameras, input_camera)

        self.num_frames = self.tokens.shape[0]
        self.num_patches = self.tokens.shape[1]
        self.token_dim = self.tokens.shape[2]

        self.tokens_mean = self.tokens.mean(dim=0)

        print(f"  tokens: {self.tokens.shape}")
        print(f"  points_map: {self.points_map.shape}")
        print(f"  num_patches (P): {self.num_patches}")
        print(f"  train cameras: {len(self.train_cam_names)}")

    def _align_points_to_llff(self, cache_dir: str, cameras: dict, input_camera: str):
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

        # Load LLFF near bound for input camera
        scene_dir = os.path.dirname(os.path.dirname(cache_dir))
        # Try to get near/far from poses_bounds
        poses_bounds_path = None
        for key in ['neu3d_scene']:
            # Walk up from cache to find config
            pass
        # Use the near/far from the cameras dict - reconstruct from LLFF
        # Simpler: just use the LLFF near bound passed from outside or compute from data
        # For now, load poses_bounds directly
        import glob
        scene_dirs = [
            os.path.join(os.path.dirname(cache_dir), '..', '..', 'datasets', 'neu3d', 'coffee_martini'),
            'D:/DecodeGaussians/datasets/neu3d/coffee_martini',
        ]
        near_llff = 6.84  # fallback
        for sd in scene_dirs:
            pb_path = os.path.join(sd, 'poses_bounds.npy')
            if os.path.exists(pb_path):
                import numpy as np_
                pb = np_.load(pb_path)
                cam_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(sd, 'cam*.mp4'))])
                cam_idx = [f.replace('.mp4', '') for f in cam_files].index(input_camera)
                near_llff = float(pb[cam_idx, -2])
                break

        scale = near_llff / float(z_near_stv2)

        print(f"  [Align] STV2 near depth (5th pct): {z_near_stv2:.3f}")
        print(f"  [Align] LLFF near bound: {near_llff:.3f}")
        print(f"  [Align] Scale factor: {scale:.2f}")

        # Print diagnostics before alignment
        valid = pts_flat.abs().sum(-1) > 1e-6
        print(f"  [Align] Before: Z_cam=[{z_stv2[valid].min():.2f}, {z_stv2[valid].max():.2f}]")

        # Apply similarity transform: p_llff = R_c2w @ (s * R_stv2_inv @ p_stv2) + t_c2w
        # Since stv2_c2w ≈ I, this simplifies to: p_llff = R_c2w @ (s * p_stv2) + t_c2w
        R_stv2_inv = stv2_inv[:3, :3]
        original_shape = self.points_map.shape
        pts = self.points_map.reshape(-1, 3)
        pts_scaled = scale * (pts @ R_stv2_inv.T + stv2_inv[:3, 3])
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

    def sample_training_batch(self, batch_frames: int, supervision_cams: int,
                              frame_sampling: str = "random"):
        if frame_sampling == "contiguous" and batch_frames > 1:
            # Sample a contiguous window of batch_frames
            max_start = self.num_frames - batch_frames
            start = random.randint(0, max_start)
            frame_indices = list(range(start, start + batch_frames))
        else:
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
