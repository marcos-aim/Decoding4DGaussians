"""
Multi-scene dataset wrapper for gradient accumulation training.
Extends the single-scene CachedSceneDataset to support multiple scenes.
"""

import os
import random
import cv2
import torch
import numpy as np
import sys
import glob

# Import single-scene dataset - use explicit module path to avoid circular import
_parent_dataset_path = os.path.join(os.path.dirname(__file__), "../../overfit_coffee_martini/src")
# Note: Don't modify sys.path - importlib handles loading without polluting global path
# if _parent_dataset_path not in sys.path:
#     sys.path.insert(0, _parent_dataset_path)

import importlib.util
spec = importlib.util.spec_from_file_location("base_dataset", os.path.join(_parent_dataset_path, "dataset.py"))
base_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_dataset_module)
BaseCachedSceneDataset = base_dataset_module.CachedSceneDataset


class CachedSceneDataset(BaseCachedSceneDataset):
    """
    Extended dataset with coordinate normalization support.
    Normalizes scene to [-1, 1]³ cube for cross-scene compatibility.

    Data Management:
    - Inherits from GPU-cached parent but reverts to CPU storage
    - Enables multi-scene training without exhausting VRAM
    - Batch data transferred CPU→GPU on-demand during training
    - VRAM savings: ~1 GB per scene compared to GPU caching
    """

    def __init__(self, cache_dir: str, cameras: dict, input_camera: str = "cam01",
                 eval_camera: str = "cam00", normalize_coords: bool = True, K: int = 1):
        # Temporarily disable CUDA to prevent parent from caching to GPU
        cuda_was_available = torch.cuda.is_available()
        if cuda_was_available:
            original_is_available = torch.cuda.is_available
            torch.cuda.is_available = lambda: False

        super().__init__(cache_dir, cameras, input_camera, eval_camera)

        if cuda_was_available:
            torch.cuda.is_available = original_is_available
            print(f"  [Multi-Scene] Data kept on CPU for memory efficiency")
            print(f"  [Multi-Scene] CPU→GPU transfers will occur per-batch during training")

        self.normalize_coords = normalize_coords
        self.norm_center = None
        self.norm_scale = None

        if normalize_coords:
            self._normalize_coordinates()

        self.xyz_anchor = None
        self.scale_anchor = None
        self._compute_anchors(K)

    def _normalize_coordinates(self):
        """
        Compute (norm_center, norm_scale) to map this scene to a [-1, 1]³ cube.

        Option A (denormalize-before-render): points and cameras are NOT mutated
        here. self.points_map stays in world space so the rasterizer (and the
        geometric loss) see valid world-space geometry. The norm_center /
        norm_scale values are used downstream by:
          - _compute_anchors (to compute xyz/scale anchors in normalized space)
          - train.py denormalize_gaussians (to map NN outputs back to world space)
        """
        print("  [Normalize] Computing scene bounds...")

        pts_flat = self.points_map.reshape(-1, 3)
        valid = pts_flat.abs().sum(-1) > 1e-6
        pts_valid = pts_flat[valid]

        if len(pts_valid) == 0:
            print("  [WARN] No valid points found, skipping normalization")
            return

        # Center of mass and max extent from valid (non-zero) world-space points
        self.norm_center = pts_valid.mean(dim=0)                     # [3] on CPU
        max_extent = (pts_valid - self.norm_center).abs().max()
        self.norm_scale = max_extent * 1.1                           # 10% margin

        print(f"  [Normalize] Center: [{self.norm_center[0]:.2f}, {self.norm_center[1]:.2f}, {self.norm_center[2]:.2f}]")
        print(f"  [Normalize] Scale: {self.norm_scale:.2f}")
        print(f"  [Normalize] points_map and cameras are left in WORLD space (Option A: denormalize before render)")


    def _compute_anchors(self, K: int):
        """
        Compute per-patch xyz and scale anchors in NORMALIZED [-1,1]³ space.
        xyz_anchor  [P*K, 3]: K sample points per patch (normalized coords).
        scale_anchor [P*K, 3]: log(nn_dist) between patch centroids (normalized),
                               broadcast across K Gaussians and 3 scale axes.
        Stored as self.xyz_anchor and self.scale_anchor (CPU tensors).

        Note: self.points_map is in WORLD space (Option A). We normalize a local
        copy here for anchor computation only; the world-space points are still
        served by sample_training_batch for the geometric loss and kept as the
        rasterizer's geometric ground truth.
        """
        # Local normalization for anchor computation only — self.points_map untouched
        if self.norm_center is not None and self.norm_scale is not None:
            pts_norm = (self.points_map - self.norm_center) / self.norm_scale
        else:
            pts_norm = self.points_map

        pts_mean = pts_norm.mean(dim=0)          # [H, W, 3] in normalized space
        H, W = pts_mean.shape[0], pts_mean.shape[1]
        pts_flat = pts_mean.reshape(-1, 3)       # [H*W, 3]
        P = self.num_patches

        # Zero/near-zero STV2 points map to a single camera-position cluster after
        # alignment. Detect them as the points within 1% of the global L2-norm range.
        # These background-cluster points corrupt patch centroids and nn_dist.
        pts_norms = pts_flat.norm(dim=-1)
        norm_threshold = pts_norms.min() + 0.01 * (pts_norms.max() - pts_norms.min())
        valid_global = pts_norms > norm_threshold
        pts_valid_global = pts_flat[valid_global]
        global_fallback = pts_valid_global.mean(dim=0) if pts_valid_global.shape[0] > 0 else pts_flat.mean(dim=0)

        # Build patch grid
        grid_w = round((P * W / H) ** 0.5)
        grid_h = round(P / grid_w)
        while grid_w * grid_h < P:
            grid_w += 1

        patch_h = H / grid_h
        patch_w = W / grid_w
        pixel_y = torch.arange(H).float()
        pixel_x = torch.arange(W).float()
        yy, xx = torch.meshgrid(pixel_y, pixel_x, indexing="ij")
        patch_idx_y = (yy / patch_h).long().clamp(0, grid_h - 1)
        patch_idx_x = (xx / patch_w).long().clamp(0, grid_w - 1)
        pixel_to_patch = (patch_idx_y * grid_w + patch_idx_x).reshape(-1)  # [H*W]

        # Sample K points per patch for xyz_anchor
        init_xyz_per_gaussian = torch.zeros(P, K, 3)
        init_xyz = torch.zeros(P, 3)   # patch centroid (valid points only)
        for p in range(P):
            mask = (pixel_to_patch == p)
            pts_p = pts_flat[mask]

            # Keep only valid (non-background-cluster) points
            valid_p = pts_p.norm(dim=-1) > norm_threshold
            pts_p_valid = pts_p[valid_p]
            if pts_p_valid.shape[0] == 0:
                pts_p_valid = global_fallback.unsqueeze(0)

            # Centroid of valid points (robust to scattered bad pixels)
            init_xyz[p] = pts_p_valid.mean(dim=0)

            if pts_p_valid.shape[0] >= K:
                idx = torch.linspace(0, pts_p_valid.shape[0] - 1, K).long()
                init_xyz_per_gaussian[p] = pts_p_valid[idx]
            else:
                repeats = K // pts_p_valid.shape[0] + 1
                init_xyz_per_gaussian[p] = pts_p_valid.repeat(repeats, 1)[:K]

        self.xyz_anchor = init_xyz_per_gaussian.reshape(P * K, 3)   # [P*K, 3]

        # NN distances between patch centroids → scale anchor
        dists = torch.cdist(init_xyz, init_xyz)   # [P, P]
        dists.fill_diagonal_(float('inf'))
        nn_dist = dists.min(dim=1).values.clamp(min=1e-6)   # [P]
        log_nn = torch.log(nn_dist)                          # [P]

        # Clamp extreme outliers: no patch worse than 2 log-units below the median
        # (guards against any remaining degenerate patches after the validity filter)
        log_nn = log_nn.clamp(min=(log_nn.median() - 2.0).item())

        # Expand: each patch's log_nn shared across K Gaussians and 3 scale axes
        scale_anchor = (
            log_nn.unsqueeze(1)          # [P, 1]
            .expand(-1, K)               # [P, K]
            .reshape(P * K)              # [P*K]
            .unsqueeze(1)                # [P*K, 1]
            .expand(-1, 3)               # [P*K, 3]
            .contiguous()
        )
        self.scale_anchor = scale_anchor   # [P*K, 3]

        print(f"  [Anchors] Patch grid: {grid_h}x{grid_w}, P={P}, K={K}")
        print(f"  [Anchors] Valid pixels: {valid_global.sum().item()}/{pts_flat.shape[0]} ({100*valid_global.float().mean():.1f}%)")
        print(f"  [Anchors] scale_anchor (log nn_dist): min={log_nn.min():.3f}, mean={log_nn.mean():.3f}, max={log_nn.max():.3f}")


class MultiSceneDataset:
    """
    Wrapper for multiple CachedSceneDataset instances.
    Enables gradient accumulation across scenes for feedforward learning.
    """

    def __init__(self, scene_configs: list, normalize_coords: bool = True, K: int = 1):
        """
        Args:
            scene_configs: List of dicts with keys: name, cache_dir, cameras, input_camera, eval_camera, weight
            normalize_coords: If True, normalize each scene to unit cube
        """
        self.scenes = []
        self.normalize_coords = normalize_coords

        print(f"\n{'='*60}")
        print(f"Initializing MultiSceneDataset ({len(scene_configs)} scenes)")
        print(f"{'='*60}")

        for i, cfg in enumerate(scene_configs):
            print(f"\n[{i+1}/{len(scene_configs)}] Loading scene: {cfg['name']}")
            print(f"  Cache: {cfg['cache_dir']}")

            dataset = CachedSceneDataset(
                cache_dir=cfg["cache_dir"],
                cameras=cfg["cameras"],
                input_camera=cfg.get("input_camera", "cam01"),
                eval_camera=cfg.get("eval_camera", "cam00"),
                normalize_coords=normalize_coords,
                K=K,
            )

            scene_info = {
                "name": cfg["name"],
                "dataset": dataset,
                "cameras": cfg["cameras"],
                "weight": cfg.get("weight", 1.0),
                "input_camera": cfg.get("input_camera", "cam01"),
                "eval_camera": cfg.get("eval_camera", "cam00"),
            }

            self.scenes.append(scene_info)

            print(f"  ✓ Loaded {dataset.num_frames} frames, {dataset.num_patches} patches")
            print(f"    Weight: {scene_info['weight']}")

        print(f"\n{'='*60}")
        print(f"✓ Multi-scene dataset initialized")
        print(f"  Total scenes: {len(self.scenes)}")
        print(f"  Total frames: {sum(s['dataset'].num_frames for s in self.scenes)}")
        print(f"{'='*60}\n")

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> dict:
        return self.scenes[idx]

    def sample_multi_scene_batch(self, batch_frames_per_scene: int, supervision_cams: int):
        """
        Sample training batches from all scenes.

        Args:
            batch_frames_per_scene: Number of frames to sample per scene
            supervision_cams: Number of cameras for multi-view supervision

        Returns:
            List of batch dicts, one per scene
        """
        batches = []

        for scene_info in self.scenes:
            batch = scene_info["dataset"].sample_training_batch(
                batch_frames_per_scene, supervision_cams
            )

            # Add scene metadata
            batch["scene_name"] = scene_info["name"]
            batch["scene_weight"] = scene_info["weight"]
            batch["cameras"] = scene_info["cameras"]

            batches.append(batch)

        return batches

    def get_scene_by_name(self, name: str):
        """Get scene info by name."""
        for scene in self.scenes:
            if scene["name"] == name:
                return scene
        return None

    def print_statistics(self):
        """Print dataset statistics."""
        print(f"\n{'='*60}")
        print("Multi-Scene Dataset Statistics")
        print(f"{'='*60}")

        for scene in self.scenes:
            dataset = scene["dataset"]
            print(f"\n{scene['name']}:")
            print(f"  Frames: {dataset.num_frames}")
            print(f"  Patches: {dataset.num_patches}")
            print(f"  Cameras: {len(dataset.cameras)}")
            print(f"  Train cameras: {len(dataset.train_cam_names)}")
            print(f"  Weight: {scene['weight']}")

            if dataset.norm_center is not None:
                print(f"  Normalized: Yes")
                print(f"    Center: {dataset.norm_center.cpu().numpy()}")
                print(f"    Scale: {dataset.norm_scale:.2f}")
            else:
                print(f"  Normalized: No")

            # Token statistics
            tokens_mean = dataset.tokens_mean
            print(f"  Token stats:")
            print(f"    Mean: {tokens_mean.mean().item():.3f}")
            print(f"    Std: {tokens_mean.std().item():.3f}")

        print(f"{'='*60}\n")
