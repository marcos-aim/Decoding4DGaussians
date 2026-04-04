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
                 eval_camera: str = "cam00", normalize_coords: bool = True):
        # Temporarily disable CUDA to prevent parent from caching to GPU
        # Multi-scene training requires CPU-based data management to avoid
        # loading all scenes into VRAM simultaneously
        cuda_was_available = torch.cuda.is_available()
        if cuda_was_available:
            # Monkey-patch torch.cuda.is_available() to return False during parent init
            original_is_available = torch.cuda.is_available
            torch.cuda.is_available = lambda: False

        # Call parent constructor (will keep data on CPU due to our patch)
        super().__init__(cache_dir, cameras, input_camera, eval_camera)

        # Restore CUDA availability check
        if cuda_was_available:
            torch.cuda.is_available = original_is_available
            print(f"  [Multi-Scene] Data kept on CPU for memory efficiency")
            print(f"  [Multi-Scene] CPU→GPU transfers will occur per-batch during training")

        self.normalize_coords = normalize_coords
        self.norm_center = None
        self.norm_scale = None

        if normalize_coords:
            self._normalize_coordinates()

    def _normalize_coordinates(self):
        """
        Normalize points and camera poses to [-1, 1]³ cube.
        This enables the model to learn scene-invariant features.
        """
        print("  [Normalize] Computing scene bounds...")

        # Compute scene center and extent from points
        pts_flat = self.points_map.reshape(-1, 3)
        valid = pts_flat.abs().sum(-1) > 1e-6
        pts_valid = pts_flat[valid]

        if len(pts_valid) == 0:
            print("  [WARN] No valid points found, skipping normalization")
            return

        # Use center of mass and max extent
        self.norm_center = pts_valid.mean(dim=0)  # Keep on CPU (points_map is on CPU)
        pts_centered = pts_valid - self.norm_center  # Both on CPU

        # Scale to fit in [-1, 1] cube (use max extent across all axes)
        max_extent = pts_centered.abs().max()
        self.norm_scale = max_extent * 1.1  # Add 10% margin (scalar, broadcasts correctly)

        print(f"  [Normalize] Center: [{self.norm_center[0]:.2f}, {self.norm_center[1]:.2f}, {self.norm_center[2]:.2f}]")
        print(f"  [Normalize] Scale: {self.norm_scale:.2f}")

        # Normalize points (compute on CPU since points_map is on CPU)
        original_shape = self.points_map.shape
        pts = self.points_map.reshape(-1, 3)
        pts_normalized = (pts - self.norm_center) / self.norm_scale
        self.points_map = pts_normalized.reshape(original_shape)

        # Verify bounds
        pts_check = self.points_map.reshape(-1, 3)
        valid_check = pts_check.abs().sum(-1) > 1e-6
        pts_valid_norm = pts_check[valid_check]
        print(f"  [Normalize] After normalization:")
        print(f"    Min: [{pts_valid_norm.min(dim=0)[0].tolist()}]")
        print(f"    Max: [{pts_valid_norm.max(dim=0)[0].tolist()}]")
        print(f"    Mean: [{pts_valid_norm.mean(dim=0).tolist()}]")

        # Transform camera poses
        # Camera centers: c_new = (c_old - center) / scale
        for cam_name, cam in self.cameras.items():
            cam_center_old = cam.camera_center.clone()
            cam.camera_center = (cam.camera_center - self.norm_center.to(cam.camera_center.device)) / self.norm_scale

            # Update world_view_transform
            # W2V transforms world points to camera space
            # Let p_world_new = (p_world_old - center) / scale
            # p_cam = W2V_old @ p_world_old = W2V_old @ (scale * p_world_new + center)
            #      = (W2V_old @ scale * p_world_new) + (W2V_old @ center)
            #      = scale * (W2V_old @ p_world_new) + W2V_old @ center
            # We want: p_cam = W2V_new @ p_world_new
            # So: W2V_new = scale * W2V_old, with adjusted translation

            w2v_old = cam.world_view_transform.clone()
            R_w2v = w2v_old[:3, :3]
            t_w2v = w2v_old[:3, 3]

            # New transformation
            R_w2v_new = self.norm_scale * R_w2v
            t_w2v_new = self.norm_scale * t_w2v + R_w2v @ self.norm_center.to(R_w2v.device)

            cam.world_view_transform[:3, :3] = R_w2v_new
            cam.world_view_transform[:3, 3] = t_w2v_new

            if cam_name == self.input_camera:
                print(f"  [Normalize] {cam_name} camera center: {cam_center_old.cpu()} → {cam.camera_center.cpu()}")


class MultiSceneDataset:
    """
    Wrapper for multiple CachedSceneDataset instances.
    Enables gradient accumulation across scenes for feedforward learning.
    """

    def __init__(self, scene_configs: list, normalize_coords: bool = True):
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
