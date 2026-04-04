"""Multi-scene dataset with configurable memory strategies."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.config import DataConfig, SceneConfig

# Map string dtype names to torch dtypes
_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_memory_strategy(strategy: str, num_scenes: int, vram_gb: float = 8.0) -> str:
    """Resolve 'auto' memory strategy based on scene count and available VRAM.

    Rules:
        - 1 scene       → "pinned"
        - 2–5 scenes    → "compressed"
        - 6+ scenes     → "streaming"
    Non-auto strategies are returned as-is.
    """
    if strategy != "auto":
        return strategy

    if num_scenes == 1:
        return "pinned"
    elif num_scenes <= 5:
        return "compressed"
    else:
        return "streaming"


class SceneDataset:
    """Dataset that holds one or more precomputed scenes in memory.

    Supports four memory strategies:
        - pinned:     All tensors loaded eagerly onto ``device`` in full precision.
        - compressed: Tokens stored as float16 to halve memory footprint.
        - rotation:   Scenes rotated in/out of GPU memory as needed.
        - streaming:  Tensors kept on CPU; transferred per-batch.
    """

    def __init__(
        self,
        config: DataConfig,
        device: str = "cuda",
        cameras: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self.device = device
        self.cameras = cameras

        num_scenes = len(config.scenes)
        self.strategy = resolve_memory_strategy(
            config.memory_strategy, num_scenes=num_scenes
        )

        token_dtype_str = getattr(config, "token_dtype", "float32")
        self.token_dtype = _DTYPE_MAP.get(token_dtype_str, torch.float32)

        # For compressed strategy, always use float16 for tokens
        if self.strategy == "compressed":
            self.token_dtype = torch.float16

        self.scenes: Dict[str, dict] = {}
        for scene_cfg in config.scenes:
            self.scenes[scene_cfg.name] = self._load_scene(scene_cfg, self.token_dtype)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_scenes(self) -> int:
        return len(self.scenes)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_scene(self, scene_cfg: SceneConfig, token_dtype: torch.dtype) -> dict:
        """Load a single precomputed scene from disk.

        Returns a dict with:
            tokens          [T, P, D]
            tokens_mean     [P, D]  (time-averaged)
            points_map      [T, H, W, 3]
            num_frames      int
            num_patches     int
            cache_dir       Path
            gt_images_cache dict
        """
        cache_dir = Path(scene_cfg.precomputed)

        tokens = torch.load(cache_dir / "tokens.pt", map_location="cpu", weights_only=True)
        points_map = torch.load(cache_dir / "points_map.pt", map_location="cpu", weights_only=True)

        # Convert tokens to requested dtype
        tokens = tokens.to(dtype=token_dtype)

        # Compute time-mean of tokens: [T, P, D] → [P, D]
        tokens_mean = tokens.float().mean(dim=0).to(dtype=token_dtype)

        # Move to device for pinned/compressed strategies
        if self.strategy in ("pinned", "compressed"):
            tokens = tokens.to(self.device)
            tokens_mean = tokens_mean.to(self.device)
            points_map = points_map.to(self.device)

        num_frames = tokens.shape[0]
        num_patches = tokens.shape[1]

        return {
            "tokens": tokens,
            "tokens_mean": tokens_mean,
            "points_map": points_map,
            "num_frames": num_frames,
            "num_patches": num_patches,
            "cache_dir": cache_dir,
            "gt_images_cache": {},
        }

    def _load_frame_image(
        self, cache_dir: Path, cam_name: str, frame_idx: int
    ) -> torch.Tensor:
        """Load a single frame image as a float32 [3, H, W] tensor in [0, 1].

        Tries .pt format first (used in tests), then falls back to JPEG.
        """
        pt_path = cache_dir / "frames" / cam_name / f"{frame_idx:06d}.pt"
        if pt_path.exists():
            img = torch.load(pt_path, map_location="cpu", weights_only=True)
            # img shape: [H, W, 3] uint8
            img = img.float() / 255.0
            img = img.permute(2, 0, 1)  # [3, H, W]
            return img

        # Fallback: JPEG via cv2
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                f"cv2 not available and no .pt file found at {pt_path}"
            ) from exc

        jpg_path = cache_dir / "frames" / cam_name / f"{frame_idx:06d}.jpg"
        bgr = cv2.imread(str(jpg_path))
        if bgr is None:
            raise FileNotFoundError(f"Frame not found: {jpg_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).float() / 255.0
        img = img.permute(2, 0, 1)  # [3, H, W]
        return img

    # ------------------------------------------------------------------
    # Batch sampling
    # ------------------------------------------------------------------

    def sample_training_batch(
        self,
        batch_scenes: int = 1,
        batch_frames: int = 1,
        supervision_cams: Optional[int] = None,
    ) -> dict:
        """Sample a random training batch across scenes and frames.

        Returns dict with:
            tokens_mean       list[Tensor [P, D]]     one per scene
            tokens_frames     list[Tensor [F, P, D]]  one per scene
            points_map_frames list[Tensor [F, H, W, 3]] one per scene
            scene_names       list[str]
            frame_indices     list[list[int]]
        """
        scene_names_all = list(self.scenes.keys())
        chosen_scenes = random.sample(
            scene_names_all, k=min(batch_scenes, len(scene_names_all))
        )

        tokens_mean_list = []
        tokens_frames_list = []
        points_map_frames_list = []
        frame_indices_list = []

        for scene_name in chosen_scenes:
            scene = self.scenes[scene_name]
            num_frames = scene["num_frames"]

            chosen_frames = [
                random.randint(0, num_frames - 1) for _ in range(batch_frames)
            ]
            frame_indices_list.append(chosen_frames)

            tokens_mean_list.append(scene["tokens_mean"])

            # Gather token frames: [F, P, D]
            t_frames = torch.stack([scene["tokens"][f] for f in chosen_frames], dim=0)
            tokens_frames_list.append(t_frames)

            # Gather points_map frames: [F, H, W, 3]
            pm_frames = torch.stack([scene["points_map"][f] for f in chosen_frames], dim=0)
            points_map_frames_list.append(pm_frames)

        return {
            "tokens_mean": tokens_mean_list,
            "tokens_frames": tokens_frames_list,
            "points_map_frames": points_map_frames_list,
            "scene_names": chosen_scenes,
            "frame_indices": frame_indices_list,
        }

    def get_eval_batch(self, scene_name: str, frame_idx: int) -> dict:
        """Return a single-frame eval batch for one scene.

        Returns dict with:
            tokens_mean   Tensor [P, D]
            tokens_frame  Tensor [P, D]
            points_map    Tensor [H, W, 3]
        """
        scene = self.scenes[scene_name]
        return {
            "tokens_mean": scene["tokens_mean"],
            "tokens_frame": scene["tokens"][frame_idx],
            "points_map": scene["points_map"][frame_idx],
        }

    # ------------------------------------------------------------------
    # GT image cache
    # ------------------------------------------------------------------

    def cache_gt_images(self, scene_name: str, cam_names: List[str]) -> None:
        """Pre-load and cache all GT images for a scene onto device."""
        scene = self.scenes[scene_name]
        cache_dir = scene["cache_dir"]
        num_frames = scene["num_frames"]
        gt_cache = scene["gt_images_cache"]

        for cam_name in cam_names:
            if cam_name not in gt_cache:
                gt_cache[cam_name] = {}
            for f_idx in range(num_frames):
                if f_idx not in gt_cache[cam_name]:
                    img = self._load_frame_image(cache_dir, cam_name, f_idx)
                    gt_cache[cam_name][f_idx] = img.to(self.device)

    def get_gt_image(
        self, scene_name: str, cam_name: str, frame_idx: int
    ) -> torch.Tensor:
        """Return GT image [3, H, W] float32 in [0,1]; loads from cache if available."""
        scene = self.scenes[scene_name]
        gt_cache = scene["gt_images_cache"]

        if cam_name in gt_cache and frame_idx in gt_cache[cam_name]:
            return gt_cache[cam_name][frame_idx]

        img = self._load_frame_image(scene["cache_dir"], cam_name, frame_idx)
        return img.to(self.device)
