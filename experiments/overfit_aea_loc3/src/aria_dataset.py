"""AriaSequenceDataset: loads 4DGT's transforms.json format (monocular posed video).

Aria coordinates are metric world-anchored (MPS closed-loop SLAM), so no
similarity alignment is needed — unlike our LLFF/STV2 coffee_martini pipeline.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch


@dataclass
class AriaFrame:
    fx: float
    fy: float
    cx: float
    cy: float
    w: int
    h: int
    image_path: str          # relative to seq_root
    transform_matrix: np.ndarray  # 4x4 camera-to-world (OpenCV/Aria convention)
    timestamp: int           # nanoseconds


class AriaSequenceDataset:
    """Loads a 4DGT-format Aria sequence directory.

    Args:
        seq_root: path to .../recording/camera-rgb-rectified-600-h1000/
        target_resolution: (W, H) to resize to. None = keep native.
        frame_indices: subset of frames to keep (list of ints into transforms.json["frames"]).
                       None = use all.
    """

    def __init__(self, seq_root: str, target_resolution: Tuple[int, int] = None,
                 frame_indices: List[int] = None):
        self.seq_root = seq_root
        tj = os.path.join(seq_root, "transforms.json")
        with open(tj) as f:
            meta = json.load(f)

        all_frames: List[AriaFrame] = []
        for f in meta["frames"]:
            all_frames.append(AriaFrame(
                fx=float(f["fx"]),
                fy=float(f["fy"]),
                cx=float(f["cx"]),
                cy=float(f["cy"]),
                w=int(f["w"]),
                h=int(f["h"]),
                image_path=f["image_path"],
                transform_matrix=np.asarray(f["transform_matrix"], dtype=np.float64),
                timestamp=int(f["timestamp"]),
            ))
        all_frames.sort(key=lambda x: x.timestamp)

        if frame_indices is None:
            self.frames = all_frames
        else:
            self.frames = [all_frames[i] for i in frame_indices]

        self.target_resolution = target_resolution
        self.orig_w = self.frames[0].w
        self.orig_h = self.frames[0].h
        if target_resolution is None:
            self.target_w, self.target_h = self.orig_w, self.orig_h
        else:
            self.target_w, self.target_h = target_resolution

    def __len__(self) -> int:
        return len(self.frames)

    def get_frame_image(self, idx: int) -> torch.Tensor:
        """Return frame as [3, H, W] float tensor in [0, 1], RGB."""
        f = self.frames[idx]
        img_path = os.path.join(self.seq_root, f.image_path)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (self.target_w, self.target_h) != (f.w, f.h):
            rgb = cv2.resize(rgb, (self.target_w, self.target_h),
                             interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t

    def get_frames_stack(self, indices: List[int]) -> torch.Tensor:
        """Return [N, 3, H, W] stacked float tensor in [0, 1]."""
        return torch.stack([self.get_frame_image(i) for i in indices], dim=0)

    def get_intrinsics(self, idx: int) -> torch.Tensor:
        """Return 3x3 intrinsic matrix scaled to target resolution."""
        f = self.frames[idx]
        sx = self.target_w / f.w
        sy = self.target_h / f.h
        K = torch.tensor([
            [f.fx * sx, 0.0,        f.cx * sx],
            [0.0,       f.fy * sy,  f.cy * sy],
            [0.0,       0.0,        1.0],
        ], dtype=torch.float32)
        return K

    def get_c2w(self, idx: int) -> torch.Tensor:
        """Return 4x4 camera-to-world matrix (float32)."""
        return torch.from_numpy(self.frames[idx].transform_matrix).float()

    def get_w2c(self, idx: int) -> torch.Tensor:
        return torch.inverse(self.get_c2w(idx))
