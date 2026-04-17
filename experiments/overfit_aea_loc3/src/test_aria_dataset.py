"""Smoke test for AriaSequenceDataset + camera builder.

Run with a mock transforms.json — does NOT require the 5-30 GB real dataset.
"""
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from aria_dataset import AriaSequenceDataset
from aria_cameras import build_cameras_from_aria


def _make_fake_seq(tmp: str, n_frames: int = 5, w: int = 500, h: int = 500):
    img_dir = os.path.join(tmp, "images")
    os.makedirs(img_dir, exist_ok=True)
    frames = []
    for i in range(n_frames):
        img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        p = os.path.join(img_dir, f"{i:05d}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # Camera walks +X by 0.1 m per frame, identity rotation
        c2w = np.eye(4)
        c2w[0, 3] = i * 0.1
        frames.append({
            "fx": 150.0, "fy": 150.0, "cx": 250.0, "cy": 250.0,
            "w": w, "h": h,
            "image_path": f"images/{i:05d}.png",
            "transform_matrix": c2w.tolist(),
            "timestamp": i * int(1e8),
        })
    with open(os.path.join(tmp, "transforms.json"), "w") as f:
        json.dump({"frames": frames}, f)


def test_load_and_shape():
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_seq(tmp, n_frames=5)
        ds = AriaSequenceDataset(tmp, target_resolution=(256, 256))
        assert len(ds) == 5
        img = ds.get_frame_image(0)
        assert img.shape == (3, 256, 256)
        assert 0.0 <= img.min() and img.max() <= 1.0
        K = ds.get_intrinsics(0)
        # Focal scales by 256/500 = 0.512
        assert torch.allclose(K[0, 0], torch.tensor(150.0 * 0.512), atol=1e-3)
        stack = ds.get_frames_stack([0, 1, 2])
        assert stack.shape == (3, 3, 256, 256)


def test_camera_center_matches_c2w_translation():
    """Camera center (in world) should equal c2w[:3, 3]."""
    if not torch.cuda.is_available():
        print("SKIP test_camera_center_matches_c2w_translation (no CUDA)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_seq(tmp, n_frames=3)
        ds = AriaSequenceDataset(tmp, target_resolution=(256, 256))
        cams = build_cameras_from_aria(ds, [0, 1, 2])
        for i, cam in enumerate(cams):
            expected = torch.tensor([i * 0.1, 0.0, 0.0], device="cuda")
            got = cam.camera_center
            assert torch.allclose(got, expected, atol=1e-4), f"frame {i}: {got} vs {expected}"


if __name__ == "__main__":
    test_load_and_shape()
    test_camera_center_matches_c2w_translation()
    print("OK")
