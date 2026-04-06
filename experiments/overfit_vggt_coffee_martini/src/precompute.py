"""Precompute VGGT features for overfit_vggt_coffee_martini.

Replaces the STV2+hook pattern. Uses facebook/vggt-1b directly:
  - model.aggregator() -> tokens [B, T, P, 2048]
  - model.point_head() -> pts3d [B, T, H, W, 3] world-space + confidence

Output files (same names as STV2 cache for train.py compatibility):
  tokens.pt        [T, P, 2048] float16
  points_map.pt    [T, H, W, 3] float16  -- world-space, no alignment needed
  unc_metric.pt    [T, H, W, 1] float16  -- confidence (alias of pts3d_conf)
  pts3d_conf.pt    [T, H, W, 1] float16  -- same, kept separately for downstream use

NOTE: poses.pt is intentionally NOT saved. The shared engine's align_points_to_llff
checks for poses.pt at startup; its absence triggers the guard that skips alignment.
VGGT pts3d are already in world space (first camera = world origin, OpenCV convention).
"""

import os
import sys
import glob
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm

# Add VGGT to path (cloned repo at refs/vggt)
VGGT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "refs", "vggt"))
sys.path.insert(0, VGGT_ROOT)

from vggt.models.vggt import VGGT

VGGT_W = 518   # 37 x 14 — exact multiple, no padding/interpolation
VGGT_H = 378   # 27 x 14 — exact multiple, no padding/interpolation
PATCH_START_IDX = 5  # 1 camera token + 4 register tokens (constant in VGGT)


def extract_frames(scene_dir: str, cache_dir: str, target_w: int = VGGT_W, target_h: int = VGGT_H):
    """Extract frames from all camera MP4s, resize to target_w x target_h, save as JPEG."""
    cam_files = sorted(glob.glob(os.path.join(scene_dir, "cam*.mp4")))
    print(f"Found {len(cam_files)} cameras")

    for cam_path in tqdm(cam_files, desc="Extracting frames"):
        cam_name = os.path.splitext(os.path.basename(cam_path))[0]
        out_dir = os.path.join(cache_dir, "frames", cam_name)
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(cam_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, f"{frame_idx:06d}.jpg"), frame)
            frame_idx += 1
        cap.release()
        print(f"  {cam_name}: {frame_idx} frames @ {target_w}x{target_h}")


def load_frames(cache_dir: str, cam_name: str, num_frames: int,
                target_w: int = VGGT_W, target_h: int = VGGT_H) -> torch.Tensor:
    """Load saved frames as float32 [T, 3, H, W] in [0, 1]."""
    frames = []
    frame_dir = os.path.join(cache_dir, "frames", cam_name)
    for i in range(num_frames):
        path = os.path.join(frame_dir, f"{i:06d}.jpg")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)  # [3, H, W]
    return torch.stack(frames)  # [T, 3, H, W]


def _run_window(model, frames_window):
    """Run VGGT on a single window. Returns (tokens [T,P,2048], pts3d [T,H,W,3], conf [T,H,W,1])."""
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            aggregated_tokens_list, patch_start_idx = model.aggregator(frames_window)
            pts3d, pts3d_conf = model.point_head(
                aggregated_tokens_list, frames_window, patch_start_idx
            )
    # Extract patch tokens: [1, T, P_full, 2048] -> [T, P, 2048]
    tokens = aggregated_tokens_list[-1][:, :, patch_start_idx:].squeeze(0).cpu()
    pts = pts3d.squeeze(0).cpu()    # [T, H, W, 3]
    conf = pts3d_conf.squeeze(0).cpu()  # [T, H, W, 1]
    return tokens, pts, conf


def precompute_vggt(cache_dir: str, cam_name: str, num_frames: int,
                    window_size: int = 8) -> None:
    """
    Run VGGT on the input camera frames and save feature cache.

    Args:
        cache_dir: path where frames/ subdir already exists and outputs will be saved
        cam_name: name of the input camera (e.g. 'cam01')
        num_frames: total frames to process
        window_size: frames per forward pass (8 fits in 12GB at 518x378)
    """
    print(f"\n=== Loading VGGT (facebook/vggt-1b) ===")
    model = VGGT.from_pretrained("facebook/vggt-1b")
    model.eval()
    model = model.cuda()
    print(f"  Model loaded. aggregator: {type(model.aggregator).__name__}")

    print(f"=== Loading frames from {cam_name} ({num_frames} frames) ===")
    all_frames = load_frames(cache_dir, cam_name, num_frames)  # [T, 3, H, W]

    all_tokens = []
    all_pts3d = []
    all_conf = []

    print(f"=== Running VGGT (window={window_size}) ===")
    for start in tqdm(range(0, num_frames, window_size), desc="VGGT inference"):
        end = min(start + window_size, num_frames)
        n = end - start
        # images: [B=1, T, 3, H, W], float32 [0,1]
        frames_window = all_frames[start:end].unsqueeze(0).cuda()

        torch.cuda.empty_cache()
        try:
            tokens, pts, conf = _run_window(model, frames_window)
            all_tokens.append(tokens[:n])
            all_pts3d.append(pts[:n])
            all_conf.append(conf[:n])
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            half = max(1, (end - start) // 2)
            print(f"\n  OOM at window {start}-{end}, retrying with window={half}")
            for sub_start in range(start, end, half):
                sub_end = min(sub_start + half, end)
                sub_n = sub_end - sub_start
                sub_frames = all_frames[sub_start:sub_end].unsqueeze(0).cuda()
                tokens, pts, conf = _run_window(model, sub_frames)
                all_tokens.append(tokens[:sub_n])
                all_pts3d.append(pts[:sub_n])
                all_conf.append(conf[:sub_n])

    # Concatenate and save
    tokens = torch.cat(all_tokens, dim=0).half()        # [T, P, 2048]
    points_map = torch.cat(all_pts3d, dim=0).half()     # [T, H, W, 3]
    conf = torch.cat(all_conf, dim=0).half()             # [T, H, W, 1]

    print(f"\n=== Cache tensor shapes ===")
    print(f"  tokens:     {tokens.shape}  ({tokens.element_size() * tokens.nelement() / 1e6:.1f} MB)")
    print(f"  points_map: {points_map.shape}  ({points_map.element_size() * points_map.nelement() / 1e6:.1f} MB)")
    print(f"  conf:       {conf.shape}")

    torch.save(tokens, os.path.join(cache_dir, "tokens.pt"))
    torch.save(points_map, os.path.join(cache_dir, "points_map.pt"))
    torch.save(conf, os.path.join(cache_dir, "unc_metric.pt"))   # alias: train.py loads this name
    torch.save(conf, os.path.join(cache_dir, "pts3d_conf.pt"))   # kept separately for opacity init

    # Save poses.pt so train.py's align_points_to_llff fires correctly.
    # VGGT places frame 0 camera exactly at world origin (c2w = identity) by convention.
    # The alignment code uses poses[0] to transform pts → cam0 space → scale → LLFF world.
    # With c2w_0 = I, the cam0 transform is a no-op and depth-matching heuristic handles scale.
    poses = torch.eye(4).unsqueeze(0).expand(len(all_tokens), -1, -1).contiguous()  # [T, 4, 4]
    torch.save(poses.float(), os.path.join(cache_dir, "poses.pt"))

    del model
    torch.cuda.empty_cache()
    print("Done.")
    print("Saved poses.pt as identity (VGGT frame-0 cam = world origin) to enable LLFF alignment.")


def main():
    parser = argparse.ArgumentParser(description="Precompute VGGT features for overfit_vggt_coffee_martini")
    parser.add_argument("--scene_dir", type=str,
                        default="D:/DecodeGaussians/datasets/neu3d/coffee_martini")
    parser.add_argument("--cache_dir", type=str,
                        default="D:/DecodeGaussians/experiments/overfit_vggt_coffee_martini/cache")
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--input_camera", type=str, default="cam01")
    parser.add_argument("--window_size", type=int, default=8,
                        help="Frames per VGGT forward pass. Reduce if OOM (retries at half automatically).")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print("=" * 60)
    print(f"Step 1: Extracting frames at {VGGT_W}x{VGGT_H}")
    print("=" * 60)
    extract_frames(args.scene_dir, args.cache_dir)

    print("\n" + "=" * 60)
    print("Step 2: Running VGGT (facebook/vggt-1b)")
    print("=" * 60)
    precompute_vggt(args.cache_dir, args.input_camera, args.num_frames, args.window_size)

    print("\n" + "=" * 60)
    print("Precompute complete!")
    print("=" * 60)
    for f in ["tokens.pt", "points_map.pt", "unc_metric.pt", "pts3d_conf.pt", "poses.pt"]:
        path = os.path.join(args.cache_dir, f)
        if os.path.exists(path):
            print(f"  {f}: {os.path.getsize(path) / 1e6:.1f} MB")
    print("\nNot saved (intentional): intrs.pt")
    print("Run training with:")
    print(f"  python src/train.py config.yaml")


if __name__ == "__main__":
    main()
