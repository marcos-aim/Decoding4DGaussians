"""Precompute SpaTrackerV2 features and extract frames for training."""

import os
import sys
import glob
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

# Add SpaTrackerV2 to path
SPATRACKER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "SpaTrackerV2"))
sys.path.insert(0, SPATRACKER_ROOT)

from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image


def extract_frames(scene_dir: str, cache_dir: str, target_w: int, target_h: int):
    """Extract frames from all camera MP4s, resize, and save as JPEG."""
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
        print(f"  {cam_name}: {frame_idx} frames")

    return cam_files


def load_frames_for_stv2(cache_dir: str, cam_name: str, num_frames: int,
                          target_w: int, target_h: int) -> torch.Tensor:
    """Load extracted frames as a tensor for SpaTrackerV2 input."""
    frames = []
    frame_dir = os.path.join(cache_dir, "frames", cam_name)
    for i in range(num_frames):
        path = os.path.join(frame_dir, f"{i:06d}.jpg")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(img).permute(2, 0, 1).float())  # [3, H, W]
    return torch.stack(frames)  # [T, 3, H, W]


def precompute_with_hooks(cache_dir: str, cam_name: str, num_frames: int,
                           target_w: int, target_h: int, window_size: int = 8):
    """
    Run VGGT4Track with a forward hook to capture aggregated_tokens_list[-1].
    """
    print(f"\n=== Loading VGGT4Track model ===")
    model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    model.eval()
    model = model.to("cuda")

    print(f"=== Loading frames from {cam_name} ===")
    all_frames = load_frames_for_stv2(cache_dir, cam_name, num_frames, target_w, target_h)

    # Storage for hook captures
    captured_tokens = {}

    def hook_fn(module, input, output):
        # aggregator returns (aggregated_tokens_list, patch_start_idx)
        aggregated_tokens_list, patch_start_idx = output
        # Take the last element (most refined features)
        captured_tokens["tokens"] = aggregated_tokens_list[-1].cpu()  # [B, T, P, 2048]

    # Register hook on the aggregator
    hook = model.aggregator.register_forward_hook(hook_fn)

    all_tokens = []
    all_points_map = []
    all_poses = []
    all_intrs = []
    all_unc = []

    print(f"=== Running VGGT4Track with token capture (window={window_size}) ===")
    for start in tqdm(range(0, num_frames, window_size), desc="VGGT4Track"):
        end = min(start + window_size, num_frames)
        frames_window = all_frames[start:end].unsqueeze(0).cuda() / 255.0

        torch.cuda.empty_cache()

        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    predictions = model(frames_window)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                half = max(1, (end - start) // 2)
                print(f"\n  OOM at window {start}-{end}, retrying with window={half}")
                for sub_start in range(start, end, half):
                    sub_end = min(sub_start + half, end)
                    sub_frames = all_frames[sub_start:sub_end].unsqueeze(0).cuda() / 255.0
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            pred = model(sub_frames)
                    n = sub_end - sub_start
                    all_tokens.append(captured_tokens["tokens"].squeeze(0)[:n].half())
                    all_points_map.append(pred["points_map"].cpu()[:n])
                    all_poses.append(pred["poses_pred"].cpu().squeeze(0)[:n])
                    all_intrs.append(pred["intrs"].cpu().squeeze(0)[:n])
                    all_unc.append(pred["unc_metric"].cpu()[:n])
                continue
            else:
                raise

        n = end - start
        all_tokens.append(captured_tokens["tokens"].squeeze(0)[:n].half())  # [T, P, 2048]
        all_points_map.append(predictions["points_map"].cpu()[:n])
        all_poses.append(predictions["poses_pred"].cpu().squeeze(0)[:n])
        all_intrs.append(predictions["intrs"].cpu().squeeze(0)[:n])
        all_unc.append(predictions["unc_metric"].cpu()[:n])

    hook.remove()

    # Concatenate
    tokens = torch.cat(all_tokens, dim=0)
    points_map = torch.cat(all_points_map, dim=0)
    poses = torch.cat(all_poses, dim=0)
    intrs = torch.cat(all_intrs, dim=0)
    unc_metric = torch.cat(all_unc, dim=0)

    print(f"\n=== Cached tensor shapes ===")
    print(f"  tokens:     {tokens.shape} ({tokens.element_size() * tokens.nelement() / 1e6:.1f} MB)")
    print(f"  points_map: {points_map.shape}")
    print(f"  poses:      {poses.shape}")
    print(f"  intrs:      {intrs.shape}")
    print(f"  unc_metric: {unc_metric.shape}")

    torch.save(tokens, os.path.join(cache_dir, "tokens.pt"))
    torch.save(points_map.half(), os.path.join(cache_dir, "points_map.pt"))
    torch.save(poses.float(), os.path.join(cache_dir, "poses.pt"))
    torch.save(intrs.float(), os.path.join(cache_dir, "intrs.pt"))
    torch.save(unc_metric.half(), os.path.join(cache_dir, "unc_metric.pt"))

    del model
    torch.cuda.empty_cache()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Precompute STV2 features for overfit test")
    parser.add_argument("--scene_dir", type=str,
                        default="D:/DecodeGaussians/datasets/neu3d/coffee_martini")
    parser.add_argument("--cache_dir", type=str,
                        default="D:/DecodeGaussians/experiments/overfit_coffee_martini/cache")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--input_camera", type=str, default="cam01")
    parser.add_argument("--window_size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Step 1: Extract frames from all cameras
    print("=" * 60)
    print("Step 1: Extracting frames")
    print("=" * 60)
    extract_frames(args.scene_dir, args.cache_dir, args.width, args.height)

    # Step 2: Run VGGT4Track with hooks to capture aggregated tokens
    print("\n" + "=" * 60)
    print("Step 2: Running VGGT4Track (with token capture hooks)")
    print("=" * 60)
    precompute_with_hooks(
        args.cache_dir, args.input_camera, args.num_frames,
        args.width, args.height, args.window_size
    )

    # Step 3: Report
    print("\n" + "=" * 60)
    print("Precompute complete!")
    print("=" * 60)
    for f in ["tokens.pt", "points_map.pt", "poses.pt", "intrs.pt", "unc_metric.pt"]:
        path = os.path.join(args.cache_dir, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {f}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
