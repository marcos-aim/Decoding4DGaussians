"""Precompute SpaTrackerV2 features and extract frames for diva360 dataset.

Produces identical output format to precompute.py (tokens.pt, points_map.pt,
poses.pt, intrs.pt, unc_metric.pt, frames/) but handles the diva360 directory
structure where videos are AVI files nested in synced/ subdirectories.

Also generates a neu3d-compatible scene directory ({cache_dir}/scene/) with:
  - cam00.mp4, cam01.mp4, ... symlinks to the original AVI files
  - poses_bounds.npy synthesized from VGGT4Track multi-camera pose estimation
  - camera_mapping.txt mapping camNN -> original camera name

This allows train.py to use a diva360 scene with zero changes by pointing
poses_bounds_path at {cache_dir}/scene/poses_bounds.npy.

Auto-discovers all cameras in {scene_dir}/synced/, extracts frames for all of
them, and uses the second camera (sorted) as the input for VGGT4Track by default.

Usage:
    python src/precompute_diva360.py \\
        --scene-dir /path/to/diva360/pan_synced \\
        --cache-dir /path/to/cached_datasets/diva360/cache_pan_synced \\
        --num-frames 300
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add SpaTrackerV2 to path
SPATRACKER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "SpaTrackerV2"))
sys.path.insert(0, SPATRACKER_ROOT)

from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image


def discover_cameras(scene_dir: str) -> list:
    """Auto-discover all camera directories in the diva360 synced/ folder.

    Returns sorted list of camera names (e.g., ['brics-sbc-001_cam0', ...]),
    excluding microphone directories (*_mic*).
    """
    synced_dir = os.path.join(scene_dir, "synced")
    cam_dirs = sorted(glob.glob(os.path.join(synced_dir, "brics-sbc-*_cam*")))
    camera_names = [os.path.basename(d) for d in cam_dirs
                    if os.path.isdir(d) and "_mic" not in os.path.basename(d)]
    return camera_names


def extract_frames(scene_dir: str, camera_names: list, cache_dir: str,
                   target_w: int, target_h: int):
    """Extract frames from diva360 camera AVI files, resize, and save as JPEG.

    Diva360 video paths: {scene_dir}/synced/{cam_name}/{cam_name}.avi
    Output: {cache_dir}/frames/{cam_name}/{frame_idx:06d}.jpg
    """
    print(f"Extracting frames for {len(camera_names)} cameras")

    for cam_name in tqdm(camera_names, desc="Extracting frames"):
        video_path = os.path.join(scene_dir, "synced", cam_name, f"{cam_name}.avi")
        if not os.path.exists(video_path):
            print(f"  WARNING: video not found: {video_path}")
            continue

        out_dir = os.path.join(cache_dir, "frames", cam_name)
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
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


def estimate_camera_poses(scene_dir: str, cache_dir: str, camera_names: list,
                           target_w: int, target_h: int) -> np.ndarray:
    """Estimate multi-camera poses by running VGGT4Track on one frame per camera.

    Feeds frame 0 from each camera as a "video" to VGGT4Track, which predicts
    relative poses and intrinsics. Converts to LLFF poses_bounds.npy format.

    Returns:
        poses_bounds: (N_cams, 17) float32 numpy array in LLFF format
    """
    print(f"Loading frame 0 from each of {len(camera_names)} cameras...")

    frames = []
    for cam_name in camera_names:
        frame_path = os.path.join(cache_dir, "frames", cam_name, "000000.jpg")
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame not found: {frame_path} — run frame extraction first")
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(img).permute(2, 0, 1).float())

    # Stack as [1, N_cams, 3, H, W] — treating cameras as "frames"
    frames_tensor = torch.stack(frames).unsqueeze(0).cuda() / 255.0

    print("Running VGGT4Track for multi-camera pose estimation...")
    model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    model.eval()
    model = model.to("cuda")

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(frames_tensor)

    # poses_pred: [1, N_cams, 4, 4] c2w matrices
    # intrs:      [1, N_cams, 3, 3] intrinsic matrices
    # points_map: used for near/far depth estimation
    poses_pred = predictions["poses_pred"].float().squeeze(0).cpu().numpy()  # [N, 4, 4]
    intrs_pred = predictions["intrs"].float().squeeze(0).cpu().numpy()       # [N, 3, 3]
    points_map = predictions["points_map"].float().squeeze(0).cpu().numpy()  # [N, H, W, 3] or [N, H*W, 3]

    del model
    torch.cuda.empty_cache()

    # Original resolution (needed for H/W/focal in poses_bounds)
    # Use first camera's original video to read resolution
    first_cam = camera_names[0]
    first_video = os.path.join(scene_dir, "synced", first_cam, f"{first_cam}.avi")
    cap = cv2.VideoCapture(first_video)
    orig_H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    orig_W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()

    # Estimate near/far from all camera point maps
    # points_map values are in VGGT camera space — use Z (depth) component
    pts = points_map.reshape(-1, 3)
    depths = pts[:, 2]
    valid_depths = depths[depths > 0]
    if len(valid_depths) > 0:
        near = float(np.percentile(valid_depths, 5))
        far = float(np.percentile(valid_depths, 95))
        near = max(near * 0.9, 1e-3)
        far = far * 1.1
    else:
        near, far = 0.01, 100.0
    print(f"  Estimated depth bounds: near={near:.3f}, far={far:.3f}")

    # Build poses_bounds array: (N, 17)
    #
    # VGGT FrontTracker poses_pred is w2c in OpenCV convention [right, down, forward].
    # We need c2w, so invert first.
    #
    # LLFF 3x5 format (per row, before axis swap in parse_llff_poses):
    #   col0: down axis
    #   col1: right axis
    #   col2: backwards axis
    #   col3: translation
    #   col4: [H, W, focal]^T
    #
    # OpenCV c2w rotation columns: [right, down, forward] in world coords.
    # Mapping to LLFF raw:
    #   col0 (down)      = c2w R[:,1]  (OpenCV down)
    #   col1 (right)     = c2w R[:,0]  (OpenCV right)
    #   col2 (backwards) = -c2w R[:,2] (backwards = -forward)
    #   col3 (t)         = c2w t
    poses_bounds = np.zeros((len(camera_names), 17), dtype=np.float32)

    for i in range(len(camera_names)):
        w2c = poses_pred[i]           # (4, 4) — w2c from FrontTracker
        c2w = np.linalg.inv(w2c)      # invert to get c2w
        K = intrs_pred[i]             # (3, 3)

        R_c2w = c2w[:3, :3]          # columns are [right, down, forward] in world coords
        t_c2w = c2w[:3, 3]

        # Average fx/fy, scaled to original resolution
        fx = K[0, 0] * (orig_W / target_w)
        fy = K[1, 1] * (orig_H / target_h)
        focal = float((fx + fy) / 2.0)

        llff_pose = np.zeros((3, 5), dtype=np.float32)
        llff_pose[:, 0] = R_c2w[:, 1]    # down = OpenCV down
        llff_pose[:, 1] = R_c2w[:, 0]    # right = OpenCV right
        llff_pose[:, 2] = -R_c2w[:, 2]   # backwards = -forward
        llff_pose[:, 3] = t_c2w          # translation
        llff_pose[0, 4] = orig_H
        llff_pose[1, 4] = orig_W
        llff_pose[2, 4] = focal

        poses_bounds[i, :15] = llff_pose.reshape(15)
        poses_bounds[i, 15] = near
        poses_bounds[i, 16] = far

    print(f"  poses_bounds shape: {poses_bounds.shape}")
    print(f"  Focal length (cam0): {poses_bounds[0, 14]:.1f}px  "
          f"H={poses_bounds[0, 4]:.0f}  W={poses_bounds[0, 9]:.0f}")

    return poses_bounds


def create_scene_directory(scene_dir: str, cache_dir: str, camera_names: list,
                            poses_bounds: np.ndarray):
    """Create a neu3d-compatible scene directory in {cache_dir}/scene/.

    Contains:
      - cam00.mp4, cam01.mp4, ... symlinks → original AVI files
      - poses_bounds.npy
      - camera_mapping.txt
    """
    scene_out_dir = os.path.join(cache_dir, "scene")
    os.makedirs(scene_out_dir, exist_ok=True)

    # Create symlinks camNN.mp4 → original AVI, in sorted camera order
    mapping_lines = []
    for i, cam_name in enumerate(camera_names):
        avi_path = os.path.abspath(
            os.path.join(scene_dir, "synced", cam_name, f"{cam_name}.avi")
        )
        link_path = os.path.join(scene_out_dir, f"cam{i:02d}.mp4")

        if os.path.islink(link_path):
            os.remove(link_path)
        os.symlink(avi_path, link_path)
        mapping_lines.append(f"cam{i:02d} -> {cam_name}  ({avi_path})")

    # Save poses_bounds.npy
    poses_path = os.path.join(scene_out_dir, "poses_bounds.npy")
    np.save(poses_path, poses_bounds)

    # Save human-readable mapping
    mapping_path = os.path.join(scene_out_dir, "camera_mapping.txt")
    with open(mapping_path, "w") as f:
        f.write("\n".join(mapping_lines) + "\n")

    print(f"  Created {len(camera_names)} symlinks in {scene_out_dir}/")
    print(f"  Saved poses_bounds.npy  ({poses_bounds.shape})")
    print(f"  Saved camera_mapping.txt")
    print(f"\n  Config hint:")
    print(f"    poses_bounds_path: {poses_path}")

    return scene_out_dir


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
    parser = argparse.ArgumentParser(description="Precompute STV2 features for a diva360 scene")
    parser.add_argument("--scene-dir", type=str, required=True,
                        help="Path to the diva360 scene root (e.g. .../diva360/pan_synced)")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Output cache directory")
    parser.add_argument("--input-camera", type=str, default=None,
                        help="Camera name to run VGGT4Track on (default: second camera, sorted)")
    parser.add_argument("--width", type=int, default=512,
                        help="Output frame width")
    parser.add_argument("--height", type=int, default=384,
                        help="Output frame height")
    parser.add_argument("--num-frames", type=int, default=300,
                        help="Number of frames to process for VGGT4Track")
    parser.add_argument("--window-size", type=int, default=8,
                        help="Window size for VGGT4Track processing")
    args = parser.parse_args()

    # Auto-discover all cameras
    camera_names = discover_cameras(args.scene_dir)
    if len(camera_names) == 0:
        print(f"ERROR: No cameras found in {args.scene_dir}/synced/")
        return

    # Default input camera is the second one (index 1), skipping the first
    input_camera = args.input_camera or camera_names[1]
    if input_camera not in camera_names:
        print(f"ERROR: --input-camera '{input_camera}' not found in discovered cameras")
        return

    print("=" * 60)
    print(f"Precomputing diva360 scene: {os.path.basename(args.scene_dir)}")
    print("=" * 60)
    print(f"  Scene directory: {args.scene_dir}")
    print(f"  Cache directory: {args.cache_dir}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Cameras found: {len(camera_names)}")
    print(f"  Input camera: {input_camera}")
    print(f"  Num frames: {args.num_frames}")
    print("=" * 60)

    os.makedirs(args.cache_dir, exist_ok=True)

    # Step 1: Extract frames from all discovered cameras
    print("\n" + "=" * 60)
    print("Step 1: Extracting frames")
    print("=" * 60)
    extract_frames(args.scene_dir, camera_names, args.cache_dir, args.width, args.height)

    # Step 2: Estimate multi-camera poses and build neu3d-compatible scene directory
    print("\n" + "=" * 60)
    print("Step 2: Estimating camera poses (one frame per camera)")
    print("=" * 60)
    poses_bounds = estimate_camera_poses(
        args.scene_dir, args.cache_dir, camera_names, args.width, args.height
    )

    print("\n" + "=" * 60)
    print("Step 3: Creating neu3d-compatible scene directory")
    print("=" * 60)
    create_scene_directory(args.scene_dir, args.cache_dir, camera_names, poses_bounds)

    # Step 4: Run VGGT4Track with hooks to capture aggregated tokens
    print("\n" + "=" * 60)
    print("Step 4: Running VGGT4Track (with token capture hooks)")
    print("=" * 60)
    precompute_with_hooks(
        args.cache_dir, input_camera, args.num_frames,
        args.width, args.height, args.window_size
    )

    # Step 5: Report
    print("\n" + "=" * 60)
    print(f"Precompute complete for: {os.path.basename(args.scene_dir)}")
    print("=" * 60)
    print(f"Cache directory: {args.cache_dir}")
    for f in ["tokens.pt", "points_map.pt", "poses.pt", "intrs.pt", "unc_metric.pt"]:
        path = os.path.join(args.cache_dir, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {f}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
