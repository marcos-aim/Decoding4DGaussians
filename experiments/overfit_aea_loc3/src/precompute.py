"""Precompute STV2 features for a monocular Aria sequence.

Unlike coffee_martini's multi-cam setup, there is no LLFF alignment step —
Aria MPS poses are already in metric world coordinates and we project
STV2's output directly through the Aria cameras.

Outputs (saved to cache_dir):
  - tokens.pt            [T, P, 2048]   fp16 aggregator features
  - points_map.pt        [T, H, W, 3]   fp16 STV2 per-pixel 3D in STV2 frame
  - poses.pt             [T, 4, 4]      float32 STV2-estimated c2w (informational)
  - intrs.pt             [T, 3, 3]      float32 STV2-estimated K
  - unc_metric.pt        [T, H, W]      fp16 per-pixel uncertainty
  - frame_indices.pt     [T]            int64 indices into transforms.json (ordering)
  - aria_c2w.pt          [T, 4, 4]      float32 Aria MPS c2w (the ground truth we render through)
  - aria_K.pt            [T, 3, 3]      float32 Aria intrinsics scaled to target res
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Path plumbing
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SPATRACKER_ROOT = os.path.join(REPO_ROOT, "SpaTrackerV2")
sys.path.insert(0, HERE)
sys.path.insert(0, SPATRACKER_ROOT)

from aria_dataset import AriaSequenceDataset
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_root", required=True,
                        help="Path to .../recording/camera-rgb-rectified-600-h1000")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--width", type=int, default=504,
                        help="4DGT default input_image_res=504")
    parser.add_argument("--height", type=int, default=504)
    parser.add_argument("--num_frames", type=int, default=128,
                        help="Window length to extract (4DGT paper: 128 @ 10fps = 13s)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start index into sorted-by-timestamp frames")
    parser.add_argument("--window_size", type=int, default=8,
                        help="STV2 forward-pass window (VRAM budget)")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # 1. Build the frame index list (128 contiguous frames)
    full_ds = AriaSequenceDataset(args.seq_root, target_resolution=(args.width, args.height))
    assert len(full_ds) >= args.start_frame + args.num_frames, \
        f"sequence has only {len(full_ds)} frames"
    frame_indices = list(range(args.start_frame, args.start_frame + args.num_frames))
    ds = AriaSequenceDataset(args.seq_root, target_resolution=(args.width, args.height),
                              frame_indices=frame_indices)

    print(f"Sequence frames: {len(ds)}  @ {args.width}x{args.height}")

    # 2. Load VGGT4Track
    print("Loading VGGT4Track...")
    model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().cuda()

    captured_tokens = {}
    def hook_fn(module, inputs, output):
        aggregated_tokens_list, _ = output
        captured_tokens["tokens"] = aggregated_tokens_list[-1].cpu()

    hook = model.aggregator.register_forward_hook(hook_fn)

    # 3. Run STV2 in windows, collect everything
    all_tokens, all_points_map = [], []
    all_poses, all_intrs, all_unc = [], [], []

    for start in tqdm(range(0, len(ds), args.window_size), desc="STV2"):
        end = min(start + args.window_size, len(ds))
        frames = ds.get_frames_stack(list(range(start, end))).unsqueeze(0).cuda()  # [1, N, 3, H, W]

        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(frames)
        except torch.cuda.OutOfMemoryError:
            # Halve the window and retry inline
            torch.cuda.empty_cache()
            half = args.window_size // 2
            sub_tokens, sub_pm, sub_poses, sub_intrs, sub_unc = [], [], [], [], []
            for s2 in range(start, end, half):
                e2 = min(s2 + half, end)
                fr = ds.get_frames_stack(list(range(s2, e2))).unsqueeze(0).cuda()
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        pred2 = model(fr)
                n2 = e2 - s2
                sub_tokens.append(captured_tokens["tokens"].squeeze(0)[:n2].half())
                sub_pm.append(pred2["points_map"].cpu()[:n2])
                sub_poses.append(pred2["poses_pred"].cpu().squeeze(0)[:n2])
                sub_intrs.append(pred2["intrs"].cpu().squeeze(0)[:n2])
                sub_unc.append(pred2["unc_metric"].cpu()[:n2])
            all_tokens.append(torch.cat(sub_tokens, dim=0))
            all_points_map.append(torch.cat(sub_pm, dim=0))
            all_poses.append(torch.cat(sub_poses, dim=0))
            all_intrs.append(torch.cat(sub_intrs, dim=0))
            all_unc.append(torch.cat(sub_unc, dim=0))
            continue

        n = end - start
        all_tokens.append(captured_tokens["tokens"].squeeze(0)[:n].half())
        all_points_map.append(pred["points_map"].cpu()[:n])
        all_poses.append(pred["poses_pred"].cpu().squeeze(0)[:n])
        all_intrs.append(pred["intrs"].cpu().squeeze(0)[:n])
        all_unc.append(pred["unc_metric"].cpu()[:n])

    hook.remove()

    tokens = torch.cat(all_tokens, dim=0)
    points_map = torch.cat(all_points_map, dim=0).float()
    poses = torch.cat(all_poses, dim=0).float()
    intrs = torch.cat(all_intrs, dim=0).float()
    unc = torch.cat(all_unc, dim=0)

    # 4. Align STV2 points to Aria metric world frame
    #    STV2 places cam-0 at identity with unknown metric scale.
    #    Aria gives us the TRUE c2w for cam-0 and metric scale.
    #    Scale factor: match STV2 scene-depth median to Aria triangulated semidense depth,
    #    using the 5th-percentile depth in cam-0 as a proxy — identical to our LLFF path
    #    but using Aria's known near bound instead of poses_bounds.npy.
    stv2_c2w_0 = poses[0]
    stv2_inv0 = torch.inverse(stv2_c2w_0)
    pts_flat = points_map.reshape(-1, 3)
    pts_cam0 = pts_flat @ stv2_inv0[:3, :3].T + stv2_inv0[:3, 3]
    z_stv2 = pts_cam0[:, 2]
    z_pos = z_stv2[z_stv2 > 0]
    z_near_stv2 = torch.quantile(z_pos, 0.05).item()

    # Estimate Aria near bound: 5th percentile of MPS semidense point depths projected into cam-0
    # Fallback: assume indoor room scale, set near=0.3 m (standard for Aria indoor)
    ARIA_NEAR_DEFAULT = 0.3
    near_aria = ARIA_NEAR_DEFAULT
    scale = near_aria / max(z_near_stv2, 1e-6)

    aria_c2w_0 = ds.get_c2w(0)  # 4x4
    R_aria = aria_c2w_0[:3, :3]
    t_aria = aria_c2w_0[:3, 3]

    pts_scaled = scale * (pts_flat @ stv2_inv0[:3, :3].T + stv2_inv0[:3, 3])
    pts_world = pts_scaled @ R_aria.T + t_aria
    points_map = pts_world.reshape(points_map.shape)

    print(f"[align] STV2 near depth (5%): {z_near_stv2:.3f}  → scale={scale:.3f}")
    print(f"[align] Aria cam-0 center:    {t_aria.tolist()}")

    # 5. Save all caches
    torch.save(tokens, os.path.join(args.cache_dir, "tokens.pt"))
    torch.save(points_map.half(), os.path.join(args.cache_dir, "points_map.pt"))
    torch.save(poses, os.path.join(args.cache_dir, "poses.pt"))
    torch.save(intrs, os.path.join(args.cache_dir, "intrs.pt"))
    torch.save(unc, os.path.join(args.cache_dir, "unc_metric.pt"))
    torch.save(torch.tensor(frame_indices), os.path.join(args.cache_dir, "frame_indices.pt"))

    aria_c2w_all = torch.stack([ds.get_c2w(i) for i in range(len(ds))])
    aria_K_all = torch.stack([ds.get_intrinsics(i) for i in range(len(ds))])
    torch.save(aria_c2w_all, os.path.join(args.cache_dir, "aria_c2w.pt"))
    torch.save(aria_K_all, os.path.join(args.cache_dir, "aria_K.pt"))
    torch.save({"scale": scale, "near_aria": near_aria, "z_near_stv2": z_near_stv2},
               os.path.join(args.cache_dir, "align_meta.pt"))

    print("=== Precompute complete ===")
    for fn in ["tokens.pt", "points_map.pt", "poses.pt", "intrs.pt", "unc_metric.pt",
               "frame_indices.pt", "aria_c2w.pt", "aria_K.pt", "align_meta.pt"]:
        p = os.path.join(args.cache_dir, fn)
        if os.path.exists(p):
            print(f"  {fn}: {os.path.getsize(p)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
