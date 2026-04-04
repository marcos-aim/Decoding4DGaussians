"""SpaTrackerV2 precomputation pipeline.

Extracts and caches STV2 latents (tokens, points_map, poses, intrinsics, uncertainty)
for training. Idempotent: skips scenes with complete manifests.

Usage:
    python -m src.precompute --config config.yaml
    python -m src.precompute --scenes datasets/neural_3d/* --output precomputed/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def extract_frames(scene_path: str, output_dir: str, resolution: tuple[int, int],
                   camera_dirs: list[str] = None) -> int:
    """Extract frames from MP4 videos and save as JPEGs.

    Args:
        scene_path: path to scene directory containing camera MP4s
        output_dir: where to save extracted frames
        resolution: (width, height) target resolution
        camera_dirs: list of camera directory names, or None to auto-detect

    Returns:
        Number of frames extracted per camera.
    """
    scene_path = Path(scene_path)
    output_dir = Path(output_dir)

    if camera_dirs is None:
        # Auto-detect camera directories with MP4 files
        camera_dirs = sorted([
            d.name for d in scene_path.iterdir()
            if d.is_dir() and any(d.glob("*.mp4"))
        ])

    num_frames = 0
    for cam_name in camera_dirs:
        cam_out = output_dir / "frames" / cam_name
        cam_out.mkdir(parents=True, exist_ok=True)

        mp4_files = sorted((scene_path / cam_name).glob("*.mp4"))
        if not mp4_files:
            continue

        cap = cv2.VideoCapture(str(mp4_files[0]))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resolution)
            cv2.imwrite(str(cam_out / f"{frame_idx:06d}.jpg"), frame)
            frame_idx += 1
        cap.release()
        num_frames = max(num_frames, frame_idx)
        print(f"  {cam_name}: {frame_idx} frames")

    return num_frames


def run_stv2_extraction(
    frames_dir: str,
    output_dir: str,
    input_camera: str,
    window_size: int = 8,
    dtype_str: str = "bfloat16",
    outputs_config: dict = None,
) -> dict:
    """Run SpaTrackerV2 (VGGT4Track) and extract tokens + predictions.

    Args:
        frames_dir: directory with extracted frames (frames/{cam_name}/{idx:06d}.jpg)
        output_dir: where to save cached tensors
        input_camera: which camera to use as STV2 input
        window_size: frames per STV2 forward pass
        dtype_str: "bfloat16" or "float16" for autocast
        outputs_config: dict of output_name -> bool for which tensors to save

    Returns:
        dict with shapes of saved tensors
    """
    # Import STV2 lazily (heavy dependency)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SpaTrackerV2"))
    from models.spatracker.predictor import SpaTrackerPredictor

    outputs_config = outputs_config or {"tokens": True, "points_map": True,
                                         "poses": True, "intrinsics": True,
                                         "uncertainty": True}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frames for input camera
    cam_dir = Path(frames_dir) / "frames" / input_camera
    frame_files = sorted(cam_dir.glob("*.jpg"))
    print(f"  Loading {len(frame_files)} frames from {input_camera}...")

    frames = []
    for ff in frame_files:
        img = cv2.imread(str(ff))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(img).permute(2, 0, 1))
    all_frames = torch.stack(frames)  # [T, 3, H, W]
    num_frames = all_frames.shape[0]

    # Load model
    print("  Loading VGGT4Track model...")
    model = SpaTrackerPredictor.from_pretrained("facebook/VGGT4Track")
    model = model.cuda().eval()

    # Hook to capture aggregated tokens
    captured_tokens = {}

    def hook_fn(module, input, output):
        if hasattr(module, "aggregated_tokens_list") and module.aggregated_tokens_list:
            captured_tokens["tokens"] = module.aggregated_tokens_list[-1]

    hook = model.aggregator.register_forward_hook(hook_fn)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    autocast_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    all_tokens = []
    all_points_map = []
    all_poses = []
    all_intrs = []
    all_unc = []

    print(f"  Running VGGT4Track (window={window_size})...")
    for start in tqdm(range(0, num_frames, window_size), desc="VGGT4Track"):
        end = min(start + window_size, num_frames)
        frames_window = all_frames[start:end].unsqueeze(0).cuda() / 255.0

        torch.cuda.empty_cache()

        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
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
                        with torch.amp.autocast("cuda", dtype=autocast_dtype):
                            pred = model(sub_frames)
                    n = sub_end - sub_start
                    all_tokens.append(captured_tokens["tokens"].squeeze(0)[:n].half().cpu())
                    all_points_map.append(pred["points_map"].cpu()[:n])
                    all_poses.append(pred["poses_pred"].cpu().squeeze(0)[:n])
                    all_intrs.append(pred["intrs"].cpu().squeeze(0)[:n])
                    all_unc.append(pred["unc_metric"].cpu()[:n])
                continue
            else:
                raise

        n = end - start
        all_tokens.append(captured_tokens["tokens"].squeeze(0)[:n].half().cpu())
        all_points_map.append(predictions["points_map"].cpu()[:n])
        all_poses.append(predictions["poses_pred"].cpu().squeeze(0)[:n])
        all_intrs.append(predictions["intrs"].cpu().squeeze(0)[:n])
        all_unc.append(predictions["unc_metric"].cpu()[:n])

    hook.remove()

    tokens = torch.cat(all_tokens, dim=0)
    points_map = torch.cat(all_points_map, dim=0)
    poses = torch.cat(all_poses, dim=0)
    intrs = torch.cat(all_intrs, dim=0)
    unc_metric = torch.cat(all_unc, dim=0)

    shapes = {}
    if outputs_config.get("tokens", True):
        torch.save(tokens, output_dir / "tokens.pt")
        shapes["tokens"] = list(tokens.shape)
    if outputs_config.get("points_map", True):
        torch.save(points_map.half(), output_dir / "points_map.pt")
        shapes["points_map"] = list(points_map.shape)
    if outputs_config.get("poses", True):
        torch.save(poses, output_dir / "poses.pt")
        shapes["poses"] = list(poses.shape)
    if outputs_config.get("intrinsics", True):
        torch.save(intrs, output_dir / "intrs.pt")
        shapes["intrinsics"] = list(intrs.shape)
    if outputs_config.get("uncertainty", True):
        torch.save(unc_metric.half(), output_dir / "unc_metric.pt")
        shapes["uncertainty"] = list(unc_metric.shape)

    # Write manifest
    manifest = {
        "backbone": "spatracker_v2",
        "num_frames": num_frames,
        "resolution": [all_frames.shape[3], all_frames.shape[2]],
        "window_size": window_size,
        "shapes": shapes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n  Saved to {output_dir}/")
    for name, shape in shapes.items():
        print(f"    {name}: {shape}")

    return shapes


def is_scene_complete(precomputed_dir: str) -> bool:
    """Check if a scene has a complete manifest (idempotent check)."""
    manifest_path = Path(precomputed_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        return "backbone" in manifest and "shapes" in manifest
    except (json.JSONDecodeError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Precompute STV2 features")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--scenes", nargs="*", help="Scene directories to process")
    parser.add_argument("--output", type=str, default="precomputed/", help="Output root directory")
    parser.add_argument("--input-camera", type=str, default="cam01")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--resolution", nargs=2, type=int, default=[512, 384], metavar=("W", "H"))
    args = parser.parse_args()

    if args.config:
        from src.config import load_engine_config
        cfg = load_engine_config(args.config)
        for scene_cfg in cfg.data.scenes:
            if is_scene_complete(scene_cfg.precomputed):
                print(f"  Skipping {scene_cfg.name} (already complete)")
                continue
            print(f"\n=== Processing {scene_cfg.name} ===")
            extract_frames(scene_cfg.path, scene_cfg.precomputed,
                           tuple(cfg.data.resolution))
            run_stv2_extraction(
                scene_cfg.precomputed, scene_cfg.precomputed,
                cfg.data.input_camera, cfg.precompute.window_size,
                cfg.precompute.dtype,
                {k: getattr(cfg.precompute.outputs, k) for k in
                 ["tokens", "points_map", "poses", "intrinsics", "uncertainty"]},
            )
    elif args.scenes:
        for scene_path in args.scenes:
            scene_name = Path(scene_path).name
            output_dir = os.path.join(args.output, scene_name)
            if is_scene_complete(output_dir):
                print(f"  Skipping {scene_name} (already complete)")
                continue
            print(f"\n=== Processing {scene_name} ===")
            extract_frames(scene_path, output_dir, tuple(args.resolution))
            run_stv2_extraction(output_dir, output_dir, args.input_camera,
                                args.window_size)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
