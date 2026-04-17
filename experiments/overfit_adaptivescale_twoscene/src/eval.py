"""Offline evaluation for a single scene from multi-scene trained model.

Renders all frames from eval camera, computes metrics (PSNR, SSIM), saves renders.

Usage:
    python src/eval.py --config config.yaml --scene coffee_martini [--checkpoint PATH] [--output-dir ./eval_output]
"""

import os
import sys
import glob
import yaml
import numpy as np
import torch
import cv2
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from cameras import build_cameras_from_llff
from dataset import CachedSceneDataset
from model import CanonicalGaussianHead, DeformationHead, compose_gaussians
from renderer import render_gaussians


def load_model(config_path: str, scene_name: str, checkpoint_path: str = None):
    """Load multi-scene trained model for a specific scene evaluation."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Find the specified scene in the config
    scene_cfg = None
    for s in cfg["scenes"]:
        if s["name"] == scene_name:
            scene_cfg = s
            break

    if scene_cfg is None:
        raise ValueError(
            f"Scene '{scene_name}' not found in config. "
            f"Available scenes: {[s['name'] for s in cfg['scenes']]}"
        )

    print(f"Loading scene: {scene_name}")
    print(f"  Cache: {scene_cfg['cache_dir']}")
    print(f"  Poses: {scene_cfg['poses_bounds_path']}")

    # Load cameras from LLFF format
    scene_dir = os.path.dirname(scene_cfg["poses_bounds_path"])
    target_w = cfg["rendering"]["image_width"]
    target_h = cfg["rendering"]["image_height"]

    cam_files = sorted([
        os.path.basename(f) for f in glob.glob(os.path.join(scene_dir, "cam*.mp4"))
    ])

    cameras = build_cameras_from_llff(
        scene_cfg["poses_bounds_path"], target_w, target_h, cam_files
    )

    # Create dataset with coordinate normalization (CRITICAL for multi-scene!)
    dataset = CachedSceneDataset(
        cache_dir=scene_cfg["cache_dir"],
        cameras=cameras,
        input_camera=scene_cfg.get("input_camera", "cam01"),
        eval_camera=scene_cfg.get("eval_camera", "cam00"),
        normalize_coords=cfg["normalization"].get("enabled", True)
    )

    print(f"  Loaded {dataset.num_frames} frames, {dataset.num_patches} patches")
    if dataset.norm_center is not None:
        print(f"  Coordinate normalization enabled")

    # Initialize model architecture
    P = dataset.num_patches
    K = cfg["model"]["num_gaussians_per_patch"]

    # Compute init_xyz from normalized points
    pts_mean = dataset.points_map.mean(dim=0)
    H, W = pts_mean.shape[0], pts_mean.shape[1]
    pts_flat = pts_mean.reshape(-1, 3)

    # Grid-based patch assignment
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
    pixel_to_patch = (patch_idx_y * grid_w + patch_idx_x).reshape(-1)

    # Compute per-patch init positions
    init_xyz_per_gaussian = torch.zeros(P, K, 3)
    for p in range(P):
        mask = (pixel_to_patch == p)
        pts_p = pts_flat[mask]
        if pts_p.shape[0] == 0:
            init_xyz_per_gaussian[p] = pts_flat.mean(dim=0).unsqueeze(0).expand(K, -1)
        elif pts_p.shape[0] >= K:
            idx = torch.linspace(0, pts_p.shape[0] - 1, K).long()
            init_xyz_per_gaussian[p] = pts_p[idx]
        else:
            repeats = K // pts_p.shape[0] + 1
            init_xyz_per_gaussian[p] = pts_p.repeat(repeats, 1)[:K]
    init_xyz_per_gaussian = init_xyz_per_gaussian.reshape(P * K, 3)

    indices = torch.linspace(0, pts_flat.shape[0] - 1, P).long()
    init_xyz = pts_flat[indices]

    # Create canonical head
    canonical_head = CanonicalGaussianHead(
        dim_in=cfg["model"]["dim_tokens"],
        dim_hidden=cfg["model"]["dim_canonical_hidden"],
        sh_degree=cfg["model"]["sh_degree"],
        init_xyz=init_xyz,
        num_gaussians_per_patch=K,
        init_xyz_per_gaussian=init_xyz_per_gaussian,
    ).cuda()

    # Create deformation head
    deformation_head = DeformationHead(
        dim_in=cfg["model"]["dim_tokens"],
        dim_hidden=cfg["model"]["dim_deformation_hidden"],
        n_attn_heads=cfg["model"]["n_attn_heads"],
        n_attn_layers=cfg["model"]["n_attn_layers"],
        sh_degree=cfg["model"]["sh_degree"],
        num_gaussians_per_patch=K,
    ).cuda()

    # Load checkpoint
    ckpt_dir = cfg["logging"]["ckpt_dir"]
    if checkpoint_path is None:
        stage3_path = os.path.join(ckpt_dir, "stage3_final.pt")
        checkpoint_path = (
            stage3_path if os.path.exists(stage3_path)
            else os.path.join(ckpt_dir, "stage2_final.pt")
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    canonical_head.load_state_dict(ckpt["canonical_head"])
    deformation_head.load_state_dict(ckpt["deformation_head"])
    canonical_head.eval()
    deformation_head.eval()
    print("Model loaded successfully!")

    scale_anneal_target = cfg["training"].get("scale_anneal_target", 1.0)

    return canonical_head, deformation_head, dataset, cfg, scale_anneal_target


def compute_metrics(pred, gt):
    """Compute PSNR and SSIM between predicted and ground truth images."""
    # PSNR
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))

    # SSIM (simplified version using cv2)
    # Convert to uint8 for cv2.SSIM
    pred_uint8 = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
    gt_uint8 = (np.clip(gt, 0, 1) * 255).astype(np.uint8)

    # Compute SSIM per channel and average
    ssims = []
    for c in range(3):
        ssim = cv2.matchTemplate(pred_uint8[..., c], gt_uint8[..., c], cv2.TM_CCOEFF_NORMED)[0, 0]
        ssims.append(ssim)
    ssim = np.mean(ssims)

    return psnr, ssim


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Offline evaluation for single scene from multi-scene trained model"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to multi-scene config.yaml"
    )
    parser.add_argument(
        "--scene", type=str, required=True,
        help="Name of scene to evaluate (must match name in config)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (default: stage3_final.pt or stage2_final.pt)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./eval_output",
        help="Directory to save renders and metrics (default: ./eval_output)"
    )
    args = parser.parse_args()

    print("="*60)
    print("Multi-Scene Gaussian Evaluation")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Scene: {args.scene}")
    print(f"Output: {args.output_dir}")
    print("="*60)

    # Load model
    canonical_head, deformation_head, dataset, cfg, scale_anneal_target = load_model(
        args.config, args.scene, args.checkpoint
    )
    num_frames = dataset.num_frames
    eval_cam_name = dataset.eval_cam_name

    print(f"\nEvaluating {num_frames} frames from camera: {eval_cam_name}")

    # Create output directories
    output_dir = Path(args.output_dir) / args.scene
    output_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    gt_dir = output_dir / "ground_truth"
    gt_dir.mkdir(exist_ok=True)

    # Rendering setup
    bg_color = torch.tensor(cfg["rendering"]["bg_color"], dtype=torch.float32, device="cuda")
    scale_modifier = cfg["rendering"]["scale_modifier"]

    # Evaluation loop
    print("\nRendering frames...")
    metrics = {"psnr": [], "ssim": []}

    with torch.no_grad():
        tokens_mean = dataset.get_tokens_mean().cuda()
        canonical = canonical_head(tokens_mean)

        for frame_idx in range(num_frames):
            # Get tokens for this frame
            tokens_t = dataset.get_tokens_frame(frame_idx).cuda()

            # Deform canonical Gaussians
            deltas = deformation_head(tokens_t)
            means3D, scales, rotations, opacity, shs = compose_gaussians(
                canonical, deltas, scale_factor=scale_anneal_target
            )

            # Render from eval camera
            eval_cam = dataset.cameras[eval_cam_name]
            rendered = render_gaussians(
                means3D, scales, rotations, opacity, shs,
                eval_cam, bg_color, scale_modifier
            )

            # Load ground truth
            gt_image = dataset.load_frame_image(eval_cam_name, frame_idx).cuda()

            # Convert to numpy for metrics and saving
            pred_np = rendered.permute(1, 2, 0).cpu().numpy()
            gt_np = gt_image.permute(1, 2, 0).cpu().numpy()

            # Compute metrics
            psnr, ssim = compute_metrics(pred_np, gt_np)
            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)

            # Save renders
            pred_uint8 = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
            gt_uint8 = (np.clip(gt_np, 0, 1) * 255).astype(np.uint8)

            cv2.imwrite(
                str(renders_dir / f"frame_{frame_idx:04d}.png"),
                cv2.cvtColor(pred_uint8, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                str(gt_dir / f"frame_{frame_idx:04d}.png"),
                cv2.cvtColor(gt_uint8, cv2.COLOR_RGB2BGR)
            )

            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}/{num_frames}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

    # Compute average metrics
    avg_psnr = np.mean(metrics["psnr"])
    avg_ssim = np.mean(metrics["ssim"])

    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Scene: {args.scene}")
    print(f"Frames: {num_frames}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*60)

    # Save metrics to file
    metrics_file = output_dir / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Scene: {args.scene}\n")
        f.write(f"Frames: {num_frames}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n\n")
        f.write("Per-frame metrics:\n")
        for i, (p, s) in enumerate(zip(metrics["psnr"], metrics["ssim"])):
            f.write(f"Frame {i:04d}: PSNR={p:.2f} dB, SSIM={s:.4f}\n")

    print(f"\nMetrics saved to: {metrics_file}")
    print(f"Renders saved to: {renders_dir}")
    print(f"Ground truth saved to: {gt_dir}")


if __name__ == "__main__":
    main()
