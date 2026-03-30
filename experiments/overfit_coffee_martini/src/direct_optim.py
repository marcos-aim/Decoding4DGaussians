"""Direct Gaussian parameter optimization (no MLP decoder).
Diagnostic: finds the upper bound achievable with our Gaussian count + initialization."""

import os
import sys
import glob
import math
import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from cameras import build_cameras_from_llff
from dataset import CachedSceneDataset
from renderer import render_gaussians
from losses import photometric_loss, compute_psnr, scale_regularization, opacity_regularization


def inverse_sigmoid(x):
    return math.log(x / (1.0 - x))


def direct_optimize(config_path: str, use_full_cloud: bool = False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    scene_dir = cfg["paths"]["neu3d_scene"]
    cache_dir = cfg["paths"]["cache_dir"]
    render_dir = cfg["paths"]["render_dir"]
    target_w, target_h = cfg["data"]["resolution"]

    cam_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(scene_dir, "cam*.mp4"))])
    cameras = build_cameras_from_llff(
        os.path.join(scene_dir, "poses_bounds.npy"), target_w, target_h, cam_files
    )
    dataset = CachedSceneDataset(cache_dir, cameras)

    pts_mean = dataset.points_map.mean(dim=0).reshape(-1, 3)

    if use_full_cloud:
        # Use ALL 196K points from the STV2 point cloud
        # Filter out zero/invalid points
        valid_mask = pts_mean.abs().sum(-1) > 1e-6
        init_xyz = pts_mean[valid_mask]
        N = init_xyz.shape[0]
        print(f"Direct optimization (FULL CLOUD): {N} Gaussians from {pts_mean.shape[0]} pixels")
    else:
        # Original: subsample to P patches, expand with K
        P = dataset.num_patches
        K = cfg["model"].get("num_gaussians_per_patch", 1)
        N = P * K
        indices = torch.linspace(0, pts_mean.shape[0] - 1, P).long()
        anchor = pts_mean[indices]
        if K > 1:
            spread = 0.05
            offsets = torch.randn(P, K, 3) * spread
            offsets[:, 0, :] = 0.0
            init_xyz = (anchor.unsqueeze(1).expand(-1, K, -1) + offsets).reshape(N, 3)
        else:
            init_xyz = anchor
        print(f"Direct optimization: {N} Gaussians ({P} patches x {cfg['model'].get('num_gaussians_per_patch', 1)})")

    # Directly optimizable parameters
    means3D = init_xyz.cuda().requires_grad_(True)
    log_scales = torch.full((N, 3), math.log(0.5), device="cuda").requires_grad_(True)
    quats = torch.zeros(N, 4, device="cuda")
    quats[:, 0] = 1.0  # identity quaternion
    quats = quats.requires_grad_(True)
    logit_opacity = torch.full((N, 1), inverse_sigmoid(0.5), device="cuda").requires_grad_(True)
    # DC color (SH=0): [N, 1, 3]
    colors = torch.zeros(N, 1, 3, device="cuda").requires_grad_(True)

    params = [
        {"params": [means3D], "lr": 1e-3},
        {"params": [log_scales], "lr": 5e-3},
        {"params": [quats], "lr": 1e-3},
        {"params": [logit_opacity], "lr": 5e-3},
        {"params": [colors], "lr": 5e-3},
    ]
    optimizer = optim.Adam(params)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

    bg_color = torch.zeros(3, device="cuda")
    eval_cam = cameras[cfg["data"]["eval_camera"]]
    train_cams = [n for n in cameras if n != cfg["data"]["eval_camera"]]

    os.makedirs(os.path.join(render_dir, "direct"), exist_ok=True)

    # Use frame 0 for single-frame direct optimization
    frame_idx = 0

    for step in tqdm(range(5000), desc="Direct Optim"):
        optimizer.zero_grad()

        scales = torch.clamp(F.softplus(log_scales), min=1e-6, max=5.0)
        rotations = F.normalize(quats, dim=-1)
        opacity = torch.sigmoid(logit_opacity)

        # Random training camera
        cam_name = train_cams[step % len(train_cams)]
        cam = cameras[cam_name]
        gt = dataset.load_frame_image(cam_name, frame_idx).cuda()

        rendered, _, _, _ = render_gaussians(
            means3D, scales, rotations, opacity, colors, cam, bg_color, 0
        )

        loss = photometric_loss(rendered, gt, lambda_ssim=0.85)
        loss = loss + 0.01 * scale_regularization(scales)
        loss = loss + 0.01 * opacity_regularization(opacity)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 500 == 0:
            psnr = compute_psnr(rendered, gt)
            # Novel view
            with torch.no_grad():
                s = torch.clamp(F.softplus(log_scales), min=1e-6, max=5.0)
                r = F.normalize(quats, dim=-1)
                o = torch.sigmoid(logit_opacity)
                gt_novel = dataset.load_frame_image(cfg["data"]["eval_camera"], frame_idx).cuda()
                rendered_novel, _, _, _ = render_gaussians(means3D, s, r, o, colors, eval_cam, bg_color, 0)
                novel_psnr = compute_psnr(rendered_novel, gt_novel)
            print(f"  Step {step}: train={psnr:.2f} dB, novel={novel_psnr:.2f} dB, loss={loss.item():.4f}")

            if step % 1000 == 0:
                import cv2
                import numpy as np
                img = (rendered_novel.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype("uint8")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(render_dir, "direct", f"step_{step:05d}.jpg"), img)

    # Final evaluation
    with torch.no_grad():
        scales = torch.clamp(F.softplus(log_scales), min=1e-6, max=5.0)
        rotations = F.normalize(quats, dim=-1)
        opacity = torch.sigmoid(logit_opacity)

        psnrs_train = []
        for cam_name in train_cams[:4]:
            gt = dataset.load_frame_image(cam_name, frame_idx).cuda()
            rendered, _, _, _ = render_gaussians(means3D, scales, rotations, opacity, colors, cameras[cam_name], bg_color, 0)
            psnrs_train.append(compute_psnr(rendered, gt))

        gt_novel = dataset.load_frame_image(cfg["data"]["eval_camera"], frame_idx).cuda()
        rendered_novel, _, _, _ = render_gaussians(means3D, scales, rotations, opacity, colors, eval_cam, bg_color, 0)
        novel_psnr = compute_psnr(rendered_novel, gt_novel)

    print("\n" + "=" * 60)
    print(f"Direct Optimization Results (frame {frame_idx}):")
    print(f"  Train PSNR (4 cams avg): {sum(psnrs_train)/len(psnrs_train):.2f} dB")
    print(f"  Novel PSNR (cam00):      {novel_psnr:.2f} dB")
    print(f"  Train/Novel gap:         {sum(psnrs_train)/len(psnrs_train) - novel_psnr:.2f} dB")
    print(f"  Mean scale: {scales.mean():.4f}, Max: {scales.max():.4f}")
    print(f"  Mean opacity: {opacity.mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="D:/DecodeGaussians/experiments/overfit_coffee_martini/config.yaml")
    parser.add_argument("--full-cloud", action="store_true",
                        help="Use all 196K STV2 points instead of subsampled patches")
    args = parser.parse_args()
    direct_optimize(args.config, use_full_cloud=args.full_cloud)
