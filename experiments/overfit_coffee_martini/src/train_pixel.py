"""Training loop for pixel-aligned Gaussians.

Key difference from train.py: uses PixelGaussianHead which upsamples
patch tokens to pixel resolution, producing one Gaussian per pixel (196K total)
with true 3D coverage from STV2's per-pixel point cloud.
"""

import os
import sys
import glob
import time
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from cameras import build_cameras_from_llff
from dataset import CachedSceneDataset
from pixel_gaussians import PixelGaussianHead, PixelDeformationHead, compose_pixel_gaussians, fourier_encode
from renderer import render_gaussians
from losses import photometric_loss, compute_psnr, scale_regularization, opacity_regularization
import lpips


def get_vram_gb():
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    return alloc, peak


def save_checkpoint(path, step, canonical_head, deformation_head, optimizer):
    torch.save({
        "step": step,
        "canonical_head": canonical_head.state_dict(),
        "deformation_head": deformation_head.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def save_render(rendered, path):
    import cv2
    img = (rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def train(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["paths"]["cache_dir"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    render_dir = cfg["paths"]["render_dir"]
    log_dir = cfg["paths"]["log_dir"]

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(render_dir, "pixel_s1"), exist_ok=True)
    os.makedirs(os.path.join(render_dir, "pixel_s2"), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    target_w, target_h = cfg["data"]["resolution"]
    total_vram = cfg["vram"]["total_gb"]
    warn_vram = cfg["vram"]["warn_threshold_gb"]

    scene_dir = cfg["paths"]["neu3d_scene"]
    cam_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(scene_dir, "cam*.mp4"))])
    cameras = build_cameras_from_llff(
        os.path.join(scene_dir, "poses_bounds.npy"), target_w, target_h, cam_files
    )

    dataset = CachedSceneDataset(
        cache_dir, cameras,
        input_camera=cfg["data"]["input_camera"],
        eval_camera=cfg["data"]["eval_camera"],
    )

    # Get the spatial grid dimensions from patch count
    P = dataset.num_patches
    H, W = target_h, target_w
    grid_w = round((P * W / H) ** 0.5)
    grid_h = round(P / grid_w)
    while grid_w * grid_h < P:
        grid_w += 1
    print(f"  Patch grid: {grid_h}x{grid_w} = {grid_h*grid_w} (P={P})")
    print(f"  Image: {H}x{W} = {H*W} pixels = {H*W} Gaussians")

    # Get mean points map for canonical positions
    pts_mean = dataset.points_map.mean(dim=0)  # [H, W, 3]

    # Pixel config from yaml (with defaults)
    pcfg = cfg.get("pixel_model", {})
    canon_dim_feat = pcfg.get("canon_dim_feat", 256)
    canon_dim_hidden = pcfg.get("canon_dim_hidden", 256)
    deform_dim_feat = pcfg.get("deform_dim_feat", 128)
    deform_dim_hidden = pcfg.get("deform_dim_hidden", 128)

    canonical_head = PixelGaussianHead(
        dim_in=2048, dim_feat=canon_dim_feat, dim_hidden=canon_dim_hidden,
        grid_h=grid_h, grid_w=grid_w, img_h=H, img_w=W,
    ).cuda()

    deformation_head = PixelDeformationHead(
        dim_in=2048, dim_feat=deform_dim_feat, dim_hidden=deform_dim_hidden,
        grid_h=grid_h, grid_w=grid_w, img_h=H, img_w=W,
    ).cuda()

    pts_mean_cuda = pts_mean.cuda()
    # Precompute Fourier-encoded XYZ once (constant, no grad needed)
    with torch.no_grad():
        xyz_encoded = fourier_encode(pts_mean_cuda, num_freqs=4)  # [H, W, 27]
        xyz_feat_precomputed = xyz_encoded.permute(2, 0, 1).unsqueeze(0)  # [1, 27, H, W]
    print(f"  Precomputed xyz features: {xyz_feat_precomputed.shape}")

    bg_color = torch.zeros(3, device="cuda")
    sh_degree = 0  # DC only

    # LPIPS perceptual loss
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    param_count = sum(p.numel() for p in canonical_head.parameters())
    param_count += sum(p.numel() for p in deformation_head.parameters())
    print(f"  Total parameters: {param_count:,} ({param_count * 4 / 1e6:.1f} MB)")

    writer = SummaryWriter(log_dir)
    start_time = time.time()

    # ===== STAGE 1: Canonical Head =====
    print("\n" + "=" * 60)
    print("STAGE 1: Pixel-Aligned Canonical Head")
    print("=" * 60)

    deformation_head.eval()
    for p in deformation_head.parameters():
        p.requires_grad = False

    canonical_head.train()
    s1_cfg = cfg["training"]["stage1"]
    optimizer_s1 = optim.AdamW(canonical_head.parameters(), lr=s1_cfg["lr"])
    scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s1, T_max=s1_cfg["steps"]
    )

    torch.cuda.reset_peak_memory_stats()
    s1_weights = s1_cfg["loss_weights"]
    grad_clip = cfg["training"].get("grad_clip_max_norm", 1.0)
    supervision_cams = cfg["training"].get("stage1_supervision_cams", 2)
    ssim_warmup = cfg["training"].get("ssim_warmup_steps", 500)
    lpips_warmup = cfg["training"].get("lpips_warmup_steps", 1000)
    target_ssim = s1_weights["ssim"]
    lpips_weight = s1_weights.get("lpips", 0.0)

    eval_cam_name = cfg["data"]["eval_camera"]
    eval_cam = cameras[eval_cam_name]

    pbar = tqdm(range(s1_cfg["steps"]), desc="Stage 1 (Pixel)")
    for step in pbar:
        optimizer_s1.zero_grad()

        lambda_ssim = target_ssim * min(step / max(ssim_warmup, 1), 1.0)
        lpips_w = lpips_weight * min(step / max(lpips_warmup, 1), 1.0)

        batch = dataset.sample_training_batch(1, supervision_cams)
        tokens_mean = batch["tokens_mean"].cuda()

        canonical = canonical_head(tokens_mean, pts_mean_cuda, xyz_feat_precomputed)
        means3D, scales, rotations, opacity, shs = compose_pixel_gaussians(canonical)

        total_loss = torch.tensor(0.0, device="cuda")
        total_psnr = 0.0
        num_renders = 0

        for i, fidx in enumerate(batch["frame_indices"]):
            for cam_name in batch["cam_names"]:
                cam = cameras[cam_name]
                gt = batch["gt_images"][cam_name][i].cuda()

                rendered, radii, depth, _ = render_gaussians(
                    means3D, scales, rotations, opacity, shs, cam, bg_color, sh_degree
                )

                total_loss = total_loss + s1_weights["rgb"] * photometric_loss(
                    rendered, gt, lambda_ssim=lambda_ssim
                )

                if lpips_w > 0:
                    lpips_val = lpips_fn(rendered.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1)
                    total_loss = total_loss + lpips_w * lpips_val.mean()

                total_psnr += compute_psnr(rendered, gt)
                num_renders += 1

        total_loss = total_loss / max(num_renders, 1)
        avg_psnr = total_psnr / max(num_renders, 1)

        if s1_weights.get("scale_reg", 0) > 0:
            total_loss = total_loss + s1_weights["scale_reg"] * scale_regularization(scales)
        if s1_weights.get("opacity_reg", 0) > 0:
            total_loss = total_loss + s1_weights["opacity_reg"] * opacity_regularization(opacity)

        if torch.isnan(total_loss):
            print(f"\n[ABORT] NaN loss at step {step}!")
            return

        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(canonical_head.parameters(), grad_clip)
        optimizer_s1.step()
        scheduler_s1.step()

        vram_alloc, vram_peak = get_vram_gb()
        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "psnr": f"{avg_psnr:.1f}",
            "vram": f"{vram_alloc:.1f}/{total_vram:.0f}GB",
        })

        writer.add_scalar("stage1/loss", total_loss.item(), step)
        writer.add_scalar("stage1/psnr", avg_psnr, step)
        writer.add_scalar("vram/allocated", vram_alloc, step)
        writer.add_scalar("vram/peak", vram_peak, step)

        if step % 500 == 0:
            save_render(rendered, os.path.join(render_dir, "pixel_s1", f"step_{step:05d}.jpg"))
            # Novel view eval
            with torch.no_grad():
                gt_novel = dataset.load_frame_image(eval_cam_name, 0).cuda()
                rendered_novel, _, _, _ = render_gaussians(
                    means3D, scales, rotations, opacity, shs, eval_cam, bg_color, sh_degree
                )
                novel_psnr = compute_psnr(rendered_novel, gt_novel)
                save_render(rendered_novel, os.path.join(render_dir, "pixel_s1", f"novel_{step:05d}.jpg"))
            writer.add_scalar("stage1/novel_psnr", novel_psnr, step)
            tqdm.write(f"  Step {step}: train={avg_psnr:.2f} dB, novel={novel_psnr:.2f} dB")

        if step % 2000 == 0 and step > 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f"pixel_s1_step{step}.pt"),
                step, canonical_head, deformation_head, optimizer_s1
            )

    s1_peak = torch.cuda.max_memory_allocated() / 1e9
    save_checkpoint(
        os.path.join(ckpt_dir, "pixel_s1_final.pt"),
        s1_cfg["steps"], canonical_head, deformation_head, optimizer_s1
    )
    print(f"\nStage 1 complete. Peak VRAM: {s1_peak:.1f}GB")
    torch.cuda.empty_cache()

    # ===== STAGE 2: Deformation Head =====
    print("\n" + "=" * 60)
    print("STAGE 2: Pixel-Aligned Deformation Head")
    print("=" * 60)

    canonical_head.eval()
    for p in canonical_head.parameters():
        p.requires_grad = False

    deformation_head.train()
    for p in deformation_head.parameters():
        p.requires_grad = True

    s2_cfg = cfg["training"]["stage2"]
    optimizer_s2 = optim.AdamW(deformation_head.parameters(), lr=s2_cfg["lr"])
    scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2, T_max=s2_cfg["steps"]
    )

    torch.cuda.reset_peak_memory_stats()
    s2_weights = s2_cfg["loss_weights"]
    supervision_cams_s2 = cfg["training"]["supervision_cams"]

    pbar = tqdm(range(s2_cfg["steps"]), desc="Stage 2 (Pixel Deform)")
    for step in pbar:
        optimizer_s2.zero_grad()

        batch = dataset.sample_training_batch(1, supervision_cams_s2)
        tokens_mean = batch["tokens_mean"].cuda()

        with torch.no_grad():
            canonical = canonical_head(tokens_mean, pts_mean_cuda, xyz_feat_precomputed)

        total_loss = torch.tensor(0.0, device="cuda")
        total_psnr = 0.0

        for i, fidx in enumerate(batch["frame_indices"]):
            tokens_t = batch["tokens_frames"][i].cuda()
            deltas = deformation_head(tokens_t)

            means3D, scales, rotations, opacity, shs = compose_pixel_gaussians(
                canonical, deltas
            )

            for cam_name in batch["cam_names"]:
                cam = cameras[cam_name]
                gt = batch["gt_images"][cam_name][i].cuda()

                rendered, radii, depth, _ = render_gaussians(
                    means3D, scales, rotations, opacity, shs, cam, bg_color, sh_degree
                )

                total_loss = total_loss + s2_weights["rgb"] * photometric_loss(
                    rendered, gt, lambda_ssim=s2_weights["ssim"]
                )
                total_psnr += compute_psnr(rendered, gt)

        num_renders = len(batch["frame_indices"]) * len(batch["cam_names"])
        total_loss = total_loss / num_renders
        avg_psnr = total_psnr / num_renders

        if s2_weights.get("scale_reg", 0) > 0:
            total_loss = total_loss + s2_weights["scale_reg"] * scale_regularization(scales)
        if s2_weights.get("opacity_reg", 0) > 0:
            total_loss = total_loss + s2_weights["opacity_reg"] * opacity_regularization(opacity)

        if torch.isnan(total_loss):
            print(f"\n[ABORT] NaN loss at step {step}!")
            return

        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(deformation_head.parameters(), grad_clip)
        optimizer_s2.step()
        scheduler_s2.step()

        vram_alloc, vram_peak = get_vram_gb()
        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "psnr": f"{avg_psnr:.1f}",
            "vram": f"{vram_alloc:.1f}/{total_vram:.0f}GB",
        })

        writer.add_scalar("stage2/loss", total_loss.item(), step)
        writer.add_scalar("stage2/psnr", avg_psnr, step)

        if step % 500 == 0:
            save_render(rendered, os.path.join(render_dir, "pixel_s2", f"step_{step:05d}.jpg"))
            with torch.no_grad():
                gt_novel = dataset.load_frame_image(eval_cam_name, 0).cuda()
                rendered_novel, _, _, _ = render_gaussians(
                    means3D, scales, rotations, opacity, shs, eval_cam, bg_color, sh_degree
                )
                novel_psnr = compute_psnr(rendered_novel, gt_novel)
                save_render(rendered_novel, os.path.join(render_dir, "pixel_s2", f"novel_{step:05d}.jpg"))
            writer.add_scalar("stage2/novel_psnr", novel_psnr, step)
            tqdm.write(f"  Step {step}: train={avg_psnr:.2f} dB, novel={novel_psnr:.2f} dB")

        if step % 2000 == 0 and step > 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f"pixel_s2_step{step}.pt"),
                step, canonical_head, deformation_head, optimizer_s2
            )

    s2_peak = torch.cuda.max_memory_allocated() / 1e9
    save_checkpoint(
        os.path.join(ckpt_dir, "pixel_s2_final.pt"),
        s2_cfg["steps"], canonical_head, deformation_head, optimizer_s2
    )

    elapsed = time.time() - start_time
    writer.close()

    print("\n" + "=" * 60)
    print("=== Pixel-Aligned Training Complete ===")
    print(f"Stage 1 peak: {s1_peak:.1f}GB / {total_vram:.0f}GB")
    print(f"Stage 2 peak: {s2_peak:.1f}GB / {total_vram:.0f}GB")
    print(f"Total Gaussians: {H * W}")
    print(f"Total training time: {elapsed/3600:.1f}h ({elapsed:.0f}s)")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="D:/DecodeGaussians/experiments/overfit_coffee_martini/config.yaml")
    args = parser.parse_args()
    train(args.config)
