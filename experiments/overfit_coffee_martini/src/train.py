"""Training loop for the overfit test."""

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
from model import CanonicalGaussianHead, DeformationHead, compose_gaussians
from renderer import render_gaussians
from losses import photometric_loss, geometric_loss, depth_loss, tv_loss, compute_psnr, scale_regularization, opacity_regularization
import lpips


def get_vram_gb():
    """Return (allocated_gb, peak_gb)."""
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    return alloc, peak


def save_checkpoint(path, step, canonical_head, deformation_head, optimizer):
    """Save training checkpoint."""
    torch.save({
        "step": step,
        "canonical_head": canonical_head.state_dict(),
        "deformation_head": deformation_head.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def save_render(rendered, path):
    """Save rendered image [3, H, W] as JPEG."""
    import cv2
    img = (rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def train(config_path: str, resume_stage: int = 1):
    """Main training function. resume_stage: 1=full, 2=skip stage1, 3=skip stage1+2."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_dir = os.path.dirname(os.path.dirname(config_path))
    cache_dir = cfg["paths"]["cache_dir"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    render_dir = cfg["paths"]["render_dir"]
    log_dir = cfg["paths"]["log_dir"]

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(render_dir, "stage1"), exist_ok=True)
    os.makedirs(os.path.join(render_dir, "stage2"), exist_ok=True)
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

    # Initialize xyz from point cloud — use actual 3D points for each patch
    P = dataset.num_patches
    K = cfg["model"].get("num_gaussians_per_patch", 1)
    pts_mean = dataset.points_map.mean(dim=0)  # [H, W, 3]
    H, W = pts_mean.shape[0], pts_mean.shape[1]

    # Compute per-patch point assignments from the full point cloud
    # STV2 ViT patches form a grid; map pixels to nearest patch
    pts_flat = pts_mean.reshape(-1, 3)  # [H*W, 3]

    # Estimate patch grid dimensions from P
    # STV2 uses patch_size ~14, so grid ≈ (W/14, H/14)
    grid_w = round((P * W / H) ** 0.5)
    grid_h = round(P / grid_w)
    while grid_w * grid_h < P:
        grid_w += 1
    print(f"  Estimated patch grid: {grid_h}x{grid_w} = {grid_h*grid_w} (P={P})")

    # Assign each pixel to its patch based on grid position
    patch_h = H / grid_h
    patch_w = W / grid_w
    pixel_y = torch.arange(H).float()
    pixel_x = torch.arange(W).float()
    yy, xx = torch.meshgrid(pixel_y, pixel_x, indexing="ij")
    patch_idx_y = (yy / patch_h).long().clamp(0, grid_h - 1)
    patch_idx_x = (xx / patch_w).long().clamp(0, grid_w - 1)
    pixel_to_patch = (patch_idx_y * grid_w + patch_idx_x).reshape(-1)  # [H*W]

    # For each patch, gather its pixel 3D points and subsample K
    init_xyz_per_gaussian = torch.zeros(P, K, 3)
    for p in range(P):
        mask = (pixel_to_patch == p)
        pts_p = pts_flat[mask]  # points belonging to patch p
        if pts_p.shape[0] == 0:
            # Fallback: use overall mean
            init_xyz_per_gaussian[p] = pts_flat.mean(dim=0).unsqueeze(0).expand(K, -1)
        elif pts_p.shape[0] >= K:
            # Subsample K points uniformly
            idx = torch.linspace(0, pts_p.shape[0] - 1, K).long()
            init_xyz_per_gaussian[p] = pts_p[idx]
        else:
            # Repeat to fill K slots
            repeats = K // pts_p.shape[0] + 1
            init_xyz_per_gaussian[p] = pts_p.repeat(repeats, 1)[:K]

    init_xyz_per_gaussian = init_xyz_per_gaussian.reshape(P * K, 3)
    print(f"  init_xyz_per_gaussian: range [{init_xyz_per_gaussian.min():.3f}, {init_xyz_per_gaussian.max():.3f}]")
    print(f"  Gaussians per patch (K): {K}, total Gaussians: {P * K}")

    # Also compute patch centers for eval compatibility
    indices = torch.linspace(0, pts_flat.shape[0] - 1, P).long()
    init_xyz = pts_flat[indices]

    canonical_head = CanonicalGaussianHead(
        dim_in=cfg["model"]["canonical"]["dim_in"],
        dim_hidden=cfg["model"]["canonical"]["dim_hidden"],
        sh_degree=cfg["model"]["canonical"]["sh_degree"],
        init_xyz=init_xyz,
        num_gaussians_per_patch=K,
        init_xyz_per_gaussian=init_xyz_per_gaussian,
    ).cuda()

    deformation_head = DeformationHead(
        dim_in=cfg["model"]["deformation"]["dim_in"],
        dim_hidden=cfg["model"]["deformation"]["dim_hidden"],
        n_attn_heads=cfg["model"]["deformation"]["n_attn_heads"],
        n_attn_layers=cfg["model"]["deformation"]["n_attn_layers"],
        sh_degree=cfg["model"]["canonical"]["sh_degree"],
        num_gaussians_per_patch=K,
    ).cuda()

    bg_color = torch.zeros(3, device="cuda")
    sh_degree = cfg["model"]["canonical"]["sh_degree"]

    # LPIPS perceptual loss (AlexNet backbone)
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    writer = SummaryWriter(log_dir)
    start_time = time.time()

    # Load checkpoint if resuming from a later stage
    if resume_stage > 1:
        ckpt_path = os.path.join(ckpt_dir, "stage1_final.pt")
        if resume_stage >= 3:
            ckpt_path = os.path.join(ckpt_dir, "stage2_final.pt")
        print(f"\n  Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False)
        canonical_head.load_state_dict(ckpt["canonical_head"])
        deformation_head.load_state_dict(ckpt["deformation_head"])

    s1_peak = 0.0
    s2_peak = 0.0

    # ===== STAGE 1: Canonical Head (MULTI-VIEW) =====
    if resume_stage > 1:
        print("\n  Skipping Stage 1 (resuming from stage {})".format(resume_stage))
    else:
        print("\n" + "=" * 60)
        print("STAGE 1: Training Canonical Head (Multi-View)")
        print("=" * 60)

    deformation_head.eval()
    for p in deformation_head.parameters():
        p.requires_grad = False

    canonical_head.train()
    optimizer_s1 = optim.AdamW(canonical_head.parameters(), lr=cfg["training"]["stage1"]["lr"])
    scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s1, T_max=cfg["training"]["stage1"]["steps"]
    )

    torch.cuda.reset_peak_memory_stats()
    batch_frames = cfg["training"]["batch_frames"]
    s1_weights = cfg["training"]["stage1"]["loss_weights"]
    grad_clip = cfg["training"].get("grad_clip_max_norm", 0.0)
    s1_supervision_cams = cfg["training"].get("stage1_supervision_cams", 4)

    ssim_warmup_steps = cfg["training"].get("ssim_warmup_steps", 500)
    lpips_warmup_steps = cfg["training"].get("lpips_warmup_steps", 1000)
    target_ssim = s1_weights["ssim"]
    lpips_weight = s1_weights.get("lpips", 0.05)

    # Scale annealing: large Gaussians early for stable training, shrink later for 3D precision
    anneal_start = cfg["training"].get("scale_anneal_start", 0)
    anneal_target = cfg["training"].get("scale_anneal_target", 1.0)
    total_s1_steps = cfg["training"]["stage1"]["steps"]

    pbar = tqdm(range(total_s1_steps if resume_stage <= 1 else 0), desc="Stage 1 (Canonical)")
    for step in pbar:
        optimizer_s1.zero_grad()

        # SSIM warmup: start L1-dominant, ramp SSIM over warmup period
        lambda_ssim = target_ssim * min(step / max(ssim_warmup_steps, 1), 1.0)
        # LPIPS warmup: delay perceptual loss to avoid destabilizing early training
        lpips_w = lpips_weight * min(step / max(lpips_warmup_steps, 1), 1.0)

        # Scale annealing: 1.0 → anneal_target after anneal_start
        if step < anneal_start or anneal_target >= 1.0:
            scale_factor = 1.0
        else:
            progress = min((step - anneal_start) / max(total_s1_steps - anneal_start, 1), 1.0)
            scale_factor = 1.0 - (1.0 - anneal_target) * progress

        # Multi-view supervision: render from multiple cameras
        batch = dataset.sample_training_batch(batch_frames, supervision_cams=s1_supervision_cams)
        tokens_mean = batch["tokens_mean"].cuda()

        canonical = canonical_head(tokens_mean)
        means3D, scales, rotations, opacity, shs = compose_gaussians(canonical, scale_factor=scale_factor)

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

                # LPIPS perceptual loss with warmup
                if lpips_w > 0:
                    lpips_val = lpips_fn(rendered.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1)
                    total_loss = total_loss + lpips_w * lpips_val.mean()

                total_psnr += compute_psnr(rendered, gt)
                num_renders += 1

            # Geometric loss (once per frame, not per camera)
            pts_map = batch["points_map_frames"][i].cuda()
            if s1_weights.get("geo", 0) > 0:
                total_loss = total_loss + s1_weights["geo"] * geometric_loss(means3D, pts_map)

            # Depth loss: rendered depth vs STV2 point cloud depth (input camera only)
            if s1_weights.get("depth", 0) > 0 and cfg["data"]["input_camera"] in batch["cam_names"]:
                input_cam = cameras[cfg["data"]["input_camera"]]
                rendered_input, _, depth_input, _ = render_gaussians(
                    means3D, scales, rotations, opacity, shs, input_cam, bg_color, sh_degree
                )
                total_loss = total_loss + s1_weights["depth"] * depth_loss(
                    depth_input, pts_map, input_cam
                )

        total_loss = total_loss / max(num_renders, 1)
        avg_psnr = total_psnr / max(num_renders, 1)

        # Regularization losses
        if s1_weights.get("scale_reg", 0) > 0:
            total_loss = total_loss + s1_weights["scale_reg"] * scale_regularization(scales)
        if s1_weights.get("opacity_reg", 0) > 0:
            total_loss = total_loss + s1_weights["opacity_reg"] * opacity_regularization(opacity)

        if torch.isnan(total_loss):
            print(f"\n[ABORT] NaN loss at step {step}!")
            print(f"  means3D range: [{means3D.min():.3f}, {means3D.max():.3f}]")
            print(f"  scales range: [{scales.min():.6f}, {scales.max():.6f}]")
            print(f"  opacity range: [{opacity.min():.3f}, {opacity.max():.3f}]")
            return

        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(canonical_head.parameters(), grad_clip)
        optimizer_s1.step()
        scheduler_s1.step()

        vram_alloc, vram_peak = get_vram_gb()
        if vram_alloc > warn_vram:
            tqdm.write(f"[WARN] VRAM {vram_alloc:.1f}GB > {warn_vram}GB threshold")

        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "psnr": f"{avg_psnr:.1f}",
            "vram": f"{vram_alloc:.1f}/{total_vram:.0f}GB",
            "peak": f"{vram_peak:.1f}GB",
        })

        writer.add_scalar("stage1/loss", total_loss.item(), step)
        writer.add_scalar("stage1/psnr", avg_psnr, step)
        writer.add_scalar("vram/allocated", vram_alloc, step)
        writer.add_scalar("vram/peak", vram_peak, step)
        writer.add_scalar("stage1/lr", scheduler_s1.get_last_lr()[0], step)

        if step % cfg["training"]["checkpointing"]["render_every"] == 0:
            save_render(rendered, os.path.join(render_dir, "stage1", f"step_{step:05d}.jpg"))

        if step % cfg["training"]["checkpointing"]["save_every"] == 0 and step > 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f"stage1_step{step}.pt"),
                step, canonical_head, deformation_head, optimizer_s1
            )

        if step == 500 and avg_psnr < 10:
            tqdm.write(f"[WARN] PSNR={avg_psnr:.1f}dB at step 500 — likely camera convention mismatch")

        # Novel view evaluation every 1000 steps
        if step % 1000 == 0:
            with torch.no_grad():
                eval_cam_name = cfg["data"]["eval_camera"]
                eval_cam = cameras[eval_cam_name]
                frame_idx = 0  # Use first frame for consistent eval
                gt_novel = dataset.load_frame_image(eval_cam_name, frame_idx).cuda()
                rendered_novel, _, _, _ = render_gaussians(
                    means3D.detach(), scales.detach(), rotations.detach(),
                    opacity.detach(), shs.detach(), eval_cam, bg_color, sh_degree
                )
                novel_psnr = compute_psnr(rendered_novel, gt_novel)
                writer.add_scalar("stage1/novel_psnr", novel_psnr, step)
                tqdm.write(f"  Step {step}: train={avg_psnr:.2f}, novel={novel_psnr:.2f} dB")
                save_render(rendered_novel, os.path.join(render_dir, "stage1", f"novel_{step:05d}.jpg"))

    s1_peak = torch.cuda.max_memory_allocated() / 1e9
    save_checkpoint(
        os.path.join(ckpt_dir, "stage1_final.pt"),
        cfg["training"]["stage1"]["steps"], canonical_head, deformation_head, optimizer_s1
    )

    print(f"\nStage 1 complete. Peak VRAM: {s1_peak:.1f}GB")
    torch.cuda.empty_cache()

    # ===== STAGE 2: Deformation Head =====
    print("\n" + "=" * 60)
    print("STAGE 2: Training Deformation Head")
    print("=" * 60)

    canonical_head.eval()
    for p in canonical_head.parameters():
        p.requires_grad = False

    deformation_head.train()
    for p in deformation_head.parameters():
        p.requires_grad = True

    optimizer_s2 = optim.AdamW(deformation_head.parameters(), lr=cfg["training"]["stage2"]["lr"])
    scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2, T_max=cfg["training"]["stage2"]["steps"]
    )

    torch.cuda.reset_peak_memory_stats()
    s2_weights = cfg["training"]["stage2"]["loss_weights"]
    supervision_cams = cfg["training"]["supervision_cams"]

    pbar = tqdm(range(cfg["training"]["stage2"]["steps"] if resume_stage <= 2 else 0), desc="Stage 2 (Deformation)")
    for step in pbar:
        optimizer_s2.zero_grad()

        batch = dataset.sample_training_batch(batch_frames, supervision_cams)

        tokens_mean = batch["tokens_mean"].cuda()
        with torch.no_grad():
            canonical = canonical_head(tokens_mean)

        total_loss = torch.tensor(0.0, device="cuda")
        total_psnr = 0.0
        prev_all_deltas = None

        for i, fidx in enumerate(batch["frame_indices"]):
            tokens_t = batch["tokens_frames"][i].cuda()
            deltas = deformation_head(tokens_t)

            means3D, scales, rotations, opacity, shs = compose_gaussians(canonical, deltas, scale_factor=anneal_target)

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

            if prev_all_deltas is not None:
                total_loss = total_loss + s2_weights["tv"] * tv_loss(
                    deltas["all_deltas"], prev_all_deltas
                )
            prev_all_deltas = deltas["all_deltas"].detach()

        num_renders = len(batch["frame_indices"]) * len(batch["cam_names"])
        total_loss = total_loss / num_renders
        avg_psnr = total_psnr / num_renders

        # Regularization losses (use last frame's Gaussians)
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
        if vram_alloc > warn_vram:
            tqdm.write(f"[WARN] VRAM {vram_alloc:.1f}GB > {warn_vram}GB threshold")

        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "psnr": f"{avg_psnr:.1f}",
            "vram": f"{vram_alloc:.1f}/{total_vram:.0f}GB",
            "peak": f"{vram_peak:.1f}GB",
        })

        writer.add_scalar("stage2/loss", total_loss.item(), step)
        writer.add_scalar("stage2/psnr", avg_psnr, step)
        writer.add_scalar("vram/allocated_s2", vram_alloc, step)
        writer.add_scalar("vram/peak_s2", vram_peak, step)

        if step % cfg["training"]["checkpointing"]["render_every"] == 0:
            save_render(rendered, os.path.join(render_dir, "stage2", f"step_{step:05d}.jpg"))

        if step % cfg["training"]["checkpointing"]["save_every"] == 0 and step > 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f"stage2_step{step}.pt"),
                step, canonical_head, deformation_head, optimizer_s2
            )

        # Novel view evaluation every 1000 steps
        if step % 1000 == 0:
            with torch.no_grad():
                eval_cam_name = cfg["data"]["eval_camera"]
                eval_cam = cameras[eval_cam_name]
                frame_idx = 0
                tokens_mean_eval = dataset.get_tokens_mean().cuda()
                canonical_eval = canonical_head(tokens_mean_eval)
                tokens_t_eval = dataset.get_tokens_frame(frame_idx).cuda()
                deltas_eval = deformation_head(tokens_t_eval)
                m3d, sc, rot, op, sh = compose_gaussians(canonical_eval, deltas_eval, scale_factor=anneal_target)
                gt_novel = dataset.load_frame_image(eval_cam_name, frame_idx).cuda()
                rendered_novel, _, _, _ = render_gaussians(m3d, sc, rot, op, sh, eval_cam, bg_color, sh_degree)
                novel_psnr = compute_psnr(rendered_novel, gt_novel)
                writer.add_scalar("stage2/novel_psnr", novel_psnr, step)
                tqdm.write(f"  Step {step}: train={avg_psnr:.2f}, novel={novel_psnr:.2f} dB")
                save_render(rendered_novel, os.path.join(render_dir, "stage2", f"novel_{step:05d}.jpg"))

    s2_peak = torch.cuda.max_memory_allocated() / 1e9
    save_checkpoint(
        os.path.join(ckpt_dir, "stage2_final.pt"),
        cfg["training"]["stage2"]["steps"], canonical_head, deformation_head, optimizer_s2
    )

    # ===== STAGE 3: Joint Fine-tuning =====
    if "stage3" in cfg["training"]:
        print("\n" + "=" * 60)
        print("STAGE 3: Joint Fine-tuning (Both Heads)")
        print("=" * 60)

        os.makedirs(os.path.join(render_dir, "stage3"), exist_ok=True)

        canonical_head.train()
        deformation_head.train()
        for p in canonical_head.parameters():
            p.requires_grad = True
        for p in deformation_head.parameters():
            p.requires_grad = True

        s3_lr = cfg["training"]["stage3"]["lr"]
        optimizer_s3 = optim.AdamW(
            list(canonical_head.parameters()) + list(deformation_head.parameters()),
            lr=s3_lr,
        )
        scheduler_s3 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_s3, T_max=cfg["training"]["stage3"]["steps"]
        )

        torch.cuda.reset_peak_memory_stats()
        s3_weights = cfg["training"]["stage3"]["loss_weights"]
        s3_steps = cfg["training"]["stage3"]["steps"]

        pbar = tqdm(range(s3_steps), desc="Stage 3 (Joint)")
        for step in pbar:
            optimizer_s3.zero_grad()

            batch = dataset.sample_training_batch(batch_frames, supervision_cams)

            tokens_mean = batch["tokens_mean"].cuda()
            canonical = canonical_head(tokens_mean)

            total_loss = torch.tensor(0.0, device="cuda")
            total_psnr = 0.0
            prev_all_deltas = None

            for i, fidx in enumerate(batch["frame_indices"]):
                tokens_t = batch["tokens_frames"][i].cuda()
                deltas = deformation_head(tokens_t)

                means3D, scales, rotations, opacity, shs = compose_gaussians(canonical, deltas, scale_factor=anneal_target)

                for cam_name in batch["cam_names"]:
                    cam = cameras[cam_name]
                    gt = batch["gt_images"][cam_name][i].cuda()

                    rendered, radii, depth, _ = render_gaussians(
                        means3D, scales, rotations, opacity, shs, cam, bg_color, sh_degree
                    )

                    total_loss = total_loss + s3_weights["rgb"] * photometric_loss(
                        rendered, gt, lambda_ssim=s3_weights["ssim"]
                    )

                    total_psnr += compute_psnr(rendered, gt)

                if prev_all_deltas is not None:
                    total_loss = total_loss + s3_weights["tv"] * tv_loss(
                        deltas["all_deltas"], prev_all_deltas
                    )
                prev_all_deltas = deltas["all_deltas"].detach()

            num_renders = len(batch["frame_indices"]) * len(batch["cam_names"])
            total_loss = total_loss / num_renders
            avg_psnr = total_psnr / num_renders

            if s3_weights.get("scale_reg", 0) > 0:
                total_loss = total_loss + s3_weights["scale_reg"] * scale_regularization(scales)
            if s3_weights.get("opacity_reg", 0) > 0:
                total_loss = total_loss + s3_weights["opacity_reg"] * opacity_regularization(opacity)

            if torch.isnan(total_loss):
                print(f"\n[ABORT] NaN loss at step {step}!")
                break

            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(canonical_head.parameters()) + list(deformation_head.parameters()),
                    grad_clip,
                )
            optimizer_s3.step()
            scheduler_s3.step()

            vram_alloc, vram_peak = get_vram_gb()
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "psnr": f"{avg_psnr:.1f}",
                "vram": f"{vram_alloc:.1f}/{total_vram:.0f}GB",
            })

            writer.add_scalar("stage3/loss", total_loss.item(), step)
            writer.add_scalar("stage3/psnr", avg_psnr, step)

            if step % cfg["training"]["checkpointing"]["render_every"] == 0:
                save_render(rendered, os.path.join(render_dir, "stage3", f"step_{step:05d}.jpg"))

            if step % cfg["training"]["checkpointing"]["save_every"] == 0 and step > 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"stage3_step{step}.pt"),
                    step, canonical_head, deformation_head, optimizer_s3
                )

            # Novel view evaluation every 1000 steps
            if step % 1000 == 0:
                with torch.no_grad():
                    eval_cam_name = cfg["data"]["eval_camera"]
                    eval_cam = cameras[eval_cam_name]
                    frame_idx = 0
                    tokens_mean_eval = dataset.get_tokens_mean().cuda()
                    canonical_eval = canonical_head(tokens_mean_eval)
                    tokens_t_eval = dataset.get_tokens_frame(frame_idx).cuda()
                    deltas_eval = deformation_head(tokens_t_eval)
                    m3d, sc, rot, op, sh = compose_gaussians(canonical_eval, deltas_eval, scale_factor=anneal_target)
                    gt_novel = dataset.load_frame_image(eval_cam_name, frame_idx).cuda()
                    rendered_novel, _, _, _ = render_gaussians(m3d, sc, rot, op, sh, eval_cam, bg_color, sh_degree)
                    novel_psnr = compute_psnr(rendered_novel, gt_novel)
                    writer.add_scalar("stage3/novel_psnr", novel_psnr, step)
                    tqdm.write(f"  Step {step}: train={avg_psnr:.2f}, novel={novel_psnr:.2f} dB")
                    save_render(rendered_novel, os.path.join(render_dir, "stage3", f"novel_{step:05d}.jpg"))

        s3_peak = torch.cuda.max_memory_allocated() / 1e9
        save_checkpoint(
            os.path.join(ckpt_dir, "stage3_final.pt"),
            s3_steps, canonical_head, deformation_head, optimizer_s3
        )
        print(f"\nStage 3 complete. Peak VRAM: {s3_peak:.1f}GB")

    elapsed = time.time() - start_time
    writer.close()

    print("\n" + "=" * 60)
    print("=== VRAM Report ===")
    print(f"Stage 1 peak: {s1_peak:.1f}GB / {total_vram:.0f}GB")
    print(f"Stage 2 peak: {s2_peak:.1f}GB / {total_vram:.0f}GB")
    print(f"Batch frames: {batch_frames}")
    print(f"Total training time: {elapsed/3600:.1f}h ({elapsed:.0f}s)")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="D:/DecodeGaussians/experiments/overfit_coffee_martini/config.yaml")
    parser.add_argument("--resume-stage", type=int, default=1,
                        help="Resume from stage N (1=full, 2=skip stage1, 3=skip stage1+2)")
    args = parser.parse_args()
    train(args.config, resume_stage=args.resume_stage)
