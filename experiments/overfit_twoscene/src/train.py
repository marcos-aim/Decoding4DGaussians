"""
Multi-scene training loop with gradient accumulation.

Extends the single-scene training to support multiple datasets, accumulating
gradients across all scenes before each optimizer step. This enables learning
true feedforward Gaussian decoding that generalizes across scenes.
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
from dataset import MultiSceneDataset
from model import CanonicalGaussianHead, DeformationHead, compose_gaussians
from renderer import render_gaussians
from losses import photometric_loss, compute_psnr, scale_regularization, opacity_regularization


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


def find_latest_checkpoint(ckpt_dir):
    """Find the most recent checkpoint in the checkpoint directory.

    Returns:
        (checkpoint_path, stage, step) or (None, None, None) if no checkpoint found
    """
    # Check in priority order: stage 3 > stage 2 > stage 1
    # Within each stage, prefer highest step number
    checkpoint_patterns = [
        ("stage3_step*.pt", 3),  # Stage 3 intermediate checkpoints
        ("stage3_final.pt", 3),  # Stage 3 final
        ("stage2_step*.pt", 2),  # Stage 2 intermediate checkpoints
        ("stage2_final.pt", 2),  # Stage 2 final
        ("stage1_step*.pt", 1),  # Stage 1 intermediate checkpoints
        ("stage1_final.pt", 1),  # Stage 1 final
    ]

    latest_ckpt = None
    latest_step = -1
    latest_stage = None

    # Iterate through stages in priority order
    for pattern, stage in checkpoint_patterns:
        # Skip to next stage if we already found a checkpoint in a higher stage
        if latest_stage is not None and stage < latest_stage:
            continue

        ckpt_paths = glob.glob(os.path.join(ckpt_dir, pattern))

        for ckpt_path in ckpt_paths:
            # Extract step number from intermediate checkpoints
            if "step" in pattern and "*" in pattern:
                basename = os.path.basename(ckpt_path)
                # Extract number from "stageN_stepXXXX.pt"
                try:
                    step = int(basename.split("step")[1].split(".pt")[0])
                except (IndexError, ValueError):
                    continue
            else:
                # For final checkpoints, load to get step
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                    step = ckpt.get("step", 0)
                except:
                    continue

            # Track the checkpoint with highest step number within this stage
            # If we found a higher stage, reset the comparison
            if latest_stage is None or stage > latest_stage:
                latest_step = step
                latest_ckpt = ckpt_path
                latest_stage = stage
            elif stage == latest_stage and step > latest_step:
                latest_step = step
                latest_ckpt = ckpt_path

    return latest_ckpt, latest_stage, latest_step


def save_render(rendered, path):
    import cv2
    img = (rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def tv_loss(deltas_t: torch.Tensor, deltas_prev: torch.Tensor) -> torch.Tensor:
    return (deltas_t - deltas_prev).pow(2).mean()


def train(config_path: str, resume_stage: int = 1, restart_latest: bool = False):
    """Main training function.

    Args:
        config_path: Path to config YAML
        resume_stage: 1=full, 2=skip stage1, 3=skip stage1+2
        restart_latest: If True, find and load the most recent checkpoint, continuing from that step
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = cfg["logging"]["ckpt_dir"]
    render_dir = cfg["logging"]["render_dir"]
    log_dir = cfg["logging"]["log_dir"]

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(render_dir, "stage1"), exist_ok=True)
    os.makedirs(os.path.join(render_dir, "stage2"), exist_ok=True)
    os.makedirs(os.path.join(render_dir, "stage3"), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    target_w, target_h = cfg["rendering"]["image_width"], cfg["rendering"]["image_height"]

    # Build scene configs with cameras
    scene_configs = []
    for scene_cfg in cfg["scenes"]:
        scene_dir = os.path.dirname(scene_cfg["poses_bounds_path"])
        cam_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(scene_dir, "cam*.mp4"))])

        cameras = build_cameras_from_llff(
            scene_cfg["poses_bounds_path"], target_w, target_h, cam_files
        )

        scene_configs.append({
            "name": scene_cfg["name"],
            "cache_dir": scene_cfg["cache_dir"],
            "cameras": cameras,
            "input_camera": scene_cfg.get("input_camera", "cam01"),
            "eval_camera": scene_cfg.get("eval_camera", "cam00"),
            "weight": scene_cfg.get("weight", 1.0),
        })

    # Create multi-scene dataset
    multi_dataset = MultiSceneDataset(
        scene_configs,
        normalize_coords=cfg["normalization"].get("enabled", True)
    )

    multi_dataset.print_statistics()

    # Initialize models using first scene's dimensions
    first_dataset = multi_dataset.scenes[0]["dataset"]
    P = first_dataset.num_patches
    K = cfg["model"]["num_gaussians_per_patch"]

    # Initialize anchor points from first scene (normalized coordinates)
    pts_mean = first_dataset.points_map.mean(dim=0)
    H, W = pts_mean.shape[0], pts_mean.shape[1]
    pts_flat = pts_mean.reshape(-1, 3)

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

    # Create models
    canonical_head = CanonicalGaussianHead(
        dim_in=cfg["model"]["dim_tokens"],
        dim_hidden=cfg["model"]["dim_canonical_hidden"],
        sh_degree=cfg["model"]["sh_degree"],
        init_xyz=init_xyz,
        num_gaussians_per_patch=K,
        init_xyz_per_gaussian=init_xyz_per_gaussian,
    ).cuda()

    deformation_head = DeformationHead(
        dim_in=cfg["model"]["dim_tokens"],
        dim_hidden=cfg["model"]["dim_deformation_hidden"],
        n_attn_heads=cfg["model"]["n_attn_heads"],
        n_attn_layers=cfg["model"]["n_attn_layers"],
        sh_degree=cfg["model"]["sh_degree"],
        num_gaussians_per_patch=K,
    ).cuda()

    print(f"\n{'='*60}")
    print(f"Model Architecture:")
    print(f"  Canonical Head: {sum(p.numel() for p in canonical_head.parameters())/1e6:.1f}M params")
    print(f"  Deformation Head: {sum(p.numel() for p in deformation_head.parameters())/1e6:.1f}M params")
    print(f"  Total Gaussians: {P * K}")
    print(f"{'='*60}\n")

    bg_color = torch.tensor(cfg["rendering"]["bg_color"], device="cuda")
    sh_degree = cfg["model"]["sh_degree"]
    grad_clip = cfg["training"].get("grad_clip_max_norm", 0.0)

    # Handle checkpoint loading
    restart_from_step = None
    restart_from_stage = None

    if restart_latest:
        # Find and load the most recent checkpoint
        latest_ckpt, latest_stage, latest_step = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt is not None:
            print(f"\n{'='*60}")
            print(f"RESTART MODE: Loading latest checkpoint")
            print(f"  Checkpoint: {os.path.basename(latest_ckpt)}")
            print(f"  Stage: {latest_stage}")
            print(f"  Step: {latest_step}")
            print(f"{'='*60}\n")
            ckpt = torch.load(latest_ckpt, weights_only=False)
            canonical_head.load_state_dict(ckpt["canonical_head"])
            deformation_head.load_state_dict(ckpt["deformation_head"])
            restart_from_step = latest_step
            restart_from_stage = latest_stage
        else:
            print(f"\n{'='*60}")
            print(f"WARNING: --restart-latest specified but no checkpoints found")
            print(f"Starting training from scratch")
            print(f"{'='*60}\n")
    elif resume_stage > 1:
        # Load checkpoint for stage skipping (original behavior)
        ckpt_path = os.path.join(ckpt_dir, "stage1_final.pt")
        if resume_stage >= 3:
            ckpt_path = os.path.join(ckpt_dir, "stage2_final.pt")
        print(f"\n{'='*60}")
        print(f"Resuming from {ckpt_path}")
        print(f"{'='*60}\n")
        ckpt = torch.load(ckpt_path, weights_only=False)
        canonical_head.load_state_dict(ckpt["canonical_head"])
        deformation_head.load_state_dict(ckpt["deformation_head"])

    writer = SummaryWriter(log_dir)
    start_time = time.time()

    # ===== STAGE 1: Canonical Head Only =====
    should_skip_stage1 = (resume_stage > 1) or (restart_from_stage and restart_from_stage > 1)

    if should_skip_stage1:
        if restart_from_stage:
            print(f"\n  Skipping Stage 1 (restarting from stage {restart_from_stage})")
        else:
            print(f"\n  Skipping Stage 1 (resuming from stage {resume_stage})")
    else:
        print("\n" + "=" * 60)
        print("STAGE 1: Training Canonical Head (Multi-Scene)")
        print("=" * 60)

        canonical_head.train()
        deformation_head.eval()

        s1_steps = cfg["training"]["stage1_steps"]
        s1_lr = cfg["training"]["stage1_lr"]
        s1_batch_frames = cfg["training"]["stage1_batch_frames"]
        s1_supervision_cams = cfg["training"]["stage1_supervision_cams"]

        optimizer_s1 = optim.AdamW(canonical_head.parameters(), lr=s1_lr)
        scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=s1_steps)

        # If restarting within stage 1, load optimizer state and advance scheduler
        should_continue_stage1 = (restart_from_stage == 1 and restart_from_step is not None)
        start_step_s1 = 0
        if should_continue_stage1:
            print(f"  Continuing Stage 1 from step {restart_from_step}")
            start_step_s1 = restart_from_step
            if "optimizer" in ckpt:
                optimizer_s1.load_state_dict(ckpt["optimizer"])
            # Advance scheduler to correct position
            for _ in range(restart_from_step):
                scheduler_s1.step()

        torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(range(start_step_s1, s1_steps), desc="Stage 1 (Canonical)")
        for step in pbar:
            optimizer_s1.zero_grad()

            # Sample batches from all scenes
            scene_batches = multi_dataset.sample_multi_scene_batch(s1_batch_frames, s1_supervision_cams)

            total_psnr = 0.0
            total_renders = 0
            total_loss = torch.tensor(0.0, device="cuda")  # Accumulate on GPU

            # Accumulate gradients across all scenes
            for scene_batch in scene_batches:
                scene_name = scene_batch["scene_name"]
                scene_weight = scene_batch["scene_weight"]
                scene_cameras = scene_batch["cameras"]

                tokens_mean = scene_batch["tokens_mean"].cuda()
                canonical = canonical_head(tokens_mean)

                # Direct loss accumulation (no list, no stack)
                scene_loss = torch.tensor(0.0, device="cuda")
                last_rendered = None

                for i, fidx in enumerate(scene_batch["frame_indices"]):
                    # For stage 1, no deformation (canonical only)
                    means3D, scales, rotations, opacity, shs = compose_gaussians(canonical)

                    for cam_name in scene_batch["cam_names"]:
                        cam = scene_cameras[cam_name]
                        gt = scene_batch["gt_images"][cam_name][i].cuda()

                        rendered, radii, depth, _ = render_gaussians(
                            means3D, scales, rotations, opacity, shs, cam, bg_color, sh_degree
                        )

                        loss = photometric_loss(rendered, gt, lambda_ssim=0.2)
                        scene_loss = scene_loss + loss  # Direct accumulation
                        total_psnr += compute_psnr(rendered, gt)
                        total_renders += 1

                        # Keep last rendered for saving, clean up others
                        last_rendered = rendered.detach()
                        del radii, depth, gt

                # Normalize scene loss by number of renders
                num_scene_renders = len(scene_batch["frame_indices"]) * len(scene_batch["cam_names"])
                scene_loss = scene_loss / max(num_scene_renders, 1)

                # Apply scene weight and accumulate into total loss
                weighted_scene_loss = scene_loss * scene_weight / len(scene_batches)
                total_loss = total_loss + weighted_scene_loss

            # Single backward call for all scenes
            total_loss.backward()

            # Single optimizer step after all scenes
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(canonical_head.parameters(), grad_clip)
            optimizer_s1.step()
            scheduler_s1.step()

            # Get loss for logging (single .item() call)
            avg_loss = total_loss.item()
            avg_psnr = total_psnr / max(total_renders, 1)

            vram_alloc, vram_peak = get_vram_gb()
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "psnr": f"{avg_psnr:.1f}",
                "vram": f"{vram_alloc:.1f}GB",
            })

            writer.add_scalar("stage1/loss", avg_loss, step)
            writer.add_scalar("stage1/psnr", avg_psnr, step)

            if step % cfg["logging"]["save_render_every"] == 0:
                save_render(last_rendered, os.path.join(render_dir, "stage1", f"step_{step:05d}.jpg"))

            if step % cfg["training"]["save_ckpt_every"] == 0 and step > 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"stage1_step{step}.pt"),
                    step, canonical_head, deformation_head, optimizer_s1
                )

        save_checkpoint(
            os.path.join(ckpt_dir, "stage1_final.pt"),
            s1_steps, canonical_head, deformation_head, optimizer_s1
        )
        print(f"\nStage 1 complete. Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        torch.cuda.empty_cache()

    # ===== STAGE 2 & 3: DISABLED FOR ADAPTIVE SCALE EXPERIMENT =====
    # Stages 2 and 3 are commented out to test stage 1 only

    # ===== STAGE 2: Deformation Head Only =====
    should_skip_stage2 = (resume_stage > 2) or (restart_from_stage and restart_from_stage > 2)
    should_continue_stage2 = (restart_from_stage == 2 and restart_from_step is not None)

    if should_skip_stage2:
        if restart_from_stage:
            print(f"\n  Skipping Stage 2 (restarting from stage {restart_from_stage})")
        else:
            print(f"\n  Skipping Stage 2 (resuming from stage {resume_stage})")
    else:
        print("\n" + "=" * 60)
        print("STAGE 2: Training Deformation Head (Multi-Scene)")
        print("=" * 60)

        canonical_head.eval()
        for p in canonical_head.parameters():
            p.requires_grad = False

        deformation_head.train()

        s2_steps = cfg["training"]["stage2_steps"]
        s2_lr = cfg["training"]["stage2_lr"]
        s2_batch_frames = cfg["training"]["stage2_batch_frames"]
        s2_supervision_cams = cfg["training"]["stage2_supervision_cams"]
        s2_weights = cfg["training"]["stage2_loss_weights"]

        optimizer_s2 = optim.AdamW(deformation_head.parameters(), lr=s2_lr)
        scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=s2_steps)

        # If restarting within stage 2, load optimizer state and advance scheduler
        start_step_s2 = 0
        if should_continue_stage2:
            print(f"  Continuing Stage 2 from step {restart_from_step}")
            start_step_s2 = restart_from_step
            if "optimizer" in ckpt:
                optimizer_s2.load_state_dict(ckpt["optimizer"])
            # Advance scheduler to correct position
            for _ in range(restart_from_step):
                scheduler_s2.step()

        torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(range(start_step_s2, s2_steps), desc="Stage 2 (Deformation)")
        for step in pbar:
            optimizer_s2.zero_grad()

            scene_batches = multi_dataset.sample_multi_scene_batch(s2_batch_frames, s2_supervision_cams)

            total_psnr = 0.0
            total_renders = 0
            total_loss = torch.tensor(0.0, device="cuda")  # Accumulate on GPU
            per_scene_losses = {}
            delta_magnitudes = []  # Track delta magnitudes for debugging

            for scene_batch in scene_batches:
                scene_name = scene_batch["scene_name"]
                scene_weight = scene_batch["scene_weight"]
                scene_cameras = scene_batch["cameras"]
    
                tokens_mean = scene_batch["tokens_mean"].cuda()
    
                with torch.no_grad():
                    canonical = canonical_head(tokens_mean)
    
                # Direct loss accumulation (no list, no stack)
                scene_loss = torch.tensor(0.0, device="cuda")
                last_rendered = None
                prev_all_deltas = None  # For TV loss between consecutive frames
    
                for i, fidx in enumerate(scene_batch["frame_indices"]):
                    tokens_t = scene_batch["tokens_frames"][i].cuda()
    
                    # Deformation: predict deltas from per-frame tokens
                    deltas = deformation_head(tokens_t)
    
                    # Track delta magnitudes for debugging
                    delta_magnitudes.append(deltas["dxyz"].abs().mean().item())
    
                    means3D, scales, rotations, opacity, shs = compose_gaussians(canonical, deltas)
    
                    for cam_name in scene_batch["cam_names"]:
                        cam = scene_cameras[cam_name]
                        gt = scene_batch["gt_images"][cam_name][i].cuda()
    
                        rendered, radii, depth, _ = render_gaussians(
                            means3D, scales, rotations, opacity, shs, cam, bg_color, sh_degree
                        )
    
                        loss = photometric_loss(rendered, gt, lambda_ssim=0.2)
                        scene_loss = scene_loss + loss  # Direct accumulation
                        total_psnr += compute_psnr(rendered, gt)
                        total_renders += 1
    
                        # Keep last rendered for saving, clean up others
                        last_rendered = rendered.detach()
                        del radii, depth, gt
    
                    # TV loss: encourage temporally smooth deformations
                    if prev_all_deltas is not None:
                        tv_weight = s2_weights.get("tv", 0.01)
                        scene_loss = scene_loss + tv_weight * tv_loss(deltas["all_deltas"], prev_all_deltas)
                    prev_all_deltas = deltas["all_deltas"].detach()
    
                # Normalize scene loss by number of renders
                num_scene_renders = len(scene_batch["frame_indices"]) * len(scene_batch["cam_names"])
                scene_loss = scene_loss / max(num_scene_renders, 1)
    
                # Store per-scene loss for logging
                per_scene_losses[scene_name] = scene_loss.item()
    
                # Apply scene weight and accumulate into total loss
                weighted_scene_loss = scene_loss * scene_weight / len(scene_batches)
                total_loss = total_loss + weighted_scene_loss
    
            # Single backward call for all scenes
            total_loss.backward()
    
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(deformation_head.parameters(), grad_clip)
            optimizer_s2.step()
            scheduler_s2.step()
    
            # Get loss for logging (single .item() call)
            avg_loss = total_loss.item()
            avg_psnr = total_psnr / max(total_renders, 1)
            avg_delta_mag = sum(delta_magnitudes) / max(len(delta_magnitudes), 1)
    
            vram_alloc, vram_peak = get_vram_gb()
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "psnr": f"{avg_psnr:.1f}",
                "Δ": f"{avg_delta_mag:.4f}",
                "vram": f"{vram_alloc:.1f}GB",
            })
    
            writer.add_scalar("stage2/loss", avg_loss, step)
            writer.add_scalar("stage2/psnr", avg_psnr, step)
            writer.add_scalar("stage2/delta_magnitude", avg_delta_mag, step)
    
            # Log per-scene losses
            for scene_name, scene_loss_val in per_scene_losses.items():
                writer.add_scalar(f"stage2/loss_{scene_name}", scene_loss_val, step)
    
            if step % cfg["logging"]["save_render_every"] == 0:
                save_render(last_rendered, os.path.join(render_dir, "stage2", f"step_{step:05d}.jpg"))
    
            if step % cfg["training"]["save_ckpt_every"] == 0 and step > 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"stage2_step{step}.pt"),
                    step, canonical_head, deformation_head, optimizer_s2
                )
    
            # Evaluation
            if step % cfg["training"]["eval_every"] == 0:
                with torch.no_grad():
                    eval_psnrs = {}
                    for scene_info in multi_dataset.scenes:
                        scene_name = scene_info["name"]
                        dataset = scene_info["dataset"]
                        cameras = scene_info["cameras"]
                        eval_cam_name = scene_info["eval_camera"]
    
                        tokens_mean_eval = dataset.get_tokens_mean().cuda()
                        canonical_eval = canonical_head(tokens_mean_eval)
                        tokens_t_eval = dataset.get_tokens_frame(0).cuda()
                        deltas_eval = deformation_head(tokens_t_eval)
                        m3d, sc, rot, op, sh = compose_gaussians(canonical_eval, deltas_eval)
    
                        eval_cam = cameras[eval_cam_name]
                        gt_novel = dataset.load_frame_image(eval_cam_name, 0).cuda()
                        rendered_novel, _, _, _ = render_gaussians(m3d, sc, rot, op, sh, eval_cam, bg_color, sh_degree)
                        novel_psnr = compute_psnr(rendered_novel, gt_novel)
                        eval_psnrs[scene_name] = novel_psnr
    
                        writer.add_scalar(f"stage2/eval_psnr_{scene_name}", novel_psnr, step)
    
                    eval_str = ", ".join([f"{name}={psnr:.1f}" for name, psnr in eval_psnrs.items()])
                    tqdm.write(f"  Step {step}: {eval_str} dB")
    
        save_checkpoint(
            os.path.join(ckpt_dir, "stage2_final.pt"),
            s2_steps, canonical_head, deformation_head, optimizer_s2
        )
        print(f"\nStage 2 complete. Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        torch.cuda.empty_cache()

    # ===== STAGE 3: Joint Fine-tuning =====
    should_skip_stage3 = (restart_from_stage and restart_from_stage > 3)
    should_continue_stage3 = (restart_from_stage == 3 and restart_from_step is not None)

    if should_skip_stage3:
        if restart_from_stage:
            print(f"\n  Skipping Stage 3 (restarting from stage {restart_from_stage})")
    elif cfg["training"].get("stage3_steps", 0) > 0:
        print("\n" + "=" * 60)
        print("STAGE 3: Joint Fine-tuning (Both Heads, Multi-Scene)")
        print("=" * 60)

        canonical_head.train()
        for p in canonical_head.parameters():
            p.requires_grad = True

        deformation_head.train()

        s3_steps = cfg["training"]["stage3_steps"]
        s3_lr = cfg["training"]["stage3_lr"]
        s3_batch_frames = cfg["training"]["stage3_batch_frames"]
        s3_supervision_cams = cfg["training"]["stage3_supervision_cams"]
        s3_weights = cfg["training"]["stage3_loss_weights"]

        optimizer_s3 = optim.AdamW(
            list(canonical_head.parameters()) + list(deformation_head.parameters()),
            lr=s3_lr
        )
        scheduler_s3 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s3, T_max=s3_steps)

        # If restarting within stage 3, load optimizer state and advance scheduler
        start_step_s3 = 0
        if should_continue_stage3:
            print(f"  Continuing Stage 3 from step {restart_from_step}")
            start_step_s3 = restart_from_step
            if "optimizer" in ckpt:
                optimizer_s3.load_state_dict(ckpt["optimizer"])
            # Advance scheduler to correct position
            for _ in range(restart_from_step):
                scheduler_s3.step()

        torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(range(start_step_s3, s3_steps), desc="Stage 3 (Joint)")
        for step in pbar:
            optimizer_s3.zero_grad()

            scene_batches = multi_dataset.sample_multi_scene_batch(s3_batch_frames, s3_supervision_cams)

            total_psnr = 0.0
            total_renders = 0
            total_loss = torch.tensor(0.0, device="cuda")  # Accumulate on GPU
            delta_magnitudes = []  # Track delta magnitudes for debugging

            for scene_batch in scene_batches:
                scene_weight = scene_batch["scene_weight"]
                scene_cameras = scene_batch["cameras"]

                tokens_mean = scene_batch["tokens_mean"].cuda()
                canonical = canonical_head(tokens_mean)

                # Direct loss accumulation (no list, no stack)
                scene_loss = torch.tensor(0.0, device="cuda")
                last_rendered = None
                prev_all_deltas = None  # For TV loss between consecutive frames

                for i, fidx in enumerate(scene_batch["frame_indices"]):
                    tokens_t = scene_batch["tokens_frames"][i].cuda()
                    deltas = deformation_head(tokens_t)

                    # Track delta magnitudes for debugging
                    delta_magnitudes.append(deltas["dxyz"].abs().mean().item())

                    means3D, scales, rotations, opacity, shs = compose_gaussians(canonical, deltas)

                    for cam_name in scene_batch["cam_names"]:
                        cam = scene_cameras[cam_name]
                        gt = scene_batch["gt_images"][cam_name][i].cuda()

                        rendered, radii, depth, _ = render_gaussians(
                            means3D, scales, rotations, opacity, shs, cam, bg_color, sh_degree
                        )

                        loss = photometric_loss(rendered, gt, lambda_ssim=0.2)
                        scene_loss = scene_loss + loss  # Direct accumulation
                        total_psnr += compute_psnr(rendered, gt)
                        total_renders += 1

                        # Keep last rendered for saving, clean up others
                        last_rendered = rendered.detach()
                        del radii, depth, gt

                    # TV loss: encourage temporally smooth deformations
                    if prev_all_deltas is not None:
                        tv_weight = s3_weights.get("tv", 0.01)
                        scene_loss = scene_loss + tv_weight * tv_loss(deltas["all_deltas"], prev_all_deltas)
                    prev_all_deltas = deltas["all_deltas"].detach()

                # Normalize scene loss by number of renders
                num_scene_renders = len(scene_batch["frame_indices"]) * len(scene_batch["cam_names"])
                scene_loss = scene_loss / max(num_scene_renders, 1)

                # Apply scene weight and accumulate into total loss
                weighted_scene_loss = scene_loss * scene_weight / len(scene_batches)
                total_loss = total_loss + weighted_scene_loss

            # Single backward call for all scenes
            total_loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(canonical_head.parameters()) + list(deformation_head.parameters()),
                    grad_clip
                )
            optimizer_s3.step()
            scheduler_s3.step()

            # Get loss for logging (single .item() call)
            avg_loss = total_loss.item()
            avg_psnr = total_psnr / max(total_renders, 1)
            avg_delta_mag = sum(delta_magnitudes) / max(len(delta_magnitudes), 1)

            vram_alloc, vram_peak = get_vram_gb()
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "psnr": f"{avg_psnr:.1f}",
                "Δ": f"{avg_delta_mag:.4f}",
                "vram": f"{vram_alloc:.1f}GB",
            })

            writer.add_scalar("stage3/loss", avg_loss, step)
            writer.add_scalar("stage3/psnr", avg_psnr, step)
            writer.add_scalar("stage3/delta_magnitude", avg_delta_mag, step)

            if step % cfg["logging"]["save_render_every"] == 0:
                save_render(last_rendered, os.path.join(render_dir, "stage3", f"step_{step:05d}.jpg"))

            if step % cfg["training"]["save_ckpt_every"] == 0 and step > 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"stage3_step{step}.pt"),
                    step, canonical_head, deformation_head, optimizer_s3
                )

        save_checkpoint(
            os.path.join(ckpt_dir, "stage3_final.pt"),
            s3_steps, canonical_head, deformation_head, optimizer_s3
        )
        print(f"\nStage 3 complete. Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete! Total time: {elapsed/3600:.1f} hours")
    print(f"{'='*60}\n")

    writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume-stage", type=int, default=1,
                        help="Resume from stage N (1=full, 2=skip stage1, 3=skip stage1+2)")
    parser.add_argument("--restart-latest", action="store_true",
                        help="Restart from latest checkpoint")
    args = parser.parse_args()

    train(args.config, resume_stage=args.resume_stage, restart_latest=args.restart_latest)
