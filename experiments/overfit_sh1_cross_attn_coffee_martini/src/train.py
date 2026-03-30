"""Training loop for cross-attention deformation head.

Loads canonical head from the original experiment's Stage 1 checkpoint,
then trains the new CrossAttentionDeformationHead which uses cross-attention
between canonical Gaussian features and per-frame STV2 tokens.
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
from model import CanonicalGaussianHead, CrossAttentionDeformationHead, compose_gaussians
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


def save_render(rendered, path):
    import cv2
    img = (rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def tv_loss(deltas_t: torch.Tensor, deltas_prev: torch.Tensor) -> torch.Tensor:
    return (deltas_t - deltas_prev).pow(2).mean()


def train(config_path: str, resume_stage: int = 2):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["paths"]["cache_dir"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    render_dir = cfg["paths"]["render_dir"]
    log_dir = cfg["paths"]["log_dir"]
    canonical_ckpt_path = cfg["paths"]["canonical_checkpoint"]

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(render_dir, "stage2"), exist_ok=True)
    os.makedirs(os.path.join(render_dir, "stage3"), exist_ok=True)
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

    # Initialize canonical head and load pretrained weights
    P = dataset.num_patches
    K = cfg["model"].get("num_gaussians_per_patch", 1)
    pts_mean = dataset.points_map.mean(dim=0)
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

    canonical_head = CanonicalGaussianHead(
        dim_in=cfg["model"]["canonical"]["dim_in"],
        dim_hidden=cfg["model"]["canonical"]["dim_hidden"],
        sh_degree=cfg["model"]["canonical"]["sh_degree"],
        init_xyz=init_xyz,
        num_gaussians_per_patch=K,
        init_xyz_per_gaussian=init_xyz_per_gaussian,
    ).cuda()

    # Load canonical checkpoint (handle sh_degree mismatch gracefully)
    print(f"  Loading canonical checkpoint: {canonical_ckpt_path}")
    ckpt = torch.load(canonical_ckpt_path, weights_only=False)
    ckpt_state = ckpt["canonical_head"]
    model_state = canonical_head.state_dict()

    # Check for SH head shape mismatch (e.g., loading sh_degree=0 into sh_degree=1)
    sh_mismatch = False
    for key in ["sh_head.weight", "sh_head.bias"]:
        if key in ckpt_state and key in model_state:
            if ckpt_state[key].shape != model_state[key].shape:
                sh_mismatch = True
                break

    if sh_mismatch:
        print(f"  [SH upgrade] Checkpoint sh_degree differs from config — loading with partial match")
        # Load everything except sh_head
        filtered = {k: v for k, v in ckpt_state.items() if "sh_head" not in k}
        canonical_head.load_state_dict(filtered, strict=False)
        # Copy DC coefficients from old sh_head into new head's DC slot
        old_bias = ckpt_state["sh_head.bias"]  # [K * old_sh_coeffs * 3]
        old_weight = ckpt_state["sh_head.weight"]  # [K * old_sh_coeffs * 3, dim_hidden]
        old_sh_coeffs = old_bias.shape[0] // (K * 3)
        new_sh_coeffs = canonical_head.num_sh_coeffs
        print(f"  [SH upgrade] Old SH coeffs: {old_sh_coeffs}, New SH coeffs: {new_sh_coeffs}")
        with torch.no_grad():
            # Zero-init new sh_head first
            canonical_head.sh_head.weight.zero_()
            canonical_head.sh_head.bias.zero_()
            # Copy DC (l=0) coefficients: for each Gaussian k and color c,
            # old layout is [k*old_sh*3 + sh*3 + c], new is [k*new_sh*3 + sh*3 + c]
            for k_idx in range(K):
                for c in range(3):
                    old_idx = k_idx * old_sh_coeffs * 3 + 0 * 3 + c  # DC is sh_idx=0
                    new_idx = k_idx * new_sh_coeffs * 3 + 0 * 3 + c
                    canonical_head.sh_head.bias[new_idx] = old_bias[old_idx]
                    canonical_head.sh_head.weight[new_idx] = old_weight[old_idx]
        print(f"  [SH upgrade] DC coefficients copied, higher-order SH initialized to zero")
    else:
        canonical_head.load_state_dict(ckpt_state)

    canonical_head.eval()
    for p in canonical_head.parameters():
        p.requires_grad = False
    print(f"  Canonical head loaded (frozen)")

    # Create cross-attention deformation head
    defo_cfg = cfg["model"]["deformation"]
    deformation_head = CrossAttentionDeformationHead(
        dim_canonical=defo_cfg["dim_canonical"],
        dim_tokens=defo_cfg["dim_tokens"],
        dim_hidden=defo_cfg["dim_hidden"],
        n_heads=defo_cfg["n_heads"],
        n_layers=defo_cfg["n_layers"],
        sh_degree=cfg["model"]["canonical"]["sh_degree"],
        num_gaussians_per_patch=K,
        max_displacement=defo_cfg.get("max_displacement", 2.0),
    ).cuda()
    print(f"  CrossAttention DeformationHead: {sum(p.numel() for p in deformation_head.parameters())/1e6:.1f}M params")
    print(f"  Max displacement: {defo_cfg.get('max_displacement', 2.0)}")
    print(f"  Total Gaussians: {P * K}")

    bg_color = torch.zeros(3, device="cuda")
    sh_degree = cfg["model"]["canonical"]["sh_degree"]
    scale_anneal_target = cfg["training"].get("scale_anneal_target", 1.0)
    grad_clip = cfg["training"].get("grad_clip_max_norm", 0.0)
    batch_frames = cfg["training"]["batch_frames"]
    supervision_cams = cfg["training"]["supervision_cams"]
    eval_cam_name = cfg["data"]["eval_camera"]

    writer = SummaryWriter(log_dir)
    start_time = time.time()

    # ===== STAGE 2: Cross-Attention Deformation Head =====
    if resume_stage <= 2:
        print("\n" + "=" * 60)
        print("STAGE 2: Training Cross-Attention Deformation Head")
        print("=" * 60)

        deformation_head.train()
        s2_cfg = cfg["training"]["stage2"]
        s2_weights = s2_cfg["loss_weights"]

        optimizer_s2 = optim.AdamW(deformation_head.parameters(), lr=s2_cfg["lr"])
        scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=s2_cfg["steps"])

        torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(range(s2_cfg["steps"]), desc="Stage 2 (CrossAttn Defo)")
        for step in pbar:
            optimizer_s2.zero_grad()

            batch = dataset.sample_training_batch(batch_frames, supervision_cams)
            tokens_mean = batch["tokens_mean"].cuda()

            with torch.no_grad():
                canonical = canonical_head(tokens_mean)

            total_loss = torch.tensor(0.0, device="cuda")
            total_psnr = 0.0
            prev_deltas_flat = None

            for i, fidx in enumerate(batch["frame_indices"]):
                tokens_t = batch["tokens_frames"][i].cuda()

                # Cross-attention: canonical features attend to frame tokens
                deltas = deformation_head(canonical["hidden"], tokens_t)

                means3D, scales, rotations, opacity, shs = compose_gaussians(
                    canonical, deltas, scale_factor=scale_anneal_target
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

                # TV loss for temporal smoothness
                deltas_flat = torch.cat([
                    deltas["dxyz"].reshape(-1),
                    deltas["dscale"].reshape(-1),
                    deltas["opacity_logit"].reshape(-1),
                ], dim=0)
                if prev_deltas_flat is not None:
                    total_loss = total_loss + s2_weights["tv"] * tv_loss(deltas_flat, prev_deltas_flat)
                prev_deltas_flat = deltas_flat.detach()

            num_renders = len(batch["frame_indices"]) * len(batch["cam_names"])
            total_loss = total_loss / max(num_renders, 1)
            avg_psnr = total_psnr / max(num_renders, 1)

            # Regularization
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

            if step % cfg["training"]["checkpointing"]["render_every"] == 0:
                save_render(rendered, os.path.join(render_dir, "stage2", f"step_{step:05d}.jpg"))

            if step % cfg["training"]["checkpointing"]["save_every"] == 0 and step > 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"stage2_step{step}.pt"),
                    step, canonical_head, deformation_head, optimizer_s2
                )

            # Novel view eval every 1000 steps
            if step % 1000 == 0:
                with torch.no_grad():
                    eval_cam = cameras[eval_cam_name]
                    frame_idx = 0
                    tokens_mean_eval = dataset.get_tokens_mean().cuda()
                    canonical_eval = canonical_head(tokens_mean_eval)
                    tokens_t_eval = dataset.get_tokens_frame(frame_idx).cuda()
                    deltas_eval = deformation_head(canonical_eval["hidden"], tokens_t_eval)
                    m3d, sc, rot, op, sh = compose_gaussians(canonical_eval, deltas_eval, scale_factor=scale_anneal_target)
                    gt_novel = dataset.load_frame_image(eval_cam_name, frame_idx).cuda()
                    rendered_novel, _, _, _ = render_gaussians(m3d, sc, rot, op, sh, eval_cam, bg_color, sh_degree)
                    novel_psnr = compute_psnr(rendered_novel, gt_novel)
                    writer.add_scalar("stage2/novel_psnr", novel_psnr, step)
                    tqdm.write(f"  Step {step}: train={avg_psnr:.2f}, novel={novel_psnr:.2f} dB")
                    save_render(rendered_novel, os.path.join(render_dir, "stage2", f"novel_{step:05d}.jpg"))

        save_checkpoint(
            os.path.join(ckpt_dir, "stage2_final.pt"),
            s2_cfg["steps"], canonical_head, deformation_head, optimizer_s2
        )
        print(f"\nStage 2 complete. Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        torch.cuda.empty_cache()

    # ===== STAGE 3: Joint Fine-tuning =====
    if "stage3" in cfg["training"]:
        print("\n" + "=" * 60)
        print("STAGE 3: Joint Fine-tuning (Both Heads)")
        print("=" * 60)

        os.makedirs(os.path.join(render_dir, "stage3"), exist_ok=True)

        if resume_stage >= 3:
            ckpt_path = os.path.join(ckpt_dir, "stage2_final.pt")
            print(f"  Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, weights_only=False)
            canonical_head.load_state_dict(ckpt["canonical_head"])
            deformation_head.load_state_dict(ckpt["deformation_head"])

        canonical_head.train()
        deformation_head.train()
        for p in canonical_head.parameters():
            p.requires_grad = True

        s3_cfg = cfg["training"]["stage3"]
        s3_weights = s3_cfg["loss_weights"]

        optimizer_s3 = optim.AdamW(
            list(canonical_head.parameters()) + list(deformation_head.parameters()),
            lr=s3_cfg["lr"],
        )
        scheduler_s3 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s3, T_max=s3_cfg["steps"])

        torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(range(s3_cfg["steps"]), desc="Stage 3 (Joint)")
        for step in pbar:
            optimizer_s3.zero_grad()

            batch = dataset.sample_training_batch(batch_frames, supervision_cams)
            tokens_mean = batch["tokens_mean"].cuda()
            canonical = canonical_head(tokens_mean)

            total_loss = torch.tensor(0.0, device="cuda")
            total_psnr = 0.0
            prev_deltas_flat = None

            for i, fidx in enumerate(batch["frame_indices"]):
                tokens_t = batch["tokens_frames"][i].cuda()
                deltas = deformation_head(canonical["hidden"], tokens_t)
                means3D, scales, rotations, opacity, shs = compose_gaussians(
                    canonical, deltas, scale_factor=scale_anneal_target
                )

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

                deltas_flat = torch.cat([
                    deltas["dxyz"].reshape(-1),
                    deltas["dscale"].reshape(-1),
                    deltas["opacity_logit"].reshape(-1),
                ], dim=0)
                if prev_deltas_flat is not None:
                    total_loss = total_loss + s3_weights["tv"] * tv_loss(deltas_flat, prev_deltas_flat)
                prev_deltas_flat = deltas_flat.detach()

            num_renders = len(batch["frame_indices"]) * len(batch["cam_names"])
            total_loss = total_loss / max(num_renders, 1)
            avg_psnr = total_psnr / max(num_renders, 1)

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

            vram_alloc, _ = get_vram_gb()
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

            if step % 1000 == 0:
                with torch.no_grad():
                    eval_cam = cameras[eval_cam_name]
                    frame_idx = 0
                    tokens_mean_eval = dataset.get_tokens_mean().cuda()
                    canonical_eval = canonical_head(tokens_mean_eval)
                    tokens_t_eval = dataset.get_tokens_frame(frame_idx).cuda()
                    deltas_eval = deformation_head(canonical_eval["hidden"], tokens_t_eval)
                    m3d, sc, rot, op, sh = compose_gaussians(canonical_eval, deltas_eval, scale_factor=scale_anneal_target)
                    gt_novel = dataset.load_frame_image(eval_cam_name, frame_idx).cuda()
                    rendered_novel, _, _, _ = render_gaussians(m3d, sc, rot, op, sh, eval_cam, bg_color, sh_degree)
                    novel_psnr = compute_psnr(rendered_novel, gt_novel)
                    writer.add_scalar("stage3/novel_psnr", novel_psnr, step)
                    tqdm.write(f"  Step {step}: train={avg_psnr:.2f}, novel={novel_psnr:.2f} dB")
                    save_render(rendered_novel, os.path.join(render_dir, "stage3", f"novel_{step:05d}.jpg"))

        save_checkpoint(
            os.path.join(ckpt_dir, "stage3_final.pt"),
            s3_cfg["steps"], canonical_head, deformation_head, optimizer_s3
        )
        print(f"\nStage 3 complete. Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    elapsed = time.time() - start_time
    writer.close()
    print(f"\nTotal training time: {elapsed/3600:.1f}h ({elapsed:.0f}s)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="D:/DecodeGaussians/experiments/overfit_cross_attn_coffee_martini/config.yaml")
    parser.add_argument("--resume-stage", type=int, default=2,
                        help="Resume from stage N (2=deformation, 3=joint only)")
    args = parser.parse_args()
    train(args.config, resume_stage=args.resume_stage)
