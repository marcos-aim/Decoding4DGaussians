"""Training loop for Aria monocular overfit.

Structural differences vs. dpt_hybrid_s3lr/train.py:
  - Single supervision camera per timestamp (Aria is monocular).
  - No eval camera held out across space — eval split is across TIME.
  - Dataset is AriaCachedDataset, cameras built from Aria poses.
"""

import argparse
import math
import os
import sys
import time
import yaml

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from aria_cached_dataset import AriaCachedDataset
from aria_cameras import build_cameras_from_aria, AriaMiniCam
from aria_dataset import AriaSequenceDataset
from model import HybridDPTCanonicalGaussianHead, CrossAttentionDeformationHead, compose_gaussians
from renderer import render_gaussians
from losses import (compute_psnr, photometric_loss, geometric_loss,
                    scale_regularization, opacity_regularization)


def select_supervision_indices(num_frames: int, step: int) -> list:
    """Monocular: pick ONE frame per step, randomly."""
    rng = np.random.default_rng(step)
    return [int(rng.integers(0, num_frames))]


def build_heads(cfg, dataset):
    """Construct canonical + deformation heads using the s3lr-reference init logic.

    Computes grid_h/grid_w from the actual P in the cached tokens, derives
    init_xyz_per_gaussian by mapping each pixel to a patch cell, and applies
    resolution-aware init_log_scale / spread.
    """
    P = dataset.num_patches
    K = cfg["model"].get("num_gaussians_per_patch", 128)
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

    pts_flat_cpu = pts_flat.detach().cpu()
    init_xyz_per_gaussian = torch.zeros(P, K, 3)
    for p in range(P):
        mask = (pixel_to_patch == p)
        pts_p = pts_flat_cpu[mask]
        if pts_p.shape[0] == 0:
            init_xyz_per_gaussian[p] = pts_flat_cpu.mean(dim=0).unsqueeze(0).expand(K, -1)
        elif pts_p.shape[0] >= K:
            idx = torch.linspace(0, pts_p.shape[0] - 1, K).long()
            init_xyz_per_gaussian[p] = pts_p[idx]
        else:
            repeats = K // pts_p.shape[0] + 1
            init_xyz_per_gaussian[p] = pts_p.repeat(repeats, 1)[:K]
    init_xyz_per_gaussian = init_xyz_per_gaussian.reshape(P * K, 3)

    indices = torch.linspace(0, pts_flat_cpu.shape[0] - 1, P).long()
    init_xyz = pts_flat_cpu[indices]

    REF_W = 512
    res_scale = REF_W / W
    init_log_scale = math.log(0.5 * res_scale)
    spread = 0.05 * res_scale

    can_cfg = cfg["model"]["canonical"]
    dpt_dim = can_cfg.get("dpt_dim", 512)
    cfg_grid_h = can_cfg.get("grid_h", grid_h)
    cfg_grid_w = can_cfg.get("grid_w", grid_w)
    if cfg_grid_h * cfg_grid_w < P:
        print(f"[build_heads] WARNING: config grid {cfg_grid_h}x{cfg_grid_w}={cfg_grid_h*cfg_grid_w} "
              f"< P={P}; falling back to computed grid {grid_h}x{grid_w}")
        cfg_grid_h, cfg_grid_w = grid_h, grid_w

    print(f"[build_heads] P={P} K={K} grid={cfg_grid_h}x{cfg_grid_w} "
          f"res_scale={res_scale:.3f} init_log_scale={init_log_scale:.3f} spread={spread:.4f}")

    canonical_head = HybridDPTCanonicalGaussianHead(
        dim_in=can_cfg["dim_in"],
        dim_hidden=can_cfg["dim_hidden"],
        grid_h=cfg_grid_h,
        grid_w=cfg_grid_w,
        sh_degree=can_cfg["sh_degree"],
        num_gaussians_per_patch=K,
        init_xyz=init_xyz,
        init_xyz_per_gaussian=init_xyz_per_gaussian,
        init_log_scale=init_log_scale,
        spread=spread,
        dpt_dim=dpt_dim,
    ).cuda()

    defo_cfg = cfg["model"]["deformation"]
    deformation_head = CrossAttentionDeformationHead(
        dim_canonical=defo_cfg["dim_canonical"],
        dim_tokens=defo_cfg["dim_tokens"],
        dim_hidden=defo_cfg["dim_hidden"],
        n_heads=defo_cfg["n_heads"],
        n_layers=defo_cfg["n_layers"],
        sh_degree=can_cfg["sh_degree"],
        num_gaussians_per_patch=K,
        max_displacement=defo_cfg.get("max_displacement", 2.0),
    ).cuda()

    return canonical_head, deformation_head


def train_one_stage(cfg, stage_name, canonical_head, deformation_head,
                    dataset, aria_seq, writer, global_step_start):
    stage_cfg = cfg["training"][stage_name]
    n_steps = stage_cfg["steps"]
    lr = float(stage_cfg["lr"])
    lw = stage_cfg["loss_weights"]

    if stage_name == "stage1":
        params = list(canonical_head.parameters())
    elif stage_name == "stage2":
        params = list(deformation_head.parameters())
        for p in canonical_head.parameters():
            p.requires_grad = False
    else:
        params = list(canonical_head.parameters()) + list(deformation_head.parameters())
        for p in canonical_head.parameters():
            p.requires_grad = True

    opt = optim.AdamW([p for p in params if p.requires_grad], lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.01)

    bg = torch.zeros(3, device="cuda")
    sh_deg = cfg["model"]["canonical"]["sh_degree"]
    scale_target = cfg["training"].get("scale_anneal_target", 1.0)

    # Pre-build one MiniCam per frame (monocular — one pose per timestamp)
    cams_per_frame = build_cameras_from_aria(
        aria_seq, list(range(dataset.num_frames))
    )

    pbar = tqdm(range(n_steps), desc=stage_name)
    for step in pbar:
        global_step = global_step_start + step
        sup_indices = select_supervision_indices(dataset.num_frames, global_step)
        t_idx = sup_indices[0]
        cam = cams_per_frame[t_idx]

        tokens_mean = dataset.get_tokens_mean()
        tokens_t = dataset.get_tokens_frame(t_idx)

        canonical = canonical_head(tokens_mean)

        if stage_name == "stage1":
            means3D, scales, rotations, opacity, shs = compose_gaussians(
                canonical, scale_factor=scale_target
            )
        else:
            deltas = deformation_head(canonical["hidden"], tokens_t)
            means3D, scales, rotations, opacity, shs = compose_gaussians(
                canonical, deltas, scale_factor=scale_target
            )

        rendered, _, _, _ = render_gaussians(
            means3D, scales, rotations, opacity, shs, cam, bg, sh_deg
        )
        gt = dataset.load_frame_image(t_idx)

        # photometric_loss combines L1 + SSIM internally (lambda_ssim weights SSIM portion).
        # Match reference s3lr: scale by lw["rgb"], pass lw["ssim"] as lambda_ssim.
        loss = lw.get("rgb", 1.0) * photometric_loss(
            rendered, gt, lambda_ssim=lw.get("ssim", 0.85)
        )
        if lw.get("geo", 0.0) > 0:
            loss = loss + lw["geo"] * geometric_loss(means3D, dataset.get_points_map_frame(t_idx))
        # TV loss skipped: monocular loop samples ONE random frame per step, so there is
        # no meaningful consecutive-frame pair. Reference s3lr tv_loss takes (deltas_t, deltas_t_prev).
        if lw.get("scale_reg", 0.0) > 0:
            loss = loss + lw["scale_reg"] * scale_regularization(scales)
        if lw.get("opacity_reg", 0.0) > 0:
            loss = loss + lw["opacity_reg"] * opacity_regularization(opacity)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in params if p.requires_grad],
            cfg["training"]["grad_clip_max_norm"]
        )
        opt.step()
        sched.step()

        with torch.no_grad():
            psnr = compute_psnr(rendered, gt)

        pbar.set_postfix(loss=f"{loss.item():.3f}", psnr=f"{psnr:.2f}")
        writer.add_scalar(f"{stage_name}/loss_total", loss.item(), global_step)
        writer.add_scalar(f"{stage_name}/psnr", psnr, global_step)
        writer.add_scalar(f"{stage_name}/lr", sched.get_last_lr()[0], global_step)

        if (step + 1) % cfg["training"]["checkpointing"]["save_every"] == 0:
            ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"],
                                      f"{stage_name}_step{step+1}.pt")
            torch.save({"canonical_head": canonical_head.state_dict(),
                        "deformation_head": deformation_head.state_dict()}, ckpt_path)

    final_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"{stage_name}_final.pt")
    torch.save({"canonical_head": canonical_head.state_dict(),
                "deformation_head": deformation_head.state_dict()}, final_path)
    return global_step_start + n_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume_stage", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)

    aria_seq = AriaSequenceDataset(
        cfg["paths"]["aria_seq_root"],
        target_resolution=tuple(cfg["data"]["resolution"]),
        frame_indices=list(range(cfg["data"]["start_frame"],
                                 cfg["data"]["start_frame"] + cfg["data"]["num_frames"])),
    )
    dataset = AriaCachedDataset(
        cfg["paths"]["cache_dir"], cfg["paths"]["aria_seq_root"],
        target_resolution=tuple(cfg["data"]["resolution"]),
    )

    canonical_head, deformation_head = build_heads(cfg, dataset)

    writer = SummaryWriter(cfg["paths"]["log_dir"])
    g = 0
    if args.resume_stage <= 1:
        g = train_one_stage(cfg, "stage1", canonical_head, deformation_head,
                             dataset, aria_seq, writer, g)
    else:
        ckpt = torch.load(os.path.join(cfg["paths"]["checkpoint_dir"], "stage1_final.pt"),
                          weights_only=False)
        canonical_head.load_state_dict(ckpt["canonical_head"])
        g = cfg["training"]["stage1"]["steps"]

    if args.resume_stage <= 2:
        g = train_one_stage(cfg, "stage2", canonical_head, deformation_head,
                             dataset, aria_seq, writer, g)
    else:
        ckpt = torch.load(os.path.join(cfg["paths"]["checkpoint_dir"], "stage2_final.pt"),
                          weights_only=False)
        canonical_head.load_state_dict(ckpt["canonical_head"])
        deformation_head.load_state_dict(ckpt["deformation_head"])
        g += cfg["training"]["stage2"]["steps"]

    g = train_one_stage(cfg, "stage3", canonical_head, deformation_head,
                         dataset, aria_seq, writer, g)

    writer.close()
    print("=== training complete ===")


if __name__ == "__main__":
    main()
