"""Training loop for Aria monocular overfit.

Structural differences vs. dpt_hybrid_s3lr/train.py:
  - Single supervision camera per timestamp (Aria is monocular).
  - No eval camera held out across space — eval split is across TIME.
  - Dataset is AriaCachedDataset, cameras built from Aria poses.
"""

import argparse
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
from losses import (compute_psnr, compute_rgb_loss, compute_ssim_loss,
                    compute_geo_loss, compute_tv_loss,
                    compute_scale_reg, compute_opacity_reg)


def build_input_indices(num_frames: int, input_subsample: int) -> list:
    """Indices into the 128-frame window that are VISIBLE to the model.
    4DGT default: every 8th frame → 16 input timestamps out of 128."""
    return list(range(0, num_frames, input_subsample))


def select_supervision_indices(num_frames: int, step: int) -> list:
    """At each training step, pick one supervised frame index (or a few).
    Monocular: pick ONE frame per step, randomly."""
    rng = np.random.default_rng(step)
    return [int(rng.integers(0, num_frames))]


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

        rgb_l = compute_rgb_loss(rendered, gt)
        ssim_l = compute_ssim_loss(rendered, gt)
        loss = lw.get("rgb", 0.0) * rgb_l + lw.get("ssim", 0.0) * ssim_l
        if lw.get("geo", 0.0) > 0:
            loss = loss + lw["geo"] * compute_geo_loss(means3D, dataset.get_points_map_frame(t_idx))
        if lw.get("tv", 0.0) > 0 and stage_name != "stage1":
            tv_l = compute_tv_loss(means3D)
            loss = loss + lw["tv"] * tv_l
        if lw.get("scale_reg", 0.0) > 0:
            loss = loss + lw["scale_reg"] * compute_scale_reg(scales)
        if lw.get("opacity_reg", 0.0) > 0:
            loss = loss + lw["opacity_reg"] * compute_opacity_reg(opacity)

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

    pts_mean = dataset.points_map.mean(dim=0)
    H, W = pts_mean.shape[0], pts_mean.shape[1]
    pts_flat = pts_mean.reshape(-1, 3)
    K = cfg["model"]["num_gaussians_per_patch"]

    canonical_head = HybridDPTCanonicalGaussianHead(
        num_patches=dataset.num_patches,
        num_gaussians_per_patch=K,
        init_xyz=pts_flat,
        **cfg["model"]["canonical"],
    ).cuda()

    deformation_head = CrossAttentionDeformationHead(
        num_patches=dataset.num_patches,
        num_gaussians_per_patch=K,
        **cfg["model"]["deformation"],
    ).cuda()

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
