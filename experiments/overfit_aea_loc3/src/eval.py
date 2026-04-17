"""Evaluate on 4DGT-style novel-time split.

4DGT paper §5.2:
  - 128-frame window W sampled every 8th → 16 input timestamps (N=16).
  - Supervise/render all 128 frames.
  - Metrics reported on the 112 held-out timestamps.

We also run the 64/64 split as a secondary protocol:
  - Every-other frame is input (64), every-other frame is held out (64).
  - This matches the 4DGT ablation table for fair 1:1 frame comparison.

Outputs:
  - metrics.json                 per-frame PSNR/SSIM/LPIPS
  - summary.json                 mean/min/max aggregates
  - renders/{input,heldout}/*.jpg
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lpips
from pytorch_msssim import ssim as compute_ssim

from aria_cached_dataset import AriaCachedDataset
from aria_cameras import build_cameras_from_aria
from aria_dataset import AriaSequenceDataset
from model import HybridDPTCanonicalGaussianHead, CrossAttentionDeformationHead, compose_gaussians
from renderer import render_gaussians
from losses import compute_psnr


def build_splits(num_frames: int, protocol: str):
    """Return (input_indices, heldout_indices)."""
    if protocol == "4dgt_16_112":
        input_idx = list(range(0, num_frames, 8))
        heldout_idx = [i for i in range(num_frames) if i not in set(input_idx)]
    elif protocol == "64_64":
        input_idx = list(range(0, num_frames, 2))
        heldout_idx = list(range(1, num_frames, 2))
    else:
        raise ValueError(protocol)
    return input_idx, heldout_idx


def evaluate(cfg_path: str, protocol: str, checkpoint_name: str = "stage3_final.pt"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    target_w, target_h = cfg["data"]["resolution"]
    aria_seq = AriaSequenceDataset(
        cfg["paths"]["aria_seq_root"],
        target_resolution=(target_w, target_h),
        frame_indices=list(range(cfg["data"]["start_frame"],
                                 cfg["data"]["start_frame"] + cfg["data"]["num_frames"])),
    )
    dataset = AriaCachedDataset(
        cfg["paths"]["cache_dir"], cfg["paths"]["aria_seq_root"],
        target_resolution=(target_w, target_h),
    )

    pts_mean = dataset.points_map.mean(dim=0)
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

    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], checkpoint_name)
    ckpt = torch.load(ckpt_path, weights_only=False)
    canonical_head.load_state_dict(ckpt["canonical_head"])
    deformation_head.load_state_dict(ckpt["deformation_head"])
    canonical_head.eval()
    deformation_head.eval()

    input_idx, heldout_idx = build_splits(dataset.num_frames, protocol)
    cams = build_cameras_from_aria(aria_seq, list(range(dataset.num_frames)))
    bg = torch.zeros(3, device="cuda")
    sh = cfg["model"]["canonical"]["sh_degree"]
    scale_target = cfg["training"]["scale_anneal_target"]

    render_root = os.path.join(cfg["paths"]["render_dir"], protocol)
    os.makedirs(os.path.join(render_root, "input"), exist_ok=True)
    os.makedirs(os.path.join(render_root, "heldout"), exist_ok=True)

    lpips_fn = lpips.LPIPS(net="alex").cuda().eval()

    per_frame = {}
    with torch.no_grad():
        tokens_mean = dataset.get_tokens_mean()
        canonical = canonical_head(tokens_mean)

        for split_name, indices in [("input", input_idx), ("heldout", heldout_idx)]:
            for i in tqdm(indices, desc=f"render {split_name}"):
                tokens_t = dataset.get_tokens_frame(i)
                deltas = deformation_head(canonical["hidden"], tokens_t)
                means3D, scales, rotations, opacity, shs = compose_gaussians(
                    canonical, deltas, scale_factor=scale_target
                )
                rendered, _, _, _ = render_gaussians(
                    means3D, scales, rotations, opacity, shs, cams[i], bg, sh
                )
                gt = dataset.load_frame_image(i)

                psnr = float(compute_psnr(rendered, gt))
                ssim_v = float(compute_ssim(rendered.unsqueeze(0), gt.unsqueeze(0),
                                             data_range=1.0, size_average=True).item())
                # LPIPS expects [-1, 1]
                lp = float(lpips_fn((rendered * 2 - 1).unsqueeze(0),
                                     (gt * 2 - 1).unsqueeze(0)).item())

                per_frame[i] = {"split": split_name, "psnr": psnr, "ssim": ssim_v, "lpips": lp}

                img_np = (rendered.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(render_root, split_name, f"{i:04d}.jpg"),
                            cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # Aggregate
    def agg(name):
        xs = [v[name] for v in per_frame.values() if v["split"] == "heldout"]
        return {"mean": float(np.mean(xs)), "min": float(np.min(xs)), "max": float(np.max(xs)),
                "n": len(xs)}

    summary = {
        "protocol": protocol,
        "checkpoint": checkpoint_name,
        "num_input": len(input_idx),
        "num_heldout": len(heldout_idx),
        "heldout": {"psnr": agg("psnr"), "ssim": agg("ssim"), "lpips": agg("lpips")},
    }
    with open(os.path.join(render_root, "metrics.json"), "w") as f:
        json.dump(per_frame, f, indent=2)
    with open(os.path.join(render_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--protocol", choices=["4dgt_16_112", "64_64"], default="4dgt_16_112")
    parser.add_argument("--checkpoint", default="stage3_final.pt")
    args = parser.parse_args()
    evaluate(args.config, args.protocol, args.checkpoint)
