"""Evaluate trained model: render novel views, compute metrics, export PLY."""

import os
import sys
import glob
import yaml
import struct
import torch
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from cameras import build_cameras_from_llff
from dataset import CachedSceneDataset
from model import CanonicalGaussianHead, DeformationHead, compose_gaussians
from renderer import render_gaussians
from losses import compute_psnr
from pytorch_msssim import ssim as compute_ssim


def write_ply(path: str, points: np.ndarray, colors: np.ndarray = None):
    """
    Write points to a PLY file.
    points: (N, 3) float32
    colors: (N, 3) uint8, optional
    """
    N = points.shape[0]
    header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
"""
    if colors is not None:
        header += """property uchar red
property uchar green
property uchar blue
"""
    header += "end_header\n"

    with open(path, "wb") as f:
        f.write(header.encode())
        for i in range(N):
            f.write(struct.pack("<fff", *points[i]))
            if colors is not None:
                f.write(struct.pack("<BBB", *colors[i]))


def evaluate(config_path: str, checkpoint_path: str = None):
    """Render all frames from eval camera and compute metrics."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["paths"]["cache_dir"]
    render_dir = cfg["paths"]["render_dir"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    target_w, target_h = cfg["data"]["resolution"]

    scene_dir = cfg["paths"]["neu3d_scene"]
    cam_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(scene_dir, "cam*.mp4"))])
    cameras = build_cameras_from_llff(
        os.path.join(scene_dir, "poses_bounds.npy"), target_w, target_h, cam_files
    )

    dataset = CachedSceneDataset(cache_dir, cameras)

    # Reconstruct init_xyz to match training
    P = dataset.num_patches
    pts_mean = dataset.points_map.mean(dim=0).reshape(-1, 3)
    indices = torch.linspace(0, pts_mean.shape[0] - 1, P).long()
    init_xyz = pts_mean[indices]

    K = cfg["model"].get("num_gaussians_per_patch", 1)

    canonical_head = CanonicalGaussianHead(
        dim_in=cfg["model"]["canonical"]["dim_in"],
        dim_hidden=cfg["model"]["canonical"]["dim_hidden"],
        sh_degree=cfg["model"]["canonical"]["sh_degree"],
        init_xyz=init_xyz,
        num_gaussians_per_patch=K,
    ).cuda()

    deformation_head = DeformationHead(
        dim_in=cfg["model"]["deformation"]["dim_in"],
        dim_hidden=cfg["model"]["deformation"]["dim_hidden"],
        n_attn_heads=cfg["model"]["deformation"]["n_attn_heads"],
        n_attn_layers=cfg["model"]["deformation"]["n_attn_layers"],
        sh_degree=cfg["model"]["canonical"]["sh_degree"],
        num_gaussians_per_patch=K,
    ).cuda()

    if checkpoint_path is None:
        stage3_path = os.path.join(ckpt_dir, "stage3_final.pt")
        checkpoint_path = stage3_path if os.path.exists(stage3_path) else os.path.join(ckpt_dir, "stage2_final.pt")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    canonical_head.load_state_dict(ckpt["canonical_head"])
    deformation_head.load_state_dict(ckpt["deformation_head"])
    canonical_head.eval()
    deformation_head.eval()

    bg_color = torch.zeros(3, device="cuda")
    sh_degree = cfg["model"]["canonical"]["sh_degree"]
    scale_anneal_target = cfg["training"].get("scale_anneal_target", 1.0)
    eval_cam_name = cfg["data"]["eval_camera"]
    eval_cam = cameras[eval_cam_name]

    eval_dir = os.path.join(render_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    psnrs = []
    ssims = []
    rendered_frames = []

    with torch.no_grad():
        tokens_mean = dataset.get_tokens_mean().cuda()
        canonical = canonical_head(tokens_mean)

        for frame_idx in tqdm(range(dataset.num_frames), desc=f"Rendering {eval_cam_name}"):
            tokens_t = dataset.get_tokens_frame(frame_idx).cuda()
            deltas = deformation_head(tokens_t)
            means3D, scales, rotations, opacity, shs = compose_gaussians(canonical, deltas, scale_factor=scale_anneal_target)

            rendered, _, _, _ = render_gaussians(
                means3D, scales, rotations, opacity, shs, eval_cam, bg_color, sh_degree
            )

            img_np = (rendered.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(eval_dir, f"{frame_idx:06d}.jpg"), img_bgr)
            rendered_frames.append(img_bgr)

            gt = dataset.load_frame_image(eval_cam_name, frame_idx).cuda()
            psnrs.append(compute_psnr(rendered, gt))
            ssim_val = compute_ssim(
                rendered.unsqueeze(0), gt.unsqueeze(0), data_range=1.0, size_average=True
            ).item()
            ssims.append(ssim_val)

    video_path = os.path.join(render_dir, "novel_view.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(video_path, fourcc, 30, (target_w, target_h))
    for frame in rendered_frames:
        out_video.write(frame)
    out_video.release()

    with torch.no_grad():
        tokens_mean = dataset.get_tokens_mean().cuda()
        canonical = canonical_head(tokens_mean)
        means3D, scales, _, opacity, _ = compose_gaussians(canonical, scale_factor=scale_anneal_target)
        pts = means3D.cpu().numpy()
        op = opacity.squeeze(-1).cpu().numpy()
        colors = (np.stack([op, op, op], axis=-1) * 255).astype(np.uint8)
        write_ply(os.path.join(render_dir, "gaussians.ply"), pts, colors)

    mean_psnr = np.mean(psnrs)
    min_psnr = np.min(psnrs)
    mean_ssim = np.mean(ssims)

    print("\n" + "=" * 60)
    print(f"=== Overfit Results: {cfg['experiment']['scene']} ===")
    print(f"Experiment: {cfg['experiment']['mode']}")
    print(f"Eval camera: {eval_cam_name}")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")
    print(f"Min  PSNR: {min_psnr:.2f} dB")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Video saved: {video_path}")
    print(f"PLY saved: {os.path.join(render_dir, 'gaussians.ply')}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="D:/DecodeGaussians/experiments/overfit_coffee_martini/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
