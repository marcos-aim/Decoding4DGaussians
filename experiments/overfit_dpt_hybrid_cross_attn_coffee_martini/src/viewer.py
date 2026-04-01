"""Interactive 3D Gaussian viewer for H2 hybrid DPT model using viser.

Loads the trained hybrid DPT+MLP decoder model and renders Gaussians in real-time.
Navigate with mouse, scrub through frames with a slider.
Open http://localhost:8080 in your browser.
"""

import math
import os
import sys
import glob
import yaml
import numpy as np
import torch
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from cameras import build_cameras_from_llff
from dataset import CachedSceneDataset
from model import HybridDPTCanonicalGaussianHead, CrossAttentionDeformationHead, compose_gaussians

import viser


def load_model(config_path: str, checkpoint_path: str = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["paths"]["cache_dir"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    scene_dir = cfg["paths"]["neu3d_scene"]
    target_w, target_h = cfg["data"]["resolution"]

    cam_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(scene_dir, "cam*.mp4"))])
    cameras = build_cameras_from_llff(
        os.path.join(scene_dir, "poses_bounds.npy"), target_w, target_h, cam_files
    )

    dataset = CachedSceneDataset(cache_dir, cameras)

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

    REF_W = 512
    res_scale = REF_W / W
    init_log_scale = math.log(0.5 * res_scale)
    spread = 0.05 * res_scale

    can_cfg = cfg["model"]["canonical"]
    dpt_dim = can_cfg.get("dpt_dim", 512)
    cfg_grid_h = can_cfg.get("grid_h", grid_h)
    cfg_grid_w = can_cfg.get("grid_w", grid_w)

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

    if checkpoint_path is None:
        stage3_path = os.path.join(ckpt_dir, "stage3_final.pt")
        checkpoint_path = stage3_path if os.path.exists(stage3_path) else os.path.join(ckpt_dir, "stage2_final.pt")

    ckpt = torch.load(checkpoint_path, weights_only=False)
    canonical_head.load_state_dict(ckpt["canonical_head"])
    deformation_head.load_state_dict(ckpt["deformation_head"])
    canonical_head.eval()
    deformation_head.eval()

    scale_anneal_target = cfg["training"].get("scale_anneal_target", 1.0)

    return canonical_head, deformation_head, dataset, cfg, scale_anneal_target


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="D:/DecodeGaussians/experiments/overfit_dpt_hybrid_cross_attn_coffee_martini/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print("Loading H2 hybrid DPT model...")
    canonical_head, deformation_head, dataset, cfg, scale_anneal_target = load_model(
        args.config, args.checkpoint
    )
    num_frames = dataset.num_frames
    print(f"Model loaded. {num_frames} frames available.")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"\nOpen http://localhost:{args.port} in your browser!")

    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=num_frames - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)

    print("Precomputing Gaussians for all frames...")
    frame_cache = {}
    with torch.no_grad():
        tokens_mean = dataset.get_tokens_mean().cuda()
        canonical = canonical_head(tokens_mean)
        canonical_hidden_cpu = canonical["hidden"].cpu()
        for fi in range(num_frames):
            tokens_t = dataset.get_tokens_frame(fi).cuda()
            canonical["hidden"] = canonical_hidden_cpu.cuda()
            deltas = deformation_head(canonical["hidden"], tokens_t)
            means3D, scales, rotations, opacity, shs = compose_gaussians(
                canonical, deltas, scale_factor=scale_anneal_target
            )
            frame_cache[fi] = {
                "positions": means3D.cpu().numpy().astype(np.float32),
                "scales": scales.cpu().numpy().astype(np.float32),
                "quats": rotations.cpu().numpy().astype(np.float32),
                "opacities": opacity.squeeze(-1).cpu().numpy().astype(np.float32),
                "colors_sh": shs[:, 0, :].cpu().numpy().astype(np.float32),
            }
            del tokens_t, deltas, means3D, scales, rotations, opacity, shs
            canonical["hidden"] = canonical_hidden_cpu
            torch.cuda.empty_cache()
            if fi % 50 == 0:
                print(f"  Frame {fi}/{num_frames}")
    print(f"Precomputed {num_frames} frames.")

    print("Precomputing covariance matrices...")
    SH_C0 = 0.28209479177387814
    for fi in range(num_frames):
        data = frame_cache[fi]
        data["covariances"] = compute_covariances(data["scales"], data["quats"])
        data["rgbs"] = (np.clip(data["colors_sh"] * SH_C0 + 0.5, 0, 1) * 255).astype(np.uint8)
        if fi % 50 == 0:
            print(f"  Covariances {fi}/{num_frames}")
    print("Covariances ready.")

    init_data = frame_cache[0]
    splat_handle = server.scene.add_gaussian_splats(
        "/gaussians",
        centers=init_data["positions"],
        covariances=init_data["covariances"],
        rgbs=init_data["rgbs"],
        opacities=init_data["opacities"].reshape(-1, 1),
    )
    current_frame = 0

    def update_splats(frame_idx, force=False):
        nonlocal current_frame
        if frame_idx == current_frame and not force:
            return
        data = frame_cache[frame_idx]
        splat_handle.centers = data["positions"]
        splat_handle.covariances = data["covariances"]
        splat_handle.rgbs = data["rgbs"]
        splat_handle.opacities = data["opacities"].reshape(-1, 1)
        current_frame = frame_idx

    @frame_slider.on_update
    def _on_frame_change(event):
        update_splats(int(frame_slider.value))

    update_splats(0)
    print("Ready! Navigate in your browser.")

    while True:
        if playing.value:
            frame = int(frame_slider.value)
            frame = (frame + 1) % num_frames
            frame_slider.value = frame
            update_splats(frame)
            time.sleep(1.0 / fps_slider.value)
        else:
            time.sleep(0.05)


def compute_covariances(scales, quats):
    N = scales.shape[0]
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    R = np.zeros((N, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)

    S = np.zeros((N, 3, 3), dtype=np.float32)
    S[:, 0, 0] = scales[:, 0] ** 2
    S[:, 1, 1] = scales[:, 1] ** 2
    S[:, 2, 2] = scales[:, 2] ** 2

    cov = np.einsum('nij,njk,nlk->nil', R, S, R)
    return cov


if __name__ == "__main__":
    main()
