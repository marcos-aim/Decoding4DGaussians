"""Interactive 3D Gaussian viewer using viser.

Built-in controls (no setup needed):
  W/S       — move forward/back
  A/D       — strafe left/right
  Q/E       — move down/up
  Mouse drag — orbit / change view angle
  Scroll    — zoom

Open http://localhost:8080 in your browser.
"""

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
from model import CanonicalGaussianHead, CrossAttentionDeformationHead, compose_gaussians

import viser


def load_model(config_path, checkpoint_path=None):
    with open(config_path) as f:
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
    K = cfg["model"].get("num_gaussians_per_patch", 1)
    pts_mean = dataset.points_map.mean(dim=0)
    H, W = pts_mean.shape[0], pts_mean.shape[1]
    pts_flat = pts_mean.reshape(-1, 3)

    grid_w = round((P * W / H) ** 0.5)
    grid_h = round(P / grid_w)
    while grid_w * grid_h < P:
        grid_w += 1

    patch_h, patch_w = H / grid_h, W / grid_w
    pixel_y, pixel_x = torch.arange(H).float(), torch.arange(W).float()
    yy, xx = torch.meshgrid(pixel_y, pixel_x, indexing="ij")
    pixel_to_patch = ((yy / patch_h).long().clamp(0, grid_h-1) * grid_w
                      + (xx / patch_w).long().clamp(0, grid_w-1)).reshape(-1)

    init_xyz_per_gaussian = torch.zeros(P, K, 3)
    for p in range(P):
        pts_p = pts_flat[pixel_to_patch == p]
        if pts_p.shape[0] == 0:
            init_xyz_per_gaussian[p] = pts_flat.mean(0).unsqueeze(0).expand(K, -1)
        elif pts_p.shape[0] >= K:
            init_xyz_per_gaussian[p] = pts_p[torch.linspace(0, pts_p.shape[0]-1, K).long()]
        else:
            init_xyz_per_gaussian[p] = pts_p.repeat(K // pts_p.shape[0] + 1, 1)[:K]
    init_xyz_per_gaussian = init_xyz_per_gaussian.reshape(P * K, 3)
    init_xyz = pts_flat[torch.linspace(0, pts_flat.shape[0]-1, P).long()]

    canonical_head = CanonicalGaussianHead(
        dim_in=cfg["model"]["canonical"]["dim_in"],
        dim_hidden=cfg["model"]["canonical"]["dim_hidden"],
        sh_degree=cfg["model"]["canonical"]["sh_degree"],
        init_xyz=init_xyz, num_gaussians_per_patch=K,
        init_xyz_per_gaussian=init_xyz_per_gaussian,
    ).cuda()

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

    if checkpoint_path is None:
        s3 = os.path.join(ckpt_dir, "stage3_final.pt")
        checkpoint_path = s3 if os.path.exists(s3) else os.path.join(ckpt_dir, "stage2_final.pt")

    ckpt = torch.load(checkpoint_path, weights_only=False)
    canonical_head.load_state_dict(ckpt["canonical_head"])
    deformation_head.load_state_dict(ckpt["deformation_head"])
    canonical_head.eval()
    deformation_head.eval()

    return canonical_head, deformation_head, dataset, cfg, cfg["training"].get("scale_anneal_target", 1.0)


def compute_covariances(scales, quats):
    N = scales.shape[0]
    w, x, y, z = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
    R = np.zeros((N,3,3), dtype=np.float32)
    R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-w*z); R[:,0,2]=2*(x*z+w*y)
    R[:,1,0]=2*(x*y+w*z);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-w*x)
    R[:,2,0]=2*(x*z-w*y);   R[:,2,1]=2*(y*z+w*x);   R[:,2,2]=1-2*(x*x+y*y)
    S = np.zeros((N,3,3), dtype=np.float32)
    S[:,0,0]=scales[:,0]**2; S[:,1,1]=scales[:,1]**2; S[:,2,2]=scales[:,2]**2
    return np.einsum('nij,njk,nlk->nil', R, S, R)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="D:/DecodeGaussians/experiments/overfit_highres_highcount_coffee_martini/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print("Loading model...")
    canonical_head, deformation_head, dataset, cfg, scale_anneal_target = load_model(args.config, args.checkpoint)
    num_frames = dataset.num_frames
    print(f"Loaded. {num_frames} frames.")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction((0, -1, 0))  # LLFF world up is -Y
    print(f"\nOpen http://localhost:{args.port}")
    print("Controls: WASD move, Q/E up/down, mouse drag to orbit, scroll to zoom")

    frame_slider = server.gui.add_slider("Frame", min=0, max=num_frames-1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)
    server.gui.add_markdown("**Controls:** WASD · Q/E up/down · drag to orbit · scroll zoom")

    print("Precomputing frames...")
    frame_cache = {}
    SH_C0 = 0.28209479177387814
    with torch.no_grad():
        tokens_mean = dataset.get_tokens_mean().cuda()
        canonical = canonical_head(tokens_mean)
        for fi in range(num_frames):
            tokens_t = dataset.get_tokens_frame(fi).cuda()
            deltas = deformation_head(canonical["hidden"], tokens_t)
            means3D, scales, rotations, opacity, shs = compose_gaussians(
                canonical, deltas, scale_factor=scale_anneal_target)
            sc = scales.cpu().numpy().astype(np.float32)
            qt = rotations.cpu().numpy().astype(np.float32)
            frame_cache[fi] = {
                "positions": means3D.cpu().numpy().astype(np.float32),
                "covariances": compute_covariances(sc, qt),
                "opacities": opacity.squeeze(-1).cpu().numpy().astype(np.float32),
                "rgbs": (np.clip(shs[:,0,:].cpu().numpy() * SH_C0 + 0.5, 0, 1) * 255).astype(np.uint8),
            }
            if fi % 50 == 0:
                print(f"  {fi}/{num_frames}")
    print("Ready! Navigate in your browser.")

    init = frame_cache[0]
    splat_handle = server.scene.add_gaussian_splats(
        "/gaussians",
        centers=init["positions"],
        covariances=init["covariances"],
        rgbs=init["rgbs"],
        opacities=init["opacities"].reshape(-1, 1),
    )
    current_frame = [0]

    def update_splats(fi, force=False):
        if fi == current_frame[0] and not force:
            return
        d = frame_cache[fi]
        splat_handle.centers = d["positions"]
        splat_handle.covariances = d["covariances"]
        splat_handle.rgbs = d["rgbs"]
        splat_handle.opacities = d["opacities"].reshape(-1, 1)
        current_frame[0] = fi

    @frame_slider.on_update
    def _(_): update_splats(int(frame_slider.value))

    update_splats(0)

    while True:
        if playing.value:
            fi = (int(frame_slider.value) + 1) % num_frames
            frame_slider.value = fi
            update_splats(fi)
            time.sleep(1.0 / fps_slider.value)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
