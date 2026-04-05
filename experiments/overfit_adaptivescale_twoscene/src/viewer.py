"""Interactive 3D Gaussian viewer for multi-scene model using viser.

Loads a trained multi-scene model and renders Gaussians from a specified scene in real-time.
Navigate with mouse, scrub through frames with a slider.
Open http://localhost:8080 in your browser.

Usage:
    python src/viewer.py --config config.yaml --scene coffee_martini [--checkpoint PATH] [--port 8080]
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
from model import CanonicalGaussianHead, DeformationHead, compose_gaussians

import viser


def load_model(config_path: str, scene_name: str, checkpoint_path: str = None):
    """Load multi-scene trained model for a specific scene visualization."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Find the specified scene in the config
    scene_cfg = None
    for s in cfg["scenes"]:
        if s["name"] == scene_name:
            scene_cfg = s
            break

    if scene_cfg is None:
        raise ValueError(
            f"Scene '{scene_name}' not found in config. "
            f"Available scenes: {[s['name'] for s in cfg['scenes']]}"
        )

    print(f"Loading scene: {scene_name}")
    print(f"  Cache: {scene_cfg['cache_dir']}")
    print(f"  Poses: {scene_cfg['poses_bounds_path']}")

    # Load cameras from LLFF format
    scene_dir = os.path.dirname(scene_cfg["poses_bounds_path"])
    target_w = cfg["rendering"]["image_width"]
    target_h = cfg["rendering"]["image_height"]

    cam_files = sorted([
        os.path.basename(f) for f in glob.glob(os.path.join(scene_dir, "cam*.mp4"))
    ])

    cameras = build_cameras_from_llff(
        scene_cfg["poses_bounds_path"], target_w, target_h, cam_files
    )

    K = cfg["model"]["num_gaussians_per_patch"]
    normalize_coords = cfg["normalization"].get("enabled", True)

    # Load dataset with same normalization as training
    dataset = CachedSceneDataset(
        cache_dir=scene_cfg["cache_dir"],
        cameras=cameras,
        input_camera=scene_cfg.get("input_camera", "cam01"),
        eval_camera=scene_cfg.get("eval_camera", "cam00"),
        normalize_coords=normalize_coords,
        K=K,
    )

    print(f"  Loaded {dataset.num_frames} frames, {dataset.num_patches} patches")
    if dataset.norm_center is not None:
        print(f"  Coordinate normalization: center={dataset.norm_center.tolist()}, scale={dataset.norm_scale:.3f}")

    P = dataset.num_patches

    # Create canonical head (anchors passed at runtime, not stored as buffers)
    canonical_head = CanonicalGaussianHead(
        dim_in=cfg["model"]["dim_tokens"],
        dim_hidden=cfg["model"]["dim_canonical_hidden"],
        sh_degree=cfg["model"]["sh_degree"],
        num_gaussians_per_patch=K,
    ).cuda()

    # Create deformation head
    deformation_head = DeformationHead(
        dim_in=cfg["model"]["dim_tokens"],
        dim_hidden=cfg["model"]["dim_deformation_hidden"],
        n_attn_heads=cfg["model"]["n_attn_heads"],
        n_attn_layers=cfg["model"]["n_attn_layers"],
        sh_degree=cfg["model"]["sh_degree"],
        num_gaussians_per_patch=K,
    ).cuda()

    # Load checkpoint
    ckpt_dir = cfg["logging"]["ckpt_dir"]
    if checkpoint_path is None:
        # Default to stage1_final.pt for this experiment (canonical head only)
        checkpoint_path = os.path.join(ckpt_dir, "stage1_final.pt")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    canonical_head.load_state_dict(ckpt["canonical_head"])
    deformation_head.load_state_dict(ckpt["deformation_head"])
    canonical_head.eval()
    deformation_head.eval()
    print("Model loaded successfully!")

    scale_anneal_target = cfg["training"].get("scale_anneal_target", 1.0)

    return canonical_head, deformation_head, dataset, cfg, scale_anneal_target


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Interactive 3D Gaussian viewer for multi-scene trained model"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to multi-scene config.yaml"
    )
    parser.add_argument(
        "--scene", type=str, required=True,
        help="Name of scene to visualize (must match name in config)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (default: stage3_final.pt or stage2_final.pt)"
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port for viser server (default: 8080)"
    )
    args = parser.parse_args()

    print("="*60)
    print("Multi-Scene Gaussian Viewer")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Scene: {args.scene}")
    print(f"Port: {args.port}")
    print("="*60)

    # Load model
    canonical_head, deformation_head, dataset, cfg, scale_anneal_target = load_model(
        args.config, args.scene, args.checkpoint
    )
    num_frames = dataset.num_frames
    print(f"\n{num_frames} frames available for visualization.")

    # Start viser server
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"\n{'='*60}")
    print(f"Open http://localhost:{args.port} in your browser!")
    print(f"{'='*60}\n")

    # Create GUI controls
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=num_frames - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)

    # Precompute all frames for smooth playback
    print("Precomputing Gaussians for all frames...")
    frame_cache = {}
    with torch.no_grad():
        tokens_mean = dataset.get_tokens_mean().cuda()
        xyz_anchor = dataset.xyz_anchor.cuda()
        scale_anchor = dataset.scale_anchor.cuda()
        canonical = canonical_head(tokens_mean, xyz_anchor=xyz_anchor, scale_anchor=scale_anchor)

        for fi in range(num_frames):
            tokens_t = dataset.get_tokens_frame(fi).cuda()
            # Deformation: predict deltas from per-frame tokens
            deltas = deformation_head(tokens_t)
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
            if fi % 50 == 0:
                print(f"  Frame {fi}/{num_frames}")
    print(f"Precomputed {num_frames} frames.")

    # Precompute covariances and RGB
    print("Precomputing covariance matrices...")
    SH_C0 = 0.28209479177387814
    for fi in range(num_frames):
        data = frame_cache[fi]
        data["covariances"] = compute_covariances(data["scales"], data["quats"])
        data["rgbs"] = (np.clip(data["colors_sh"] * SH_C0 + 0.5, 0, 1) * 255).astype(np.uint8)
        if fi % 50 == 0:
            print(f"  Covariances {fi}/{num_frames}")
    print("Covariances ready.")

    # Create splats ONCE — update properties in-place for smooth playback
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
    print("\nReady! Navigate in your browser.")
    print(f"Scene: {args.scene}")
    print(f"Use slider to scrub through {num_frames} frames")
    print(f"Check 'Play' to animate")
    print("\nPress Ctrl+C to stop.\n")

    # Main loop
    try:
        while True:
            if playing.value:
                frame = int(frame_slider.value)
                frame = (frame + 1) % num_frames
                frame_slider.value = frame
                update_splats(frame)
                time.sleep(1.0 / fps_slider.value)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


def compute_covariances(scales, quats):
    """Compute 3x3 covariance matrices from scales and quaternions."""
    N = scales.shape[0]
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    # Rotation matrix from quaternion
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

    # Scale matrix
    S = np.zeros((N, 3, 3), dtype=np.float32)
    S[:, 0, 0] = scales[:, 0] ** 2
    S[:, 1, 1] = scales[:, 1] ** 2
    S[:, 2, 2] = scales[:, 2] ** 2

    # Covariance = R @ S @ R^T
    cov = np.einsum('nij,njk,nlk->nil', R, S, R)
    return cov


if __name__ == "__main__":
    main()
