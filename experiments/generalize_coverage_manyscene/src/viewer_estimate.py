"""Interactive 3D Gaussian viewer with VGGT4Track pose estimation.

Same as viewer.py but estimates camera poses at startup by running VGGT4Track
on frame 0 from each camera, rather than loading from poses_bounds.npy.
Useful for diva360 or other scenes without pre-calibrated camera poses.

Usage:
    python src/viewer_estimate.py --config config.yaml --scene pan_synced [--checkpoint PATH] [--port 8080]
"""

import os
import sys
import glob
import yaml
import numpy as np
import torch
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from cameras import MiniCam, llff_pose_to_R_T
from dataset import CachedSceneDataset
from model import CanonicalGaussianHead, DeformationHead, compose_gaussians

import cv2
import viser

# Add SpaTrackerV2 to path
SPATRACKER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "SpaTrackerV2"))
sys.path.insert(0, SPATRACKER_ROOT)

from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track


def vggt_poses_to_cameras(cache_dir: str, target_w: int, target_h: int) -> dict:
    """Estimate camera poses via VGGT4Track and build MiniCam objects.

    Loads frame 0 from each cam* directory in {cache_dir}/frames/, feeds them
    to VGGT4Track as a single "video" to get per-camera w2c poses and intrinsics,
    then builds MiniCam objects directly without going through LLFF format.

    VGGT FrontTracker convention: poses_pred is w2c in OpenCV [right, down, forward].
    Conversion to 4DGS MiniCam convention:
      1. Invert w2c → c2w
      2. Convert OpenCV c2w to LLFF post-axis-swap pose [right, up, backwards, t]
         (right=col0, up=-col1, backwards=-col2, t=c2w[:3,3])
      3. Apply llff_pose_to_R_T: R = -R; R[:,0] = -R[:,0]; T = -t @ R
      4. Build MiniCam(R, T, focal, ...)

    Returns:
        dict mapping cam_name -> MiniCam
    """
    frames_root = os.path.join(cache_dir, "frames")
    cam_dirs = sorted(glob.glob(os.path.join(frames_root, "cam*")))
    camera_names = [os.path.basename(d) for d in cam_dirs if os.path.isdir(d)]

    if len(camera_names) == 0:
        raise FileNotFoundError(f"No cam* directories found in {frames_root}")

    print(f"  Estimating poses for {len(camera_names)} cameras via VGGT4Track...")

    # Load frame 0 from each camera
    frames = []
    for cam_name in camera_names:
        frame_path = os.path.join(frames_root, cam_name, "000000.jpg")
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(img).permute(2, 0, 1).float())

    frames_tensor = torch.stack(frames).unsqueeze(0).cuda() / 255.0  # [1, N, 3, H, W]

    # Run VGGT4Track
    model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    model.eval()
    model = model.to("cuda")

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(frames_tensor)

    poses_pred = predictions["poses_pred"].float().squeeze(0).cpu().numpy()  # [N, 4, 4] w2c
    intrs_pred  = predictions["intrs"].float().squeeze(0).cpu().numpy()      # [N, 3, 3]

    del model
    torch.cuda.empty_cache()

    # Get original frame resolution from first frame file on disk
    first_frame_path = os.path.join(frames_root, camera_names[0], "000000.jpg")
    orig_img = cv2.imread(first_frame_path)
    orig_H, orig_W = orig_img.shape[:2]

    # Build MiniCam for each camera
    cameras = {}
    for i, cam_name in enumerate(camera_names):
        w2c = poses_pred[i]          # (4, 4)
        c2w = np.linalg.inv(w2c)     # invert to camera-to-world
        K   = intrs_pred[i]          # (3, 3)

        R_opencv = c2w[:3, :3]       # columns: [right, down, forward] in world coords
        t_c2w    = c2w[:3, 3]

        # Convert OpenCV c2w to LLFF post-axis-swap pose [right, up, backwards, t]
        # then apply llff_pose_to_R_T for the 4DGS convention
        llff_pose_3x4 = np.zeros((3, 4), dtype=np.float32)
        llff_pose_3x4[:, 0] = R_opencv[:, 0]   # right  = OpenCV right
        llff_pose_3x4[:, 1] = -R_opencv[:, 1]  # up     = -OpenCV down
        llff_pose_3x4[:, 2] = -R_opencv[:, 2]  # back   = -OpenCV forward
        llff_pose_3x4[:, 3] = t_c2w

        R, T = llff_pose_to_R_T(llff_pose_3x4)

        # Average fx/fy — intrinsics are at target resolution already
        focal_at_target = float((K[0, 0] + K[1, 1]) / 2.0)
        # Scale focal to original resolution for MiniCam (it re-scales internally)
        focal_orig = focal_at_target * (orig_W / target_w)

        cameras[cam_name] = MiniCam(
            R=R,
            T=T,
            focal=focal_orig,
            width=target_w,
            height=target_h,
            orig_W=float(orig_W),
            orig_H=float(orig_H),
        )

    print(f"  Built {len(cameras)} MiniCam objects from VGGT4Track predictions.")
    return cameras


def load_model(config_path: str, scene_name: str, checkpoint_path: str = None):
    """Load multi-scene trained model for a specific scene visualization.

    Estimates camera poses via VGGT4Track instead of loading from poses_bounds.npy.
    """
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

    target_w = cfg["rendering"]["image_width"]
    target_h = cfg["rendering"]["image_height"]
    cache_dir = scene_cfg["cache_dir"]

    print(f"Loading scene: {scene_name}")
    print(f"  Cache: {cache_dir}")
    print(f"  Resolution: {target_w}x{target_h}")
    print(f"  Estimating camera poses via VGGT4Track (no poses_bounds.npy needed)...")

    # Estimate cameras from VGGT4Track instead of loading from file
    cameras = vggt_poses_to_cameras(cache_dir, target_w, target_h)

    # Create dataset
    dataset = CachedSceneDataset(
        cache_dir=cache_dir,
        cameras=cameras,
        input_camera=scene_cfg.get("input_camera", "cam01"),
        eval_camera=scene_cfg.get("eval_camera", "cam00"),
        normalize_coords=False,
    )

    print(f"  Loaded {dataset.num_frames} frames, {dataset.num_patches} patches")

    # Initialize model architecture
    P = dataset.num_patches
    K = cfg["model"]["num_gaussians_per_patch"]

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

    ckpt_dir = cfg["logging"]["ckpt_dir"]
    if checkpoint_path is None:
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
        description="Interactive 3D Gaussian viewer with VGGT4Track pose estimation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True,
                        help="Name of scene to visualize (must match name in config)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print("="*60)
    print("Multi-Scene Gaussian Viewer (VGGT4Track pose estimation)")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Scene: {args.scene}")
    print(f"Port: {args.port}")
    print("="*60)

    canonical_head, deformation_head, dataset, cfg, scale_anneal_target = load_model(
        args.config, args.scene, args.checkpoint
    )
    num_frames = dataset.num_frames
    print(f"\n{num_frames} frames available for visualization.")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"\n{'='*60}")
    print(f"Open http://localhost:{args.port} in your browser!")
    print(f"{'='*60}\n")

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

        for fi in range(num_frames):
            tokens_t = dataset.get_tokens_frame(fi).cuda()
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
    print("\nReady! Navigate in your browser.")
    print(f"Scene: {args.scene}")
    print(f"Use slider to scrub through {num_frames} frames")
    print(f"Check 'Play' to animate")
    print("\nPress Ctrl+C to stop.\n")

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
