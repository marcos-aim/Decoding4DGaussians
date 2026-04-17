"""
Diagnostic script: visualize raw/aligned points_map and verify STV2→LLFF alignment.

Default mode — outputs to experiments/overfit_synthetic_1/diagnostics/:
  raw_points_frame0.ply       — VGGT point cloud before alignment
  aligned_points_frame0.ply   — after alignment, with camera positions
  overlay_cam00_frame0.jpg    — projected points on cam00 GT image
  overlay_cam01_frame0.jpg    — projected points on cam01 GT image

Viewer mode (--viewer) — interactive browser-based 3D point cloud viewer:
  Opens http://localhost:8080. Frame slider, depth/RGB color toggle.

Usage:
  conda run -n SpaTrack2 python experiments/overfit_synthetic_1/src/diagnose_alignment.py
  conda run -n SpaTrack2 python experiments/overfit_synthetic_1/src/diagnose_alignment.py --viewer
"""

import os
import sys
import struct
import glob
import numpy as np
import torch
import cv2
import yaml

# Allow importing sibling modules
sys.path.insert(0, os.path.dirname(__file__))
from cameras import build_cameras_from_llff, parse_llff_poses

SCRIPT_DIR = os.path.dirname(__file__)
EXP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(os.path.dirname(EXP_DIR))
CONFIG_PATH = os.path.join(EXP_DIR, "config.yaml")
OUT_DIR = os.path.join(EXP_DIR, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# PLY writer
# ---------------------------------------------------------------------------

def write_ply(path: str, points: np.ndarray, colors: np.ndarray = None):
    """Write binary PLY. points: (N,3) float32, colors: (N,3) uint8."""
    N = len(points)
    has_color = colors is not None
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
    )
    if has_color:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        for i in range(N):
            f.write(struct.pack("fff", *points[i]))
            if has_color:
                f.write(struct.pack("BBB", *colors[i]))
    print(f"  Saved {N} points → {path}")


# ---------------------------------------------------------------------------
# Depth colormap: blue (near) → red (far)
# ---------------------------------------------------------------------------

def depth_to_color(z: np.ndarray) -> np.ndarray:
    """Map depth values to BGR colors for overlay."""
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    colors_bgr = np.zeros((len(z), 3), dtype=np.uint8)
    colors_bgr[:, 0] = (255 * (1 - z_norm)).astype(np.uint8)  # B: near
    colors_bgr[:, 2] = (255 * z_norm).astype(np.uint8)         # R: far
    return colors_bgr


# ---------------------------------------------------------------------------
# Numeric diagnostics helper
# ---------------------------------------------------------------------------

def depth_stats(label: str, z: torch.Tensor):
    z_pos = z[z > 0.01]
    if len(z_pos) == 0:
        print(f"  [{label}] no positive-depth points")
        return
    z_sorted = z_pos.sort().values
    n = len(z_sorted)
    print(f"  [{label}] n={n:,}  "
          f"min={z_sorted[0]:.3f}  "
          f"p5={z_sorted[n//20]:.3f}  "
          f"p50={z_sorted[n//2]:.3f}  "
          f"p95={z_sorted[int(n*0.95)]:.3f}  "
          f"max={z_sorted[-1]:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    scene_dir = cfg["paths"]["neu3d_scene"]
    cache_dir = cfg["paths"]["cache_dir"]
    target_w, target_h = cfg["data"]["resolution"]
    input_camera = cfg["data"]["input_camera"]

    poses_bounds_path = os.path.join(scene_dir, "poses_bounds.npy")
    cam_files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(scene_dir, "cam*.mp4"))])
    print(f"Found {len(cam_files)} cameras: {cam_files[:4]}...")

    cameras = build_cameras_from_llff(poses_bounds_path, target_w, target_h, cam_files)

    # Load LLFF near/far for reference
    parsed = parse_llff_poses(poses_bounds_path)
    near_fars = parsed["near_fars"]
    cam_names = [f.replace(".mp4", "") for f in cam_files]
    cam01_idx = cam_names.index(input_camera)
    print(f"\nLLFF near/far for {input_camera}: near={near_fars[cam01_idx,0]:.4f}, far={near_fars[cam01_idx,1]:.4f}")
    print(f"LLFF near/far range across all cams: near=[{near_fars[:,0].min():.4f}, {near_fars[:,0].max():.4f}], "
          f"far=[{near_fars[:,1].min():.2f}, {near_fars[:,1].max():.2f}]")

    # Load raw tensors
    print("\nLoading cached tensors...")
    points_map = torch.load(os.path.join(cache_dir, "points_map.pt"), weights_only=True).float()
    stv2_poses = torch.load(os.path.join(cache_dir, "poses.pt"), weights_only=True).float()
    print(f"  points_map: {tuple(points_map.shape)}  dtype={points_map.dtype}")
    print(f"  stv2_poses: {tuple(stv2_poses.shape)}")

    # -----------------------------------------------------------------------
    # 1. Raw point cloud diagnostics (VGGT space, frame 0)
    # -----------------------------------------------------------------------
    print("\n--- Raw VGGT point cloud (frame 0) ---")
    stv2_c2w_0 = stv2_poses[0]
    stv2_inv = torch.inverse(stv2_c2w_0)

    pts_raw = points_map[0].reshape(-1, 3)  # (H*W, 3)

    # Camera-space depth (Z axis) from STV2 cam0 frame
    pts_cam = pts_raw @ stv2_inv[:3, :3].T + stv2_inv[:3, 3]
    z_stv2 = pts_cam[:, 2]
    depth_stats("STV2 cam-space Z", z_stv2)

    # Subsample for PLY export
    idx = torch.randperm(len(pts_raw))[:50000]
    pts_sub = pts_raw[idx].numpy().astype(np.float32)
    # Color by depth
    z_sub = z_stv2[idx].numpy()
    valid_sub = z_sub > 0.01
    colors_raw = np.full((len(pts_sub), 3), 128, dtype=np.uint8)
    if valid_sub.sum() > 0:
        colors_raw[valid_sub] = depth_to_color(z_sub[valid_sub])[:, ::-1]  # BGR→RGB

    write_ply(os.path.join(OUT_DIR, "raw_points_frame0.ply"), pts_sub, colors_raw)

    # -----------------------------------------------------------------------
    # 2. Alignment step-by-step
    # -----------------------------------------------------------------------
    print("\n--- Alignment (STV2 → LLFF) ---")

    cam01 = cameras[input_camera]
    w2v_llff = cam01.world_view_transform.T.cpu()
    c2w_llff = torch.inverse(w2v_llff)
    R_c2w = c2w_llff[:3, :3]
    t_c2w = c2w_llff[:3, 3]

    # Scale factor
    z_positive = z_stv2[z_stv2 > 0.01]
    z_sorted = z_positive.sort().values
    z_near_stv2 = z_sorted[len(z_sorted) // 20]

    # cam_center_llff approach (as used in synthetic dataset.py)
    cam_center_llff = torch.inverse(w2v_llff)[:3, 3]
    near_from_cam_center = float(cam_center_llff.norm())

    print(f"  STV2 5th-pct depth: {z_near_stv2:.4f}")
    print(f"  LLFF cam01 distance from origin: {near_from_cam_center:.4f}")
    print(f"  LLFF near bound (from poses_bounds.npy): {near_fars[cam01_idx, 0]:.4f}")
    print(f"  Scale (cam_center method): {near_from_cam_center / float(z_near_stv2):.2f}")
    print(f"  Scale (near_bound method): {near_fars[cam01_idx, 0] / float(z_near_stv2):.2f}")

    scale = near_from_cam_center / float(z_near_stv2)

    # Lateral correction: match VGGT focal to LLFF focal
    lateral_x = lateral_y = 1.0
    intrs_path = os.path.join(cache_dir, "intrs.pt")
    if os.path.exists(intrs_path):
        import math
        intrs = torch.load(intrs_path, weights_only=True).float()
        fx_vggt = intrs[0, 0, 0].item()
        fy_vggt = intrs[0, 1, 1].item()
        cam01_obj = cameras[input_camera]
        fx_llff = cam01_obj.image_width / (2.0 * math.tan(cam01_obj.FoVx / 2.0))
        fy_llff = cam01_obj.image_height / (2.0 * math.tan(cam01_obj.FoVy / 2.0))
        lateral_x = fx_vggt / fx_llff
        lateral_y = fy_vggt / fy_llff
        print(f"  VGGT focal: fx={fx_vggt:.2f}, fy={fy_vggt:.2f}")
        print(f"  LLFF focal: fx={fx_llff:.2f}, fy={fy_llff:.2f}")
        print(f"  Lateral correction: x*={lateral_x:.3f}, y*={lateral_y:.3f}")

    # Apply transform to all frames (for stats), frame 0 for PLY
    pts_all = points_map.reshape(-1, 3)
    R_stv2_inv = stv2_inv[:3, :3]
    pts_cam_all = pts_all @ R_stv2_inv.T + stv2_inv[:3, 3]
    pts_cam_all = pts_cam_all * torch.tensor([lateral_x, lateral_y, 1.0], dtype=pts_cam_all.dtype)
    pts_scaled = scale * pts_cam_all
    pts_aligned_all = pts_scaled @ R_c2w.T + t_c2w

    pts_frame0 = pts_aligned_all[:points_map.shape[1] * points_map.shape[2]]

    # Verify depth in cam01 space after alignment
    pts_v = pts_frame0 @ w2v_llff[:3, :3].T + w2v_llff[:3, 3]
    z_after = pts_v[:, 2]
    depth_stats("Aligned cam01 depth", z_after)
    print(f"  Expected near ≈ {near_from_cam_center:.3f} (cam_center method)")

    # Depth in cam00 space
    cam00 = cameras["cam00"]
    w2v_cam00 = cam00.world_view_transform.T.cpu()
    pts_v00 = pts_frame0 @ w2v_cam00[:3, :3].T + w2v_cam00[:3, 3]
    z_cam00 = pts_v00[:, 2]
    depth_stats("Aligned cam00 depth", z_cam00)

    # -----------------------------------------------------------------------
    # 3. Aligned PLY with camera positions
    # -----------------------------------------------------------------------
    print("\n--- Exporting aligned PLY ---")
    idx2 = torch.randperm(len(pts_frame0))[:50000]
    pts_al_sub = pts_frame0[idx2].numpy().astype(np.float32)
    z_al_sub = z_after[idx2].numpy()
    valid_al = z_al_sub > 0
    colors_al = np.full((len(pts_al_sub), 3), 128, dtype=np.uint8)
    if valid_al.sum() > 0:
        colors_al[valid_al] = depth_to_color(z_al_sub[valid_al])[:, ::-1]  # BGR→RGB

    # Add camera positions as red points
    cam_positions = []
    for cam_name, cam in cameras.items():
        pos = torch.inverse(cam.world_view_transform.T.cpu())[:3, 3].numpy()
        cam_positions.append(pos)
    cam_positions = np.array(cam_positions, dtype=np.float32)
    cam_colors = np.tile([255, 0, 0], (len(cam_positions), 1)).astype(np.uint8)  # red

    all_pts = np.concatenate([pts_al_sub, cam_positions], axis=0)
    all_colors = np.concatenate([colors_al, cam_colors], axis=0)
    write_ply(os.path.join(OUT_DIR, "aligned_points_frame0.ply"), all_pts, all_colors)

    # -----------------------------------------------------------------------
    # 4. Projection overlays
    # -----------------------------------------------------------------------
    print("\n--- Projection overlays ---")

    # Use LLFF focal (matches GT images and training cameras)
    focal = parsed["focal"]
    fx = focal * (target_w / parsed["W"])
    fy = focal * (target_h / parsed["H"])
    cx = target_w / 2.0
    cy = target_h / 2.0
    print(f"  Using LLFF intrinsics: fx={fx:.2f}, fy={fy:.2f}")

    pts_np = pts_frame0.numpy()  # (H*W, 3)

    for cam_name in ["cam00", "cam01"]:
        cam = cameras[cam_name]
        frame_path = os.path.join(cache_dir, "frames", cam_name, "000000.jpg")
        if not os.path.exists(frame_path):
            print(f"  [WARN] frame not found: {frame_path}")
            continue

        img = cv2.imread(frame_path)  # BGR, (H, W, 3)

        w2v = cam.world_view_transform.T.cpu().numpy()  # (4,4) column-major W2V
        pts_c = pts_np @ w2v[:3, :3].T + w2v[:3, 3]    # (N, 3) in camera space

        # Keep only points in front of camera
        z = pts_c[:, 2]
        valid = z > 0.1
        pts_c_v = pts_c[valid]
        z_v = z[valid]

        # Project to pixel coordinates
        u = fx * pts_c_v[:, 0] / z_v + cx
        v = fy * pts_c_v[:, 1] / z_v + cy

        # Keep within image bounds
        in_bounds = (u >= 0) & (u < target_w) & (v >= 0) & (v < target_h)
        u = u[in_bounds].astype(int)
        v = v[in_bounds].astype(int)
        z_final = z_v[in_bounds]

        # Color by depth
        colors_bgr = depth_to_color(z_final)

        # Draw dots on image (radius 2)
        overlay = img.copy()
        for i in range(len(u)):
            cv2.circle(overlay, (u[i], v[i]), 2, (int(colors_bgr[i, 0]),
                                                    int(colors_bgr[i, 1]),
                                                    int(colors_bgr[i, 2])), -1)

        out_path = os.path.join(OUT_DIR, f"overlay_{cam_name}_frame0.jpg")
        cv2.imwrite(out_path, overlay)
        print(f"  {cam_name}: projected {len(u):,} points → {out_path}")

    # -----------------------------------------------------------------------
    # 5. Coffee-martini comparison (if available)
    # -----------------------------------------------------------------------
    cm_cache = os.path.join(REPO_DIR, "cached_datasets", "neu3d", "cache_coffee_martini")
    if os.path.exists(os.path.join(cm_cache, "points_map.pt")):
        print("\n--- Coffee-martini comparison ---")
        cm_pts = torch.load(os.path.join(cm_cache, "points_map.pt"), weights_only=True).float()
        cm_poses = torch.load(os.path.join(cm_cache, "poses.pt"), weights_only=True).float()
        cm_inv = torch.inverse(cm_poses[0])
        cm_pts_flat = cm_pts[0].reshape(-1, 3)
        cm_pts_cam = cm_pts_flat @ cm_inv[:3, :3].T + cm_inv[:3, 3]
        depth_stats("CM raw STV2 Z", cm_pts_cam[:, 2])
    else:
        print(f"\n  [INFO] Coffee-martini cache not found at {cm_cache}, skipping comparison")

    print("\nDone. Check diagnostics/ folder.")


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

def run_viewer(port: int = 8080):
    """Launch an interactive viser viewer showing the aligned point cloud."""
    import viser
    import time

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    scene_dir = cfg["paths"]["neu3d_scene"]
    cache_dir = cfg["paths"]["cache_dir"]
    target_w, target_h = cfg["data"]["resolution"]
    input_camera = cfg["data"]["input_camera"]

    poses_bounds_path = os.path.join(scene_dir, "poses_bounds.npy")
    cam_files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(scene_dir, "cam*.mp4"))])
    cameras = build_cameras_from_llff(poses_bounds_path, target_w, target_h, cam_files)
    parsed = parse_llff_poses(poses_bounds_path)

    print("Loading cached tensors...")
    points_map = torch.load(os.path.join(cache_dir, "points_map.pt"), weights_only=True).float()
    stv2_poses = torch.load(os.path.join(cache_dir, "poses.pt"), weights_only=True).float()
    num_frames = points_map.shape[0]
    print(f"  {num_frames} frames, {points_map.shape[1]}x{points_map.shape[2]} points each")

    # Compute alignment transform (same as dataset.py)
    stv2_c2w_0 = stv2_poses[0]
    stv2_inv = torch.inverse(stv2_c2w_0)
    cam01 = cameras[input_camera]
    w2v_llff = cam01.world_view_transform.T.cpu()
    c2w_llff = torch.inverse(w2v_llff)
    R_c2w = c2w_llff[:3, :3]
    t_c2w = c2w_llff[:3, 3]
    R_stv2_inv = stv2_inv[:3, :3]

    pts_raw0 = points_map[0].reshape(-1, 3)
    pts_cam0 = pts_raw0 @ stv2_inv[:3, :3].T + stv2_inv[:3, 3]
    z_stv2 = pts_cam0[:, 2]
    z_pos = z_stv2[z_stv2 > 0.01].sort().values
    z_near_stv2 = z_pos[len(z_pos) // 20]
    near_llff = float(torch.inverse(w2v_llff)[:3, 3].norm())
    scale = near_llff / float(z_near_stv2)
    print(f"  Alignment scale: {scale:.3f}")

    # Pre-align all frames, subsample to ~20K points for display performance
    DISPLAY_PTS = 20000
    H_pts, W_pts = points_map.shape[1], points_map.shape[2]
    total_pts = H_pts * W_pts
    step = max(1, total_pts // DISPLAY_PTS)
    sample_idx = torch.arange(0, total_pts, step)

    print("Pre-aligning all frames...")
    aligned_frames = []
    raw_frames = []
    for fi in range(num_frames):
        pts = points_map[fi].reshape(-1, 3)[sample_idx]
        # Raw (VGGT space, subsampled)
        raw_frames.append(pts.numpy().astype(np.float32))
        # Aligned
        pts_sc = scale * (pts @ R_stv2_inv.T + stv2_inv[:3, 3])
        pts_al = pts_sc @ R_c2w.T + t_c2w
        aligned_frames.append(pts_al.numpy().astype(np.float32))

    # Load GT image colors for cam01 (for RGB coloring mode)
    print("Loading GT image colors...")
    image_colors = []
    for fi in range(num_frames):
        img_path = os.path.join(cache_dir, "frames", input_camera, f"{fi:06d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            image_colors.append(None)
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # (H, W, 3)
        # Flatten and subsample to match point indices
        img_flat = img_rgb.reshape(-1, 3)[sample_idx.numpy()]
        image_colors.append(img_flat.astype(np.uint8))

    def make_depth_colors(pts: np.ndarray) -> np.ndarray:
        """Color points by depth (Z value), blue=near red=far."""
        z = pts[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        rgb = np.zeros((len(z), 3), dtype=np.uint8)
        rgb[:, 0] = (255 * z_norm).astype(np.uint8)   # R: far
        rgb[:, 2] = (255 * (1 - z_norm)).astype(np.uint8)  # B: near
        return rgb

    # Camera positions in world space
    cam_positions = np.array([
        torch.inverse(cameras[c].world_view_transform.T.cpu())[:3, 3].numpy()
        for c in sorted(cameras.keys())
    ], dtype=np.float32)
    cam_colors = np.tile(np.array([[255, 80, 80]], dtype=np.uint8), (len(cam_positions), 1))

    # --- Viser server ---
    server = viser.ViserServer(host="0.0.0.0", port=port)
    print(f"\nOpen http://localhost:{port} in your browser!")

    frame_slider = server.gui.add_slider("Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
    color_mode = server.gui.add_dropdown("Color", options=["Depth", "RGB"], initial_value="Depth")
    show_aligned = server.gui.add_checkbox("Aligned (vs Raw)", initial_value=True)
    show_cameras = server.gui.add_checkbox("Show Cameras", initial_value=True)
    point_size = server.gui.add_slider("Point Size", min=0.01, max=0.2, step=0.005, initial_value=0.04)

    # Static camera markers
    cam_cloud = server.scene.add_point_cloud(
        "/cameras", points=cam_positions, colors=cam_colors, point_size=0.12
    )

    def get_colors(fi: int, pts: np.ndarray) -> np.ndarray:
        if color_mode.value == "RGB" and image_colors[fi] is not None:
            return image_colors[fi]
        return make_depth_colors(pts)

    def refresh(_=None):
        fi = int(frame_slider.value)
        pts = aligned_frames[fi] if show_aligned.value else raw_frames[fi]
        colors = get_colors(fi, pts)
        server.scene.add_point_cloud("/points", points=pts, colors=colors,
                                      point_size=point_size.value)
        cam_cloud.visible = show_cameras.value

    frame_slider.on_update(refresh)
    color_mode.on_update(refresh)
    show_aligned.on_update(refresh)
    show_cameras.on_update(refresh)
    point_size.on_update(refresh)

    refresh()
    print(f"Showing {len(sample_idx):,} points per frame. Use the Frame slider to scrub.")
    print("Color=Depth shows depth structure. Color=RGB shows GT texture projected onto points.")

    while True:
        time.sleep(0.05)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="Launch interactive viser viewer")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.viewer:
        run_viewer(port=args.port)
    else:
        main()
