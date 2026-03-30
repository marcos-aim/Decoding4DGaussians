"""Direct Gaussian optimization using COLMAP point cloud as initialization.
Comparison: if COLMAP points → high PSNR but STV2 points → low PSNR,
the issue is point quality. If both similar, issue is optimization."""

import os
import sys
import glob
import math
import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from cameras import build_cameras_from_llff
from dataset import CachedSceneDataset
from renderer import render_gaussians
from losses import photometric_loss, compute_psnr, scale_regularization, opacity_regularization


def inverse_sigmoid(x):
    return math.log(x / (1.0 - x))


def load_colmap_points(ply_path: str) -> torch.Tensor:
    """Load COLMAP fused point cloud from PLY file."""
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    x = np.array(ply['vertex']['x'], dtype=np.float32)
    y = np.array(ply['vertex']['y'], dtype=np.float32)
    z = np.array(ply['vertex']['z'], dtype=np.float32)
    pts = np.stack([x, y, z], axis=-1)
    return torch.from_numpy(pts)


def load_colmap_points_txt(txt_path: str) -> torch.Tensor:
    """Load COLMAP sparse points from points3D.txt."""
    pts = []
    with open(txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            # Format: POINT3D_ID X Y Z R G B ERROR TRACK[]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            pts.append([x, y, z])
    if not pts:
        raise ValueError(f"No points found in {txt_path}")
    return torch.tensor(pts, dtype=torch.float32)


def load_colmap_points_bin(bin_path: str) -> torch.Tensor:
    """Load COLMAP sparse points from points3D.bin."""
    import struct
    pts = []
    with open(bin_path, 'rb') as f:
        num_points = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_points):
            point3d_id = struct.unpack('<Q', f.read(8))[0]
            x, y, z = struct.unpack('<ddd', f.read(24))
            r, g, b = struct.unpack('<BBB', f.read(3))
            error = struct.unpack('<d', f.read(8))[0]
            track_length = struct.unpack('<Q', f.read(8))[0]
            for _ in range(track_length):
                f.read(8)  # image_id (4) + point2d_idx (4)
            pts.append([x, y, z])
    if not pts:
        raise ValueError(f"No points found in {bin_path}")
    return torch.tensor(pts, dtype=torch.float32)


def find_colmap_points(scene_dir: str) -> tuple:
    """Find COLMAP point cloud in various possible locations."""
    candidates = [
        # Dense fused
        (os.path.join(scene_dir, "colmap", "dense", "workspace", "fused.ply"), "dense_ply"),
        # Sparse bin (COLMAP 4.0 puts it in 0/0/)
        (os.path.join(scene_dir, "colmap", "sparse", "0", "0", "points3D.bin"), "sparse_bin"),
        (os.path.join(scene_dir, "colmap", "sparse", "0", "points3D.bin"), "sparse_bin"),
        # Sparse txt
        (os.path.join(scene_dir, "colmap", "sparse", "0", "points3D.txt"), "sparse_txt"),
        (os.path.join(scene_dir, "colmap", "sparse_custom", "points3D.txt"), "sparse_txt"),
    ]
    for path, fmt in candidates:
        if os.path.exists(path):
            return path, fmt
    raise FileNotFoundError(f"No COLMAP points found in {scene_dir}/colmap/")


def colmap_to_llff_coords(colmap_pts: torch.Tensor, scene_dir: str,
                           cameras: dict, cam_files: list) -> torch.Tensor:
    """
    Transform COLMAP points to LLFF world space.

    COLMAP mapper creates its own coordinate system. If we used llff2colmap.py
    to set up the COLMAP reconstruction, the COLMAP cameras should match the
    LLFF cameras (same poses were used). So COLMAP points should already be in
    a compatible coordinate system.

    However, we need to verify this. Let's check by projecting through our
    LLFF cameras and seeing if depths make sense.
    """
    # First, just check if COLMAP points are already in LLFF-compatible space
    # by projecting through cam01
    cam01 = cameras.get("cam01")
    if cam01 is None:
        cam01 = list(cameras.values())[0]

    w2v = cam01.world_view_transform.T.cpu()  # 4x4 column-major W2V
    pts_cam = colmap_pts @ w2v[:3, :3].T + w2v[:3, 3]
    z = pts_cam[:, 2]
    z_pos = z[z > 0]

    print(f"  COLMAP points projected through LLFF cam01:")
    print(f"    Z range: [{z.min():.2f}, {z.max():.2f}]")
    print(f"    Z>0: {(z > 0).sum()}/{len(z)} points")
    if len(z_pos) > 0:
        print(f"    Z>0 range: [{z_pos.min():.2f}, {z_pos.max():.2f}]")

    # Check if points project to reasonable screen coordinates
    fx = cam01.image_width / (2.0 * math.tan(cam01.FoVx / 2.0))
    fy = cam01.image_height / (2.0 * math.tan(cam01.FoVy / 2.0))
    cx, cy = cam01.image_width / 2.0, cam01.image_height / 2.0

    valid = z > 0.1
    if valid.sum() > 0:
        u = fx * pts_cam[valid, 0] / pts_cam[valid, 2] + cx
        v = fy * pts_cam[valid, 1] / pts_cam[valid, 2] + cy
        in_bounds = ((u >= 0) & (u < cam01.image_width) &
                     (v >= 0) & (v < cam01.image_height))
        print(f"    In-bounds (cam01): {in_bounds.sum()}/{valid.sum()} = {100*in_bounds.float().mean():.1f}%")

    return colmap_pts


def direct_optimize_colmap(config_path: str, num_steps: int = 5000,
                            densify: bool = True):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    scene_dir = cfg["paths"]["neu3d_scene"]
    cache_dir = cfg["paths"]["cache_dir"]
    render_dir = cfg["paths"]["render_dir"]
    target_w, target_h = cfg["data"]["resolution"]

    cam_files = sorted([os.path.basename(f) for f in
                        glob.glob(os.path.join(scene_dir, "cam*.mp4"))])
    cameras = build_cameras_from_llff(
        os.path.join(scene_dir, "poses_bounds.npy"), target_w, target_h, cam_files
    )

    # Load COLMAP points
    print("=== Loading COLMAP point cloud ===")
    pt_path, pt_fmt = find_colmap_points(scene_dir)
    print(f"  Source: {pt_path} ({pt_fmt})")

    if pt_fmt == "dense_ply":
        colmap_pts = load_colmap_points(pt_path)
    elif pt_fmt == "sparse_txt":
        colmap_pts = load_colmap_points_txt(pt_path)
    elif pt_fmt == "sparse_bin":
        colmap_pts = load_colmap_points_bin(pt_path)

    print(f"  Raw points: {colmap_pts.shape[0]}")
    print(f"  XYZ range: X=[{colmap_pts[:,0].min():.2f}, {colmap_pts[:,0].max():.2f}], "
          f"Y=[{colmap_pts[:,1].min():.2f}, {colmap_pts[:,1].max():.2f}], "
          f"Z=[{colmap_pts[:,2].min():.2f}, {colmap_pts[:,2].max():.2f}]")

    # Transform to LLFF coordinate space if needed
    init_xyz = colmap_to_llff_coords(colmap_pts, scene_dir, cameras, cam_files)

    # Filter points that are behind all cameras (invalid)
    # Keep only points with positive depth in at least one training camera
    train_cams = [n for n in cameras if n != cfg["data"]["eval_camera"]]
    valid_mask = torch.zeros(len(init_xyz), dtype=torch.bool)
    for cam_name in train_cams[:4]:  # check first 4 cams
        cam = cameras[cam_name]
        w2v = cam.world_view_transform.T.cpu()
        z = (init_xyz @ w2v[:3, :3].T + w2v[:3, 3])[:, 2]
        valid_mask |= (z > 0.1)

    init_xyz = init_xyz[valid_mask]
    N = init_xyz.shape[0]
    print(f"  After filtering: {N} valid points")

    if densify and N < 50000:
        # Upsample by adding jittered copies
        target_N = 50000
        repeats = target_N // N + 1
        xyz_repeated = init_xyz.repeat(repeats, 1)[:target_N]
        noise = torch.randn_like(xyz_repeated) * 0.05
        xyz_repeated = xyz_repeated + noise
        # Keep originals exact
        xyz_repeated[:N] = init_xyz
        init_xyz = xyz_repeated
        N = init_xyz.shape[0]
        print(f"  After densification: {N} points")

    # Also load STV2 dataset for GT images
    dataset = CachedSceneDataset(cache_dir, cameras)

    # Initialize Gaussian parameters
    means3D = init_xyz.cuda().requires_grad_(True)
    log_scales = torch.full((N, 3), math.log(0.3), device="cuda").requires_grad_(True)
    quats = torch.zeros(N, 4, device="cuda")
    quats[:, 0] = 1.0
    quats = quats.requires_grad_(True)
    logit_opacity = torch.full((N, 1), inverse_sigmoid(0.5), device="cuda").requires_grad_(True)
    colors = torch.zeros(N, 1, 3, device="cuda").requires_grad_(True)

    params = [
        {"params": [means3D], "lr": 5e-4},
        {"params": [log_scales], "lr": 5e-3},
        {"params": [quats], "lr": 1e-3},
        {"params": [logit_opacity], "lr": 5e-3},
        {"params": [colors], "lr": 5e-3},
    ]
    optimizer = optim.Adam(params)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    bg_color = torch.zeros(3, device="cuda")
    eval_cam = cameras[cfg["data"]["eval_camera"]]
    frame_idx = 0

    out_dir = os.path.join(render_dir, "direct_colmap")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Direct optimization with COLMAP points ({N} Gaussians, {num_steps} steps) ===")

    for step in tqdm(range(num_steps), desc="Direct Optim (COLMAP)"):
        optimizer.zero_grad()

        scales = torch.clamp(F.softplus(log_scales), min=1e-6, max=5.0)
        rotations = F.normalize(quats, dim=-1)
        opacity = torch.sigmoid(logit_opacity)

        cam_name = train_cams[step % len(train_cams)]
        cam = cameras[cam_name]
        gt = dataset.load_frame_image(cam_name, frame_idx).cuda()

        rendered, _, _, _ = render_gaussians(
            means3D, scales, rotations, opacity, colors, cam, bg_color, 0
        )

        loss = photometric_loss(rendered, gt, lambda_ssim=0.85)
        loss = loss + 0.01 * scale_regularization(scales)
        loss = loss + 0.01 * opacity_regularization(opacity)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 500 == 0:
            psnr = compute_psnr(rendered, gt)
            with torch.no_grad():
                s = torch.clamp(F.softplus(log_scales), min=1e-6, max=5.0)
                r = F.normalize(quats, dim=-1)
                o = torch.sigmoid(logit_opacity)
                gt_novel = dataset.load_frame_image(cfg["data"]["eval_camera"], frame_idx).cuda()
                rendered_novel, _, _, _ = render_gaussians(
                    means3D, s, r, o, colors, eval_cam, bg_color, 0
                )
                novel_psnr = compute_psnr(rendered_novel, gt_novel)
            print(f"  Step {step}: train={psnr:.2f} dB, novel={novel_psnr:.2f} dB, loss={loss.item():.4f}")

            if step % 1000 == 0:
                import cv2
                img = (rendered_novel.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_dir, f"colmap_step_{step:05d}.jpg"), img)

    # Final evaluation
    with torch.no_grad():
        scales = torch.clamp(F.softplus(log_scales), min=1e-6, max=5.0)
        rotations = F.normalize(quats, dim=-1)
        opacity = torch.sigmoid(logit_opacity)

        psnrs_train = []
        for cam_name in train_cams[:4]:
            gt = dataset.load_frame_image(cam_name, frame_idx).cuda()
            rendered, _, _, _ = render_gaussians(
                means3D, scales, rotations, opacity, colors,
                cameras[cam_name], bg_color, 0
            )
            psnrs_train.append(compute_psnr(rendered, gt))

        gt_novel = dataset.load_frame_image(cfg["data"]["eval_camera"], frame_idx).cuda()
        rendered_novel, _, _ = render_gaussians(
            means3D, scales, rotations, opacity, colors, eval_cam, bg_color, 0
        )
        novel_psnr = compute_psnr(rendered_novel, gt_novel)

    print("\n" + "=" * 60)
    print(f"COLMAP Direct Optimization Results (frame {frame_idx}):")
    print(f"  Points source: {pt_path}")
    print(f"  Num Gaussians: {N}")
    print(f"  Train PSNR (4 cams avg): {sum(psnrs_train)/len(psnrs_train):.2f} dB")
    print(f"  Novel PSNR (cam00):      {novel_psnr:.2f} dB")
    print(f"  Train/Novel gap:         {sum(psnrs_train)/len(psnrs_train) - novel_psnr:.2f} dB")
    print(f"  Mean scale: {scales.mean():.4f}, Max: {scales.max():.4f}")
    print(f"  Mean opacity: {opacity.mean():.4f}")
    print("=" * 60)

    # Also run STV2 points for direct comparison
    print("\n\n=== NOW RUNNING STV2 POINTS FOR COMPARISON ===")
    run_stv2_comparison(cfg, cameras, dataset, train_cams, eval_cam,
                        bg_color, frame_idx, num_steps, out_dir)


def run_stv2_comparison(cfg, cameras, dataset, train_cams, eval_cam,
                         bg_color, frame_idx, num_steps, out_dir):
    """Run STV2 points with same optimization for fair comparison."""
    pts_mean = dataset.points_map.mean(dim=0).reshape(-1, 3)
    valid_mask = pts_mean.abs().sum(-1) > 1e-6
    init_xyz = pts_mean[valid_mask]
    N = init_xyz.shape[0]
    print(f"  STV2 points: {N}")

    means3D = init_xyz.cuda().requires_grad_(True)
    log_scales = torch.full((N, 3), math.log(0.3), device="cuda").requires_grad_(True)
    quats = torch.zeros(N, 4, device="cuda")
    quats[:, 0] = 1.0
    quats = quats.requires_grad_(True)
    logit_opacity = torch.full((N, 1), inverse_sigmoid(0.5), device="cuda").requires_grad_(True)
    colors = torch.zeros(N, 1, 3, device="cuda").requires_grad_(True)

    params = [
        {"params": [means3D], "lr": 5e-4},
        {"params": [log_scales], "lr": 5e-3},
        {"params": [quats], "lr": 1e-3},
        {"params": [logit_opacity], "lr": 5e-3},
        {"params": [colors], "lr": 5e-3},
    ]
    optimizer = optim.Adam(params)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    for step in tqdm(range(num_steps), desc="Direct Optim (STV2)"):
        optimizer.zero_grad()

        scales = torch.clamp(F.softplus(log_scales), min=1e-6, max=5.0)
        rotations = F.normalize(quats, dim=-1)
        opacity = torch.sigmoid(logit_opacity)

        cam_name = train_cams[step % len(train_cams)]
        cam = cameras[cam_name]
        gt = dataset.load_frame_image(cam_name, frame_idx).cuda()

        rendered, _, _, _ = render_gaussians(
            means3D, scales, rotations, opacity, colors, cam, bg_color, 0
        )

        loss = photometric_loss(rendered, gt, lambda_ssim=0.85)
        loss = loss + 0.01 * scale_regularization(scales)
        loss = loss + 0.01 * opacity_regularization(opacity)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 500 == 0:
            psnr = compute_psnr(rendered, gt)
            with torch.no_grad():
                s = torch.clamp(F.softplus(log_scales), min=1e-6, max=5.0)
                r = F.normalize(quats, dim=-1)
                o = torch.sigmoid(logit_opacity)
                gt_novel = dataset.load_frame_image(cfg["data"]["eval_camera"], frame_idx).cuda()
                rendered_novel, _, _, _ = render_gaussians(
                    means3D, s, r, o, colors, eval_cam, bg_color, 0
                )
                novel_psnr = compute_psnr(rendered_novel, gt_novel)
            print(f"  Step {step}: train={psnr:.2f} dB, novel={novel_psnr:.2f} dB")

    # Final
    with torch.no_grad():
        scales = torch.clamp(F.softplus(log_scales), min=1e-6, max=5.0)
        rotations = F.normalize(quats, dim=-1)
        opacity = torch.sigmoid(logit_opacity)

        psnrs_train = []
        for cam_name in train_cams[:4]:
            gt = dataset.load_frame_image(cam_name, frame_idx).cuda()
            rendered, _, _, _ = render_gaussians(
                means3D, scales, rotations, opacity, colors,
                cameras[cam_name], bg_color, 0
            )
            psnrs_train.append(compute_psnr(rendered, gt))

        gt_novel = dataset.load_frame_image(cfg["data"]["eval_camera"], frame_idx).cuda()
        rendered_novel, _, _ = render_gaussians(
            means3D, scales, rotations, opacity, colors, eval_cam, bg_color, 0
        )
        novel_psnr = compute_psnr(rendered_novel, gt_novel)

    print("\n" + "=" * 60)
    print(f"STV2 Direct Optimization Results (frame {frame_idx}):")
    print(f"  Num Gaussians: {N}")
    print(f"  Train PSNR (4 cams avg): {sum(psnrs_train)/len(psnrs_train):.2f} dB")
    print(f"  Novel PSNR (cam00):      {novel_psnr:.2f} dB")
    print(f"  Train/Novel gap:         {sum(psnrs_train)/len(psnrs_train) - novel_psnr:.2f} dB")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="D:/DecodeGaussians/experiments/overfit_coffee_martini/config.yaml")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--no-densify", action="store_true")
    args = parser.parse_args()
    direct_optimize_colmap(args.config, num_steps=args.steps,
                           densify=not args.no_densify)
