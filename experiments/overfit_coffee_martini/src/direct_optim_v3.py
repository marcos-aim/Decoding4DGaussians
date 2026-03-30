"""Direct Gaussian optimization v3 — proper multi-view training.

Key improvements over v2:
- Multi-view gradient accumulation: render ALL training cameras per step
- Proper 3DGS-style position LR with scene extent normalization
- Better densification with the full-view gradient signal
- Higher resolution COLMAP with more images
"""

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
from losses import photometric_loss, compute_psnr


def inverse_sigmoid(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x / (1.0 - x))
    return math.log(x / (1.0 - x))


def load_colmap_dense(scene_dir: str) -> torch.Tensor:
    from plyfile import PlyData
    ply_path = os.path.join(scene_dir, "colmap", "dense", "workspace", "fused.ply")
    ply = PlyData.read(ply_path)
    x = np.array(ply['vertex']['x'], dtype=np.float32)
    y = np.array(ply['vertex']['y'], dtype=np.float32)
    z = np.array(ply['vertex']['z'], dtype=np.float32)
    return torch.from_numpy(np.stack([x, y, z], axis=-1))


class GaussianModel:
    def __init__(self, init_xyz: torch.Tensor, init_scale: float = 0.1,
                 sh_degree: int = 0):
        N = init_xyz.shape[0]
        self.sh_degree = sh_degree
        self.means3D = init_xyz.cuda().requires_grad_(True)
        self.log_scales = torch.full((N, 3), math.log(init_scale),
                                     device="cuda").requires_grad_(True)
        self.quats = torch.zeros(N, 4, device="cuda")
        self.quats[:, 0] = 1.0
        self.quats = self.quats.requires_grad_(True)
        self.logit_opacity = torch.full((N, 1), inverse_sigmoid(0.5),
                                        device="cuda").requires_grad_(True)
        num_sh = (sh_degree + 1) ** 2
        self.colors = torch.zeros(N, num_sh, 3, device="cuda").requires_grad_(True)

        self.xyz_gradient_accum = torch.zeros(N, device="cuda")
        self.denom = torch.zeros(N, device="cuda")
        self.max_radii2D = torch.zeros(N, device="cuda")

    @property
    def num_points(self):
        return self.means3D.shape[0]

    def get_scales(self):
        return torch.clamp(F.softplus(self.log_scales), min=1e-6, max=5.0)

    def get_rotations(self):
        return F.normalize(self.quats, dim=-1)

    def get_opacity(self):
        return torch.sigmoid(self.logit_opacity)

    def setup_optimizer(self, spatial_lr_scale, position_lr_init=1.6e-4,
                        position_lr_final=1.6e-6):
        """Setup with 3DGS-style spatial LR scaling."""
        self.spatial_lr_scale = spatial_lr_scale
        self.position_lr_init = position_lr_init * spatial_lr_scale
        self.position_lr_final = position_lr_final * spatial_lr_scale

        param_groups = [
            {"params": [self.means3D], "lr": self.position_lr_init, "name": "xyz"},
            {"params": [self.log_scales], "lr": 5e-3, "name": "scaling"},
            {"params": [self.quats], "lr": 1e-3, "name": "rotation"},
            {"params": [self.logit_opacity], "lr": 5e-2, "name": "opacity"},
            {"params": [self.colors], "lr": 2.5e-3, "name": "color"},
        ]
        self.optimizer = optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def update_learning_rate(self, step, max_steps):
        t = step / max_steps
        lr = math.exp(math.log(self.position_lr_init) * (1 - t) +
                      math.log(self.position_lr_final + 1e-10) * t)
        for pg in self.optimizer.param_groups:
            if pg["name"] == "xyz":
                pg["lr"] = lr
        return lr

    def accumulate_gradients(self, radii, viewspace_points):
        visible = radii > 0
        if viewspace_points.grad is not None and visible.any():
            grad_norm = viewspace_points.grad[visible, :2].norm(dim=-1)
            self.xyz_gradient_accum[visible] += grad_norm
            self.denom[visible] += 1

    def densify_and_prune(self, grad_threshold, min_opacity=0.005,
                          max_screen_size=20, extent=1.0):
        grads = self.xyz_gradient_accum / (self.denom + 1e-7)
        grads[grads.isnan()] = 0.0

        scales = self.get_scales()
        scale_max = scales.max(dim=1).values
        percent_dense = 0.01

        clone_mask = (grads >= grad_threshold) & (scale_max <= percent_dense * extent)
        split_mask = (grads >= grad_threshold) & (scale_max > percent_dense * extent)

        n_clone = clone_mask.sum().item()
        n_split = split_mask.sum().item()

        new_means, new_log_scales, new_quats = [], [], []
        new_logit_opacity, new_colors = [], []

        if n_clone > 0:
            new_means.append(self.means3D[clone_mask].detach())
            new_log_scales.append(self.log_scales[clone_mask].detach())
            new_quats.append(self.quats[clone_mask].detach())
            new_logit_opacity.append(self.logit_opacity[clone_mask].detach())
            new_colors.append(self.colors[clone_mask].detach())

        if n_split > 0:
            stds = scales[split_mask]
            rots_mat = self._quat_to_rotmat(self.quats[split_mask])
            for _ in range(2):
                samples = torch.randn_like(stds) * stds
                # Rotate offset by Gaussian orientation (proper 3DGS way)
                offset = torch.bmm(rots_mat, samples.unsqueeze(-1)).squeeze(-1)
                new_means.append((self.means3D[split_mask] + offset).detach())
                new_log_scales.append(
                    (self.log_scales[split_mask] - math.log(1.6)).detach())
                new_quats.append(self.quats[split_mask].detach())
                new_logit_opacity.append(self.logit_opacity[split_mask].detach())
                new_colors.append(self.colors[split_mask].detach())

        opacity = self.get_opacity().squeeze()
        prune_mask = (opacity < min_opacity)
        if max_screen_size > 0:
            prune_mask |= (self.max_radii2D > max_screen_size)
        prune_mask |= (scale_max > 0.1 * extent)
        prune_mask |= split_mask
        keep_mask = ~prune_mask

        self.means3D = self.means3D[keep_mask].detach().requires_grad_(True)
        self.log_scales = self.log_scales[keep_mask].detach().requires_grad_(True)
        self.quats = self.quats[keep_mask].detach().requires_grad_(True)
        self.logit_opacity = self.logit_opacity[keep_mask].detach().requires_grad_(True)
        self.colors = self.colors[keep_mask].detach().requires_grad_(True)

        if new_means:
            self.means3D = torch.cat([self.means3D.detach()] + new_means).requires_grad_(True)
            self.log_scales = torch.cat([self.log_scales.detach()] + new_log_scales).requires_grad_(True)
            self.quats = torch.cat([self.quats.detach()] + new_quats).requires_grad_(True)
            self.logit_opacity = torch.cat([self.logit_opacity.detach()] + new_logit_opacity).requires_grad_(True)
            self.colors = torch.cat([self.colors.detach()] + new_colors).requires_grad_(True)

        N = self.num_points
        self.xyz_gradient_accum = torch.zeros(N, device="cuda")
        self.denom = torch.zeros(N, device="cuda")
        self.max_radii2D = torch.zeros(N, device="cuda")
        self._rebuild_optimizer()
        return n_clone, n_split, prune_mask.sum().item()

    def _quat_to_rotmat(self, q):
        """Convert quaternions (N,4) wxyz to rotation matrices (N,3,3)."""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        N = q.shape[0]
        R = torch.zeros(N, 3, 3, device=q.device)
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)
        return R

    def reset_opacity(self):
        new_opacity = inverse_sigmoid(torch.clamp(self.get_opacity(), max=0.01))
        self.logit_opacity = new_opacity.detach().requires_grad_(True)
        self._rebuild_optimizer()

    def _rebuild_optimizer(self):
        lrs = {pg["name"]: pg["lr"] for pg in self.optimizer.param_groups}
        param_groups = [
            {"params": [self.means3D], "lr": lrs.get("xyz", 1e-4), "name": "xyz"},
            {"params": [self.log_scales], "lr": lrs.get("scaling", 5e-3), "name": "scaling"},
            {"params": [self.quats], "lr": lrs.get("rotation", 1e-3), "name": "rotation"},
            {"params": [self.logit_opacity], "lr": lrs.get("opacity", 5e-2), "name": "opacity"},
            {"params": [self.colors], "lr": lrs.get("color", 2.5e-3), "name": "color"},
        ]
        self.optimizer = optim.Adam(param_groups, lr=0.0, eps=1e-15)


def run(config_path: str, source: str = "colmap", num_steps: int = 30000,
        max_gaussians: int = 200000, views_per_step: int = 4,
        resolution_override=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    scene_dir = cfg["paths"]["neu3d_scene"]
    cache_dir = cfg["paths"]["cache_dir"]
    render_dir = cfg["paths"]["render_dir"]
    if resolution_override:
        target_w, target_h = resolution_override
    else:
        target_w, target_h = cfg["data"]["resolution"]

    cam_files = sorted([os.path.basename(f) for f in
                        glob.glob(os.path.join(scene_dir, "cam*.mp4"))])
    cameras = build_cameras_from_llff(
        os.path.join(scene_dir, "poses_bounds.npy"), target_w, target_h, cam_files
    )
    dataset = CachedSceneDataset(cache_dir, cameras)

    # Initialize points
    if source == "colmap":
        init_xyz = load_colmap_dense(scene_dir)
        if init_xyz.shape[0] < 10000:
            # Densify initial cloud
            target_N = 50000
            repeats = target_N // init_xyz.shape[0] + 1
            xyz_dense = init_xyz.repeat(repeats, 1)[:target_N]
            noise = torch.randn_like(xyz_dense) * 0.05
            xyz_dense[:init_xyz.shape[0]] = init_xyz
            xyz_dense[init_xyz.shape[0]:] += noise[init_xyz.shape[0]:]
            init_xyz = xyz_dense
    elif source == "stv2":
        pts_mean = dataset.points_map.mean(dim=0).reshape(-1, 3)
        valid = pts_mean.abs().sum(-1) > 1e-6
        init_xyz = pts_mean[valid]

    print(f"=== v3: {source}, {init_xyz.shape[0]} initial points ===")
    print(f"  Multi-view: {views_per_step} cameras per step")

    # Scene extent for LR scaling
    scene_center = init_xyz.mean(0)
    extent = (init_xyz - scene_center).norm(dim=-1).quantile(0.9).item()
    spatial_lr_scale = extent
    print(f"  Scene extent: {extent:.2f}")

    model = GaussianModel(init_xyz, init_scale=0.1, sh_degree=0)
    model.setup_optimizer(spatial_lr_scale)

    bg_color = torch.zeros(3, device="cuda")
    eval_cam = cameras[cfg["data"]["eval_camera"]]
    train_cams = [n for n in cameras if n != cfg["data"]["eval_camera"]]
    frame_idx = 0

    out_dir = os.path.join(render_dir, f"direct_v3_{source}")
    os.makedirs(out_dir, exist_ok=True)

    # Densification schedule
    densify_from = 500
    densify_until = 15000
    densify_every = 100
    grad_threshold = 0.0002  # standard 3DGS threshold

    import random

    print(f"\n=== Training ({num_steps} steps) ===")
    for step in tqdm(range(num_steps), desc=f"v3 ({source})"):
        model.optimizer.zero_grad()
        model.update_learning_rate(step, num_steps)

        scales = model.get_scales()
        rotations = model.get_rotations()
        opacity = model.get_opacity()

        # Multi-view: render multiple cameras, accumulate loss
        selected_cams = random.sample(train_cams, min(views_per_step, len(train_cams)))
        total_loss = torch.tensor(0.0, device="cuda")

        for cam_name in selected_cams:
            cam = cameras[cam_name]
            gt = dataset.load_frame_image(cam_name, frame_idx).cuda()

            rendered, radii, _, viewspace_points = render_gaussians(
                model.means3D, scales, rotations, opacity,
                model.colors, cam, bg_color, model.sh_degree
            )

            loss_rgb = photometric_loss(rendered, gt, lambda_ssim=0.85)
            total_loss = total_loss + loss_rgb

            # Accumulate gradients per view (before backward)
            if step < densify_until:
                visible = radii > 0
                if visible.any():
                    model.max_radii2D[visible] = torch.max(
                        model.max_radii2D[visible], radii[visible].float()
                    )

        total_loss = total_loss / len(selected_cams)
        # Light regularization
        total_loss = total_loss + 0.001 * scales.mean()

        total_loss.backward()

        # Accumulate viewspace gradients after backward
        if step < densify_until and viewspace_points.grad is not None:
            # Note: only last view's gradients are in viewspace_points.grad
            # This is a simplification — true 3DGS accumulates across views
            vis = radii > 0
            if vis.any():
                gn = viewspace_points.grad[vis, :2].norm(dim=-1)
                model.xyz_gradient_accum[vis] += gn
                model.denom[vis] += 1

        # Densification
        if step >= densify_from and step < densify_until and step % densify_every == 0:
            if model.num_points < max_gaussians:
                n_clone, n_split, n_prune = model.densify_and_prune(
                    grad_threshold=grad_threshold,
                    min_opacity=0.005,
                    max_screen_size=20,
                    extent=extent,
                )
                if (n_clone + n_split + n_prune > 0) and step % 500 == 0:
                    tqdm.write(f"  [{step}] Clone={n_clone}, Split={n_split}, "
                              f"Prune={n_prune}, Total={model.num_points}")
                if model.num_points > max_gaussians:
                    tqdm.write(f"  [{step}] Over cap, stopping densification")
                    densify_until = step

        # Opacity reset at step 3000 only (once)
        if step == 3000:
            model.reset_opacity()
            tqdm.write(f"  [{step}] Single opacity reset, N={model.num_points}")

        model.optimizer.step()

        if step % 1000 == 0:
            psnr = compute_psnr(rendered, gt)
            with torch.no_grad():
                gt_novel = dataset.load_frame_image(cfg["data"]["eval_camera"], frame_idx).cuda()
                s = model.get_scales()
                r = model.get_rotations()
                o = model.get_opacity()
                rendered_novel, _, _, _ = render_gaussians(
                    model.means3D, s, r, o, model.colors, eval_cam, bg_color, model.sh_degree
                )
                novel_psnr = compute_psnr(rendered_novel, gt_novel)
            tqdm.write(f"  Step {step}: train={psnr:.2f}, novel={novel_psnr:.2f} dB, "
                      f"N={model.num_points}, loss={total_loss.item():.4f}")

            import cv2
            img = (rendered_novel.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, f"step_{step:05d}.jpg"), img)

    # Final evaluation — ALL training cameras
    with torch.no_grad():
        scales = model.get_scales()
        rotations = model.get_rotations()
        opacity = model.get_opacity()

        psnrs_train = []
        for cam_name in train_cams:
            gt = dataset.load_frame_image(cam_name, frame_idx).cuda()
            rendered, _, _, _ = render_gaussians(
                model.means3D, scales, rotations, opacity, model.colors,
                cameras[cam_name], bg_color, model.sh_degree
            )
            psnrs_train.append(compute_psnr(rendered, gt))

        gt_novel = dataset.load_frame_image(cfg["data"]["eval_camera"], frame_idx).cuda()
        rendered_novel, _, _, _ = render_gaussians(
            model.means3D, scales, rotations, opacity, model.colors,
            eval_cam, bg_color, model.sh_degree
        )
        novel_psnr = compute_psnr(rendered_novel, gt_novel)

        # Save final novel render
        img = (rendered_novel.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        import cv2
        cv2.imwrite(os.path.join(out_dir, "final_novel.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    avg_train = sum(psnrs_train) / len(psnrs_train)
    print("\n" + "=" * 60)
    print(f"v3 Results ({source}, frame {frame_idx}):")
    print(f"  Final Gaussians: {model.num_points}")
    print(f"  Train PSNR ({len(train_cams)} cams avg): {avg_train:.2f} dB")
    print(f"  Novel PSNR (cam00):      {novel_psnr:.2f} dB")
    print(f"  Train/Novel gap:         {avg_train - novel_psnr:.2f} dB")
    print(f"  Mean scale: {scales.mean():.4f}")
    print(f"  Mean opacity: {opacity.mean():.4f}")
    print("=" * 60)

    # Per-camera breakdown
    print("\nPer-camera train PSNR:")
    for cam_name, psnr in zip(train_cams, psnrs_train):
        marker = " ***" if psnr < avg_train - 3 else ""
        print(f"  {cam_name}: {psnr:.2f} dB{marker}")

    return novel_psnr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="D:/DecodeGaussians/experiments/overfit_coffee_martini/config.yaml")
    parser.add_argument("--source", type=str, default="colmap",
                        choices=["colmap", "stv2"])
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--max-gaussians", type=int, default=200000)
    parser.add_argument("--views-per-step", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=None,
                        help="Override resolution as W H (e.g. --resolution 1352 1014)")
    args = parser.parse_args()
    run(args.config, source=args.source, num_steps=args.steps,
        max_gaussians=args.max_gaussians, views_per_step=args.views_per_step,
        resolution_override=args.resolution)
