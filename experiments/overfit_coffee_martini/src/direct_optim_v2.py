"""Direct Gaussian optimization v2 — with 3DGS-style adaptive density control.

Key differences from v1:
- Densification: clone small high-grad Gaussians, split large high-grad ones
- Pruning: remove near-transparent Gaussians periodically
- 30K steps with exponential LR decay for positions
- Opacity reset every 3K steps
- Proper gradient accumulation for densification decisions
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
    """Load COLMAP dense fused point cloud."""
    from plyfile import PlyData
    ply_path = os.path.join(scene_dir, "colmap", "dense", "workspace", "fused.ply")
    ply = PlyData.read(ply_path)
    x = np.array(ply['vertex']['x'], dtype=np.float32)
    y = np.array(ply['vertex']['y'], dtype=np.float32)
    z = np.array(ply['vertex']['z'], dtype=np.float32)
    return torch.from_numpy(np.stack([x, y, z], axis=-1))


class GaussianModel:
    """Manages Gaussian parameters with densification support."""

    def __init__(self, init_xyz: torch.Tensor, init_scale: float = 0.3,
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
        num_sh_coeffs = (sh_degree + 1) ** 2
        self.colors = torch.zeros(N, num_sh_coeffs, 3, device="cuda").requires_grad_(True)

        # Gradient accumulation for densification
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

    def get_params(self):
        return [self.means3D, self.log_scales, self.quats,
                self.logit_opacity, self.colors]

    def setup_optimizer(self, position_lr_init=1.6e-4, position_lr_final=1.6e-6,
                        scale_lr=5e-3, rotation_lr=1e-3, opacity_lr=5e-2,
                        color_lr=2.5e-3):
        self.position_lr_init = position_lr_init
        self.position_lr_final = position_lr_final

        param_groups = [
            {"params": [self.means3D], "lr": position_lr_init, "name": "xyz"},
            {"params": [self.log_scales], "lr": scale_lr, "name": "scaling"},
            {"params": [self.quats], "lr": rotation_lr, "name": "rotation"},
            {"params": [self.logit_opacity], "lr": opacity_lr, "name": "opacity"},
            {"params": [self.colors], "lr": color_lr, "name": "color"},
        ]
        self.optimizer = optim.Adam(param_groups, lr=0.0, eps=1e-15)
        return self.optimizer

    def update_learning_rate(self, step, max_steps):
        """Exponential decay for position LR."""
        lr = self.position_lr_final + (self.position_lr_init - self.position_lr_final) * \
             math.exp(-step / (max_steps * 0.3))
        for pg in self.optimizer.param_groups:
            if pg["name"] == "xyz":
                pg["lr"] = lr
        return lr

    def accumulate_gradients(self, radii, viewspace_points):
        """Accumulate position gradients for densification decisions."""
        visible = radii > 0
        if viewspace_points.grad is not None:
            grad_norm = viewspace_points.grad[visible, :2].norm(dim=-1)
            self.xyz_gradient_accum[visible] += grad_norm
            self.denom[visible] += 1

    def densify_and_prune(self, grad_threshold=0.0002, min_opacity=0.005,
                          max_screen_size=20, extent=1.0):
        """3DGS-style densification: clone small, split large, prune transparent."""
        grads = self.xyz_gradient_accum / (self.denom + 1e-7)
        grads[grads.isnan()] = 0.0

        scales = self.get_scales()
        scale_max = scales.max(dim=1).values

        # Clone: small Gaussians with large gradients
        clone_mask = (grads >= grad_threshold) & (scale_max <= extent * 0.01)
        # Split: large Gaussians with large gradients
        split_mask = (grads >= grad_threshold) & (scale_max > extent * 0.01)

        n_clone = clone_mask.sum().item()
        n_split = split_mask.sum().item()

        new_means = []
        new_log_scales = []
        new_quats = []
        new_logit_opacity = []
        new_colors = []

        # Clone: duplicate as-is
        if n_clone > 0:
            new_means.append(self.means3D[clone_mask].detach())
            new_log_scales.append(self.log_scales[clone_mask].detach())
            new_quats.append(self.quats[clone_mask].detach())
            new_logit_opacity.append(self.logit_opacity[clone_mask].detach())
            new_colors.append(self.colors[clone_mask].detach())

        # Split: create 2 smaller copies offset along principal axis
        if n_split > 0:
            stds = scales[split_mask]  # (n_split, 3)
            # Sample 2 offset positions
            for _ in range(2):
                offset = torch.randn_like(stds) * stds
                new_means.append((self.means3D[split_mask] + offset).detach())
                # Reduce scale by factor of 1.6
                new_log_scales.append(
                    (self.log_scales[split_mask] - math.log(1.6)).detach())
                new_quats.append(self.quats[split_mask].detach())
                new_logit_opacity.append(self.logit_opacity[split_mask].detach())
                new_colors.append(self.colors[split_mask].detach())

        # Prune: remove transparent and too-large Gaussians
        opacity = self.get_opacity().squeeze()
        prune_mask = (opacity < min_opacity)
        if max_screen_size > 0:
            prune_mask |= (self.max_radii2D > max_screen_size)
        # World-space scale pruning (from 4DGS: scale > 0.1 * extent)
        prune_mask |= (scale_max > 0.1 * extent)
        # Also remove split originals
        prune_mask |= split_mask
        keep_mask = ~prune_mask

        # Apply pruning
        self.means3D = self.means3D[keep_mask].detach().requires_grad_(True)
        self.log_scales = self.log_scales[keep_mask].detach().requires_grad_(True)
        self.quats = self.quats[keep_mask].detach().requires_grad_(True)
        self.logit_opacity = self.logit_opacity[keep_mask].detach().requires_grad_(True)
        self.colors = self.colors[keep_mask].detach().requires_grad_(True)

        # Append new points
        if new_means:
            all_new_means = torch.cat(new_means)
            all_new_log_scales = torch.cat(new_log_scales)
            all_new_quats = torch.cat(new_quats)
            all_new_logit_opacity = torch.cat(new_logit_opacity)
            all_new_colors = torch.cat(new_colors)

            self.means3D = torch.cat([self.means3D.detach(), all_new_means]).requires_grad_(True)
            self.log_scales = torch.cat([self.log_scales.detach(), all_new_log_scales]).requires_grad_(True)
            self.quats = torch.cat([self.quats.detach(), all_new_quats]).requires_grad_(True)
            self.logit_opacity = torch.cat([self.logit_opacity.detach(), all_new_logit_opacity]).requires_grad_(True)
            self.colors = torch.cat([self.colors.detach(), all_new_colors]).requires_grad_(True)

        N = self.num_points
        self.xyz_gradient_accum = torch.zeros(N, device="cuda")
        self.denom = torch.zeros(N, device="cuda")
        self.max_radii2D = torch.zeros(N, device="cuda")

        # Rebuild optimizer with new params
        self._rebuild_optimizer()

        return n_clone, n_split, prune_mask.sum().item()

    def reset_opacity(self):
        """Reset opacity to a low value — forces pruning of useless Gaussians."""
        new_opacity = inverse_sigmoid(torch.clamp(self.get_opacity(), max=0.01))
        self.logit_opacity = new_opacity.detach().requires_grad_(True)
        self._rebuild_optimizer()

    def _rebuild_optimizer(self):
        """Rebuild optimizer after densification changes parameter tensors."""
        # Preserve LR from current optimizer
        lrs = {}
        for pg in self.optimizer.param_groups:
            lrs[pg["name"]] = pg["lr"]

        param_groups = [
            {"params": [self.means3D], "lr": lrs.get("xyz", 1e-4), "name": "xyz"},
            {"params": [self.log_scales], "lr": lrs.get("scaling", 5e-3), "name": "scaling"},
            {"params": [self.quats], "lr": lrs.get("rotation", 1e-3), "name": "rotation"},
            {"params": [self.logit_opacity], "lr": lrs.get("opacity", 5e-2), "name": "opacity"},
            {"params": [self.colors], "lr": lrs.get("color", 2.5e-3), "name": "color"},
        ]
        self.optimizer = optim.Adam(param_groups, lr=0.0, eps=1e-15)


def run_optimization(config_path: str, source: str = "colmap",
                     num_steps: int = 30000, max_gaussians: int = 200000,
                     sh_degree: int = 0, no_opacity_reset: bool = False):
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
    dataset = CachedSceneDataset(cache_dir, cameras)

    # Initialize points
    if source == "colmap":
        print("=== Using COLMAP dense points ===")
        init_xyz = load_colmap_dense(scene_dir)
        # Densify sparse COLMAP cloud
        if init_xyz.shape[0] < 10000:
            target_N = 50000
            repeats = target_N // init_xyz.shape[0] + 1
            xyz_dense = init_xyz.repeat(repeats, 1)[:target_N]
            noise = torch.randn_like(xyz_dense) * 0.05
            xyz_dense[:init_xyz.shape[0]] = init_xyz  # keep originals
            xyz_dense[init_xyz.shape[0]:] += noise[init_xyz.shape[0]:]
            init_xyz = xyz_dense
    elif source == "stv2":
        print("=== Using STV2 aligned points ===")
        pts_mean = dataset.points_map.mean(dim=0).reshape(-1, 3)
        valid = pts_mean.abs().sum(-1) > 1e-6
        init_xyz = pts_mean[valid]
    elif source == "random":
        print("=== Using random points in scene bbox ===")
        # Get bbox from COLMAP dense
        colmap_pts = load_colmap_dense(scene_dir)
        bbox_min = colmap_pts.min(0).values
        bbox_max = colmap_pts.max(0).values
        init_xyz = torch.rand(50000, 3) * (bbox_max - bbox_min) + bbox_min

    print(f"  Initial points: {init_xyz.shape[0]}")

    model = GaussianModel(init_xyz, init_scale=0.1, sh_degree=sh_degree)
    print(f"  SH degree: {sh_degree} ({(sh_degree+1)**2} coefficients per Gaussian)")
    model.setup_optimizer(
        position_lr_init=1.6e-4 * 10,  # scale up since our scene is larger
        position_lr_final=1.6e-6 * 10,
    )

    bg_color = torch.zeros(3, device="cuda")
    eval_cam = cameras[cfg["data"]["eval_camera"]]
    train_cams = [n for n in cameras if n != cfg["data"]["eval_camera"]]
    frame_idx = 0

    out_dir = os.path.join(render_dir, f"direct_v2_{source}")
    os.makedirs(out_dir, exist_ok=True)

    # Compute scene extent for densification threshold
    scene_center = init_xyz.mean(0)
    extent = (init_xyz - scene_center).norm(dim=-1).quantile(0.9).item()
    print(f"  Scene extent: {extent:.2f}")

    # Densification schedule (matching 3DGS)
    densify_from = 500
    densify_until = 15000
    densify_every = 100
    opacity_reset_every = 3000
    grad_threshold = 0.0005  # higher threshold to avoid explosion

    print(f"\n=== Training {source} ({num_steps} steps, densify {densify_from}-{densify_until}) ===")

    for step in tqdm(range(num_steps), desc=f"v2 ({source})"):
        model.optimizer.zero_grad()
        model.update_learning_rate(step, num_steps)

        scales = model.get_scales()
        rotations = model.get_rotations()
        opacity = model.get_opacity()

        cam_name = train_cams[step % len(train_cams)]
        cam = cameras[cam_name]
        gt = dataset.load_frame_image(cam_name, frame_idx).cuda()

        rendered, radii, _, viewspace_points = render_gaussians(
            model.means3D, scales, rotations, opacity,
            model.colors, cam, bg_color, model.sh_degree
        )

        loss_rgb = photometric_loss(rendered, gt, lambda_ssim=0.85)
        # Light scale regularization
        loss_scale = 0.001 * scales.mean()
        loss = loss_rgb + loss_scale

        loss.backward()

        # Accumulate gradients for densification
        if step < densify_until:
            model.accumulate_gradients(radii, viewspace_points)
            # Track max screen-space radius
            visible = radii > 0
            if visible.any():
                model.max_radii2D[visible] = torch.max(
                    model.max_radii2D[visible], radii[visible].float()
                )

        # Densification
        if step >= densify_from and step < densify_until and step % densify_every == 0:
            if model.num_points < max_gaussians:
                n_clone, n_split, n_prune = model.densify_and_prune(
                    grad_threshold=grad_threshold,
                    min_opacity=0.005,
                    max_screen_size=20,
                    extent=extent,
                )
                if n_clone + n_split + n_prune > 0:
                    tqdm.write(f"  [{step}] Clone={n_clone}, Split={n_split}, "
                              f"Prune={n_prune}, Total={model.num_points}")
                # Hard cap: if we overshot, prune aggressively
                if model.num_points > max_gaussians:
                    tqdm.write(f"  [{step}] Over cap ({model.num_points}>{max_gaussians}), stopping densification")
                    densify_until = step  # stop further densification

        # Opacity reset
        if not no_opacity_reset and step > 0 and step % opacity_reset_every == 0 and step < densify_until:
            model.reset_opacity()
            tqdm.write(f"  [{step}] Opacity reset, N={model.num_points}")

        model.optimizer.step()

        # Logging
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
                      f"N={model.num_points}, loss={loss.item():.4f}")

            import cv2
            img = (rendered_novel.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, f"step_{step:05d}.jpg"), img)

    # Final evaluation
    with torch.no_grad():
        scales = model.get_scales()
        rotations = model.get_rotations()
        opacity = model.get_opacity()

        psnrs_train = []
        for cam_name in train_cams[:6]:
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

    print("\n" + "=" * 60)
    print(f"v2 Direct Optimization Results ({source}, frame {frame_idx}):")
    print(f"  Final Gaussians: {model.num_points}")
    print(f"  Train PSNR (6 cams avg): {sum(psnrs_train)/len(psnrs_train):.2f} dB")
    print(f"  Novel PSNR (cam00):      {novel_psnr:.2f} dB")
    print(f"  Train/Novel gap:         {sum(psnrs_train)/len(psnrs_train) - novel_psnr:.2f} dB")
    print(f"  Mean scale: {scales.mean():.4f}")
    print(f"  Mean opacity: {opacity.mean():.4f}")
    print("=" * 60)

    return novel_psnr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="D:/DecodeGaussians/experiments/overfit_coffee_martini/config.yaml")
    parser.add_argument("--source", type=str, default="colmap",
                        choices=["colmap", "stv2", "random"])
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--max-gaussians", type=int, default=200000)
    parser.add_argument("--sh-degree", type=int, default=0)
    parser.add_argument("--no-opacity-reset", action="store_true")
    args = parser.parse_args()
    run_optimization(args.config, source=args.source,
                     num_steps=args.steps, max_gaussians=args.max_gaussians,
                     sh_degree=args.sh_degree, no_opacity_reset=args.no_opacity_reset)
