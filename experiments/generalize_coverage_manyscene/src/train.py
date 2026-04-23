"""
Many-scene training loop with scene swapping.

Extends multi-scene training to support arbitrarily many scenes by loading
scenes_per_batch scenes at a time and swapping them from disk every swap_interval
steps. VRAM/RAM footprint stays constant regardless of total scene count.
"""

import gc
import os
import sys
import glob
import time
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from cameras import build_cameras_from_llff
from dataset import MultiSceneDataset
from model import CanonicalGaussianHead, DeformationHead, compose_gaussians
from renderer import render_gaussians
from losses import photometric_loss, coverage_loss, compute_psnr, scale_regularization, opacity_regularization


def get_vram_gb():
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    return alloc, peak


def save_checkpoint(path, step, canonical_head, deformation_head, optimizer):
    torch.save({
        "step": step,
        "canonical_head": canonical_head.state_dict(),
        "deformation_head": deformation_head.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def find_latest_checkpoint(ckpt_dir):
    """Find the most recent checkpoint. Returns (path, stage, step) or (None, None, None)."""
    checkpoint_patterns = [
        ("stage3_step*.pt", 3),
        ("stage3_final.pt", 3),
        ("stage2_step*.pt", 2),
        ("stage2_final.pt", 2),
        ("stage1_step*.pt", 1),
        ("stage1_final.pt", 1),
    ]

    latest_ckpt = None
    latest_step = -1
    latest_stage = None

    for pattern, stage in checkpoint_patterns:
        if latest_stage is not None and stage < latest_stage:
            continue

        ckpt_paths = glob.glob(os.path.join(ckpt_dir, pattern))

        for ckpt_path in ckpt_paths:
            if "step" in pattern and "*" in pattern:
                basename = os.path.basename(ckpt_path)
                try:
                    step = int(basename.split("step")[1].split(".pt")[0])
                except (IndexError, ValueError):
                    continue
            else:
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                    step = ckpt.get("step", 0)
                except Exception:
                    continue

            if latest_stage is None or stage > latest_stage:
                latest_step = step
                latest_ckpt = ckpt_path
                latest_stage = stage
            elif stage == latest_stage and step > latest_step:
                latest_step = step
                latest_ckpt = ckpt_path

    return latest_ckpt, latest_stage, latest_step


def save_render(rendered, path):
    import cv2
    img = (rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def tv_loss(deltas_t: torch.Tensor, deltas_prev: torch.Tensor) -> torch.Tensor:
    return (deltas_t - deltas_prev).pow(2).mean()


def compute_anchors(dataset, P: int, K: int):
    """
    Compute per-patch positional anchors from a scene's point map.

    Args:
        dataset: A CachedSceneDataset with .points_map [T, H, W, 3] and .num_patches
        P: Number of patches
        K: Number of Gaussians per patch

    Returns:
        init_xyz: [P, 3] one anchor point per patch
        init_xyz_per_gaussian: [P*K, 3] K anchor points per patch
    """
    pts_mean = dataset.points_map.mean(dim=0)  # [H, W, 3]
    H, W = pts_mean.shape[0], pts_mean.shape[1]
    pts_flat = pts_mean.reshape(-1, 3)  # [H*W, 3]

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

    return init_xyz, init_xyz_per_gaussian


class SceneScheduler:
    """
    Manages scene loading/unloading for many-scene training.

    Holds the full list of scene configs. Keeps scenes_per_batch scenes loaded
    at any time, swapping to the next batch from disk every swap_interval steps.
    Cycles through all scenes in order, wrapping around.
    """

    def __init__(self, all_scene_configs: list, scenes_per_batch: int,
                 swap_interval: int, stage_mode: str, normalize_coords: bool,
                 target_w: int, target_h: int):
        self.all_scene_configs = all_scene_configs  # list of dicts with cameras already built
        self.scenes_per_batch = scenes_per_batch
        self.swap_interval = swap_interval
        self.stage_mode = stage_mode
        self.normalize_coords = normalize_coords
        self.n_total = len(all_scene_configs)
        self.batch_start_idx = 0  # index into all_scene_configs of current batch start
        self._active_dataset = None

        print(f"\n{'='*60}")
        print(f"SceneScheduler: {self.n_total} total scenes, {scenes_per_batch} per batch")
        print(f"  Swap every {swap_interval} steps | Stage mode: {stage_mode}")
        print(f"{'='*60}")

        # Load initial batch
        self._load_batch(0)

    def _load_batch(self, start_idx: int):
        """Load a batch of scenes starting at start_idx (wraps around)."""
        # Free previous dataset
        if self._active_dataset is not None:
            del self._active_dataset
            gc.collect()
            torch.cuda.empty_cache()

        self.batch_start_idx = start_idx % self.n_total
        batch_indices = [(self.batch_start_idx + i) % self.n_total
                         for i in range(self.scenes_per_batch)]
        batch_configs = [self.all_scene_configs[i] for i in batch_indices]

        scene_names = [c["name"] for c in batch_configs]
        print(f"\n  [SceneScheduler] Loading batch: {scene_names}")

        self._active_dataset = MultiSceneDataset(
            batch_configs,
            normalize_coords=self.normalize_coords,
        )

    def maybe_swap(self, step: int) -> bool:
        """Check if a swap is needed at this step. Returns True if a swap occurred."""
        if step > 0 and step % self.swap_interval == 0:
            next_start = self.batch_start_idx + self.scenes_per_batch
            self._load_batch(next_start)
            return True
        return False

    def get_active_dataset(self) -> MultiSceneDataset:
        return self._active_dataset

    def get_active_scene_names(self) -> list:
        return [s["name"] for s in self._active_dataset.scenes]

    def total_steps_for_stage(self, base_steps: int) -> int:
        """
        For all_scenes_per_stage mode: scale step count so every scene is trained
        for at least base_steps steps.
        """
        if self.stage_mode == "all_scenes_per_stage":
            import math
            n_batches = math.ceil(self.n_total / self.scenes_per_batch)
            steps_per_rotation = n_batches * self.swap_interval
            n_rotations = math.ceil(base_steps / self.swap_interval)
            return n_rotations * steps_per_rotation
        return base_steps


def train(config_path: str, resume_stage: int = 1, restart_latest: bool = False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = cfg["logging"]["ckpt_dir"]
    render_dir = cfg["logging"]["render_dir"]
    log_dir = cfg["logging"]["log_dir"]

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(render_dir, "stage1"), exist_ok=True)
    os.makedirs(os.path.join(render_dir, "stage2"), exist_ok=True)
    os.makedirs(os.path.join(render_dir, "stage3"), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    target_w = cfg["rendering"]["image_width"]
    target_h = cfg["rendering"]["image_height"]

    # Build all scene configs with cameras up front (cheap — just parses poses)
    all_scene_configs = []
    for scene_cfg in cfg["scenes"]:
        scene_dir = os.path.dirname(scene_cfg["poses_bounds_path"])
        cam_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(scene_dir, "cam*.mp4"))])
        cameras = build_cameras_from_llff(
            scene_cfg["poses_bounds_path"], target_w, target_h, cam_files
        )
        all_scene_configs.append({
            "name": scene_cfg["name"],
            "cache_dir": scene_cfg["cache_dir"],
            "cameras": cameras,
            "input_camera": scene_cfg.get("input_camera", "cam01"),
            "eval_camera": scene_cfg.get("eval_camera", "cam00"),
            "weight": scene_cfg.get("weight", 1.0),
        })

    scenes_per_batch = cfg["training"].get("scenes_per_batch", 2)
    swap_interval = cfg["training"].get("swap_interval", 500)
    stage_mode = cfg["training"].get("stage_mode", "global_steps")
    normalize_coords = cfg["normalization"].get("enabled", False)

    scheduler = SceneScheduler(
        all_scene_configs=all_scene_configs,
        scenes_per_batch=scenes_per_batch,
        swap_interval=swap_interval,
        stage_mode=stage_mode,
        normalize_coords=normalize_coords,
        target_w=target_w,
        target_h=target_h,
    )

    # Derive model dimensions from first loaded scene
    first_dataset = scheduler.get_active_dataset().scenes[0]["dataset"]
    P = first_dataset.num_patches
    K = cfg["model"]["num_gaussians_per_patch"]

    # Compute initial anchors from first scene
    init_xyz, init_xyz_per_gaussian = compute_anchors(first_dataset, P, K)

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

    print(f"\n{'='*60}")
    print(f"Model Architecture:")
    print(f"  Canonical Head: {sum(p.numel() for p in canonical_head.parameters())/1e6:.1f}M params")
    print(f"  Deformation Head: {sum(p.numel() for p in deformation_head.parameters())/1e6:.1f}M params")
    print(f"  Total Gaussians: {P * K}")
    print(f"{'='*60}\n")

    bg_color = torch.tensor(cfg["rendering"]["bg_color"], device="cuda")
    sh_degree = cfg["model"]["sh_degree"]
    grad_clip = cfg["training"].get("grad_clip_max_norm", 0.0)

    # Handle checkpoint loading
    restart_from_step = None
    restart_from_stage = None
    ckpt = None

    if restart_latest:
        latest_ckpt, latest_stage, latest_step = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt is not None:
            print(f"\n{'='*60}")
            print(f"RESTART MODE: Loading latest checkpoint")
            print(f"  Checkpoint: {os.path.basename(latest_ckpt)}")
            print(f"  Stage: {latest_stage}, Step: {latest_step}")
            print(f"{'='*60}\n")
            ckpt = torch.load(latest_ckpt, weights_only=False)
            canonical_head.load_state_dict(ckpt["canonical_head"])
            deformation_head.load_state_dict(ckpt["deformation_head"])
            restart_from_step = latest_step
            restart_from_stage = latest_stage
        else:
            print(f"\nWARNING: --restart-latest specified but no checkpoints found. Starting fresh.\n")
    elif resume_stage > 1:
        ckpt_path = os.path.join(ckpt_dir, "stage1_final.pt")
        if resume_stage >= 3:
            ckpt_path = os.path.join(ckpt_dir, "stage2_final.pt")
        print(f"\nResuming from {ckpt_path}\n")
        ckpt = torch.load(ckpt_path, weights_only=False)
        canonical_head.load_state_dict(ckpt["canonical_head"])
        deformation_head.load_state_dict(ckpt["deformation_head"])

    writer = SummaryWriter(log_dir)
    start_time = time.time()

    def do_swap_and_update_anchors(step):
        """Perform a scene swap and update model anchor buffers."""
        swapped = scheduler.maybe_swap(step)
        if swapped:
            active_dataset = scheduler.get_active_dataset()
            ref_dataset = active_dataset.scenes[0]["dataset"]
            _, new_xyz_per_gaussian = compute_anchors(ref_dataset, P, K)
            if canonical_head.xyz_anchor is not None:
                canonical_head.xyz_anchor.copy_(new_xyz_per_gaussian.cuda())
            scene_names = scheduler.get_active_scene_names()
            tqdm.write(f"  [Step {step}] Swapped to scenes: {scene_names}")
            writer.add_text("scene_swap", str(scene_names), step)
        return swapped

    # ===== STAGE 1: Canonical Head Only =====
    should_skip_stage1 = (resume_stage > 1) or (restart_from_stage and restart_from_stage > 1)

    if should_skip_stage1:
        print(f"\n  Skipping Stage 1")
    else:
        print("\n" + "=" * 60)
        print("STAGE 1: Training Canonical Head (Many-Scene)")
        print("=" * 60)

        canonical_head.train()
        deformation_head.eval()

        s1_steps = scheduler.total_steps_for_stage(cfg["training"]["stage1_steps"])
        s1_lr = cfg["training"]["stage1_lr"]
        s1_batch_frames = cfg["training"]["stage1_batch_frames"]
        s1_supervision_cams = cfg["training"]["stage1_supervision_cams"]
        s1_weights = cfg["training"].get("stage1_loss_weights", {})

        optimizer_s1 = optim.AdamW(canonical_head.parameters(), lr=s1_lr)
        scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=s1_steps)

        start_step_s1 = 0
        if restart_from_stage == 1 and restart_from_step is not None:
            print(f"  Continuing Stage 1 from step {restart_from_step}")
            start_step_s1 = restart_from_step
            if ckpt and "optimizer" in ckpt:
                optimizer_s1.load_state_dict(ckpt["optimizer"])
            for _ in range(restart_from_step):
                scheduler_s1.step()

        torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(range(start_step_s1, s1_steps), desc="Stage 1 (Canonical)")
        for step in pbar:
            do_swap_and_update_anchors(step)

            optimizer_s1.zero_grad()

            scene_batches = scheduler.get_active_dataset().sample_multi_scene_batch(
                s1_batch_frames, s1_supervision_cams
            )

            total_psnr = 0.0
            total_renders = 0
            total_loss = torch.tensor(0.0, device="cuda")
            last_rendered = None

            for scene_batch in scene_batches:
                scene_weight = scene_batch["scene_weight"]
                scene_cameras = scene_batch["cameras"]

                tokens_mean = scene_batch["tokens_mean"].cuda()
                canonical = canonical_head(tokens_mean)

                scene_loss = torch.tensor(0.0, device="cuda")

                for i, fidx in enumerate(scene_batch["frame_indices"]):
                    means3D, scales, rotations, opacity, shs = compose_gaussians(canonical)

                    for cam_name in scene_batch["cam_names"]:
                        cam = scene_cameras[cam_name]
                        gt = scene_batch["gt_images"][cam_name][i].cuda()

                        rendered, radii, depth, _ = render_gaussians(
                            means3D, scales, rotations, opacity, shs, cam, bg_color, sh_degree
                        )

                        loss = photometric_loss(rendered, gt, lambda_ssim=0.2)
                        scene_loss = scene_loss + loss
                        total_psnr += compute_psnr(rendered, gt)
                        total_renders += 1

                        last_rendered = rendered.detach()
                        del radii, depth, gt

                    coverage_w = s1_weights.get("coverage", 0.0)
                    if coverage_w > 0:
                        pts_map = scene_batch["points_map_frames"][i].cuda()
                        scene_loss = scene_loss + coverage_w * coverage_loss(means3D, pts_map)

                num_scene_renders = len(scene_batch["frame_indices"]) * len(scene_batch["cam_names"])
                scene_loss = scene_loss / max(num_scene_renders, 1)
                total_loss = total_loss + scene_loss * scene_weight / len(scene_batches)

            total_loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(canonical_head.parameters(), grad_clip)
            optimizer_s1.step()
            scheduler_s1.step()

            avg_loss = total_loss.item()
            avg_psnr = total_psnr / max(total_renders, 1)
            vram_alloc, _ = get_vram_gb()
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "psnr": f"{avg_psnr:.1f}", "vram": f"{vram_alloc:.1f}GB"})

            writer.add_scalar("stage1/loss", avg_loss, step)
            writer.add_scalar("stage1/psnr", avg_psnr, step)

            if step % cfg["logging"]["save_render_every"] == 0 and last_rendered is not None:
                save_render(last_rendered, os.path.join(render_dir, "stage1", f"step_{step:05d}.jpg"))

            if step % cfg["training"]["save_ckpt_every"] == 0 and step > 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"stage1_step{step}.pt"),
                    step, canonical_head, deformation_head, optimizer_s1
                )

        save_checkpoint(
            os.path.join(ckpt_dir, "stage1_final.pt"),
            s1_steps, canonical_head, deformation_head, optimizer_s1
        )
        print(f"\nStage 1 complete. Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        torch.cuda.empty_cache()

    # ===== STAGE 2: Deformation Head Only =====
    should_skip_stage2 = (resume_stage > 2) or (restart_from_stage and restart_from_stage > 2)

    if should_skip_stage2:
        print(f"\n  Skipping Stage 2")
    else:
        print("\n" + "=" * 60)
        print("STAGE 2: Training Deformation Head (Many-Scene)")
        print("=" * 60)

        canonical_head.eval()
        for p in canonical_head.parameters():
            p.requires_grad = False

        deformation_head.train()

        s2_steps = scheduler.total_steps_for_stage(cfg["training"]["stage2_steps"])
        s2_lr = cfg["training"]["stage2_lr"]
        s2_batch_frames = cfg["training"]["stage2_batch_frames"]
        s2_supervision_cams = cfg["training"]["stage2_supervision_cams"]
        s2_weights = cfg["training"]["stage2_loss_weights"]

        optimizer_s2 = optim.AdamW(deformation_head.parameters(), lr=s2_lr)
        scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=s2_steps)

        start_step_s2 = 0
        if restart_from_stage == 2 and restart_from_step is not None:
            print(f"  Continuing Stage 2 from step {restart_from_step}")
            start_step_s2 = restart_from_step
            if ckpt and "optimizer" in ckpt:
                optimizer_s2.load_state_dict(ckpt["optimizer"])
            for _ in range(restart_from_step):
                scheduler_s2.step()

        torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(range(start_step_s2, s2_steps), desc="Stage 2 (Deformation)")
        for step in pbar:
            do_swap_and_update_anchors(step)

            optimizer_s2.zero_grad()

            scene_batches = scheduler.get_active_dataset().sample_multi_scene_batch(
                s2_batch_frames, s2_supervision_cams
            )

            total_psnr = 0.0
            total_renders = 0
            total_loss = torch.tensor(0.0, device="cuda")
            per_scene_losses = {}
            delta_magnitudes = []
            last_rendered = None

            for scene_batch in scene_batches:
                scene_name = scene_batch["scene_name"]
                scene_weight = scene_batch["scene_weight"]
                scene_cameras = scene_batch["cameras"]

                tokens_mean = scene_batch["tokens_mean"].cuda()

                with torch.no_grad():
                    canonical = canonical_head(tokens_mean)

                scene_loss = torch.tensor(0.0, device="cuda")
                prev_all_deltas = None

                for i, fidx in enumerate(scene_batch["frame_indices"]):
                    tokens_t = scene_batch["tokens_frames"][i].cuda()
                    deltas = deformation_head(tokens_t)
                    delta_magnitudes.append(deltas["dxyz"].abs().mean().item())

                    means3D, scales, rotations, opacity, shs = compose_gaussians(canonical, deltas)

                    for cam_name in scene_batch["cam_names"]:
                        cam = scene_cameras[cam_name]
                        gt = scene_batch["gt_images"][cam_name][i].cuda()

                        rendered, radii, depth, _ = render_gaussians(
                            means3D, scales, rotations, opacity, shs, cam, bg_color, sh_degree
                        )

                        loss = photometric_loss(rendered, gt, lambda_ssim=0.2)
                        scene_loss = scene_loss + loss
                        total_psnr += compute_psnr(rendered, gt)
                        total_renders += 1

                        last_rendered = rendered.detach()
                        del radii, depth, gt

                    if prev_all_deltas is not None:
                        tv_weight = s2_weights.get("tv", 0.01)
                        scene_loss = scene_loss + tv_weight * tv_loss(deltas["all_deltas"], prev_all_deltas)
                    prev_all_deltas = deltas["all_deltas"].detach()

                num_scene_renders = len(scene_batch["frame_indices"]) * len(scene_batch["cam_names"])
                scene_loss = scene_loss / max(num_scene_renders, 1)
                per_scene_losses[scene_name] = scene_loss.item()
                total_loss = total_loss + scene_loss * scene_weight / len(scene_batches)

            total_loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(deformation_head.parameters(), grad_clip)
            optimizer_s2.step()
            scheduler_s2.step()

            avg_loss = total_loss.item()
            avg_psnr = total_psnr / max(total_renders, 1)
            avg_delta_mag = sum(delta_magnitudes) / max(len(delta_magnitudes), 1)
            vram_alloc, _ = get_vram_gb()
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}", "psnr": f"{avg_psnr:.1f}",
                "Δ": f"{avg_delta_mag:.4f}", "vram": f"{vram_alloc:.1f}GB"
            })

            writer.add_scalar("stage2/loss", avg_loss, step)
            writer.add_scalar("stage2/psnr", avg_psnr, step)
            writer.add_scalar("stage2/delta_magnitude", avg_delta_mag, step)
            for sname, sloss in per_scene_losses.items():
                writer.add_scalar(f"stage2/loss_{sname}", sloss, step)

            if step % cfg["logging"]["save_render_every"] == 0 and last_rendered is not None:
                save_render(last_rendered, os.path.join(render_dir, "stage2", f"step_{step:05d}.jpg"))

            if step % cfg["training"]["save_ckpt_every"] == 0 and step > 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"stage2_step{step}.pt"),
                    step, canonical_head, deformation_head, optimizer_s2
                )

            if step % cfg["training"]["eval_every"] == 0:
                with torch.no_grad():
                    eval_psnrs = {}
                    for scene_info in scheduler.get_active_dataset().scenes:
                        sname = scene_info["name"]
                        dataset = scene_info["dataset"]
                        cameras = scene_info["cameras"]
                        eval_cam_name = scene_info["eval_camera"]

                        tokens_mean_eval = dataset.get_tokens_mean().cuda()
                        canonical_eval = canonical_head(tokens_mean_eval)
                        tokens_t_eval = dataset.get_tokens_frame(0).cuda()
                        deltas_eval = deformation_head(tokens_t_eval)
                        m3d, sc, rot, op, sh = compose_gaussians(canonical_eval, deltas_eval)

                        eval_cam = cameras[eval_cam_name]
                        gt_novel = dataset.load_frame_image(eval_cam_name, 0).cuda()
                        rendered_novel, _, _, _ = render_gaussians(m3d, sc, rot, op, sh, eval_cam, bg_color, sh_degree)
                        novel_psnr = compute_psnr(rendered_novel, gt_novel)
                        eval_psnrs[sname] = novel_psnr
                        writer.add_scalar(f"stage2/eval_psnr_{sname}", novel_psnr, step)

                    eval_str = ", ".join([f"{n}={v:.1f}" for n, v in eval_psnrs.items()])
                    tqdm.write(f"  Step {step}: {eval_str} dB")

        save_checkpoint(
            os.path.join(ckpt_dir, "stage2_final.pt"),
            s2_steps, canonical_head, deformation_head, optimizer_s2
        )
        print(f"\nStage 2 complete. Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        torch.cuda.empty_cache()

    # ===== STAGE 3: Joint Fine-tuning =====
    should_skip_stage3 = (restart_from_stage and restart_from_stage > 3)

    if not should_skip_stage3 and cfg["training"].get("stage3_steps", 0) > 0:
        print("\n" + "=" * 60)
        print("STAGE 3: Joint Fine-tuning (Many-Scene)")
        print("=" * 60)

        canonical_head.train()
        for p in canonical_head.parameters():
            p.requires_grad = True
        deformation_head.train()

        s3_steps = scheduler.total_steps_for_stage(cfg["training"]["stage3_steps"])
        s3_lr = cfg["training"]["stage3_lr"]
        s3_batch_frames = cfg["training"]["stage3_batch_frames"]
        s3_supervision_cams = cfg["training"]["stage3_supervision_cams"]
        s3_weights = cfg["training"]["stage3_loss_weights"]

        optimizer_s3 = optim.AdamW(
            list(canonical_head.parameters()) + list(deformation_head.parameters()),
            lr=s3_lr
        )
        scheduler_s3 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s3, T_max=s3_steps)

        start_step_s3 = 0
        if restart_from_stage == 3 and restart_from_step is not None:
            print(f"  Continuing Stage 3 from step {restart_from_step}")
            start_step_s3 = restart_from_step
            if ckpt and "optimizer" in ckpt:
                optimizer_s3.load_state_dict(ckpt["optimizer"])
            for _ in range(restart_from_step):
                scheduler_s3.step()

        torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(range(start_step_s3, s3_steps), desc="Stage 3 (Joint)")
        for step in pbar:
            do_swap_and_update_anchors(step)

            optimizer_s3.zero_grad()

            scene_batches = scheduler.get_active_dataset().sample_multi_scene_batch(
                s3_batch_frames, s3_supervision_cams
            )

            total_psnr = 0.0
            total_renders = 0
            total_loss = torch.tensor(0.0, device="cuda")
            delta_magnitudes = []
            last_rendered = None

            for scene_batch in scene_batches:
                scene_weight = scene_batch["scene_weight"]
                scene_cameras = scene_batch["cameras"]

                tokens_mean = scene_batch["tokens_mean"].cuda()
                canonical = canonical_head(tokens_mean)

                scene_loss = torch.tensor(0.0, device="cuda")
                prev_all_deltas = None

                for i, fidx in enumerate(scene_batch["frame_indices"]):
                    tokens_t = scene_batch["tokens_frames"][i].cuda()
                    deltas = deformation_head(tokens_t)
                    delta_magnitudes.append(deltas["dxyz"].abs().mean().item())

                    means3D, scales, rotations, opacity, shs = compose_gaussians(canonical, deltas)

                    for cam_name in scene_batch["cam_names"]:
                        cam = scene_cameras[cam_name]
                        gt = scene_batch["gt_images"][cam_name][i].cuda()

                        rendered, radii, depth, _ = render_gaussians(
                            means3D, scales, rotations, opacity, shs, cam, bg_color, sh_degree
                        )

                        loss = photometric_loss(rendered, gt, lambda_ssim=0.2)
                        scene_loss = scene_loss + loss
                        total_psnr += compute_psnr(rendered, gt)
                        total_renders += 1

                        last_rendered = rendered.detach()
                        del radii, depth, gt

                    if prev_all_deltas is not None:
                        tv_weight = s3_weights.get("tv", 0.01)
                        scene_loss = scene_loss + tv_weight * tv_loss(deltas["all_deltas"], prev_all_deltas)
                    prev_all_deltas = deltas["all_deltas"].detach()

                num_scene_renders = len(scene_batch["frame_indices"]) * len(scene_batch["cam_names"])
                scene_loss = scene_loss / max(num_scene_renders, 1)
                total_loss = total_loss + scene_loss * scene_weight / len(scene_batches)

            total_loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(canonical_head.parameters()) + list(deformation_head.parameters()),
                    grad_clip
                )
            optimizer_s3.step()
            scheduler_s3.step()

            avg_loss = total_loss.item()
            avg_psnr = total_psnr / max(total_renders, 1)
            avg_delta_mag = sum(delta_magnitudes) / max(len(delta_magnitudes), 1)
            vram_alloc, _ = get_vram_gb()
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}", "psnr": f"{avg_psnr:.1f}",
                "Δ": f"{avg_delta_mag:.4f}", "vram": f"{vram_alloc:.1f}GB"
            })

            writer.add_scalar("stage3/loss", avg_loss, step)
            writer.add_scalar("stage3/psnr", avg_psnr, step)
            writer.add_scalar("stage3/delta_magnitude", avg_delta_mag, step)

            if step % cfg["logging"]["save_render_every"] == 0 and last_rendered is not None:
                save_render(last_rendered, os.path.join(render_dir, "stage3", f"step_{step:05d}.jpg"))

            if step % cfg["training"]["save_ckpt_every"] == 0 and step > 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"stage3_step{step}.pt"),
                    step, canonical_head, deformation_head, optimizer_s3
                )

        save_checkpoint(
            os.path.join(ckpt_dir, "stage3_final.pt"),
            s3_steps, canonical_head, deformation_head, optimizer_s3
        )
        print(f"\nStage 3 complete. Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete! Total time: {elapsed/3600:.1f} hours")
    print(f"{'='*60}\n")

    writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume-stage", type=int, default=1,
                        help="Resume from stage N (1=full, 2=skip stage1, 3=skip stage1+2)")
    parser.add_argument("--restart-latest", action="store_true",
                        help="Restart from latest checkpoint")
    args = parser.parse_args()

    train(args.config, resume_stage=args.resume_stage, restart_latest=args.restart_latest)
