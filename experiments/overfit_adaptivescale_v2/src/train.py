"""Adaptive scale (per-patch NN distance init) using shared engine infrastructure.

CanonicalGaussianHead (basic MLP) with 3DGS-style scale anchors,
scale_anneal_target=0.5, and all shared engine fixes:
- float32 tokens, LLFF alignment, always include input_camera.
"""

import math
import os
import sys
import glob
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from cameras import build_cameras_from_llff
from model import CanonicalGaussianHead, CrossAttentionDeformationHead, compose_gaussians
from src.config import load_engine_config
from src.metrics import MetricAccumulator
from src.checkpoint import CheckpointManager
from src.engine import _apply_phase_freezing, _build_optimizer, _build_scheduler
from src.compile_utils import compile_modules
from losses import photometric_loss, compute_psnr, scale_regularization, opacity_regularization


def get_vram_gb():
    return torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9


def save_render(rendered, path):
    import cv2
    img = (rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def align_points_to_llff(points_map: torch.Tensor, cache_dir: str, cameras: dict,
                          input_camera: str, scene_path: str) -> torch.Tensor:
    """Transform STV2 points from VGGT coordinate system to LLFF world space."""
    import numpy as np_
    poses_path = os.path.join(cache_dir, "poses.pt")
    if not os.path.exists(poses_path):
        print("  [Align] No poses.pt found, skipping alignment")
        return points_map

    stv2_poses = torch.load(poses_path, weights_only=True).float()
    stv2_c2w_0 = stv2_poses[0]
    stv2_inv = torch.inverse(stv2_c2w_0)

    cam01 = cameras[input_camera]
    w2v_llff = cam01.world_view_transform.T.cpu()
    c2w_llff = torch.inverse(w2v_llff)
    R_c2w = c2w_llff[:3, :3]
    t_c2w = c2w_llff[:3, 3]

    pts_flat = points_map.reshape(-1, 3)
    pts_cam = pts_flat @ stv2_inv[:3, :3].T + stv2_inv[:3, 3]
    z_stv2 = pts_cam[:, 2]
    z_positive = z_stv2[z_stv2 > 0.01]
    z_sorted = z_positive.sort().values
    z_near_stv2 = z_sorted[len(z_sorted) // 20]

    near_llff = 6.84
    pb_path = os.path.join(scene_path, "poses_bounds.npy")
    if os.path.exists(pb_path):
        pb = np_.load(pb_path)
        cam_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(scene_path, "cam*.mp4"))])
        cam_idx = [f.replace(".mp4", "") for f in cam_files].index(input_camera)
        near_llff = float(pb[cam_idx, -2])

    scale = near_llff / float(z_near_stv2)
    print(f"  [Align] STV2 near: {z_near_stv2:.3f}, LLFF near: {near_llff:.3f}, scale: {scale:.2f}")

    R_stv2_inv = stv2_inv[:3, :3]
    original_shape = points_map.shape
    pts = points_map.reshape(-1, 3)
    pts_scaled = scale * (pts @ R_stv2_inv.T + stv2_inv[:3, 3])
    pts_aligned = pts_scaled @ R_c2w.T + t_c2w
    return pts_aligned.reshape(original_shape)


def build_heads(cfg, tokens, points_map):
    """Build model heads with adaptive per-patch NN scale initialization."""
    model_cfg = cfg.get("model", {})
    can_cfg = model_cfg.get("canonical", {})
    defo_cfg = model_cfg.get("deformation", {})

    P = tokens.shape[1]
    K = model_cfg.get("num_gaussians_per_patch", 128)

    pts_mean = points_map.mean(dim=0)  # [H, W, 3]
    H, W = pts_mean.shape[0], pts_mean.shape[1]
    pts_flat = pts_mean.reshape(-1, 3)

    # Compute patch grid
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
    spread = 0.05 * res_scale
    ssim_win_size = max(11, round(11 / res_scale)) | 1

    # Adaptive scale init: per-patch nearest-neighbor distance (3DGS approach)
    pts_per_patch = init_xyz.cpu()
    dists = torch.cdist(pts_per_patch, pts_per_patch)
    dists.fill_diagonal_(float("inf"))
    nn_dist = dists.min(dim=1).values.clamp(min=1e-6)
    log_nn = torch.log(nn_dist)
    log_nn_pk = log_nn.unsqueeze(1).expand(-1, K).reshape(P * K)
    scale_anchor = log_nn_pk.unsqueeze(1).expand(-1, 3).contiguous()  # [P*K, 3]

    print(f"  Patch grid: {grid_h}x{grid_w}, P={P}, K={K}, total Gaussians={P*K}")
    print(f"  Resolution scaling: W={W}, res_scale={res_scale:.2f}")
    print(f"  Scale anchor (log NN dist): min={log_nn.min():.3f}, mean={log_nn.mean():.3f}, max={log_nn.max():.3f}")

    canonical_head = CanonicalGaussianHead(
        dim_in=can_cfg.get("dim_in", 2048),
        dim_hidden=can_cfg.get("dim_hidden", 1024),
        sh_degree=can_cfg.get("sh_degree", 0),
        num_gaussians_per_patch=K,
        init_xyz=init_xyz,
        init_xyz_per_gaussian=init_xyz_per_gaussian,
        spread=spread,
        scale_anchor=scale_anchor,
    )

    deformation_head = CrossAttentionDeformationHead(
        dim_canonical=defo_cfg.get("dim_canonical", 1024),
        dim_tokens=defo_cfg.get("dim_tokens", 2048),
        dim_hidden=defo_cfg.get("dim_hidden", 256),
        n_heads=defo_cfg.get("n_heads", 8),
        n_layers=defo_cfg.get("n_layers", 3),
        sh_degree=can_cfg.get("sh_degree", 0),
        num_gaussians_per_patch=K,
        max_displacement=defo_cfg.get("max_displacement", 2.0),
    )

    print(f"  CanonicalHead: {sum(p.numel() for p in canonical_head.parameters())/1e6:.1f}M params")
    print(f"  CrossAttentionDeformHead: {sum(p.numel() for p in deformation_head.parameters())/1e6:.1f}M params")

    return canonical_head, deformation_head, res_scale, ssim_win_size


def train(config_path: str, start_phase: int = 0):
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    engine_config = load_engine_config(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = engine_config.engine.amp and device == "cuda"
    use_compile = engine_config.engine.compile and device == "cuda"

    experiment_dir = os.path.dirname(os.path.abspath(config_path))
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    render_dir = os.path.join(experiment_dir, "renders")
    log_dir = os.path.join(experiment_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    scene_cfg = engine_config.data.scenes[0]
    cache_dir = scene_cfg.precomputed

    tokens = torch.load(os.path.join(cache_dir, "tokens.pt"), map_location="cpu", weights_only=True).float()
    points_map = torch.load(os.path.join(cache_dir, "points_map.pt"), map_location="cpu", weights_only=True)
    tokens_mean = tokens.mean(dim=0)  # [P, D]

    scene_path = scene_cfg.path
    target_w, target_h = engine_config.data.resolution
    cam_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(scene_path, "cam*.mp4"))])
    cameras = build_cameras_from_llff(
        os.path.join(scene_path, "poses_bounds.npy"), target_w, target_h, cam_files
    )

    input_cam_name = engine_config.data.input_camera or "cam01"
    points_map = align_points_to_llff(points_map.float(), cache_dir, cameras, input_cam_name, scene_path)

    tokens = tokens.to(device)
    tokens_mean = tokens_mean.to(device)
    points_map_cpu = points_map
    points_map = points_map.to(device)

    canonical_head, deformation_head, res_scale, ssim_win_size = build_heads(
        raw_cfg, tokens.cpu(), points_map_cpu
    )

    modules = {
        "canonical_head": canonical_head,
        "deformation_head": deformation_head,
    }
    for mod in modules.values():
        mod.to(device)

    if use_compile:
        modules = compile_modules(modules, compile_enabled=True,
                                  compile_mode=engine_config.engine.compile_mode,
                                  skip_names={"renderer"})

    ckpt_mgr = CheckpointManager(ckpt_dir)
    metrics_acc = MetricAccumulator()

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
    except ImportError:
        print("[engine] TensorBoard not available.")

    scaler = None  # bfloat16/float32 don't need GradScaler

    from renderer import render_gaussians

    bg_color = torch.zeros(3, device=device)
    sh_degree = raw_cfg["model"]["canonical"].get("sh_degree", 0)
    scale_anneal_target = raw_cfg.get("model", {}).get("scale_anneal_target", 0.5)
    supervision_cams = engine_config.data.supervision_cams or 4
    batch_frames = engine_config.data.batch_frames or 1
    eval_cam_name = engine_config.data.eval_camera or "cam00"
    num_frames = tokens.shape[0]

    grad_clip = engine_config.engine.grad_clip_max_norm or 1.0
    log_every = engine_config.engine.log_every_n or 100
    save_every = engine_config.engine.save_every_n or 2000
    eval_every = engine_config.engine.eval_every_n or 1000

    all_cams = sorted(cameras.keys())
    train_cams = [c for c in all_cams if c != eval_cam_name]

    print(f"\n[engine] Device: {device}, AMP: {use_amp}, Compile: {use_compile}")
    print(f"[engine] Cameras: {len(all_cams)} total, {len(train_cams)} train, eval={eval_cam_name}")
    print(f"[engine] Tokens: {tokens.shape}, Points: {points_map.shape}")
    print(f"[engine] scale_anneal_target: {scale_anneal_target}")

    import cv2
    import numpy as np
    import random

    print("[engine] Pre-caching GT frames into CPU pinned memory...")
    gt_cache = {}
    cache_start = time.time()
    for cam_name in all_cams:
        gt_cache[cam_name] = {}
        frames_dir = os.path.join(cache_dir, "frames", cam_name)
        if not os.path.exists(frames_dir):
            continue
        for fidx in range(num_frames):
            jpg_path = os.path.join(frames_dir, f"{fidx:06d}.jpg")
            bgr = cv2.imread(jpg_path)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1)
                gt_cache[cam_name][fidx] = img.pin_memory()
    cache_elapsed = time.time() - cache_start
    total_cached = sum(len(v) for v in gt_cache.values())
    print(f"[engine] Cached {total_cached} frames in {cache_elapsed:.1f}s (CPU pinned)")

    def load_gt_frame(cam_name, frame_idx):
        return gt_cache[cam_name][frame_idx].to(device, non_blocking=True)

    input_cam = input_cam_name

    def sample_batch():
        frame_indices = [random.randint(0, num_frames - 1) for _ in range(batch_frames)]
        other_cams = [c for c in train_cams if c != input_cam]
        extra = random.sample(other_cams, min(supervision_cams - 1, len(other_cams)))
        cam_names = [input_cam] + extra
        return frame_indices, cam_names

    start_time = time.time()
    global_step = 0
    phase_times = {}

    for phase_idx, phase in enumerate(engine_config.phases):
        if phase_idx < start_phase:
            global_step += phase.steps
            print(f"\n[engine] Skipping Phase {phase_idx}: {phase.name} (already completed)")
            continue

        phase_start = time.time()
        phase_name = phase.name

        print(f"\n{'='*60}")
        print(f"Phase {phase_idx}: {phase_name} ({phase.steps} steps)")
        print(f"  Trainable: {phase.trainable}")
        print(f"{'='*60}")

        if phase_idx > 0 and phase_idx == start_phase:
            prev_ckpt_dir = os.path.join(ckpt_dir, f"phase_{phase_idx-1}_{engine_config.phases[phase_idx-1].name}")
            latest_pt = os.path.join(prev_ckpt_dir, "latest.pt")
            if os.path.exists(latest_pt):
                print(f"[engine] Loading weights from Phase {phase_idx-1} checkpoint...")
                ckpt = torch.load(latest_pt, map_location=device, weights_only=False)
                for name, mod in modules.items():
                    if name in ckpt.get("modules", {}):
                        mod.load_state_dict(ckpt["modules"][name])
                print(f"[engine] Loaded.")

        _apply_phase_freezing(modules, phase.trainable)

        optimizer = _build_optimizer(modules, phase)
        scheduler = _build_scheduler(optimizer, phase)

        loss_weights = {name: lc.weight for name, lc in phase.losses.items()}

        phase_render_dir = os.path.join(render_dir, phase_name)
        os.makedirs(phase_render_dir, exist_ok=True)

        torch.cuda.reset_peak_memory_stats()
        metrics_acc = MetricAccumulator()

        is_canonical_only = "deformation_head" not in phase.trainable and phase_name == "canonical"

        pbar = tqdm(range(phase.steps), desc=f"Phase: {phase_name}")
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)

            frame_indices, cam_names = sample_batch()

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    canonical = modules["canonical_head"](tokens_mean)
            else:
                canonical = modules["canonical_head"](tokens_mean)

            total_loss = torch.tensor(0.0, device=device)
            total_psnr = 0.0
            num_renders = 0

            if is_canonical_only:
                means3D, scales, rotations, opacity, shs = compose_gaussians(
                    canonical, scale_factor=scale_anneal_target
                )
                for cam_name in cam_names:
                    cam = cameras[cam_name]
                    for fidx in frame_indices:
                        gt = load_gt_frame(cam_name, fidx)
                        rendered, _, _, _ = render_gaussians(
                            means3D.float(), scales.float(), rotations.float(),
                            opacity.float(), shs.float(),
                            cam, bg_color, sh_degree
                        )
                        total_loss = total_loss + loss_weights.get("rgb", 1.0) * photometric_loss(
                            rendered, gt, lambda_ssim=loss_weights.get("ssim", 0.85),
                            win_size=ssim_win_size
                        )
                        total_psnr += compute_psnr(rendered, gt)
                        num_renders += 1

                if loss_weights.get("scale_reg", 0) > 0:
                    total_loss = total_loss + loss_weights["scale_reg"] * res_scale * scale_regularization(scales.float())
                if loss_weights.get("opacity_reg", 0) > 0:
                    total_loss = total_loss + loss_weights["opacity_reg"] * opacity_regularization(opacity.float())

            else:
                if "canonical_head" not in phase.trainable:
                    with torch.no_grad():
                        if use_amp:
                            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                canonical = modules["canonical_head"](tokens_mean)
                        else:
                            canonical = modules["canonical_head"](tokens_mean)

                prev_deltas_flat = None
                for fidx in frame_indices:
                    tokens_t = tokens[fidx]
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            deltas = modules["deformation_head"](canonical["hidden"], tokens_t)
                    else:
                        deltas = modules["deformation_head"](canonical["hidden"], tokens_t)

                    means3D, scales, rotations, opacity, shs = compose_gaussians(
                        canonical, deltas, scale_factor=scale_anneal_target
                    )

                    for cam_name in cam_names:
                        cam = cameras[cam_name]
                        gt = load_gt_frame(cam_name, fidx)
                        rendered, _, _, _ = render_gaussians(
                            means3D.float(), scales.float(), rotations.float(),
                            opacity.float(), shs.float(),
                            cam, bg_color, sh_degree
                        )
                        total_loss = total_loss + loss_weights.get("rgb", 1.0) * photometric_loss(
                            rendered, gt, lambda_ssim=loss_weights.get("ssim", 0.85),
                            win_size=ssim_win_size
                        )
                        total_psnr += compute_psnr(rendered, gt)
                        num_renders += 1

                    if loss_weights.get("tv", 0) > 0:
                        deltas_flat = torch.cat([
                            deltas["dxyz"].reshape(-1),
                            deltas["dscale"].reshape(-1),
                            deltas["opacity_logit"].reshape(-1),
                        ], dim=0)
                        if prev_deltas_flat is not None:
                            tv = (deltas_flat - prev_deltas_flat).pow(2).mean()
                            total_loss = total_loss + loss_weights["tv"] * tv
                        prev_deltas_flat = deltas_flat.detach()

                if loss_weights.get("scale_reg", 0) > 0:
                    total_loss = total_loss + loss_weights["scale_reg"] * res_scale * scale_regularization(scales.float())
                if loss_weights.get("opacity_reg", 0) > 0:
                    total_loss = total_loss + loss_weights["opacity_reg"] * opacity_regularization(opacity.float())

            total_loss = total_loss / max(num_renders, 1)
            avg_psnr = total_psnr / max(num_renders, 1)

            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    for mod in modules.values():
                        nn.utils.clip_grad_norm_(mod.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if grad_clip > 0:
                    for mod in modules.values():
                        nn.utils.clip_grad_norm_(mod.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()
            metrics_acc.update("loss", total_loss)

            if step % log_every == 0:
                metrics = metrics_acc.flush()
                if writer is not None:
                    for k, v in metrics.items():
                        writer.add_scalar(f"{phase_name}/{k}", v, global_step + step)
                    writer.add_scalar(f"{phase_name}/psnr", avg_psnr, global_step + step)
                lr = optimizer.param_groups[0]["lr"]
                vram_alloc, _ = get_vram_gb()
                loss_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                pbar.set_postfix_str(f"{loss_str} psnr={avg_psnr:.1f} lr={lr:.2e} vram={vram_alloc:.1f}GB")

            if step > 0 and step % save_every == 0:
                snap = metrics_acc.flush() if metrics_acc._sums else {}
                ckpt_mgr.save(phase_idx, phase_name, step, modules, optimizer, scheduler, snap)

            if step % eval_every == 0:
                with torch.no_grad():
                    eval_cam = cameras[eval_cam_name]
                    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.no_grad()
                    with autocast_ctx:
                        canonical_eval = modules["canonical_head"](tokens_mean)
                        if is_canonical_only:
                            m3d, sc, rot, op, sh = compose_gaussians(
                                canonical_eval, scale_factor=scale_anneal_target
                            )
                        else:
                            tokens_t_eval = tokens[0]
                            deltas_eval = modules["deformation_head"](canonical_eval["hidden"], tokens_t_eval)
                            m3d, sc, rot, op, sh = compose_gaussians(
                                canonical_eval, deltas_eval, scale_factor=scale_anneal_target
                            )
                    gt_novel = load_gt_frame(eval_cam_name, 0)
                    rendered_novel, _, _, _ = render_gaussians(
                        m3d.float(), sc.float(), rot.float(), op.float(), sh.float(),
                        eval_cam, bg_color, sh_degree
                    )
                    novel_psnr = -10 * torch.log10(F.mse_loss(rendered_novel, gt_novel) + 1e-8).item()
                    if writer:
                        writer.add_scalar(f"{phase_name}/novel_psnr", novel_psnr, global_step + step)
                    tqdm.write(f"  Step {step}: novel PSNR = {novel_psnr:.2f} dB")
                    save_render(rendered_novel, os.path.join(phase_render_dir, f"novel_{step:05d}.jpg"))

        final_metrics = metrics_acc.flush() if metrics_acc._sums else {}
        ckpt_mgr.save(phase_idx, phase_name, phase.steps, modules, optimizer, scheduler, final_metrics)

        phase_elapsed = time.time() - phase_start
        phase_times[phase_name] = phase_elapsed
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPhase {phase_name} complete: {phase_elapsed:.0f}s ({phase_elapsed/60:.1f}min), Peak VRAM: {peak_vram:.1f}GB")
        global_step += phase.steps
        torch.cuda.empty_cache()

    total_elapsed = time.time() - start_time
    if writer:
        writer.close()

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — AdaptiveScale v2 (AMP={use_amp}, Compile={use_compile})")
    print(f"{'='*60}")
    for name, t in phase_times.items():
        steps = next(p.steps for p in engine_config.phases if p.name == name)
        print(f"  {name:20s}: {t:8.0f}s ({t/60:6.1f}min)  [{steps} steps, {steps/t:.1f} steps/s]")
    print(f"  {'TOTAL':20s}: {total_elapsed:8.0f}s ({total_elapsed/3600:.2f}h)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?",
                        default=os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    parser.add_argument("--start_phase", type=int, default=0)
    args = parser.parse_args()
    train(args.config, start_phase=args.start_phase)
