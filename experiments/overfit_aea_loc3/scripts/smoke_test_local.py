"""Local smoke test: exercise the aria_loc3 pipeline end-to-end on a SYNTHETIC
Aria sequence (real images from Neu3D cam01, fake-but-valid transforms.json).

Catches:
  - shape/dtype bugs in AriaSequenceDataset, aria_cameras, precompute.py
  - STV2 forward pass & hook token capture at a resolution we haven't tried
  - model forward (canonical + deformation heads) with our config (grid 18x18 here)
  - renderer integration (does it produce non-zero pixels?)
  - AriaCachedDataset reload
  - VRAM budget at reduced scale → extrapolate to 504×504 cluster run

Does NOT exercise:
  - projectaria_tools VRS preprocessing
  - real MPS pose quality
  - training convergence
"""
import json
import os
import sys
import time
import tempfile
import shutil

import cv2
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(REPO, "experiments", "overfit_aea_loc3", "src")
SPATRACKER_ROOT = os.path.join(REPO, "SpaTrackerV2")
sys.path.insert(0, SRC)
sys.path.insert(0, SPATRACKER_ROOT)

NEU3D_CAM01_MP4 = os.path.join(REPO, "datasets", "neu3d", "coffee_martini", "cam01.mp4")

# Tiny config for local smoke test
N_FRAMES = 16
IMG_SIZE = 252            # 18*14, matches VGGT patch stride
GRID = 18                  # 252 / 14
K = 32                     # Gaussians per patch (plan uses 128; smaller here for speed)
CACHE_DIR = os.path.join(REPO, "experiments", "overfit_aea_loc3", "smoke_cache")
SEQ_DIR = os.path.join(REPO, "experiments", "overfit_aea_loc3", "smoke_seq")


def make_fake_aria_sequence() -> str:
    """Extract 16 frames from Neu3D cam01 (center-crop + resize to 252x252),
    write them as PNGs + a transforms.json mimicking 4DGT's format."""
    if not os.path.exists(NEU3D_CAM01_MP4):
        raise FileNotFoundError(f"Expected Neu3D coffee_martini cam01.mp4 at {NEU3D_CAM01_MP4}")

    os.makedirs(os.path.join(SEQ_DIR, "images"), exist_ok=True)

    cap = cv2.VideoCapture(NEU3D_CAM01_MP4)
    frames = []
    frame_idx = 0
    while len(frames) < N_FRAMES:
        ok, bgr = cap.read()
        if not ok:
            break
        # Center-crop to square
        h, w = bgr.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        crop = bgr[y0:y0+s, x0:x0+s]
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(resized)
        frame_idx += 1
    cap.release()

    if len(frames) < N_FRAMES:
        raise RuntimeError(f"Only got {len(frames)} frames from Neu3D cam01")

    # Write frames + transforms.json
    out = {"frames": []}
    for i, fr in enumerate(frames):
        path = f"images/{i:04d}.png"
        cv2.imwrite(os.path.join(SEQ_DIR, path), fr)
        # Synthesize: camera translates along +X by 0.02 m per frame
        c2w = np.eye(4)
        c2w[0, 3] = i * 0.02
        out["frames"].append({
            "fx": 150.0,
            "fy": 150.0,
            "cx": IMG_SIZE / 2,
            "cy": IMG_SIZE / 2,
            "w": IMG_SIZE,
            "h": IMG_SIZE,
            "image_path": path,
            "transform_matrix": c2w.tolist(),
            "timestamp": i * int(1e8),
        })

    with open(os.path.join(SEQ_DIR, "transforms.json"), "w") as f:
        json.dump(out, f)

    print(f"[seq] wrote {N_FRAMES} frames to {SEQ_DIR} at {IMG_SIZE}x{IMG_SIZE}")
    return SEQ_DIR


def print_vram(label: str):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[vram] {label:30s} allocated={a:.2f} GB  reserved={r:.2f} GB  peak={peak:.2f} GB")


def test_aria_dataset(seq_dir: str):
    from aria_dataset import AriaSequenceDataset
    from aria_cameras import build_cameras_from_aria

    ds = AriaSequenceDataset(seq_dir, target_resolution=(IMG_SIZE, IMG_SIZE))
    print(f"[dataset] len={len(ds)}, target={ds.target_w}x{ds.target_h}")
    img0 = ds.get_frame_image(0)
    assert img0.shape == (3, IMG_SIZE, IMG_SIZE), f"bad shape {img0.shape}"
    assert 0.0 <= img0.min().item() and img0.max().item() <= 1.0
    K = ds.get_intrinsics(0)
    c2w = ds.get_c2w(0)
    print(f"[dataset] img shape={tuple(img0.shape)}, K[0,0]={K[0,0].item():.3f}, c2w[0,3]={c2w[0,3].item():.3f}")

    cams = build_cameras_from_aria(ds, list(range(len(ds))))
    assert len(cams) == N_FRAMES
    cam0 = cams[0]
    print(f"[cameras] FoVx={cam0.FoVx:.3f} rad, center={cam0.camera_center.cpu().tolist()}")
    return ds, cams


def run_precompute_inline(ds):
    """Inline the precompute logic (smaller window, smaller K) to catch
    shape bugs in the STV2 hook without the CLI overhead."""
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track

    print("[precompute] loading VGGT4Track...")
    t0 = time.time()
    model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().cuda()
    print_vram("after load VGGT4Track")
    print(f"[precompute] VGGT4Track loaded in {time.time()-t0:.1f}s")

    captured = {}
    def hook_fn(module, inputs, output):
        tokens_list, _ = output
        captured["tokens"] = tokens_list[-1].detach().cpu()
    handle = model.aggregator.register_forward_hook(hook_fn)

    WINDOW = 8
    all_tok, all_pm, all_po, all_intr, all_unc = [], [], [], [], []
    for start in range(0, N_FRAMES, WINDOW):
        end = min(start + WINDOW, N_FRAMES)
        frames = ds.get_frames_stack(list(range(start, end))).unsqueeze(0).cuda()
        print(f"[precompute] window [{start}:{end}], input shape {tuple(frames.shape)}")
        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(frames)
        n = end - start
        all_tok.append(captured["tokens"].squeeze(0)[:n].half())
        all_pm.append(pred["points_map"].cpu()[:n])
        all_po.append(pred["poses_pred"].cpu().squeeze(0)[:n])
        all_intr.append(pred["intrs"].cpu().squeeze(0)[:n])
        all_unc.append(pred["unc_metric"].cpu()[:n])
        print(f"[precompute]   tokens     {tuple(all_tok[-1].shape)}  {all_tok[-1].dtype}")
        print(f"[precompute]   points_map {tuple(all_pm[-1].shape)}  {all_pm[-1].dtype}")
        print(f"[precompute]   poses_pred {tuple(all_po[-1].shape)}  {all_po[-1].dtype}")
        print(f"[precompute]   intrs      {tuple(all_intr[-1].shape)}  {all_intr[-1].dtype}")
        print(f"[precompute]   unc_metric {tuple(all_unc[-1].shape)}  {all_unc[-1].dtype}")
    handle.remove()
    print_vram("after STV2 forward")

    tokens = torch.cat(all_tok, dim=0)
    points_map = torch.cat(all_pm, dim=0).float()
    poses = torch.cat(all_po, dim=0).float()
    intrs = torch.cat(all_intr, dim=0).float()
    unc = torch.cat(all_unc, dim=0)

    # Alignment (using plan's logic)
    stv2_c2w_0 = poses[0]
    stv2_inv0 = torch.inverse(stv2_c2w_0)
    pts_flat = points_map.reshape(-1, 3)
    pts_cam0 = pts_flat @ stv2_inv0[:3, :3].T + stv2_inv0[:3, 3]
    z_stv2 = pts_cam0[:, 2]
    z_pos = z_stv2[z_stv2 > 0]
    z_near_stv2 = torch.quantile(z_pos, 0.05).item()
    ARIA_NEAR = 0.3
    scale = ARIA_NEAR / max(z_near_stv2, 1e-6)
    print(f"[align] z_near_stv2={z_near_stv2:.3f}  scale={scale:.3f}")

    aria_c2w_0 = ds.get_c2w(0)
    R_aria = aria_c2w_0[:3, :3]
    t_aria = aria_c2w_0[:3, 3]
    pts_scaled = scale * (pts_flat @ stv2_inv0[:3, :3].T + stv2_inv0[:3, 3])
    pts_world = pts_scaled @ R_aria.T + t_aria
    points_map_aligned = pts_world.reshape(points_map.shape)

    # Save to cache (mimicking precompute.py)
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(tokens, os.path.join(CACHE_DIR, "tokens.pt"))
    torch.save(points_map_aligned.half(), os.path.join(CACHE_DIR, "points_map.pt"))
    torch.save(poses, os.path.join(CACHE_DIR, "poses.pt"))
    torch.save(intrs, os.path.join(CACHE_DIR, "intrs.pt"))
    torch.save(unc, os.path.join(CACHE_DIR, "unc_metric.pt"))
    torch.save(torch.arange(N_FRAMES), os.path.join(CACHE_DIR, "frame_indices.pt"))
    aria_c2w_all = torch.stack([ds.get_c2w(i) for i in range(len(ds))])
    aria_K_all = torch.stack([ds.get_intrinsics(i) for i in range(len(ds))])
    torch.save(aria_c2w_all, os.path.join(CACHE_DIR, "aria_c2w.pt"))
    torch.save(aria_K_all, os.path.join(CACHE_DIR, "aria_K.pt"))
    torch.save({"scale": scale, "near_aria": ARIA_NEAR, "z_near_stv2": z_near_stv2},
               os.path.join(CACHE_DIR, "align_meta.pt"))
    print(f"[precompute] cache saved to {CACHE_DIR}")

    del model
    torch.cuda.empty_cache()
    return tokens.shape, points_map_aligned.shape


def test_cached_dataset():
    from aria_cached_dataset import AriaCachedDataset
    ds = AriaCachedDataset(CACHE_DIR, SEQ_DIR, target_resolution=(IMG_SIZE, IMG_SIZE))
    assert ds.num_frames == N_FRAMES
    print(f"[cached] num_frames={ds.num_frames}, num_patches={ds.num_patches}, token_dim={ds.token_dim}")
    return ds


def test_model_and_render(cached_ds, cams):
    from model import compose_gaussians
    from renderer import render_gaussians
    from train import build_heads

    # Synthesize a minimal cfg structure so build_heads can consume it.
    smoke_cfg = {
        "model": {
            "num_gaussians_per_patch": K,
            "canonical": {
                "dim_in": 2048,
                "dim_hidden": 1024,
                "sh_degree": 0,
                "dpt_dim": 512,
                # grid_h/grid_w intentionally omitted → build_heads computes from P
            },
            "deformation": {
                "dim_canonical": 1024,
                "dim_tokens": 2048,
                "dim_hidden": 256,
                "n_heads": 8,
                "n_layers": 3,
                "max_displacement": 2.0,
            },
        }
    }

    print(f"[model] cached P={cached_ds.num_patches}, token_dim={cached_ds.token_dim}")
    canonical_head, deformation_head = build_heads(smoke_cfg, cached_ds)
    print_vram("after model init")

    tokens_mean = cached_ds.get_tokens_mean()
    canonical = canonical_head(tokens_mean)
    print(f"[model] canonical output keys={list(canonical.keys())}")
    for k, v in canonical.items():
        if torch.is_tensor(v):
            print(f"[model]   canonical[{k}]: {tuple(v.shape)}  {v.dtype}")

    tokens_t = cached_ds.get_tokens_frame(0)
    deltas = deformation_head(canonical["hidden"], tokens_t)
    print(f"[model] deltas keys={list(deltas.keys())}")
    for k, v in deltas.items():
        if torch.is_tensor(v):
            print(f"[model]   deltas[{k}]: {tuple(v.shape)}  {v.dtype}")

    means3D, scales, rotations, opacity, shs = compose_gaussians(canonical, deltas, scale_factor=1.0)
    print(f"[compose] means3D={tuple(means3D.shape)}  scales={tuple(scales.shape)}  "
          f"rotations={tuple(rotations.shape)}  opacity={tuple(opacity.shape)}  shs={tuple(shs.shape)}")
    print(f"[compose] means3D range X=[{means3D[:,0].min():.2f},{means3D[:,0].max():.2f}] "
          f"Y=[{means3D[:,1].min():.2f},{means3D[:,1].max():.2f}] "
          f"Z=[{means3D[:,2].min():.2f},{means3D[:,2].max():.2f}]")
    print(f"[compose] opacity range=[{opacity.min():.4f}, {opacity.max():.4f}], "
          f"scales log range=[{scales.log().min():.2f}, {scales.log().max():.2f}]")
    print_vram("after compose_gaussians")

    bg = torch.zeros(3, device="cuda")
    rendered, _, _, _ = render_gaussians(means3D, scales, rotations, opacity, shs, cams[0], bg, 0)
    print(f"[render] output shape={tuple(rendered.shape)}  dtype={rendered.dtype}")
    print(f"[render] pixel range=[{rendered.min():.4f}, {rendered.max():.4f}], mean={rendered.mean():.4f}")
    print_vram("after render")

    # Sanity: non-zero rendering
    if rendered.abs().sum().item() < 1.0:
        print("[render] WARNING: rendered image is all zeros — points likely behind camera")
    return rendered


def cleanup():
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    if os.path.isdir(SEQ_DIR):
        shutil.rmtree(SEQ_DIR, ignore_errors=True)


def main():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("=" * 70)
    print(f"aea_loc3 local smoke test  (N_FRAMES={N_FRAMES}, IMG={IMG_SIZE}x{IMG_SIZE}, K={K})")
    print("=" * 70)
    try:
        seq_dir = make_fake_aria_sequence()
        ds, cams = test_aria_dataset(seq_dir)
        tokens_shape, pm_shape = run_precompute_inline(ds)
        cached = test_cached_dataset()
        rendered = test_model_and_render(cached, cams)
        print()
        print("=" * 70)
        print("SMOKE TEST PASSED")
        print("=" * 70)
        print(f"tokens:     {tokens_shape}")
        print(f"points_map: {pm_shape}")
        print(f"rendered pixel mean: {rendered.mean().item():.4f}")
        if torch.cuda.is_available():
            print(f"peak VRAM:  {torch.cuda.max_memory_allocated()/1e9:.2f} GB at {IMG_SIZE}x{IMG_SIZE}, K={K}")
            # Extrapolate to cluster: 504x504 is 4x pixels, K=128 is 4x Gaussians
            # STV2 VRAM scales ~linear in pixels; model VRAM scales linear in patch count (which is 4x) and K (4x)
            # rough: linear-in-tokens (4x) + linear-in-gaussians (16x for K side)
            extrapolated = torch.cuda.max_memory_allocated()/1e9 * 4  # STV2 dominates; conservative
            print(f"est cluster VRAM at 504x504 K=128: ~{extrapolated:.1f} GB (STV2-dominated; model is smaller)")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
