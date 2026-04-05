"""Shared loading helpers for memorization diagnostics.

Not a script. Imported by diagnose_swap.py and diagnose_null.py.

These helpers rebuild the exact same multi-scene eval state as train.py's
Stage-1 eval block, so the diagnostics see the model under the same conditions
it was trained in (same anchors, same normalization mode, same cameras).
"""

import os
import sys
import glob
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cameras import build_cameras_from_llff
from dataset import MultiSceneDataset
from model import CanonicalGaussianHead, compose_gaussians
from renderer import render_gaussians


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_multi_scene(cfg: dict) -> MultiSceneDataset:
    """Mirror of train.py scene-construction logic. Uses every scene listed in cfg."""
    target_w = cfg["rendering"]["image_width"]
    target_h = cfg["rendering"]["image_height"]

    scene_configs = []
    for scene_cfg in cfg["scenes"]:
        scene_dir = os.path.dirname(scene_cfg["poses_bounds_path"])
        cam_files = sorted([
            os.path.basename(f)
            for f in glob.glob(os.path.join(scene_dir, "cam*.mp4"))
        ])
        cameras = build_cameras_from_llff(
            scene_cfg["poses_bounds_path"], target_w, target_h, cam_files
        )
        scene_configs.append({
            "name": scene_cfg["name"],
            "cache_dir": scene_cfg["cache_dir"],
            "cameras": cameras,
            "input_camera": scene_cfg.get("input_camera", "cam01"),
            "eval_camera": scene_cfg.get("eval_camera", "cam00"),
            "weight": scene_cfg.get("weight", 1.0),
        })

    K = cfg["model"]["num_gaussians_per_patch"]
    return MultiSceneDataset(
        scene_configs,
        normalize_coords=cfg["normalization"].get("enabled", False),
        K=K,
    )


def build_scene_meta(multi_dataset: MultiSceneDataset) -> dict:
    """Per-scene lookup: anchors, normalization constants, cameras, eval cam name."""
    meta = {}
    for scene in multi_dataset.scenes:
        ds = scene["dataset"]
        has_norm = (ds.norm_center is not None) and (ds.norm_scale is not None)
        meta[scene["name"]] = {
            "dataset": ds,
            "cameras": scene["cameras"],
            "eval_camera": scene["eval_camera"],
            "xyz_anchor":   ds.xyz_anchor.cuda(),
            "scale_anchor": ds.scale_anchor.cuda(),
            "norm_center":  ds.norm_center.cuda() if has_norm else None,
            "norm_scale":   torch.tensor(float(ds.norm_scale), device="cuda") if has_norm else None,
        }
    return meta


def build_canonical_head(cfg: dict, ckpt_path: str) -> CanonicalGaussianHead:
    K = cfg["model"]["num_gaussians_per_patch"]
    head = CanonicalGaussianHead(
        dim_in=cfg["model"]["dim_tokens"],
        dim_hidden=cfg["model"]["dim_canonical_hidden"],
        sh_degree=cfg["model"]["sh_degree"],
        num_gaussians_per_patch=K,
    ).cuda()

    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    head.load_state_dict(ckpt["canonical_head"])
    head.eval()
    print(f"[load] {os.path.basename(ckpt_path)} (step={ckpt.get('step', '?')})")
    return head


def denormalize(means3D, scales, norm_center, norm_scale):
    """No-op when normalization is disabled (norm_center/norm_scale are None)."""
    if norm_center is None or norm_scale is None:
        return means3D, scales
    return means3D * norm_scale + norm_center, scales * norm_scale


@torch.no_grad()
def render_config(
    head: CanonicalGaussianHead,
    tokens_mean: torch.Tensor,
    xyz_anchor: torch.Tensor,
    scale_anchor: torch.Tensor,
    norm_center,
    norm_scale,
    camera,
    bg_color: torch.Tensor,
    sh_degree: int,
) -> torch.Tensor:
    """Single forward pass: tokens -> canonical -> compose -> denorm -> render.

    Returns a clamped [C, H, W] rgb image ready for save_image.
    """
    canonical = head(tokens_mean, xyz_anchor=xyz_anchor, scale_anchor=scale_anchor)
    means3D, scales, rot, opacity, shs = compose_gaussians(canonical, scale_factor=1.0)
    means3D, scales = denormalize(means3D, scales, norm_center, norm_scale)
    rendered, _, _, _ = render_gaussians(
        means3D, scales, rot, opacity, shs, camera, bg_color, sh_degree
    )
    return rendered.clamp(0.0, 1.0)
