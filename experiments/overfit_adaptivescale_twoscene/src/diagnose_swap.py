"""Token/anchor swap diagnostic.

For every unordered pair of scenes (A, B) in the config, render four images:

  A_tokens_A_anchors  (baseline from scene A, camera = A's eval_camera)
  B_tokens_A_anchors  (B's tokens flowing through A's anchor container, camera = A)
  A_tokens_B_anchors  (A's tokens flowing through B's anchor container, camera = B)
  B_tokens_B_anchors  (baseline from scene B, camera = B's eval_camera)

How to read the results:

  - If B_tokens_A_anchors looks identical to the A baseline:
      -> the NN is ignoring its token input; anchors + biases carry all the information.
      -> fix: reduce capacity, force the NN to use tokens (noise augmentation on tokens,
         stronger bottleneck, token dropout).

  - If B_tokens_A_anchors looks like scene B (textures, colors, even geometry):
      -> the tokens carry scene identity end-to-end; anchors are cosmetic.
      -> fix: normalize tokens per-scene (z-score), information bottleneck, reduce
         hidden dim so the NN can't memorize a lookup table.

  - If B_tokens_A_anchors looks like a clean mashup (B's colors/textures on A's
    geometry, or vice versa):
      -> clean disentanglement. This is the regime you want. Overfitting on a
         4-scene training set probably comes from something other than retrieval.

Run:
    python diagnose_swap.py \\
        --config /path/to/experiment/config.yaml \\
        --checkpoint /path/to/stage1_final.pt \\
        --output_dir ./diagnose_swap_out
"""

import os
import sys
import argparse
import itertools
import torch
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_utils import (
    load_config,
    build_multi_scene,
    build_scene_meta,
    build_canonical_head,
    render_config,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment config.yaml")
    ap.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
    ap.add_argument("--output_dir", default="./diagnose_swap_out",
                    help="Where to write per-pair output directories")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    multi = build_multi_scene(cfg)
    meta = build_scene_meta(multi)
    head = build_canonical_head(cfg, args.checkpoint)

    bg_color = torch.tensor(cfg["rendering"]["bg_color"], device="cuda")
    sh_degree = cfg["model"]["sh_degree"]

    scene_names = [s["name"] for s in multi.scenes]
    if len(scene_names) < 2:
        raise RuntimeError(f"Need at least 2 scenes for swap diagnostic, got {len(scene_names)}")

    for a, b in itertools.combinations(scene_names, 2):
        print(f"\n[swap] {a}  <->  {b}")
        ma, mb = meta[a], meta[b]

        tokens_a = ma["dataset"].get_tokens_mean().cuda()
        tokens_b = mb["dataset"].get_tokens_mean().cuda()
        cam_a = ma["cameras"][ma["eval_camera"]]
        cam_b = mb["cameras"][mb["eval_camera"]]

        # IMPORTANT: the camera must match the anchor coordinate frame, because
        # anchors are in scene-specific world space. Rendering scene A's anchors
        # through scene B's camera would just produce garbage (wrong world frame).
        configs = [
            (f"{a}_tokens__{a}_anchors",  tokens_a, ma, cam_a),  # baseline A
            (f"{b}_tokens__{a}_anchors",  tokens_b, ma, cam_a),  # swap tokens into A's container
            (f"{a}_tokens__{b}_anchors",  tokens_a, mb, cam_b),  # swap tokens into B's container
            (f"{b}_tokens__{b}_anchors",  tokens_b, mb, cam_b),  # baseline B
        ]

        out_sub = os.path.join(args.output_dir, f"{a}__vs__{b}")
        os.makedirs(out_sub, exist_ok=True)
        for name, tokens, anc_meta, cam in configs:
            rendered = render_config(
                head,
                tokens,
                anc_meta["xyz_anchor"],
                anc_meta["scale_anchor"],
                anc_meta["norm_center"],
                anc_meta["norm_scale"],
                cam,
                bg_color,
                sh_degree,
            )
            out_path = os.path.join(out_sub, f"{name}.png")
            save_image(rendered, out_path)
            print(f"  saved {os.path.basename(out_path)}")

    print(f"\nDone. Outputs in {args.output_dir}")
    print("Compare the four images in each pair folder to interpret (see docstring at top of file).")


if __name__ == "__main__":
    main()
