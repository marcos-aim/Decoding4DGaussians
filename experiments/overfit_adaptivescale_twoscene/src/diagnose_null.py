"""Null-input diagnostic.

Does the canonical head actually *use* its input tokens, or does it produce an
output determined almost entirely by its learned biases plus the runtime anchors?

For each scene in the config, render four images from the scene's eval camera:

  real_tokens      : tokens = the dataset's real STV2 tokens           (baseline)
  zero_tokens      : tokens = all zeros                                (bias-only)
  random_tokens    : tokens = gaussian noise scaled to per-dim std     (random input)
  shuffled_tokens  : tokens = real tokens, but patch order permuted    (bag-of-patches test)

How to read the results:

  - If zero_tokens renders a *recognizable scene* (any scene, even a training one):
      -> bias terms + anchors are doing most of the work. The NN barely needs
         its tokens. Expect bad generalization and memorization.
      -> fix: reduce capacity, add token noise during training, weight decay.

  - If zero_tokens and real_tokens are visually identical for a given scene:
      -> tokens contribute near zero in that regime. Same fix.

  - If random_tokens produces garbage (solid color, scattered gaussians):
      -> the NN is sensitive to its input distribution. Good sign for "uses tokens"
         but still says nothing about whether it generalizes.

  - If shuffled_tokens looks the same as real_tokens:
      -> the per-patch identity is irrelevant; the NN treats tokens as a bag.
      -> fix: add positional encoding, encourage per-patch structure.

  - If shuffled_tokens is a scrambled mess:
      -> the NN uses per-patch identity. Expected for a working model.

Run:
    python diagnose_null.py \\
        --config /path/to/experiment/config.yaml \\
        --checkpoint /path/to/stage1_final.pt \\
        --output_dir ./diagnose_null_out
"""

import os
import sys
import argparse
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
    ap.add_argument("--output_dir", default="./diagnose_null_out")
    ap.add_argument("--noise_std_mult", type=float, default=1.0,
                    help="Scale random-token noise by this * per-dim std")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    multi = build_multi_scene(cfg)
    meta = build_scene_meta(multi)
    head = build_canonical_head(cfg, args.checkpoint)

    bg_color = torch.tensor(cfg["rendering"]["bg_color"], device="cuda")
    sh_degree = cfg["model"]["sh_degree"]

    for scene_name in [s["name"] for s in multi.scenes]:
        print(f"\n[null] {scene_name}")
        m = meta[scene_name]

        tokens_real = m["dataset"].get_tokens_mean().cuda()  # [P, D]
        P = tokens_real.shape[0]

        # Match noise to per-dimension token stats so magnitudes are comparable.
        token_std = tokens_real.std(dim=0, keepdim=True)
        tokens_zero = torch.zeros_like(tokens_real)
        tokens_random = torch.randn_like(tokens_real) * token_std * args.noise_std_mult
        perm = torch.randperm(P, device=tokens_real.device)
        tokens_shuffled = tokens_real[perm]

        cam = m["cameras"][m["eval_camera"]]

        configs = [
            ("real_tokens",     tokens_real),
            ("zero_tokens",     tokens_zero),
            ("random_tokens",   tokens_random),
            ("shuffled_tokens", tokens_shuffled),
        ]

        out_sub = os.path.join(args.output_dir, scene_name)
        os.makedirs(out_sub, exist_ok=True)
        for name, tokens in configs:
            rendered = render_config(
                head,
                tokens,
                m["xyz_anchor"],
                m["scale_anchor"],
                m["norm_center"],
                m["norm_scale"],
                cam,
                bg_color,
                sh_degree,
            )
            out_path = os.path.join(out_sub, f"{name}.png")
            save_image(rendered, out_path)
            print(f"  saved {os.path.basename(out_path)}")

    print(f"\nDone. Outputs in {args.output_dir}")
    print("Compare the four images per scene folder to interpret (see docstring at top of file).")


if __name__ == "__main__":
    main()
