"""Read 4DGT render outputs + GT Aria frames, compute PSNR/SSIM/LPIPS
on the heldout indices under a chosen protocol.

4DGT's tlod/run.py writes renders to: <output_dir>/<seq>/renders/<frame_idx>.png
(exact layout may vary — this script auto-detects the render subdir).
"""
import argparse
import json
import os
import sys

import cv2
import lpips
import numpy as np
import torch
from pytorch_msssim import ssim as compute_ssim
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from aria_dataset import AriaSequenceDataset


def build_splits(num_frames: int, protocol: str):
    if protocol == "4dgt_16_112":
        input_idx = list(range(0, num_frames, 8))
    elif protocol == "64_64":
        input_idx = list(range(0, num_frames, 2))
    else:
        raise ValueError(protocol)
    heldout_idx = [i for i in range(num_frames) if i not in set(input_idx)]
    return input_idx, heldout_idx


def find_render_dir(out_root: str, seq_name: str) -> str:
    """Auto-detect 4DGT output dir."""
    candidates = [
        os.path.join(out_root, seq_name, "renders"),
        os.path.join(out_root, seq_name, "render"),
        os.path.join(out_root, seq_name),
    ]
    for c in candidates:
        if os.path.isdir(c) and any(f.endswith(".png") for f in os.listdir(c)):
            return c
    raise RuntimeError(f"could not find 4DGT renders under {out_root}/{seq_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fourdgt_out", required=True,
                    help="--output_dir passed to tlod.run")
    ap.add_argument("--seq_name", default="loc3_script3_seq1_rec1")
    ap.add_argument("--aria_seq_root", required=True)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=128)
    ap.add_argument("--protocol", default="4dgt_16_112", choices=["4dgt_16_112", "64_64"])
    ap.add_argument("--target_w", type=int, default=504)
    ap.add_argument("--target_h", type=int, default=504)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    aria = AriaSequenceDataset(
        args.aria_seq_root, target_resolution=(args.target_w, args.target_h),
        frame_indices=list(range(args.start_frame, args.start_frame + args.num_frames)),
    )

    render_dir = find_render_dir(args.fourdgt_out, args.seq_name)
    input_idx, heldout_idx = build_splits(args.num_frames, args.protocol)

    lp = lpips.LPIPS(net="alex").cuda().eval()
    per_frame = {}
    for i in tqdm(heldout_idx, desc="4DGT heldout"):
        rp = os.path.join(render_dir, f"{i:04d}.png")
        if not os.path.exists(rp):
            rp = os.path.join(render_dir, f"{i:06d}.png")
        rendered = cv2.imread(rp)
        rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        if rendered.shape[:2] != (args.target_h, args.target_w):
            rendered = cv2.resize(rendered, (args.target_w, args.target_h))
        rendered = torch.from_numpy(rendered).permute(2, 0, 1).float().cuda() / 255.0
        gt = aria.get_frame_image(i).cuda()

        mse = torch.mean((rendered - gt) ** 2).item()
        psnr = -10.0 * np.log10(max(mse, 1e-10))
        ssim_v = float(compute_ssim(rendered.unsqueeze(0), gt.unsqueeze(0),
                                     data_range=1.0, size_average=True).item())
        lp_v = float(lp((rendered * 2 - 1).unsqueeze(0),
                        (gt * 2 - 1).unsqueeze(0)).item())
        per_frame[i] = {"psnr": psnr, "ssim": ssim_v, "lpips": lp_v}

    def agg(name):
        xs = [v[name] for v in per_frame.values()]
        return {"mean": float(np.mean(xs)), "min": float(np.min(xs)),
                "max": float(np.max(xs)), "n": len(xs)}

    summary = {
        "system": "4DGT",
        "protocol": args.protocol,
        "num_heldout": len(heldout_idx),
        "heldout": {"psnr": agg("psnr"), "ssim": agg("ssim"), "lpips": agg("lpips")},
        "per_frame": per_frame,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["heldout"], indent=2))


if __name__ == "__main__":
    main()
