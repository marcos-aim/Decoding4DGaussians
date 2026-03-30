"""Compute PSNR metrics from pre-rendered eval frames vs GT."""
import cv2
import numpy as np
from pathlib import Path

def compute_metrics(render_dir, gt_dir, label):
    psnrs = []
    for i in range(300):
        fname = f"{i:06d}.jpg"
        r = cv2.imread(str(Path(render_dir) / fname))
        g = cv2.imread(str(Path(gt_dir) / fname))
        if r is None or g is None:
            continue
        if r.shape != g.shape:
            g = cv2.resize(g, (r.shape[1], r.shape[0]))
        r = r.astype(np.float64) / 255.0
        g = g.astype(np.float64) / 255.0
        mse = np.mean((r - g) ** 2)
        psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
        psnrs.append(psnr)

    psnrs = np.array(psnrs)
    worst10 = np.argsort(psnrs)[:10]
    print(f"\n=== {label} ===")
    print(f"  Frames: {len(psnrs)}")
    print(f"  Mean PSNR: {psnrs.mean():.2f} dB")
    print(f"  Min  PSNR: {psnrs.min():.2f} dB (frame {np.argmin(psnrs)})")
    print(f"  Max  PSNR: {psnrs.max():.2f} dB (frame {np.argmax(psnrs)})")
    print(f"  Std  PSNR: {psnrs.std():.2f} dB")
    print(f"  Worst 10 frames: {worst10.tolist()}")
    print(f"  Worst 10 PSNRs:  {[f'{psnrs[i]:.2f}' for i in worst10]}")

gt_lo = "D:/DecodeGaussians/experiments/overfit_coffee_martini/cache/frames/cam00"
gt_hi = "D:/DecodeGaussians/experiments/overfit_highres_cross_attn_coffee_martini/cache/frames/cam00"

compute_metrics(
    "D:/DecodeGaussians/experiments/overfit_cross_attn_coffee_martini/renders/eval",
    gt_lo, "cross_attn (512x384, K=128) - PRIMARY BASELINE")

compute_metrics(
    "D:/DecodeGaussians/experiments/overfit_coffee_martini/renders/eval",
    gt_lo, "self_attn (512x384, K=128) - REFERENCE")

compute_metrics(
    "D:/DecodeGaussians/experiments/overfit_highres_cross_attn_coffee_martini/renders/eval",
    gt_hi, "highres_cross_attn (1024x768, K=128) - REFERENCE")

compute_metrics(
    "D:/DecodeGaussians/experiments/overfit_highres_highcount_coffee_martini/renders/eval",
    gt_hi, "highres_highcount (1024x768, K=512) - REFERENCE")
