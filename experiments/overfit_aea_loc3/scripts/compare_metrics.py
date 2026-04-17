"""Compare ours (experiments/overfit_aea_loc3/renders/<proto>/summary.json)
vs 4DGT (4dgt_baseline/metrics_<proto>.json) side-by-side.

Prints a markdown table; writes docs/superpowers/plans/2026-04-17-aria-4dgt-comparison-results.md.
"""
import argparse
import json
import os


def fmt(x, dec=3):
    return f"{x:.{dec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_summary", required=True)
    ap.add_argument("--fourdgt_summary", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    with open(args.ours_summary) as f:
        ours = json.load(f)
    with open(args.fourdgt_summary) as f:
        fgt = json.load(f)

    assert ours["protocol"] == fgt["protocol"], "protocol mismatch"

    lines = [
        f"# Aria vs 4DGT Comparison — protocol: {ours['protocol']}",
        "",
        "| Metric  | Ours (heldout) | 4DGT (heldout) | Δ (ours − 4dgt) |",
        "|---------|----------------|-----------------|------------------|",
    ]
    for key, dec in [("psnr", 2), ("ssim", 4), ("lpips", 4)]:
        a = ours["heldout"][key]["mean"]
        b = fgt["heldout"][key]["mean"]
        d = a - b
        lines.append(f"| {key.upper()} | {fmt(a, dec)} | {fmt(b, dec)} | {fmt(d, dec)} |")

    lines += ["", f"- #heldout frames: ours={ours['heldout']['psnr']['n']}, 4dgt={fgt['heldout']['psnr']['n']}"]

    md = "\n".join(lines) + "\n"
    with open(args.out_md, "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
