"""Batch evaluation for all scenes from multi-scene trained model.

Evaluates all scenes in the config, computes metrics, generates summary report.

Usage:
    python src/eval_all.py --config config.yaml [--checkpoint PATH] [--output-dir ./eval_all_output]
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml
import numpy as np


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Batch evaluation for all scenes from multi-scene trained model"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to multi-scene config.yaml"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (default: stage3_final.pt or stage2_final.pt)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./eval_all_output",
        help="Directory to save all evaluation results (default: ./eval_all_output)"
    )
    args = parser.parse_args()

    print("="*60)
    print("Multi-Scene Batch Evaluation")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print("="*60)

    # Load config to get scene names
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    scenes = [s["name"] for s in cfg["scenes"]]
    print(f"\nFound {len(scenes)} scenes: {scenes}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each scene
    all_metrics = {}

    for i, scene_name in enumerate(scenes):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(scenes)}] Evaluating scene: {scene_name}")
        print(f"{'='*60}\n")

        # Build command for eval.py
        cmd = [
            "python", "src/eval.py",
            "--config", args.config,
            "--scene", scene_name,
            "--output-dir", str(output_dir)
        ]

        if args.checkpoint is not None:
            cmd.extend(["--checkpoint", args.checkpoint])

        # Run evaluation
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"ERROR: Evaluation failed for scene '{scene_name}'")
            continue

        # Parse metrics from output file
        metrics_file = output_dir / scene_name / "metrics.txt"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Average PSNR:"):
                        psnr = float(line.split(":")[1].strip().split()[0])
                    elif line.startswith("Average SSIM:"):
                        ssim = float(line.split(":")[1].strip())

                all_metrics[scene_name] = {"psnr": psnr, "ssim": ssim}

    # Generate summary report
    print(f"\n{'='*60}")
    print("Summary Report")
    print(f"{'='*60}")

    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("Multi-Scene Evaluation Summary\n")
        f.write("="*60 + "\n\n")

        # Per-scene results
        f.write("Per-Scene Results:\n")
        f.write("-"*60 + "\n")
        print("\nPer-Scene Results:")
        print("-"*60)

        for scene_name, metrics in all_metrics.items():
            line = f"{scene_name:20s}  PSNR: {metrics['psnr']:6.2f} dB  SSIM: {metrics['ssim']:.4f}"
            f.write(line + "\n")
            print(line)

        # Average across all scenes
        if all_metrics:
            avg_psnr = np.mean([m["psnr"] for m in all_metrics.values()])
            avg_ssim = np.mean([m["ssim"] for m in all_metrics.values()])

            f.write("\n" + "="*60 + "\n")
            f.write("Average Across All Scenes:\n")
            f.write("-"*60 + "\n")
            f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write("="*60 + "\n")

            print("\n" + "="*60)
            print("Average Across All Scenes:")
            print("-"*60)
            print(f"Average PSNR: {avg_psnr:.2f} dB")
            print(f"Average SSIM: {avg_ssim:.4f}")
            print("="*60)

        f.write(f"\nTotal scenes evaluated: {len(all_metrics)}/{len(scenes)}\n")
        f.write(f"\nDetailed results for each scene can be found in:\n")
        for scene_name in all_metrics.keys():
            f.write(f"  - {output_dir / scene_name}\n")

    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_file}")
    print(f"Individual results in: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
