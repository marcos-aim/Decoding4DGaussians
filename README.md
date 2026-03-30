# Decoding4DGaussians

Feed-forward dynamic 3D Gaussian splatting from monocular video. A frozen SpaTrackerV2 backbone extracts per-frame tokens and 3D point maps, which learnable decoder heads convert into dynamic Gaussians — no per-scene optimization required.

## Setup

```bash
git clone --recurse-submodules https://github.com/marcos-aim/Decoding4DGaussians.git
cd Decoding4DGaussians

# Install dependencies
pip install torch torchvision --index-url https://cu126.pip.pytorch.org  # match your CUDA version
pip install pytorch_msssim lpips open3d imageio[ffmpeg] viser pyyaml tqdm tensorboard opencv-python

# Build diff-gaussian-rasterization (from 4DGaussians submodule)
pip install 4DGaussians/submodules/diff-gaussian-rasterization
pip install 4DGaussians/submodules/simple-knn
```

## Dataset

Download the **Neu3D** dataset (coffee_martini scene) from the [Neural 3D Video Synthesis](https://github.com/facebookresearch/Neural_3D_Video) project page.

Place it at `datasets/neu3d/coffee_martini/` with structure:
```
datasets/neu3d/coffee_martini/
  cam00.mp4 ... cam20.mp4
  poses_bounds.npy
```

## Running an Experiment

Each experiment lives in `experiments/<name>/` with a `config.yaml` and `src/` directory.

```bash
# 1. Precompute SpaTrackerV2 features (once per scene/resolution)
python experiments/overfit_coffee_martini/src/precompute.py --scene coffee_martini

# 2. Train (3 stages: canonical -> deformation -> joint)
python experiments/<experiment>/src/train.py --config experiments/<experiment>/config.yaml

# 3. Evaluate (renders video, computes PSNR/SSIM, exports PLY)
python experiments/<experiment>/src/eval.py --config experiments/<experiment>/config.yaml
```

Edit `config.yaml` to change resolution, Gaussian count (K), learning rates, and training schedule.

## Experiments

| Experiment | Resolution | K | Mean PSNR | Min PSNR | Status |
|---|---|---|---|---|---|
| `overfit_coffee_martini` | 512x384 | 128 | 24.52 | 23.70 | Done |
| `overfit_cross_attn_coffee_martini` | 512x384 | 128 | 24.61 | 24.14 | Done (primary baseline) |
| `overfit_highres_cross_attn_coffee_martini` | 1024x768 | 128 | 24.06 | 23.54 | Done |
| `overfit_highres_highcount_coffee_martini` | 1024x768 | 512 | 24.74 | 24.20 | Done |
| `overfit_sh1_cross_attn_coffee_martini` | 512x384 | 128 | 24.66 | 22.92 | Done |
| `overfit_depthloss_cross_attn_coffee_martini` | 512x384 | 128 | 24.55 | 23.99 | Done |
| **`overfit_k64_cross_attn_coffee_martini`** | 512x384 | **64** | **25.20** | **24.51** | **Done (best)** |
| `overfit_contig4_cross_attn_coffee_martini` | 512x384 | 128 | 23.76 | 23.05 | Stage 2 only |
| `overfit_k256_cross_attn_coffee_martini` | 512x384 | 256 | — | — | Ready |
| `overfit_coordnorm_hybrid_cross_attn_coffee_martini` | 512x384 | 128 | — | — | Ready |
| `overfit_dpt_full_cross_attn_coffee_martini` | 512x384 | 128 | — | — | Ready |
| `overfit_dpt_hybrid_cross_attn_coffee_martini` | 512x384 | 128 | — | — | Ready |

Full results with per-experiment lessons in [`experiments/leaderboard.csv`](experiments/leaderboard.csv).
