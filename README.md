# Decoding4DGaussians

Feed-forward dynamic 3D Gaussian splatting from monocular video. A frozen SpaTrackerV2 backbone extracts per-frame tokens and 3D point maps, which MLP decoder heads convert into dynamic Gaussians — no per-scene optimization required.

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
cd experiments/overfit_highres_highcount_coffee_martini

# 1. Precompute SpaTrackerV2 features (once per scene/resolution)
python src/precompute.py --scene_dir ../../datasets/neu3d/coffee_martini --cache_dir cache

# 2. Train (3 stages: canonical -> deformation -> joint)
python src/train.py

# 3. Evaluate (renders video, computes PSNR/SSIM, exports PLY)
python src/eval.py

# 4. Interactive viewer (opens at http://localhost:8080)
python src/viewer.py
```

Edit `config.yaml` to change resolution, Gaussian count (K), learning rates, and training schedule.

## Experiments

| Directory | Resolution | K | Description |
|---|---|---|---|
| `overfit_coffee_martini` | 512x384 | 128 | Baseline with aggregator-only deformation |
| `overfit_cross_attn_coffee_martini` | 512x384 | 128 | Cross-attention deformation head |
| `overfit_highres_cross_attn_coffee_martini` | 1024x768 | 128 | High-res with resolution-aware scaling |
| `overfit_highres_highcount_coffee_martini` | 1024x768 | 512 | High-res + 4x Gaussians (best quality) |
