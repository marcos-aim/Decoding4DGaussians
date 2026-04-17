#!/usr/bin/env bash
set -euxo pipefail

# Cluster environment setup for Aria vs 4DGT comparison experiment.
# Run once on the head node (or login node with internet access) before any jobs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_NAME="aria-compare"

cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "FATAL: conda not on PATH. Module-load miniconda first." >&2
  exit 1
fi

conda env create -f experiments/overfit_aea_loc3/environment.aria-compare.yml || \
  conda env update -f experiments/overfit_aea_loc3/environment.aria-compare.yml --prune

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Build diff_gaussian_rasterization (required for our renderer)
pip install --no-build-isolation ./4DGaussians/submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./4DGaussians/submodules/simple-knn

# Download 4DGT pretrained checkpoint
cd 4DGT
python -m tlod.download_model
cd -

# Sanity-check CUDA
python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"
# Sanity-check projectaria_tools
python -c "import projectaria_tools.core as pa; print('projectaria_tools OK')"
# Sanity-check VGGT4Track download
python -c "
import sys; sys.path.insert(0, 'SpaTrackerV2')
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
m = VGGT4Track.from_pretrained('Yuxihenry/SpatialTrackerV2_Front')
print('VGGT4Track OK')
"
echo '=== env setup complete ==='
