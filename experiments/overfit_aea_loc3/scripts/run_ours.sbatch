#!/usr/bin/env bash
#SBATCH --job-name=aea-ours
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=experiments/overfit_aea_loc3/logs/slurm-%j.out

set -euxo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate aria-compare

REPO_ROOT="$(pwd)"
CACHE="$REPO_ROOT/experiments/overfit_aea_loc3/cache"
SEQ_ROOT="$REPO_ROOT/datasets/aria/loc3_script3_seq1_rec1/recording/camera-rgb-rectified-600-h1000"

# 1. Precompute STV2 features (once)
if [[ ! -f "$CACHE/tokens.pt" ]]; then
  python experiments/overfit_aea_loc3/src/precompute.py \
    --seq_root "$SEQ_ROOT" --cache_dir "$CACHE" \
    --width 504 --height 504 --num_frames 128 --window_size 8
fi

# 2. Train 3 stages
python experiments/overfit_aea_loc3/src/train.py \
  experiments/overfit_aea_loc3/config.yaml

# 3. Eval under both protocols
python experiments/overfit_aea_loc3/src/eval.py \
  experiments/overfit_aea_loc3/config.yaml --protocol 4dgt_16_112
python experiments/overfit_aea_loc3/src/eval.py \
  experiments/overfit_aea_loc3/config.yaml --protocol 64_64

echo "=== ours job complete ==="
