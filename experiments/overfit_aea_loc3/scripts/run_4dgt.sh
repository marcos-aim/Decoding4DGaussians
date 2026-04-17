#!/usr/bin/env bash
# Run 4DGT pretrained on the same 128-frame Aria window.
# Output: 4DGT renders for all 128 timestamps (same window) from the same
# set of cameras our system sees.
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SEQ_NAME="loc3_script3_seq1_rec1"
ARIA_DIR="$REPO_ROOT/datasets/aria/${SEQ_NAME}"
OUT_DIR="$REPO_ROOT/experiments/overfit_aea_loc3/4dgt_baseline"

mkdir -p "$OUT_DIR"

cd "$REPO_ROOT/4DGT"
python -m tlod.run \
  data_path="../datasets/aria" \
  seq_list=["${SEQ_NAME}"] \
  output_dir="${OUT_DIR}" \
  model.checkpoint="checkpoints/4dgt_full.pth" \
  input_image_res=504 \
  image_num_per_batch=128 \
  output_image_num=128 \
  sample_interval=128 \
  loaded_to_seconds=1e9 \
  loaded_to_meters=1.0 \
  start_frame=0
cd -

echo "=== 4DGT baseline run complete. Outputs at $OUT_DIR ==="
