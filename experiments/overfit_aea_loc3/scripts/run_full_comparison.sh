#!/usr/bin/env bash
# Orchestrate the full head-to-head: our system + 4DGT + metrics compare.
# Assumes env already set up (task 0) and Aria data downloaded (task 1).
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# 1. Our system — blocking SLURM submit (or direct if interactive)
if command -v sbatch >/dev/null 2>&1; then
  OURS_JOB=$(sbatch --parsable experiments/overfit_aea_loc3/scripts/run_ours.sbatch)
  echo "Submitted ours as job $OURS_JOB"
  # Wait for completion
  while squeue -j "$OURS_JOB" | grep -q "$OURS_JOB"; do sleep 60; done
else
  bash experiments/overfit_aea_loc3/scripts/run_ours.sbatch
fi

# 2. 4DGT baseline
bash experiments/overfit_aea_loc3/scripts/run_4dgt.sh

# 3. Extract 4DGT metrics
for PROTO in 4dgt_16_112 64_64; do
  python experiments/overfit_aea_loc3/scripts/extract_4dgt_heldout_frames.py \
    --fourdgt_out experiments/overfit_aea_loc3/4dgt_baseline \
    --seq_name loc3_script3_seq1_rec1 \
    --aria_seq_root datasets/aria/loc3_script3_seq1_rec1/recording/camera-rgb-rectified-600-h1000 \
    --protocol "$PROTO" \
    --out_json experiments/overfit_aea_loc3/4dgt_baseline/metrics_${PROTO}.json

  # 4. Compare
  python experiments/overfit_aea_loc3/scripts/compare_metrics.py \
    --ours_summary experiments/overfit_aea_loc3/renders/${PROTO}/summary.json \
    --fourdgt_summary experiments/overfit_aea_loc3/4dgt_baseline/metrics_${PROTO}.json \
    --out_md docs/superpowers/plans/2026-04-17-aria-4dgt-results-${PROTO}.md
done

echo "=== full comparison complete — see docs/superpowers/plans/2026-04-17-aria-4dgt-results-*.md ==="
