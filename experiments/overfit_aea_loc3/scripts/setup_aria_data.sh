#!/usr/bin/env bash
# Download AEA loc3_script3_seq1_rec1 VRS + run 4DGT's VRS preprocessing.
# Produces: datasets/aria/loc3_script3_seq1_rec1/recording/camera-rgb-rectified-600-h1000/{images,transforms.json}
#
# Pre-req: AEA access agreement accepted at https://www.projectaria.com/datasets/aea/ and
# a valid CDN URL file downloaded to $AEA_CDN_URLS (set below).
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SEQ="loc3_script3_seq1_rec1"
DATA_DIR="datasets/aria/${SEQ}"
AEA_CDN_URLS="${AEA_CDN_URLS:-${HOME}/aria_access/aea_urls.json}"

if [[ ! -f "$AEA_CDN_URLS" ]]; then
  echo "FATAL: AEA_CDN_URLS=$AEA_CDN_URLS not found." >&2
  echo "Fetch your CDN URLs from https://www.projectaria.com/datasets/aea/ first." >&2
  exit 1
fi

mkdir -p "$DATA_DIR"

# Step 1: download VRS + MPS artifacts via aria_dataset_downloader
aria_dataset_downloader \
  --cdn-file "$AEA_CDN_URLS" \
  --output-folder "$DATA_DIR" \
  --sequence-names "$SEQ" \
  --data-types 0 1 2 3 4    # 0=main_vrs 1=mps_slam_trajectory 2=mps_slam_points 3=mps_slam_calibration 4=speech2text

# Step 2: rectify RGB + write transforms.json using 4DGT's preprocessing
cd 4DGT
bash tlod/scripts/run_vrs_preprocessing.sh \
  "../$DATA_DIR/$SEQ/recording.vrs" \
  "../$DATA_DIR/$SEQ/mps/slam/closed_loop_trajectory.csv" \
  "../$DATA_DIR/$SEQ/mps/slam/online_calibration.jsonl" \
  "../$DATA_DIR/$SEQ/mps/slam/semidense_points.csv.gz" \
  "../$DATA_DIR/$SEQ/recording"
cd -

# Step 3: sanity check
python - <<'PY'
import json, os, sys
seq = "datasets/aria/loc3_script3_seq1_rec1/recording/camera-rgb-rectified-600-h1000"
tj = os.path.join(seq, "transforms.json")
with open(tj) as f:
    data = json.load(f)
frames = data["frames"]
print(f"Frame count: {len(frames)}")
print(f"First frame keys: {sorted(frames[0].keys())}")
print(f"First frame fx/fy/cx/cy: {frames[0]['fx']:.1f}/{frames[0]['fy']:.1f}/{frames[0]['cx']:.1f}/{frames[0]['cy']:.1f}")
print(f"First frame image exists: {os.path.exists(os.path.join(seq, frames[0]['image_path']))}")
assert len(frames) > 1000, "Expected >1000 frames for loc3_script3_seq1_rec1"
print("OK")
PY
