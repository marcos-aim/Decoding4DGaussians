# Cluster runbook: overfit_aea_loc3 (Aria vs 4DGT)

Paste this entire file into your cluster Claude Code session as context.

---

## Task

Run a head-to-head comparison between our feed-forward Gaussian-splat system (DecodeGaussians) and Meta's 4DGT on the Project Aria `loc3_script3_seq1_rec1` sequence. Both systems consume the **same monocular 128-frame window** and render the same timestamps. Metrics (PSNR / SSIM / LPIPS) are computed on held-out frames using 4DGT's two protocols (`4dgt_16_112` and `64_64`). Results land as Markdown tables in `docs/superpowers/plans/`.

The experiment code lives under `experiments/overfit_aea_loc3/` on `main`. It has already been implemented and smoke-tested locally on a synthetic Neu3D-as-Aria sequence. **Your job on the cluster is to run it end-to-end on real Aria data.**

---

## Step 0 — Get the latest code

Experiments live directly on `main` in this repo (no per-experiment branches). If you don't have the repo yet:

```bash
git clone git@github.com:<org>/DecodeGaussians.git
cd DecodeGaussians
```

Otherwise update:

```bash
cd <path-to>/DecodeGaussians
git checkout main
git pull --ff-only
git log --oneline -5        # top commit should be `fix(aea_loc3): add nodefaults to aria-compare channels...`
```

Update submodules:

```bash
git submodule update --init --recursive
```

---

## Step 1 — Clone 4DGT (if not already in the repo)

4DGT is Meta's NeurIPS 2025 feed-forward 4D Gaussian Transformer. Code is at:

  **https://github.com/facebookresearch/4dgt**

(Paper arXiv:2506.08015. Pretrained ckpt on HuggingFace: `projectaria/4DGT`.)

If the URL ever changes, websearch "4DGT facebookresearch github" — the package should be named `tlod` and have a `tlod/run.py` entrypoint.

```bash
cd <path-to>/DecodeGaussians
if [ ! -d 4DGT ]; then
  git clone https://github.com/facebookresearch/4dgt.git 4DGT
fi
```

Do **not** add 4DGT to the repo's submodules — our scripts assume it is a sibling directory at `DecodeGaussians/4DGT/`.

---

## Step 2 — Environment

There is a single dedicated conda env `aria-compare` for this experiment. Spec: `experiments/overfit_aea_loc3/environment.aria-compare.yml`. CUDA 12.4 / torch 2.4.1+cu124 / xformers 0.0.28 / flash-attn 2.6.3 / projectaria_tools 1.5.5 / gsplat 1.4.0.

Run the bundled setup script (idempotent — creates env if missing, updates if present, builds diff-gaussian-rasterization from our 4DGaussians submodule, downloads the 4DGT pretrained checkpoint, and smoke-tests CUDA + projectaria_tools + VGGT4Track):

```bash
bash experiments/overfit_aea_loc3/scripts/setup_env.sh
```

Then activate:

```bash
conda activate aria-compare
```

The 4DGT checkpoint will land at `4DGT/checkpoints/4dgt_full.pth` (≈2 GB).

---

## Step 3 — Download + preprocess the Aria sequence

Aria Everyday Activities (AEA) is gated behind a Meta form. You need to:

1. Request access at https://www.projectaria.com/datasets/aea/ (Meta reviews in ~1 day).
2. Get the AEA download CLI: `pip install projectaria-tools[all]`.
3. Run our preprocess script:

```bash
bash experiments/overfit_aea_loc3/scripts/setup_aria_data.sh
```

This fetches `loc3_script3_seq1_rec1.vrs`, runs `tlod/scripts/run_vrs_preprocessing.sh` from the 4DGT repo, and writes the output into:

```
datasets/aria/loc3_script3_seq1_rec1/recording/camera-rgb-rectified-600-h1000/
  transforms.json   # per-frame 4x4 c2w + fx/fy/cx/cy + image_path + timestamp
  images/*.png      # rectified RGB, 600×1000
```

If AEA access is not yet approved but you want to dry-run the pipeline, you can substitute any 4DGT-format monocular sequence — just point `experiment.paths.aria_seq_root` in the config at its `camera-rgb-rectified-600-h1000/` directory.

---

## Step 4 — Understand the config

Read `experiments/overfit_aea_loc3/config.yaml` once. Key fields:

- `data.resolution: [504, 504]` — 4DGT default. **Do not change** without also rerunning STV2 precompute; the STV2 cache is resolution-dependent.
- `data.num_frames: 128` — 4DGT paper window. `data.input_subsample: 8` gives N=16 input frames.
- `training.stage1.steps: 20000`, `stage2.steps: 20000`, `stage3.steps: 10000` — 50 K steps total.
- `training.scale_anneal_target: 1.0` — flagged as a known issue (disables the resolution-scale coupling that CLAUDE.md calls out). Leave alone for this first run; we'll iterate after getting baseline numbers.
- `model.canonical.grid_h` / `grid_w` — intentionally **not set**. `build_heads()` in `train.py` derives grid from the actual patch count `P` in the cached STV2 tokens (VGGT internally pads to 518×518 → `P=1374`). Do not hardcode.
- `vram.total_gb: 80.0` — A100/H100 assumption.

---

## Step 5 — Precompute STV2 features (one-shot, ~5 min)

```bash
cd <path-to>/DecodeGaussians
python experiments/overfit_aea_loc3/src/precompute.py \
  --seq_root datasets/aria/loc3_script3_seq1_rec1/recording/camera-rgb-rectified-600-h1000 \
  --cache_dir experiments/overfit_aea_loc3/cache \
  --width 504 --height 504 \
  --num_frames 128 --start_frame 0 \
  --window_size 8
```

This loads `VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")`, runs forward passes in 8-frame windows, captures the aggregator tokens via a forward hook, and saves:

```
experiments/overfit_aea_loc3/cache/
  tokens.pt         [128, 1374, 2048]   fp16
  points_map.pt     [128, 504, 504, 3]  fp16 (aligned to Aria metric world frame)
  poses.pt          [128, 4, 4]         STV2-estimated (informational)
  intrs.pt          [128, 3, 3]         STV2-estimated
  unc_metric.pt     [128, 504, 504]     fp16
  frame_indices.pt  [128]
  aria_c2w.pt       [128, 4, 4]         GT Aria poses
  aria_K.pt         [128, 3, 3]         GT Aria intrinsics scaled to 504×504
  align_meta.pt     {scale, near_aria, z_near_stv2}
```

Alignment sets STV2 cam-0 → Aria cam-0 via rigid transform plus a scale factor so the 5th-percentile STV2 depth matches Aria's near bound (default 0.3 m).

**VRAM:** measured peak 11 GB at 252×252 / K=32 locally. Cluster extrapolation to 504×504 / K=128 ≈ 15–20 GB (STV2 dominates and is roughly res-independent because VGGT resizes to 518 internally). A100-80GB has ample headroom.

**Expected tensor shapes (sanity-check against smoke test):**
```
[precompute]   tokens     (8, 1374, 2048)  torch.float16
[precompute]   points_map (8, 504, 504, 3) torch.float32
[build_heads] P=1374 K=128 grid=37x38 res_scale=1.016 ...
```

---

## Step 6 — Train (50 K steps, ~8–12 h on A100)

Two options. Pick one.

### 6a — Direct run (foreground, recommended first time)

```bash
python experiments/overfit_aea_loc3/src/train.py \
  experiments/overfit_aea_loc3/config.yaml
```

Checkpoints land every 2000 steps in `experiments/overfit_aea_loc3/checkpoints/`. TensorBoard logs in `experiments/overfit_aea_loc3/logs/`.

### 6b — SLURM submit

```bash
sbatch experiments/overfit_aea_loc3/scripts/run_ours.sbatch
```

(Edit `run_ours.sbatch` header for your partition/account if needed.)

### Training loop behavior to know about

- **Monocular — one supervised frame per step**, picked at random. There is no multi-view supervision — we only have the single Aria RGB stream.
- **TV loss is disabled** in stages 2 & 3 (`tv: 0.0` in config) because TV compares consecutive-frame deltas and the monocular loop samples one frame per step, so there is no valid pair. This is intentional; do not re-enable without adding a two-frame sampler.
- Stage-gate behavior:
  - Stage 1 (20 K): trains `canonical_head` only, `deformation_head` frozen.
  - Stage 2 (20 K): trains `deformation_head` only, `canonical_head` frozen.
  - Stage 3 (10 K): joint refinement.
- Resume with `--resume_stage {2,3}` — it will load `stage{N-1}_final.pt` and pick up.

### Sanity checks during training

- First 100 steps: loss should drop monotonically, PSNR should climb above ~15 dB.
- If PSNR is stuck at ~6 dB the entire first epoch, alignment probably failed — re-inspect `align_meta.pt` and check that `scale` is a reasonable positive number.
- Peak VRAM during render ≈ 3–5 GB on top of the ~5 GB steady model state. If OOM during Stage 2/3, reduce `num_gaussians_per_patch` (currently K=128) to K=64 in the config.

---

## Step 7 — Evaluate our system (both protocols)

```bash
python experiments/overfit_aea_loc3/src/eval.py \
  experiments/overfit_aea_loc3/config.yaml \
  --protocol 4dgt_16_112 \
  --checkpoint stage3_final.pt

python experiments/overfit_aea_loc3/src/eval.py \
  experiments/overfit_aea_loc3/config.yaml \
  --protocol 64_64 \
  --checkpoint stage3_final.pt
```

Outputs per protocol: `experiments/overfit_aea_loc3/renders/<protocol>/`:
- `summary.json` — mean/min/max PSNR/SSIM/LPIPS over **held-out** frames only
- `metrics.json` — per-frame
- `input/*.jpg`, `heldout/*.jpg` — rendered images

`4dgt_16_112`: every 8th frame is input (16 total); metrics on the other 112.
`64_64`: every other frame is input; metrics on the other 64 (the 4DGT ablation protocol).

---

## Step 8 — Run the 4DGT baseline on the same window

```bash
bash experiments/overfit_aea_loc3/scripts/run_4dgt.sh
```

This `cd`s into `4DGT/` and invokes `python -m tlod.run` with matched `input_image_res=504`, `image_num_per_batch=128`, `output_image_num=128`, `sample_interval=128`, `start_frame=0`. Output images and JSON per-frame metrics land at `experiments/overfit_aea_loc3/4dgt_baseline/`.

If `tlod.run` errors on missing config fields, check `4DGT/configs/config.yaml` against our CLI overrides — the upstream config key names may have drifted since we wrote the script.

---

## Step 9 — Extract matching metrics + compare

```bash
for PROTO in 4dgt_16_112 64_64; do
  python experiments/overfit_aea_loc3/scripts/extract_4dgt_heldout_frames.py \
    --fourdgt_out experiments/overfit_aea_loc3/4dgt_baseline \
    --seq_name loc3_script3_seq1_rec1 \
    --aria_seq_root datasets/aria/loc3_script3_seq1_rec1/recording/camera-rgb-rectified-600-h1000 \
    --protocol "$PROTO" \
    --out_json experiments/overfit_aea_loc3/4dgt_baseline/metrics_${PROTO}.json

  python experiments/overfit_aea_loc3/scripts/compare_metrics.py \
    --ours_summary experiments/overfit_aea_loc3/renders/${PROTO}/summary.json \
    --fourdgt_summary experiments/overfit_aea_loc3/4dgt_baseline/metrics_${PROTO}.json \
    --out_md docs/superpowers/plans/2026-04-17-aria-4dgt-results-${PROTO}.md
done
```

Final deliverable: two Markdown tables at `docs/superpowers/plans/2026-04-17-aria-4dgt-results-*.md` with side-by-side mean PSNR / SSIM / LPIPS on held-out frames.

### One-shot wrapper (sbatch + 4DGT + compare)

```bash
bash experiments/overfit_aea_loc3/scripts/run_full_comparison.sh
```

Polls SLURM every 60 s until the train job finishes, then runs 4DGT, then compare. Use this if the cluster is set up and you want to queue+walk-away.

---

## Known caveats to flag when reporting results

1. **Input-advantage asymmetry.** Our STV2 precompute sees **all 128 frames** of the window; 4DGT only sees 16. Our numbers are therefore **optimistic** vs. a strict "same-inputs" comparison. Label tables clearly ("Ours with N=128 precompute input" vs "4DGT with N=16 input"). If the human wants strict parity, the precompute needs to be restricted to the 16-frame subset — that's a follow-up, not part of this run.
2. **`scale_anneal_target=1.0`** — CLAUDE.md flags that resolution-scale coupling should be on. The first run inherits 1.0 by design (stable); iterate after baseline.
3. **AEA near bound fallback.** If MPS semidense points aren't available, `precompute.py` falls back to `ARIA_NEAR_DEFAULT = 0.3 m`. For most indoor AEA sequences this is fine; for outdoor sequences it won't be.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `HybridDPTCanonicalGaussianHead.__init__() got an unexpected keyword argument 'num_patches'` | Old cached code | `git pull`. The constructor takes `grid_h/grid_w/init_xyz/init_xyz_per_gaussian/init_log_scale/spread/dpt_dim`. `build_heads()` in `train.py` handles this. |
| `ModuleNotFoundError: models` in precompute | `SpaTrackerV2/` not on path | precompute.py adds it via `sys.path.insert` — if you moved the script, keep the `REPO_ROOT` math intact. |
| PSNR stuck at 6 dB all of Stage 1 | Alignment failure | Check `align_meta.pt`: `scale` must be positive and finite; `z_near_stv2` must be positive. Try bumping `--start_frame` to avoid static-camera windows where STV2 depth is unreliable. |
| CUDA OOM in precompute | Window too large | precompute already auto-halves on OOM; if it still fails, pass `--window_size 4`. |
| `tlod.run` errors on hydra key | 4DGT upstream config drift | Diff `4DGT/configs/config.yaml` against `scripts/run_4dgt.sh` CLI overrides. |

---

## Handoff checklist for when you're done

- [ ] Both `renders/4dgt_16_112/summary.json` and `renders/64_64/summary.json` exist and have non-null metrics.
- [ ] Both `4dgt_baseline/metrics_*.json` exist.
- [ ] Both comparison markdown files written under `docs/superpowers/plans/`.
- [ ] TensorBoard log directory (`experiments/overfit_aea_loc3/logs/`) is copied off-cluster if the cluster's scratch is ephemeral.
- [ ] `stage3_final.pt` checkpoint preserved (~1–2 GB).
- [ ] Any deviations from this runbook (e.g., you had to reduce K, change num_frames, swap sequence) noted in the final report.
