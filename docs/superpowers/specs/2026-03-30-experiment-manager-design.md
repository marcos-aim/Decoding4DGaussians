# Experiment Manager Design Spec
**Date:** 2026-03-30
**Project:** DecodeGaussians
**Status:** Approved — ready for implementation

---

## 1. Problem Statement

The `experiments/` directory contains many self-contained experiment directories. Running them currently requires manual babysitting: launching train.py by hand, watching VRAM, deciding when to start the next job. There is no queue, no VRAM-aware packing, no dashboard, and no durable state.

This spec defines an **experiment management system** — `expmanager` — that wraps the existing experiment workflow without modifying it.

---

## 2. Goals

- Automatically discover experiments from `experiments/`
- Queue and schedule jobs based on available GPU VRAM
- Support multiple concurrent jobs on one GPU when safe (conservative packing)
- Design for multi-GPU servers from the start (single-GPU is the degenerate case)
- Provide a Streamlit dashboard showing live status, loss curves, renders, and metrics
- All control actions go through a backend layer; Streamlit is a thin frontend
- Durable SQLite state survives scheduler restarts
- Zero changes to existing experiment directories

---

## 3. Non-Goals (v1)

- REST API or remote control
- Full daemon / systemd integration (upgrade path exists, not v1)
- Full event sourcing
- Multi-machine distributed scheduling
- Experiment directory modification (configs, train scripts, etc.)

---

## 4. Architecture Overview

```
D:/DecodeGaussians/
├── .expmanager/                  ← gitignored runtime dir
│   ├── expmanager.db             ← SQLite (WAL mode)
│   └── expmanager.log            ← scheduler process log
├── expmanager/                   ← new top-level package
│   ├── __init__.py
│   ├── __main__.py               ← python -m expmanager scheduler run
│   ├── expmanager.yaml           ← manager-level config
│   ├── db/
│   │   ├── schema.py             ← schema, migrations, WAL setup
│   │   └── connection.py         ← thread-safe SQLite connection
│   ├── core/
│   │   ├── types.py              ← JobSpec, RunStateSnapshot, GpuSnapshot dataclasses
│   │   ├── discovery.py          ← scan experiments/, infer jobs from config.yaml
│   │   ├── metadata.py           ← config.yaml → JobSpec
│   │   ├── gpu.py                ← pynvml / nvidia-smi / estimate-only backends
│   │   ├── scheduler.py          ← main 5s tick loop
│   │   ├── launcher.py           ← subprocess launch with CUDA_VISIBLE_DEVICES
│   │   ├── monitor.py            ← process watchdog + artifact indexing (two concerns, one module for v1)
│   │   └── actions.py            ← public control API (enqueue/cancel/retry/etc.)
│   └── dashboard/
│       ├── app.py                ← Streamlit entry point
│       ├── state_reader.py       ← read-only DB + filesystem queries
│       └── components/
│           ├── job_table.py
│           ├── run_detail.py
│           ├── gpu_panel.py
│           ├── loss_curves.py
│           └── log_viewer.py
└── experiments/                  ← UNTOUCHED
```

### Data flow

```
experiments/X/config.yaml ──discovery──► jobs table (job_id = hash of experiment_dir)

scheduler tick:
  GPU state → find fitting GPU → launch subprocess
  subprocess env: CUDA_VISIBLE_DEVICES=N, cwd=experiments/X/src/
  subprocess writes: logs/, renders/, checkpoints/ (unchanged experiment layout)

monitor thread (per active run):
  reads TensorBoard events → run_state (upsert)
  reads checkpoint sentinels → stage detection
  reads renders/stageN/*.jpg → last_render_path
  checks PID liveness

Streamlit dashboard:
  reads SQLite (jobs, runs, run_state, logs)
  reads renders/ JPEGs directly for image display
  reads TensorBoard events for detailed loss curves
  control buttons → actions.py → SQLite + operator_log
```

---

## 5. Manager Configuration

**File:** `expmanager/expmanager.yaml`

```yaml
scheduler:
  tick_interval_seconds: 5
  packing_policy: pack_conservative   # sequential | one_per_gpu | pack_conservative
  vram_safety_margin: 0.20            # fractional overhead added to effective estimate
  low_confidence_multiplier: 1.30     # multiplier when vram_confidence = low
  stale_heartbeat_seconds: 30         # mark run stale if monitor has not updated for this long
  max_concurrent_per_gpu: 4           # hard cap regardless of VRAM fit

discovery:
  experiments_root: D:/DecodeGaussians/experiments
  config_filename: config.yaml
  exclude_dirs: []

launcher:
  python_venv: D:/DecodeGaussians/SpaTrackerV2/.venv
  train_script: src/train.py
  config_arg: --config

gpu:
  monitoring_backend: auto            # auto | pynvml | nvidia_smi | estimate_only
  reserved_system_vram_gb: 0.5

paths:
  db: .expmanager/expmanager.db
  log: .expmanager/expmanager.log
```

---

## 6. Shared Types

**File:** `expmanager/core/types.py`

```python
@dataclass
class JobSpec:
    job_id: str                        # sha1 of experiment_dir
    experiment_name: str               # from config.experiment.name
    experiment_dir: Path
    config_path: Path
    resolution: tuple[int, int]        # (W, H) from config.data.resolution
    k_gaussians: int                   # config.model.num_gaussians_per_patch
    total_gaussians: int
    stage_steps: dict[str, int]        # {"stage1": 20000, "stage2": 15000, "stage3": 10000}
    estimated_vram_gb: float           # from config.vram.total_gb or formula
    vram_confidence: str               # low | medium | high
    mode: str                          # config.experiment.mode
    notes: str

@dataclass
class GpuSnapshot:
    gpu_id: int
    name: str
    total_vram_gb: float
    used_vram_gb: float                # measured (pynvml/nvidia-smi) or None
    free_vram_gb: float | None         # measured free; None if estimate_only
    reserved_by_scheduler_gb: float    # sum of reserved_vram_gb for running jobs
    running_run_ids: list[str]
    monitoring_confidence: str         # pynvml | nvidia_smi | estimate_only
    sampled_at: datetime

@dataclass
class RunStateSnapshot:
    run_id: str
    current_stage: int
    completed_stage: int               # highest stage with a _final.pt checkpoint
    current_step: int
    total_steps: int
    latest_loss: float | None
    latest_psnr: float | None
    latest_novel_psnr: float | None
    latest_vram_alloc_gb: float | None
    latest_vram_peak_gb: float | None
    latest_lr: float | None
    eta_seconds: int | None
    stage1_complete: bool
    stage2_complete: bool
    stage3_complete: bool
    last_render_path: str | None
    last_heartbeat: datetime
```

---

## 7. SQLite Schema

**WAL mode + foreign keys enabled on every connection.**

```sql
CREATE TABLE jobs (
    job_id                TEXT PRIMARY KEY,
    experiment_name       TEXT NOT NULL,
    experiment_dir        TEXT NOT NULL UNIQUE,
    config_path           TEXT NOT NULL,
    status                TEXT NOT NULL DEFAULT 'discovered',
    -- discovered | pending | queued | running | completed | failed | cancelled | blocked
    priority              INTEGER NOT NULL DEFAULT 50,
    estimated_vram_gb     REAL,
    observed_peak_vram_gb REAL,
    vram_confidence       TEXT NOT NULL DEFAULT 'low',   -- low | medium | high | oom_observed
    resolution            TEXT,          -- "512x384"
    k_gaussians           INTEGER,
    total_gaussians       INTEGER,
    stage_steps           TEXT,          -- JSON
    mode                  TEXT,
    notes                 TEXT,
    tags                  TEXT,          -- JSON array
    discovered_at         TEXT NOT NULL,
    last_seen_at          TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);

CREATE TABLE runs (
    run_id                TEXT PRIMARY KEY,
    job_id                TEXT NOT NULL REFERENCES jobs(job_id),
    attempt               INTEGER NOT NULL DEFAULT 1,
    status                TEXT NOT NULL DEFAULT 'pending',
    -- pending | launching | running | completed | failed | cancelled | oom
    gpu_id                INTEGER,
    pid                   INTEGER,
    cuda_visible_devices  TEXT,
    resume_stage          INTEGER NOT NULL DEFAULT 1,
    estimated_vram_gb     REAL,          -- snapshot at launch time
    reserved_vram_gb      REAL,          -- effective reservation (estimate * multiplier * margin)
    observed_peak_vram_gb REAL,          -- from run_state at end
    launch_command        TEXT,          -- exact command string for reproducibility
    working_dir           TEXT,          -- cwd used for launch
    python_executable     TEXT,          -- full path to python binary used
    started_at            TEXT,
    completed_at          TEXT,
    exit_code             INTEGER,
    failure_reason        TEXT,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);

CREATE TABLE run_state (
    run_id                TEXT PRIMARY KEY REFERENCES runs(run_id),
    current_stage         INTEGER,
    completed_stage       INTEGER,       -- highest stage with _final.pt present
    current_step          INTEGER,
    total_steps           INTEGER,
    latest_loss           REAL,
    latest_psnr           REAL,
    latest_novel_psnr     REAL,
    latest_vram_alloc_gb  REAL,
    latest_vram_peak_gb   REAL,
    latest_lr             REAL,
    eta_seconds           INTEGER,       -- approximate; uses current-stage rate
    last_render_path      TEXT,
    last_heartbeat        TEXT,
    stage1_complete       INTEGER NOT NULL DEFAULT 0,
    stage2_complete       INTEGER NOT NULL DEFAULT 0,
    stage3_complete       INTEGER NOT NULL DEFAULT 0,
    updated_at            TEXT NOT NULL
);

CREATE TABLE operator_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action    TEXT NOT NULL,
    -- enqueue | cancel | retry | reprioritize | pause | resume | discover | block | unblock
    job_id    TEXT,
    run_id    TEXT,
    actor     TEXT NOT NULL DEFAULT 'user',   -- user | scheduler
    payload   TEXT,          -- JSON
    result    TEXT           -- ok | error:<reason>
);

CREATE TABLE scheduler_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event     TEXT NOT NULL,
    -- launched | completed | failed | oom | tick | gpu_snapshot | stale_run_detected | stage_transition
    job_id    TEXT,
    run_id    TEXT,
    payload   TEXT           -- JSON
);

-- Indexes for common dashboard queries
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_runs_job_id ON runs(job_id);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_operator_log_job ON operator_log(job_id, timestamp);
CREATE INDEX idx_scheduler_log_run ON scheduler_log(run_id, timestamp);
```

---

## 8. Scheduler Logic

### 8.1 Main tick loop

```
every tick_interval_seconds:
  1. refresh_gpu_state()
     - query each GPU via monitoring backend
     - cross-reference reserved_vram from scheduler accounting
     - if telemetry available: use min(measured_free, total - reserved) as headroom
     - if estimate_only: use total - reserved_system - sum(reserved_vram for running runs)

  2. reconcile_running_runs()
     - for each run with status=running:
       * check os.kill(pid, 0) liveness
       * check last_heartbeat age vs stale_heartbeat_seconds
       * if dead: read exit_code, classify (completed / failed / oom), update DB
       * if stale but alive: log warning, continue

  3. pull_queued_jobs()
     - SELECT jobs WHERE status='queued' ORDER BY priority DESC, created_at ASC

  4. for each queued job:
       gpu = find_fitting_gpu(job)
       if gpu is not None:
         launch(job, gpu)
         break   # one launch per tick to avoid race conditions

  5. log scheduler_log tick event with gpu_snapshot JSON
```

### 8.2 VRAM fit decision

```python
def find_fitting_gpu(job: JobSpec, gpus: list[GpuSnapshot]) -> GpuSnapshot | None:
    effective_estimate = job.estimated_vram_gb
    if job.vram_confidence == "low":
        effective_estimate *= cfg.low_confidence_multiplier  # 1.3x

    needed = effective_estimate * (1 + cfg.vram_safety_margin)

    for gpu in sorted(gpus, key=lambda g: g.free_vram_gb or 0, reverse=True):
        # Scheduler-side reservation accounting
        scheduler_headroom = (
            gpu.total_vram_gb
            - cfg.reserved_system_vram_gb
            - gpu.reserved_by_scheduler_gb
        )
        # If live telemetry is available, use the more conservative of the two
        if gpu.free_vram_gb is not None:
            measured_headroom = gpu.free_vram_gb - cfg.reserved_system_vram_gb
            headroom = min(scheduler_headroom, measured_headroom)
        else:
            headroom = scheduler_headroom

        fits_vram = needed <= headroom
        fits_cap = len(gpu.running_run_ids) < cfg.max_concurrent_per_gpu
        if fits_vram and fits_cap:
            return gpu
    return None
```

### 8.3 VRAM estimate source priority

| Source | Confidence | When used |
|--------|-----------|-----------|
| `observed_peak_vram_gb` from prior completed run | `high` | Always preferred |
| `vram.total_gb` from experiment `config.yaml` | `medium` | No observed peak yet |
| Formula: `base + K * res_scale` | `low` | No config hint; triggers 1.3x multiplier |

### 8.4 OOM handling

- Detect via: exit code 137, or `CUDA out of memory` in stderr (monitor reads first ~4KB of stderr)
- Set `runs.status = 'oom'`
- Set `jobs.vram_confidence = 'oom_observed'`
- **Do not** set `observed_peak_vram_gb` to total GPU VRAM automatically
- Instead: increase `reserved_vram_gb` by 25% for next attempt, log the OOM event
- Require explicit `actions.retry()` — no auto-retry for OOM
- Log to both `operator_log` and `scheduler_log`

### 8.5 Packing policies

| Policy | Behavior |
|--------|---------|
| `sequential` | One job running at a time across all GPUs |
| `one_per_gpu` | One job per GPU, no sharing |
| `pack_conservative` | Multiple jobs per GPU when fit is confident (default) |

v1 implements `sequential`, `one_per_gpu`, and `pack_conservative`.

---

## 9. Launcher

```python
def launch(job: JobSpec, gpu: GpuSnapshot, resume_stage: int = 1) -> Run:
    venv_python = Path(cfg.launcher.python_venv) / "Scripts" / "python"  # Windows
    # fallback: "bin/python" for Linux/Mac

    src_dir = job.experiment_dir / "src"
    cmd = [
        str(venv_python),
        "train.py",
        "--config", str(job.config_path),
        "--resume-stage", str(resume_stage),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu.gpu_id)

    proc = subprocess.Popen(
        cmd,
        cwd=str(src_dir),
        env=env,
        stdout=open(job.experiment_dir / "train.log", "a"),
        stderr=subprocess.STDOUT,
    )

    run = Run(
        run_id=str(uuid4()),
        job_id=job.job_id,
        attempt=next_attempt_number(job.job_id),
        status="running",
        gpu_id=gpu.gpu_id,
        pid=proc.pid,
        cuda_visible_devices=str(gpu.gpu_id),
        resume_stage=resume_stage,
        estimated_vram_gb=job.estimated_vram_gb,
        reserved_vram_gb=effective_reservation(job),
        launch_command=" ".join(cmd),
        working_dir=str(src_dir),
        python_executable=str(venv_python),
        started_at=utcnow(),
    )
    db.upsert_run(run)
    db.set_job_status(job.job_id, "running")
    scheduler_log("launched", job_id=job.job_id, run_id=run.run_id, payload={
        "gpu_id": gpu.gpu_id, "pid": proc.pid, "cmd": " ".join(cmd)
    })
    return run
```

---

## 10. Monitor Thread

One monitor thread runs per active run. It wakes every 10s and:

1. **Liveness check:** `os.kill(pid, 0)` — if `ProcessLookupError`, run has exited
2. **Exit classification:** read `proc.wait(timeout=0)` exit code; detect OOM from stderr
3. **TensorBoard events:** scan `job.log_dir/` for latest event file; extract scalars:
   - `stage1/loss`, `stage1/psnr`, `stage1/novel_psnr`
   - `stage2/loss`, `stage2/psnr`, `stage2/novel_psnr`
   - `stage3/loss`, `stage3/psnr`, `stage3/novel_psnr`
   - `vram/allocated`, `vram/peak`
   - `stageN/lr`
4. **Stage detection:**
   ```python
   completed_stage = 0
   if (ckpt_dir / "stage1_final.pt").exists(): completed_stage = 1
   if (ckpt_dir / "stage2_final.pt").exists(): completed_stage = 2
   if (ckpt_dir / "stage3_final.pt").exists(): completed_stage = 3

   # Active stage: one above completed, but only if process is alive and TB events are recent
   if process_alive and tb_event_age_seconds < 60:
       current_stage = completed_stage + 1
   else:
       current_stage = completed_stage  # stalled or completed
   ```
5. **Current step:** latest `stage{N}_step{K}.pt` filename; cross-check with TB global step
6. **Latest render:** `max(renders/stageN/*.jpg, key=mtime)` → `run_state.last_render_path`
7. **ETA:** uses current-stage rate for better accuracy:
   ```python
   stage_steps_done = current_step - steps_completed_in_prior_stages
   stage_steps_total = stage_steps[f"stage{current_stage}"]
   stage_elapsed = now - stage_start_time   # approximated from first checkpoint in stage
   rate = stage_steps_done / stage_elapsed if stage_elapsed > 0 else None
   stage_eta = (stage_steps_total - stage_steps_done) / rate if rate else None
   # Add remaining stages' total steps / (same rate) for full ETA
   ```
8. **Upsert** `run_state` with all updated fields

---

## 11. GPU Monitoring Backends

Auto-detected in order:

| Backend | Detection | VRAM accuracy | Per-process? |
|---------|-----------|---------------|-------------|
| `pynvml` | `import pynvml; pynvml.nvmlInit()` | Real-time, precise | Yes |
| `nvidia_smi` | `subprocess.run(["nvidia-smi", ...])` | ~100ms latency | No (total only) |
| `estimate_only` | fallback | None | No |

- `estimate_only`: scheduler uses reservation accounting only; logs a startup warning
- Backend is detected once at startup; stored in `expmanager.yaml` cache field
- `GpuSnapshot.free_vram_gb = None` when `estimate_only` — fit decision uses scheduler math only

---

## 12. Actions API

**File:** `expmanager/core/actions.py` — all functions are synchronous, write to SQLite, log to `operator_log`.

| Function | Description | Preconditions |
|----------|-------------|---------------|
| `discover()` | Rescan `experiments/`; insert new jobs; update `last_seen_at` | Any time |
| `enqueue(job_id, priority=None)` | Set job status to `queued` | `discovered/pending/failed/cancelled` |
| `enqueue_all(status_filter=['discovered'])` | Enqueue all matching jobs | — |
| `cancel(run_id)` | Send SIGTERM to PID; set status `cancelled` | `queued/running` |
| `retry(job_id, resume_stage=None)` | Re-enqueue with new run attempt | `failed/cancelled/oom` |
| `set_priority(job_id, priority)` | Update priority (1–100) | Any status |
| `set_blocked(job_id, reason)` | Set status `blocked` | Any queued/discovered |
| `unblock(job_id)` | Return to `queued` | `blocked` |
| `pause_scheduler()` | Set scheduler pause flag in DB | — |
| `resume_scheduler()` | Clear scheduler pause flag | — |

All actions:
- Validate preconditions; return `(ok: bool, error: str | None)`
- Append to `operator_log`
- Are idempotent where possible

---

## 13. Streamlit Dashboard

**Entry point:** `streamlit run expmanager/dashboard/app.py`

**Pages:**

| Page | Key content |
|------|-------------|
| Queue | Jobs table: name, status, priority, resolution, K, VRAM est/obs, action buttons |
| Active Runs | Running jobs: stage indicator, step/total progress bar, live loss/PSNR, VRAM bar, ETA, latest render thumbnail |
| Run Detail | Loss curves (from TB events), stage renders gallery, checkpoints list, log tail (last 50 lines), metrics summary |
| GPUs | Per-device card: VRAM bar (measured vs reserved), running job list, monitoring backend label |
| History | Completed/failed/OOM runs table with final metrics |
| Audit Log | operator_log table, filterable by action/job |

**Auto-refresh:** use `streamlit-autorefresh` component (pip installable, no JS needed) or `st.empty()` + loop pattern. 5s interval for Active/Queue pages; manual refresh for detail pages.

**Control actions:** All buttons call `actions.py` functions. Destructive actions (cancel, retry after OOM) show `st.warning` confirmation before executing. Buttons are disabled when action is invalid (e.g., Enqueue disabled if job is already running).

**Dashboard is read-only over experiment outputs.** It may only write through `actions.py`.

---

## 14. Startup & Recovery

On scheduler start:
1. Load `expmanager.yaml`
2. Open SQLite (WAL mode); run migrations
3. `discover()` — scan `experiments/`, insert new jobs, update `last_seen_at`
4. Reconcile runs: for each `status=running` run in DB:
   - Check if PID exists and is a Python process
   - If dead: classify and mark failed/completed based on checkpoint state
   - If alive: reattach monitor thread
5. Log startup event to `scheduler_log`
6. Begin main tick loop

---

## 15. Usability

**Start scheduler:**
```bash
cd D:/DecodeGaussians
python -m expmanager scheduler run
```

**Start dashboard (separate terminal/tmux pane):**
```bash
cd D:/DecodeGaussians
streamlit run expmanager/dashboard/app.py
```

**CLI convenience commands (v1):**
```bash
python -m expmanager discover           # scan and print discovered jobs
python -m expmanager enqueue <name>     # enqueue one job by experiment name
python -m expmanager enqueue-all        # enqueue all discovered jobs
python -m expmanager status             # print current job/run table
python -m expmanager cancel <run_id>    # cancel a run
```

---

## 16. Assumptions & Constraints

| Assumption | Impact if wrong |
|------------|----------------|
| All experiments use `SpaTrackerV2/.venv/` | Launcher will fail; add per-job python_executable override |
| `src/train.py --config ../config.yaml` launch pattern is universal | Discovery fails for non-standard experiments; add manifest override |
| `config.yaml` is the canonical experiment marker | Non-config dirs silently ignored; acceptable |
| `CUDA_VISIBLE_DEVICES` correctly remaps devices | On some drivers may not work as expected; fallback to `--device` arg |
| TensorBoard scalars use `stage1/loss` naming | TB reader returns None for variants with different scalar names; dashboard shows "N/A" gracefully |

---

## 17. Future Extensions (not v1)

- REST API wrapper around `actions.py` (FastAPI, thin layer)
- Rich TUI using `textual` or `rich`
- Daemon/systemd wrapper
- Multi-machine scheduling with shared NFS SQLite or Postgres backend
- `artifacts` table for repeated filesystem scan elimination
- Auto-experiment-generation from parameter sweep configs
- Slack/webhook notifications on run completion or failure
