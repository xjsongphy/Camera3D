# Camera3D

Camera3D labs managed by `uv`.

## Quick Start

```bash
# install dependencies
uv sync

# run task pipelines
uv run lab1 task1 --videos S1-2 --fps 30 --stage all
uv run lab1 task2 --source-fps 30 --stage all
uv run lab1 task3 --videos S2-1 S2-2 --fps 5 --methods raw
```

## Project Layout

```text
Camera3D/
├─ src/lab1/                  # lab1 implementation
├─ docs/lab1/assets/videos/   # input videos (S1-*.mp4, S2-*.mp4)
├─ scripts/                   # helper scripts
└─ outputs/lab1/              # generated outputs
```

## Output Conventions

| Task | Output Path |
|---|---|
| task1 | `outputs/lab1/task1/<video>_<fps>/` |
| task1 merge | `outputs/lab1/task1/merged/<video>/` |
| task2 | `outputs/lab1/task2/S1-2_<fps>/` |
| task3-mask | `outputs/lab1/task3/masks/<source>/<video>_<fps>/` |
| task3 | `outputs/lab1/task3/<video>_<fps>/<method>/` |

## Environment

```bash
uv sync
uv run lab1 --help
```

External tools required in PATH:

| Tool | Check | Purpose |
|---|---|---|
| `colmap` | `colmap -h` | SfM reconstruction |
| `ffmpeg` | `ffmpeg -version` | frame extraction |

## CLI Overview

Aliases:

| Full command | Alias |
|---|---|
| `task1` | `q1` |
| `task2` | `q2` |
| `task3` | `q3` |
| `task4` | `q4` |

## Task1

```bash
# full run
uv run lab1 task1 --videos S1-1 S1-2 S1-3 --fps 30 --stage all

# stages
uv run lab1 task1 --videos S1-2 --fps 30 --stage extract
uv run lab1 task1 --videos S1-2 --fps 30 --stage sfm

# merge trajectories across fps (Sim3)
uv run lab1 task1 merge --videos S1-2

# redraw trajectory plots from existing outputs
uv run lab1 task1 plot --videos S1-2 --fps 30

# generate sparse point-cloud plot from existing sparse output only
uv run lab1 task1 cloud --videos S1-2 --fps 30
```

`task1 cloud` only reads existing `sparse/0/{images,points3D}.txt` and writes:

- `sparse_points.png`

## Task2

```bash
# full run
uv run lab1 task2 --source-fps 30 --stage all

# stages
uv run lab1 task2 --source-fps 30 --stage prepare
uv run lab1 task2 --source-fps 30 --stage sfm
uv run lab1 task2 --source-fps 30 --stage analyze
```

Main outputs:

- `summary.csv`
- per-sequence `metrics.txt`
- per-sequence `trajectory_overlay.png`

## Task3

```bash
# 1) default camera ROI mask
uv run lab1 task3-mask --source default --videos S2-1 S2-2 --fps 5

# 2) motion mask
uv run lab1 task3-mask --source motion --videos S2-1 S2-2 --fps 5

# 3) YOLO mask
uv sync --extra task3-yolo
uv run lab1 task3-mask --source yolo --videos S2-1 S2-2 --fps 5

# 4) raw reconstruction
uv run lab1 task3 --videos S2-1 S2-2 --fps 5 --methods raw

# 5) masked reconstruction (default/motion/yolo)
uv run lab1 task3 --videos S2-1 S2-2 --fps 5 --methods mask --mask-source default
uv run lab1 task3 --videos S2-1 S2-2 --fps 5 --methods mask --mask-source motion
uv run lab1 task3 --videos S2-1 S2-2 --fps 5 --methods mask --mask-source yolo
```

Notes:

- `task3-mask generates masks and an overlay preview video (20 sampled frames) under each mask directory.
- `task3` consumes existing masks only; missing masks will raise an error with a suggested command.

Task3 method outputs include:

- `trajectory_raw.png`
- `trajectory_with_directions.png`
- `sparse_points.png`
- `analysis.txt`
- `analysis.csv`
- `method_summary.csv`

## Scripts

### Task1 full sweep

```powershell
# Windows
./scripts/task1_fps_sweep_full.ps1
./scripts/task1_fps_sweep_full.ps1 -Videos S1-2

# Linux/macOS
bash ./scripts/task1_fps_sweep_full.sh
bash ./scripts/task1_fps_sweep_full.sh S1-2
```

Behavior:

- runs `task1 --stage all --force`
- then runs `task1 cloud --force`
- writes benchmark summary CSV under `outputs/lab1/task1/benchmarks/`

### Task2 full pipeline

```powershell
# Windows
./scripts/task2_full_pipeline.ps1
./scripts/task2_full_pipeline.ps1 -Force

# Linux/macOS
bash ./scripts/task2_full_pipeline.sh
bash ./scripts/task2_full_pipeline.sh --force
```

### Task3 full pipeline

```powershell
# Windows
./scripts/task3_full_pipeline.ps1
./scripts/task3_full_pipeline.ps1 -Force
./scripts/task3_full_pipeline.ps1 -SkipYolo

# Linux/macOS
bash ./scripts/task3_full_pipeline.sh
bash ./scripts/task3_full_pipeline.sh --force
bash ./scripts/task3_full_pipeline.sh --skip-yolo
```

Behavior:

- generates masks in order: `default`, `motion`, `yolo` (unless skip yolo)
- runs reconstruction: `raw`, then `mask + default/motion/yolo`

## Logs and Timing

| Type | Location |
|---|---|
| run logs | `outputs/lab1/<task>/logs/<task>_YYYYMMDD_HHMMSS.log` |
| stage timings | task output `timing.csv` |
