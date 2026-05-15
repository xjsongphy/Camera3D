# Camera3D

`Camera3D` 是一个基于 `uv` 管理的课程实验仓库，当前已实现 Lab1 的四个任务：

- `task1`：静态场景 SfM 与轨迹可视化
- `task2`：子序列重建、Sim(3) 对齐与 ATE 分析
- `task3`：动态场景 SfM 与掩膜改进实验
- `task4`：无 GT 位姿质量评估

核心实现位于 `src/lab1/`，实验文档与报告位于 `docs/lab1/`，运行结果输出到 `outputs/lab1/`。

## 快速开始

```bash
uv sync
uv run lab1 --help
```

常用命令：

```bash
# task1: 静态场景 SfM
uv run lab1 task1 --videos S1-2 --fps 30 --stage all

# task2: 子序列分析
uv run lab1 task2 --source-fps 30 --stage all

# task3: 动态场景掩膜 + 重建
uv sync --extra task3-yolo
uv run lab1 task3-mask --source motion --videos S2-1 S2-2 --fps 30
uv run lab1 task3 --videos S2-1 S2-2 --fps 30 --methods raw
uv run lab1 task3 --videos S2-1 S2-2 --fps 30 --methods mask --mask-source motion

# task4: 位姿质量评估
uv run lab1 task4
uv run lab1 task4 plot
```

别名：

| Full command | Alias |
|---|---|
| `task1` | `q1` |
| `task2` | `q2` |
| `task3` | `q3` |
| `task4` | `q4` |

## 环境依赖

Python 依赖由 `uv` 管理：

```bash
uv sync
```

外部工具需要已在 PATH 中可用：

| Tool | Check | Purpose |
|---|---|---|
| `colmap` | `colmap -h` | 稀疏重建 / SfM |
| `ffmpeg` | `ffmpeg -version` | 视频抽帧 |

`task3` 的 YOLO 掩膜依赖可选安装：

```bash
uv sync --extra task3-yolo
```

## 项目结构

```text
Camera3D/
├─ src/lab1/                  # Lab1 CLI 与任务实现
├─ docs/lab1/                 # 题目、报告、报告配图
├─ docs/lab1/assets/videos/   # 实验视频
├─ docs/lab1/assets/annotations/ # task4 标注数据
├─ scripts/                   # 批处理脚本与辅助脚本
└─ outputs/lab1/              # 运行输出
```

## 命令说明

### Task1

```bash
# 完整流程
uv run lab1 task1 --videos S1-1 S1-2 S1-3 --fps 30 --stage all

# 分阶段运行
uv run lab1 task1 --videos S1-2 --fps 30 --stage extract
uv run lab1 task1 --videos S1-2 --fps 30 --stage sfm

# 基于已有结果生成附加可视化
uv run lab1 task1 plot --videos S1-2 --fps 30
uv run lab1 task1 cloud --videos S1-2 --fps 30

# 对多个 fps 结果做 Sim(3) 对齐叠加
uv run lab1 task1 merge --videos S1-2

# 合并模式下指定要合并的 fps 列表（默认 4/8/16/30）
uv run lab1 task1 merge --videos S1-2 --fps 4 8 16
```

输出目录：

- `outputs/lab1/task1/<video>_fps<fps>/`
- `outputs/lab1/task1/merged/<video>/`

主要文件：

- `trajectory.png`
- `trajectory_with_directions.png`
- `sparse_points.png`
- `frame_map.csv`
- `timing.csv`

### Task2

```bash
# 完整流程
uv run lab1 task2 --source-fps 30 --stage all

# 分阶段运行
uv run lab1 task2 --source-fps 30 --stage prepare
uv run lab1 task2 --source-fps 30 --stage sfm
uv run lab1 task2 --source-fps 30 --stage analyze

# 自定义子序列，格式 START:END:NAME（1-based, inclusive）
uv run lab1 task2 --source-fps 30 --subseq 211:930:return_mid
```

输出目录：

- `outputs/lab1/task2/S1-2_fps<fps>/`
- 子目录形如 `seq01_return_mid_000211-000930/`

主要文件：

- `summary.csv`
- `trajectory_overlay.png`
- `metrics.txt`
- `timing.csv`

### Task3

```bash
# 先生成掩膜
uv run lab1 task3-mask --source default --videos S2-1 S2-2 --fps 30
uv run lab1 task3-mask --source motion --videos S2-1 S2-2 --fps 30
uv run lab1 task3-mask --source yolo --videos S2-1 S2-2 --fps 30

# 原始重建
uv run lab1 task3 --videos S2-1 S2-2 --fps 30 --methods raw

# 使用不同掩膜重建
uv run lab1 task3 --videos S2-1 S2-2 --fps 30 --methods mask --mask-source default
uv run lab1 task3 --videos S2-1 S2-2 --fps 30 --methods mask --mask-source motion
uv run lab1 task3 --videos S2-1 S2-2 --fps 30 --methods mask --mask-source yolo
```

输出目录：

- `outputs/lab1/task3/masks/<source>/<video>_fps<fps>/`
- `outputs/lab1/task3/<video>_fps<fps>/raw/`
- `outputs/lab1/task3/<video>_fps<fps>/mask_default/`
- `outputs/lab1/task3/<video>_fps<fps>/mask_motion/`
- `outputs/lab1/task3/<video>_fps<fps>/mask_yolo/`

主要文件：

- `method_summary.csv`
- `trajectory_overlay.png`
- `trajectory_overlay_summary.csv`
- 每种方法目录下的 `analysis.txt`、`analysis.csv`、`trajectory_raw.png`、`trajectory_with_directions.png`、`sparse_points.png`

说明：

- `task3-mask` 只负责生成掩膜与叠加预览。
- `task3` 只消费现有掩膜；若掩膜缺失，会直接报错并给出建议命令。

### Task4

```bash
# 跑全部 10 个标注 case（含质量评估 + 轨迹绘制）
uv run lab1 task4

# 只跑部分 case
uv run lab1 task4 --cases 01 02 06

# 仅重新绘制轨迹图（无需重新计算指标）
uv run lab1 task4 plot

# 调整方向箭头数量
uv run lab1 task4 plot --direction-arrows 20

# 指标阈值参数
uv run lab1 task4 --compose-threshold-deg 1.0 --zigzag-residual-threshold 2.0 --accel-jump-ratio 4.0

# 视频三角化几何指标：fps16 抽帧、10 个 case 并行
uv run python scripts/task4_video_geometry_fps16.py --workers 10 --target-fps 16 --max-pairs 160
```

输出目录：

- `outputs/lab1/task4/`

主要文件：

- `case_metrics.csv`
- `summary.txt`
- `quality_scores.png`
- `timing.csv`
- `trajectories/trajectories.png` — 10 条轨迹总览
- `trajectories/trajectory_*.png` — 逐 case 轨迹图

当前实现的质量指标：

- `smooth_jump_ratio`
- `traj_smoothness`
- `zigzag_score`
- `epi_dist_px`
- `reproj_err_px`
- `reproj_median_px`
- `compose_rot_err_deg`

视频三角化几何指标输出到 `outputs/lab1/task4_geometry_fps16/`，报告图输出到 `docs/lab1/report_assets/task4_geometry_fps16/`。

## 批处理脚本

Windows:

```powershell
./scripts/task1_fps_sweep_full.ps1
./scripts/task2_full_pipeline.ps1
./scripts/task3_full_pipeline.ps1
./scripts/run_lab1_pipeline.ps1
```

Linux / macOS:

```bash
bash ./scripts/task1_fps_sweep_full.sh
bash ./scripts/task2_full_pipeline.sh
bash ./scripts/task3_full_pipeline.sh
bash ./scripts/run_lab1_pipeline.sh
```

说明：

- `task1_fps_sweep_full` 会批量跑 `4 / 8 / 16 / 30 fps`，并生成 benchmark CSV。
- `task2_full_pipeline` 会先确保 `S1-2` 全量结果存在，再运行默认三段子序列分析。
- `task3_full_pipeline` 会依次生成 `default / motion / yolo` 掩膜，并运行 `raw + mask_*` 重建。
- `run_lab1_pipeline` 用于串联执行 `task1~task3` 的完整实验流程。

报告配图位于 `docs/lab1/report_assets/`，例如：

- `docs/lab1/report_assets/task1/task1_fps_sweep.png`
- `docs/lab1/report_assets/task2/seq01_global_fps_grid.png`
- `docs/lab1/report_assets/task3/S2-1_sparse_raw_vs_mask_motion.png`
- `docs/lab1/report_assets/task4/task4_quality_score.png`

## 日志与输出约定

日志默认写入：

- `outputs/lab1/<task>/logs/<task>_YYYYMMDD_HHMMSS.log`

所有任务的阶段耗时都会写入各自输出目录下的 `timing.csv`。
