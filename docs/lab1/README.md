# Lab 1 README

Lab1 聚焦单目视频场景的 SfM 重建、子序列分析、动态场景掩膜实验与无 GT 质量评估。

## 依赖与工具

```bash
# 安装通用依赖
uv sync

# 若需要 YOLO 掩膜（task3）
uv sync --extra task3-yolo
```

系统工具（需在 PATH 中可用）：

- `colmap`（SfM 重建）
- `ffmpeg`（视频抽帧）

## 常用命令

### Task1: 静态场景重建

```bash
# 完整流程
uv run lab1 task1 --videos S1-1 S1-2 S1-3 --fps 30 --stage all

# 分阶段
uv run lab1 task1 --videos S1-2 --fps 30 --stage extract
uv run lab1 task1 --videos S1-2 --fps 30 --stage sfm

# 可视化与合并
uv run lab1 task1 plot --videos S1-2 --fps 30
uv run lab1 task1 cloud --videos S1-2 --fps 30
uv run lab1 task1 merge --videos S1-2
```

### Task2: 子序列重建与 Sim(3) 对齐

```bash
# 完整流程
uv run lab1 task2 --source-fps 30 --stage all

# 分阶段
uv run lab1 task2 --source-fps 30 --stage prepare
uv run lab1 task2 --source-fps 30 --stage sfm
uv run lab1 task2 --source-fps 30 --stage analyze
```

### Task3: 动态场景掩膜与重建

```bash
# 生成掩膜
uv run lab1 task3-mask --source default --videos S2-1 S2-2 --fps 30
uv run lab1 task3-mask --source motion --videos S2-1 S2-2 --fps 30
uv run lab1 task3-mask --source yolo --videos S2-1 S2-2 --fps 30

# raw 与 mask 方法重建
uv run lab1 task3 --videos S2-1 S2-2 --fps 30 --methods raw
uv run lab1 task3 --videos S2-1 S2-2 --fps 30 --methods mask --mask-source motion
```

### Task4: 无 GT 位姿质量评估

```bash
# 全量 case
uv run lab1 task4

# 指定 case
uv run lab1 task4 --cases 01 02 06

# 仅重绘轨迹
uv run lab1 task4 plot
```

## 批处理脚本

Windows:

```powershell
./scripts/task1_fps_sweep_full.ps1
./scripts/task2_full_pipeline.ps1
./scripts/task3_full_pipeline.ps1
./scripts/run_lab1_pipeline.ps1
```

Linux/macOS:

```bash
bash ./scripts/task1_fps_sweep_full.sh
bash ./scripts/task2_full_pipeline.sh
bash ./scripts/task3_full_pipeline.sh
bash ./scripts/run_lab1_pipeline.sh
```

## 输出目录

- `outputs/lab1/task1/`
- `outputs/lab1/task2/`
- `outputs/lab1/task3/`
- `outputs/lab1/task4/`
