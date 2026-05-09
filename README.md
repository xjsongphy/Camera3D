# Camera3D

三维计算成像实验 - Structure from Motion 与相机位姿估计

使用 `uv` 管理环境与命令入口。

---

## 快速开始

```bash
# 环境安装
uv sync

# 运行任务
uv run lab1 task1 --videos S1-2 --fps 30 --stage all
uv run lab1 task2 --source-fps 30 --stage all
uv run lab1 task3
```

---

## 目录结构

```
Camera3D/
├── src/lab1/          # Lab1 实现代码
├── docs/
│   ├── lab1/
│   │   ├── assets/
│   │   │   └── videos/    # 输入视频（S1-*.mp4, S2-*.mp4）
│   │   ├── report.md      # 报告
│   │   └── report_assets/ # 报告素材
│   └── lab2/
├── scripts/           # 批处理脚本
└── outputs/lab1/      # 运行输出（默认不提交）
```

**输出目录约定：**

| 任务 | 输出路径 |
|------|----------|
| task1 | `outputs/lab1/task1/<video>_<fps>/` |
| task1 merge | `outputs/lab1/task1/merged/<video>/` |
| task2 | `outputs/lab1/task2/S1-2_<fps>/` |
| task3 | `outputs/lab1/task3/<video>_<fps>/<method>/` |

---

## 环境准备

### Python 依赖

```bash
uv sync
uv run lab1 --help
```

### 外部工具

需安装并配置到 PATH：

| 工具 | 检查命令 | 作用 |
|------|----------|------|
| `colmap` | `colmap -h` | SfM 重建 |
| `ffmpeg` | `ffmpeg -version` | 视频抽帧 |

**未在 PATH 时**：运行时通过参数指定

```bash
uv run lab1 task1 --colmap-bin /path/to/colmap --ffmpeg-bin /path/to/ffmpeg
```

---

## Task 命令参考

### 命令别名速查

| 完整命令 | 别名 |
|----------|------|
| `task1` | `q1` |
| `task1 merge` | `q2` |
| `task2` | `q3` |
| `task3` | `q4` |

### Task1：多 FPS 轨迹重建

```bash
# 全流程
uv run lab1 task1 --videos S1-1 S1-2 S1-3 --fps 30 --stage all

# 分阶段
uv run lab1 task1 --videos S1-2 --fps 30 --stage extract  # 抽帧
uv run lab1 task1 --videos S1-2 --fps 30 --stage sfm      # SfM

# 合并多 FPS 轨迹（Sim(3) 对齐）
uv run lab1 task1 merge --videos S1-2
```

### Task2：子序列位姿分析

默认使用三段更有区分度的子序列：
- `return_local`：局部折返短段，容易暴露独立 SfM 的失败模式
- `scan_stable`：单向扫描段，通常能稳定独立重建
- `return_long`：最长的长程折返段，是最接近“全局回返”结构的子序列

```bash
# 全流程
uv run lab1 task2 --source-fps 30 --stage all

# 分阶段
uv run lab1 task2 --source-fps 30 --stage prepare  # 准备子序列
uv run lab1 task2 --source-fps 30 --stage sfm      # 子序列重建
uv run lab1 task2 --source-fps 30 --stage analyze  # 对齐与统计
```

**输出：** `summary.csv`、`metrics.txt`、`trajectory_overlay.png`

### Task3

```bash
# 原始 SfM
uv run lab1 task3 --videos S2-1 S2-2 --fps 5 --methods raw

# 基于静态先验区域的遮罩
uv run lab1 task3 --videos S2-1 S2-2 --fps 5 --methods static_roi_mask

# 基于帧间差分的动态区域遮罩
uv run lab1 task3 --videos S2-1 S2-2 --fps 5 --methods motion_mask

# 外部语义分割掩码（需预先生成）
uv run lab1 task3 --videos S2-1 S2-2 --fps 5 --methods semantic_mask \
  --semantic-mask-root /path/to/semantic_masks
```

**Task3 输出：**
- `trajectory_raw.png`
- `trajectory_with_directions.png`
- `sparse_points.png`
- `analysis.txt`
- `analysis.csv`
- `method_summary.csv`

---

## 一键脚本

### Task1 全量构建（S1-2）

```powershell
# Windows
./scripts/task1_build_full_result.ps1 -Video S1-2 -Fps 30

# Linux/macOS
bash ./scripts/task1_build_full_result.sh S1-2 30
```

### Task2 全流程（自动检测前置依赖）

脚本会检测 task1 结果，缺失时自动构建，避免重复。

```powershell
# Windows
./scripts/task2_full_pipeline.ps1

# Linux/macOS
bash ./scripts/task2_full_pipeline.sh

# 强制重跑
./scripts/task2_full_pipeline.ps1 -Force
bash ./scripts/task2_full_pipeline.sh --force
```

### Task3 全流程

```powershell
# Windows
./scripts/task3_full_pipeline.ps1
./scripts/task3_full_pipeline.ps1 -Methods raw,static_roi_mask,motion_mask -Force

# Linux/macOS
bash ./scripts/task3_full_pipeline.sh
bash ./scripts/task3_full_pipeline.sh --methods raw static_roi_mask motion_mask --force
```

---

## 日志与耗时

| 类型 | 位置 |
|------|------|
| 运行日志 | `outputs/lab1/<task>/logs/<task>_YYYYMMDD_HHMMSS.log` |
| 阶段耗时 | 各任务目录下的 `timing.csv` |
