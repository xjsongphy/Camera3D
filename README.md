# Camera3D

使用 `uv` 管理环境与命令入口。

## 项目结构

- `src/lab1/`: Lab1 实现代码（`task1`、`task2`、`task3`、CLI）
- `lab1/`: Lab1 题目文档与报告
- `scripts/`: 批处理与一键运行脚本
- `outputs/lab1/`: 运行产物（已在 `.gitignore` 中忽略）

## 目录存放约定

- 输入视频固定放在 `lab1/assets/videos/`，文件名按题目约定：
  - `S1-1.mp4`、`S1-2.mp4`、`S1-3.mp4`
  - `S2-1.mp4`、`S2-2.mp4`
- 报告与报告素材放在 `lab1/` 下：
  - 报告：`lab1/report.md`
  - 报告图与脚本：`lab1/report_assets/`
- 运行脚本放在 `scripts/`：
  - `task1_build_full_result.*`
  - `task2_full_pipeline.*`
- 运行输出统一写到 `outputs/lab1/`（默认不提交）：
  - `task1`：`outputs/lab1/task1/<video>_<param_tag>/`
  - `task1 merge`：`outputs/lab1/task1/merged/<video>/`
  - `task2`：`outputs/lab1/task2/S1-2_<param_tag>/`

## 环境准备

### 1. Python 依赖

```bash
uv sync
uv run lab1 --help
```

### 2. 外部工具

需要安装并可在命令行调用：

- `colmap`
- `ffmpeg`

可用以下命令检查：

```bash
colmap -h
ffmpeg -version
```

如不在 PATH，可运行时指定：

```bash
uv run lab1 task1 --colmap-bin /path/to/colmap --ffmpeg-bin /path/to/ffmpeg
```

## 常用命令

```bash
uv run lab1 task1
uv run lab1 task1 merge
uv run lab1 task2
uv run lab1 task3
```

别名：

```bash
uv run lab1 q1
uv run lab1 q2
uv run lab1 q3
uv run lab1 q4
```

## Task1 运行

### 全流程（默认）

```bash
uv run lab1 task1 --videos S1-1 S1-2 S1-3 --fps 30 --stage all
```

### 分阶段

```bash
uv run lab1 task1 --videos S1-2 --fps 30 --stage extract
uv run lab1 task1 --videos S1-2 --fps 30 --stage sfm
```

### 合并不同 FPS 轨迹（Sim(3) 对齐）

```bash
uv run lab1 task1 merge --videos S1-2
```

输出目录示例：

- `outputs/lab1/task1/S1-2_fps30/`
- `outputs/lab1/task1/merged/S1-2/`

## Task2 运行（S1-2 子序列分析）

`task2` 使用三段固定子序列：前 `1/3`、中间 `1/2`、后 `1/4`。  
流程包含方法 A 位姿截取、方法 B 子序列重建、Sim(3) 对齐和 ATE 统计。

### 直接运行 task2

```bash
uv run lab1 task2 --source-fps 30 --stage all
```

### 分阶段运行

```bash
uv run lab1 task2 --source-fps 30 --stage prepare
uv run lab1 task2 --source-fps 30 --stage sfm
uv run lab1 task2 --source-fps 30 --stage analyze
```

输出目录示例：

- `outputs/lab1/task2/S1-2_fps30/summary.csv`
- `outputs/lab1/task2/S1-2_fps30/<subseq>/metrics.txt`
- `outputs/lab1/task2/S1-2_fps30/<subseq>/trajectory_overlay.png`

## 一键脚本

### Task1 全量结果构建（S1-2）

Windows:

```powershell
./scripts/task1_build_full_result.ps1 -Video S1-2 -Fps 30
```

Linux/macOS:

```bash
bash ./scripts/task1_build_full_result.sh S1-2 30
```

### Task2 全流程（带前置检测）

脚本会先检测 `task1` 全量结果是否存在，缺失时自动调用上面的 `task1` 构建脚本；存在则复用，避免重复重建。

Windows:

```powershell
./scripts/task2_full_pipeline.ps1 -SourceFps 30
```

Linux/macOS:

```bash
bash ./scripts/task2_full_pipeline.sh 30
```

强制重跑可加 `-Force`（PowerShell）或 `--force`（bash 第 2 个参数）。

## 日志与耗时

每次 `uv run lab1 ...` 会写日志到：

- `outputs/lab1/<task>/logs/<task>_YYYYMMDD_HHMMSS.log`

阶段耗时会写到各自目录的 `timing.csv`。
