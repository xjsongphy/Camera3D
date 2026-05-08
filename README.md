# Camera3D

当前仓库使用 `uv` 管理 Python 环境与命令行入口。

## 项目结构

- `src/lab1/`: Lab1 代码（当前优先实现）
- `src/lab2/`: Lab2 代码目录（仅占位，后续实现）
- `lab1/`: 课程给定的 Lab1 文档与素材
- `lab2/`: 课程给定的 Lab2 文档与素材
- `outputs/lab1/`: Lab1 运行产物

## 安装

```bash
uv sync
uv run lab1 --help
```

## 安装 COLMAP 与 ffmpeg

`task1` 依赖外部命令行工具：`colmap` 和 `ffmpeg`。

macOS（推荐，Homebrew）：

```bash
# 如果还没有 Homebrew，先安装：
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装依赖
brew install colmap ffmpeg
```

Linux（Debian/Ubuntu）：

```bash
sudo apt update
sudo apt install -y colmap ffmpeg
```

安装后自检：

```bash
colmap -h
ffmpeg -version
```

若 `colmap` 不在 PATH，可在运行时显式指定：

```bash
uv run lab1 task1 --colmap-bin /your/path/to/colmap
```

## Lab1 运行

```bash
uv run lab1 task1 [--fps 2] [--force]
uv run lab1 task2 [--source-fps 4] [--force]
uv run lab1 task3
uv run lab1 task4
```

简写：

```bash
uv run lab1 q1
uv run lab1 q2
uv run lab1 q3
uv run lab1 q4
```

## 题目一（已实现）

按 `lab1/lab1.md` 要求，`task1` 会对以下三个静态视频执行流程：

- `lab1/assets/videos/S1-1.mp4`
- `lab1/assets/videos/S1-2.mp4`
- `lab1/assets/videos/S1-3.mp4`

流程：

1. 用 `ffmpeg` 抽帧到 `images/`
2. 用 COLMAP 运行稀疏重建（`feature_extractor + sequential_matcher + mapper`）
3. 导出 COLMAP 文本格式结果（`images.txt/cameras.txt/points3D.txt`）
4. 绘制并保存相机轨迹图 `trajectory.png`

输出目录（每个视频一份）：

```txt
outputs/lab1/task1/<video_name>_<param_tag>/
├── images/
├── sparse/
│   └── 0/
│       ├── images.txt
│       ├── cameras.txt
│       └── points3D.txt
└── trajectory.png
```

其中 `param_tag` 由运行参数生成（例如 `fps4`、`fps2p5`），用于隔离不同参数的实验结果。
同样参数重复运行时会自动复用已有结果（除非传 `--force`）。

可选参数：

- `--fps`: 抽帧帧率（默认 `2.0`）
- `--skip-sfm`: 兼容旧参数（已弃用），等价于 `--stage extract`
- `--stage`: 控制阶段，`all`（默认）/`extract`（仅抽帧）/`sfm`（仅跑SfM，复用已有抽帧）
- `--colmap-bin`: COLMAP 可执行文件路径（默认 `colmap`）
- `--ffmpeg-bin`: ffmpeg 可执行文件路径（默认 `ffmpeg`）
- `--force`: 覆盖已有输出
- `--dry-run`: 只打印命令，不执行

示例（避免重复抽帧）：

```bash
# 第一步：仅抽帧
uv run lab1 task1 --videos S1-1 --fps 4 --stage extract --force

# 第二步：只跑 SfM（复用已有 frames）
uv run lab1 task1 --videos S1-1 --fps 4 --stage sfm --force
```

说明：`task1` 参数只对 `task1/q1` 生效；`task2/3/4` 不再接受 `--fps` 等 `task1` 专属参数，避免误用。

## 视频文件放置说明

请将 Lab1 视频放在以下路径：

- `lab1/assets/videos/S1-1.mp4`
- `lab1/assets/videos/S1-2.mp4`
- `lab1/assets/videos/S1-3.mp4`
- `lab1/assets/videos/S2-1.mp4`
- `lab1/assets/videos/S2-2.mp4`

若视频在其他位置，请先移动或建立软链接到上述路径。

## 当前范围

- 已实现：Lab1 题目一、题目二
- 未实现：Lab1 题目三/四 与 Lab2

## 题目二（已实现）

`task2` 针对 `S1-2` 执行“三段子序列 × 两种位姿来源”的对比：

1. 方法 A：从 `task1` 的完整序列重建中截取子序列位姿；
2. 方法 B：仅用该子序列帧重新跑一遍 SfM；
3. 对齐与对比：用 Sim(3)（Umeyama）将方法 B 对齐到方法 A，并计算 ATE，导出叠加轨迹图。

输入依赖（先运行 task1）：

```bash
uv run lab1 task1 --videos S1-2 --fps 4 --stage all
```

运行 task2：

```bash
uv run lab1 task2 --source-fps 4 --stage all
```

分阶段运行（避免重复）：

```bash
# 准备三段子序列（复制子帧 + 方法A位姿子集）
uv run lab1 task2 --source-fps 4 --stage prepare

# 仅跑方法B的 SfM
uv run lab1 task2 --source-fps 4 --stage sfm

# 仅做对齐与ATE分析
uv run lab1 task2 --source-fps 4 --stage analyze
```

输出目录：

```txt
outputs/lab1/task2/S1-2_fps4/
├── seq01_front_1over3_000001-000105/
│   ├── method_a/sparse/0/{images.txt,cameras.txt,points3D.txt}
│   ├── method_b/{images/,database.db,sparse/0/...}
│   ├── trajectory_overlay.png
│   └── metrics.txt
├── seq02_middle_1over2_000079-000235/...
├── seq03_back_1over4_000235-000312/...
└── summary.csv
```
