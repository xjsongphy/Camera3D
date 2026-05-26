# Lab 1: 基于视频的3D重建

基于单目视频的静态与动态场景3D重建与位姿评估实验。

## 实验任务

- `task1`：静态场景 SfM 与轨迹可视化
- `task2`：子序列重建、Sim(3) 对齐与 ATE 分析
- `task3`：动态场景 SfM 与掩膜改进实验
- `task4`：无 GT 位姿质量评估

## 快速开始

```bash
# 同步依赖
uv sync --group lab1

# 运行task1示例
uv run lab1 task1 --videos S1-2 --fps 30 --stage all
```

## 常用命令

```bash
# task1: 静态场景 SfM
uv run lab1 task1 --videos S1-2 --fps 30 --stage all

# task2: 子序列分析
uv run lab1 task2 --source-fps 30 --stage all

# task3: 动态场景掩膜 + 重建
uv sync --group lab1 --group lab1-yolo
uv run lab1 task3-mask --source motion --videos S2-1 S2-2 --fps 30
uv run lab1 task3 --videos S2-1 S2-2 --fps 30 --methods mask --mask-source motion

# task4: 位姿质量评估
uv run lab1 task4
uv run lab1 task4 plot
```

## 项目结构

```
docs/lab1/
├─ assets/videos/           # 实验视频
├─ assets/annotations/      # task4 标注数据
├─ report_assets/           # 报告配图
├─ lab1.md                  # 作业要求
├─ report.md                # 实验报告
└─ README.md                # 本文件
```

## 环境依赖

### Python依赖
```bash
uv sync --group lab1
```

### 外部工具
| 工具 | 检查 | 用途 |
|------|------|------|
| `colmap` | `colmap -h` | 稀疏重建/SfM |
| `ffmpeg` | `ffmpeg -version` | 视频抽帧 |

### 可选依赖（task3）
```bash
uv sync --group lab1 --group lab1-yolo
```

## 作业文档

- [作业要求](lab1.md)
- [实验报告](report.md)
- [讲义](lab1-讲义.md)
- [原始PDF](三维计算成像基础 - 课程作业.pdf)

详细命令说明请参考[主仓库README](../../README.md)。
