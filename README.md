# Camera3D

当前仓库使用 `uv` 管理 Python 环境与命令行入口。

## 项目结构

- `src/lab1/`: Lab1 代码与命令入口（已实现）
- `src/lab2/`: Lab2 代码目录（仅占位，后续实现）
- `lab1/`: 课程给定的 Lab1 文档与素材
- `lab2/`: 课程给定的 Lab2 文档与素材
- `outputs/lab1/`: Lab1 运行产物（自动创建）

## 安装与运行

```bash
uv sync
uv run lab1 --help
```

运行 Lab1 各题：

```bash
uv run lab1 task1
uv run lab1 task2
uv run lab1 task3
uv run lab1 task4
```

也支持简写参数：

```bash
uv run lab1 q1
uv run lab1 q2
uv run lab1 q3
uv run lab1 q4
```

## 视频文件放置说明

请将 Lab1 视频放在以下路径：

- `lab1/assets/videos/S1-1.mp4`
- `lab1/assets/videos/S1-2.mp4`
- `lab1/assets/videos/S1-3.mp4`
- `lab1/assets/videos/S2-1.mp4`
- `lab1/assets/videos/S2-2.mp4`

若你的视频放在其他位置，请先移动或建立软链接到上述路径，再执行 `uv run lab1 ...`。

## 当前范围

本次初始化只完成 Lab1 命令入口与仓库结构整理，不包含 Lab2 实现。
