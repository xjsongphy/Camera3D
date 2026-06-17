# Camera3D

基于 `uv` 管理的三维计算成像课程实验仓库，包含 Lab1（视频3D重建）、Lab2（结构光优化）和 Lab3（多表示三维场景重建）三个独立实验。

## 整体设计

- **依赖管理**：使用 `uv` 统一管理Python依赖
- **模块化**：每个lab有独立的源码、文档和输出目录
- **类型安全**：Python 3.12+ with type hints
- **测试驱动**：每个lab包含自检测试

## 实验模块

### [Lab 1: 基于视频的3D重建](docs/lab1/README.md)

基于单目视频的静态与动态场景3D重建与位姿评估。

- 静态场景 SfM 与轨迹可视化
- 子序列重建与 Sim(3) 对齐
- 动态场景掩膜改进
- 无 GT 位姿质量评估

### [Lab 2: 结构光Pattern自动优化](docs/lab2/README.md)

复现Optical SGD论文，实现合成环境下的结构光图案优化闭环。

- Mitsuba合成渲染器
- 有限差分 vs 自动微分梯度计算
- ZNCC vs ZNCC-NN Decoder对比
- 多材质鲁棒性测试

### [Lab 3: 多表示三维场景重建](docs/lab3/README.md)

面向自采图片/视频的 SfM 点云、3DGS、Nerfacto 三表示统一重建与公平评测框架。

- 图片/视频统一抽帧、manifest、train/test 划分；位姿共享（单次 COLMAP 供三方法复用）
- COLMAP SfM（点云/dense）、GraphDeco 3DGS 外部仓库包装、nerfstudio Nerfacto 集成
- 统一指标（PSNR/SSIM/LPIPS）、渲染 FPS、模型大小、GPU 信息 → `metrics.csv`
- 几何分期（点云/mesh/depth）与 GT-方法-误差定性对比
- 交互式 viewer：`lab3 --view-run` 开 Open3D（点云/高斯/mesh）+ nerfstudio web viewer（NeRF），3DGS 可用 SIBR
- 输出 `metrics.csv`/`qualitative/`/`geometry/`/`logs/`，对应作业 §6 报告骨架（`docs/lab3/report_template.md`）

## 项目结构

```
Camera3D/
├─ pyproject.toml           # uv配置与依赖分组
├─ README.md                # 本文件
├─ src/                     # 源码
│  ├─ lab1/                 # Lab1实现
│  ├─ lab2/                 # Lab2实现
│  └─ lab3/                 # Lab3实现
├─ docs/                    # 文档
│  ├─ lab1/                 # Lab1文档
│  ├─ lab2/                 # Lab2文档
│  └─ lab3/                 # Lab3文档
├─ scripts/                 # 脚本
│  ├─ lab1/                 # Lab1脚本
│  ├─ lab2/                 # Lab2脚本
│  └─ lab3/                 # Lab3脚本
├─ tests/                   # 测试
│  ├─ lab1/                 # Lab1测试
│  ├─ lab2/                 # Lab2测试
│  └─ lab3/                 # Lab3测试
├─ outputs/                 # 运行输出
│  ├─ lab1/                 # Lab1输出
│  ├─ lab2/                 # Lab2输出
│  └─ lab3/                 # Lab3输出
└─ assets/                  # 资源文件
   ├─ scenes/               # Lab2场景文件
   └─ videos/               # Lab1视频
```

## 依赖管理

项目使用 `uv` 进行依赖管理，依赖按lab分组：

```bash
# 查看所有依赖组
uv sync --help

# 同步特定lab的依赖
uv sync --group lab1       # OpenCV, Matplotlib, Pillow
uv sync --group lab2       # PyTorch, Mitsuba, Drjit
uv sync --group lab3 --extra cu124  # Matplotlib, Pillow, lpips, nerfstudio + CUDA torch
uv sync --group lab1-yolo  # Lab1的YOLO依赖
uv sync --all-groups       # 所有依赖
```

### 依赖分组

| 分组 | 内容 |
|------|------|
| `lab1` | OpenCV, Matplotlib, Pillow |
| `lab1-yolo` | Ultralytics YOLO (可选) |
| `lab2` | PyTorch, Mitsuba, Drjit, OpenCV, Matplotlib |
| `lab3` | Matplotlib, Pillow, lpips, nerfstudio（torch 经 `--extra cpu/cu124` 提供）；外部 COLMAP、FFmpeg、3DGS 仓库 |

## Python版本要求

- **Lab1**: Python >= 3.10
- **Lab2**: Python 3.12（PyTorch和Mitsuba的限制）

## 开发指南

### 添加新依赖

```bash
# 添加到lab1组
uv add --group lab1 <package>

# 添加开发依赖
uv add --dev pytest
```

### 运行测试

```bash
# Lab1测试
uv run pytest tests/lab1/

# Lab2测试（需要Mitsuba）
uv sync --group lab2
uv run pytest tests/lab2/
```

### 代码风格

项目使用类型注解和docstring：

```python
from __future__ import annotations

def example_function(param: str) -> str:
    """Example function with type hints."""
    return param.upper()
```

## License

本仓库仅用于课程教学，请勿用于商业用途。
