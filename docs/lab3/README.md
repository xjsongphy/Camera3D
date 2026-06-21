# Lab 3 README

Lab 3 面向自采图片/视频的多表示三维场景重建与评测，统一支持 SfM、3DGS、NeRF、NeuS 四种方法。

## 依赖与工具

项目 `pyproject.toml` 要求 Python 3.12。

在用 `uv` 创建 `lab3` 环境前，先确保 GraphDeco 的 `gaussian-splatting` 仓库已经放在项目根目录下的 `./gaussian-splatting`，并且其 `submodules/` 已经初始化完成；`lab3` 依赖中的 `diff-gaussian-rasterization`、`simple-knn`、`fused-ssim` 都通过这个仓库内的本地路径安装，因此这些目录必须已经就绪，并且能被当前工作区中的 Git/文件路径正常定位。

CPU 环境：

```bash
uv sync --python 3.12 --group lab3 --extra cpu
```

CUDA 12.4 环境（推荐，3DGS / NeRF / NeuS 训练需要 GPU）：

```bash
uv sync --python 3.12 --group lab3 --extra cu124
```

`lab3` 组依赖包括：

- `matplotlib`
- `pillow`
- `numpy`
- `lpips`
- `nerfstudio`
- `pycolmap`
- `HLOC`
- `tiny-cuda-nn`
- `diff-gaussian-rasterization`
- `simple-knn`
- `fused-ssim`

系统工具（需在 PATH 中可用）：

- `colmap`
- `ffmpeg`

完成一次 `uv sync` 后，下面的命令默认都直接使用 `uv run lab3 ...`。

## 常用命令

### 1. 命令链自检

```bash
uv run lab3 \
  --config configs/lab3/default.json \
  --methods sfm \
  --dry-run
```

### 2. 完整运行

```bash
uv run lab3 \
  --input-dir path/to/captured_scene \
  --scene-name my_scene \
  --methods sfm 3dgs nerf neus \
  --dgs-repo ./gaussian-splatting \
  --fps 2
```

如果 `gaussian-splatting` 不在项目根目录，可以把 `--dgs-repo` 改成实际路径。

### 3. 只跑部分方法

```bash
uv run lab3 \
  --config configs/lab3/default.json \
  --methods sfm nerf
```

### 4. 仅重跑评测 / 后处理

```bash
uv run lab3 \
  --run-dir outputs/lab3/<timestamp>_<scene>
```

也可以按需关闭后处理阶段：

```bash
uv run lab3 \
  --config configs/lab3/default.json \
  --no-evaluate \
  --no-geometry \
  --no-qualitative
```

### 5. 交互查看已有结果

统一入口：

```bash
uv run lab3 \
  --view-run outputs/lab3/<timestamp>_<scene>
```

`SfM`：查看器是 Open3D。进入方式：

```bash
uv run lab3 \
  --view-run outputs/lab3/<timestamp>_<scene> \
  --methods sfm \
  --no-nerfstudio-viewer
```

`3DGS`：查看器是 GraphDeco SIBR viewer；`lab3 --view-run` 会自动尝试进入，若未编译会先提示并构建。进入方式：

```bash
uv run lab3 \
  --view-run outputs/lab3/<timestamp>_<scene> \
  --methods 3dgs \
  --no-nerfstudio-viewer
```

`NeRF`：查看器是 nerfstudio web viewer。进入方式：

```bash
uv run lab3 \
  --view-run outputs/lab3/<timestamp>_<scene> \
  --methods nerf
```

`NeuS`：查看器是 nerfstudio web viewer。进入方式：

```bash
uv run lab3 \
  --view-run outputs/lab3/<timestamp>_<scene> \
  --methods neus
```

### 6. 常见参数覆盖

SfM / HLOC：

```bash
uv run lab3 \
  --config configs/lab3/boya_close.json \
  --methods sfm \
  --sfm-feature-extractor aliked-n16 \
  --sfm-feature-matcher aliked+lightglue
```

3DGS：

```bash
uv run lab3 \
  --config configs/lab3/default.json \
  --methods 3dgs \
  --dgs-repo ./gaussian-splatting \
  --dgs-iterations 7000 \
  --dgs-save-iterations 2000 4000 6000 7000 \
  --dgs-camera-cache-size 4
```

NeRF：

```bash
uv run lab3 \
  --config configs/lab3/default.json \
  --methods nerf \
  --nerf-iterations 30000 \
  --nerf-save-iterations 2000 5000 10000 20000 30000 \
  --nerf-train-rays-per-batch 4096
```

NeuS：

```bash
uv run lab3 \
  --config configs/lab3/extra.json \
  --methods neus \
  --neus-iterations 20001 \
  --neus-save-iterations 2000 5000 10000 20000 \
  --neus-train-rays-per-batch 4096 \
  --neus-train-num-images 256 \
  --neus-train-repeat-images 100
```

### 7. 常用配置入口

- 主配置：`configs/lab3/default.json`
- Boya 场景配置：`configs/lab3/boya_close.json`、`configs/lab3/boya_far.json`
- 额外实验配置：`configs/lab3/extra.json`
