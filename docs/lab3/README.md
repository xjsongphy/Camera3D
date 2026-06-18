# Lab 3 README

Lab3 是面向自采图片/视频的多表示三维场景重建与评测框架。一条命令走完：抽帧/划分 → COLMAP 位姿 → SfM 点云 / 3DGS / Nerfacto 三种表示 → 渲染 held-out → 统一指标（PSNR/SSIM/LPIPS）+ 效率（训练时间/迭代数/GPU 显存峰值/FPS/模型大小）+ 几何（可视化 + Chamfer/F-score）→ 出 `metrics.csv`、`geometry_metrics.csv`、`qualitative/`、`geometry/`、`timings.json`、`logs/`。对应作业 `docs/lab3/lab3.md`。

## 依赖与工具

```bash
# CPU 环境
uv sync --group lab3 --extra cpu

# CUDA 12.4 环境（推荐，3DGS/NeRF 训练需要 GPU）
uv sync --group lab3 --extra cu124
```

`lab3` 组：Matplotlib、Pillow、NumPy、lpips、nerfstudio、tiny-cuda-nn，以及 3DGS 依赖的本地扩展 `diff-gaussian-rasterization` / `simple-knn` / `fused-ssim`（torch 经 `--extra cpu/cu124` 提供）。Windows + Python 3.12 下通过 `tool.uv.override-dependencies` 覆盖到 `pymeshlab==2023.12.post1`，并在 `tool.uv.prerelease = "allow"` 下接受 `nerfstudio 1.1.5` 传递依赖的 `newrawpy>=1.0.0b0` 预发布版本。`tiny-cuda-nn` 通过 Git 源码编译安装，要求本机 CUDA/MSVC 工具链可用；3DGS 三个扩展通过仓库内 `gaussian-splatting/submodules/` 本地路径构建安装，因此同步前需保证该目录存在。lpips 缺失时评测自动降级为只算 PSNR/SSIM。

系统工具（需在 PATH 中可用）：

- `colmap`（SfM、共享位姿、dense；CUDA 版用于 MVS）
- `ffmpeg`（视频抽帧）

外部代码库（完整训练/评测需要）：

- `nerfstudio`：已随 `uv sync --group lab3 --extra cpu|cu124` 安装（提供 `ns-process-data`/`ns-train`/`ns-eval`/`ns-render`）
- GraphDeco `gaussian-splatting`：默认读取当前工作目录下的相对路径 `./gaussian-splatting`；若 clone 到别处，用 `--dgs-repo` 显式指向（提供 `convert.py`/`train.py`/`render.py`/`metrics.py`）

> 不装这些工具也能用 `--dry-run` 验证整条命令链（见下）。

## 常用命令

### 命令链自检（dry-run，无需重工具）

```bash
uv run lab3 --config configs/lab3/default.json --methods sfm --dry-run
```

### 完整三方法（需先装好 lab3 依赖，并在当前目录准备 `./gaussian-splatting` 或手动传 `--dgs-repo`）

```bash
# Linux/macOS
uv run lab3 --input-dir path/to/captured_scene --scene-name my_scene \
  --methods sfm 3dgs nerf --dgs-repo /path/to/gaussian-splatting --fps 2

# Windows PowerShell
uv run lab3 `
  --input-dir path\to\captured_scene `
  --scene-name my_scene `
  --methods sfm 3dgs nerf `
  --dgs-repo D:\path\to\gaussian-splatting `
  --fps 2
```

### 仅重跑评测（跳过训练）

```bash
uv run lab3 --run-dir outputs/lab3/<timestamp>_my_scene
```

### 常用开关

| 开关 | 作用 |
|------|------|
| `--no-share-poses` | 关闭位姿共享（每方法各自跑 COLMAP） |
| `--no-evaluate` / `--no-geometry` / `--no-qualitative` | 关闭对应后处理阶段 |
| `--no-lpips` | 跳过 LPIPS，只算 PSNR/SSIM |
| `--eval-size H W` | 统一缩放到该分辨率再算指标（公平） |
| `--test-ratio 0.1` | held-out 比例（写 `prepared/test.txt`） |
| `--dgs-iterations 7000` / `--nerf-iterations 30000` | 训练迭代数 |
| `--timestamp <tag>` | 固定输出目录后缀，便于复现 |

### 批处理脚本

Windows：

```powershell
./scripts/lab3/run_lab3_sample.ps1
```

## 交互式可视化（作业 §8 加分）

除固定 PNG 外，可对重建结果做交互式查看（orbit / 缩放 / 选中）：

```bash
# Open3D 几何窗口（SfM 点云 / 3DGS Gaussian / mesh）+ nerfstudio web viewer（NeRF）
uv run lab3 --view-run outputs/lab3/<ts>_<scene>

# 只看几何，不开 nerfstudio viewer
uv run lab3 --view-run outputs/lab3/<ts>_<scene> --methods sfm 3dgs --no-nerfstudio-viewer
```

- **Open3D**（`pip install open3d`，需显示器）：SfM 点云、COLMAP dense、Poisson mesh、3DGS Gaussian `.ply` 都能交互查看。
  - 注意：Open3D 忽略 3DGS 的 SH 颜色通道，Gaussian 只显示为点云（仍可观察分布、浮点、穿透）。要看真正的实时 splatting 用下面的 SIBR。
- **nerfstudio web viewer**：`ns-viewer --load-config results/nerf/train/.../config.yml`，浏览器交互查看 NeRF。
- **3DGS 原生 SIBR viewer**（真正的实时 splatting 渲染）：在 `gaussian-splatting/SIBR_viewers` 编译后运行
  `./install/bin/SIBR_gaussianViewer_app -m outputs/lab3/<ts>_<scene>/results/3dgs`。

## 输出目录

`outputs/lab3/<timestamp>_<scene>/`：

- `configs/`：`run_config.json`、`prepared_dataset.json`
- `prepared/`：`images/`、`sparse/0/`（共享 COLMAP）、`manifest.csv`、`train.txt`、`test.txt`
- `results/{sfm,3dgs,nerf}/`：各方法训练/渲染产物
- `qualitative/comparison_*.png`：GT vs 方法渲染 vs 误差图（≥3 视角）
- `geometry/{sfm,3dgs,nerf}/`：点云/mesh/Gaussian `.ply` 汇总
- `geometry_metrics.csv`：各方法点云 vs COLMAP dense proxy 的 Chamfer / F-score
- `logs/<method>_<step>.log`：每步外部命令日志
- `metrics.csv`：作业必交，列 `method, psnr, ssim, lpips, metric_source, held_out, train_time_sec, iterations, gpu_mem_peak_gb, render_fps, model_size_mb, gpu, notes`
- `timings.json`：各阶段耗时

## 输出与对应指标

框架最终产出下面这些文件，每个文件直接对应作业 `docs/lab3/lab3.md` 的一条评价要求：

| 产物 | 内容 | 对应作业指标 |
|------|------|-------------|
| `metrics.csv` | 每方法一行：PSNR / SSIM / LPIPS、训练时间、迭代数、GPU 显存峰值、渲染 FPS、模型大小、GPU 型号 | §5.1 PSNR/SSIM/LPIPS；§5.2 训练时间/迭代数/显存峰值/FPS/模型大小/GPU；§6 结果表 |
| `geometry_metrics.csv` | 各方法点云 vs COLMAP dense proxy 的 Chamfer（`chamfer_to_proxy`/`from_proxy`/`sym`）与 F-score（bbox 对角线 0.5% / 1%） | §5.3 几何定量分析（proxy，需诚实披露偏差） |
| `qualitative/comparison_*.png` | GT / 各方法渲染 / 误差图，≥3 个 held-out 视角 | §5.1 定性对比 + 误差图 |
| `geometry/{sfm,3dgs,nerf}/` | COLMAP dense `.ply`、Poisson mesh、3DGS Gaussian `.ply`、nerfstudio 导出 | §5.3 几何可视化（孔洞/浮点/边界锐利度） |
| `timings.json` | 各阶段耗时：feature/match/mapper/dense、convert、process_data、train、render、eval | §5.2 预处理 / 训练 / 渲染时间分解 |
| `results/3dgs/test/.../results.json`、`results/_eval/nerf_eval.json` | 3DGS 原版 `metrics.py` 与 nerfstudio `ns-eval` 的官方指标 | 交叉校验自实现指标的可信度 |
| `configs/run_config.json`、`prepared_dataset.json` | 完整运行配置与训练/测试划分清单 | §8 可复现性 |
| `logs/<method>_<step>.log` | 每条外部命令的完整 stdout/stderr | §8 命令可追踪 |

**测量方式与保真度**

- **PSNR/SSIM**：纯 NumPy 自实现（`lab3.metrics`），SSIM 用 11×11 高斯窗；所有方法用同一套实现、同一 `eval_size` 分辨率、同一 RGB [0,1] 色彩空间。**LPIPS** 懒加载 `lpips` 包，装不上或权重下载失败时返回空并只报 PSNR/SSIM（作业允许）。
- **GPU 显存峰值** `gpu_mem_peak_gb`：训练/MVS 命令运行期间后台每 0.5 s 轮询一次 `nvidia-smi memory.used`，取最大值（GiB）。这是 GPU **全局已用**显存，含同卡其他进程，属 best-effort 近似（作业 §5.2 允许“近似观察”）；无 NVIDIA 驱动 / 无 `nvidia-smi`（如纯 CPU）时留空并在报告说明。
- **渲染 FPS** `render_fps`：3DGS 的 `render.py` 计时**含磁盘 I/O**（写 PNG），数值偏保守；nerfstudio 的 `ns-eval` 计时含渲染故帧数未知、该格常为空，以 `ns-eval` 原生为准。
- **几何指标**：以 COLMAP dense 点云为 **proxy 而非真值**，比较前双方下采样到 4096 点（`downsample_cap`）；F-score 阈值取 proxy bbox 对角线的 0.5% / 1%，尺度无关。`.ply` 加载需 Open3D，缺失则跳过并写说明行。
- **公平性**：三方法共享同一套 COLMAP 位姿（`share_poses=true`）；但 held-out 集合不完全相同——3DGS 用 `train.py --eval` 每 8 张留 1，nerfstudio 用其原生 eval split。指标实现与分辨率统一，例外（划分不同）已在 `metrics.csv` 的 `metric_source` / `held_out` / `notes` 列标注，报告需讨论。SfM 为显式点云，RGB 新视角 PSNR 标 `N/A`。
