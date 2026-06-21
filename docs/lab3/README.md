# Lab 3 README

Lab3 是面向自采图片/视频的多表示三维场景重建与评测框架。一条命令走完：抽帧/划分 → 共享 COLMAP 位姿 → SfM 点云 / 3DGS / Nerfacto / NeuS 四种表示 → 渲染同一 held-out → 统一指标（PSNR/SSIM/LPIPS）+ 效率（训练时间/迭代数/GPU 显存峰值/FPS/模型大小）+ 几何（可视化 + Chamfer/F-score）→ 出 `metrics.csv`、`geometry_metrics.csv`、`qualitative/`、`geometry/`、`timings.json`、`logs/`。对应作业 `docs/lab3/lab3.md`。

## 依赖与工具

```bash
# CPU 环境
uv sync --group lab3 --extra cpu

# CUDA 12.4 环境（推荐，3DGS/NeRF 训练需要 GPU）
uv sync --group lab3 --extra cu124
```

`lab3` 组：Matplotlib、Pillow、NumPy、lpips、nerfstudio、pycolmap、HLOC、tiny-cuda-nn，以及 3DGS 依赖的本地扩展 `diff-gaussian-rasterization` / `simple-knn` / `fused-ssim`（torch 经 `--extra cpu/cu124` 提供）。Windows + Python 3.12 下通过 `tool.uv.override-dependencies` 覆盖到 `pymeshlab==2023.12.post1`，并在 `tool.uv.prerelease = "allow"` 下接受 `nerfstudio 1.1.5` 传递依赖的 `newrawpy>=1.0.0b0` 预发布版本。`tiny-cuda-nn` 通过 Git 源码编译安装，要求本机 CUDA/MSVC 工具链可用；3DGS 三个扩展通过仓库内 `gaussian-splatting/submodules/` 本地路径构建安装，因此同步前需保证该目录存在。lpips 缺失时评测自动降级为只算 PSNR/SSIM。

系统工具（需在 PATH 中可用）：

- `colmap`（SfM、共享位姿、dense；CUDA 版用于 MVS）
- `ffmpeg`（视频抽帧）

基于机器学习的 SfM（HLOC / pycolmap）依赖已经并入 `lab3` 组；只要执行上面的 `uv sync --group lab3 --extra cpu|cu124`，就会和其他 lab3 依赖一起安装。

外部代码库（完整训练/评测需要）：

- `nerfstudio`：已随 `uv sync --group lab3 --extra cpu|cu124` 安装（提供 `ns-process-data`/`ns-train`/`ns-eval`/`ns-render`）
- GraphDeco `gaussian-splatting`：默认读取当前工作目录下的相对路径 `./gaussian-splatting`；若 clone 到别处，用 `--dgs-repo` 显式指向（提供 `convert.py`/`train.py`/`render.py`/`metrics.py`）

## 常用命令

### 命令链自检（dry-run，无需重工具）

```bash
uv run lab3 --config configs/lab3/default.json --methods sfm --dry-run
```

### 完整四方法（需先装好 lab3 依赖，并在当前目录准备 `./gaussian-splatting` 或手动传 `--dgs-repo`）

```bash
# Linux/macOS
uv run lab3 --input-dir path/to/captured_scene --scene-name my_scene \
  --methods sfm 3dgs nerf neus --dgs-repo /path/to/gaussian-splatting --fps 2

# Windows PowerShell
uv run lab3 `
  --input-dir path\to\captured_scene `
  --scene-name my_scene `
  --methods sfm 3dgs nerf neus `
  --dgs-repo D:\path\to\gaussian-splatting `
  --fps 2
```

### 仅重跑评测（跳过训练）

```bash
uv run lab3 --run-dir outputs/lab3/<timestamp>_my_scene
```

### NeuS 数据与方法

`neus` 是独立重建方法，默认调用 nerfstudio 1.1.5 的 `neus-facto`；如需原版可在
`reconstruction.neus.method` 中改为 `neus`。流水线复用 SfM 的 COLMAP TXT 模型，自动生成
SDFStudio 所需的 `meta_data.json` 并归一化场景，因此应同时启用 `sfm` 和 `share_poses`。
默认不使用单目深度/法线先验（`has_mono_prior=false`），训练完成后自动从 SDF 零水平集导出 mesh。

### 常用开关

| 开关 | 作用 |
|------|------|
| `--no-share-poses` | 显式关闭公平位姿模式（3DGS/NeRF 各自跑 COLMAP；NeuS 需另给 `colmap_model`） |
| `--no-evaluate` / `--no-geometry` / `--no-qualitative` | 关闭对应后处理阶段 |
| `--no-lpips` | 跳过 LPIPS，只算 PSNR/SSIM |
| `--native-crosscheck` | 额外运行 3DGS/nerfstudio 原生指标作交叉校验；会重复部分计算，默认关闭 |
| `--eval-size H W` | 统一缩放到该分辨率再算指标（公平） |
| `--test-ratio 0.1` | held-out 比例（写 `prepared/test.txt`） |
| `--crop-ratio 0.8` | 从图像中心保留宽、高各 80%，去除广角边缘畸变 |
| `--image-size 900 1600` | 中心裁剪后统一缩放到 1600×900 |
| `--sfm-feature-extractor superpoint_aachen` | 可选切换 SfM 局部特征提取器；默认仍是 `sift` |
| `--sfm-feature-matcher superpoint+lightglue` | 可选切换 SfM 学习式匹配器；默认仍走当前 COLMAP 匹配流程 |

`reconstruction.nerf.num_downscales` 默认是 `0`：统一预处理已输出 1600×900，因而不再额外保存未使用的 1/2、1/4、1/8 图像金字塔。

3DGS、NeRF、NeuS 均使用显式 `save_iterations` 列表，而不是固定保存间隔。3DGS 的这些节点也会作为 `test_iterations`，在日志中记录每个快照对应的 train/test PSNR，便于报告绘制收敛曲线。
| `--eval-size 900 1600` | 将各方法 GT/渲染统一到 1600×900 后计算公平指标（不改变训练分辨率） |
| `--dgs-iterations 7000` / `--nerf-iterations 30000` / `--neus-iterations 20001` | 训练迭代数 |
| `--dgs-camera-cache-size 4` | 3DGS 图像留在 CPU，每次只缓存指定数量到 GPU；块内用尽才切换。显存紧张可降到 2 |
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
- **nerfstudio web viewer**（NeRF 交互预览）
  - **快速预览**：直接用 `uv run lab3 --view-run <run-dir>`（见上方命令）。框架会递归扫 `results/nerf/train/` 下最新的 `config.yml` 并拉起 viewer，**默认 png（无损，停下后更清晰）**，无需手填路径。
  - **手动指定 config 时**直接调 `lab3-viewer`（`--view-run` 不透传画质参数）。nerfstudio 训练把 config 写在 `<run-dir>/results/nerf/train/processed/nerfacto/<时间戳>/config.yml`——`processed` 来自数据目录名，`<时间戳>` 是 nerfstudio 按训练时间生成的 `YYYY-MM-DD_HHMMSS`。例：
    ```bash
    lab3-viewer --load-config outputs/lab3/20260618_172030_dormitory/results/nerf/train/processed/nerfacto/2026-06-18_195843/config.yml
    ```
    `lab3-viewer` 是仓库封装入口（`src/lab3/nerfstudio_viewer.py`）：在 import nerfstudio 前压掉当前栈几条低价值 warning，并把 viewer 默认从 jpeg 改成 png（命令行显式传 `--viewer.image-format` 仍优先生效）。原生 `ns-viewer` 也能用，只是终端更吵、且仍是 jpeg 默认。
  - **画质说明**：默认 png 无损、停下后更清晰，代价是交互可能比 jpeg 略卡；viewer 还会动态降分辨率保帧率，所以拖动时发糊属正常，停下后逐步补细，峰值仍可能低于离线 `ns-render`。若 png 明显卡顿，显式退回 jpeg：
    ```bash
    lab3-viewer --load-config <config.yml> --viewer.image-format jpeg --viewer.jpeg-quality 100
    ```
    判画质别在相机运动中看，停 1–3 秒再观察；停住仍明显模糊通常是模型/数据质量问题，不是 viewer 参数。
- **3DGS 原生 SIBR viewer**（真正的实时 splatting 渲染）：在 `gaussian-splatting/SIBR_viewers` 编译后运行
  `./install/bin/SIBR_gaussianViewer_app -m outputs/lab3/<ts>_<scene>/results/3dgs`。
  Windows 下可先用仓库脚本编译：
  `./scripts/lab3/build_sibr_viewer.ps1`

## 输出目录

`outputs/lab3/<timestamp>_<scene>/`：

- `configs/`：`run_config.json`、`prepared_dataset.json`
- `prepared/`：`images/`、`sparse/0/`（共享 COLMAP）、`manifest.csv`、`train.txt`、`test.txt`
- `blur_threshold`（可选）：对输入图片/视频抽帧计算 Laplacian variance 清晰度分数，低于阈值的模糊图会被自动跳过；保留下来的分数写进 `prepared/manifest.csv`
- 阈值不确定时，可先跑 `./scripts/lab3/inspect_blur_scores.sh <input-dir>` 查看输入图片的清晰度分布和建议阈值
- `results/{sfm,3dgs,nerf,neus}/`：各方法训练/渲染产物；NeuS mesh 位于 `results/neus/mesh/sdf_mesh.ply`
- `qualitative/comparison_*.png`：GT vs 方法渲染 vs 误差图（≥3 视角）
- `geometry/{sfm,3dgs,nerf,neus}/`：点云/mesh/Gaussian `.ply` 汇总
- `geometry_metrics.csv`：各方法点云 vs COLMAP dense proxy 的 Chamfer / F-score
- `logs/<method>_<step>.log`：每步外部命令日志
- `logs/*_train_scalars.csv`：从 TensorBoard `tfevents` 导出的训练标量长表（tag / step / value / wall_time）
- `logs/*_train_loss_curve.csv`、`logs/*_train_loss_curve.png`：训练 loss 曲线导出；`3dgs` 与 `nerf` 都会生成
- `metrics.csv`：作业必交，列 `method, psnr, ssim, lpips, metric_source, held_out, train_time_sec, iterations, gpu_mem_peak_gb, render_fps, model_size_mb, gpu, notes`
- `timings.json`：各阶段耗时

## 输出与对应指标

框架最终产出下面这些文件，每个文件直接对应作业 `docs/lab3/lab3.md` 的一条评价要求：

| 产物 | 内容 | 对应作业指标 |
|------|------|-------------|
| `metrics.csv` | 每方法一行：PSNR / SSIM / LPIPS、训练时间、迭代数、GPU 显存峰值、渲染 FPS、模型大小、GPU 型号 | §5.1 PSNR/SSIM/LPIPS；§5.2 训练时间/迭代数/显存峰值/FPS/模型大小/GPU；§6 结果表 |
| `geometry_metrics.csv` | 各方法点云 vs COLMAP dense proxy 的 Chamfer（`chamfer_to_proxy`/`from_proxy`/`sym`）与 F-score（bbox 对角线 0.5% / 1%） | §5.3 几何定量分析（proxy，需诚实披露偏差） |
| `qualitative/comparison_*.png` | GT / 各方法渲染 / 误差图，≥3 个 held-out 视角 | §5.1 定性对比 + 误差图 |
| `geometry/{sfm,3dgs,nerf,neus}/` | COLMAP dense `.ply`、Poisson mesh、3DGS Gaussian `.ply`、nerfstudio/NeuS 导出 | §5.3 几何可视化（孔洞/浮点/边界锐利度） |
| `timings.json` | 各阶段耗时：feature/match/mapper/dense、convert、process_data、train、render、eval | §5.2 预处理 / 训练 / 渲染时间分解 |
| `logs/*_train_scalars.csv`、`logs/*_train_loss_curve.{csv,png}` | 训练过程标量与 loss 曲线导出，便于直接画图/写报告，无需再手动开 TensorBoard | §5.2 训练过程记录；§6 报告图表 |
| `results/3dgs/test/.../results.json`、`results/_eval/{nerf,neus}_eval.json` | 启用 `--native-crosscheck` 后生成的框架官方指标 | 交叉校验统一指标的可信度（默认关闭以节省时间） |
| `configs/run_config.json`、`prepared_dataset.json` | 完整运行配置与训练/测试划分清单 | §8 可复现性 |
| `logs/<method>_<step>.log` | 每条外部命令的完整 stdout/stderr | §8 命令可追踪 |

**测量方式与保真度**

- **PSNR/SSIM**：纯 NumPy 自实现（`lab3.metrics`），SSIM 用 11×11 高斯窗；所有方法用同一套实现、同一 `eval_size` 分辨率、同一 RGB [0,1] 色彩空间。**LPIPS** 懒加载 `lpips` 包，装不上或权重下载失败时返回空并只报 PSNR/SSIM（作业允许）。
- **GPU 显存峰值** `gpu_mem_peak_gb`：训练/MVS 命令运行期间后台每 0.5 s 轮询一次 `nvidia-smi memory.used`，取最大值（GiB）。这是 GPU **全局已用**显存，含同卡其他进程，属 best-effort 近似（作业 §5.2 允许“近似观察”）；无 NVIDIA 驱动 / 无 `nvidia-smi`（如纯 CPU）时留空并在报告说明。
- **渲染 FPS** `render_fps`：3DGS 的 `render.py` 计时**含磁盘 I/O**（写 PNG），数值偏保守；nerfstudio 的 `ns-eval` 计时含渲染故帧数未知、该格常为空，以 `ns-eval` 原生为准。
- **几何指标**：以 COLMAP dense 点云为 **proxy 而非真值**，比较前双方下采样到 4096 点（`downsample_cap`）；F-score 阈值取 proxy bbox 对角线的 0.5% / 1%，尺度无关。`.ply` 加载需 Open3D，缺失则跳过并写说明行。
- **公平性**：默认四方法共享唯一一套全视角 COLMAP 位姿；相机位姿估计可读取全部图片，但 3DGS、NeRF、NeuS 的模型训练只消费 `prepared/train.txt`，并统一在 `prepared/test.txt` 上评测。3DGS adapter 通过其已有的 `sparse/0/test.txt` 能力读取 manifest，NeRF 使用 transforms 显式 split，NeuS 使用同一归一化坐标下的 train/test 元数据。`share_poses=true` 却缺少 SfM 会提前失败，避免静默重复 COLMAP。SfM 为显式点云，RGB 新视角 PSNR 标 `N/A`。
- **分层**：`pipeline` 只遍历 reconstruction registry 并调用统一生命周期（训练、评测、几何、viewer）；命令、split 适配、输出查找和进程入口均由对应 adapter 持有。配置也按 registry 存储和解析，新增方法不需要在 pipeline 增加方法分支。

## 学习式 SfM 提取器 / 匹配器

默认仍是当前流程：`feature_extractor=sift` + `matcher=sequential|exhaustive`，因此老配置无需改动。

如需切到学习式特征，可在 `reconstruction.sfm` 中显式指定，例如：

```json
{
  "feature_extractor": "superpoint_aachen",
  "feature_matcher": "superpoint+lightglue"
}
```

当前支持的 HLOC 预设：

- 提取器：`superpoint_aachen`、`superpoint_max`、`superpoint_inloc`、`disk`、`aliked-n16`
- 匹配器：`superpoint+lightglue`、`disk+lightglue`、`aliked+lightglue`、`superglue`、`superglue-fast`、`NN-mutual`、`adalam`

经验上可优先尝试：

- `superpoint_aachen + superpoint+lightglue`：最稳妥，适合作为学习式默认候选
- `disk + disk+lightglue`：纹理较丰富时通常也很强
- `aliked-n16 + aliked+lightglue`：弱纹理或光照变化更大时值得试

此外也支持 `feature_extractor=sift_dsp`，它仍属于经典 SIFT 路线，但会启用 DSP-SIFT 相关增强选项，通常比 plain SIFT 更容易拿到更多匹配。
