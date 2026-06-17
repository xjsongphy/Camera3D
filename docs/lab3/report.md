# 实验报告模板：多表示三维场景重建与评测

> 本模板对齐 `docs/lab3/lab3.md` §6 的八节结构。代码产出（`metrics.csv`、`qualitative/`、`geometry/`、`logs/`、`timings.json`、`configs/run_config.json`）按节内标注填入。报告建议 6–10 页，中文/英文均可。

## 1. 摘要（100–200 字）

- 场景：______（室内/室外/物体级，自采）。
- 三种表示：COLMAP 点云（显式几何）、3D Gaussian Splatting（splatting）、Nerfacto（neural radiance field）。
- 核心发现：新视角合成最优 = ______；几何最可信 = ______；训练/渲染最快 = ______。
- 位姿来源：共享同一 COLMAP `sparse/0`（`share_poses=true`）。

## 2. 数据采集

- 拍摄设备/分辨率/张数：______（见 `configs/run_config.json` 的 `fps`/`image_limit`、`prepared/manifest.csv` 的 `image_count`）。
- 训练/测试划分：`prepared/train.txt`、`prepared/test.txt`（held-out ≈ `test_ratio`）。
- 相机轨迹图：______（建议附 COLMAP `model_converter` 导出的稀疏点云+位姿截图，或 nerfstudio viewer）。
- 预处理：裁剪/降采样/mask/去模糊说明 ______。

## 3. 方法（数学对象 + 实现）

- **SfM 点云**：COLMAP SfM 得相机位姿与稀疏点云；`dense=true` 时 MVS 融合 dense 点云（`results/sfm/dense/fused.ply`）。优化变量 = 相机内外参与 3D 点；输入图像、输出点云。失败模式：弱纹理/反光处空洞、噪声。
- **3DGS**：带位置/协方差/opacity/SH 颜色的各向异性高斯集合；`train.py` 梯度优化 + 致密化/剪枝；输出 `point_cloud/iteration_N/point_cloud.ply`。失败模式：浮点高斯、几何/normal 不可靠、显存高。
- **Nerfacto**：坐标/视角 → density/color 的 neural field（含 proposal sampler/瞬时外观等）；`ns-train`。失败模式：表面 fuzzy、训练/渲染较慢。

## 4. 实验设置

- 硬件：见 `metrics.csv` 的 `gpu` 列（如 NVIDIA RTX 4070 Ti SUPER, 16 GB）。
- 软件/版本：COLMAP ______、nerfstudio ______、gaussian-splatting commit ______、torch ______。
- 关键超参：3DGS `iterations` ______；nerfstudio `max_num_iterations` ______；`eval_size` ______。见 `configs/run_config.json`。
- 评测分辨率：`eval_size` 或原生 ______；指标实现：`lab3.metrics`（统一 PSNR/SSIM，LPIPS 视安装）。

## 5. 结果

定量表（取自 `metrics.csv`）：

| 方法 | PSNR | SSIM | LPIPS | 训练时间(s) | FPS | 模型大小(MB) | 备注 |
|------|------|------|-------|------------|-----|-------------|------|
| SfM 点云 | N/A | N/A | N/A | `train_time_sec` | N/A | `model_size_mb` | 几何为主 |
| 3DGS | | | | | | | 统一指标 + results.json 交叉校验 |
| Nerfacto | | | | | | | ns-eval 原生指标 |

效率/资源：训练时间见 `timings.json`，渲染 FPS 见 `metrics.csv`（说明测量口径：固定分辨率、batch/单帧、是否含 I/O）。模型大小见 `metrics.csv`。峰值显存 best-effort 从 `logs/3dgs_train.log`、`logs/nerf_train.log` 读取。

定性：`qualitative/comparison_*.png`（GT | 方法渲染 | 误差图，≥3 视角）。几何：`geometry/{sfm,3dgs}/` 的点云/mesh；可附深度/法向图。

## 6. 讨论（七维框架 + Lecture 11 失败模式）

用 Lecture 10 七维分析三者取舍：Differentiability / Memory Efficiency / Rendering Speed / Geometric Fidelity / Data Acquisition / Topology Flexibility / ML Integration。结合 Lecture 11 讨论 NeRF（表面 fuzzy）、NeuS/隐式面（几何好但慢，本次未做可作对比引用）、3DGS/2DGS（实时但 normal/mesh 不可靠）的具体失败模式。

诚实说明：3DGS（每 8 张留一）与 nerfstudio（原生 eval split）held-out 集合不完全相同；位姿共享、指标实现/分辨率统一。

## 7. 结论

对该场景最推荐表示 = ______，理由 ______。下一步改进：增加数据/先验/正则项 ______。

## 8. 附录

- 关键命令、配置（`configs/run_config.json`）、代码链接与 commit、`metrics.csv`、`timings.json`、`logs/`、失败样例。

## 加分项（§8 额外加分，+10）

- 交互式 viewer：`lab3 --view-run outputs/lab3/<run>` 开 Open3D（SfM 点云/3DGS Gaussian/mesh）+ nerfstudio web viewer（NeRF）；3DGS 真·实时 splatting 用 SIBR viewer（见 `docs/lab3/README.md`）。
- 评测/几何/定性/位姿共享为自实现模块（`lab3.metrics`/`lab3.evaluate`/`lab3.geometry`/`lab3.qualitative`）。
