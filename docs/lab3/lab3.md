# 作业：多表示三维场景重建与评估

基于 Lecture 10: 3D Representations 与 Lecture 11: NeRF & 3DGS

## 三维计算成像基础

截止时间：以课程平台通知为准

一句话任务：选取一个真实静态场景，自行拍摄多视角数据，使用三种不同三维表示或重建方法建立场景模型，并从视觉质量、几何质量、训练效率、渲染效率和表示特性等角度做公平比较，最终提交一份实验报告。

## 1 作业目标

本作业对应最近两次课的核心问题：三维场景没有唯一最优表示。点云、Mesh、Voxel、SDF、NeRF、3DGS、2DGS等表示分别在采集、优化、渲染、编辑、几何精度和内存效率上有不同取舍。完成本作业后，应能够：

1. 从真实图像采集出发，走完整个多视角三维重建流程，包括拍摄、相机位姿估计、训练、渲染和结果评估。
2. 至少实践三种不同表示，并解释它们的表示对象、优化变量、输入输出和失败模式。
3. 使用定量指标和定性图表比较方法，而不是只展示看起来最好的结果。
4. 用 Lecture 10 的七个维度分析方法差异: Differentiability、Memory Efficiency、Rendering Speed、Geometric Fidelity、Data Acquisition、Topology Flexibility、ML Integration。
5. 形成研究报告中的基本实验习惯：公平设置、可复现实验、清晰表格、失败案例和诚实讨论。

## 2 任务概述

### 2.1 场景选择

请选择一个你能够合法、稳定、反复拍摄的真实场景。可选例子包括：

- 宿舍、书桌、书架、实验室角落等室内小场景；
- 北京大学校园内的建筑立面、雕塑、小庭院、道路拐角等室外静态场景；
- 一个物体级场景，如椅子、自行车、仪器、手办、盆栽等。

场景应尽量静态、有纹理、有足够视角覆盖。避免大面积镜面、透明玻璃、强反光、纯白墙、重复纹理、动态人群和强烈光照变化。不得拍摄涉及隐私、安防、涉密实验内容或未经同意的人脸、车牌、屏幕内容。

### 2.2 数据采集最低要求

- 输入数据必须由本人或本组自行拍摄，不能直接使用公开 demo 数据集。
- 建议拍摄 50–150 张图像，或拍摄 30–90 秒视频后抽帧。物体级场景应尽量 360 度环绕；房间或校园场景至少覆盖主要可见表面。
- 图像之间要有充分重叠，建议相邻视角重叠超过 $60\%$ 。拍摄时缓慢移动，避免运动模糊。
- 至少预留 $10\%$ 的图像作为 held-out test views, 不参与训练, 只用于 PSNR/SSIM/LPIPS 等评价。
- 报告中必须记录拍摄设备、图像数量、分辨率、拍摄路径、训练/测试划分方式，以及是否做过裁剪、降采样、去模糊或 mask。

## 3 三种表示的选择

必须完成三种不同表示或方法的重建。三种方法应尽量体现不同的表示假设，而不是同一代码库换三个小参数。推荐至少包含一种 Lecture 11 中的 neural rendering/splatting 方法，以及一种显式几何或几何导向方法。

| 类别 | 表示 | 可选方法 | 主要比较点 |
|------|------|----------|-----------|
| 显式点/面 | Point cloud / Mesh | COLMAP sparse/dense reconstruction, Open3D Poisson/alpha shape, MeshLab 等 | 采集和几何直观,渲染快;可能有孔洞、噪声、纹理不完整。 |
| 隐式表面 | SDF / Neural Surface | NeuS, VolSDF, MonoSDF, Gaussian Opacity Field 等 | 几何表面更清晰,可提mesh/normal;训练通常更慢,对 mask/位姿/超参敏感。 |
| Neural radiance field | NeRF / Instant-NGP / Nerfacto / Mip-NeRF | nerfstudio, instant-ngp, 自行实现简化版 NeRF 等 | 新视角合成质量高、可微;表面可能 fuzzy,渲染/训练效率需重点记录。 |
| Splatting | 3DGS / 2DGS | 原版 3DGS、gsplat、2D Gaussian Splatting 等 | 训练和渲染快,显式Gaussian 易可视化;可能占显存,几何/normal 不一定可靠。 |
| 体表示 | Voxel / TSDF | RGB-D/深度相机融合、TSDF fusion、voxel grid baseline 等 | 规则结构清楚,适合体数据;分辨率和内存代价高。 |

允许使用公开代码库和已有工具，但必须在报告中说明代码来源、版本或 commit、关键命令、主要配置和硬件环境。完全闭源的一键商业工具不能作为三种必做方法之一，除非提前获得教师同意；可以作为额外 qualitative reference。

## 4 建议流程

1. 采集与整理：拍摄图像或视频，去掉模糊帧、动态遮挡帧和重复帧。统一命名，建立 images/、images_test/、configs/、outputs/ 等目录。
2. 相机位姿估计: 使用 COLMAP、nerfstudio 的 ns-process-data、手机 LiDAR/VISLAM 或其他工具估计相机位姿。若某些方法使用不同位姿来源，必须说明原因。
3. 训练三种表示：尽量使用相同训练图像、相同图像分辨率和相同测试视角。若某个方法必须使用 mask、深度或不同预处理，需要在报告中单独列出。
4. 渲染与导出：对 held-out test views 渲染图像；能导出几何的，请导出 point cloud、mesh、normal map、depth map 或 Gaussian .ply 等中间结果。
5. 评价与分析：计算图像指标，记录训练/渲染效率，比较几何质量，给出失败案例，并解释这些差异来自表示假设还是实现细节。

## 5 评价要求

### 5.1 新视角合成质量

对所有能够从指定相机视角渲染 RGB 图像的方法，在相同 held-out test views 上计算：

- PSNR: 反映像素级重建误差，越高越好。
- SSIM: 反映结构相似度，越高越好。
- LPIPS: 反映感知差异，越低越好。若环境无法安装 LPIPS，可说明原因并至少提交 PSNR/SSIM。

评价时应统一分辨率、色彩空间、mask/crop 策略。请同时展示至少 3 个测试视角的 qualitative comparison，包括 ground truth、三种方法渲染结果和误差图或局部放大。

### 5.2 效率与资源

每种方法都需要记录:

- 预处理时间、训练时间、训练迭代数或终止条件；
- GPU/CPU 型号、显存或内存峰值（若无法精确记录，可给出近似观察）；
- 模型/场景文件大小，如 .pth、.ckpt、.ply、mesh 文件大小；
- 渲染速度：固定分辨率下每帧时间或 FPS，说明是否包含 I/O 和后处理。

不要只写 "快/慢"。应给出数值，并说明测量方式。例如：800x800，100 test views, batch rendering, average 0.18 s/frame。

### 5.3 几何质量

至少进行一种几何质量分析。可选方式包括：

- 可视化点云、mesh、depth map、normal map 或 Gaussian 分布，观察孔洞、浮点、表面厚度、边界锐利度和细结构；
- 若有 RGB-D、LiDAR、COLMAP dense point cloud 或其他几何 proxy, 可计算 Chamfer Distance、F-score、normal consistency 等指标;
- 对校园建筑、桌面物体等场景，可人工标注几处结构线或平面，比较墙面/桌面是否平整、边缘是否锐利、尺度是否一致；
- 对 3DGS/2DGS，可额外讨论 Gaussian 是否漂浮、是否穿透表面、normal/mesh 导出是否可信。

如果没有可靠几何真值, 可以使用 "proxy + 定性分析" 的方式, 但必须诚实说明 proxy 的偏差, 不能把 COLMAP 或某一方法的输出当成绝对真值。

## 6 实验报告要求

最终报告建议 6-10 页，不含附录。中文或英文均可，中文报告可保留英文技术术语。报告必须是一个完整实验故事，而不是命令截图堆叠。

1. 摘要：100–200 字概括场景、三种表示、核心发现。
2. 数据采集：说明场景、拍摄路径、图像数量、训练/测试划分、相机位姿估计是否成功。建议给出一张相机轨迹图或采集示意图。
3. 方法：逐一说明三种表示的数学对象和实现选择。例如 NeRF 是坐标和视角到 density/color 的 neural field，3DGS 是带位置、协方差、opacity、SH 颜色的 Gaussian 集合，NeuS 是 SDF 零等值面加 volume rendering。
4. 实验设置：列出硬件、软件版本、关键超参、训练分辨率、迭代数、评价分辨率和评价脚本。
5. 结果：给出定量表格和定性图。定量表至少包含 PSNR/SSIM/LPIPS、训练时间、渲染 FPS、模型大小。几何结果可单独成表或图。
6. 讨论: 使用 Lecture 10 的七个维度分析三种表示的取舍, 并结合 Lecture 11 讨论 NeRF、NeuS、3DGS/2DGS 的具体失败模式。
7. 结论：给出你对该场景最推荐的表示，并解释如果要继续改进，下一步会增加什么数据、先验或正则项。
8. 附录：放关键命令、配置、代码链接、commit、额外图表和失败样例。

### 建议结果表格式

| 方法 | PSNR | SSIM | LPIPS | 训练时间 | FPS | 主要观察 |
|------|------|------|-------|----------|-----|----------|
| Method A |      |      |       |          |     |          |
| Method B |      |      |       |          |     |          |
| Method C |      |      |       |          |     |          |

## 7 提交内容

请提交一个压缩包和一份报告 PDF。推荐目录结构如下：

```ignorefile
student_id_name_assignment/
report.pdf
README.md
configs/
scripts/
results/
metrics.csv
qualitative/
geometry/
logs/
```

README.md 至少包含环境说明、复现实验的主要命令、数据路径说明和每种方法的输出位置。原始图像或视频若体积过大，可不直接上传，但需要提交代表性样例、训练/测试文件列表，并按课程平台要求提供可访问链接。禁止提交与作业无关的大型缓存文件。

## 8 评分标准

| 项目 | 分值 | 要求 |
|------|------|------|
| 数据采集与预处理 | 15 | 自行采集、覆盖充分、划分合理、记录完整;相机位姿或替代方案说明清楚。 |
| 三种表示实现 | 25 | 三种方法确实对应不同表示;训练流程正确;输出可视化和中间几何/模型文件完整;公开代码使用有引用和版本说明。 |
| 评价设计与公平性 | 25 | 有held-out test views;图像指标、效率指标、几何分析完整;评价设置一致,例外有解释。 |
| 结果分析与洞察 | 25 | 能从表示本身解释优劣,而不是只罗列数值;包含失败案例;能联系七维选择框架和 Lecture 11 的 NeRF/NeuS/3DGS 机制。 |
| 报告质量与可复现性 | 10 | 图表清晰、文字准确、命令和配置可追踪、结论可信。 |
| 额外加分 | +10 | 自己实现一个简化模块;加入深度/法向/语义先验;做消融实验;完成可交互 viewer 或高质量视频;对失败原因提出并验证改进方案。 |

以下情况会显著扣分：未使用自采数据；少于三种方法；没有 held-out 评价；只提交截图没有实验记录；指标不可复现；伪造结果；未注明公开代码或他人素材来源。

## 9 提示与常见问题

- 拍摄比调参更重要。多数失败来自图像模糊、重叠不足、反光、重复纹理或动态物体。请先确认 COLMAP 或位姿估计是否稳定，再开始长时间训练。
- 保持输入公平。不同方法可以有各自推荐配置，但训练图像、测试图像、分辨率、mask和裁剪策略应尽量一致。
- 不要迷信 PSNR。PSNR 高不代表几何好。NeRF 可能渲染好但表面 fuzzy；3DGS 可能实时渲染好但 Gaussian normal 不准；NeuS/2DGS 可能几何好但训练更慢或外观不如 3DGS。
- 几何 proxy 不是绝对真值。如果用 COLMAP dense point cloud 评价 NeuS 或 3DGS mesh，应说明 COLMAP 自身也可能有孔洞和噪声。
- 硬件限制要诚实说明。部分 3DGS/2DGS/instant-ngp 实现强依赖 CUDA GPU。如果只能在 CPU、Mac 或低显存环境运行，可以降低图像数量和分辨率，或使用课程允许的服务器；报告中说明限制即可。

## 10 可选工具与参考资料

- COLMAP: Structure-from-Motion 和 Multi-View Stereo 工具，可用于相机位姿、稀疏点云和 dense reconstruction。官方文档：https://colmap.github.io/；官方 GitHub: https://github.com/colmap/colmap。
- nerfstudio：支持自定义图像/视频数据处理和 Nerfacto 等 NeRF 训练流程；常用数据处理命令为 ns-process-data，可调用 COLMAP/FFmpeg。文档：https://docs.nerf.studio/。
- instant-ngp: NVIDIA 的 multiresolution hash encoding 实现, 包含 NeRF、SDF、neural image、neural volume 等 primitives。GitHub: https://github.com/NVlabs/instant-ngp。
- 3D Gaussian Splatting: SIGGRAPH 2023 原始项目页与代码。项目页: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/；代码: https://github.com/graphdeco-inria/gaussian-splatting。
- NeuS: 使用 SDF 零等值面和 volume rendering 做 neural surface reconstruction。NeurIPS 论文页：https://proceedings.neurips.cc/paper/2021/hash/e41e164f7485ec4a28741a2d0ea41c74-Abstract.html。
- 2D Gaussian Splatting: 用 2D oriented disks/surfels 改善几何重建质量。GitHub:https://github.com/hbb1/2d-gaussian-splatting。
- Open3D / MeshLab / Blender: 用于点云、mesh、法向、纹理和可视化检查。使用时请在报告中注明具体版本和处理步骤。

## 11 完成信号

如果你的报告能够回答下面几个问题，基本就达到了本作业的目标：

1. 同一个自采场景中，三种表示分别学到了什么，丢掉了什么？
2. 哪个方法新视角合成最好？哪个方法几何最可信？哪个方法训练或渲染最快？
3. 这些差异能否用表示结构解释，而不是只用"实现好坏"解释？
4. 如果要把结果用于 AR/VR、校园数字孪生、机器人导航或后续编辑，你会选择哪种表示，为什么？
