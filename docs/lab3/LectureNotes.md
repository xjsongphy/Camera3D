# Lecture 10–11：三维表示的取舍与失败模式笔记

## 0. 总论：三维表示没有唯一最优解

Lecture 10 的核心观点是：**三维世界没有一个 universally best 的 representation**。2D 图像里，image grid 是非常自然、统一、稳定的表示；但 3D 不一样，点云、mesh、voxel、SDF、NeRF、3DGS/2DGS 都是在不同应用需求下发展出来的。不同表示分别偏向采集、渲染、编辑、几何、压缩、机器学习或工业部署，所以不能简单说某一种“最先进”或“最好”。课程作业说明中也明确把点云、Mesh、Voxel、SDF、NeRF、3DGS、2DGS 归为有不同采集、优化、渲染、编辑、几何精度和内存效率取舍的表示。

从表示对象看，可以粗略分成三类：

**显式几何表示**，例如 point cloud、mesh、voxel/TSDF。它们直接存点、面、格子或体素，几何直观，容易可视化和编辑，但可能有孔洞、噪声、拓扑限制或高内存开销。作业参考表中也把 point cloud/mesh 的特点概括为采集和几何直观、渲染快，但可能有孔洞、噪声和纹理不完整。

**隐式神经场表示**，例如 SDF、NeuS、NeRF。它们不直接存表面三角形，而是把空间中的每个点映射到某种属性，比如 SDF 值、density、color。Lecture 11 里讲的 neural field 就是“给一个坐标，网络输出对应场景属性”的思想：二维时可以输入 ((x,y)) 输出 RGB，三维时可以输入 ((x,y,z)) 或再加 view direction，输出 density/color/SDF 等。

**Splatting 表示**，例如 3DGS/2DGS。它们用一组显式 Gaussian 或 oriented disk/surfel 来近似场景。它比纯神经场更工程化，渲染快，容易交互展示，但这些 primitive 不一定等价于真实几何表面。

---

## 1. Lecture 10 的七个维度

Lecture 10/作业要求的七个维度是：

1. **Differentiability**
2. **Memory Efficiency**
3. **Rendering Speed**
4. **Geometric Fidelity**
5. **Data Acquisition**
6. **Topology Flexibility**
7. **ML Integration**

这七个维度不是独立的，经常互相冲突。例如，一个表示越适合实时渲染，未必越适合几何测量；一个表示越适合精确几何，未必越容易优化和部署。作业要求也强调要使用这七个维度分析方法差异，而不是只看渲染图好不好看。

---

## 2. Differentiability：是否容易参与梯度优化

**NeRF** 的 differentiability 很强。它把场景表示成一个 neural radiance field，输入空间点和视角方向，输出 density 和 color；再通过 volume rendering 合成图像。整条链路从像素 loss 到网络参数基本可微，所以可以直接用图像监督优化场景表示。Lecture 11 里也强调，NeRF 这类方法的核心是把 3D 表达转化成神经网络形式，并用神经网络优化。

**NeuS** 也很可微，但比 NeRF 多了一层几何约束：它使用 SDF 零等值面表示表面，再把 SDF 转换成 volume rendering 中的 opacity/density。这样既保留了可微渲染，又加入了“表面”概念。问题是这个转换并不简单，Lecture 11 里提到 NeuS 的核心问题就是如何把 SDF “无偏”地转换成 volume render 的值。

**3DGS/2DGS** 也是可微的，但它的可微性更偏工程实现。Gaussian 的位置、尺度、旋转、opacity、颜色参数可以通过 differentiable rasterization/splatting 优化；但 densification、pruning、splitting 等过程通常带有启发式和离散操作。因此它不是像 NeRF 那样纯粹的连续 neural field，而是“可微优化 + 显式 primitive 管理”的混合系统。

---

## 3. Memory Efficiency：表示是否省内存

**NeRF** 的参数量通常不大，一个 MLP 或 hash grid 可以比较紧凑地编码整个场景。但它的内存效率不能只看模型大小，因为渲染时需要沿每条 ray 采样大量点，并对每个点 query 网络。因此 NeRF 的瓶颈常常不是“文件很大”，而是“渲染和训练计算很重”。

**NeuS** 类似，SDF 网络本身可以很紧凑，但为了得到稳定表面，需要大量空间查询、采样和正则化。它的内存表示不一定大，但优化过程开销高。

**3DGS** 的模型往往更大。因为每个 Gaussian 都要存位置、尺度、旋转、opacity、颜色或 spherical harmonics 系数。场景越复杂，densification 之后 Gaussian 数量越多，显存和模型文件都会增长。它牺牲了一部分存储效率，换来了快速 rasterization 和高质量新视角渲染。作业说明也把 3DGS/2DGS 的问题概括为“可能占显存，几何/normal 不一定可靠”。

**2DGS** 相比 3DGS 更偏表面化，用 oriented disks/surfels 表示场景，几何约束更强，但也需要存大量 surfel 参数。它不一定比 3DGS 更省内存，主要优势不在压缩，而在几何一致性。

---

## 4. Rendering Speed：渲染速度

**原始 NeRF 渲染慢**。它需要对每条 ray 上许多采样点调用神经网络，再做体渲染积分。Lecture 11 的录音稿里明确提到，如果有大量像素、每条 ray 又采很多点，就需要计算非常多次神经网络，因此训练和优化都很 heavy；早期 NeRF 训练甚至需要一天，后来才有 Instant-NGP 等方法大幅加速。

**NeuS 通常比 NeRF 更慢或至少不更快**。原因是它不仅要渲染图像，还要维护 SDF 表面结构，通常还会加入 eikonal regularization、surface sampling、mask loss 等约束。它是用效率换几何。

**3DGS 的最大优势就是渲染快**。它不需要沿 ray 密集采样网络，而是把一组 Gaussian 投影到屏幕上做 splatting/rasterization。因此它非常适合实时 viewer、移动端和交互应用。Lecture 11 里也提到 3DGS 之所以火，是因为它能更容易搬到手机端或工业场景中。

**2DGS** 通常也保留 splatting 的快速渲染优势，但由于它更强调 surface-aligned primitives 和几何 regularization，训练或优化可能比普通 3DGS 更复杂。

---

## 5. Geometric Fidelity：几何真实性

这是三者差异最关键的维度。

**NeRF 的图像质量好，不等于几何好。** NeRF 优化的是图像重建误差。它学到的是 density field 和 view-dependent color，而不是严格表面。因此它可以在图像上看起来对，但内部几何可能是一团 fuzzy density。作业提示也明确指出：PSNR 高不代表几何好，NeRF 可能渲染好但表面 fuzzy。

NeRF 的几何问题本质上来自 **radiance field 与 surface geometry 的不等价**。一个 density field 可以通过一团半透明体积解释图像；只要合成后的像素颜色对，优化目标未必强迫它形成清晰表面。因此 NeRF 很适合 novel view synthesis，但不一定适合 mesh extraction、normal estimation 或物理仿真。

**NeuS 的几何 fidelity 更强。** 它使用 SDF 零等值面表示 surface，因此天然可以提取 mesh 和 normal。作业表中也把 NeuS/Neural Surface 的主要优点写成“几何表面更清晰，可提 mesh/normal”。

但 NeuS 的问题是：它强行假设场景可以被一个清晰的 SDF surface 表达。对于透明玻璃、水面、强反光、毛发、树叶、半透明物体、薄结构、多层遮挡等，单一清晰表面的假设会被破坏。它的几何更“像几何”，但未必更能解释复杂真实图像。

**3DGS 的视觉质量高，但 Gaussian 不等于真实表面。** 3D Gaussian 是为了图像 splatting 优化的 primitive，不一定对应真实物体表面的局部切平面。它可以通过一堆半透明椭球把图像渲染得很锐利，但 Gaussian 的 normal、scale、opacity 并不能直接解释成真实几何。因此作业提示里也特别说：3DGS 可能实时渲染好，但 Gaussian normal 不准。

**2DGS** 的设计动机就是修正 3DGS 的几何弱点。它用 2D oriented disks/surfels 改善几何重建质量，因此比 3DGS 更接近 surface representation。作业参考资料中也把 2DGS 描述为用 oriented disks/surfels 改善几何重建质量。 但如果场景不是清晰表面，2DGS 的表面化约束反而可能让外观变差。

---

## 6. Data Acquisition：数据采集难度

三维重建的失败常常不是模型本身造成的，而是输入数据造成的。Lecture 10 里强调，2D 拍照很容易，但 3D 数据采集很难，真实 3D 数据通常很脏；点云、深度、位姿都会有噪声。

对 **NeRF / NeuS / 3DGS** 来说，最基础的前提都是可靠 camera poses。实际流程里通常依赖 COLMAP/SfM。如果场景纹理少、重复纹理多、反光强、透明物体多、运动模糊明显、动态人群多，COLMAP 就可能失败。一旦位姿错了，三种方法都会继承错误：

* NeRF 会出现 ghosting、double edges、floating density；
* NeuS 会出现破碎表面、厚壳、错误 mesh；
* 3DGS 会出现漂浮 Gaussian、边缘毛刺、背景泄漏。

作业 FAQ 也总结了这一点：多数失败来自图像模糊、重叠不足、反光、重复纹理或动态物体，应该先确认 COLMAP 或位姿估计是否稳定。

**NeRF** 对数据的容忍度相对强一点，因为它可以用 density 和 view-dependent appearance 去“糊”出可看的结果；但这也会掩盖几何错误。

**NeuS** 对数据更挑，因为它试图恢复清晰表面。如果相机位姿或 mask 错了，SDF 会被迫拟合矛盾观测，表面就容易破碎。

**3DGS** 对初始化和视角覆盖也敏感。如果 COLMAP 点云稀疏或错误，高斯初始化就差；后续 densification 可能会放大错误，形成漂浮物或孔洞。

---

## 7. Topology Flexibility：拓扑灵活性

**NeRF** 的拓扑灵活性很强，因为它不是显式 surface，而是连续体场。只要 density field 能表达，复杂拓扑、多物体、洞、非流形结构都可以被编码。但代价是拓扑不清楚：它不直接给出一个 clean manifold surface。

**NeuS** 的拓扑也比较灵活，因为 SDF 零等值面可以表达复杂拓扑；课程里讲 SDF 时也强调它可以处理 complex topology，并且 neural network friendly、implicit encoding。Lecture 11 的 SDF 总结页中列出了 SDF 的优点，包括 differentiable、neural network friendly、implicit encoding、handles complex topology，同时缺点包括 indirect access、geometry only、expensive dense queries。

但 SDF 的拓扑灵活不等于任何东西都能重建。开放表面、薄片、半透明结构、多层反射结构都会让 SDF 的“内外”和“最近表面距离”概念变得不稳定。

**3DGS** 的拓扑极其灵活，因为它根本不要求 manifold。你可以用任意数量的 Gaussian 拼出任意复杂视觉效果。但这也是问题：太灵活会导致它不受真实表面约束，容易形成漂浮 Gaussian、半透明雾状结构或几何不一致。

**2DGS** 相比 3DGS 更接近 surface，因此拓扑约束更强。它更像一堆 surface splats，而不是体状 Gaussian。因此几何更稳定，但对于复杂体效应、毛发、树叶、透明物体和非表面外观，表达能力可能下降。

---

## 8. ML Integration：与机器学习系统的兼容性

**NeRF** 是 neural field，天然适合和深度学习结合。它可以接入各种网络结构、位置编码、hash encoding、latent code、regularization 和生成模型。Lecture 11 里也说，NeRF 重要的地方在于它启发了后续很多 3D 表达工作，让大家看到 3D 表达可以和 learning 结合。

**NeuS** 同样是 neural surface representation，也适合加入深度、法向、mask、语义、Eikonal loss 等先验。它比 NeRF 更几何导向，因此更适合几何学习、mesh extraction、normal supervision 等任务。

**3DGS/2DGS** 的 ML integration 稍微复杂。它们不是规则 grid，也不是简单 MLP，而是一组数量可变、结构不规则的 primitives。它们可以优化，也可以可微渲染，但如果要和 transformer、大模型、机器人 memory、语义理解结合，就需要设计额外的数据结构或 tokenization。Lecture 10 里也提到，3D 表示难的一个原因是我们没有像 2D image 那样统一、容易被神经网络利用的表示。

---

# 9. NeRF 的具体失败模式

## 9.1 表面 fuzzy

NeRF 学的是 density，不是显式表面。体渲染只要求 ray 积分后的颜色正确，不要求 density 集中在一个薄表面上。因此它容易出现：

* 物体边界模糊；
* 空间中有 floating density；
* 提取 mesh 时表面厚、破碎或像雾；
* PSNR 高但 depth/normal 不可信。

这就是“渲染好但几何差”的典型情况。课程和作业提示都强调不能迷信 PSNR，因为 NeRF 可能渲染好但表面 fuzzy。

## 9.2 View-dependent appearance 掩盖几何错误

NeRF 输入 view direction，可以学习视角相关颜色。因此高光、反射、透明边缘等现象可以被网络用 appearance 解释，而不一定需要正确几何。结果就是：

* 训练视角看起来不错；
* 新视角稍微远离训练分布就出错；
* 几何表面与真实物体位置不一致；
* 反光被“烘焙”进视角相关颜色。

## 9.3 训练和渲染重

原始 NeRF 需要大量 ray sampling 和网络查询。Lecture 11 里明确提到这种方法计算量很大，训练和优化不容易，早期 NeRF 训练甚至需要一天。 因此 NeRF 的失败不一定是结果完全错，也可能是工程上太慢，难以用于实时交互。

## 9.4 依赖准确相机位姿

NeRF 通常不是自己解决 pose estimation，而是假设已经有相机位姿。位姿来自 COLMAP/SfM 时，如果输入图像有低纹理、重复纹理、模糊、动态物体，NeRF 会直接继承这些错误。Lecture 11 里也提到不能直接做 NeRF，因为需要知道相机 pose，而 COLMAP 在某些场景会失效，动态场景尤其困难。

---

# 10. NeuS 的具体失败模式

## 10.1 SDF 到 opacity 的转换不稳定

NeuS 的目标是把 SDF 表面接入 volume rendering。问题是 SDF 是几何量，volume rendering 需要 opacity/density。如何从 SDF 转成无偏的渲染权重，是 NeuS 的核心技术难点。Lecture 11 里也提到 NeuS 的核心问题就是如何把 SDF 无偏地转换成 volume render 的值。

如果转换或采样策略不稳定，就会出现：

* 表面偏移；
* 表面厚度不对；
* thin structure 消失；
* mesh 有洞或多层壳。

## 10.2 对 mask、位姿、超参敏感

NeuS 要重建清晰表面，因此比 NeRF 更依赖干净输入。如果 mask 把背景、阴影、地面误当作前景，SDF 可能会把这些区域也解释成 surface。如果 pose 有误，多视角观测不一致，SDF 会被迫折中，导致：

* 表面破碎；
* 物体边缘膨胀；
* 出现 ghost shell；
* mesh 和纹理错位。

作业表里也明确总结 NeuS/Neural Surface：几何表面更清晰、可提 mesh/normal，但训练通常更慢，并且对 mask、位姿、超参敏感。

## 10.3 清晰表面假设失效

NeuS 更适合 opaque、静态、有清楚边界的物体。如果场景中有水面、玻璃、镜面、半透明塑料、毛发、树叶、烟雾等，它的 SDF surface 假设就会变差。因为这些对象的图像形成机制不是“光线打到某个单一表面后反射”，而是折射、反射、散射、多层遮挡或体效应。

## 10.4 外观可能不如 NeRF/3DGS

NeuS 牺牲一部分 appearance flexibility 来换几何。它的 mesh 可能更合理，但图像可能不如 3DGS 锐利，PSNR/LPIPS 不一定最好。Lecture 11 里也有一个重要判断：渲染好的几何未必好，几何好的渲染未必好，几何和渲染之间存在 trade-off。

---

# 11. 3DGS/2DGS 的具体失败模式

## 11.1 Gaussian normal 不等于真实 normal

3DGS 的 Gaussian 是为了图像 splatting 优化的，不是为了严格拟合真实表面。一个 Gaussian 的长短轴、旋转和 opacity 不一定对应局部物体表面的 tangent plane。因此：

* normal map 可能不可信；
* mesh extraction 可能不稳定；
* Gaussian 看起来贴在物体上，但几何并不准确；
* 视觉边缘锐利，不代表几何边缘正确。

作业提示明确指出：3DGS 可能实时渲染好，但 Gaussian normal 不准。

## 11.2 漂浮 Gaussian 和背景泄漏

3DGS 优化的是图像误差。如果某个区域视角不足、纹理弱、位姿有误或背景复杂，它可能通过在空间中放置一些半透明 Gaussian 来解释训练图像。这会产生：

* floating artifacts；
* 半透明雾状区域；
* 物体边缘毛刺；
* 背景颜色泄漏到前景；
* 新视角下出现孔洞或重影。

## 11.3 显存和模型大小问题

3DGS 的效率优势来自大量显式 Gaussian primitives。一旦场景复杂，densification 会产生很多 Gaussian。于是会有：

* 显存占用高；
* 模型文件大；
* 训练后期 primitive 数量膨胀；
* 低显存设备上难以跑完整分辨率。

作业说明也把 3DGS/2DGS 的主要比较点概括为训练和渲染快、显式 Gaussian 易可视化，但可能占显存，几何/normal 不一定可靠。

## 11.4 训练视角过拟合

3DGS 很擅长把训练视角渲染得锐利，因此容易过拟合输入视角。若相机轨迹覆盖不足，未观察到的背面、侧面、遮挡区域会出现：

* 新视角露洞；
* 物体背面模糊；
* texture 被拉伸；
* Gaussian 分布不符合真实空间结构。

## 11.5 2DGS 的额外取舍

2DGS 用 oriented disks/surfels 代替 3D ellipsoidal Gaussians，目的是提高几何表面质量。它的失败模式和 3DGS 有重叠，但重点不同：

* 如果 surfel normal 估计错，会出现破面、闪烁、边界破碎；
* 如果场景不是清晰表面，比如水面、玻璃、毛发、树叶，2D surface primitive 会表达困难；
* 因为几何约束更强，它可能不如 3DGS 那样容易“糊出”好看的图像；
* 对正则项和训练策略更敏感。

因此，2DGS 可以看作是在 3DGS 的“实时渲染优势”和 NeuS 的“几何表面优势”之间取中间路线：比 3DGS 更几何化，但不一定比 3DGS 更好看；比 NeuS 更快，但几何理论上未必像 SDF 那么干净。

---

# 12. 三种表示的整体对比

| 维度   | NeRF                             | NeuS                        | 3DGS / 2DGS                           |
| ---- | -------------------------------- | --------------------------- | ------------------------------------- |
| 表示对象 | 连续 radiance field                | SDF 零等值面 + volume rendering | Gaussian / disk / surfel primitive    |
| 核心优势 | 新视角合成质量好，ML 友好                   | 几何表面清晰，可提 mesh/normal       | 训练和渲染快，实时性强                           |
| 主要牺牲 | 几何 fuzzy，训练/渲染重                  | 训练慢，对 mask/pose/超参敏感        | 几何/normal 不一定可靠，显存可能大                 |
| 更适合  | 图像合成、view-dependent appearance   | 几何重建、mesh、normal            | 实时展示、交互 viewer、移动端                    |
| 典型失败 | floating density、模糊表面、pose 错导致鬼影 | 表面破碎、厚壳、mask 错、薄结构失败        | floating Gaussians、normal 错、边缘毛刺、显存膨胀 |

---

## 13. 核心结论

NeRF、NeuS、3DGS/2DGS 的差异，本质上不是“哪个方法更强”，而是三种目标函数和表示假设的差异：

**NeRF** 优先解决“图像怎么看起来对”。它通过 density/color field 和 volume rendering 拟合多视角图像，因此 appearance 表达强，但 geometry 不一定可靠。

**NeuS** 优先解决“表面在哪里”。它把 SDF 零等值面作为核心，因此 geometry 更清楚，但对数据质量、mask、pose 和优化细节更敏感。

**3DGS/2DGS** 优先解决“如何快速渲染”。它用显式 splats 代替密集 neural ray marching，因此速度很快、视觉锐利，但 splat primitive 未必等价于真实几何。

所以，七个维度可以压缩成一句话：

**NeRF 偏可微学习和新视角合成，NeuS 偏几何表面，3DGS/2DGS 偏工程实时渲染；三者的失败模式分别来自 density field 不等于 surface、SDF surface 假设过强、Gaussian/surfel primitive 不等于真实几何。**
