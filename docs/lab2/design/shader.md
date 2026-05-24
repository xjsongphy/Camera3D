对，这样更合理。`depth` 平时不进入训练数据流，只在自检和可视化时单独计算，可以减少每次渲染的开销。下面是调整后的版本。

---

## 渲染器设计

实现一个 `StructuredLightRenderer` 类，用于模拟结构光投影和相机成像。渲染器接收当前结构光 pattern，输出相机观测图和真值对应关系图，并提供数值差分接口，用于 OpticalSGD 的有限差分梯度估计。深度图只在可视化或自检时单独计算，默认训练流程中不输出。

---

## 主要数据

```python
patterns: Tensor[K, Wp]
```

表示 (K) 张一维结构光图案，`Wp` 是投影仪列数。

```python
patterns[k, x]
```

表示第 `k` 张图案在投影仪第 `x` 列的灰度值。

---

## 核心函数

### `load_scene(scene_path)`

加载场景几何和材质。

作用：

```text
读入带材质的三维场景，作为结构光投影和相机拍摄的对象。
```

---

### `set_camera(camera_config)`

设置相机参数。

包括：

```text
相机内参
相机外参
相机分辨率 Hc × Wc
```

作用：

```text
定义相机的位置、朝向、焦距和成像分辨率。
```

---

### `set_projector(projector_config)`

设置投影仪参数。

包括：

```text
投影仪内参
投影仪外参
投影仪分辨率 Hp × Wp
```

作用：

```text
定义投影仪的位置、朝向、分辨率，以及如何把 pattern 投射到场景上。
```

---

### `set_lights(light_config)`

设置普通光源和环境光。

作用：

```text
控制场景中的背景照明，用来模拟不同光照条件。
```

---

### `set_patterns(patterns)`

设置当前使用的结构光图案。

输入：

```python
patterns: Tensor[K, Wp]
```

作用：

```text
保存当前 K 张待投影的结构光 pattern。
```

---

### `update_patterns(patterns)`

更新结构光图案。

作用：

```text
优化器每次更新 pattern 后，调用这个函数把新的 pattern 写入渲染器。
```

实际实现中，`set_patterns()` 和 `update_patterns()` 可以合并。

---

### `render_images()`

渲染当前 pattern 下的相机图像。

输出：

```python
images: Tensor[K, Hc, Wc]
```

作用：

```text
依次投影 K 张结构光 pattern，并生成对应的 K 张相机观测图。
```

也就是：

```text
第 1 张 pattern → 第 1 张相机图
第 2 张 pattern → 第 2 张相机图
...
第 K 张 pattern → 第 K 张相机图
```

这是训练过程中最频繁调用的函数，应尽量保持轻量，不默认计算 depth。

---

### `compute_gt_corr()`

计算真值对应关系图。

输出：

```python
gt_corr: Tensor[Hc, Wc]
```

作用：

```text
对每个相机像素，计算它看到的三维点对应投影仪的哪一列。
```

即：

[
gt_corr[u_c,v_c]=x_p
]

其中 `x_p` 是该相机像素对应的投影仪列坐标。

这个量用于监督 ZNCC / ZNCC-NN decoder。decoder 预测出 `pred_corr` 后，与 `gt_corr` 对比计算 correspondence error。

`gt_corr` 只依赖相机、投影仪和场景几何，不依赖当前 pattern 的灰度值，因此可以预计算或缓存。

---

### `render_train_batch()`

输出训练所需的最小数据。

输出：

```python
{
    "images": images,
    "gt_corr": gt_corr
}
```

作用：

```text
作为 OpticalSGD 优化器的主要调用接口。
```

训练时不返回 depth，避免每次迭代都计算额外几何信息。

---

## 深度图函数

### `render_depth_for_visualization()`

单独计算相机视角下的深度图。

输出：

```python
depth: Tensor[Hc, Wc]
```

作用：

```text
只在自检、调试或保存可视化结果时调用，用于检查场景几何和相机视角是否正确。
```

注意：

```text
depth 不参与默认训练流程；
depth 不作为 ZNCC / OpticalSGD loss 的直接监督；
训练主要使用 gt_corr。
```

---

## 数值差分函数

### `finite_difference(patterns, direction, eps)`

计算 pattern 沿某个扰动方向的数值差分。

输入：

```python
patterns: Tensor[K, Wp]
direction: Tensor[K, Wp]
eps: float
```

输出：

```python
d_images: Tensor[K, Hc, Wc]
```

作用：

```text
扰动当前 pattern，比较扰动前后的渲染图像差异，
估计 images 对 patterns 在 direction 方向上的方向导数。
```

计算方式：

[
\frac{
I(\text{patterns}+\epsilon\cdot \text{direction})
-------------------------------------------------

I(\text{patterns})
}{\epsilon}
]

伪代码：

```python
def finite_difference(patterns, direction, eps):
    self.set_patterns(patterns)
    images0 = self.render_images()

    self.set_patterns(patterns + eps * direction)
    images1 = self.render_images()

    d_images = (images1 - images0) / eps

    self.set_patterns(patterns)
    return d_images
```

---

### `finite_difference_column(patterns, k, x, eps)`

对某一张 pattern 的某一列做数值差分。

输入：

```python
patterns: Tensor[K, Wp]
k: int
x: int
eps: float
```

输出：

```python
d_images: Tensor[K, Hc, Wc]
```

作用：

```text
只扰动第 k 张 pattern 的第 x 列，
估计该列灰度变化对相机观测图的影响。
```

伪代码：

```python
def finite_difference_column(patterns, k, x, eps):
    direction = torch.zeros_like(patterns)
    direction[k, x] = 1.0
    return self.finite_difference(patterns, direction, eps)
```

---

### `finite_difference_batch(patterns, directions, eps)`

批量计算多个扰动方向的数值差分。

输入：

```python
patterns: Tensor[K, Wp]
directions: Tensor[B, K, Wp]
eps: float
```

输出：

```python
d_images: Tensor[B, K, Hc, Wc]
```

作用：

```text
一次计算多个方向上的数值差分，用于加速有限差分梯度估计。
```

---

## 可视化函数

### `save_visualization(output_dir)`

保存可视化结果。

包括：

```text
当前结构光 pattern
渲染得到的相机图像
gt_corr 热力图
depth 深度图
数值差分结果图
```

作用：

```text
用于检查渲染器是否正确，也用于实验报告展示。
```

这里可以调用 `render_depth_for_visualization()`，但不影响训练阶段性能。

---

## 自检函数

### `self_check()`

渲染器自检函数。

作用：

```text
用简单 pattern 验证相机、投影仪和真值对应关系是否合理。
```

至少包括：

```text
常量 pattern 测试
条纹 pattern 测试
gt_corr 可视化
depth 可视化
数值差分可视化
```

检查标准：

```text
常量 pattern 下不应出现结构光条纹；
条纹 pattern 下相机图中应看到条纹；
gt_corr 应该沿投影方向连续变化；
depth 应该反映场景几何深度；
数值差分结果应该只在受扰动 projector 列影响的区域附近有明显响应。
```

---

## 简化数据流

训练数据流：

```text
patterns
   ↓
render_train_batch()
   ↓
images   : K 张相机观测图
gt_corr  : 每个相机像素对应的投影仪列坐标
```

可视化数据流：

```text
patterns
   ↓
save_visualization()
   ↓
patterns 图
rendered images
gt_corr 热力图
depth 深度图
finite difference 可视化
```

有限差分数据流：

```text
patterns
direction
eps
   ↓
finite_difference(patterns, direction, eps)
   ↓
d_images = [render(patterns + eps·direction) - render(patterns)] / eps
```

---

## 总结

这个设计中，渲染器默认只输出训练必需的 `images` 和 `gt_corr`。`depth` 被移出主流程，只在自检和可视化时单独计算，从而减少训练迭代中的渲染开销。
