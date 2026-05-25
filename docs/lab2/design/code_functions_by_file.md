# Lab2 代码设计说明（按文件）

本文档按文件解释 Lab2 当前实现中每个函数/方法的作用、输入输出与调用位置。

## 文件 1: `src/lab2/shader.py`

### 数据类

#### `CameraConfig`
- 作用: 保存相机参数（分辨率、内参、外参）。
- 字段: `width, height, fx, fy, cx, cy, R, t`。

#### `ProjectorConfig`
- 作用: 保存投影仪参数（分辨率、内参、外参）。
- 字段: `width, height, fx, fy, cx, cy, R, t`。

#### `LightConfig`
- 作用: 保存全局光照配置。
- 字段: `ambient`。

### 主类 `StructuredLightRenderer`

#### `__init__(device=None, dtype=torch.float32, spp=64, mi_variant="cuda_ad_rgb")`
- 作用: 初始化渲染器状态（默认优先 GPU）。

#### `_require_mitsuba()`
- 作用: 延迟导入 Mitsuba 并设置 variant；失败回退 `llvm_ad_rgb`。

#### `_to_tensor(x, shape=None)`
- 作用: 统一转为当前 device/dtype tensor，支持形状校验。

#### `_parse_intrinsics_extrinsics(config, is_camera)`
- 作用: 解析设备配置为 `CameraConfig`/`ProjectorConfig`。

#### `set_camera(camera_config)`
- 作用: 设置相机参数并清空 `gt_corr` 缓存。

#### `set_projector(projector_config)`
- 作用: 设置投影仪参数并清空 `gt_corr` 缓存。

#### `set_lights(light_config)`
- 作用: 设置环境光参数。

#### `set_patterns(patterns)`
- 作用: 写入当前结构光 pattern。
- 输入: `[K, Wp]`。

#### `update_patterns(patterns)`
- 作用: pattern 更新别名接口。

#### `load_scene(scene_path=None, cache_path=None)`
- 作用: 加载/初始化场景。
- 支持:
  - `.xml` 作为 Mitsuba scene（必须配套 `.npz` cache，保证图像与监督几何一致）。
  - `.npz` 读取 `depth/gt_corr` 缓存。
  - `None` 使用默认深度。

#### `_pixel_rays_world()`
- 作用: 根据相机参数计算每像素世界坐标系光线。
- 输出: `rays_o, rays_d`，形状均为 `[H, W, 3]`。

#### `compute_gt_corr()`
- 作用: 计算相机像素到投影仪列坐标的真值图。
- 输出: `gt_corr: [H, W]`（无效像素为 `NaN`，避免误监督到第 0 列）。

#### `gt_corr`（property）
- 作用: 对外暴露懒计算的真值对应图。

#### `_look_at_from_rt(mi, R, t)`
- 作用: 将 `R,t` 外参转换为 Mitsuba `look_at` 变换。

#### `_pattern_to_image(pattern_1d, hp)`
- 作用: 将一维 pattern 扩展为二维 RGB 纹理，供 projector emitter 使用。

#### `_make_default_scene_dict(mi, pattern_path)`
- 作用: 构建默认 Mitsuba scene（相机+环境光+投影仪+目标平面）。

#### `_make_scene_with_pattern(pattern_path)`
- 作用: 根据 pattern 构建场景（优先外部 xml，否则默认 scene dict）。

#### `_render_single_pattern(pattern_1d)`
- 作用: 渲染一张 pattern 对应的相机图。
- 输出: `image: [H, W]`。

#### `_prepare_patterns(patterns=None)`
- 作用: 统一处理 `scene loaded` 校验、pattern 来源选择与 shape/device/dtype 规范化。
- 目的: 消除 `render_images / render_train_batch / save_visualization` 的重复前置逻辑。

#### `render_images_and_gt(patterns=None)`
- 作用: 统一观测入口，一次调用同时得到 `images` 和 `gt_corr`。
- 输出: `{ \"images\": [K,H,W], \"gt_corr\": [H,W] }`。
- 说明: `render_images`、`render_train_batch`、`save_visualization` 均复用该接口，避免重复准备流程。

#### `render_images(patterns=None)`
- 作用: 渲染 K 张 pattern 的相机观测。
- 输入: `[K, Wp]`。
- 输出: `[K, H, W]`。

#### `render_images_batch(patterns_batch)`
- 作用: 渲染 B 组 pattern（批量接口）。
- 输入: `[B, K, Wp]`。
- 输出: `[B, K, H, W]`。

#### `render_train_batch()`
- 作用: 产出训练闭环所需最小数据。
- 输出: `{ "images", "gt_corr" }`。
- 说明: 内部直接复用 `render_images_and_gt()`。

#### `finite_difference(patterns, direction, eps)`
- 作用: 单方向有限差分。
- 输出: `[K, H, W]`。

#### `finite_difference_column(patterns, k, x, eps)`
- 作用: 单列扰动有限差分。

#### `finite_difference_batch(patterns, directions, eps)`
- 作用: 批量方向有限差分。
- 输入: `directions [B, K, Wp]`。
- 输出: `[B, K, H, W]`。

#### `finite_difference_batch_chunked(patterns, directions, eps, chunk_size=64)`
- 作用: 大 batch 分块有限差分，降低峰值内存。

#### `render_depth_for_visualization()`
- 作用: 返回深度可视化缓存。
- 输出: `[H, W]`。

#### `save_visualization(output_dir)`
- 作用: 根据当前 `self.patterns` 与渲染结果输出自检图。
- 输出文件:
  - `constant_pattern.png`
  - `stripe_pattern.png`
  - `constant_pattern_render.png`
  - `stripe_pattern_render.png`
  - `gt_corr_vis.png`

---

## 文件 2: `src/lab2/__init__.py`

#### 模块导出
- 作用: 导出 `StructuredLightRenderer` 供外部直接使用。

---

## 文件 3: `tests/lab2/test_shader_self_check.py`

### 类 `TestStructuredLightRendererSelfCheck`

#### `test_self_check_outputs()`
- 作用: 在 test 侧自行构造“常量 pattern + 条纹 pattern”完成自检流程。
- 调用接口:
  - `set_camera / set_projector / load_scene / set_patterns`
  - `render_images / gt_corr / render_depth_for_visualization / save_visualization`
- 检查项:
  - 张量形状和数值范围
  - 关键自检图片是否生成

---

## 调用关系（简版）

- 训练路径: `set_* -> load_scene -> set_patterns -> render_train_batch`
- 渲染路径: `render_images_and_gt -> _prepare_patterns -> _render_single_pattern -> _make_scene_with_pattern -> _require_mitsuba`
- 真值路径: `gt_corr(property) -> compute_gt_corr -> _pixel_rays_world`
- 自检路径（在 test 中实现）: `set_patterns -> render_images_and_gt + save_visualization`
- 有限差分路径: `finite_difference* -> render_images / render_images_batch`
