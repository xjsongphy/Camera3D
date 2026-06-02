# Lab 2 README

Lab 2 复现的是**模拟环境中的**结构光 pattern 优化闭环，不涉及真实相机/投影仪采集。整体流程包括场景渲染、自监督优化、解码器对比、梯度对比和结果可视化。

## 当前实现范围

- 渲染部分：使用 Mitsuba 做模拟渲染
- 训练部分：使用可微的模拟代理链路优化 pattern
- 梯度对比：比较有限差分和自动微分
- decoder：支持 `ZNCC`、`ZNCC-NN`，以及可选的可学习投影仪响应曲线
- 材质场景：包含训练板、大理石、木头/玻璃等场景

## 依赖安装

项目 `pyproject.toml` 要求 Python 3.12。当前 Lab 2 依赖除了 `lab2` group 之外，还需要安装 `torch`。

CPU 环境推荐：

```bash
uv sync --python 3.12 --group lab2 --extra cpu
```

如果已经同步过环境，后续运行命令建议也显式指定 Python 3.12：

```bash
uv run --python 3.12 --group lab2 --extra cpu python -V
```

## 场景与配置

- 场景定义：`src/lab2/scenes/`
- 场景入口：`src/lab2/scene_genertor.py` 中的 `SCENE_PRESETS`
- 训练配置：`configs/lab2/*.yaml`

常用训练参数：

- `decoder`
  - `zncc`
  - `zncc_nn`
  - `zncc_nn_response`
  - `both`：按顺序运行 `zncc`、`zncc_nn`、`zncc_nn_response`
- `training.use_projector_response_curve`
  - 对 `zncc_nn` 生效；为 `true` 时启用 32 段单调分段线性投影仪响应曲线
- `training.max_frequency_ratio`
  - 相对 Nyquist 的频率上限；`0.5` 表示限制到 `1/2 Nyquist`
- `training.frequency_weight`
  - 高频惩罚权重；同时仍会在每步更新后做硬低通

当前常用场景：

- `sl_training_board`
- `sl_vases`
- `sl_marble_objects`
- `sl_wood_glass`
- `sl_statue`

## 常用命令

### 1. 生成场景 XML

```bash
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/generate_lab2_scenes.py
```

### 2. 生成场景缓存

```bash
# 单个场景
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/generate_scene_cache.py --scene sl_marble_objects

# 全部场景
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/generate_scene_cache.py --all
```

### 3. 渲染器自检

```bash
uv run --python 3.12 --group lab2 --extra cpu python tests/lab2/test_shader_self_check.py
# 或
uv run --python 3.12 --group lab2 --extra cpu pytest tests/lab2/test_shader_self_check.py -v
```

### 4. 训练

```bash
# 使用场景配置
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/run_training.py --config configs/lab2/sl_marble_objects.yaml

# 用默认配置并覆盖参数
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/run_training.py --config configs/lab2/default.yaml --scene sl_wood_glass --decoder zncc_nn_response --iterations 500
```

### 5. 梯度对比

```bash
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/compare_gradients.py --config configs/lab2/sl_marble_objects.yaml --num-samples 4
```

如果只想看参数说明：

```bash
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/compare_gradients.py --help
```

## 输出目录

训练输出位于 `outputs/lab2/runs/`，每次 run 会生成一个时间戳目录，例如：

```text
outputs/lab2/runs/20260531_225443_sl_marble_objects_zncc_nn_l1/
```

梯度对比输出位于 `outputs/lab2/gradient_comparison/`。当 `decoder=both` 时，会同时生成三组 run，并在 comparison 目录下汇总。

现在默认不会再周期性保存大量 `patterns_iter_*.png` / `spectrum_iter_*.png`。训练过程改为：

- 完整记录每一轮的 pattern `c`
- 完整记录每一轮 decoder 网络参数
- 在训练结束后一次性生成完整 GIF

## 主要输出文件说明

### 配置与日志

- `config.json`
  - 本次 run 的场景、训练参数、渲染参数快照
- `training_log.csv`
  - 每轮训练日志，包含 `iteration`、`loss`、`elapsed_sec`
- `timing.csv`
  - 各阶段耗时统计
- `logs/*.log`
  - 终端完整日志

### Pattern 与参数历史

- `pattern_history.pt`
  - 完整 pattern 历史，包含：
  - `iterations`：记录的迭代号，含初始状态 `0`
  - `patterns`：形状 `[T, K, Wp]`
  - `loss_iterations` / `loss_history`：与训练轮次对齐的 loss 序列
- `pattern_history_summary.csv`
  - 每轮每个 pattern 的统计量：均值、标准差、最小值、最大值、L2 范数
- `decoder_parameter_history.pt`
  - 完整 decoder 参数历史；`ZNCC` 无可学习参数时不会生成
- `decoder_parameter_history_summary.csv`
  - 每轮每个网络参数张量的统计量

### 可视化图片

- `patterns_final.png`
  - 最终 pattern 可视化
- `spectrum_final.png`
  - 最终 pattern 频谱图
- `projector_response_curve.png`
  - 仅在启用可学习投影仪响应曲线时生成，展示最终学习到的响应曲线
- `loss_curve.png`
  - 全部训练迭代的 loss 曲线
- `depth_gt.png`
  - 场景真值深度图与深度分布直方图
- `gt_corr.png`
  - 真值 projector correspondence 图
- `rendered_final.png`
  - 用最终 pattern 渲染得到的相机观测图像
- `correspondence_final.png`
  - 最终预测 correspondence 与 GT 对照图
- `depth_error_map.png`
  - 最终 correspondence 误差热力图

### GIF 动画

- `pattern_evolution.gif`
  - 完整迭代过程的 pattern 演化动画
- `spectrum_evolution.gif`
  - 完整迭代过程的频谱演化动画

### Checkpoint

- `checkpoints/iter_*/checkpoint.pt`
  - 周期性 checkpoint，用于中断恢复
- `checkpoint_final/checkpoint.pt`
  - 最终 checkpoint

checkpoint 中除了当前模型状态，也会保存当前已累计的 pattern 历史和 decoder 参数历史，便于 `resume` 后继续追加。

## 自检输出

`outputs/lab2/self_check/<scene>/` 下的图片用于验证渲染器：

- `pattern.png`：测试条纹 pattern
- `render.png`：投影条纹后的渲染结果
- `normal_render.png`：近似白光照明下的材质可视化
- `depth_map.png`：深度图
- `gt_corr.png`：真值 correspondence
