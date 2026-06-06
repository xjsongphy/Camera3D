# Lab 2 README

Lab 2 复现的是模拟环境中的结构光 pattern 优化闭环，不涉及真实相机或投影仪采集。当前代码已经统一到 Mitsuba：

- 渲染部分：使用 Mitsuba 做前向渲染
- `autodiff` 训练：使用 Mitsuba 可微渲染
- `finite_difference` 训练：使用 Mitsuba 重复前向渲染做差分估计
- decoder：支持 `ZNCC`、`ZNCC-NN`、`ZNCC-NN + projector response`
- 梯度对比：比较 Mitsuba autodiff 和 Mitsuba finite difference

说明：
- 旧的 PyTorch 代理渲染路径已经移除。
- 旧实验结果如果来自代理渲染版本，需要基于当前代码重新运行。

## 依赖安装

项目 `pyproject.toml` 要求 Python 3.12。

CPU 环境推荐：

```bash
uv sync --python 3.12 --group lab2 --extra cpu
```

如果已经同步过环境，后续命令也建议显式指定 Python 3.12：

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
  - `both`
- `training.gradient_mode`
  - `autodiff`：Mitsuba 可微渲染
  - `finite_difference`：Mitsuba 有限差分
- `training.use_projector_response_curve`
  - 对 `zncc_nn` 生效
- `training.max_frequency_ratio`
  - 相对 Nyquist 的频率上限
- `training.frequency_weight`
  - 高频罚项权重

当前常用场景：

- `sl_marble_objects`
- `sl_diffuse_objects`

## 常用命令

### 1. 生成场景 XML

```bash
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/generate_lab2_scenes.py
```

### 2. 生成场景缓存

```bash
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/generate_scene_cache.py --scene sl_marble_objects
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/generate_scene_cache.py --all
```

### 3. 自检

```bash
uv run --python 3.12 --group lab2 --extra cpu pytest tests/lab2/test_shader_self_check.py -v
```

自检现在会检查两件事：

- Mitsuba 普通前向渲染是否正常
- Mitsuba autodiff 是否能对 pattern 产生非零梯度

### 4. 单次训练

```bash
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/run_training.py --config configs/lab2/sl_marble_objects.yaml
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/run_training.py --config configs/lab2/sl_diffuse_objects.yaml
```

例如强制指定 Mitsuba autodiff：

```bash
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/run_training.py --config configs/lab2/default.yaml --scene sl_marble_objects --decoder zncc_nn_response --gradient-mode autodiff --iterations 500
```

### 5. 梯度对比

```bash
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/compare_gradients.py --config configs/lab2/sl_marble_objects.yaml --num-samples 4
```

### 6. 一键运行全部任务

```bash
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/run_assignment_experiments.py
```

当前 `configs/lab2/assignment_plan.yaml` 默认会：

- 运行 `tests/lab2/test_shader_self_check.py`
- 运行一次梯度对比
- 运行大理石和漫反射场景的全部训练实验
- 自动从已有 checkpoint 恢复未完成任务

如果中途中断，可以继续：

```bash
uv run --python 3.12 --group lab2 --extra cpu python scripts/lab2/run_assignment_experiments.py --run-root outputs/lab2/assignment_runs/<已有目录>
```

## 输出目录

- 训练输出：`outputs/lab2/runs/`
- 梯度对比输出：`outputs/lab2/gradient_comparison/`
- 一键运行汇总：`outputs/lab2/assignment_runs/`

主要文件：

- `config.json`
- `training_log.csv`
- `timing.csv`
- `pattern_history.pt`
- `decoder_parameter_history.pt`
- `patterns_final.png`
- `spectrum_final.png`
- `loss_curve.png`
- `depth_gt.png`
- `gt_corr.png`
- `rendered_final.png`
- `correspondence_final.png`
- `depth_error_map.png`
- `pattern_evolution.gif`
- `spectrum_evolution.gif`
- `checkpoint_final/checkpoint.pt`

## 运行前建议

- 先单独跑一次 `pytest tests/lab2/test_shader_self_check.py -v`
- 再跑一次 `compare_gradients.py`
- 最后再启动 `run_assignment_experiments.py`

原因很简单：现在 `autodiff` 已经切到 Mitsuba 可微渲染，正确性更好，但算力开销也更高。先确认环境和梯度链路正常，再开始全量任务更稳妥。
