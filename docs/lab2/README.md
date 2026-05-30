# Lab 2 README

Lab2 复现结构光 Pattern 优化闭环，包括场景渲染、自监督优化、解码器对比与结果可视化。

## 依赖安装

```bash
# Lab2 依赖（推荐 Python 3.12）
uv sync --group lab2
```

## 场景与配置

- 场景定义：`src/lab2/scenes/`
- 场景入口：`src/lab2/scene_genertor.py`（`SCENE_PRESETS`）
- 训练配置：`configs/lab2/*.yaml`

当前常用场景示例：

- `sl_training_board`
- `sl_vases`
- `sl_marble_objects`
- `sl_wood_glass`
- `sl_statue`

## 常用命令

### 1) 生成场景 XML 资源

```bash
uv run python scripts/lab2/generate_lab2_scenes.py
```

### 2) 生成场景缓存（depth / gt_corr）

```bash
# 单个场景
uv run python scripts/lab2/generate_scene_cache.py --scene sl_marble_objects

# 全部场景
uv run python scripts/lab2/generate_scene_cache.py --all
```

### 3) 渲染器自检

```bash
uv run python tests/lab2/test_shader_self_check.py
# 或
uv run pytest tests/lab2/test_shader_self_check.py -v
```

### 4) 训练

```bash
# 使用场景配置
uv run python scripts/lab2/run_training.py --config configs/lab2/sl_marble_objects.yaml

# 用默认配置并覆盖参数
uv run python scripts/lab2/run_training.py --config configs/lab2/default.yaml --scene sl_wood_glass --decoder zncc_nn --iterations 500
```

### 5) 梯度对比（若使用）

```bash
uv run python scripts/lab2/compare_gradients.py --help
```

## 输出目录

训练输出位于：`results/lab2/runs/`

典型 run 目录内容：

- `config.json`
- `training_log.csv`
- `patterns_initial.pt`
- `patterns_final.pt`
- `patterns_final.png`
- `loss_curve.png`
- `metrics_final.json`
- `rendered_final.png`

## 常见问题

- `jitc_llvm_init(): LLVM API initialization failed ..`：通常是后端初始化警告，若训练/测试继续且通过，可先忽略。
- Mitsuba 不可用：先确认 `uv sync --group lab2` 完整执行，且 Python 版本为 3.12。
