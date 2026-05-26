# Lab 2: 结构光Pattern自动优化

基于Optical SGD的结构光投影图案优化闭环实验，复现Chen et al., CVPR 2020论文核心思想。

## 实验内容

- **合成渲染器**：Mitsuba渲染的投影仪-相机-场景系统
- **OpticalSGD优化器**：有限差分与自动微分两种梯度计算方式
- **Decoder对比**：ZNCC vs ZNCC-NN
- **材质鲁棒性**：大理石、木头、玻璃等材质测试

## 快速开始

```bash
# 同步依赖（需要Python 3.12）
uv sync --group lab2

# 生成场景（环境自动设置）
uv run python scripts/lab2/generate_lab2_scenes.py

# 运行训练
uv run python scripts/lab2/run_training.py --scene sl_plane_diffuse --decoder zncc --penalty l1

# 梯度对比
uv run python scripts/lab2/compare_gradients.py --scene sl_plane_diffuse

# 渲染器自检
uv run pytest tests/lab2/test_shader_self_check.py -v
```

## 环境依赖

### Python版本
- **必须使用 Python 3.12**

### Python依赖
```bash
uv sync --group lab2
```

包含：PyTorch, Mitsuba 3.x, Drjit, OpenCV, Matplotlib等

### macOS额外依赖

**重要**：Lab2使用统一的环境设置模块，环境变量会自动配置！

运行任何lab2脚本时会自动：
- 检测并设置 `DRJIT_LIBLLVM_PATH`
- 启用 `OPENCV_IO_ENABLE_OPENEXR=1`

**仅需手动安装LLVM**：
```bash
brew install llvm
```

然后直接运行脚本即可，无需手动设置环境变量。

### 验证安装
```bash
# 检查依赖状态
uv run python -c "from lab2 import env; env.print_dependency_status()"

# 运行自检测试
uv run pytest tests/lab2/test_shader_self_check.py -v
```

## 作业文档

- [作业要求](lab2.md)
- [论文笔记](paper.md)
- [讲义](lab2-讲义.md)
- [原始论文](Auto-Tuning Structured Light by Optical Stochastic Gradient Descent.pdf)

## 输出结构

```
results/lab2/
├─ 20250526_183000_sl_plane_diffuse_zncc_l1/
│  ├─ config.json           # 训练配置
│  ├─ scene_info.txt        # 场景信息
│  ├─ patterns_final.png    # 最终图案
│  ├─ training_log.csv      # 训练日志
│  └─ checkpoints/          # 检查点
└─ gradient_comparison/
   └─ 20250526_183000_gradient_comparison/
      ├─ config.json
      └─ gradient_comparison.png
```

## 常见问题

### Q: 测试跳过并提示Mitsuba不可用？
A: 确保已安装LLVM并设置环境变量：
```bash
export DRJIT_LIBLLVM_PATH=$(brew --prefix llvm)/lib
```

### Q: OpenEXR codec错误？
A: 设置环境变量：
```bash
export OPENCV_IO_ENABLE_OPENEXR=1
```

### Q: Python版本不匹配？
A: 确保使用Python 3.12：
```bash
uv python install 3.12
uv sync --group lab2
```
