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

# 生成场景
uv run python scripts/lab2/generate_lab2_scenes.py

# 运行训练
uv run python scripts/lab2/run_training.py --scene sl_plane_diffuse --decoder zncc --penalty l1

# 梯度对比
uv run python scripts/lab2/compare_gradients.py --scene sl_plane_diffuse

# 渲染器自检
uv run pytest tests/lab2/test_shader_self_check.py -v
```

## 项目结构

```
docs/lab2/
├─ images/                  # 论文图片
├─ design/                  # 设计文档
├─ lab2.md                  # 作业要求
├─ paper.md                 # 论文笔记
├─ lab2-讲义.md             # 讲义
└─ README.md                # 本文件
```

## 核心脚本

| 脚本 | 功能 |
|------|------|
| `generate_lab2_scenes.py` | 生成不同材质场景 |
| `run_training.py` | 运行Optical SGD训练 |
| `compare_gradients.py` | 梯度方式对比 |

## 环境依赖

### Python版本
- **必须使用 Python 3.12**

### Python依赖
```bash
uv sync --group lab2
```

包含：PyTorch, Mitsuba 3.x, Drjit, OpenCV, Matplotlib等

### macOS额外依赖

Mitsuba在macOS上需要LLVM：

```bash
# 安装LLVM
brew install llvm

# 设置环境变量（添加到~/.zshrc）
# 注意：DRJIT_LIBLLVM_PATH需要指向libLLVM.dylib文件的完整路径
export DRJIT_LIBLLVM_PATH=$(brew --prefix llvm)/lib/libLLVM.dylib
export OPENCV_IO_ENABLE_OPENEXR=1

# 重新加载配置
source ~/.zshrc
```

### 验证安装
```bash
# 检查Mitsuba
uv run python -c "import mitsuba; print(mitsuba.__version__)"

# 检查PyTorch
uv run python -c "import torch; print(torch.__version__)"

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
