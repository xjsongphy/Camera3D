# Lab 3: 多表示三维场景重建

本目录对应作业三。代码入口在 `src/lab3`，运行脚本在 `scripts/lab3`，默认配置在 `configs/lab3/default.json`。

## 设计

- `lab3.extract`: 从指定目录递归收集图片和视频；图片复制到统一数据目录，视频用 FFmpeg 抽帧；写出 `manifest.csv`、`train.txt`、`test.txt`。
- `lab3.reconstruction.sfm`: 使用 COLMAP CLI 完成 feature extraction、matching、mapper 和模型文本导出。
- `lab3.reconstruction.dgs`: 调用 GraphDeco 官方 `gaussian-splatting/train.py`，以准备好的数据目录作为输入。
- `lab3.reconstruction.nerf`: 调用 nerfstudio `ns-process-data images` 和 `ns-train nerfacto`。
- `lab3.visualization`: 只保留后处理查看入口，不在重建流程中自动启动交互式可视化。

Lab 3 不 import Lab 1 代码；COLMAP 命令封装、抽帧和路径组织在 `src/lab3` 内独立实现。

## 输出结构

一次运行会写入：

```text
outputs/lab3/<scene>_<timestamp>/
├─ configs/
│  ├─ run_config.json
│  └─ prepared_dataset.json
├─ prepared/
│  ├─ images/
│  ├─ manifest.csv
│  ├─ train.txt
│  └─ test.txt
├─ results/
│  ├─ sfm/
│  ├─ 3dgs/
│  └─ nerf/
└─ timings.json
```

## 运行

先做命令检查：

```powershell
uv run lab3 --config configs/lab3/default.json --methods sfm --timestamp dryrun --dry-run
```

使用 Lab 1 视频做 3DGS 测试时，需要先准备官方 3DGS 仓库及环境，然后传入路径：

```powershell
uv run lab3 `
  --input-dir docs/lab1/assets/videos `
  --scene-name lab1_sample `
  --methods 3dgs `
  --fps 2 `
  --dgs-repo D:\path\to\gaussian-splatting `
  --dgs-iterations 7000
```

完整三方法运行：

```powershell
uv run lab3 `
  --input-dir path\to\captured_scene `
  --scene-name my_scene `
  --methods sfm 3dgs nerf `
  --fps 2 `
  --dgs-repo D:\path\to\gaussian-splatting
```

依赖外部工具：

- FFmpeg: 视频抽帧。
- COLMAP: SfM 和 3DGS/NeRF 常用位姿来源。
- nerfstudio: `ns-process-data` 和 `ns-train nerfacto`。
- GraphDeco gaussian-splatting: 官方 3DGS 训练脚本。

