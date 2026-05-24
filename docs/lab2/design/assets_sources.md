# Lab2 素材来源建议（可直接用）

目标：给 `sl_marble_objects` / `sl_wood_glass` 场景准备可复用、授权清晰的素材。

## 推荐来源

1. Poly Haven（CC0）
- 主页: https://polyhaven.com/
- 说明: 纹理/HDRI/模型，CC0。
- 示例木纹素材: https://polyhaven.com/a/medieval_wood

2. ambientCG（CC0）
- 主页: https://ambientcg.com/
- 说明: 大量 PBR 纹理与少量模型/HDRI，CC0。
- 示例大理石素材: https://ambientcg.com/view?id=Marble025

3. Stanford Bunny（研究常用模型）
- 官方页面: https://graphics.stanford.edu/software/scanview/models/bunny.html
- 用途: 几何复杂度测试、自检可视化。

4. Mitsuba 场景格式文档（include 模块化）
- https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html
- 用途: `scene.xml + include` 拆分场景结构。

## 建议放置路径

```text
assets/
  meshes/
  textures/
  hdri/
  materials/
```

## 最小素材清单

1. `textures/marble_*`（basecolor/roughness/normal）
2. `textures/wood_*`（basecolor/roughness/normal）
3. `meshes/bunny.obj`（或 sphere/cube 先替代）
4. `hdri/studio_*.hdr`（用于环境光对比实验）

## 下载后建议

1. 统一贴图分辨率到 `2K` 或 `4K`，避免训练时 IO 过重。
2. 保留原始授权说明文件到 `assets/LICENSES/`。
3. 在 `docs/lab2/report` 中记录素材来源 URL 和下载日期。
