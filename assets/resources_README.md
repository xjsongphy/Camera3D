# Assets Resources

本文件说明 `assets/` 目录下供 Lab2 使用的几何、材质、场景模板和纹理资源。

## 目录结构

```text
assets/
├── geometry/
│   ├── materials/     # 材质 XML 定义（基于 Mitsuba 3）
│   └── obj/           # 下载的 .obj 几何模型文件
├── scenes/
│   ├── simple/        # 简单场景（内联几何体）
│   └── complex/       # 复杂场景（使用 .obj 文件）
└── textures/          # 纹理贴图文件（可选）
```

## 材质文件说明

已包含的材质定义（基于 Mitsuba 3 官方 BSDF 测试文件）：

- `diffuse_white.xml` - 白色漫反射材质（基准）
- `marble.xml` - 大理石材质（粗糙介质）
- `glass_clear.xml` - 清晰玻璃材质
- `glass_rough.xml` - 磨砂玻璃材质
- `wood.xml` - 木头材质（漫反射）
- `conductor_metal.xml` - 金属材质

## .obj 模型下载指南

### 推荐资源网站

1. **Sketchfab** (最推荐)
   - 网址: https://sketchfab.com
   - 搜索关键词: "marble statue", "sculpture", "cultural heritage"
   - 许可证: 选择 CC0 或 CC-BY 模型
   - 下载格式: OBJ

2. **Free3D**
   - 网址: https://free3d.com/3d-models/obj
   - 免费 .obj 模型库

3. **Smithsonian 3D Digitization**
   - CC0 文化遗产模型
   - 网址: https://3d.si.edu

### 具体推荐模型（Sketchfab）

1. **女性大理石雕塑**
   - URL: https://sketchfab.com/3d-models/female-marble-statue-818a96836b5148eba0b394c12c171830
   - 许可: CC BY-NC-SA
   - 三角面数: 66.5k
   - 下载后重命名为: `assets/geometry/obj/marble_statue.obj`

2. **CC0 雕塑集合**
   - URL: https://sketchfab.com/HannahT25/collections/cc0-statues-913ddbd5777b4430acfc79ed9d500275
   - 公共领域，无需署名

3. **克利夫兰艺术博物馆 CC0 藏品**
   - URL: https://sketchfab.com/clevelandart/collections/cma-cc0-objects-906ce25f39ba487ea11aa024f8123478
   - 高质量扫描文物

### 下载步骤

1. 在 Sketchfab 上找到合适的模型
2. 确认许可证（优先选择 CC0）
3. 点击 "Download 3D Model"
4. 选择 "OBJ" 格式
5. 解压下载的文件
6. 将 .obj 文件复制到 `assets/geometry/obj/` 目录
7. 将纹理文件（如有）复制到 `assets/textures/` 目录

### 使用下载的模型

参考 `assets/scenes/complex/obj_loader_template.xml` 或 `marble_statue_template.xml`，修改文件名为实际下载的模型。

## 材质来源说明

所有材质 XML 定义基于：
- Mitsuba 3 渲染器官方文档: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html
- Mitsuba 官方测试文件: https://github.com/mitsuba-renderer/mitsuba/blob/master/data/tests/test_bsdf.xml

## 实验要求

根据 Lab2 讲义，必须测试的材质：
- ✅ 大理石材质（marble.xml）
- ✅ 木头材质（wood.xml）
- ✅ 玻璃材质（glass_clear.xml, glass_rough.xml）
- ✅ 漫反射基准（diffuse_white.xml）

简单场景使用内联几何体（sphere, cube）即可满足基本实验要求。
复杂场景需要下载 .obj 模型以获得更真实的深度变化和几何细节。
