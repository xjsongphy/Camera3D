建议不要把每个实验场景都写成一个巨大 XML。更好的方式是：**用“场景模板 + 子文件 include + 配置文件”的方式存储**。

Mitsuba 的 XML 场景本身支持 `<include filename="..."/>`，可以把一个场景拆成多个 XML 文件；也支持通过命令行或加载参数替换变量，例如 `nested-scene-$version.xml` 这种形式，方便在不改主 XML 的情况下切换场景组件。([Mitsuba 3][1])

推荐目录结构：

```text
assets/
  meshes/
    plane.obj
    sphere.obj
    bunny.obj
  textures/
    marble.png
    wood.png
  materials/
    diffuse_white.xml
    marble.xml
    wood.xml
  cameras/
    cam_default.xml
  projectors/
    proj_default.xml
  lights/
    env_default.xml

scenes/
  base/
    integrator.xml
    sensor.xml
    emitter.xml

  sl_plane_diffuse/
    scene.xml
    geometry.xml
    material.xml
    config.yaml

  sl_marble_objects/
    scene.xml
    geometry.xml
    material.xml
    config.yaml

  sl_diffuse_objects/
    scene.xml
    geometry.xml
    material.xml
    config.yaml
```

其中每个 `scene.xml` 只负责组合模块：

```xml
<scene version="3.0.0">
    <include filename="../../base/integrator.xml"/>
    <include filename="../../base/sensor.xml"/>
    <include filename="../../base/emitter.xml"/>

    <include filename="geometry.xml"/>
    <include filename="material.xml"/>
</scene>
```

每个场景再配一个 `config.yaml`，存放你们自己的结构光实验参数：

```yaml
name: sl_marble_objects

camera:
  width: 640
  height: 480
  fx: 700
  fy: 700
  cx: 320
  cy: 240
  pose: [ ... ]

projector:
  width: 512
  height: 384
  fx: 700
  fy: 700
  cx: 256
  cy: 192
  pose: [ ... ]

patterns:
  K: 4
  Wp: 512

renderer:
  mitsuba_xml: scenes/sl_marble_objects/scene.xml
  spp: 16
  variant: cuda_ad_rgb
```

这样做的好处是：

```text
scene.xml      管 Mitsuba 能不能正确加载和渲染；
config.yaml    管你们的算法参数、相机/投影仪参数、pattern 分辨率；
assets/        存 mesh、texture、材质等可复用资源；
outputs/       存渲染图、gt_corr、训练曲线，不要和 scenes 混在一起。
```

我建议你们把 **Mitsuba XML 只当作“视觉场景描述”**，不要把所有算法参数都塞进 XML。因为 OpticalSGD 里还需要 `gt_corr`、pattern 尺寸、decoder 参数、有限差分步长、频率约束等，这些更适合放在 YAML/JSON 配置中。

最后，每个场景建议固定保存一份预计算缓存：

```text
cache/
  sl_marble_objects/
    gt_corr.pt
    albedo.pt
    geometry_buffer.pt
```

训练时直接读取这些缓存，不要每次都重新从 Mitsuba 计算几何真值。Mitsuba 3 也支持在 Python 中加载 XML 后访问和修改场景参数，官方文档提到可以通过 `traverse` 访问场景参数，适合做少量场景编辑；但大规模实验配置还是建议用外部 YAML 统一管理。([Mitsuba 3][2])

[1]: https://mitsuba.readthedocs.io/en/latest/src/key_topics/scene_format.html?utm_source=chatgpt.com "Scene XML file format - Mitsuba 3"
[2]: https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html?utm_source=chatgpt.com "Editing a scene - Mitsuba 3"
