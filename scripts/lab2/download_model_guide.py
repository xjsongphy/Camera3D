#!/usr/bin/env python3
"""
模型下载指南 - Lab2 结构光实验

这个脚本提供了从 Sketchfab 和其他来源下载 .obj 模型的指导。
"""

import click

@click.group()
def cli():
    """模型下载和准备工具"""
    pass

@cli.command()
def list_recommended():
    """列出推荐的 .obj 模型下载来源"""
    click.echo("=" * 60)
    click.echo("推荐的大理石/复杂几何体模型下载来源")
    click.echo("=" * 60)

    models = [
        {
            "name": "女性大理石雕塑",
            "url": "https://sketchfab.com/3d-models/female-marble-statue-818a96836b5148eba0b394c12c171830",
            "license": "CC BY-NC-SA",
            "faces": "66.5k triangles",
            "target": "marble_statue.obj"
        },
        {
            "name": "CC0 雕塑集合",
            "url": "https://sketchfab.com/HannahT25/collections/cc0-statues",
            "license": "CC0 (无需署名)",
            "faces": "各种",
            "target": "选择任意一个雕塑"
        },
        {
            "name": "阿波罗·贝尔维德雷",
            "url": "https://sketchfab.com/3d-models/apollo-belvedere-fe5c0cffdc2a4f3985872212c692af0c",
            "license": "CC0",
            "faces": "高精度扫描",
            "target": "apollo_belvedere.obj"
        },
        {
            "name": "Smithsonian 3D (文化遗产)",
            "url": "https://3d.si.edu",
            "license": "CC0",
            "faces": "各种",
            "target": "搜索 marble 或 statue"
        },
    ]

    for i, model in enumerate(models, 1):
        click.echo(f"\n{i}. {model['name']}")
        click.echo(f"   URL: {model['url']}")
        click.echo(f"   许可: {model['license']}")
        click.echo(f"   面数: {model['faces']}")
        click.echo(f"   建议: 下载后重命名为 {model['target']}")

    click.echo("\n" + "=" * 60)
    click.echo("下载步骤:")
    click.echo("1. 访问上述网址")
    click.echo("2. 确认模型可下载且有合适的许可证")
    click.echo("3. 点击 Download 3D Model")
    click.echo("4. 选择 OBJ 格式下载")
    click.echo("5. 解压并将 .obj 文件放到 resources/geometry/obj/")
    click.echo("6. 如有纹理，放到 resources/textures/")
    click.echo("=" * 60)

@cli.command()
@click.argument('model_name', type=str)
def setup_scene(model_name):
    """为下载的模型创建场景文件"""
    click.echo(f"\n为 {model_name} 创建场景模板...")

    template = f"""<!-- Auto-generated scene for {model_name} -->
<scene version="3.0.0">
    <integrator type="path"/>

    <sensor type="perspective">
        <float name="fov" value="50"/>
        <transform name="to_world">
            <lookat origin="0, 0.5, 2" target="0, 0, 0" up="0, 1, 0"/>
        </transform>
        <film type="hdrfilm">
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
        </film>
        <sampler type="independent">
            <integer name="sample_count" value="64"/>
        </sampler>
    </sensor>

    <emitter type="constant">
        <rgb name="radiance" value="0.5 0.5 0.5"/>
    </emitter>

    <shape type="obj">
        <string name="filename" value="resources/geometry/obj/{model_name}"/>
        <transform name="to_world">
            <!-- 调整缩放和旋转 -->
            <scale value="0.01"/>
        </transform>
        <bsdf type="roughdielectric">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.5"/>
            <float name="extIOR" value="1.0"/>
        </bsdf>
    </shape>
</scene>
"""

    output_path = f"resources/scenes/complex/{model_name.replace('.obj', '')}.xml"
    with open(output_path, 'w') as f:
        f.write(template)

    click.echo(f"✅ 场景文件已创建: {output_path}")
    click.echo("\n接下来:")
    click.echo("1. 将模型文件放到 resources/geometry/obj/")
    click.echo(f"2. 确认文件名与场景中一致: {model_name}")
    click.echo("3. 使用 mitsuba 渲染测试场景")

@cli.command()
def check_resources():
    """检查 resources 目录状态"""
    from pathlib import Path

    click.echo("\n检查 resources 目录状态:")
    click.echo("-" * 40)

    paths = {
        "材质目录": Path("resources/geometry/materials"),
        "OBJ 模型目录": Path("resources/geometry/obj"),
        "简单场景": Path("resources/scenes/simple"),
        "复杂场景": Path("resources/scenes/complex"),
        "纹理目录": Path("resources/textures"),
    }

    for name, path in paths.items():
        if path.exists():
            count = len(list(path.iterdir()))
            click.echo(f"✅ {name}: {count} 个文件")
        else:
            click.echo(f"❌ {name}: 不存在")

    # 检查 .obj 文件
    obj_dir = Path("resources/geometry/obj")
    if obj_dir.exists():
        obj_files = list(obj_dir.glob("*.obj"))
        if obj_files:
            click.echo(f"\n已下载的 .obj 模型:")
            for obj in obj_files:
                size = obj.stat().st_size / 1024
                click.echo(f"  - {obj.name} ({size:.1f} KB)")
        else:
            click.echo("\n⚠️  尚未下载任何 .obj 模型")
            click.echo("   运行 'python scripts/lab2/download_model_guide.py list-recommended' 查看推荐")

if __name__ == '__main__':
    cli()
