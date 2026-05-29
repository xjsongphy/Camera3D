"""
Marble objects scene - marble-like mixed objects on a table.

Features a ground plane with sphere and cube objects made of marble-like materials.
"""

def define_scene(mi):
    """
    Define marble objects scene geometry and materials.

    Args:
        mi: Mitsuba module instance

    Returns:
        dict: Scene objects (geometry + materials)
    """
    # Rotation for horizontal ground/table
    _horizontal_rot = mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=-90)

    return {
        "ground": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0.0, -0.45, 1.6)) @ _horizontal_rot @ mi.ScalarTransform4f.scale((1.2, 0.8, 1.0)),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.76, 0.76, 0.76]}},
        },
        "sphere": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate((-0.35, -0.15, 1.9)) @ mi.ScalarTransform4f.scale((0.24, 0.24, 0.24)),
            "bsdf": {
                "type": "roughplastic",
                "alpha": 0.14,
                "diffuse_reflectance": {"type": "rgb", "value": [0.92, 0.90, 0.88]},
            },
        },
        "cube": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate((0.35, -0.2, 2.05)) @ mi.ScalarTransform4f.scale((0.24, 0.24, 0.24)),
            "bsdf": {
                "type": "roughplastic",
                "alpha": 0.19,
                "diffuse_reflectance": {"type": "rgb", "value": [0.88, 0.86, 0.84]},
            },
        },
    }
