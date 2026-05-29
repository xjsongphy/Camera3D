"""
Wood + glass scene - mixed material scene with glass sphere and wood cube.

Demonstrates different material properties: transparent dielectric vs diffuse wood.
"""

def define_scene(mi):
    """
    Define wood + glass scene geometry and materials.

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
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [0.56, 0.42, 0.30]},
            },
        },
        "glass_sphere": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate((-0.35, -0.15, 1.9)) @ mi.ScalarTransform4f.scale((0.24, 0.24, 0.24)),
            "bsdf": {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0},
        },
        "wood_cube": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate((0.35, -0.2, 2.05)) @ mi.ScalarTransform4f.scale((0.24, 0.24, 0.24)),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [0.58, 0.40, 0.25]},
            },
        },
    }
