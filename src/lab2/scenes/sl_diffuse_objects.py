"""
Diffuse objects scene - simple matte baseline for material robustness tests.

Uses only diffuse BSDFs to provide a reference scene without specular or refractive effects.
"""


def define_scene(mi):
    """
    Define diffuse baseline scene geometry and materials.

    Args:
        mi: Mitsuba module instance

    Returns:
        dict: Scene objects (geometry + materials)
    """
    _horizontal_rot = mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=-90)

    return {
        "ground": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0.0, -0.45, 1.6))
            @ _horizontal_rot
            @ mi.ScalarTransform4f.scale((1.2, 0.8, 1.0)),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.70, 0.70, 0.70]}},
        },
        "sphere": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate((-0.35, -0.15, 1.9))
            @ mi.ScalarTransform4f.scale((0.24, 0.24, 0.24)),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.78, 0.78, 0.80]}},
        },
        "cube": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate((0.35, -0.2, 2.05))
            @ mi.ScalarTransform4f.scale((0.24, 0.24, 0.24)),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.62, 0.64, 0.68]}},
        },
    }
