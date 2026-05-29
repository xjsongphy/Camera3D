"""
Vases scene - inspired by paper Figure 9: multiple vessels + organic foreground detail.

Features three vases (outer+inner cylinders) with flowers and foreground detail.
"""


def _upright_cyl(mi, pos, radius, half_height):
    """Create an upright (Y-axis) cylinder transform with uniform cross-section.

    Mitsuba's default cylinder is along the Z axis.
    We rotate it to Y and use uniform XY scale to avoid the non-uniform scaling warning.
    """
    return (
        mi.ScalarTransform4f.translate(pos)
        @ mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=-90)
        @ mi.ScalarTransform4f.scale((radius, radius, half_height))
    )


def define_scene(mi):
    """
    Define vases scene geometry and materials.

    Args:
        mi: Mitsuba module instance

    Returns:
        dict: Scene objects (geometry + materials)
    """
    # Rotation for horizontal ground
    _horizontal_rot = mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=-90)

    return {
        "ground": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0.0, -0.5, 1.6)) @ _horizontal_rot @ mi.ScalarTransform4f.scale((2.5, 1.5, 1.0)),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.7, 0.7, 0.7]}},
        },
        "vase_left_outer": {
            "type": "cylinder",
            "to_world": _upright_cyl(mi, (-0.42, -0.20, 1.9), 0.12, 0.30),
            "bsdf": {
                "type": "roughplastic",
                "alpha": 0.08,
                "diffuse_reflectance": {"type": "rgb", "value": [0.92, 0.90, 0.88]},
            },
        },
        "vase_left_inner": {
            "type": "cylinder",
            "to_world": _upright_cyl(mi, (-0.42, -0.21, 1.9), 0.085, 0.29),
            "bsdf": {
                "type": "roughplastic",
                "alpha": 0.08,
                "diffuse_reflectance": {"type": "rgb", "value": [0.06, 0.06, 0.06]},
            },
        },
        "vase_center_outer": {
            "type": "cylinder",
            "to_world": _upright_cyl(mi, (-0.02, -0.16, 1.88), 0.17, 0.34),
            "bsdf": {
                "type": "roughplastic",
                "alpha": 0.11,
                "diffuse_reflectance": {"type": "rgb", "value": [0.85, 0.82, 0.80]},
            },
        },
        "vase_center_inner": {
            "type": "cylinder",
            "to_world": _upright_cyl(mi, (-0.02, -0.17, 1.88), 0.12, 0.33),
            "bsdf": {
                "type": "roughplastic",
                "alpha": 0.11,
                "diffuse_reflectance": {"type": "rgb", "value": [0.08, 0.08, 0.08]},
            },
        },
        "vase_right_outer": {
            "type": "cylinder",
            "to_world": _upright_cyl(mi, (0.42, -0.20, 1.92), 0.13, 0.30),
            "bsdf": {
                "type": "roughplastic",
                "alpha": 0.09,
                "diffuse_reflectance": {"type": "rgb", "value": [0.80, 0.78, 0.76]},
            },
        },
        "vase_right_inner": {
            "type": "cylinder",
            "to_world": _upright_cyl(mi, (0.42, -0.21, 1.92), 0.09, 0.29),
            "bsdf": {
                "type": "roughplastic",
                "alpha": 0.09,
                "diffuse_reflectance": {"type": "rgb", "value": [0.07, 0.07, 0.07]},
            },
        },
        # Flowers: thin stems + bright buds so vases are not empty
        "stem_1": {
            "type": "cylinder",
            "to_world": _upright_cyl(mi, (-0.02, 0.10, 1.88), 0.012, 0.22),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.20, 0.45, 0.18]}},
        },
        "stem_2": {
            "type": "cylinder",
            "to_world": _upright_cyl(mi, (0.03, 0.10, 1.90), 0.010, 0.20),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.19, 0.43, 0.17]}},
        },
        "flower_1": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate((-0.02, 0.20, 1.88)) @ mi.ScalarTransform4f.scale((0.045, 0.045, 0.045)),
            "bsdf": {"type": "roughplastic", "alpha": 0.15, "diffuse_reflectance": {"type": "rgb", "value": [0.95, 0.90, 0.30]}},
        },
        "flower_2": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate((0.03, 0.17, 1.90)) @ mi.ScalarTransform4f.scale((0.040, 0.040, 0.040)),
            "bsdf": {"type": "roughplastic", "alpha": 0.15, "diffuse_reflectance": {"type": "rgb", "value": [0.92, 0.34, 0.34]}},
        },
        # Foreground detail element to avoid over-clean synthetic look
        "foreground_blob": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate((-0.18, -0.44, 1.82)) @ mi.ScalarTransform4f.scale((0.07, 0.04, 0.06)),
            "bsdf": {"type": "roughplastic", "alpha": 0.2, "diffuse_reflectance": {"type": "rgb", "value": [0.75, 0.68, 0.58]}},
        },
    }
