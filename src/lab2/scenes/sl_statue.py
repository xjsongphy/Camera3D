"""
Statue scene - approximation of paper Figure 11 composition.

Features candles + central wax-like face cast for complex geometry.
"""


def _upright_cyl(mi, pos, radius, half_height):
    """Create an upright (Y-axis) cylinder transform with uniform cross-section."""
    return (
        mi.ScalarTransform4f.translate(pos)
        @ mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=-90)
        @ mi.ScalarTransform4f.scale((radius, radius, half_height))
    )


def define_scene(mi):
    """
    Define statue scene geometry and materials.

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
        "candle_left": {
            "type": "cylinder",
            "to_world": _upright_cyl(mi, (-0.42, -0.20, 1.9), 0.12, 0.30),
            "bsdf": {"type": "roughplastic", "alpha": 0.08, "diffuse_reflectance": {"type": "rgb", "value": [0.82, 0.82, 0.80]}},
        },
        "candle_right": {
            "type": "cylinder",
            "to_world": _upright_cyl(mi, (0.42, -0.20, 1.9), 0.12, 0.30),
            "bsdf": {"type": "roughplastic", "alpha": 0.08, "diffuse_reflectance": {"type": "rgb", "value": [0.82, 0.82, 0.80]}},
        },
        "face_cast": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate((0.0, -0.12, 1.90)) @ mi.ScalarTransform4f.scale((0.19, 0.25, 0.16)),
            "bsdf": {"type": "roughplastic", "alpha": 0.10, "diffuse_reflectance": {"type": "rgb", "value": [0.88, 0.86, 0.80]}},
        },
        "face_nose": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate((0.02, -0.08, 2.03)) @ mi.ScalarTransform4f.scale((0.035, 0.045, 0.030)),
            "bsdf": {"type": "roughplastic", "alpha": 0.10, "diffuse_reflectance": {"type": "rgb", "value": [0.88, 0.86, 0.80]}},
        },
        "face_base": {
            "type": "cylinder",
            "to_world": _upright_cyl(mi, (0.0, -0.42, 1.88), 0.16, 0.08),
            "bsdf": {"type": "roughplastic", "alpha": 0.14, "diffuse_reflectance": {"type": "rgb", "value": [0.78, 0.76, 0.72]}},
        },
    }
