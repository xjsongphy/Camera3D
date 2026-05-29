"""
Training board scene - simple plane with random texture (from paper).

The board is tilted against a wall corner (like an easel).
This scene matches the training board described in the structured light paper.
"""


def define_scene(mi):
    """
    Define training board scene geometry and materials.

    Args:
        mi: Mitsuba module instance

    Returns:
        dict: Scene objects (geometry + materials)
    """
    # Rotation for boards facing camera
    _face_camera_rot = mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=180)

    # Tilt the board like an easel
    tilt = mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=-15)

    return {
        "board": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0.0, 0.0, 2.2)) @ tilt @ _face_camera_rot @ mi.ScalarTransform4f.scale((2.0, 2.0, 1.0)),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.75, 0.75, 0.75]}},
        }
    }
