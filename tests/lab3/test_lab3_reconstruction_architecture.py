from __future__ import annotations

from pathlib import Path

from lab3.pipeline import config_from_dict
from lab3.reconstruction import RECONSTRUCTIONS, create_reconstructor


def test_reconstruction_configs_are_registry_keyed() -> None:
    config = config_from_dict({"input_dir": "input"})

    assert set(config.reconstruction) == set(RECONSTRUCTIONS)


def test_every_registered_backend_exposes_the_same_lifecycle() -> None:
    config = config_from_dict({"input_dir": Path("input")})
    lifecycle = (
        "run", "evaluate", "stage_geometry", "qualitative_render_dir",
        "viewer_targets", "with_shared_poses", "validate_standalone_poses",
    )

    for name in RECONSTRUCTIONS:
        reconstructor = create_reconstructor(name, config)
        assert all(callable(getattr(reconstructor, method)) for method in lifecycle)
