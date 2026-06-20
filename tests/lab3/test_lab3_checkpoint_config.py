from __future__ import annotations

from pathlib import Path

import pytest

from lab3.common import Lab3Error
from lab3.reconstruction.dgs import DGSConfig, _validate_save_iterations
from lab3.reconstruction.nerf import NeRFConfig
from lab3.reconstruction.nerfstudio import scheduled_train_command, validate_save_iterations


def test_dgs_save_iterations_are_explicit_sorted_and_unique() -> None:
    cfg = DGSConfig(
        repo_dir=Path("repo"), iterations=7000,
        save_iterations=(7000, 2000, 4000, 2000),
    )

    assert _validate_save_iterations(cfg.iterations, cfg.save_iterations) == [2000, 4000, 7000]


def test_dgs_save_iterations_reject_nodes_after_training() -> None:
    with pytest.raises(Lab3Error, match="cannot exceed"):
        _validate_save_iterations(7000, (2000, 8000))


def test_nerf_config_defaults_keep_intermediate_checkpoints() -> None:
    cfg = NeRFConfig()

    assert cfg.save_iterations == (2000, 5000, 10000, 20000, 30000)
    assert cfg.train_num_rays_per_batch is None


def test_nerfstudio_explicit_schedule_command() -> None:
    command = scheduled_train_command(
        ["ns-train", "nerfacto", "--data", "scene"], (2000, 5000)
    )

    assert command[-4:] == ["--", "nerfacto", "--data", "scene"]
    assert command[command.index("--save-iterations") + 1:command.index("--")] == ["2000", "5000"]
    assert validate_save_iterations(5000, (5000, 2000, 2000)) == (2000, 5000)
