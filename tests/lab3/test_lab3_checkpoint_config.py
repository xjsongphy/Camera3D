from __future__ import annotations

from pathlib import Path

from lab3.reconstruction.dgs import DGSConfig, _save_iterations
from lab3.reconstruction.nerf import NeRFConfig


def test_dgs_save_iterations_follow_interval_and_include_final_multiple() -> None:
    cfg = DGSConfig(repo_dir=Path("repo"), iterations=7000, save_every=2000)

    assert _save_iterations(cfg.iterations, cfg.save_every) == [2000, 4000, 6000]


def test_dgs_save_iterations_disable_when_interval_missing() -> None:
    assert _save_iterations(7000, None) == []


def test_nerf_config_defaults_keep_intermediate_checkpoints() -> None:
    cfg = NeRFConfig()

    assert cfg.save_every == 2000
    assert cfg.save_only_latest_checkpoint is False
