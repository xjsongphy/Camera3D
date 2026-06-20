from __future__ import annotations

from pathlib import Path

import pytest

from lab3.common import Lab3Error
from lab3.reconstruction.sfm import (
    SfMConfig,
    _colmap_feature_extractor_command,
    _resolve_hloc_matcher,
    _validate_hloc_combination,
    config_from_dict,
)


def test_sfm_config_defaults_to_current_classic_pipeline() -> None:
    cfg = SfMConfig()

    assert cfg.matcher == "sequential"
    assert cfg.feature_extractor == "sift"
    assert cfg.feature_matcher is None
    assert cfg.pair_overlap == 10
    assert cfg.quadratic_overlap is True


def test_sfm_config_parses_optional_learned_feature_fields() -> None:
    cfg = config_from_dict(
        {
            "matcher": "exhaustive",
            "feature_extractor": "superpoint_aachen",
            "feature_matcher": "superpoint+lightglue",
            "python_bin": "python3.11",
            "pair_overlap": 6,
            "quadratic_overlap": False,
        }
    )

    assert cfg.matcher == "exhaustive"
    assert cfg.feature_extractor == "superpoint_aachen"
    assert cfg.feature_matcher == "superpoint+lightglue"
    assert cfg.python_bin == "python3.11"
    assert cfg.pair_overlap == 6
    assert cfg.quadratic_overlap is False


def test_sift_dsp_enables_affine_and_domain_size_pooling() -> None:
    cmd = _colmap_feature_extractor_command(
        SfMConfig(feature_extractor="sift_dsp"),
        Path("database.db"),
        Path("images"),
    )

    assert "--SiftExtraction.estimate_affine_shape" in cmd
    assert "--SiftExtraction.domain_size_pooling" in cmd


def test_hloc_matcher_defaults_follow_extractor_family() -> None:
    assert _resolve_hloc_matcher("superpoint_aachen", None) == "superpoint+lightglue"
    assert _resolve_hloc_matcher("disk", None) == "disk+lightglue"
    assert _resolve_hloc_matcher("aliked-n16", None) == "aliked+lightglue"


def test_hloc_matcher_requires_explicit_choice_for_unknown_family() -> None:
    with pytest.raises(Lab3Error, match="No default learned matcher"):
        _resolve_hloc_matcher("r2d2", None)


def test_hloc_matcher_rejects_incompatible_feature_family() -> None:
    with pytest.raises(Lab3Error, match="expects extractor 'disk'"):
        _validate_hloc_combination("superpoint_aachen", "disk+lightglue")
