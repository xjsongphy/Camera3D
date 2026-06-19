from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ReconstructionContext:
    run_dir: Path
    prepared_dir: Path
    images_dir: Path
    output_dir: Path
    config_dir: Path
    dry_run: bool = False
    force: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    # Peak GPU memory (GiB) per stage, recorded by ``monitored_block`` around the
    # heavy training/MVS commands. Shared across stages like ``timings`` so the
    # evaluate stage can surface a per-method peak in ``metrics.csv`` (§5.2).
    peaks: dict[str, float] = field(default_factory=dict)
    # When set (pose sharing), COLMAP SfM writes the shared ``sparse/0`` model
    # under this directory so 3DGS / nerfstudio reuse identical camera poses.
    shared_colmap_dir: Path | None = None


@dataclass(frozen=True)
class ViewerTarget:
    """A viewable artifact exposed by a reconstructor."""

    method: str
    kind: Literal["geometry", "nerfstudio"]
    path: Path


class Reconstructor:
    """Uniform lifecycle implemented by every reconstruction backend.

    Optional stages default to no-ops so orchestration never needs to know
    whether a method is radiance-, surface-, or geometry-based.
    """

    name: str
    config: Any
    shared_pose_priority: int = 100
    writes_shared_poses: bool = False

    def run(self, context: ReconstructionContext) -> None:
        raise NotImplementedError

    def evaluate(
        self, context: ReconstructionContext, eval_config: Any, eval_dir: Path
    ) -> dict[str, Any] | None:
        return None

    def stage_geometry(self, context: ReconstructionContext) -> list[Path]:
        return []

    def qualitative_render_dir(self, run_dir: Path) -> Path | None:
        return None

    def viewer_targets(self, run_dir: Path) -> list[ViewerTarget]:
        return []

    def with_shared_poses(self, shared_dir: Path) -> Reconstructor:
        return self
