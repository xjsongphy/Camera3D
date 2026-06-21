from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class DatasetSplit:
    """Canonical image split shared by every reconstruction backend."""

    train: tuple[str, ...] = ()
    test: tuple[str, ...] = ()

    @classmethod
    def from_files(cls, train_path: Path, test_path: Path) -> DatasetSplit:
        return cls(_read_names(train_path), _read_names(test_path))

    @property
    def all(self) -> tuple[str, ...]:
        return self.train + self.test


def _read_names(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


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
    split: DatasetSplit = field(default_factory=DatasetSplit)


@dataclass(frozen=True)
class ViewerTarget:
    """A viewable artifact exposed by a reconstructor."""

    method: str
    kind: Literal["geometry", "nerfstudio", "sibr"]
    path: Path
    launcher_args: tuple[str, ...] = ()

    def command(self, executable: str) -> list[str]:
        return [executable, *self.launcher_args]


class Reconstructor:
    """Uniform lifecycle implemented by every reconstruction backend.

    Optional stages default to no-ops so orchestration never needs to know
    whether a method is radiance-, surface-, or geometry-based.
    """

    name: str
    config: Any
    shared_pose_priority: int = 100
    writes_shared_poses: bool = False
    consumes_shared_poses: bool = False
    geometry_reference: bool = False

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

    def validate_standalone_poses(self) -> None:
        return None
