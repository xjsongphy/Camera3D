from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


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


class Reconstructor(Protocol):
    name: str

    def run(self, context: ReconstructionContext) -> None:
        pass
