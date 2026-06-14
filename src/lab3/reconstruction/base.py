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


class Reconstructor(Protocol):
    name: str

    def run(self, context: ReconstructionContext) -> None:
        pass
