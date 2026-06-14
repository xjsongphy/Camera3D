from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from lab3.common import Lab3Error, require_tool, run_cmd, timed_block
from lab3.reconstruction.base import ReconstructionContext


@dataclass(frozen=True)
class DGSConfig:
    repo_dir: Path | None = None
    python_bin: str = sys.executable
    iterations: int | None = 7000
    resolution: int | None = None
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class DGSReconstructor:
    config: DGSConfig
    name: str = "3dgs"

    def run(self, context: ReconstructionContext) -> None:
        if self.config.repo_dir is None:
            raise Lab3Error(
                "3DGS requires --dgs-repo or config.reconstruction.3dgs.repo_dir "
                "pointing to the GraphDeco gaussian-splatting checkout."
            )
        train_py = self.config.repo_dir / "train.py"
        if not context.dry_run:
            require_tool(self.config.python_bin)
            if not train_py.exists():
                raise Lab3Error(f"3DGS train.py not found: {train_py}")
            context.output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.config.python_bin,
            str(train_py),
            "-s",
            str(context.prepared_dir),
            "-m",
            str(context.output_dir),
        ]
        if self.config.iterations is not None:
            cmd.extend(["--iterations", str(self.config.iterations)])
        if self.config.resolution is not None:
            cmd.extend(["--resolution", str(self.config.resolution)])
        cmd.extend(self.config.extra_args)

        with timed_block("3dgs_train", context.timings):
            run_cmd(cmd, dry_run=context.dry_run, cwd=self.config.repo_dir)
