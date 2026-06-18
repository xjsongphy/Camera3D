from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from lab3.common import Lab3Error, require_tool, run_cmd, monitored_block, timed_block
from lab3.reconstruction.base import ReconstructionContext


@dataclass(frozen=True)
class DGSConfig:
    repo_dir: Path | None = Path("gaussian-splatting")
    python_bin: str = sys.executable
    iterations: int | None = 7000
    resolution: int | None = None
    extra_args: tuple[str, ...] = ()
    # Shared COLMAP source (has ``images/`` + ``sparse/0``). When set, train.py
    # uses these poses directly; otherwise convert.py builds them first.
    colmap_source: Path | None = None
    colmap_bin: str = "colmap"
    # Hold out every 8th image (train.py --eval) so render.py has a genuine
    # test split to evaluate, instead of training on every view.
    eval_split: bool = True


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
        repo_dir = self.config.repo_dir.resolve()
        train_py = repo_dir / "train.py"
        convert_py = repo_dir / "convert.py"
        logs = context.run_dir / "logs" if context.run_dir else None
        if not context.dry_run:
            require_tool(self.config.python_bin)
            if not train_py.exists():
                raise Lab3Error(f"3DGS train.py not found: {train_py}")
            context.output_dir.mkdir(parents=True, exist_ok=True)

        source = (self.config.colmap_source or context.prepared_dir).resolve()
        output_dir = context.output_dir.resolve()

        # train.py needs a COLMAP sparse model under the source. When poses are
        # not shared, build one with convert.py (runs COLMAP). Without this the
        # previous code pointed train.py at a bare images/ dir and always failed.
        if self.config.colmap_source is None:
            if not context.dry_run and not convert_py.exists():
                raise Lab3Error(f"3DGS convert.py not found: {convert_py}")
            convert_cmd = [self.config.python_bin, "convert.py", "-s", str(source)]
            if self.config.colmap_bin != "colmap":
                convert_cmd.extend(["--colmap_executable", self.config.colmap_bin])
            with timed_block("3dgs_convert", context.timings):
                run_cmd(
                    convert_cmd,
                    dry_run=context.dry_run,
                    cwd=repo_dir,
                    log_path=logs / "3dgs_convert.log" if logs else None,
                )

        cmd = [
            self.config.python_bin,
            "train.py",
            "-s",
            str(source),
            "-m",
            str(output_dir),
        ]
        if self.config.eval_split:
            cmd.append("--eval")
        if self.config.iterations is not None:
            cmd.extend(["--iterations", str(self.config.iterations)])
        if self.config.resolution is not None:
            cmd.extend(["--resolution", str(self.config.resolution)])
        cmd.extend(self.config.extra_args)

        with monitored_block("3dgs_train", context.timings, context.peaks, enabled=not context.dry_run):
            run_cmd(
                cmd,
                dry_run=context.dry_run,
                cwd=repo_dir,
                log_path=logs / "3dgs_train.log" if logs else None,
            )
