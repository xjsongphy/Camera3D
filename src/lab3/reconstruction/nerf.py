from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from lab3.common import require_tool, run_cmd, monitored_block, timed_block
from lab3.reconstruction.base import ReconstructionContext
from lab3.training_artifacts import export_training_scalar_artifacts


@dataclass(frozen=True)
class NeRFConfig:
    process_bin: str = "ns-process-data"
    train_bin: str = "ns-train"
    method: str = "nerfacto"
    max_num_iterations: int | None = 30000
    save_every: int | None = 2000
    save_only_latest_checkpoint: bool = False
    downscale_factor: int | None = None
    skip_process_data: bool = False
    # Shared COLMAP ``sparse/0`` dir. When set, ns-process-data reuses these
    # poses (``--skip-colmap --colmap-model-path``) instead of re-running COLMAP.
    colmap_model: Path | None = None


@dataclass(frozen=True)
class NeRFReconstructor:
    config: NeRFConfig
    name: str = "nerf"

    def run(self, context: ReconstructionContext) -> None:
        processed_dir = (context.output_dir / "processed").resolve()
        train_dir = (context.output_dir / "train").resolve()
        images_dir = context.images_dir.resolve()
        colmap_model = self.config.colmap_model.resolve() if self.config.colmap_model is not None else None
        logs = context.run_dir / "logs" if context.run_dir else None
        if not context.dry_run:
            require_tool(self.config.process_bin)
            require_tool(self.config.train_bin)
            context.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.config.skip_process_data:
            cmd = [
                self.config.process_bin,
                "images",
                "--data",
                str(images_dir),
                "--output-dir",
                str(processed_dir),
            ]
            if colmap_model is not None:
                cmd.extend(
                    ["--skip-colmap", "--colmap-model-path", str(colmap_model)]
                )
            if self.config.downscale_factor is not None:
                cmd.extend(["--downscale-factor", str(self.config.downscale_factor)])
            with timed_block("nerf_process_data", context.timings):
                run_cmd(
                    cmd,
                    dry_run=context.dry_run,
                    log_path=logs / "nerf_process_data.log" if logs else None,
                )

        train_cmd = [
            self.config.train_bin,
            self.config.method,
            "--data",
            str(processed_dir),
            "--output-dir",
            str(train_dir),
            "--vis",
            "tensorboard",
        ]
        if self.config.max_num_iterations is not None:
            train_cmd.extend(["--max-num-iterations", str(self.config.max_num_iterations)])
        if self.config.save_every is not None:
            train_cmd.extend(["--steps-per-save", str(self.config.save_every)])
        train_cmd.extend(
            ["--save-only-latest-checkpoint", str(self.config.save_only_latest_checkpoint)]
        )

        if logs is not None:
            wrapped_cmd = [
                sys.executable,
                "-m",
                "lab3.ns_train_wrapper",
                "--log-path",
                str(logs / "nerf_train.log"),
                "--",
                *train_cmd,
            ]
            train_log_path: Path | None = None
        else:
            wrapped_cmd = train_cmd
            train_log_path = None

        with monitored_block("nerf_train", context.timings, context.peaks, enabled=not context.dry_run):
            run_cmd(
                wrapped_cmd,
                dry_run=context.dry_run,
                log_path=train_log_path,
            )
        if logs is not None and not context.dry_run:
            export_training_scalar_artifacts("nerf", train_dir, logs)
