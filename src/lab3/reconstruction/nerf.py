from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from lab3.common import Lab3Error, require_tool, run_cmd, monitored_block, timed_block
from lab3.reconstruction.base import ReconstructionContext, Reconstructor, ViewerTarget
from lab3.reconstruction.nerfstudio import evaluate_nerfstudio, train_monitor_command
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


def config_from_dict(values: dict) -> NeRFConfig:
    return NeRFConfig(
        process_bin=str(values.get("process_bin", "ns-process-data")),
        train_bin=str(values.get("train_bin", "ns-train")),
        method=str(values.get("method", "nerfacto")),
        max_num_iterations=_optional_int(values.get("max_num_iterations", 30000)),
        save_every=_optional_int(values.get("save_every", 2000)),
        save_only_latest_checkpoint=bool(values.get("save_only_latest_checkpoint", False)),
        downscale_factor=_optional_int(values.get("downscale_factor")),
        skip_process_data=bool(values.get("skip_process_data", False)),
        colmap_model=(
            None if values.get("colmap_model") in (None, "")
            else Path(str(values["colmap_model"]))
        ),
    )


def add_cli_arguments(parser) -> None:
    parser.add_argument("--nerf-iterations", type=int, help="nerfstudio training iterations")
    parser.add_argument("--nerf-save-every", type=int, help="save nerfstudio checkpoint every N iterations")


def cli_overrides(arguments) -> dict:
    return {
        key: value for key, value in {
            "max_num_iterations": arguments.nerf_iterations,
            "save_every": arguments.nerf_save_every,
        }.items() if value is not None
    }


@dataclass(frozen=True)
class NeRFReconstructor(Reconstructor):
    config: NeRFConfig
    name: str = "nerf"
    consumes_shared_poses: bool = True

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

        transforms_path = processed_dir / "transforms.json"
        if context.dry_run:
            print(f"$ inject canonical split into {transforms_path}")
        else:
            apply_nerfstudio_split(transforms_path, context.split.train, context.split.test)

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
            wrapped_cmd = train_monitor_command(train_cmd, logs / "nerf_train.log")
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

    def evaluate(self, context: ReconstructionContext, eval_config, eval_dir: Path):
        return evaluate_nerfstudio(
            self.name,
            context,
            eval_config,
            self.config,
            eval_dir,
            notes="transforms.json contains the repository's explicit train/val/test filenames",
        )

    def stage_geometry(self, context: ReconstructionContext) -> list[Path]:
        from lab3.geometry import stage_exported_geometry

        return stage_exported_geometry(context, self.name)

    def qualitative_render_dir(self, run_dir: Path) -> Path:
        return run_dir / "results" / self.name / "renders"

    def viewer_targets(self, run_dir: Path) -> list[ViewerTarget]:
        train_root = run_dir / "results" / self.name / "train"
        configs = sorted(train_root.rglob("config.yml")) if train_root.exists() else []
        return [
            ViewerTarget(self.name, "external", path, ("--load-config", str(path)))
            for path in configs[-1:]
        ]

    def with_shared_poses(self, shared_dir: Path) -> Reconstructor:
        return replace(self, config=replace(self.config, colmap_model=shared_dir / "sparse" / "0"))


def apply_nerfstudio_split(
    transforms_path: Path, train_names: tuple[str, ...], test_names: tuple[str, ...]
) -> None:
    """Inject explicit filename splits supported by NerfstudioDataParser."""
    if not transforms_path.is_file():
        raise Lab3Error(f"nerfstudio transforms.json not found: {transforms_path}")
    metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = metadata.get("frames", [])
    paths_by_name = {
        Path(str(frame["file_path"])).name: str(frame["file_path"])
        for frame in frames
        if "file_path" in frame
    }

    def resolve(names: tuple[str, ...]) -> list[str]:
        missing = [name for name in names if name not in paths_by_name]
        if missing:
            raise Lab3Error(
                f"Canonical split references images absent from {transforms_path}: {missing[:5]}"
            )
        return [paths_by_name[name] for name in names]

    train_paths = resolve(train_names)
    test_paths = resolve(test_names)
    metadata["train_filenames"] = train_paths
    metadata["val_filenames"] = test_paths
    metadata["test_filenames"] = test_paths
    transforms_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _optional_int(value: object) -> int | None:
    return None if value in (None, "") else int(value)
