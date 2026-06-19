from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

from lab3.common import Lab3Error, require_tool, run_cmd, monitored_block, timed_block
from lab3.reconstruction.base import ReconstructionContext, Reconstructor, ViewerTarget
from lab3.training_artifacts import export_training_scalar_artifacts


@dataclass(frozen=True)
class DGSConfig:
    repo_dir: Path | None = Path("gaussian-splatting")
    python_bin: str = sys.executable
    iterations: int | None = 7000
    save_every: int | None = 2000
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
class DGSReconstructor(Reconstructor):
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

        train_cmd = [
            self.config.python_bin,
            "train.py",
            "-s",
            str(source),
            "-m",
            str(output_dir),
        ]
        if self.config.eval_split:
            train_cmd.append("--eval")
        if self.config.iterations is not None:
            train_cmd.extend(["--iterations", str(self.config.iterations)])
        save_iterations = _save_iterations(self.config.iterations, self.config.save_every)
        if save_iterations:
            train_cmd.extend(["--save_iterations", *[str(step) for step in save_iterations]])
        if self.config.resolution is not None:
            train_cmd.extend(["--resolution", str(self.config.resolution)])
        train_cmd.extend(self.config.extra_args)

        if logs is not None:
            wrapped_cmd = [
                self.config.python_bin,
                "-m",
                "lab3.gs_train_wrapper",
                "--cwd",
                str(repo_dir),
                "--log-path",
                str(logs / "3dgs_train.log"),
                "--curve-path",
                str(logs / "3dgs_train_curve.csv"),
                "--",
                *train_cmd,
            ]
            train_cwd: Path | None = None
            train_log_path: Path | None = None
        else:
            wrapped_cmd = train_cmd
            train_cwd = repo_dir
            train_log_path = None

        with monitored_block("3dgs_train", context.timings, context.peaks, enabled=not context.dry_run):
            run_cmd(
                wrapped_cmd,
                dry_run=context.dry_run,
                cwd=train_cwd,
                log_path=train_log_path,
            )
        if logs is not None and not context.dry_run:
            export_training_scalar_artifacts("3dgs", output_dir, logs)

    def evaluate(self, context: ReconstructionContext, eval_config, eval_dir: Path):
        from lab3.evaluate import evaluate_3dgs

        return evaluate_3dgs(
            context, eval_config, self.config, context.timings, context.peaks, eval_dir
        )

    def stage_geometry(self, context: ReconstructionContext) -> list[Path]:
        from lab3.geometry import copy_geometry, find_3dgs_pointcloud

        point_cloud = find_3dgs_pointcloud(context.run_dir / "results" / self.name)
        if point_cloud is None:
            return []
        return [copy_geometry(point_cloud, context.run_dir / "geometry" / self.name, "gaussians.ply")]

    def qualitative_render_dir(self, run_dir: Path) -> Path:
        test_dir = run_dir / "results" / self.name / "test"
        matches = sorted(test_dir.glob("ours_*")) if test_dir.is_dir() else []
        return matches[-1] if matches else run_dir / "results" / self.name

    def viewer_targets(self, run_dir: Path) -> list[ViewerTarget]:
        candidates = [run_dir / "geometry" / self.name / "gaussians.ply"]
        candidates += sorted(
            (run_dir / "results" / self.name / "point_cloud").glob("iteration_*/point_cloud.ply")
        )
        return [ViewerTarget(self.name, "geometry", path) for path in candidates if path.exists()]

    def with_shared_poses(self, shared_dir: Path) -> Reconstructor:
        return replace(self, config=replace(self.config, colmap_source=shared_dir))


def _save_iterations(total_iterations: int | None, save_every: int | None) -> list[int]:
    if total_iterations is None or save_every is None or save_every <= 0:
        return []
    return list(range(save_every, total_iterations + 1, save_every))
