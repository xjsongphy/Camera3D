from __future__ import annotations

import argparse
import csv
import json
import os
import re
import runpy
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from lab3.common import Lab3Error, require_tool, run_cmd, monitored_block, timed_block
from lab3.evaluate import config_iterations, metrics_row, model_size_mb, render_fps
from lab3.metrics import compute_image_metrics
from lab3.reconstruction.base import ReconstructionContext, Reconstructor, ViewerTarget
from lab3.training_artifacts import export_training_scalar_artifacts


@dataclass(frozen=True)
class DGSConfig:
    repo_dir: Path | None = Path("gaussian-splatting")
    python_bin: str = sys.executable
    iterations: int | None = 7000
    # Explicit report checkpoints. These nodes only save snapshots; held-out
    # benchmarking is delegated to the pipeline's final evaluate stage.
    save_iterations: tuple[int, ...] = (2000, 4000, 6000, 7000)
    resolution: int | None = None
    data_device: str = "cpu"
    camera_cache_size: int = 4
    extra_args: tuple[str, ...] = ()
    # Shared COLMAP source (has ``images/`` + ``sparse/0``). When set, train.py
    # uses these poses directly; otherwise convert.py builds them first.
    colmap_source: Path | None = None
    colmap_bin: str = "colmap"
    # Enable Graphdeco's evaluation path. This adapter replaces Graphdeco's
    # fixed LLFF holdout with the repository's canonical test.txt manifest.
    eval_split: bool = True


def config_from_dict(values: dict) -> DGSConfig:
    iterations = _optional_int(values.get("iterations", 7000))
    return DGSConfig(
        repo_dir=(
            Path("gaussian-splatting")
            if values.get("repo_dir") in (None, "") else Path(str(values["repo_dir"]))
        ),
        python_bin=(
            sys.executable if values.get("python_bin") in (None, "", "python")
            else str(values["python_bin"])
        ),
        iterations=iterations,
        save_iterations=_configured_save_iterations(values, iterations),
        resolution=_optional_int(values.get("resolution")),
        data_device=str(values.get("data_device", "cpu")),
        camera_cache_size=int(values.get("camera_cache_size", 4)),
        extra_args=tuple(str(item) for item in values.get("extra_args", ())),
        colmap_source=(
            None if values.get("colmap_source") in (None, "")
            else Path(str(values["colmap_source"]))
        ),
        colmap_bin=str(values.get("colmap_bin", "colmap")),
        eval_split=bool(values.get("eval_split", True)),
    )


def add_cli_arguments(parser) -> None:
    parser.add_argument(
        "--dgs-repo", type=Path,
        help="Graphdeco gaussian-splatting repository path (default: ./gaussian-splatting)",
    )
    parser.add_argument("--dgs-iterations", type=int, help="3DGS training iterations")
    parser.add_argument(
        "--dgs-save-iterations", nargs="+", type=int,
        help="explicit 3DGS snapshot/evaluation iterations",
    )
    parser.add_argument(
        "--dgs-camera-cache-size", type=int,
        help="number of CPU-backed camera images kept on CUDA per 3DGS cache chunk",
    )


def cli_overrides(arguments) -> dict:
    return {
        key: value for key, value in {
            "repo_dir": arguments.dgs_repo,
            "iterations": arguments.dgs_iterations,
            "save_iterations": arguments.dgs_save_iterations,
            "camera_cache_size": arguments.dgs_camera_cache_size,
        }.items() if value is not None
    }


@dataclass(frozen=True)
class DGSReconstructor(Reconstructor):
    config: DGSConfig
    name: str = "3dgs"
    consumes_shared_poses: bool = True

    def run(self, context: ReconstructionContext) -> None:
        if self.config.data_device not in {"cpu", "cuda"}:
            raise Lab3Error("3DGS data_device must be 'cpu' or 'cuda'")
        if self.config.camera_cache_size <= 0:
            raise Lab3Error("3DGS camera_cache_size must be positive")
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

        train_cmd = _graphdeco_command(
            self.config.python_bin,
            repo_dir,
            "train.py",
            [
            "-s",
            str(source),
            "-m",
            str(output_dir),
            ],
        )
        if context.split.test and not self.config.eval_split:
            raise Lab3Error(
                "3DGS eval_split=false conflicts with the canonical held-out split; enable eval_split."
            )
        if context.split.test:
            train_cmd.append("--eval")
            split_path = source / "sparse" / "0" / "test.txt"
            if context.dry_run:
                print(f"$ write canonical 3DGS split {split_path}")
            else:
                split_path.parent.mkdir(parents=True, exist_ok=True)
                split_path.write_text("".join(f"{name}\n" for name in context.split.test), encoding="utf-8")
        if self.config.iterations is not None:
            train_cmd.extend(["--iterations", str(self.config.iterations)])
        save_iterations = _validate_save_iterations(
            self.config.iterations, self.config.save_iterations
        )
        if save_iterations:
            train_cmd.extend(["--save_iterations", *[str(step) for step in save_iterations]])
        if self.config.resolution is not None:
            train_cmd.extend(["--resolution", str(self.config.resolution)])
        train_cmd.extend(["--data_device", self.config.data_device])
        if self.config.data_device == "cpu":
            train_cmd.extend(["--camera_cache_size", str(self.config.camera_cache_size)])
        train_cmd.extend(self.config.extra_args)

        if logs is not None:
            wrapped_cmd = [
                self.config.python_bin,
                str(Path(__file__).resolve()),
                "train-monitor",
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
        if self.config.repo_dir is None:
            raise Lab3Error("3DGS repository is required for evaluation")
        repo_dir = self.config.repo_dir.resolve()
        model_dir = (context.run_dir / "results" / self.name).resolve()
        logs = context.run_dir / "logs"
        render_command = _graphdeco_command(
            self.config.python_bin, repo_dir, "render.py", ["-m", str(model_dir)]
        )
        with timed_block("3dgs_render", context.timings):
            run_cmd(
                render_command, dry_run=context.dry_run, cwd=repo_dir,
                log_path=logs / "3dgs_render.log",
            )
        pairs = [] if context.dry_run else _pair_renders(model_dir)
        metrics = (
            compute_image_metrics(
                [ground_truth for ground_truth, _ in pairs],
                [prediction for _, prediction in pairs],
                eval_size=eval_config.eval_size,
                lpips_enabled=eval_config.lpips,
            )
            if pairs else {"psnr": float("nan"), "ssim": float("nan"), "lpips": None, "n": 0}
        )
        if eval_config.native_crosscheck and not context.dry_run:
            with timed_block("3dgs_native_metrics", context.timings):
                run_cmd(
                    [self.config.python_bin, "metrics.py", "-m", str(model_dir)],
                    cwd=repo_dir,
                    log_path=logs / "3dgs_metrics.log",
                )
        return metrics_row(
            self.name,
            metrics,
            metric_source="lab3.metrics",
            held_out="prepared/test.txt (canonical split)",
            train_time_sec=context.timings.get("3dgs_train"),
            iterations=config_iterations(self.config),
            gpu_mem_peak_gb=context.peaks.get("3dgs_train"),
            render_fps=render_fps(metrics.get("n"), context.timings.get("3dgs_render", 0.0)),
            model_size_mb=(
                model_size_mb([model_dir / "point_cloud"])
                if not context.dry_run else float("nan")
            ),
            notes=(
                "3dgs consumes the canonical sparse/0/test.txt manifest; "
                + ("native results.json cross-check enabled; " if eval_config.native_crosscheck else "")
                + "render_fps includes render.py disk I/O"
            ),
        )

    def stage_geometry(self, context: ReconstructionContext) -> list[Path]:
        from lab3.geometry import copy_geometry

        point_cloud = _find_point_cloud(context.run_dir / "results" / self.name)
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
        targets = [ViewerTarget(self.name, "geometry", path) for path in candidates if path.exists()]
        model_dir = run_dir / "results" / self.name
        if model_dir.exists():
            targets.append(
                ViewerTarget(
                    self.name,
                    "sibr",
                    model_dir,
                    (str(_viewer_repo_dir(run_dir)),),
                )
            )
        return targets

    def with_shared_poses(self, shared_dir: Path) -> Reconstructor:
        return replace(self, config=replace(self.config, colmap_source=shared_dir))


def _validate_save_iterations(
    total_iterations: int | None, save_iterations: tuple[int, ...]
) -> list[int]:
    nodes = sorted(set(save_iterations))
    if any(node <= 0 for node in nodes):
        raise Lab3Error("3DGS save_iterations must contain only positive integers")
    if total_iterations is not None and any(node > total_iterations for node in nodes):
        raise Lab3Error(
            f"3DGS save_iterations cannot exceed iterations={total_iterations}: {nodes}"
        )
    return nodes


def _configured_save_iterations(
    values: dict, total_iterations: int | None
) -> tuple[int, ...]:
    configured = values.get("save_iterations")
    if configured is not None:
        if not isinstance(configured, (list, tuple)):
            raise Lab3Error("3DGS save_iterations must be a list of iteration numbers")
        return tuple(int(node) for node in configured)
    if total_iterations is None:
        return ()
    return tuple(node for node in (2000, 4000, 6000, total_iterations) if node <= total_iterations)


def _optional_int(value: object) -> int | None:
    return None if value in (None, "") else int(value)


def _find_point_cloud(model_dir: Path) -> Path | None:
    matches = (
        sorted((model_dir / "point_cloud").glob("iteration_*/point_cloud.ply"))
        if model_dir.is_dir() else []
    )
    return matches[-1] if matches else None


def _viewer_repo_dir(run_dir: Path) -> Path:
    config_path = run_dir / "configs" / "run_config.json"
    if not config_path.is_file():
        return Path("gaussian-splatting")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return Path("gaussian-splatting")
    reconstruction = data.get("reconstruction", {})
    dgs = reconstruction.get("3dgs", {}) if isinstance(reconstruction, dict) else {}
    configured = dgs.get("repo_dir") if isinstance(dgs, dict) else None
    return Path("gaussian-splatting") if configured in (None, "") else Path(str(configured))


def _graphdeco_command(
    python_bin: str, repo_dir: Path, script: str, arguments: list[str]
) -> list[str]:
    return [
        python_bin, str(Path(__file__).resolve()), "graphdeco",
        "--repo-dir", str(repo_dir), "--script", script, "--", *arguments,
    ]


def _pair_renders(model_dir: Path) -> list[tuple[Path, Path]]:
    test_dir = model_dir / "test"
    matches = sorted(test_dir.glob("ours_*")) if test_dir.is_dir() else []
    if not matches:
        return []
    renders_dir, ground_truth_dir = matches[-1] / "renders", matches[-1] / "gt"
    if not renders_dir.is_dir() or not ground_truth_dir.is_dir():
        return []
    return [
        (ground_truth_dir / render.name, render)
        for render in sorted(renders_dir.iterdir())
        if (ground_truth_dir / render.name).is_file()
    ]


def _run_graphdeco(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    repo_dir = options.repo_dir.resolve()
    script = repo_dir / options.script
    if not script.is_file():
        raise SystemExit(f"Graphdeco script not found: {script}")
    sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

    from scene import dataset_readers

    original = dataset_readers.readColmapSceneInfo

    class ManifestPath(str):
        def __contains__(self, item: object) -> bool:
            return False if item == "360" else super().__contains__(item)

    def read_manifest(path, images, depths, eval, train_test_exp):
        return original(ManifestPath(path), images, depths, eval, train_test_exp, llffhold=0)

    dataset_readers.sceneLoadTypeCallbacks["Colmap"] = read_manifest
    forwarded = options.args[1:] if options.args[:1] == ["--"] else options.args
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


CAMERA_RE = re.compile(r"^Reading camera\s+(\d+)/(\d+)")
PROGRESS_RE = re.compile(
    r"^Training progress:\s+\d+%.*?\|\s*(\d+)/(\d+)\s*\[[^\]]*Loss=([0-9.]+),\s*Depth Loss=([0-9.]+)\]"
)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class ProgressFilter:
    def __init__(self) -> None:
        self.active_inline = False
        self.last_train_iter: int | None = None
        self.curves: list[dict[str, object]] = []

    def handle(self, line: str) -> tuple[str | None, str | None]:
        clean = ANSI_RE.sub("", line)
        camera = CAMERA_RE.match(clean)
        if camera:
            self.active_inline = True
            return f"\rReading camera {camera.group(1)}/{camera.group(2)}", None
        progress = PROGRESS_RE.match(clean)
        if progress:
            iteration, total = int(progress.group(1)), int(progress.group(2))
            loss, depth_loss = float(progress.group(3)), float(progress.group(4))
            self.active_inline = True
            if self.last_train_iter != iteration:
                self.last_train_iter = iteration
                self.curves.append({
                    "iteration": iteration, "total_iterations": total,
                    "loss": loss, "depth_loss": depth_loss,
                })
            return (
                f"\rTraining progress: {iteration}/{total} | Loss={loss:.7f} | "
                f"Depth Loss={depth_loss:.7f}", None
            )
        prefix = "\n" if self.active_inline else ""
        self.active_inline = False
        return prefix + clean, None

    def finalize(self) -> tuple[str | None, str | None]:
        if not self.active_inline:
            return None, None
        self.active_inline = False
        return "\n", None


def _run_train_monitor(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--curve-path", type=Path, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not command:
        raise SystemExit("train-monitor requires a command after '--'")
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    progress = ProgressFilter()
    tail: deque[str] = deque(maxlen=120)
    with args.log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
        process = subprocess.Popen(
            command, cwd=str(args.cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            log.write(raw_line)
            tail.append(ANSI_RE.sub("", raw_line).rstrip("\n"))
            output, _ = progress.handle(raw_line)
            if output:
                _safe_write(output)
        output, _ = progress.finalize()
        if output:
            _safe_write(output)
        return_code = process.wait()
    _write_curve_csv(args.curve_path, progress.curves)
    if return_code:
        raise SystemExit(
            f"Command failed with exit code {return_code}: {' '.join(command)}\n"
            + "\n".join(tail)
        )


def _write_curve_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["iteration", "total_iterations", "loss", "depth_loss"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _safe_write(text: str) -> None:
    try:
        print(text, end="")
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode(sys.stdout.encoding or "utf-8", errors="replace"))
        sys.stdout.flush()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: dgs.py {graphdeco|train-monitor} ...")
    command, arguments = sys.argv[1], sys.argv[2:]
    if command == "graphdeco":
        _run_graphdeco(arguments)
    elif command == "train-monitor":
        _run_train_monitor(arguments)
    else:
        raise SystemExit(f"unknown 3DGS adapter command: {command}")


if __name__ == "__main__":
    main()
