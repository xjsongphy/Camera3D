from __future__ import annotations

import json
import math
import argparse
import importlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from lab3.common import Lab3Error, monitored_block, require_tool, run_cmd, timed_block
from lab3.evaluate import config_iterations, metrics_row, model_size_mb
from lab3.reconstruction.base import ReconstructionContext, Reconstructor, ViewerTarget
from lab3.reconstruction.nerfstudio import (
    evaluate_nerfstudio,
    latest_config,
    model_files,
    scheduled_train_command,
    train_monitor_command,
    validate_save_iterations,
)
from lab3.training_artifacts import export_training_scalar_artifacts


@dataclass(frozen=True)
class NeuSConfig:
    """nerfstudio NeuS/NeuS-facto configuration for COLMAP image captures."""

    train_bin: str = "ns-train"
    method: str = "neus-facto"
    max_num_iterations: int | None = 20001
    save_iterations: tuple[int, ...] = (2000, 5000, 10000, 20000)
    train_num_rays_per_batch: int | None = None
    colmap_model: Path | None = None
    scene_scale: float = 2.0
    export_mesh: bool = True
    mesh_resolution: int = 512


def config_from_dict(values: dict) -> NeuSConfig:
    return NeuSConfig(
        train_bin=str(values.get("train_bin", "ns-train")),
        method=str(values.get("method", "neus-facto")),
        max_num_iterations=_optional_int(values.get("max_num_iterations", 20001)),
        save_iterations=tuple(int(step) for step in values.get("save_iterations", (2000, 5000, 10000, 20000))),
        train_num_rays_per_batch=_optional_int(values.get("train_num_rays_per_batch")),
        colmap_model=(
            None if values.get("colmap_model") in (None, "")
            else Path(str(values["colmap_model"]))
        ),
        scene_scale=float(values.get("scene_scale", 2.0)),
        export_mesh=bool(values.get("export_mesh", True)),
        mesh_resolution=int(values.get("mesh_resolution", 512)),
    )


def add_cli_arguments(parser) -> None:
    parser.add_argument("--neus-iterations", type=int, help="NeuS/NeuS-facto training iterations")
    parser.add_argument("--neus-save-iterations", nargs="+", type=int, help="explicit NeuS checkpoint iterations")
    parser.add_argument(
        "--neus-train-rays-per-batch",
        type=int,
        help="override NeuS train_num_rays_per_batch",
    )
    parser.add_argument(
        "--neus-export-mesh",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="export NeuS mesh after training (default: on)",
    )


def cli_overrides(arguments) -> dict:
    return {
        key: value for key, value in {
            "max_num_iterations": arguments.neus_iterations,
            "save_iterations": arguments.neus_save_iterations,
            "train_num_rays_per_batch": arguments.neus_train_rays_per_batch,
            "export_mesh": arguments.neus_export_mesh,
        }.items() if value is not None
    }


@dataclass(frozen=True)
class NeuSReconstructor(Reconstructor):
    config: NeuSConfig
    name: str = "neus"
    consumes_shared_poses: bool = True

    def run(self, context: ReconstructionContext) -> None:
        if self.config.method not in {"neus", "neus-facto"}:
            raise Lab3Error("NeuS method must be 'neus' or 'neus-facto'")
        if self.config.mesh_resolution < 512 or self.config.mesh_resolution % 512 != 0:
            raise Lab3Error("NeuS mesh_resolution must be a positive multiple of 512")

        model_dir = self.config.colmap_model
        if model_dir is None:
            model_dir = context.prepared_dir / "sparse" / "0"
        model_dir = model_dir.resolve()
        train_dataset_dir = (context.output_dir / "processed" / "train").resolve()
        test_dataset_dir = (context.output_dir / "processed" / "test").resolve()
        train_dir = (context.output_dir / "train").resolve()
        logs = context.run_dir / "logs"

        if not context.dry_run:
            require_tool(self.config.train_bin)
            for filename in ("cameras.txt", "images.txt", "points3D.txt"):
                if not (model_dir / filename).is_file():
                    raise Lab3Error(
                        f"NeuS requires a COLMAP TXT model; missing {model_dir / filename}. "
                        "Include sfm in methods and keep share_poses enabled."
                    )
            context.output_dir.mkdir(parents=True, exist_ok=True)

        with timed_block("neus_process_data", context.timings):
            if context.dry_run:
                print(f"$ lab3 NeuS COLMAP-to-SDFStudio {model_dir} -> {train_dataset_dir}")
            else:
                build_sdfstudio_dataset(
                    model_dir,
                    context.images_dir.resolve(),
                    train_dataset_dir,
                    scene_scale=self.config.scene_scale,
                    image_names=context.split.train,
                )
                if context.split.test:
                    build_sdfstudio_dataset(
                        model_dir,
                        context.images_dir.resolve(),
                        test_dataset_dir,
                        scene_scale=self.config.scene_scale,
                        image_names=context.split.test,
                    )

        train_cmd = [
            self.config.train_bin,
            self.config.method,
            "--data",
            str(train_dataset_dir),
            "--output-dir",
            str(train_dir),
            "--vis",
            "tensorboard",
        ]
        if self.config.max_num_iterations is not None:
            train_cmd.extend(["--max-num-iterations", str(self.config.max_num_iterations)])
        if self.config.train_num_rays_per_batch is not None:
            if self.config.train_num_rays_per_batch <= 0:
                raise Lab3Error("NeuS train_num_rays_per_batch must be positive")
            train_cmd.extend(
                [
                    "--pipeline.datamanager.train-num-rays-per-batch",
                    str(self.config.train_num_rays_per_batch),
                ]
            )
        save_iterations = validate_save_iterations(
            self.config.max_num_iterations, self.config.save_iterations
        )
        train_cmd.extend(["--save-only-latest-checkpoint", "False"])
        train_cmd.extend(
            ["sdfstudio-data", "--data", str(train_dataset_dir), "--auto-orient", "False"]
        )
        if save_iterations:
            train_cmd = scheduled_train_command(train_cmd, save_iterations)
        wrapped_cmd = train_monitor_command(train_cmd, logs / "neus_train.log")
        with monitored_block(
            "neus_train", context.timings, context.peaks, enabled=not context.dry_run
        ):
            run_cmd(wrapped_cmd, dry_run=context.dry_run)

        if not context.dry_run:
            export_training_scalar_artifacts("neus", train_dir, logs)

        if self.config.export_mesh:
            config_path = latest_config(train_dir)
            if config_path is None and not context.dry_run:
                raise Lab3Error(f"NeuS training config.yml not found under {train_dir}")
            export_cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "export-mesh",
                "--load-config",
                str(config_path or train_dir / "config.yml"),
                "--output-path",
                str((context.output_dir / "mesh" / "sdf_mesh.ply").resolve()),
                "--resolution",
                str(self.config.mesh_resolution),
                "--bound",
                str(self.config.scene_scale / 2.0),
            ]
            with monitored_block(
                "neus_export_mesh", context.timings, context.peaks, enabled=not context.dry_run
            ):
                run_cmd(
                    export_cmd,
                    dry_run=context.dry_run,
                    log_path=logs / "neus_export_mesh.log",
                )

    def evaluate(self, context: ReconstructionContext, eval_config, eval_dir: Path):
        training_config = latest_config(context.run_dir / "results" / self.name / "train")
        evaluation_config = context.run_dir / "results" / self.name / "eval_config.yml"
        if not context.dry_run:
            if training_config is None:
                raise Lab3Error("NeuS training config.yml is missing")
            write_neus_eval_config(
                training_config,
                context.run_dir / "results" / self.name / "processed" / "test",
                evaluation_config,
            )

        try:
            return evaluate_nerfstudio(
                self.name,
                context,
                eval_config,
                self.config,
                eval_dir,
                notes="NeuS trains on train-only metadata and evaluates with a held-out config in the same coordinates",
                config_path=evaluation_config if not context.dry_run else None,
            )
        except Lab3Error as exc:
            return metrics_row(
                self.name,
                {"psnr": "N/A", "ssim": "N/A", "lpips": "N/A"},
                metric_source="geometry-only",
                held_out="render skipped",
                train_time_sec=context.timings.get("neus_train"),
                iterations=config_iterations(self.config, "max_num_iterations"),
                gpu_mem_peak_gb=context.peaks.get("neus_train"),
                render_fps=float("nan"),
                model_size_mb=(
                    model_size_mb(model_files(context.run_dir / "results" / self.name))
                    if not context.dry_run else float("nan")
                ),
                notes=(
                    "NeuS training and mesh export completed, but held-out RGB rendering was skipped: "
                    f"{exc}"
                ),
            ) | {"psnr": "N/A", "ssim": "N/A", "lpips": "N/A", "render_fps": "N/A"}

    def stage_geometry(self, context: ReconstructionContext) -> list[Path]:
        from lab3.geometry import stage_exported_geometry

        return stage_exported_geometry(context, self.name)

    def qualitative_render_dir(self, run_dir: Path) -> Path:
        return run_dir / "results" / self.name / "renders"

    def viewer_targets(self, run_dir: Path) -> list[ViewerTarget]:
        train_root = run_dir / "results" / self.name / "train"
        configs = sorted(train_root.rglob("config.yml")) if train_root.exists() else []
        targets = [
            ViewerTarget(self.name, "external", path, ("--load-config", str(path)))
            for path in configs[-1:]
        ]
        mesh = run_dir / "geometry" / self.name / "sdf_mesh.ply"
        if mesh.exists():
            targets.insert(0, ViewerTarget(self.name, "geometry", mesh))
        return targets

    def with_shared_poses(self, shared_dir: Path) -> Reconstructor:
        return replace(self, config=replace(self.config, colmap_model=shared_dir / "sparse" / "0"))

    def validate_standalone_poses(self) -> None:
        if self.config.colmap_model is None:
            raise Lab3Error(
                "NeuS cannot estimate poses itself. Enable share_poses with sfm, or set "
                "reconstruction.neus.colmap_model explicitly."
            )


def build_sdfstudio_dataset(
    colmap_model: Path,
    images_dir: Path,
    output_dir: Path,
    *,
    scene_scale: float = 2.0,
    image_names: tuple[str, ...] | None = None,
) -> Path:
    """Convert COLMAP TXT cameras into nerfstudio's SDFStudio ``meta_data.json``.

    Monocular depth/normals are intentionally optional: the core NeuS configs use
    ``include_mono_prior=False``. Sparse COLMAP points are used only to choose a
    robust origin and scale for the SDF bounding cube.
    """
    cameras = _read_cameras(colmap_model / "cameras.txt")
    images = _read_images(colmap_model / "images.txt")
    points = _read_points(colmap_model / "points3D.txt")
    if not images:
        raise Lab3Error(f"No registered images in {colmap_model / 'images.txt'}")

    centers = np.stack([entry[1][:3, 3] for entry in images], axis=0)
    if len(points):
        center = np.median(points, axis=0)
        radius = float(np.quantile(np.linalg.norm(points - center, axis=1), 0.95))
    else:
        center = np.median(centers, axis=0)
        radius = float(np.quantile(np.linalg.norm(centers - center, axis=1), 0.95))
    if not math.isfinite(radius) or radius <= 1e-8:
        radius = 1.0
    scale = (scene_scale * 0.45) / radius

    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_names = None if image_names is None else set(image_names)
    frames = []
    width = height = None
    for image_name, c2w, camera_id in images:
        if allowed_names is not None and image_name not in allowed_names:
            continue
        if camera_id not in cameras:
            raise Lab3Error(f"COLMAP image {image_name} references missing camera {camera_id}")
        intrinsics, camera_width, camera_height = cameras[camera_id]
        source = images_dir / image_name
        if not source.is_file():
            raise Lab3Error(f"Registered COLMAP image is missing: {source}")
        width, height = camera_width, camera_height
        normalized = c2w.copy()
        normalized[:3, 3] = (normalized[:3, 3] - center) * scale
        frames.append(
            {
                "rgb_path": str(source),
                "camtoworld": normalized.tolist(),
                "intrinsics": intrinsics.tolist(),
            }
        )
    if not frames:
        raise Lab3Error(f"NeuS split contains no registered images under {images_dir}")

    half = scene_scale / 2.0
    metadata = {
        "camera_model": "OPENCV",
        "height": height,
        "width": width,
        "has_mono_prior": False,
        "has_foreground_mask": False,
        "scene_box": {"aabb": [[-half, -half, -half], [half, half, half]], "near": 0.05, "far": 6.0},
        "frames": frames,
        "worldtogt": np.eye(4).tolist(),
    }
    path = output_dir / "meta_data.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def write_neus_eval_config(training_config: Path, test_data_dir: Path, output_path: Path) -> Path:
    """Clone a trained NeuS config while pointing its parser at held-out frames."""
    # nerfstudio serializes typed config objects with Python YAML tags. The file
    # is produced locally by ns-train, so using its matching loader is expected.
    config = yaml.load(training_config.read_text(encoding="utf-8"), Loader=yaml.Loader)
    test_data_dir = test_data_dir.resolve()
    try:
        config.data = test_data_dir
        config.pipeline.datamanager.data = test_data_dir
        dataparser = config.pipeline.datamanager.dataparser
    except AttributeError as exc:
        raise Lab3Error(f"Unexpected NeuS config structure: {training_config}") from exc
    dataparser.data = test_data_dir
    dataparser.auto_orient = False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(config), encoding="utf-8")
    return output_path


def _read_cameras(path: Path) -> dict[int, tuple[np.ndarray, int, int]]:
    cameras: dict[int, tuple[np.ndarray, int, int]] = {}
    for line in _data_lines(path):
        fields = line.split()
        camera_id, model = int(fields[0]), fields[1]
        width, height = int(fields[2]), int(fields[3])
        params = [float(v) for v in fields[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            fx, cx, cy = params[:3]
            fy = fx
        elif model in {"OPENCV", "FULL_OPENCV"}:
            fx, fy, cx, cy = params[:4]
        else:
            raise Lab3Error(f"Unsupported COLMAP camera model for NeuS: {model}")
        intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        cameras[camera_id] = (intrinsics, width, height)
    return cameras


def _optional_int(value: object) -> int | None:
    return None if value in (None, "") else int(value)


def _read_images(path: Path) -> list[tuple[str, np.ndarray, int]]:
    images: list[tuple[str, np.ndarray, int]] = []
    # COLMAP stores exactly two records per image. Keep blank 2D-point records;
    # filtering empty lines first would shift every following image header.
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        header = lines[index].strip()
        index += 1
        if not header or header.startswith("#"):
            continue
        fields = header.split()
        if len(fields) < 10:
            continue
        qvec = np.array([float(v) for v in fields[1:5]])
        tvec = np.array([float(v) for v in fields[5:8]])
        camera_id, name = int(fields[8]), fields[9]
        rotation = _qvec_to_rotmat(qvec)
        c2w = np.eye(4)
        c2w[:3, :3] = rotation.T
        c2w[:3, 3] = -rotation.T @ tvec
        images.append((name, c2w, camera_id))
        if index < len(lines):
            index += 1  # skip POINTS2D[] record (possibly blank)
    return images


def _read_points(path: Path) -> np.ndarray:
    values = []
    for line in _data_lines(path):
        fields = line.split()
        if len(fields) >= 4:
            values.append([float(v) for v in fields[1:4]])
    return np.asarray(values, dtype=np.float64).reshape((-1, 3))


def _data_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec / np.linalg.norm(qvec)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def _export_mesh(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Export a NeuS SDF zero-level mesh")
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--bound", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.resolution < 512 or args.resolution % 512 != 0:
        raise SystemExit("resolution must be a positive multiple of 512")

    generate_mesh_with_multires_marching_cubes, SDFField, eval_setup = _load_nerfstudio_export_modules()

    _, pipeline, _, _ = eval_setup(args.load_config, test_mode="inference")
    field = getattr(pipeline.model, "field", None)
    if not isinstance(field, SDFField):
        raise RuntimeError(f"Loaded field is {type(field).__name__}, expected SDFField")
    bound = float(args.bound)
    mesh = generate_mesh_with_multires_marching_cubes(
        geometry_callable_field=lambda points: field.forward_geonetwork(points)[:, 0].contiguous(),
        resolution=args.resolution,
        bounding_box_min=(-bound, -bound, -bound),
        bounding_box_max=(bound, bound, bound),
        isosurface_threshold=0.0,
        coarse_mask=None,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output_path)
    print(f"NeuS mesh: {args.output_path}")


def _load_nerfstudio_export_modules():
    """Import nerfstudio exporter modules even when this script is named ``neus.py``.

    Executing this file directly prepends its directory to ``sys.path``. If a
    local shadow module named ``nerfstudio`` is already present in
    ``sys.modules``, importing installed nerfstudio packages can fail with
    ``'nerfstudio' is not a package``. Temporarily remove the helper directory
    and clear only the shadowed module entry.
    """
    helper_dir = str(Path(__file__).resolve().parent)
    original_path = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(helper_dir)]
        shadow = sys.modules.get("nerfstudio")
        if shadow is not None:
            shadow_file = getattr(shadow, "__file__", None)
            if shadow_file and Path(shadow_file).resolve() == Path(__file__).resolve():
                sys.modules.pop("nerfstudio", None)
        marching_cubes = importlib.import_module("nerfstudio.exporter.marching_cubes")
        sdf_field = importlib.import_module("nerfstudio.fields.sdf_field")
        eval_utils = importlib.import_module("nerfstudio.utils.eval_utils")
        return (
            marching_cubes.generate_mesh_with_multires_marching_cubes,
            sdf_field.SDFField,
            eval_utils.eval_setup,
        )
    finally:
        sys.path[:] = original_path


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "export-mesh":
        raise SystemExit("usage: neus.py export-mesh ...")
    _export_mesh(sys.argv[2:])


if __name__ == "__main__":
    main()
