from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from lab3.common import Lab3Error, monitored_block, require_tool, run_cmd, timed_block
from lab3.reconstruction.base import ReconstructionContext, Reconstructor, ViewerTarget
from lab3.training_artifacts import export_training_scalar_artifacts


@dataclass(frozen=True)
class NeuSConfig:
    """nerfstudio NeuS/NeuS-facto configuration for COLMAP image captures."""

    train_bin: str = "ns-train"
    method: str = "neus-facto"
    max_num_iterations: int | None = 20001
    save_every: int | None = 2000
    save_only_latest_checkpoint: bool = False
    colmap_model: Path | None = None
    scene_scale: float = 2.0
    export_mesh: bool = True
    mesh_resolution: int = 512


@dataclass(frozen=True)
class NeuSReconstructor(Reconstructor):
    config: NeuSConfig
    name: str = "neus"

    def run(self, context: ReconstructionContext) -> None:
        if self.config.method not in {"neus", "neus-facto"}:
            raise Lab3Error("NeuS method must be 'neus' or 'neus-facto'")
        if self.config.mesh_resolution < 512 or self.config.mesh_resolution % 512 != 0:
            raise Lab3Error("NeuS mesh_resolution must be a positive multiple of 512")

        model_dir = self.config.colmap_model
        if model_dir is None:
            model_dir = context.prepared_dir / "sparse" / "0"
        model_dir = model_dir.resolve()
        dataset_dir = (context.output_dir / "processed").resolve()
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
                print(f"$ lab3 NeuS COLMAP-to-SDFStudio {model_dir} -> {dataset_dir}")
            else:
                build_sdfstudio_dataset(
                    model_dir,
                    context.images_dir.resolve(),
                    dataset_dir,
                    scene_scale=self.config.scene_scale,
                )

        train_cmd = [
            self.config.train_bin,
            self.config.method,
            "--data",
            str(dataset_dir),
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
        wrapped_cmd = [
            sys.executable,
            "-m",
            "lab3.ns_train_wrapper",
            "--log-path",
            str(logs / "neus_train.log"),
            "--",
            *train_cmd,
        ]
        with monitored_block(
            "neus_train", context.timings, context.peaks, enabled=not context.dry_run
        ):
            run_cmd(wrapped_cmd, dry_run=context.dry_run)

        if not context.dry_run:
            export_training_scalar_artifacts("neus", train_dir, logs)

        if self.config.export_mesh:
            config_path = _latest_config(train_dir)
            if config_path is None and not context.dry_run:
                raise Lab3Error(f"NeuS training config.yml not found under {train_dir}")
            export_cmd = [
                sys.executable,
                "-m",
                "lab3.reconstruction.neus_export",
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
        from lab3.evaluate import evaluate_nerfstudio

        return evaluate_nerfstudio(
            self.name,
            context,
            eval_config,
            self.config,
            context.timings,
            context.peaks,
            eval_dir,
            held_out="SDFStudio eval views (also seen during training)",
            notes=(
                "SDFStudioDataParser does not exclude eval frames from training; NeuS RGB metrics "
                "are training-view diagnostics, while mesh/geometry is the primary result"
            ),
            train_timing_key="neus_train",
            train_peak_key="neus_train",
        )

    def stage_geometry(self, context: ReconstructionContext) -> list[Path]:
        from lab3.geometry import stage_exported_geometry

        return stage_exported_geometry(context, self.name)

    def qualitative_render_dir(self, run_dir: Path) -> Path:
        return run_dir / "results" / self.name / "renders"

    def viewer_targets(self, run_dir: Path) -> list[ViewerTarget]:
        train_root = run_dir / "results" / self.name / "train"
        configs = sorted(train_root.rglob("config.yml")) if train_root.exists() else []
        targets = [ViewerTarget(self.name, "nerfstudio", path) for path in configs[-1:]]
        mesh = run_dir / "geometry" / self.name / "sdf_mesh.ply"
        if mesh.exists():
            targets.insert(0, ViewerTarget(self.name, "geometry", mesh))
        return targets

    def with_shared_poses(self, shared_dir: Path) -> Reconstructor:
        return replace(self, config=replace(self.config, colmap_model=shared_dir / "sparse" / "0"))


def build_sdfstudio_dataset(
    colmap_model: Path,
    images_dir: Path,
    output_dir: Path,
    *,
    scene_scale: float = 2.0,
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
    frames = []
    width = height = None
    for image_name, c2w, camera_id in images:
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


def _latest_config(train_dir: Path) -> Path | None:
    matches = sorted(train_dir.rglob("config.yml")) if train_dir.exists() else []
    return matches[-1] if matches else None
