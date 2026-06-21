#!/usr/bin/env python3
"""Render per-method camera-path videos from one completed Lab 3 run.

This is a post-processing script. It does not retrain anything: it loads the
final 3DGS / Nerfacto / NeuS artifacts from an existing run directory, fits one
continuous camera trajectory from the original discrete shared poses, renders
that same trajectory for each method, encodes one MP4 per method, and writes a
timing summary.

Design constraints for this script:
- keep everything in one file for convenience
- every important parameter has an editable default below
- every default is overridable from the CLI
- do not assume the run name contains "fps4" even if that is the current use
- do not stitch multi-method comparison videos; emit one video per method
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


# Editable defaults. CLI flags override these values.
DEFAULTS = {
    "run_dir": ROOT / "outputs" / "lab3" / "20260620_200014_dormitory_fps4",
    "fps": 24.0,
    "duration_sec": 30.0,
    "width": None,
    "height": None,
    "output_dirname": "renders/camera_path_videos",
    "methods": ("3dgs", "nerf", "neus"),
    "ffmpeg_bin": "ffmpeg",
    "dgs_repo": ROOT / "gaussian-splatting",
    "dgs_iteration": -1,
    "dgs_white_background": False,
    "dgs_sh_degree": 3,
    "dgs_antialiasing": False,
    "device": "cuda",
    "video_codec": "libx264",
    "video_pix_fmt": "yuv420p",
    "overwrite": False,
    "keep_frames": True,
}


class CameraPathVideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    run_dir: Path
    fps: float
    duration_sec: float
    width: int | None
    height: int | None
    output_dirname: str
    methods: tuple[str, ...]
    ffmpeg_bin: str
    dgs_repo: Path
    dgs_iteration: int
    dgs_white_background: bool
    dgs_sh_degree: int
    dgs_antialiasing: bool
    device: str
    video_codec: str
    video_pix_fmt: str
    overwrite: bool
    keep_frames: bool


@dataclass(frozen=True)
class PoseSample:
    image_name: str
    t: float
    center: np.ndarray  # (3,)
    c2w: np.ndarray  # (4, 4)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class FramePose:
    index: int
    timestamp_sec: float
    center: np.ndarray  # (3,)
    c2w: np.ndarray  # (4, 4)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    prepared_dir: Path
    images_dir: Path
    manifest_path: Path
    sparse_dir: Path
    output_root: Path
    nerf_config: Path
    neus_config: Path
    dgs_model_dir: Path
    width: int
    height: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULTS["run_dir"])
    parser.add_argument("--fps", type=float, default=DEFAULTS["fps"])
    parser.add_argument("--duration-sec", type=float, default=DEFAULTS["duration_sec"])
    parser.add_argument("--width", type=int, default=DEFAULTS["width"])
    parser.add_argument("--height", type=int, default=DEFAULTS["height"])
    parser.add_argument("--output-dirname", default=DEFAULTS["output_dirname"])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULTS["methods"]))
    parser.add_argument("--ffmpeg-bin", default=DEFAULTS["ffmpeg_bin"])
    parser.add_argument("--dgs-repo", type=Path, default=DEFAULTS["dgs_repo"])
    parser.add_argument("--dgs-iteration", type=int, default=DEFAULTS["dgs_iteration"])
    parser.add_argument(
        "--dgs-white-background",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["dgs_white_background"],
    )
    parser.add_argument(
        "--dgs-antialiasing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["dgs_antialiasing"],
    )
    parser.add_argument("--device", default=DEFAULTS["device"])
    parser.add_argument("--video-codec", default=DEFAULTS["video_codec"])
    parser.add_argument("--video-pix-fmt", default=DEFAULTS["video_pix_fmt"])
    parser.add_argument("--overwrite", action="store_true", default=DEFAULTS["overwrite"])
    parser.add_argument(
        "--keep-frames",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["keep_frames"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config(
        run_dir=args.run_dir.resolve(),
        fps=float(args.fps),
        duration_sec=float(args.duration_sec),
        width=args.width,
        height=args.height,
        output_dirname=str(args.output_dirname),
        methods=tuple(str(method).lower() for method in args.methods),
        ffmpeg_bin=str(args.ffmpeg_bin),
        dgs_repo=args.dgs_repo.resolve(),
        dgs_iteration=int(args.dgs_iteration),
        dgs_white_background=bool(args.dgs_white_background),
        dgs_sh_degree=int(DEFAULTS["dgs_sh_degree"]),
        dgs_antialiasing=bool(args.dgs_antialiasing),
        device=str(args.device),
        video_codec=str(args.video_codec),
        video_pix_fmt=str(args.video_pix_fmt),
        overwrite=bool(args.overwrite),
        keep_frames=bool(args.keep_frames),
    )
    validate_config(cfg)
    artifacts = resolve_run_artifacts(cfg)
    pose_samples = load_pose_samples(artifacts)
    frames = sample_camera_path(pose_samples, cfg, artifacts.width, artifacts.height)
    write_trajectory_manifests(artifacts.output_root, frames, cfg)

    summaries: list[dict[str, Any]] = []
    for method in cfg.methods:
        if method == "3dgs":
            summaries.append(render_3dgs_video(cfg, artifacts, frames))
        elif method == "nerf":
            summaries.append(render_nerfstudio_video("nerf", cfg, artifacts, frames))
        elif method == "neus":
            summaries.append(render_nerfstudio_video("neus", cfg, artifacts, frames))
        else:
            raise CameraPathVideoError(f"Unsupported method: {method}")

    write_summary(artifacts.output_root, summaries)
    print(f"Done: {artifacts.output_root}")
    return 0


def validate_config(cfg: Config) -> None:
    if cfg.fps <= 0:
        raise CameraPathVideoError(f"fps must be positive, got {cfg.fps}")
    if cfg.duration_sec <= 0:
        raise CameraPathVideoError(f"duration_sec must be positive, got {cfg.duration_sec}")
    if (cfg.width is None) ^ (cfg.height is None):
        raise CameraPathVideoError("width and height must be set together")
    if cfg.width is not None and cfg.width <= 0:
        raise CameraPathVideoError(f"width must be positive, got {cfg.width}")
    if cfg.height is not None and cfg.height <= 0:
        raise CameraPathVideoError(f"height must be positive, got {cfg.height}")
    if not cfg.methods:
        raise CameraPathVideoError("methods must not be empty")
    if shutil.which(cfg.ffmpeg_bin) is None:
        raise CameraPathVideoError(f"ffmpeg not found in PATH: {cfg.ffmpeg_bin}")


def resolve_run_artifacts(cfg: Config) -> RunArtifacts:
    run_dir = cfg.run_dir
    if not run_dir.is_dir():
        raise CameraPathVideoError(f"Run directory not found: {run_dir}")
    prepared_dir = run_dir / "prepared"
    images_dir = prepared_dir / "images"
    manifest_path = prepared_dir / "manifest.csv"
    sparse_dir = prepared_dir / "sparse" / "0"
    if not images_dir.is_dir():
        raise CameraPathVideoError(f"Prepared images directory not found: {images_dir}")
    if not manifest_path.is_file():
        raise CameraPathVideoError(f"Prepared manifest not found: {manifest_path}")
    if not sparse_dir.is_dir():
        raise CameraPathVideoError(f"Shared COLMAP sparse model not found: {sparse_dir}")

    run_config = read_json(run_dir / "configs" / "run_config.json", required=False)
    width, height = resolve_output_size(cfg, run_config, images_dir)
    output_root = run_dir / cfg.output_dirname
    output_root.mkdir(parents=True, exist_ok=True)

    nerf_config = latest_config(run_dir / "results" / "nerf" / "train")
    neus_config = latest_config(run_dir / "results" / "neus" / "train")
    if nerf_config is None:
        raise CameraPathVideoError("Nerfacto config.yml not found under results/nerf/train")
    if neus_config is None:
        raise CameraPathVideoError("NeuS config.yml not found under results/neus/train")
    dgs_model_dir = run_dir / "results" / "3dgs"
    if not dgs_model_dir.is_dir():
        raise CameraPathVideoError(f"3DGS model directory not found: {dgs_model_dir}")

    return RunArtifacts(
        run_dir=run_dir,
        prepared_dir=prepared_dir,
        images_dir=images_dir,
        manifest_path=manifest_path,
        sparse_dir=sparse_dir,
        output_root=output_root,
        nerf_config=nerf_config,
        neus_config=neus_config,
        dgs_model_dir=dgs_model_dir,
        width=width,
        height=height,
    )


def resolve_output_size(
    cfg: Config,
    run_config: dict[str, Any] | None,
    images_dir: Path,
) -> tuple[int, int]:
    if cfg.width is not None and cfg.height is not None:
        return cfg.width, cfg.height
    if run_config is not None:
        eval_size = run_config.get("eval_size")
        if isinstance(eval_size, list) and len(eval_size) == 2:
            return int(eval_size[1]), int(eval_size[0])
    image_files = sorted(path for path in images_dir.iterdir() if path.is_file())
    if not image_files:
        raise CameraPathVideoError("Cannot infer output size: prepared/images is empty")
    with Image.open(image_files[0]) as image:
        return image.width, image.height


def read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise CameraPathVideoError(f"JSON file not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise CameraPathVideoError(f"Expected JSON object in {path}")
    return data


def latest_config(root: Path) -> Path | None:
    matches = sorted(root.rglob("config.yml")) if root.exists() else []
    return matches[-1] if matches else None


def load_pose_samples(artifacts: RunArtifacts) -> list[PoseSample]:
    ordered_names = load_manifest_image_names(artifacts.manifest_path)
    name_order = {name: index for index, name in enumerate(ordered_names)}
    reconstruction = load_colmap_reconstruction(artifacts.sparse_dir)

    samples: list[PoseSample] = []
    for image in reconstruction.images.values():
        image_name = image.name
        if image_name not in name_order:
            continue
        camera = reconstruction.cameras[image.camera_id]
        width = int(camera.width)
        height = int(camera.height)
        fx, fy, cx, cy = camera_intrinsics(camera)
        cam_from_world = image.cam_from_world()
        c2w_3x4 = np.asarray(cam_from_world.inverse().matrix(), dtype=np.float64)
        c2w_rot = c2w_3x4[:, :3]
        center = c2w_3x4[:, 3]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = c2w_rot
        c2w[:3, 3] = center
        samples.append(
            PoseSample(
                image_name=image_name,
                t=float(name_order[image_name]),
                center=center,
                c2w=c2w,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=width,
                height=height,
            )
        )
    samples.sort(key=lambda item: item.t)
    if len(samples) < 2:
        raise CameraPathVideoError("Need at least two valid poses from the shared sparse model")
    return assign_arc_length_parameter(samples)


def load_manifest_image_names(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("image_name", "").strip()
            if name:
                rows.append(name)
    if not rows:
        raise CameraPathVideoError(f"No image_name rows found in {path}")
    return rows


def load_colmap_reconstruction(sparse_dir: Path):
    try:
        import pycolmap  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency/runtime guard
        raise CameraPathVideoError(
            "pycolmap is required to load the shared sparse reconstruction for camera-path rendering"
        ) from exc
    return pycolmap.Reconstruction(str(sparse_dir))


def camera_intrinsics(camera: Any) -> tuple[float, float, float, float]:
    raw_model = getattr(camera, "model", "")
    model = str(raw_model).upper()
    if "." in model:
        model = model.rsplit(".", 1)[-1]
    params = list(camera.params)
    if model == "SIMPLE_PINHOLE":
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif model == "PINHOLE":
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    elif model == "SIMPLE_RADIAL":
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif model == "RADIAL":
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif model == "OPENCV":
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    else:
        raise CameraPathVideoError(f"Unsupported COLMAP camera model for rendering script: {camera.model}")
    return fx, fy, cx, cy


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def assign_arc_length_parameter(samples: list[PoseSample]) -> list[PoseSample]:
    centers = np.stack([sample.center for sample in samples], axis=0)
    deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    if cumulative[-1] <= 1e-9:
        cumulative = np.arange(len(samples), dtype=np.float64)
    out: list[PoseSample] = []
    for sample, value in zip(samples, cumulative, strict=True):
        out.append(
            PoseSample(
                image_name=sample.image_name,
                t=float(value),
                center=sample.center,
                c2w=sample.c2w,
                fx=sample.fx,
                fy=sample.fy,
                cx=sample.cx,
                cy=sample.cy,
                width=sample.width,
                height=sample.height,
            )
        )
    return out


def sample_camera_path(
    samples: list[PoseSample],
    cfg: Config,
    width: int,
    height: int,
) -> list[FramePose]:
    frame_count = max(1, round(cfg.fps * cfg.duration_sec))
    key_t = np.array([sample.t for sample in samples], dtype=np.float64)
    centers = np.stack([sample.center for sample in samples], axis=0)
    quats = np.stack([rotation_to_quaternion(sample.c2w[:3, :3]) for sample in samples], axis=0)
    ts = np.linspace(key_t[0], key_t[-1], num=frame_count)

    if len(samples) >= 4:
        center_samples = catmull_rom_spline(key_t, centers, ts)
    else:
        center_samples = np.column_stack(
            [np.interp(ts, key_t, centers[:, axis]) for axis in range(3)]
        )

    frames: list[FramePose] = []
    for index, t_value in enumerate(ts):
        left, right, alpha = neighbor_indices(key_t, t_value)
        quat = slerp(quats[left], quats[right], alpha)
        rot = quaternion_to_rotation(quat)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rot
        c2w[:3, 3] = center_samples[index]
        ref = samples[left]
        frames.append(
            FramePose(
                index=index,
                timestamp_sec=index / cfg.fps,
                center=center_samples[index],
                c2w=c2w,
                fx=scale_intrinsic(ref.fx, ref.width, width),
                fy=scale_intrinsic(ref.fy, ref.height, height),
                cx=scale_intrinsic(ref.cx, ref.width, width),
                cy=scale_intrinsic(ref.cy, ref.height, height),
                width=width,
                height=height,
            )
        )
    return frames


def scale_intrinsic(value: float, src: int, dst: int) -> float:
    return float(value) * float(dst) / float(src)


def colmap_to_nerfstudio_c2w(c2w: np.ndarray) -> np.ndarray:
    """Convert a COLMAP (OpenCV) camera-to-world matrix to Nerfstudio (OpenGL) convention.

    COLMAP uses  +X right, +Y down,  +Z forward  (OpenCV).
    Nerfstudio uses +X right, +Y up,    -Z forward  (OpenGL).

    The conversion negates the camera's Y and Z basis vectors (columns 1 & 2),
    which is equivalent to ``c2w @ diag(1, -1, -1, 1)``.
    """
    result = c2w.copy()
    result[:3, 1] *= -1.0
    result[:3, 2] *= -1.0
    return result


def neighbor_indices(key_t: np.ndarray, target: float) -> tuple[int, int, float]:
    right = int(np.searchsorted(key_t, target, side="right"))
    left = max(0, right - 1)
    right = min(len(key_t) - 1, right)
    if left == right:
        return left, right, 0.0
    denom = key_t[right] - key_t[left]
    alpha = 0.0 if denom <= 0.0 else float((target - key_t[left]) / denom)
    return left, right, alpha


def catmull_rom_spline(key_t: np.ndarray, points: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    result = np.empty((len(query_t), 3), dtype=np.float64)
    for idx, t_value in enumerate(query_t):
        i1 = max(0, int(np.searchsorted(key_t, t_value, side="right")) - 1)
        i2 = min(len(key_t) - 1, i1 + 1)
        i0 = max(0, i1 - 1)
        i3 = min(len(key_t) - 1, i2 + 1)
        t1 = key_t[i1]
        t2 = key_t[i2]
        u = 0.0 if t2 <= t1 else float((t_value - t1) / (t2 - t1))
        p0 = points[i0]
        p1 = points[i1]
        p2 = points[i2]
        p3 = points[i3]
        result[idx] = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * u
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * (u * u)
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * (u * u * u)
        )
    return result


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        diag = np.diag(rotation)
        axis = int(np.argmax(diag))
        if axis == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / s,
                    0.25 * s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    0.25 * s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    return quat / np.linalg.norm(quat)


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    a = q0 / np.linalg.norm(q0)
    b = q1 / np.linalg.norm(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + alpha * (b - a)
        return out / np.linalg.norm(out)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return s0 * a + s1 * b


def write_trajectory_manifests(output_root: Path, frames: list[FramePose], cfg: Config) -> None:
    generic_payload = {
        "fps": cfg.fps,
        "duration_sec": cfg.duration_sec,
        "frame_count": len(frames),
        "frames": [
            {
                "index": frame.index,
                "timestamp_sec": frame.timestamp_sec,
                "camera_to_world": frame.c2w.tolist(),
                "center": frame.center.tolist(),
                "fx": frame.fx,
                "fy": frame.fy,
                "cx": frame.cx,
                "cy": frame.cy,
                "width": frame.width,
                "height": frame.height,
            }
            for frame in frames
        ],
    }
    write_json(output_root / "camera_path.json", generic_payload)

    nerfstudio_payload = {
        "camera_type": "perspective",
        "render_height": frames[0].height,
        "render_width": frames[0].width,
        "fps": cfg.fps,
        "seconds": cfg.duration_sec,
        "is_cycle": False,
        "camera_path": [
            {
                "camera_to_world": colmap_to_nerfstudio_c2w(frame.c2w).tolist(),
                "fov": 2.0 * math.atan(frame.height / (2.0 * frame.fy)),
                "aspect": frame.width / frame.height,
            }
            for frame in frames
        ],
    }
    write_json(output_root / "nerfstudio_camera_path.json", nerfstudio_payload)


def render_nerfstudio_video(
    method: str,
    cfg: Config,
    artifacts: RunArtifacts,
    frames: list[FramePose],
) -> dict[str, Any]:
    load_config = artifacts.nerf_config if method == "nerf" else artifacts.neus_config
    method_root = artifacts.output_root / method
    renders_dir = method_root / "renders"
    video_path = method_root / f"{method}.mp4"
    if cfg.overwrite and method_root.exists():
        shutil.rmtree(method_root)
    method_root.mkdir(parents=True, exist_ok=True)

    render_cmd = [
        "ns-render",
        "camera-path",
        "--load-config",
        str(load_config),
        "--camera-path-filename",
        str(artifacts.output_root / "nerfstudio_camera_path.json"),
        "--output-path",
        str(renders_dir),
        "--rendered-output-names",
        "rgb",
    ]
    render_started = perf_counter()
    run_cmd(render_cmd)
    render_time_sec = perf_counter() - render_started

    direct_video_path = renders_dir.with_suffix(".mp4")
    if direct_video_path.is_file():
        if video_path.exists():
            video_path.unlink()
        shutil.move(str(direct_video_path), str(video_path))
        if not cfg.keep_frames and renders_dir.exists():
            shutil.rmtree(renders_dir, ignore_errors=True)
        return build_summary_row(method, cfg, frames, artifacts, video_path, render_time_sec, 0.0)

    frame_dir = resolve_nerfstudio_frame_dir(renders_dir)
    encode_cmd = ffmpeg_encode_command(cfg, frame_dir, video_path)
    encode_started = perf_counter()
    run_cmd(encode_cmd)
    encode_time_sec = perf_counter() - encode_started

    if not cfg.keep_frames:
        shutil.rmtree(renders_dir, ignore_errors=True)

    return build_summary_row(method, cfg, frames, artifacts, video_path, render_time_sec, encode_time_sec)


def resolve_nerfstudio_frame_dir(renders_dir: Path) -> Path:
    rgb_dir = renders_dir / "rgb"
    if rgb_dir.is_dir():
        return rgb_dir
    test_rgb_dir = renders_dir / "test" / "rgb"
    if test_rgb_dir.is_dir():
        return test_rgb_dir
    pngs = list(renders_dir.rglob("*.png"))
    if pngs:
        return pngs[0].parent
    raise CameraPathVideoError(f"Could not locate rendered PNG frames under {renders_dir}")


def render_3dgs_video(
    cfg: Config,
    artifacts: RunArtifacts,
    frames: list[FramePose],
) -> dict[str, Any]:
    method = "3dgs"
    method_root = artifacts.output_root / method
    frames_dir = method_root / "frames"
    video_path = method_root / f"{method}.mp4"
    if cfg.overwrite and method_root.exists():
        shutil.rmtree(method_root)
    frames_dir.mkdir(parents=True, exist_ok=True)

    render_started = perf_counter()
    render_3dgs_frames(cfg, artifacts, frames, frames_dir)
    render_time_sec = perf_counter() - render_started

    encode_cmd = ffmpeg_encode_command(cfg, frames_dir, video_path)
    encode_started = perf_counter()
    run_cmd(encode_cmd)
    encode_time_sec = perf_counter() - encode_started

    if not cfg.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    return build_summary_row(method, cfg, frames, artifacts, video_path, render_time_sec, encode_time_sec)


def render_3dgs_frames(
    cfg: Config,
    artifacts: RunArtifacts,
    frames: list[FramePose],
    frames_dir: Path,
) -> None:
    repo_dir = cfg.dgs_repo
    if not repo_dir.is_dir():
        raise CameraPathVideoError(f"3DGS repository not found: {repo_dir}")
    sys.path.insert(0, str(repo_dir))
    try:
        import torch  # type: ignore
        import torchvision  # type: ignore
        from arguments import PipelineParams  # type: ignore
        from gaussian_renderer import render as gs_render  # type: ignore
        from scene.cameras import MiniCam  # type: ignore
        from gaussian_renderer import GaussianModel  # type: ignore
        from utils.graphics_utils import getProjectionMatrix, getWorld2View2, focal2fov  # type: ignore
        from utils.system_utils import searchForMaxIteration  # type: ignore
        from argparse import ArgumentParser  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise CameraPathVideoError(
            "Failed to import gaussian-splatting runtime modules for 3DGS rendering"
        ) from exc

    parser = ArgumentParser()
    pipeline_params = PipelineParams(parser)
    pipeline_args = parser.parse_args([])
    pipeline_args.antialiasing = cfg.dgs_antialiasing
    pipeline = pipeline_params.extract(pipeline_args)

    iteration = cfg.dgs_iteration
    if iteration == -1:
        iteration = searchForMaxIteration(str(artifacts.dgs_model_dir / "point_cloud"))
    ply_path = artifacts.dgs_model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.is_file():
        raise CameraPathVideoError(f"3DGS point cloud not found: {ply_path}")

    with torch.no_grad():
        gaussians = GaussianModel(cfg.dgs_sh_degree)
        gaussians.load_ply(str(ply_path), False)
        bg_color = [1, 1, 1] if cfg.dgs_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for frame in frames:
            fovx = focal2fov(frame.fx, frame.width)
            fovy = focal2fov(frame.fy, frame.height)
            c2w = frame.c2w
            world_to_view = camera_to_world_view_transform(c2w, getWorld2View2, torch)
            projection = getProjectionMatrix(
                znear=0.01,
                zfar=100.0,
                fovX=fovx,
                fovY=fovy,
            ).transpose(0, 1).cuda()
            full_proj = world_to_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
            view = MiniCam(
                frame.width,
                frame.height,
                fovy,
                fovx,
                0.01,
                100.0,
                world_to_view,
                full_proj,
            )
            rendering = gs_render(
                view,
                gaussians,
                pipeline,
                background,
                separate_sh=False,
                use_trained_exp=False,
            )["render"]
            torchvision.utils.save_image(rendering, frames_dir / f"{frame.index:06d}.png")


def camera_to_world_view_transform(c2w: np.ndarray, get_world2view2, torch_module):
    w2c = np.linalg.inv(c2w)
    # Graphdeco stores Camera.R transposed w.r.t. the actual world-to-camera rotation.
    stored_r = w2c[:3, :3].T
    stored_t = w2c[:3, 3]
    return torch_module.tensor(get_world2view2(stored_r, stored_t)).transpose(0, 1).cuda()


def ffmpeg_encode_command(cfg: Config, frames_dir: Path, output_path: Path) -> list[str]:
    return [
        cfg.ffmpeg_bin,
        "-y" if cfg.overwrite else "-n",
        "-framerate",
        str(cfg.fps),
        "-i",
        str(frames_dir / "%06d.png"),
        "-c:v",
        cfg.video_codec,
        "-pix_fmt",
        cfg.video_pix_fmt,
        str(output_path),
    ]


def build_summary_row(
    method: str,
    cfg: Config,
    frames: list[FramePose],
    artifacts: RunArtifacts,
    video_path: Path,
    render_time_sec: float,
    encode_time_sec: float,
) -> dict[str, Any]:
    frame_count = len(frames)
    return {
        "method": method,
        "frame_count": frame_count,
        "target_fps": cfg.fps,
        "render_time_sec": render_time_sec,
        "encode_time_sec": encode_time_sec,
        "total_time_sec": render_time_sec + encode_time_sec,
        "effective_render_fps": frame_count / max(render_time_sec, 1e-9),
        "width": artifacts.width,
        "height": artifacts.height,
        "video_path": str(video_path),
    }


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    json_path = output_root / "timing_summary.json"
    csv_path = output_root / "timing_summary.csv"
    write_json(json_path, rows)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise CameraPathVideoError(
            f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
