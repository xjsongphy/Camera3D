from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab3.common import Lab3Error, build_run_dir, read_json, write_json
from lab3.extract import ExtractionConfig, prepare_dataset
from lab3.reconstruction import (
    DGSConfig,
    DGSReconstructor,
    NeRFConfig,
    NeRFReconstructor,
    ReconstructionContext,
    SfMConfig,
    SfMReconstructor,
)


@dataclass(frozen=True)
class Lab3PipelineConfig:
    input_dir: Path
    scene_name: str
    output_root: Path = Path("outputs/lab3")
    methods: tuple[str, ...] = ("sfm", "3dgs", "nerf")
    fps: float = 2.0
    test_ratio: float = 0.1
    image_limit: int | None = None
    ffmpeg_bin: str = "ffmpeg"
    force: bool = False
    dry_run: bool = False
    timestamp: str | None = None
    sfm: SfMConfig = field(default_factory=SfMConfig)
    dgs: DGSConfig = field(default_factory=DGSConfig)
    nerf: NeRFConfig = field(default_factory=NeRFConfig)


def run_pipeline(cfg: Lab3PipelineConfig) -> Path:
    run_dir = build_run_dir(cfg.output_root, cfg.scene_name, cfg.timestamp)
    prepared_dir = run_dir / "prepared"
    configs_dir = run_dir / "configs"
    results_dir = run_dir / "results"

    if not cfg.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        configs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    write_json(configs_dir / "run_config.json", cfg)

    prepared = prepare_dataset(
        ExtractionConfig(
            input_dir=cfg.input_dir,
            output_dir=prepared_dir,
            fps=cfg.fps,
            image_limit=cfg.image_limit,
            test_ratio=cfg.test_ratio,
            ffmpeg_bin=cfg.ffmpeg_bin,
            force=cfg.force,
            dry_run=cfg.dry_run,
        )
    )
    write_json(configs_dir / "prepared_dataset.json", prepared)

    timings: dict[str, float] = {}
    for method in cfg.methods:
        normalized = normalize_method(method)
        reconstructor = build_reconstructor(normalized, cfg)
        method_dir = results_dir / normalized
        context = ReconstructionContext(
            run_dir=run_dir,
            prepared_dir=prepared.root,
            images_dir=prepared.images_dir,
            output_dir=method_dir,
            config_dir=configs_dir,
            dry_run=cfg.dry_run,
            force=cfg.force,
            timings=timings,
        )
        reconstructor.run(context)

    write_json(run_dir / "timings.json", timings)
    return run_dir


def normalize_method(method: str) -> str:
    value = method.strip().lower()
    aliases = {"dgs": "3dgs", "gaussian": "3dgs", "gaussian-splatting": "3dgs"}
    value = aliases.get(value, value)
    if value not in {"sfm", "3dgs", "nerf"}:
        raise Lab3Error(f"Unsupported method: {method}. Choose from sfm, 3dgs, nerf.")
    return value


def build_reconstructor(method: str, cfg: Lab3PipelineConfig):
    if method == "sfm":
        return SfMReconstructor(cfg.sfm)
    if method == "3dgs":
        return DGSReconstructor(cfg.dgs)
    if method == "nerf":
        return NeRFReconstructor(cfg.nerf)
    raise Lab3Error(f"Unsupported method: {method}")


def load_pipeline_config(path: Path, overrides: dict[str, Any] | None = None) -> Lab3PipelineConfig:
    data = read_json(path)
    if overrides:
        data.update({key: value for key, value in overrides.items() if value is not None})
    return config_from_dict(data)


def config_from_dict(data: dict[str, Any]) -> Lab3PipelineConfig:
    reconstruction = data.get("reconstruction", {})
    if reconstruction is None:
        reconstruction = {}
    if not isinstance(reconstruction, dict):
        raise Lab3Error("config.reconstruction must be an object")

    sfm_data = _dict_value(reconstruction, "sfm")
    dgs_data = _dict_value(reconstruction, "3dgs")
    nerf_data = _dict_value(reconstruction, "nerf")

    input_dir = data.get("input_dir")
    scene_name = data.get("scene_name")
    if input_dir is None:
        raise Lab3Error("config.input_dir is required")
    if scene_name is None:
        scene_name = Path(str(input_dir)).name

    methods = tuple(data.get("methods", ("sfm", "3dgs", "nerf")))
    return Lab3PipelineConfig(
        input_dir=Path(str(input_dir)),
        scene_name=str(scene_name),
        output_root=Path(str(data.get("output_root", "outputs/lab3"))),
        methods=tuple(normalize_method(str(method)) for method in methods),
        fps=float(data.get("fps", 2.0)),
        test_ratio=float(data.get("test_ratio", 0.1)),
        image_limit=_optional_int(data.get("image_limit")),
        ffmpeg_bin=str(data.get("ffmpeg_bin", "ffmpeg")),
        force=bool(data.get("force", False)),
        dry_run=bool(data.get("dry_run", False)),
        timestamp=data.get("timestamp"),
        sfm=SfMConfig(
            colmap_bin=str(sfm_data.get("colmap_bin", "colmap")),
            matcher=str(sfm_data.get("matcher", "sequential")),
            camera_model=str(sfm_data.get("camera_model", "PINHOLE")),
            single_camera=bool(sfm_data.get("single_camera", True)),
            dense=bool(sfm_data.get("dense", False)),
        ),
        dgs=DGSConfig(
            repo_dir=None if dgs_data.get("repo_dir") in (None, "") else Path(str(dgs_data["repo_dir"])),
            python_bin=str(dgs_data.get("python_bin", "python")),
            iterations=_optional_int(dgs_data.get("iterations", 7000)),
            resolution=_optional_int(dgs_data.get("resolution")),
            extra_args=tuple(str(item) for item in dgs_data.get("extra_args", ())),
        ),
        nerf=NeRFConfig(
            process_bin=str(nerf_data.get("process_bin", "ns-process-data")),
            train_bin=str(nerf_data.get("train_bin", "ns-train")),
            method=str(nerf_data.get("method", "nerfacto")),
            max_num_iterations=_optional_int(nerf_data.get("max_num_iterations", 30000)),
            downscale_factor=_optional_int(nerf_data.get("downscale_factor")),
            skip_process_data=bool(nerf_data.get("skip_process_data", False)),
        ),
    )


def _dict_value(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise Lab3Error(f"config.reconstruction.{key} must be an object")
    return value


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
