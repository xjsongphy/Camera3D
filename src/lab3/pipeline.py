from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab3.common import Lab3Error, build_run_dir, read_json, write_json
from lab3.evaluate import EvaluateConfig, evaluate_run
from lab3.extract import ExtractionConfig, prepare_dataset
from lab3.geometry import stage_geometry, write_geometry_metrics
from lab3.qualitative import save_qualitative
from lab3.reconstruction import (
    DGSConfig,
    DatasetSplit,
    NeRFConfig,
    NeuSConfig,
    RECONSTRUCTIONS,
    ReconstructionContext,
    Reconstructor,
    SfMConfig,
    create_reconstructor,
    normalize_reconstruction_name,
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
    blur_threshold: float | None = None
    ffmpeg_bin: str = "ffmpeg"
    force: bool = False
    dry_run: bool = False
    timestamp: str | None = None
    # Pose fairness: run COLMAP once (via SfM) and share ``sparse/0`` with 3DGS
    # and nerfstudio so all methods use identical camera poses.
    share_poses: bool = True
    # Post-training stages (all dry-run aware).
    evaluate: bool = True
    geometry: bool = True
    qualitative: bool = True
    eval_size: tuple[int, int] | None = None
    lpips: bool = True
    sfm: SfMConfig = field(default_factory=SfMConfig)
    dgs: DGSConfig = field(default_factory=DGSConfig)
    nerf: NeRFConfig = field(default_factory=NeRFConfig)
    neus: NeuSConfig = field(default_factory=NeuSConfig)


def run_pipeline(cfg: Lab3PipelineConfig) -> Path:
    run_dir = build_run_dir(cfg.output_root, cfg.scene_name, cfg.timestamp)
    prepared_dir = run_dir / "prepared"
    configs_dir = run_dir / "configs"
    results_dir = run_dir / "results"

    if not cfg.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        configs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    write_json(configs_dir / "run_config.json", cfg)

    prepared = prepare_dataset(
        ExtractionConfig(
            input_dir=cfg.input_dir,
            output_dir=prepared_dir,
            fps=cfg.fps,
            image_limit=cfg.image_limit,
            blur_threshold=cfg.blur_threshold,
            test_ratio=cfg.test_ratio,
            ffmpeg_bin=cfg.ffmpeg_bin,
            force=cfg.force,
            dry_run=cfg.dry_run,
        )
    )
    write_json(configs_dir / "prepared_dataset.json", prepared)

    timings: dict[str, float] = {}
    gpu_peaks: dict[str, float] = {}
    shared_dir = prepared.root
    reconstructors = _build_reconstructors(cfg, shared_dir)
    split = DatasetSplit.from_files(prepared.train_list, prepared.test_list)

    for reconstructor in reconstructors:
        method_dir = results_dir / reconstructor.name
        shared_colmap = shared_dir if cfg.share_poses and reconstructor.writes_shared_poses else None
        context = ReconstructionContext(
            run_dir=run_dir,
            prepared_dir=prepared.root,
            images_dir=prepared.images_dir,
            output_dir=method_dir,
            config_dir=configs_dir,
            dry_run=cfg.dry_run,
            force=cfg.force,
            timings=timings,
            peaks=gpu_peaks,
            shared_colmap_dir=shared_colmap,
            split=split,
        )
        reconstructor.run(context)

    eval_context = ReconstructionContext(
        run_dir=run_dir,
        prepared_dir=prepared.root,
        images_dir=prepared.images_dir,
        output_dir=results_dir,
        config_dir=configs_dir,
        dry_run=cfg.dry_run,
        force=cfg.force,
        timings=timings,
        peaks=gpu_peaks,
        split=split,
    )

    if cfg.evaluate:
        eval_cfg = EvaluateConfig(
            enabled=True,
            eval_size=cfg.eval_size,
            lpips=cfg.lpips,
        )
        evaluate_run(
            eval_context,
            eval_cfg=eval_cfg,
            reconstructors=reconstructors,
        )

    if cfg.geometry:
        staged = stage_geometry(eval_context, reconstructors)
        write_geometry_metrics(eval_context, staged)

    if cfg.qualitative and not cfg.dry_run:
        method_render_dirs = _qualitative_render_dirs(reconstructors, run_dir)
        test_names = _read_test_names(prepared.test_list)
        save_qualitative(
            eval_context,
            None,
            prepared_images_dir=prepared.images_dir,
            test_names=test_names,
            method_render_dirs=method_render_dirs,
            eval_size=cfg.eval_size,
        )

    write_json(run_dir / "timings.json", timings)
    return run_dir


def evaluate_run_dir(run_dir: Path, cfg_overrides: dict[str, Any] | None = None) -> Path:
    """Re-run only the evaluate/geometry/qualitative stages on an existing run."""
    if not run_dir.exists():
        raise Lab3Error(f"Run directory not found: {run_dir}")
    cfg = load_pipeline_config(run_dir / "configs" / "run_config.json", cfg_overrides)
    prepared_root = run_dir / "prepared"
    test_list = prepared_root / "test.txt"
    timings_path = run_dir / "timings.json"
    timings = read_json(timings_path) if timings_path.exists() else {}

    reconstructors = _build_reconstructors(cfg, prepared_root)
    split = DatasetSplit.from_files(prepared_root / "train.txt", prepared_root / "test.txt")
    context = ReconstructionContext(
        run_dir=run_dir,
        prepared_dir=prepared_root,
        images_dir=prepared_root / "images",
        output_dir=run_dir / "results",
        config_dir=run_dir / "configs",
        dry_run=False,
        timings=timings,
        split=split,
    )
    eval_cfg = EvaluateConfig(
        enabled=True,
        eval_size=cfg.eval_size,
        lpips=cfg.lpips,
    )
    evaluate_run(context, eval_cfg, reconstructors)
    if cfg.geometry:
        staged = stage_geometry(context, reconstructors)
        write_geometry_metrics(context, staged)
    if cfg.qualitative:
        method_render_dirs = _qualitative_render_dirs(reconstructors, run_dir)
        test_names = _read_test_names(test_list)
        save_qualitative(
            context,
            None,
            prepared_images_dir=prepared_root / "images",
            test_names=test_names,
            method_render_dirs=method_render_dirs,
            eval_size=cfg.eval_size,
        )
    write_json(timings_path, timings)
    return run_dir


def _build_reconstructors(cfg: Lab3PipelineConfig, shared_dir: Path) -> list[Reconstructor]:
    reconstructors = [create_reconstructor(method, cfg) for method in cfg.methods]
    has_pose_provider = any(reconstructor.writes_shared_poses for reconstructor in reconstructors)
    if cfg.share_poses and has_pose_provider:
        reconstructors = [reconstructor.with_shared_poses(shared_dir) for reconstructor in reconstructors]
        reconstructors.sort(key=lambda reconstructor: reconstructor.shared_pose_priority)
    return reconstructors


def _qualitative_render_dirs(
    reconstructors: list[Reconstructor], run_dir: Path
) -> dict[str, Path]:
    directories: dict[str, Path] = {}
    for reconstructor in reconstructors:
        directory = reconstructor.qualitative_render_dir(run_dir)
        if directory is not None:
            directories[reconstructor.name] = directory
    return directories


def normalize_method(method: str) -> str:
    value = method.strip().lower()
    value = normalize_reconstruction_name(value)
    if value not in RECONSTRUCTIONS:
        choices = ", ".join(RECONSTRUCTIONS)
        raise Lab3Error(f"Unsupported method: {method}. Choose from {choices}.")
    return value


# Kept for backwards compatibility with tests/external callers.
def build_reconstructor(method: str, cfg: Lab3PipelineConfig):
    return create_reconstructor(method, cfg)


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

    # Support both the user-facing nested config schema and the run_config.json
    # snapshots written by this pipeline, which store method configs at top level.
    sfm_source = _coalesce_config_section(data, reconstruction, "sfm")
    dgs_source = _coalesce_config_section(data, reconstruction, "3dgs")
    nerf_source = _coalesce_config_section(data, reconstruction, "nerf")
    neus_source = _coalesce_config_section(data, reconstruction, "neus")
    sfm_data = _dict_value(sfm_source)
    dgs_data = _dict_value(dgs_source)
    nerf_data = _dict_value(nerf_source)
    neus_data = _dict_value(neus_source)

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
        blur_threshold=_optional_float(data.get("blur_threshold")),
        ffmpeg_bin=str(data.get("ffmpeg_bin", "ffmpeg")),
        force=bool(data.get("force", False)),
        dry_run=bool(data.get("dry_run", False)),
        timestamp=data.get("timestamp"),
        share_poses=bool(data.get("share_poses", True)),
        evaluate=bool(data.get("evaluate", True)),
        geometry=bool(data.get("geometry", True)),
        qualitative=bool(data.get("qualitative", True)),
        eval_size=_parse_eval_size(data.get("eval_size")),
        lpips=bool(data.get("lpips", True)),
        sfm=SfMConfig(
            colmap_bin=str(sfm_data.get("colmap_bin", "colmap")),
            matcher=str(sfm_data.get("matcher", "sequential")),
            camera_model=str(sfm_data.get("camera_model", "PINHOLE")),
            single_camera=bool(sfm_data.get("single_camera", True)),
            dense=bool(sfm_data.get("dense", False)),
        ),
        dgs=DGSConfig(
            repo_dir=(
                Path("gaussian-splatting")
                if dgs_data.get("repo_dir") in (None, "")
                else Path(str(dgs_data["repo_dir"]))
            ),
            python_bin=(
                sys.executable
                if dgs_data.get("python_bin") in (None, "", "python")
                else str(dgs_data.get("python_bin"))
            ),
            iterations=_optional_int(dgs_data.get("iterations", 7000)),
            save_every=_optional_int(dgs_data.get("save_every", 2000)),
            resolution=_optional_int(dgs_data.get("resolution")),
            extra_args=tuple(str(item) for item in dgs_data.get("extra_args", ())),
            colmap_source=None
            if dgs_data.get("colmap_source") in (None, "")
            else Path(str(dgs_data["colmap_source"])),
            colmap_bin=str(dgs_data.get("colmap_bin", "colmap")),
            eval_split=bool(dgs_data.get("eval_split", True)),
        ),
        nerf=NeRFConfig(
            process_bin=str(nerf_data.get("process_bin", "ns-process-data")),
            train_bin=str(nerf_data.get("train_bin", "ns-train")),
            method=str(nerf_data.get("method", "nerfacto")),
            max_num_iterations=_optional_int(nerf_data.get("max_num_iterations", 30000)),
            save_every=_optional_int(nerf_data.get("save_every", 2000)),
            save_only_latest_checkpoint=bool(
                nerf_data.get("save_only_latest_checkpoint", False)
            ),
            downscale_factor=_optional_int(nerf_data.get("downscale_factor")),
            skip_process_data=bool(nerf_data.get("skip_process_data", False)),
            colmap_model=None
            if nerf_data.get("colmap_model") in (None, "")
            else Path(str(nerf_data["colmap_model"])),
        ),
        neus=NeuSConfig(
            train_bin=str(neus_data.get("train_bin", "ns-train")),
            method=str(neus_data.get("method", "neus-facto")),
            max_num_iterations=_optional_int(neus_data.get("max_num_iterations", 20001)),
            save_every=_optional_int(neus_data.get("save_every", 2000)),
            save_only_latest_checkpoint=bool(
                neus_data.get("save_only_latest_checkpoint", False)
            ),
            colmap_model=None
            if neus_data.get("colmap_model") in (None, "")
            else Path(str(neus_data["colmap_model"])),
            scene_scale=float(neus_data.get("scene_scale", 2.0)),
            export_mesh=bool(neus_data.get("export_mesh", True)),
            mesh_resolution=int(neus_data.get("mesh_resolution", 512)),
        ),
    )


def _dict_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise Lab3Error("method config must be an object")
    return value


def _coalesce_config_section(
    top_level: dict[str, Any], reconstruction: dict[str, Any], key: str
) -> Any:
    if key in reconstruction:
        return reconstruction.get(key)
    if key == "3dgs" and "dgs" in reconstruction:
        return reconstruction.get("dgs")
    if key == "3dgs" and "dgs" in top_level:
        return top_level.get("dgs")
    return top_level.get(key)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_eval_size(value: Any) -> tuple[int, int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise Lab3Error(f"eval_size must be a [height, width] pair, got {value!r}")


def _read_test_names(test_list: Path) -> list[str]:
    if not test_list.exists():
        return []
    return [line.strip() for line in test_list.read_text(encoding="utf-8").splitlines() if line.strip()]
