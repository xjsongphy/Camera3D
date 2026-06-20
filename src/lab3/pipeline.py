from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab3.common import Lab3Error, build_run_dir, read_json, write_json
from lab3.evaluate import EvaluateConfig, evaluate_run
from lab3.extract import ExtractionConfig, prepare_dataset
from lab3.geometry import stage_geometry, write_geometry_metrics
from lab3.qualitative import save_qualitative
from lab3.reconstruction import (
    DatasetSplit,
    RECONSTRUCTIONS,
    ReconstructionContext,
    Reconstructor,
    create_reconstructor,
    default_reconstruction_configs,
    normalize_reconstruction_name,
    parse_reconstruction_configs,
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
    crop_ratio: float = 1.0
    image_size: tuple[int, int] | None = None
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
    eval_size: tuple[int, int] | None = (900, 1600)
    lpips: bool = True
    native_crosscheck: bool = False
    reconstruction: dict[str, Any] = field(default_factory=default_reconstruction_configs)

    def __getattr__(self, name: str) -> Any:
        """Compatibility access; storage remains registry-keyed and extensible."""
        for method, registration in RECONSTRUCTIONS.items():
            if registration.config_attr == name:
                return self.reconstruction[method]
        raise AttributeError(name)


def run_pipeline(cfg: Lab3PipelineConfig) -> Path:
    run_dir = build_run_dir(cfg.output_root, cfg.scene_name, cfg.timestamp)
    prepared_dir = run_dir / "prepared"
    configs_dir = run_dir / "configs"
    results_dir = run_dir / "results"
    # Validate method/pose topology before extraction or any heavy subprocess.
    reconstructors = _build_reconstructors(cfg, prepared_dir)

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
            crop_ratio=cfg.crop_ratio,
            image_size=cfg.image_size,
            test_ratio=cfg.test_ratio,
            ffmpeg_bin=cfg.ffmpeg_bin,
            force=cfg.force,
            dry_run=cfg.dry_run,
        )
    )
    write_json(configs_dir / "prepared_dataset.json", prepared)

    timings: dict[str, float] = {}
    gpu_peaks: dict[str, float] = {}
    split = DatasetSplit(prepared.train_names, prepared.test_names)
    if not cfg.dry_run and (cfg.evaluate or cfg.qualitative) and not split.test:
        raise Lab3Error(
            "Evaluation requires a non-empty canonical test split; increase test_ratio or add more images."
        )

    for reconstructor in reconstructors:
        method_dir = results_dir / reconstructor.name
        shared_colmap = prepared.root if cfg.share_poses and reconstructor.writes_shared_poses else None
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
            native_crosscheck=cfg.native_crosscheck,
        )
        evaluate_run(
            eval_context,
            eval_cfg=eval_cfg,
            reconstructors=reconstructors,
        )

    if cfg.geometry:
        staged = stage_geometry(eval_context, reconstructors)
        write_geometry_metrics(
            eval_context, staged, proxy_method=_geometry_reference(reconstructors)
        )

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
        native_crosscheck=cfg.native_crosscheck,
    )
    evaluate_run(context, eval_cfg, reconstructors)
    if cfg.geometry:
        staged = stage_geometry(context, reconstructors)
        write_geometry_metrics(
            context, staged, proxy_method=_geometry_reference(reconstructors)
        )
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
    has_pose_consumers = any(reconstructor.consumes_shared_poses for reconstructor in reconstructors)
    if cfg.share_poses and has_pose_consumers and not has_pose_provider:
        raise Lab3Error(
            "share_poses=true requires a pose provider (include sfm in methods), or explicitly use "
            "--no-share-poses for independent reconstruction."
        )
    if cfg.share_poses and has_pose_provider:
        reconstructors = [reconstructor.with_shared_poses(shared_dir) for reconstructor in reconstructors]
        reconstructors.sort(key=lambda reconstructor: reconstructor.shared_pose_priority)
    else:
        for reconstructor in reconstructors:
            reconstructor.validate_standalone_poses()
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


def _geometry_reference(reconstructors: list[Reconstructor]) -> str | None:
    return next(
        (reconstructor.name for reconstructor in reconstructors if reconstructor.geometry_reference),
        None,
    )


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
    reconstruction_data = data.get("reconstruction", {})
    if reconstruction_data is None:
        reconstruction_data = {}
    if not isinstance(reconstruction_data, dict):
        raise Lab3Error("config.reconstruction must be an object")
    # Older run snapshots stored method configs at top level. Merge those into
    # the registry-shaped mapping without teaching this module backend schemas.
    merged_reconstruction = dict(reconstruction_data)
    for method, registration in RECONSTRUCTIONS.items():
        if method not in merged_reconstruction:
            legacy = data.get(registration.config_attr, data.get(method))
            if isinstance(legacy, dict):
                merged_reconstruction[method] = legacy

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
        crop_ratio=float(data.get("crop_ratio", 1.0)),
        image_size=_parse_size(data.get("image_size"), "image_size"),
        ffmpeg_bin=str(data.get("ffmpeg_bin", "ffmpeg")),
        force=bool(data.get("force", False)),
        dry_run=bool(data.get("dry_run", False)),
        timestamp=data.get("timestamp"),
        share_poses=bool(data.get("share_poses", True)),
        evaluate=bool(data.get("evaluate", True)),
        geometry=bool(data.get("geometry", True)),
        qualitative=bool(data.get("qualitative", True)),
        eval_size=_parse_eval_size(data.get("eval_size", (900, 1600))),
        lpips=bool(data.get("lpips", True)),
        native_crosscheck=bool(data.get("native_crosscheck", False)),
        reconstruction=parse_reconstruction_configs(merged_reconstruction),
    )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_size(value: Any, field: str) -> tuple[int, int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise Lab3Error(f"{field} must be a [height, width] pair, got {value!r}")


def _parse_eval_size(value: Any) -> tuple[int, int] | None:
    return _parse_size(value, "eval_size")


def _read_test_names(test_list: Path) -> list[str]:
    if not test_list.exists():
        return []
    return [line.strip() for line in test_list.read_text(encoding="utf-8").splitlines() if line.strip()]
