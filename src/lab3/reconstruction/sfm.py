from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from lab3.common import Lab3Error, require_tool, run_cmd, monitored_block, timed_block
from lab3.reconstruction.base import ReconstructionContext, Reconstructor, ViewerTarget

COLMAP_FEATURE_EXTRACTORS = {"sift", "sift_dsp"}
HLOC_FEATURE_EXTRACTORS = {
    "sift",
    "superpoint_aachen",
    "superpoint_max",
    "superpoint_inloc",
    "disk",
    "aliked-n16",
}
HLOC_MATCHERS = {
    "superpoint+lightglue",
    "disk+lightglue",
    "aliked+lightglue",
    "superglue",
    "superglue-fast",
    "NN-superpoint",
    "NN-ratio",
    "NN-mutual",
    "adalam",
}


@dataclass(frozen=True)
class SfMConfig:
    colmap_bin: str = "colmap"
    matcher: str = "sequential"
    feature_extractor: str = "sift"
    feature_matcher: str | None = None
    python_bin: str = sys.executable
    pair_overlap: int = 10
    quadratic_overlap: bool = True
    camera_model: str = "PINHOLE"
    single_camera: bool = True
    dense: bool = False


def config_from_dict(values: dict) -> SfMConfig:
    return SfMConfig(
        colmap_bin=str(values.get("colmap_bin", "colmap")),
        matcher=str(values.get("matcher", "sequential")),
        feature_extractor=str(values.get("feature_extractor", "sift")),
        feature_matcher=(
            None if values.get("feature_matcher") in (None, "")
            else str(values.get("feature_matcher"))
        ),
        python_bin=(
            sys.executable if values.get("python_bin") in (None, "", "python")
            else str(values.get("python_bin"))
        ),
        pair_overlap=int(values.get("pair_overlap", 10)),
        quadratic_overlap=bool(values.get("quadratic_overlap", True)),
        camera_model=str(values.get("camera_model", "PINHOLE")),
        single_camera=bool(values.get("single_camera", True)),
        dense=bool(values.get("dense", False)),
    )


def add_cli_arguments(parser) -> None:
    parser.add_argument("--colmap-bin", help="COLMAP executable")
    parser.add_argument(
        "--sfm-feature-extractor",
        help="SfM local feature extractor: sift / sift_dsp / superpoint_aachen / disk / aliked-n16 ...",
    )
    parser.add_argument(
        "--sfm-feature-matcher",
        help="SfM local feature matcher: default COLMAP, or HLOC presets such as superpoint+lightglue / disk+lightglue / aliked+lightglue / superglue",
    )
    parser.add_argument("--sfm-python-bin", help="Python executable used for optional HLOC SfM helpers")


def cli_overrides(arguments) -> dict:
    return {
        key: value for key, value in {
            "colmap_bin": arguments.colmap_bin,
            "feature_extractor": arguments.sfm_feature_extractor,
            "feature_matcher": arguments.sfm_feature_matcher,
            "python_bin": arguments.sfm_python_bin,
        }.items() if value is not None
    }


@dataclass(frozen=True)
class SfMReconstructor(Reconstructor):
    config: SfMConfig
    name: str = "sfm"
    shared_pose_priority: int = 0
    writes_shared_poses: bool = True
    geometry_reference: bool = True

    def run(self, context: ReconstructionContext) -> None:
        if self.config.matcher not in {"sequential", "exhaustive"}:
            raise Lab3Error(f"Unsupported COLMAP matcher: {self.config.matcher}")
        if self.config.pair_overlap <= 0:
            raise Lab3Error(f"pair_overlap must be positive, got {self.config.pair_overlap}")
        if self.config.feature_extractor not in (COLMAP_FEATURE_EXTRACTORS | HLOC_FEATURE_EXTRACTORS):
            raise Lab3Error(
                "Unsupported SfM feature_extractor: "
                f"{self.config.feature_extractor}. "
                f"Supported classical presets: {sorted(COLMAP_FEATURE_EXTRACTORS)}; "
                f"supported HLOC presets: {sorted(HLOC_FEATURE_EXTRACTORS)}"
            )
        if self.config.feature_matcher is not None and self.config.feature_matcher not in HLOC_MATCHERS:
            raise Lab3Error(
                "Unsupported SfM feature_matcher: "
                f"{self.config.feature_matcher}. Supported learned presets: {sorted(HLOC_MATCHERS)}"
            )
        if not context.dry_run:
            require_tool(self.config.colmap_bin)
            context.output_dir.mkdir(parents=True, exist_ok=True)

        # When pose sharing is enabled the sparse model is written next to the
        # prepared images (``<shared>/sparse/0``) so 3DGS / nerfstudio consume
        # the exact same camera poses. Otherwise it stays under the sfm result.
        colmap_root = context.shared_colmap_dir or context.output_dir
        db_path = context.output_dir / "database.db"
        sparse_root = colmap_root / "sparse"
        hloc_root = context.output_dir / "hloc"
        if context.force and not context.dry_run:
            if db_path.exists():
                db_path.unlink()
            if sparse_root.exists():
                shutil.rmtree(sparse_root)
            if hloc_root.exists():
                shutil.rmtree(hloc_root)
        if not context.dry_run:
            sparse_root.mkdir(parents=True, exist_ok=True)

        if self._uses_hloc():
            model_dir = self._run_hloc(context, sparse_root)
        else:
            with timed_block("sfm_feature_extractor", context.timings):
                run_cmd(
                    _colmap_feature_extractor_command(self.config, db_path, context.images_dir),
                    dry_run=context.dry_run,
                    log_path=context.run_dir / "logs" / "sfm_feature_extractor.log" if context.run_dir else None,
                )

            matcher_cmd = "sequential_matcher" if self.config.matcher == "sequential" else "exhaustive_matcher"
            with timed_block(f"sfm_{matcher_cmd}", context.timings):
                run_cmd(
                    [self.config.colmap_bin, matcher_cmd, "--database_path", str(db_path)],
                    dry_run=context.dry_run,
                    log_path=context.run_dir / "logs" / f"sfm_{matcher_cmd}.log" if context.run_dir else None,
                )

            with timed_block("sfm_mapper", context.timings):
                run_cmd(
                    [
                        self.config.colmap_bin,
                        "mapper",
                        "--database_path",
                        str(db_path),
                        "--image_path",
                        str(context.images_dir),
                        "--output_path",
                        str(sparse_root),
                    ],
                    dry_run=context.dry_run,
                    log_path=context.run_dir / "logs" / "sfm_mapper.log" if context.run_dir else None,
                )

            if not context.dry_run:
                model_dir = _canonicalize_best_sparse_model(self.config.colmap_bin, sparse_root)
            else:
                model_dir = sparse_root / "0"

        # mapper writes binary cameras.bin/images.bin/points3D.bin (what 3DGS and
        # nerfstudio expect); also export a human-readable TXT copy for debugging.
        with timed_block("sfm_model_converter", context.timings):
            run_cmd(
                [
                    self.config.colmap_bin,
                    "model_converter",
                    "--input_path",
                    str(model_dir),
                    "--output_path",
                    str(model_dir),
                    "--output_type",
                    "TXT",
                ],
                dry_run=context.dry_run,
                log_path=context.run_dir / "logs" / "sfm_model_converter.log" if context.run_dir else None,
            )

        if self.config.dense:
            self._run_dense(context, model_dir)

    def _uses_hloc(self) -> bool:
        return (
            self.config.feature_extractor not in COLMAP_FEATURE_EXTRACTORS
            or self.config.feature_matcher is not None
        )

    def _run_hloc(self, context: ReconstructionContext, sparse_root: Path) -> Path:
        if self.config.feature_extractor not in HLOC_FEATURE_EXTRACTORS:
            raise Lab3Error(
                f"HLOC backend does not expose feature extractor '{self.config.feature_extractor}'"
            )
        feature_matcher = _resolve_hloc_matcher(self.config.feature_extractor, self.config.feature_matcher)
        if feature_matcher not in HLOC_MATCHERS:
            raise Lab3Error(f"HLOC matcher '{feature_matcher}' is not in the supported preset set")
        _validate_hloc_combination(self.config.feature_extractor, feature_matcher)
        helper = Path(__file__).with_name("hloc_bridge.py")
        logs = context.run_dir / "logs" if context.run_dir else None
        work_dir = context.output_dir / "hloc"
        model_dir = sparse_root / "0"
        cmd = [
            self.config.python_bin,
            str(helper),
            "build-sfm",
            "--images",
            str(context.images_dir),
            "--work-dir",
            str(work_dir),
            "--model-dir",
            str(model_dir),
            "--extractor",
            self.config.feature_extractor,
            "--matcher",
            feature_matcher,
            "--pairing",
            self.config.matcher,
            "--pair-overlap",
            str(self.config.pair_overlap),
            "--camera-model",
            self.config.camera_model,
            "--single-camera",
            "1" if self.config.single_camera else "0",
            "--quadratic-overlap",
            "1" if self.config.quadratic_overlap else "0",
        ]
        with timed_block("sfm_feature_extractor", context.timings):
            run_cmd(
                cmd,
                dry_run=context.dry_run,
                log_path=logs / "sfm_hloc.log" if logs else None,
            )
        context.timings.setdefault("sfm_mapper", context.timings["sfm_feature_extractor"])
        context.timings[f"sfm_{self.config.matcher}_matcher"] = context.timings["sfm_feature_extractor"]
        return model_dir

    def _run_dense(self, context: ReconstructionContext, model_dir: Path) -> None:
        dense_dir = context.output_dir / "dense"
        logs = context.run_dir / "logs" if context.run_dir else None
        with timed_block("sfm_image_undistorter", context.timings):
            run_cmd(
                [
                    self.config.colmap_bin,
                    "image_undistorter",
                    "--image_path",
                    str(context.images_dir),
                    "--input_path",
                    str(model_dir),
                    "--output_path",
                    str(dense_dir),
                ],
                dry_run=context.dry_run,
                log_path=logs / "sfm_image_undistorter.log" if logs else None,
            )
        with monitored_block("sfm_patch_match_stereo", context.timings, context.peaks, enabled=not context.dry_run):
            run_cmd(
                [self.config.colmap_bin, "patch_match_stereo", "--workspace_path", str(dense_dir)],
                dry_run=context.dry_run,
                log_path=logs / "sfm_patch_match_stereo.log" if logs else None,
            )
        with timed_block("sfm_stereo_fusion", context.timings):
            run_cmd(
                [
                    self.config.colmap_bin,
                    "stereo_fusion",
                    "--workspace_path",
                    str(dense_dir),
                    "--output_path",
                    str(dense_dir / "fused.ply"),
                ],
                dry_run=context.dry_run,
                log_path=logs / "sfm_stereo_fusion.log" if logs else None,
            )

    def evaluate(self, context: ReconstructionContext, eval_config, eval_dir: Path):
        from lab3.evaluate import geometry_only_row

        return geometry_only_row(
            self.name,
            self.config,
            context.timings,
            context.peaks,
            train_timing_key="sfm_mapper",
            train_peak_key="sfm_patch_match_stereo",
        )

    def stage_geometry(self, context: ReconstructionContext) -> list[Path]:
        from lab3.geometry import copy_geometry, maybe_poisson_mesh

        dense = _find_dense(context.run_dir / "results" / self.name)
        if dense is None:
            return []
        destination = context.run_dir / "geometry" / self.name
        collected = [copy_geometry(dense, destination, "dense.ply")]
        mesh = maybe_poisson_mesh(dense, destination / "poisson_mesh.ply")
        if mesh is not None:
            collected.append(mesh)
        return collected

    def viewer_targets(self, run_dir: Path) -> list[ViewerTarget]:
        candidates = [
            run_dir / "geometry" / self.name / "dense.ply",
            run_dir / "geometry" / self.name / "poisson_mesh.ply",
            run_dir / "results" / self.name / "dense" / "fused.ply",
        ]
        return [ViewerTarget(self.name, "geometry", path) for path in candidates if path.exists()]


def _find_dense(method_dir: Path) -> Path | None:
    fused = method_dir / "dense" / "fused.ply"
    return fused if fused.exists() else None


def _colmap_feature_extractor_command(
    config: SfMConfig, database_path: Path, image_path: Path
) -> list[str]:
    cmd = [
        config.colmap_bin,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.single_camera",
        "1" if config.single_camera else "0",
        "--ImageReader.camera_model",
        config.camera_model,
    ]
    if config.feature_extractor == "sift_dsp":
        cmd.extend(
            [
                "--SiftExtraction.estimate_affine_shape",
                "1",
                "--SiftExtraction.domain_size_pooling",
                "1",
            ]
        )
    return cmd


def _resolve_hloc_matcher(feature_extractor: str, configured_matcher: str | None) -> str:
    if configured_matcher is not None:
        return configured_matcher
    if feature_extractor.startswith("superpoint"):
        return "superpoint+lightglue"
    if feature_extractor == "disk":
        return "disk+lightglue"
    if feature_extractor.startswith("aliked"):
        return "aliked+lightglue"
    if feature_extractor == "sift":
        return "NN-mutual"
    raise Lab3Error(
        f"No default learned matcher is defined for extractor '{feature_extractor}'. "
        "Set reconstruction.sfm.feature_matcher explicitly."
    )


def _validate_hloc_combination(feature_extractor: str, feature_matcher: str) -> None:
    if feature_matcher.startswith("superpoint+") or feature_matcher.startswith("NN-superpoint"):
        if not feature_extractor.startswith("superpoint"):
            raise Lab3Error(
                f"HLOC matcher '{feature_matcher}' expects a SuperPoint-family extractor, got '{feature_extractor}'"
            )
    elif feature_matcher.startswith("disk+"):
        if feature_extractor != "disk":
            raise Lab3Error(
                f"HLOC matcher '{feature_matcher}' expects extractor 'disk', got '{feature_extractor}'"
            )
    elif feature_matcher.startswith("aliked+"):
        if not feature_extractor.startswith("aliked"):
            raise Lab3Error(
                f"HLOC matcher '{feature_matcher}' expects an ALIKED-family extractor, got '{feature_extractor}'"
            )
    elif feature_matcher.startswith("superglue"):
        if not feature_extractor.startswith("superpoint"):
            raise Lab3Error(
                f"HLOC matcher '{feature_matcher}' is typically paired with SuperPoint features, got '{feature_extractor}'"
            )


def _canonicalize_best_sparse_model(colmap_bin: str, sparse_root: Path) -> Path:
    """Pick the largest registered COLMAP model and expose it canonically as sparse/0."""
    candidates = sorted(path for path in sparse_root.iterdir() if path.is_dir())
    if not candidates:
        raise Lab3Error(f"COLMAP mapper produced no sparse models under {sparse_root}")

    scored: list[tuple[int, int, Path]] = []
    for path in candidates:
        registered, points = _analyze_sparse_model(colmap_bin, path)
        scored.append((registered, points, path))
    best_registered, _, best_path = max(scored, key=lambda item: (item[0], item[1], -int(item[2].name)))
    if best_registered <= 0:
        raise Lab3Error(f"COLMAP mapper produced no registered images under {sparse_root}")

    canonical = sparse_root / "0"
    if best_path.resolve() != canonical.resolve():
        replacement = sparse_root / "_best_model_tmp"
        if replacement.exists():
            shutil.rmtree(replacement)
        shutil.copytree(best_path, replacement)
        shutil.rmtree(canonical)
        replacement.rename(canonical)
    return canonical


def _analyze_sparse_model(colmap_bin: str, model_dir: Path) -> tuple[int, int]:
    result = subprocess.run(
        [colmap_bin, "model_analyzer", "--path", str(model_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    registered = _extract_model_stat(result.stdout + result.stderr, "Registered images")
    points = _extract_model_stat(result.stdout + result.stderr, "Points")
    return registered, points


def _extract_model_stat(text: str, label: str) -> int:
    match = re.search(rf"{re.escape(label)}:\s*(\d+)", text)
    if match is None:
        raise Lab3Error(f"Unable to parse '{label}' from COLMAP model_analyzer output")
    return int(match.group(1))
