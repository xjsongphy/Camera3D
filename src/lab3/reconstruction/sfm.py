from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from lab3.common import Lab3Error, require_tool, run_cmd, monitored_block, timed_block
from lab3.reconstruction.base import ReconstructionContext, Reconstructor, ViewerTarget


@dataclass(frozen=True)
class SfMConfig:
    colmap_bin: str = "colmap"
    matcher: str = "sequential"
    camera_model: str = "PINHOLE"
    single_camera: bool = True
    dense: bool = False


@dataclass(frozen=True)
class SfMReconstructor(Reconstructor):
    config: SfMConfig
    name: str = "sfm"
    shared_pose_priority: int = 0
    writes_shared_poses: bool = True

    def run(self, context: ReconstructionContext) -> None:
        if self.config.matcher not in {"sequential", "exhaustive"}:
            raise Lab3Error(f"Unsupported COLMAP matcher: {self.config.matcher}")
        if not context.dry_run:
            require_tool(self.config.colmap_bin)
            context.output_dir.mkdir(parents=True, exist_ok=True)

        # When pose sharing is enabled the sparse model is written next to the
        # prepared images (``<shared>/sparse/0``) so 3DGS / nerfstudio consume
        # the exact same camera poses. Otherwise it stays under the sfm result.
        colmap_root = context.shared_colmap_dir or context.output_dir
        db_path = context.output_dir / "database.db"
        sparse_root = colmap_root / "sparse"
        if context.force and not context.dry_run:
            if db_path.exists():
                db_path.unlink()
            if sparse_root.exists():
                shutil.rmtree(sparse_root)
        if not context.dry_run:
            sparse_root.mkdir(parents=True, exist_ok=True)

        with timed_block("sfm_feature_extractor", context.timings):
            run_cmd(
                [
                    self.config.colmap_bin,
                    "feature_extractor",
                    "--database_path",
                    str(db_path),
                    "--image_path",
                    str(context.images_dir),
                    "--ImageReader.single_camera",
                    "1" if self.config.single_camera else "0",
                    "--ImageReader.camera_model",
                    self.config.camera_model,
                ],
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

        # mapper writes binary cameras.bin/images.bin/points3D.bin (what 3DGS and
        # nerfstudio expect); also export a human-readable TXT copy for debugging.
        model_dir = sparse_root / "0"
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
        from lab3.geometry import copy_geometry, find_sfm_dense, maybe_poisson_mesh

        dense = find_sfm_dense(context.run_dir / "results" / self.name)
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
