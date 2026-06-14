from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from lab3.common import Lab3Error, require_tool, run_cmd, timed_block
from lab3.reconstruction.base import ReconstructionContext


@dataclass(frozen=True)
class SfMConfig:
    colmap_bin: str = "colmap"
    matcher: str = "sequential"
    camera_model: str = "PINHOLE"
    single_camera: bool = True
    dense: bool = False


@dataclass(frozen=True)
class SfMReconstructor:
    config: SfMConfig
    name: str = "sfm"

    def run(self, context: ReconstructionContext) -> None:
        if self.config.matcher not in {"sequential", "exhaustive"}:
            raise Lab3Error(f"Unsupported COLMAP matcher: {self.config.matcher}")
        if not context.dry_run:
            require_tool(self.config.colmap_bin)
            context.output_dir.mkdir(parents=True, exist_ok=True)

        db_path = context.output_dir / "database.db"
        sparse_root = context.output_dir / "sparse"
        if context.force and not context.dry_run:
            if db_path.exists():
                db_path.unlink()
            if sparse_root.exists():
                shutil.rmtree(sparse_root)

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
            )

        matcher_cmd = "sequential_matcher" if self.config.matcher == "sequential" else "exhaustive_matcher"
        with timed_block(f"sfm_{matcher_cmd}", context.timings):
            run_cmd(
                [self.config.colmap_bin, matcher_cmd, "--database_path", str(db_path)],
                dry_run=context.dry_run,
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
            )

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
            )

        if self.config.dense:
            self._run_dense(context, model_dir)

    def _run_dense(self, context: ReconstructionContext, model_dir: Path) -> None:
        dense_dir = context.output_dir / "dense"
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
            )
        with timed_block("sfm_patch_match_stereo", context.timings):
            run_cmd(
                [self.config.colmap_bin, "patch_match_stereo", "--workspace_path", str(dense_dir)],
                dry_run=context.dry_run,
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
            )
