from __future__ import annotations

import argparse
from pathlib import Path

from lab3.common import Lab3Error
from lab3.pipeline import Lab3PipelineConfig, config_from_dict, load_pipeline_config, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera3D Lab3 reconstruction runner")
    parser.add_argument("--config", type=Path, help="JSON config file. CLI flags override top-level fields.")
    parser.add_argument("--input-dir", type=Path, help="directory containing images and/or videos")
    parser.add_argument("--scene-name", help="scene name used in the timestamped output directory")
    parser.add_argument("--output-root", type=Path, default=None, help="output root, default outputs/lab3")
    parser.add_argument("--methods", nargs="+", choices=["sfm", "3dgs", "dgs", "nerf"], help="methods to run")
    parser.add_argument("--fps", type=float, help="frame extraction fps for videos")
    parser.add_argument("--test-ratio", type=float, help="held-out list ratio written to prepared/test.txt")
    parser.add_argument("--image-limit", type=int, help="optional maximum prepared image count")
    parser.add_argument("--ffmpeg-bin", help="ffmpeg executable")
    parser.add_argument("--colmap-bin", help="COLMAP executable")
    parser.add_argument("--dgs-repo", type=Path, help="GraphDeco gaussian-splatting repository path")
    parser.add_argument("--dgs-iterations", type=int, help="3DGS training iterations")
    parser.add_argument("--nerf-iterations", type=int, help="nerfstudio training iterations")
    parser.add_argument("--timestamp", help="fixed timestamp tag for reproducible output paths")
    parser.add_argument("--force", action="store_true", help="overwrite prepared and method outputs")
    parser.add_argument("--dry-run", action="store_true", help="print commands and write configs where possible")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        cfg = _build_config(args)
        run_dir = run_pipeline(cfg)
    except Lab3Error as exc:
        parser.exit(2, f"lab3 failed: {exc}\n")
    print(f"Lab3 output: {run_dir.resolve()}")


def _build_config(args: argparse.Namespace) -> Lab3PipelineConfig:
    overrides = {
        "input_dir": args.input_dir,
        "scene_name": args.scene_name,
        "output_root": args.output_root,
        "methods": args.methods,
        "fps": args.fps,
        "test_ratio": args.test_ratio,
        "image_limit": args.image_limit,
        "ffmpeg_bin": args.ffmpeg_bin,
        "timestamp": args.timestamp,
        "force": True if args.force else None,
        "dry_run": True if args.dry_run else None,
    }
    if args.config is not None:
        cfg = load_pipeline_config(args.config, overrides)
    else:
        data = {key: value for key, value in overrides.items() if value is not None}
        cfg = config_from_dict(data)

    if args.colmap_bin is not None:
        cfg = Lab3PipelineConfig(**{**cfg.__dict__, "sfm": cfg.sfm.__class__(**{**cfg.sfm.__dict__, "colmap_bin": args.colmap_bin})})
    if args.dgs_repo is not None or args.dgs_iterations is not None:
        cfg = Lab3PipelineConfig(
            **{
                **cfg.__dict__,
                "dgs": cfg.dgs.__class__(
                    **{
                        **cfg.dgs.__dict__,
                        "repo_dir": args.dgs_repo if args.dgs_repo is not None else cfg.dgs.repo_dir,
                        "iterations": args.dgs_iterations if args.dgs_iterations is not None else cfg.dgs.iterations,
                    }
                ),
            }
        )
    if args.nerf_iterations is not None:
        cfg = Lab3PipelineConfig(
            **{
                **cfg.__dict__,
                "nerf": cfg.nerf.__class__(
                    **{**cfg.nerf.__dict__, "max_num_iterations": args.nerf_iterations}
                ),
            }
        )
    return cfg


if __name__ == "__main__":
    main()
