from __future__ import annotations

import argparse
from pathlib import Path

from lab3.common import Lab3Error
from lab3.pipeline import (
    Lab3PipelineConfig,
    config_from_dict,
    evaluate_run_dir,
    load_pipeline_config,
    normalize_method,
    run_pipeline,
)
from lab3.reconstruction import (
    METHOD_ALIASES,
    RECONSTRUCTIONS,
    add_reconstruction_cli_arguments,
    apply_reconstruction_cli_overrides,
)
from lab3.visualization import view_run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera3D Lab3 reconstruction runner")
    parser.add_argument("--config", type=Path, help="JSON config file. CLI flags override top-level fields.")
    parser.add_argument("--input-dir", type=Path, help="directory containing images and/or videos")
    parser.add_argument("--scene-name", help="scene name used in the output directory name")
    parser.add_argument("--output-root", type=Path, default=None, help="output root, default outputs/lab3")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[*RECONSTRUCTIONS, *METHOD_ALIASES],
        help="methods to run",
    )
    parser.add_argument("--fps", type=float, help="frame extraction fps for videos")
    parser.add_argument("--test-ratio", type=float, help="held-out list ratio written to prepared/test.txt")
    parser.add_argument("--image-limit", type=int, help="optional maximum prepared image count")
    parser.add_argument(
        "--blur-threshold",
        type=float,
        help="drop images/frames whose Laplacian-variance sharpness score falls below this threshold",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        help="center-crop this fraction of both image axes before reconstruction",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="resize prepared images after cropping",
    )
    parser.add_argument("--ffmpeg-bin", help="ffmpeg executable")
    add_reconstruction_cli_arguments(parser)
    parser.add_argument("--timestamp", help="fixed timestamp tag for reproducible output paths")
    parser.add_argument("--force", action="store_true", help="overwrite prepared and method outputs")
    parser.add_argument("--dry-run", action="store_true", help="print commands and write configs where possible")
    # Pose fairness + post-training stages.
    parser.add_argument(
        "--share-poses",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run COLMAP once and share sparse/0 with 3DGS + nerfstudio (default: on)",
    )
    parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="render held-out views and write metrics.csv (default: on)",
    )
    parser.add_argument(
        "--geometry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="stage point cloud/mesh artifacts under geometry/ (default: on)",
    )
    parser.add_argument(
        "--qualitative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="build GT-vs-method-vs-error comparison figures (default: on)",
    )
    parser.add_argument(
        "--lpips",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="compute LPIPS when the lpips package is available (default: on)",
    )
    parser.add_argument(
        "--native-crosscheck",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="also run framework-native metrics (slower; default: off)",
    )
    parser.add_argument(
        "--eval-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="resample all eval images to this size for fair metric comparison",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="skip training and re-run evaluate/geometry on an existing run directory",
    )
    parser.add_argument(
        "--view-run",
        type=Path,
        default=None,
        help="open interactive viewers (Open3D geometry + nerfstudio ns-viewer) for an existing run, then exit",
    )
    parser.add_argument(
        "--no-nerfstudio-viewer",
        action="store_true",
        help="skip launching the nerfstudio web viewer (Open3D geometry only)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.view_run is not None:
            methods = (
                tuple(normalize_method(m) for m in args.methods)
                if args.methods
                else tuple(RECONSTRUCTIONS)
            )
            view_run_dir(
                args.view_run.resolve(),
                methods,
                nerfstudio_viewer=not args.no_nerfstudio_viewer,
            )
            print(f"Opened viewers for: {args.view_run.resolve()}")
            return
        if args.run_dir is not None:
            run_dir = evaluate_run_dir(args.run_dir.resolve())
        else:
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
        "blur_threshold": args.blur_threshold,
        "crop_ratio": args.crop_ratio,
        "image_size": tuple(args.image_size) if args.image_size else None,
        "ffmpeg_bin": args.ffmpeg_bin,
        "timestamp": args.timestamp,
        "force": True if args.force else None,
        "dry_run": True if args.dry_run else None,
        "share_poses": args.share_poses,
        "evaluate": args.evaluate,
        "geometry": args.geometry,
        "qualitative": args.qualitative,
        "lpips": args.lpips,
        "native_crosscheck": args.native_crosscheck,
        "eval_size": tuple(args.eval_size) if args.eval_size else None,
    }
    if args.config is not None:
        cfg = load_pipeline_config(args.config, overrides)
    else:
        data = {key: value for key, value in overrides.items() if value is not None}
        cfg = config_from_dict(data)

    return apply_reconstruction_cli_overrides(cfg, args)


if __name__ == "__main__":
    main()
