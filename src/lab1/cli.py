from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lab1.logging_utils import build_timestamped_log_path, tee_console_output
from lab1.task1 import Task1Config, Task1Error, run_task1
from lab1.task2 import Task2Config, run_task2
from lab1.task3 import Task3Config, Task3MaskConfig, run_task3, run_task3_masks
from lab1.task4 import Task4Config, run_task4

LAB1_ROOT = Path("docs/lab1")
OUTPUT_ROOT = Path("outputs/lab1")

TASK_HELP = {
    "task1": "题目一：静态场景 SfM（S1-1/S1-2/S1-3）",
    "task2": "题目二：子序列分析（S1-2）",
    "task3": "题目三：动态场景 SfM（S2-1/S2-2）",
    "task3-mask": "题目三：为动态场景生成 mask（default/motion/yolo）",
    "task4": "题目四：位姿质量评估（annotations 01-10）",
}

ALIASES = {
    "q1": "task1",
    "q2": "task2",
    "q3": "task3",
    "q4": "task4",
}


def _normalize_task(task: str) -> str:
    key = task.lower().strip()
    return ALIASES.get(key, key)


def _ensure_output_dir(task_name: str) -> Path:
    path = OUTPUT_ROOT / task_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_task_for_logging(argv: list[str]) -> str:
    if not argv:
        return "cli"
    first = argv[0].strip()
    if not first or first.startswith("-"):
        return "cli"
    task = _normalize_task(first)
    return task if task in TASK_HELP else "cli"


def _run_placeholder(task: str) -> int:
    out_dir = _ensure_output_dir(task)
    print(f"Running {task}: {TASK_HELP[task]}")
    print(f"Lab materials: {LAB1_ROOT.resolve()}")
    print(f"Output directory: {out_dir.resolve()}")
    print("This task is not implemented yet.")
    return 0


def run_task1_entry(args: argparse.Namespace) -> int:
    out_dir = _ensure_output_dir("task1")
    cfg = Task1Config(
        lab1_root=LAB1_ROOT,
        output_root=out_dir,
        fps=args.fps,
        colmap_bin=args.colmap_bin,
        ffmpeg_bin=args.ffmpeg_bin,
        force=args.force,
        dry_run=args.dry_run,
        videos=args.videos,
        stage=args.stage,
        mode=args.mode,
        direction_arrows=args.direction_arrows,
        max_points_plot=args.max_points_plot,
    )
    try:
        return run_task1(cfg)
    except Task1Error as exc:
        print(f"Task1 failed: {exc}")
        return 2


def run_task2_entry(args: argparse.Namespace) -> int:
    out_dir = _ensure_output_dir("task2")
    cfg = Task2Config(
        lab1_root=LAB1_ROOT,
        task1_output_root=OUTPUT_ROOT / "task1",
        output_root=out_dir,
        source_fps=args.source_fps,
        colmap_bin=args.colmap_bin,
        force=args.force,
        dry_run=args.dry_run,
        stage=args.stage,
        subseq_specs=tuple(args.subseq or ()),
    )
    try:
        return run_task2(cfg)
    except Task1Error as exc:
        print(f"Task2 failed: {exc}")
        return 2


def run_task3_entry(args: argparse.Namespace) -> int:
    out_dir = _ensure_output_dir("task3")
    cfg = Task3Config(
        lab1_root=LAB1_ROOT,
        output_root=out_dir,
        fps=args.fps,
        colmap_bin=args.colmap_bin,
        ffmpeg_bin=args.ffmpeg_bin,
        force=args.force,
        dry_run=args.dry_run,
        stage=args.stage,
        videos=args.videos,
        methods=tuple(args.methods or ("raw",)),
        mask_source=args.mask_source,
        direction_arrows=args.direction_arrows,
        max_points_plot=args.max_points_plot,
    )
    try:
        return run_task3(cfg)
    except Task1Error as exc:
        print(f"Task3 failed: {exc}")
        return 2


def run_task3_masks_entry(args: argparse.Namespace) -> int:
    out_dir = _ensure_output_dir("task3")
    cfg = Task3MaskConfig(
        lab1_root=LAB1_ROOT,
        output_root=out_dir,
        fps=args.fps,
        ffmpeg_bin=args.ffmpeg_bin,
        force=args.force,
        dry_run=args.dry_run,
        videos=args.videos,
        source=args.source,
        motion_threshold=args.motion_threshold,
        motion_dilation=args.motion_dilation,
        model=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        yolo_dilation=args.yolo_dilation,
    )
    try:
        return run_task3_masks(cfg)
    except Task1Error as exc:
        print(f"Task3 mask generation failed: {exc}")
        return 2


def run_task4_entry(args: argparse.Namespace) -> int:
    out_dir = _ensure_output_dir("task4")
    cfg = Task4Config(
        lab1_root=LAB1_ROOT,
        output_root=out_dir,
        force=args.force,
        dry_run=args.dry_run,
        cases=args.cases,
    )
    try:
        return run_task4(cfg)
    except Task1Error as exc:
        print(f"Task4 failed: {exc}")
        return 2


def _add_task3_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fps", type=float, default=5.0, help="frame sampling rate for task3 dynamic videos")
    parser.add_argument("--colmap-bin", default="colmap", help="colmap executable name/path for task3")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name/path for task3")
    parser.add_argument("--force", action="store_true", help="overwrite previous task3 outputs")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing for task3")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "extract", "sfm", "analyze"],
        help="task3 stage control: all (default), extract only, sfm only, or analyze only",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        help="task3 only: choose subset videos, e.g. --videos S2-1 S2-2",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="task3 methods to run: raw mask",
    )
    parser.add_argument(
        "--mask-source",
        default="default",
        choices=["default", "motion", "yolo"],
        help="for --methods mask: read masks from outputs/lab1/task3/masks/<source>",
    )
    parser.add_argument(
        "--direction-arrows",
        type=int,
        default=10,
        help="number of camera-direction arrows to draw on task3 directional trajectory plots",
    )
    parser.add_argument(
        "--max-points-plot",
        type=int,
        default=12000,
        help="maximum number of sparse 3D points to display in point-cloud plots",
    )


def _add_task3_mask_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fps", type=float, default=5.0, help="frame sampling rate shared with task3")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name/path")
    parser.add_argument("--force", action="store_true", help="overwrite previous extracted frames and generated masks")
    parser.add_argument("--dry-run", action="store_true", help="print steps without executing")
    parser.add_argument(
        "--source",
        default="default",
        choices=["default", "motion", "yolo"],
        help="mask generation source",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        help="choose subset videos, e.g. --videos S2-1 S2-2",
    )
    parser.add_argument(
        "--motion-threshold",
        type=int,
        default=28,
        help="for --source motion: grayscale threshold for automatic motion-mask generation",
    )
    parser.add_argument(
        "--motion-dilation",
        type=int,
        default=9,
        help="for --source motion: kernel size used to dilate automatic motion masks",
    )
    parser.add_argument(
        "--model",
        default="models/yolo11s-seg.pt",
        help="for --source yolo: Ultralytics segmentation model checkpoint or local path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="for --source yolo: YOLO confidence threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="for --source yolo: YOLO inference image size",
    )
    parser.add_argument(
        "--yolo-dilation",
        type=int,
        default=7,
        help="for --source yolo: kernel size used to dilate YOLO dynamic masks",
    )


def _add_task4_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--force", action="store_true", help="overwrite previous task4 outputs")
    parser.add_argument("--dry-run", action="store_true", help="print steps without executing for task4")
    parser.add_argument(
        "--cases",
        nargs="+",
        help="optional subset of annotation cases, e.g. --cases 01 02 06",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera3D Lab1 task runner")
    subparsers = parser.add_subparsers(dest="task", required=True)

    task1_parser = subparsers.add_parser("task1", help=TASK_HELP["task1"])
    task1_parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=["run", "merge", "plot", "cloud"],
        help="task1 mode: run (default), merge trajectories, redraw trajectories, or generate sparse point-cloud plot from existing outputs",
    )
    task1_parser.add_argument("--fps", type=float, default=-1.0, help="frame sampling rate for task1 (default: -1 for all available fps)")
    task1_parser.add_argument("--colmap-bin", default="colmap", help="colmap executable name/path for task1")
    task1_parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name/path for task1")
    task1_parser.add_argument("--force", action="store_true", help="overwrite previous task1 outputs")
    task1_parser.add_argument("--dry-run", action="store_true", help="print commands without executing for task1")
    task1_parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "extract", "sfm"],
        help="task1 stage control: all (default), extract only, or sfm only",
    )
    task1_parser.add_argument(
        "--videos",
        nargs="+",
        help="task1 only: choose subset videos, e.g. --videos S1-1 S1-2",
    )
    task1_parser.add_argument(
        "--direction-arrows",
        type=int,
        default=12,
        help="number of camera-direction arrows to draw on the directional trajectory plot",
    )
    task1_parser.add_argument(
        "--max-points-plot",
        type=int,
        default=12000,
        help="maximum number of sparse 3D points to display in point-cloud plots",
    )
    task1_parser.set_defaults(handler=run_task1_entry)

    task2_parser = subparsers.add_parser("task2", help=TASK_HELP["task2"])
    task2_parser.add_argument("--source-fps", type=float, default=4.0, help="use task1 S1-2 results from this fps tag")
    task2_parser.add_argument("--colmap-bin", default="colmap", help="colmap executable name/path for task2")
    task2_parser.add_argument("--force", action="store_true", help="overwrite previous task2 outputs")
    task2_parser.add_argument("--dry-run", action="store_true", help="print commands without executing for task2")
    task2_parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "prepare", "sfm", "analyze"],
        help="task2 stage control: all (default), prepare only, sfm only, or analyze only",
    )
    task2_parser.add_argument(
        "--subseq",
        action="append",
        help="optional custom subsequence in START:END:NAME format (1-based inclusive); repeatable",
    )
    task2_parser.set_defaults(handler=run_task2_entry)

    task3_parser = subparsers.add_parser("task3", help=TASK_HELP["task3"])
    _add_task3_args(task3_parser)
    task3_parser.set_defaults(handler=run_task3_entry)

    task3_mask_parser = subparsers.add_parser("task3-mask", help=TASK_HELP["task3-mask"])
    _add_task3_mask_args(task3_mask_parser)
    task3_mask_parser.set_defaults(handler=run_task3_masks_entry)

    task4_parser = subparsers.add_parser("task4", help=TASK_HELP["task4"])
    _add_task4_args(task4_parser)
    task4_parser.set_defaults(handler=run_task4_entry)

    for alias, target in ALIASES.items():
        if alias.startswith("q"):
            sub = subparsers.add_parser(alias, help=f"alias of {target}")
            if target == "task1":
                sub.add_argument(
                    "mode",
                    nargs="?",
                    default="run",
                    choices=["run", "merge", "plot", "cloud"],
                    help="task1 mode: run (default), merge trajectories, redraw trajectories, or generate sparse point-cloud plot from existing outputs",
                )
                sub.add_argument("--fps", type=float, default=2.0, help="frame sampling rate for task1")
                sub.add_argument("--colmap-bin", default="colmap", help="colmap executable name/path for task1")
                sub.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name/path for task1")
                sub.add_argument("--force", action="store_true", help="overwrite previous task1 outputs")
                sub.add_argument("--dry-run", action="store_true", help="print commands without executing for task1")
                sub.add_argument(
                    "--stage",
                    default="all",
                    choices=["all", "extract", "sfm"],
                    help="task1 stage control: all (default), extract only, or sfm only",
                )
                sub.add_argument(
                    "--videos",
                    nargs="+",
                    help="task1 only: choose subset videos, e.g. --videos S1-1 S1-2",
                )
                sub.add_argument(
                    "--direction-arrows",
                    type=int,
                    default=12,
                    help="number of camera-direction arrows to draw on the directional trajectory plot",
                )
                sub.add_argument(
                    "--max-points-plot",
                    type=int,
                    default=12000,
                    help="maximum number of sparse 3D points to display in point-cloud plots",
                )
                sub.set_defaults(handler=run_task1_entry)
            elif target == "task2":
                sub.add_argument("--source-fps", type=float, default=4.0, help="use task1 S1-2 results from this fps tag")
                sub.add_argument("--colmap-bin", default="colmap", help="colmap executable name/path for task2")
                sub.add_argument("--force", action="store_true", help="overwrite previous task2 outputs")
                sub.add_argument("--dry-run", action="store_true", help="print commands without executing for task2")
                sub.add_argument(
                    "--stage",
                    default="all",
                    choices=["all", "prepare", "sfm", "analyze"],
                    help="task2 stage control: all (default), prepare only, sfm only, or analyze only",
                )
                sub.add_argument(
                    "--subseq",
                    action="append",
                    help="optional custom subsequence in START:END:NAME format (1-based inclusive); repeatable",
                )
                sub.set_defaults(handler=run_task2_entry)
            elif target == "task3":
                _add_task3_args(sub)
                sub.set_defaults(handler=run_task3_entry)
            elif target == "task4":
                _add_task4_args(sub)
                sub.set_defaults(handler=run_task4_entry)
            else:
                sub.set_defaults(handler=lambda _args, task_name=target: _run_placeholder(task_name))
    return parser


def main() -> None:
    task_name = _resolve_task_for_logging(sys.argv[1:])
    log_root = _ensure_output_dir(task_name) / "logs"
    log_path = build_timestamped_log_path(log_root, task_name)

    with tee_console_output(log_path):
        print(f"Log file: {log_path.resolve()}")
        parser = build_parser()
        args = parser.parse_args()
        raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
