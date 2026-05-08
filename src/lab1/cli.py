from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lab1.logging_utils import build_timestamped_log_path, tee_console_output
from lab1.task1 import Task1Config, Task1Error, run_task1
from lab1.task2 import Task2Config, run_task2

LAB1_ROOT = Path("lab1")
OUTPUT_ROOT = Path("outputs/lab1")

TASK_HELP = {
    "task1": "题目一：静态场景 SfM（S1-1/S1-2/S1-3）",
    "task2": "题目二：子序列分析（S1-2）",
    "task3": "题目三：动态场景 SfM（S2-1/S2-2）",
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
    )
    try:
        return run_task2(cfg)
    except Task1Error as exc:
        print(f"Task2 failed: {exc}")
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera3D Lab1 task runner")
    subparsers = parser.add_subparsers(dest="task", required=True)

    task1_parser = subparsers.add_parser("task1", help=TASK_HELP["task1"])
    task1_parser.add_argument("--fps", type=float, default=2.0, help="frame sampling rate for task1")
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
    task2_parser.set_defaults(handler=run_task2_entry)

    for task in ("task3", "task4"):
        sub = subparsers.add_parser(task, help=TASK_HELP[task])
        sub.set_defaults(handler=lambda _args, task_name=task: _run_placeholder(task_name))

    for alias, target in ALIASES.items():
        if alias.startswith("q"):
            sub = subparsers.add_parser(alias, help=f"alias of {target}")
            if target == "task1":
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
                sub.set_defaults(handler=run_task2_entry)
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
