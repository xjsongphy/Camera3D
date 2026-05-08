from __future__ import annotations

import argparse
from pathlib import Path

from lab1.task1 import Task1Config, Task1Error, run_task1

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
        skip_sfm=args.skip_sfm,
        force=args.force,
        dry_run=args.dry_run,
    )
    try:
        return run_task1(cfg)
    except Task1Error as exc:
        print(f"Task1 failed: {exc}")
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera3D Lab1 task runner")
    parser.add_argument(
        "task",
        help="task id: task1|task2|task3|task4 (or alias: q1|q2|q3|q4)",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="frame sampling rate for task1")
    parser.add_argument("--colmap-bin", default="colmap", help="colmap executable name/path for task1")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name/path for task1")
    parser.add_argument("--skip-sfm", action="store_true", help="only extract frames for task1")
    parser.add_argument("--force", action="store_true", help="overwrite previous task1 outputs")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing for task1")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    task = _normalize_task(args.task)

    if task not in TASK_HELP:
        print(f"Unsupported task: {args.task}")
        print("Supported tasks:", ", ".join(TASK_HELP.keys()))
        print("Aliases:", ", ".join(f"{k}->{v}" for k, v in ALIASES.items()))
        raise SystemExit(2)

    if task == "task1":
        raise SystemExit(run_task1_entry(args))

    raise SystemExit(_run_placeholder(task))


if __name__ == "__main__":
    main()
