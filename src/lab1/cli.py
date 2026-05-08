from __future__ import annotations

import argparse
from pathlib import Path

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


def run_task(task_name: str) -> int:
    task = _normalize_task(task_name)
    if task not in TASK_HELP:
        print(f"Unsupported task: {task_name}")
        print("Supported tasks:", ", ".join(TASK_HELP.keys()))
        print("Aliases:", ", ".join(f"{k}->{v}" for k, v in ALIASES.items()))
        return 2

    out_dir = _ensure_output_dir(task)
    print(f"Running {task}: {TASK_HELP[task]}")
    print(f"Lab materials: {LAB1_ROOT.resolve()}")
    print(f"Output directory: {out_dir.resolve()}")

    if task == "task1":
        print("Next step: extract frames from S1 videos and run SfM sparse reconstruction.")
    elif task == "task2":
        print("Next step: select 3 sub-sequences from S1-2 and compare full-vs-subset SfM trajectories.")
    elif task == "task3":
        print("Next step: run dynamic-scene baseline and try a masking/improvement strategy.")
    elif task == "task4":
        print("Next step: compute no-GT pose quality metrics for annotations/01~10.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera3D Lab1 task runner")
    parser.add_argument(
        "task",
        help="task id: task1|task2|task3|task4 (or alias: q1|q2|q3|q4)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(run_task(args.task))


if __name__ == "__main__":
    main()
