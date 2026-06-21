"""Shared implementation details for Nerfstudio-backed reconstructors."""

from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Sequence

from lab3.common import Lab3Error, run_cmd, timed_block
from lab3.evaluate import (
    config_iterations,
    metrics_row,
    model_size_mb,
    pair_rendered_views,
    render_fps,
)
from lab3.metrics import compute_image_metrics
from lab3.reconstruction.base import ReconstructionContext


def train_monitor_command(command: list[str], log_path: Path) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "train-monitor",
        "--log-path", str(log_path), "--", *command,
    ]


def scheduled_train_command(command: list[str], save_iterations: tuple[int, ...]) -> list[str]:
    """Run ns-train with explicit checkpoint steps via a local adapter."""
    if not command or Path(command[0]).name != "ns-train":
        raise Lab3Error("Explicit save_iterations currently require train_bin='ns-train'")
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "train-scheduled",
        "--save-iterations",
        *[str(step) for step in save_iterations],
        "--",
        *command[1:],
    ]


def validate_save_iterations(
    max_iterations: int | None, save_iterations: tuple[int, ...]
) -> tuple[int, ...]:
    nodes = tuple(sorted(set(save_iterations)))
    if any(node <= 0 for node in nodes):
        raise Lab3Error("Nerfstudio save_iterations must contain only positive integers")
    if max_iterations is not None and any(node > max_iterations for node in nodes):
        raise Lab3Error(
            f"Nerfstudio save_iterations cannot exceed max_num_iterations={max_iterations}: {nodes}"
        )
    return nodes


def evaluate_nerfstudio(
    method: str,
    context: ReconstructionContext,
    eval_config: Any,
    method_config: Any,
    eval_dir: Path,
    *,
    notes: str,
    config_path: Path | None = None,
    test_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Common evaluator used behind NeRF and NeuS's identical interface."""
    train_bin = method_config.train_bin
    config_path = config_path or latest_config(context.run_dir / "results" / method / "train")
    logs = context.run_dir / "logs"

    if eval_config.native_crosscheck:
        eval_bin = "ns-eval" if train_bin == "ns-train" else train_bin.replace("ns-train", "ns-eval")
        with timed_block(f"{method}_native_eval", context.timings):
            run_cmd(
                [eval_bin, "--load-config", str(config_path or Path("config.yml")),
                 "--output-path", str(eval_dir / f"{method}_eval.json")],
                dry_run=context.dry_run,
                log_path=logs / f"{method}_native_eval.log",
            )

    render_dir = context.run_dir / "results" / method / "renders"
    render_bin = "ns-render" if train_bin == "ns-train" else train_bin.replace("ns-train", "ns-render")
    render_command = [
        render_bin, "dataset", "--load-config", str(config_path or Path("config.yml")),
        "--output-path", str(render_dir), "--split", "test",
        "--rendered-output-names", "rgb",
    ]
    with timed_block(f"{method}_render", context.timings):
        run_cmd(render_command, dry_run=context.dry_run, log_path=logs / f"{method}_render.log")

    pairs = [] if context.dry_run else pair_rendered_views(
        context.images_dir,
        tuple(test_names) if test_names is not None else context.split.test,
        render_dir,
    )
    metrics = (
        compute_image_metrics(
            [ground_truth for ground_truth, _ in pairs],
            [prediction for _, prediction in pairs],
            eval_size=eval_config.eval_size,
            lpips_enabled=eval_config.lpips,
        )
        if pairs else {"psnr": float("nan"), "ssim": float("nan"), "lpips": None, "n": 0}
    )
    return metrics_row(
        method,
        metrics,
        metric_source="lab3.metrics",
        held_out="prepared/test.txt (canonical split)",
        train_time_sec=context.timings.get(f"{method}_train"),
        iterations=config_iterations(method_config, "max_num_iterations"),
        gpu_mem_peak_gb=context.peaks.get(f"{method}_train"),
        render_fps=render_fps(metrics.get("n"), context.timings.get(f"{method}_render", 0.0)),
        model_size_mb=(
            model_size_mb(model_files(context.run_dir / "results" / method))
            if not context.dry_run else float("nan")
        ),
        notes=notes,
    )


def latest_config(train_dir: Path) -> Path | None:
    matches = sorted(train_dir.rglob("config.yml")) if train_dir.exists() else []
    return matches[-1] if matches else None


def model_files(method_dir: Path) -> list[Path]:
    """Return the latest model artifact, excluding report-only snapshots."""
    files: list[Path] = []
    for suffix in ("*.pt", "*.ckpt", "*.safetensors"):
        matches = sorted(method_dir.rglob(suffix))
        if matches:
            files.append(matches[-1])
    return files


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_PREFIX_RE = re.compile(r"^(\d+)\s+\(([\d.]+)%\)$")


class NerfstudioProgressFilter:
    def __init__(self) -> None:
        self.active_inline = False
        self.last_step: int | None = None

    def handle(self, line: str) -> str | None:
        clean = ANSI_RE.sub("", line)
        stripped = clean.strip()
        if not stripped or stripped.startswith("Step (% Done)") or set(stripped) == {"-"}:
            return None
        parts = re.split(r"\s{2,}", stripped)
        match = STEP_PREFIX_RE.match(parts[0]) if len(parts) >= 4 else None
        if match:
            step = int(match.group(1))
            if self.last_step == step:
                return None
            self.last_step = step
            self.active_inline = True
            suffix = f" | test {parts[4]}" if len(parts) >= 5 else ""
            return (
                f"\rNerfstudio train: step {step} ({match.group(2)}%) | "
                f"iter {parts[1]} | ETA {parts[2]} | train {parts[3]}{suffix}"
            )
        prefix = "\n" if self.active_inline else ""
        self.active_inline = False
        return prefix + clean

    def finalize(self) -> str | None:
        if not self.active_inline:
            return None
        self.active_inline = False
        return "\n"


def _run_train_monitor(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not command:
        raise SystemExit("train-monitor requires a command after '--'")
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    progress = NerfstudioProgressFilter()
    tail: deque[str] = deque(maxlen=120)
    with args.log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            log.write(raw_line)
            tail.append(ANSI_RE.sub("", raw_line).rstrip("\n"))
            if output := progress.handle(raw_line):
                _safe_write(output)
        if output := progress.finalize():
            _safe_write(output)
        return_code = process.wait()
    if return_code:
        raise SystemExit(
            f"Command failed with exit code {return_code}: {' '.join(command)}\n"
            + "\n".join(tail)
        )


def _run_scheduled_train(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-iterations", nargs="+", type=int, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not forwarded:
        raise SystemExit("train-scheduled requires ns-train arguments after '--'")

    # Trainer calls its imported step_check for logging, evaluation and saving.
    # A unique sentinel changes only the checkpoint branch; all other schedules
    # retain Nerfstudio's native behavior.
    sentinel = -2_147_483_647
    scheduled = set(args.save_iterations)
    trainer_module, entrypoint = _load_nerfstudio_train_entrypoint()

    native_step_check = trainer_module.step_check
    native_trainer_init = trainer_module.Trainer.__init__

    def explicit_step_check(step: int, step_size: int, run_at_zero: bool = False) -> bool:
        if step_size == sentinel:
            return step in scheduled
        return native_step_check(step, step_size, run_at_zero=run_at_zero)

    trainer_module.step_check = explicit_step_check

    def scheduled_trainer_init(self, config, *init_args, **init_kwargs):
        config.steps_per_save = sentinel
        native_trainer_init(self, config, *init_args, **init_kwargs)

    trainer_module.Trainer.__init__ = scheduled_trainer_init
    sys.argv = ["ns-train", *forwarded]
    entrypoint()


def _load_nerfstudio_train_entrypoint():
    """Import nerfstudio even when this helper file is named ``nerfstudio.py``.

    When Python executes this file directly, its directory is inserted at the
    front of ``sys.path``. A later ``import nerfstudio`` can then resolve back to
    this file instead of the installed package, producing ``'nerfstudio' is not a
    package``. Temporarily remove that directory and clear the shadow module.
    """
    helper_dir = str(Path(__file__).resolve().parent)
    original_path = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(helper_dir)]
        shadow = sys.modules.get("nerfstudio")
        if shadow is not None:
            shadow_file = getattr(shadow, "__file__", None)
            if shadow_file and Path(shadow_file).resolve() == Path(__file__).resolve():
                sys.modules.pop("nerfstudio", None)
        trainer_module = importlib.import_module("nerfstudio.engine.trainer")
        entrypoint = importlib.import_module("nerfstudio.scripts.train").entrypoint
        return trainer_module, entrypoint
    finally:
        sys.path[:] = original_path


def _safe_write(text: str) -> None:
    try:
        print(text, end="")
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode(sys.stdout.encoding or "utf-8", errors="replace"))
        sys.stdout.flush()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: nerfstudio.py {train-monitor|train-scheduled} ...")
    if sys.argv[1] == "train-monitor":
        _run_train_monitor(sys.argv[2:])
    elif sys.argv[1] == "train-scheduled":
        _run_scheduled_train(sys.argv[2:])
    else:
        raise SystemExit("usage: nerfstudio.py {train-monitor|train-scheduled} ...")


if __name__ == "__main__":
    main()
