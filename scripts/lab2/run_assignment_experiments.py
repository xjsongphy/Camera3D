from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab2.OpticalSGD import initialize_patterns
from lab2.common import auto_detect_device, load_config, set_random_seed
from lab2.scene_genertor import SCENE_PRESETS, create_standard_renderer

DEFAULT_ASSIGNMENT_CONFIG = Path("configs/lab2/assignment_plan.yaml")


def run_command(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=cwd, check=True)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_training_log(path: Path) -> tuple[list[int], list[float]]:
    rows: list[tuple[int, float]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["iteration"]), float(row["loss"])))
    if not rows:
        return [], []
    iterations, losses = zip(*rows)
    return list(iterations), list(losses)


def generate_scene_loss_comparison(run_root: Path, summary: dict[str, Any]) -> None:
    scene_runs: dict[str, list[dict[str, Any]]] = {}
    for run in summary.get("runs", []):
        output_dir = Path(run["output_dir"])
        log_path = output_dir / "training_log.csv"
        if not log_path.exists():
            continue
        scene_runs.setdefault(str(run["scene"]), []).append(run)

    for scene_name, runs in scene_runs.items():
        scene_root = run_root / scene_name
        fig, ax = plt.subplots(figsize=(10, 5))
        metrics_bundle: dict[str, Any] = {}

        plotted = 0
        for run in runs:
            output_dir = Path(run["output_dir"])
            log_path = output_dir / "training_log.csv"
            if not log_path.exists():
                continue
            iterations, losses = load_training_log(log_path)
            if not iterations:
                continue
            label = str(run["name"])
            ax.plot(iterations, losses, linewidth=2, label=label)
            metrics_path = output_dir / "metrics_final.json"
            if metrics_path.exists():
                with open(metrics_path, encoding="utf-8") as f:
                    metrics_bundle[label] = json.load(f)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        ax.set_title(f"Loss Curves - {scene_name}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(scene_root / "loss_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        if metrics_bundle:
            save_json(scene_root / "loss_comparison_metrics.json", metrics_bundle)


def create_shared_init_patterns(config_path: Path, output_path: Path, seed_override: int | None, device_override: str | None) -> dict[str, object]:
    cfg = load_config(str(config_path))
    scene_name = cfg["scene"]
    if scene_name not in SCENE_PRESETS:
        raise ValueError(f"Unknown scene '{scene_name}' in {config_path}")

    training = cfg.setdefault("training", {})
    rendering = cfg.setdefault("rendering", {})
    seed = int(seed_override if seed_override is not None else training.get("seed", 0))
    num_patterns = int(training.get("num_patterns", 4))
    init_mode = str(training.get("init_mode", "random"))
    spp = int(rendering.get("spp", 64))
    device = auto_detect_device(device_override or rendering.get("device", "auto"))

    set_random_seed(seed)
    renderer = create_standard_renderer(
        scene_name,
        device=str(device),
        spp=spp,
        backend=str(rendering.get("backend", "pytorch")),
    )
    patterns = initialize_patterns(
        num_patterns=num_patterns,
        projector_width=int(renderer.projector.width),
        device=renderer.device,
        dtype=renderer.dtype,
        init_mode=init_mode,
    ).detach().cpu()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(patterns, output_path)

    meta = {
        "config": str(config_path),
        "scene": scene_name,
        "seed": seed,
        "num_patterns": num_patterns,
        "projector_width": int(renderer.projector.width),
        "init_mode": init_mode,
        "device_for_generation": str(device),
    }
    save_json(output_path.with_suffix(".json"), meta)
    return meta


def ensure_shared_init_patterns(config_path: Path, output_path: Path, seed_override: int | None, device_override: str | None) -> dict[str, object]:
    meta_path = output_path.with_suffix(".json")
    if output_path.exists() and meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return create_shared_init_patterns(config_path, output_path, seed_override, device_override)


def latest_checkpoint(output_dir: Path) -> Path | None:
    final_ckpt = output_dir / "checkpoint_final" / "checkpoint.pt"
    if final_ckpt.exists():
        return final_ckpt

    iter_ckpts = sorted(output_dir.glob("checkpoints/iter_*/checkpoint.pt"))
    if iter_ckpts:
        return iter_ckpts[-1]
    return None


def training_finished(output_dir: Path) -> bool:
    return (output_dir / "metrics_final.json").exists() and (output_dir / "checkpoint_final" / "checkpoint.pt").exists()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab2 assignment experiments with Mitsuba forward rendering and Mitsuba autodiff / finite-difference training.")
    parser.add_argument("--assignment-config", type=str, default=str(DEFAULT_ASSIGNMENT_CONFIG))
    parser.add_argument("--run-root", type=str, default=None, help="Reuse an existing assignment run directory")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override shared seed for initialization and training")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    script_root = repo_root / "scripts" / "lab2"

    assignment_cfg = load_config(args.assignment_config)
    output_cfg = assignment_cfg.get("output", {})
    execution_cfg = assignment_cfg.get("execution", {})
    gradient_cfg = assignment_cfg.get("gradient_comparison", {})
    experiments_cfg = assignment_cfg.get("experiments", [])

    base_dir = Path(output_cfg.get("base_dir", "outputs/lab2/assignment_runs"))
    run_root = Path(args.run_root) if args.run_root is not None else base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    shared_device = args.device or assignment_cfg.get("shared", {}).get("device")
    shared_seed = args.seed if args.seed is not None else assignment_cfg.get("shared", {}).get("seed")
    skip_completed = bool(execution_cfg.get("skip_completed", True))
    auto_resume = bool(execution_cfg.get("auto_resume", True))
    default_backend = str(assignment_cfg.get("shared", {}).get("backend", "pytorch"))

    print("=" * 80)
    print("Lab2 assignment runner")
    print(f"run_root: {run_root}")
    print(f"shared_device: {shared_device}")
    print(f"shared_seed: {shared_seed}")
    print(f"default renderer backend: {default_backend}")
    if default_backend == "mitsuba":
        print("autodiff mode: Mitsuba differentiable rendering")
        print("finite_difference mode: repeated Mitsuba forward renders")
    else:
        print("autodiff mode: PyTorch differentiable renderer")
        print("finite_difference mode: repeated PyTorch forward renders")
    print("=" * 80)

    if execution_cfg.get("run_self_check", True):
        self_check_marker = run_root / "self_check.done.json"
        if skip_completed and self_check_marker.exists():
            print(f"SKIP self-check: {self_check_marker}")
        else:
            print("Running self-check: Mitsuba forward render + Mitsuba autodiff gradient connectivity")
            run_command(
                [sys.executable, "-m", "pytest", "tests/lab2/test_shader_self_check.py", "-q"],
                cwd=repo_root,
            )
            save_json(self_check_marker, {"status": "completed"})

    if execution_cfg.get("run_gradient_comparison", True):
        gradient_marker = run_root / "gradient_comparison.done.json"
        if skip_completed and gradient_marker.exists():
            print(f"SKIP gradient comparison: {gradient_marker}")
        else:
            print("Running gradient comparison: Mitsuba autodiff vs Mitsuba finite difference")
            gradient_cmd = [
                sys.executable,
                str(script_root / "compare_gradients.py"),
                "--config",
                str(Path(gradient_cfg["scene_config"])),
                "--decoder",
                str(gradient_cfg.get("decoder", "zncc")),
                "--num-samples",
                str(int(gradient_cfg.get("num_samples", 4))),
                "--fd-num-coords",
                str(int(gradient_cfg.get("fd_num_coords", 16))),
                "--output-dir",
                str(run_root / "gradient_comparison"),
            ]
            if shared_device is not None:
                gradient_cmd.extend(["--device", str(shared_device)])
            if shared_seed is not None:
                gradient_cmd.extend(["--seed", str(shared_seed)])
            run_command(gradient_cmd, cwd=repo_root)
            save_json(
                gradient_marker,
                {
                    "status": "completed",
                    "scene_config": gradient_cfg["scene_config"],
                    "decoder": gradient_cfg.get("decoder", "zncc"),
                },
            )

    summary: dict[str, Any] = {
        "assignment_config": str(Path(args.assignment_config)),
        "run_root": str(run_root),
        "device": shared_device,
        "seed": shared_seed,
        "runs": [],
    }

    if execution_cfg.get("run_training_experiments", True):
        for experiment in experiments_cfg:
            if not experiment.get("enabled", True):
                continue

            scene_config = Path(experiment["scene_config"])
            cfg = load_config(str(scene_config))
            scene_name = cfg["scene"]
            experiment_name = str(experiment["name"])
            decoder = str(experiment["decoder"])
            gradient_mode = str(experiment["gradient_mode"])

            scene_root = run_root / scene_name
            output_dir = scene_root / experiment_name
            init_patterns_path = scene_root / "shared_init_patterns.pt"
            init_meta = ensure_shared_init_patterns(scene_config, init_patterns_path, shared_seed, shared_device)

            run_entry: dict[str, Any] = {
                "name": experiment_name,
                "scene": scene_name,
                "decoder": decoder,
                "gradient_mode": gradient_mode,
                "output_dir": str(output_dir),
                "shared_init_patterns": str(init_patterns_path),
                "shared_init_meta": init_meta,
            }

            if skip_completed and training_finished(output_dir):
                print(f"SKIP completed training: {output_dir}")
                run_entry["status"] = "skipped_completed"
                summary["runs"].append(run_entry)
                continue

            train_cmd = [
                sys.executable,
                str(script_root / "run_training.py"),
                "--config",
                str(scene_config),
                "--decoder",
                decoder,
                "--gradient-mode",
                gradient_mode,
                "--init-patterns",
                str(init_patterns_path),
                "--output-dir",
                str(output_dir),
            ]

            if "iterations" in experiment:
                train_cmd.extend(["--iterations", str(int(experiment["iterations"]))])
                run_entry["iterations"] = int(experiment["iterations"])
            if gradient_mode == "finite_difference":
                if "fd_num_coords" in experiment:
                    train_cmd.extend(["--fd-num-coords", str(int(experiment["fd_num_coords"]))])
                    run_entry["fd_num_coords"] = int(experiment["fd_num_coords"])
                if "fd_epsilon" in experiment:
                    train_cmd.extend(["--fd-epsilon", str(float(experiment["fd_epsilon"]))])
                    run_entry["fd_epsilon"] = float(experiment["fd_epsilon"])
            if shared_device is not None:
                train_cmd.extend(["--device", str(shared_device)])
            if shared_seed is not None:
                train_cmd.extend(["--seed", str(shared_seed)])

            if auto_resume:
                resume_ckpt = latest_checkpoint(output_dir)
                if resume_ckpt is not None and not training_finished(output_dir):
                    train_cmd.extend(["--resume", str(resume_ckpt)])
                    run_entry["resume_from"] = str(resume_ckpt)

            run_command(train_cmd, cwd=repo_root)
            run_entry["status"] = "completed"
            summary["runs"].append(run_entry)

    generate_scene_loss_comparison(run_root, summary)
    save_json(run_root / "summary.json", summary)
    print(f"\nAll experiments completed. Summary: {run_root / 'summary.json'}")


if __name__ == "__main__":
    main()
