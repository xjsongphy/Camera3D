from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Set matplotlib backend to non-interactive Agg backend for headless operation
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab2.OpticalSGD import DecoderType, OptimizerConfig, OpticalSGDOptimizer
from lab2.analysis import (
    export_history_summaries,
    export_raw_histories,
    generate_pattern_evolution_gif,
    generate_spectrum_evolution_gif,
    plot_frequency_spectrum,
    plot_loss_curve,
    plot_patterns,
    plot_projector_response_curve,
)
from lab2.decoder import correspondence_metrics, hard_decode
from lab2.run_logging import TimingTracker, start_run_logging
from lab2.scene_genertor import SCENE_PRESETS, create_standard_renderer


def auto_detect_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def create_output_dir(base_dir: str, scene: str, decoder: str, penalty: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / "runs" / f"{timestamp}_{scene}_{decoder}_{penalty}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_decoder_variant(decoder_name: str, training_cfg: dict) -> tuple[DecoderType, bool, str]:
    if decoder_name == "zncc":
        return DecoderType.ZNCC, False, "zncc"
    if decoder_name == "zncc_nn_response":
        return DecoderType.ZNCC_NN, True, "zncc_nn_response"
    if decoder_name == "zncc_nn":
        use_curve = bool(training_cfg.get("use_projector_response_curve", False))
        variant_name = "zncc_nn_response" if use_curve else "zncc_nn"
        return DecoderType.ZNCC_NN, use_curve, variant_name
    raise ValueError(f"Unknown decoder variant: {decoder_name}")


def resolve_max_frequency(training_cfg: dict, projector_width: int) -> float | None:
    max_frequency = training_cfg.get("max_frequency")
    if max_frequency is not None:
        return float(max_frequency)

    ratio = training_cfg.get("max_frequency_ratio")
    if ratio is None:
        return None

    nyquist = projector_width / 2.0
    return float(ratio) * nyquist


def init_renderer(scene_name: str, device: torch.device, spp: int):
    renderer = create_standard_renderer(scene_name, device=str(device), spp=spp)
    renderer._depth = None
    renderer._gt_corr = None
    renderer.render_depth_for_visualization()
    renderer.calibrate_autodiff_gain()
    return renderer


def save_config_snapshot(output_dir: Path, cfg: dict) -> None:
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)


def save_training_log(log_rows: list[dict], output_dir: Path) -> None:
    with open(output_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "loss", "elapsed_sec"])
        writer.writeheader()
        writer.writerows(log_rows)


def save_checkpoint(optimizer: OpticalSGDOptimizer, log_rows: list[dict], ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "iteration": optimizer.iteration,
        "patterns": optimizer.patterns.detach().cpu(),
        "patterns_grad": optimizer.patterns.grad.detach().cpu() if optimizer.patterns.grad is not None else None,
        "decoder_state_dict": optimizer.decoder.state_dict() if optimizer.decoder is not None else None,
        "optimizer_state_dict": optimizer.optimizer.state_dict() if optimizer.optimizer is not None else None,
        "scheduler_state_dict": optimizer.scheduler.state_dict() if optimizer.scheduler is not None else None,
        "loss_history": optimizer.loss_history,
        "history_iterations": optimizer.history_iterations,
        "pattern_history": optimizer.pattern_history,
        "decoder_param_history": optimizer.decoder_param_history,
    }
    torch.save(state, ckpt_dir / "checkpoint.pt")
    if log_rows:
        save_training_log(log_rows, ckpt_dir.parent.parent)


def load_checkpoint(ckpt_path: Path, optimizer: OpticalSGDOptimizer) -> int:
    state = torch.load(ckpt_path, map_location=optimizer.renderer.device, weights_only=False)
    optimizer.patterns = torch.nn.Parameter(state["patterns"].to(device=optimizer.renderer.device, dtype=optimizer.renderer.dtype))
    if state.get("decoder_state_dict") is not None and optimizer.decoder is not None:
        optimizer.decoder.load_state_dict(state["decoder_state_dict"])

    opt_ctor = torch.optim.RMSprop if optimizer.config.optimizer_name == "rmsprop" else torch.optim.Adam
    if optimizer.config.decoder_type == DecoderType.ZNCC_NN:
        optimizer.optimizer = opt_ctor([
            {"params": [optimizer.patterns], "lr": optimizer.config.learning_rate},
            {"params": list(optimizer.decoder.parameters()), "lr": optimizer.config.decoder_learning_rate},
        ])
    else:
        optimizer.optimizer = opt_ctor([optimizer.patterns], lr=optimizer.config.learning_rate)

    if state.get("optimizer_state_dict") is not None:
        optimizer.optimizer.load_state_dict(state["optimizer_state_dict"])
    if state.get("scheduler_state_dict") is not None and optimizer.scheduler is not None:
        optimizer.scheduler.load_state_dict(state["scheduler_state_dict"])

    optimizer.loss_history = state.get("loss_history", [])
    optimizer.iteration = state.get("iteration", 0)
    optimizer.history_iterations = state.get("history_iterations", [])
    optimizer.pattern_history = state.get("pattern_history", [])
    optimizer.decoder_param_history = state.get("decoder_param_history", {})
    optimizer.renderer.update_patterns(optimizer.patterns.detach())
    return optimizer.iteration


def render_and_save_gt(renderer, output_dir: Path) -> None:
    depth = renderer.render_depth_for_visualization().cpu().numpy()
    gt_corr = renderer.gt_corr.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    im1 = ax1.imshow(depth, cmap="plasma")
    ax1.set_title(f"GT Depth ({depth.min():.2f}-{depth.max():.2f}m)")
    plt.colorbar(im1, ax=ax1)
    ax2.hist(depth.flatten(), bins=50, edgecolor="black")
    ax2.set_title("Depth Distribution")
    plt.tight_layout()
    fig.savefig(output_dir / "depth_gt.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    display = np.where(np.isfinite(gt_corr), gt_corr, -1)
    im = ax.imshow(display, cmap="viridis")
    ax.set_title("GT Correspondence")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(output_dir / "gt_corr.png", dpi=150)
    plt.close(fig)


def save_final_visualizations(optimizer: OpticalSGDOptimizer, output_dir: Path) -> None:
    for plot_fn, name in [
        (lambda: plot_patterns(optimizer), "patterns_final"),
        (lambda: plot_frequency_spectrum(optimizer), "spectrum_final"),
        (lambda: plot_loss_curve(optimizer), "loss_curve"),
    ]:
        fig = plot_fn()
        fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    response_curve_fig = plot_projector_response_curve(optimizer)
    if response_curve_fig is not None:
        response_curve_fig.savefig(output_dir / "projector_response_curve.png", dpi=150, bbox_inches="tight")
        plt.close(response_curve_fig)


def evaluate_and_save(renderer, optimizer: OpticalSGDOptimizer, output_dir: Path) -> dict:
    patterns = optimizer.patterns
    images = renderer.render_images_autodiff(patterns)
    gt_corr = renderer.gt_corr
    scores = optimizer.decoder(images, patterns)
    pred_corr = hard_decode(scores)
    metrics = {k: v.item() for k, v in correspondence_metrics(pred_corr, gt_corr).items()}

    with open(output_dir / "metrics_final.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_np = pred_corr.cpu().numpy()
    gt_np = gt_corr.cpu().numpy()
    valid = np.isfinite(gt_np)
    gt_display = np.where(valid, gt_np, -1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(pred_np, cmap="viridis"); ax1.set_title("Predicted Correspondence")
    ax2.imshow(gt_display, cmap="viridis"); ax2.set_title("GT Correspondence")
    plt.tight_layout(); fig.savefig(output_dir / "correspondence_final.png", dpi=150); plt.close(fig)

    err = np.where(valid, np.abs(pred_np - gt_np), 0)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(err, cmap="hot")
    ax.set_title(f"Error Map (MAE={metrics['mae']:.2f})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); fig.savefig(output_dir / "depth_error_map.png", dpi=150); plt.close(fig)

    return metrics


def save_rendered_images(renderer, patterns: torch.Tensor, output_dir: Path) -> None:
    images = renderer.render_images(patterns).detach().cpu().numpy()
    k = images.shape[0]
    fig, axes = plt.subplots(k, 1, figsize=(12, 3 * k))
    if k == 1:
        axes = [axes]
    for i in range(k):
        axes[i].imshow(images[i] if images[i].ndim == 3 else images[i], cmap=None if images[i].ndim == 3 else "gray")
        axes[i].set_title(f"Pattern {i + 1} Render")
    plt.tight_layout(); fig.savefig(output_dir / "rendered_final.png", dpi=150); plt.close(fig)


def run_single_decoder(scene_name: str, decoder_name: str, cfg: dict, device: torch.device, output_dir: Path | None) -> Path:
    training = cfg["training"]
    rendering = cfg["rendering"]
    output_cfg = cfg["output"]
    seed = training.get("seed")

    decoder_type, use_projector_response_curve, variant_name = resolve_decoder_variant(decoder_name, training)
    penalty = training.get("penalty", "l1")
    if output_dir is None:
        output_dir = create_output_dir(output_cfg.get("base_dir", "output/lab2"), scene_name, variant_name, penalty)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_session = start_run_logging(output_dir, prefix=f"{scene_name}_{variant_name}")
    timer = TimingTracker()
    run_t0 = time.perf_counter()
    timing_saved = False

    try:
        print(f"\n{'='*60}")
        print(f"  scene: {scene_name}")
        print(f"  decoder: {variant_name}")
        print(f"  iterations: {training['iterations']}")
        print(f"  device: {device}")
        print(f"  output: {output_dir}")
        print(f"  log: {log_session.log_path}")
        if seed is not None:
            print(f"  seed: {seed}")
        print(f"{'='*60}\n")

        if seed is not None:
            set_random_seed(int(seed))

        with timer.phase("save_config_snapshot"):
            save_config_snapshot(output_dir, {
                "scene": scene_name,
                "decoder": variant_name,
                "training": training,
                "rendering": rendering,
                "output": output_cfg,
            })
        with timer.phase("init_renderer"):
            renderer = init_renderer(scene_name, device, rendering.get("spp", 64))
        with timer.phase("render_and_save_gt"):
            render_and_save_gt(renderer, output_dir)

        optimizer_config = OptimizerConfig(
            num_iterations=training["iterations"],
            learning_rate=training.get("learning_rate", 0.01),
            decoder_learning_rate=training.get("decoder_lr", 0.01),
            tau=training.get("tau", 50.0),
            decoder_type=decoder_type,
            neighborhood_size=training.get("neighborhood", 1),
            use_projector_response_curve=use_projector_response_curve,
            penalty=penalty,
            max_frequency=training.get("max_frequency"),
            frequency_weight=training.get("frequency_weight", 1.0),
            log_interval=output_cfg.get("log_interval", 10),
            save_interval=output_cfg.get("save_interval", 50),
            output_dir=str(output_dir),
            optimizer_name=training.get("optimizer", "adam"),
            lr_decay_step=training.get("lr_decay_step", 0),
            lr_decay_gamma=training.get("lr_decay_gamma", 1.0),
            projector_height=renderer.projector.height,
        )

        with timer.phase("create_optimizer"):
            optimizer = OpticalSGDOptimizer(renderer, optimizer_config)
            projector_width = renderer.projector.width
            optimizer.config.max_frequency = resolve_max_frequency(training, projector_width)

        with timer.phase("initialize_patterns"):
            patterns = optimizer.initialize_patterns(training.get("num_patterns", 4), projector_width, training.get("init_mode", "random"))

        resume_ckpt = cfg.get("resume")
        start_iter = 0
        if resume_ckpt:
            with timer.phase("load_checkpoint"):
                ckpt = Path(resume_ckpt)
                if not ckpt.exists():
                    alt = output_dir / "checkpoint_final" / "checkpoint.pt"
                    ckpt = alt if alt.exists() else ckpt
                if ckpt.exists():
                    start_iter = load_checkpoint(ckpt, optimizer)
                    print(f"  resumed from checkpoint, completed {start_iter} iterations")

        log_rows: list[dict] = []
        start_time = time.time()
        iterations = training["iterations"]
        log_interval = output_cfg.get("log_interval", 10)
        save_interval = output_cfg.get("save_interval", 50)

        with timer.phase("training_loop"):
            for i in range(start_iter, iterations):
                loss = optimizer.step()
                elapsed = time.time() - start_time
                done = (i + 1) - start_iter
                eta = (iterations - (i + 1)) * (elapsed / max(1, done))
                log_rows.append({"iteration": i + 1, "loss": loss, "elapsed_sec": round(elapsed, 2)})
                if (i + 1) % log_interval == 0 or i == 0:
                    print(f"  Iter {i+1}/{iterations}: loss={loss:.6f} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)")
                if (i + 1) % save_interval == 0:
                    save_checkpoint(optimizer, log_rows, output_dir / "checkpoints" / f"iter_{i+1}")

        total_time = time.time() - start_time
        with timer.phase("save_training_log"):
            save_training_log(log_rows, output_dir)
        with timer.phase("save_final_checkpoint"):
            save_checkpoint(optimizer, log_rows, output_dir / "checkpoint_final")
        with timer.phase("save_final_visualizations"):
            save_final_visualizations(optimizer, output_dir)
        with timer.phase("export_raw_histories"):
            export_raw_histories(optimizer, output_dir)
        with timer.phase("export_history_summaries"):
            export_history_summaries(optimizer, output_dir)
        with timer.phase("generate_pattern_gif"):
            generate_pattern_evolution_gif(optimizer)
        with timer.phase("generate_spectrum_gif"):
            generate_spectrum_evolution_gif(optimizer)
        with timer.phase("evaluate_and_save"):
            metrics = evaluate_and_save(renderer, optimizer, output_dir)
        try:
            with timer.phase("save_rendered_images"):
                save_rendered_images(renderer, optimizer.patterns, output_dir)
        except Exception as e:
            print(f"  [warn] save_rendered_images failed: {e}")

        timer.add("total_run", time.perf_counter() - run_t0)
        timer.save_csv(output_dir / "timing.csv")
        timing_saved = True
        print(f"\n  done in {total_time:.1f}s")
        print(f"  MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} acc<1px={metrics['acc_1']:.3f}")
        return output_dir
    finally:
        try:
            if not timing_saved:
                timer.add("total_run", time.perf_counter() - run_t0)
                timer.save_csv(output_dir / "timing.csv")
        except Exception:
            pass
        log_session.close()


def generate_comparison(run_dirs: dict[str, Path], comparison_dir: Path) -> None:
    comparison_dir.mkdir(parents=True, exist_ok=True)

    def load_log(d: Path) -> tuple[list[int], list[float]]:
        rows = []
        with open(d / "training_log.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((int(row["iteration"]), float(row["loss"])))
        if not rows:
            return [], []
        a, b = zip(*rows)
        return list(a), list(b)

    metrics_bundle = {}
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, run_dir in run_dirs.items():
        iters, losses = load_log(run_dir)
        with open(run_dir / "metrics_final.json", encoding="utf-8") as f:
            metrics_bundle[label] = json.load(f)
        ax.plot(iters, losses, label=label)

    with open(comparison_dir / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_bundle, f, indent=2)

    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(comparison_dir / "comparison_loss_curves.png", dpi=150)
    plt.close(fig)

    print(f"\ncomparison saved to: {comparison_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Structured-light training")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--decoder", type=str, choices=["zncc", "zncc_nn", "zncc_nn_response", "both"], default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config)

    if args.scene is not None:
        cfg["scene"] = args.scene
    if args.decoder is not None:
        cfg["decoder"] = args.decoder
    if args.iterations is not None:
        cfg.setdefault("training", {})["iterations"] = args.iterations
    if args.device is not None:
        cfg.setdefault("rendering", {})["device"] = args.device
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed
    if args.resume is not None:
        cfg["resume"] = args.resume

    scene_name = cfg["scene"]
    if scene_name not in SCENE_PRESETS:
        print(f"error: unknown scene '{scene_name}'. available: {list(SCENE_PRESETS.keys())}")
        sys.exit(1)

    rendering = cfg.setdefault("rendering", {})
    device = auto_detect_device(rendering.get("device", "auto"))
    print(f"device: {device}")

    output_cfg = cfg.setdefault("output", {})
    training = cfg.setdefault("training", {})
    penalty = training.get("penalty", "l1")
    decoder = cfg.get("decoder", "zncc")

    if decoder == "both":
        base_dir = output_cfg.get("base_dir", "output/lab2")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dir = Path(base_dir) / "runs" / f"{ts}_{scene_name}_comparison_{penalty}"
        zncc_cfg = copy.deepcopy(cfg)
        zncc_cfg.setdefault("training", {})["use_projector_response_curve"] = False

        zncc_nn_cfg = copy.deepcopy(cfg)
        zncc_nn_cfg.setdefault("training", {})["use_projector_response_curve"] = False

        zncc_nn_response_cfg = copy.deepcopy(cfg)
        zncc_nn_response_cfg.setdefault("training", {})["use_projector_response_curve"] = True

        zncc_dir = run_single_decoder(scene_name, "zncc", zncc_cfg, device, None)
        nn_dir = run_single_decoder(scene_name, "zncc_nn", zncc_nn_cfg, device, None)
        nn_response_dir = run_single_decoder(scene_name, "zncc_nn_response", zncc_nn_response_cfg, device, None)

        generate_comparison(
            {
                "zncc": zncc_dir,
                "zncc_nn": nn_dir,
                "zncc_nn_response": nn_response_dir,
            },
            comparison_dir,
        )
        with open(comparison_dir / "comparison_runs.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "zncc": str(zncc_dir),
                    "zncc_nn": str(nn_dir),
                    "zncc_nn_response": str(nn_response_dir),
                },
                f,
                indent=2,
            )
    else:
        _, _, variant_name = resolve_decoder_variant(decoder, training)
        out = create_output_dir(output_cfg.get("base_dir", "output/lab2"), scene_name, variant_name, penalty)
        run_single_decoder(scene_name, decoder, cfg, device, out)


if __name__ == "__main__":
    main()
