from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab2.common import auto_detect_device, load_config
from lab2.scene_genertor import SCENE_PRESETS
from lab2.training_pipeline import create_run_output_dir, resolve_decoder_variant, run_decoder_comparison, run_single_decoder


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Structured-light training with Mitsuba forward rendering and Mitsuba autodiff or finite difference gradients")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--decoder", type=str, choices=["zncc", "zncc_nn", "zncc_nn_response", "both"], default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gradient-mode", type=str, choices=["autodiff", "finite_difference"], default=None,
                   help="`autodiff` uses Mitsuba differentiable rendering; `finite_difference` uses repeated Mitsuba forward renders")
    p.add_argument("--fd-num-coords", type=int, default=None)
    p.add_argument("--fd-epsilon", type=float, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--init-patterns", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
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
    if args.gradient_mode is not None:
        cfg.setdefault("training", {})["gradient_mode"] = args.gradient_mode
    if args.fd_num_coords is not None:
        cfg.setdefault("training", {})["fd_num_coords"] = args.fd_num_coords
    if args.fd_epsilon is not None:
        cfg.setdefault("training", {})["fd_epsilon"] = args.fd_epsilon
    if args.resume is not None:
        cfg["resume"] = args.resume
    if args.init_patterns is not None:
        cfg["init_patterns_path"] = args.init_patterns

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
        from datetime import datetime

        base_dir = output_cfg.get("base_dir", "output/lab2")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dir = Path(base_dir) / "runs" / f"{ts}_{scene_name}_comparison_{penalty}"
        run_decoder_comparison(scene_name, cfg, device, comparison_dir)
    else:
        _, _, variant_name = resolve_decoder_variant(decoder, training)
        out = Path(args.output_dir) if args.output_dir is not None else create_run_output_dir(
            output_cfg.get("base_dir", "output/lab2"),
            scene_name,
            variant_name,
            penalty,
        )
        run_single_decoder(scene_name, decoder, cfg, device, out)


if __name__ == "__main__":
    main()
