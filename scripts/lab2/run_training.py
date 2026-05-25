"""
Run Optical SGD training with specified config and scene.

Usage:
    python scripts/lab2/run_training.py --scene sl_plane_diffuse --decoder zncc --penalty l1
    python scripts/lab2/run_training.py --scene sl_marble_objects --decoder zncc_nn --penalty zero_tolerance --iterations 200

Output:
    - results/lab2/{timestamp}_{scene}_{decoder}_{penalty}/
        - config.json: Training configuration
        - scene_info.txt: Scene description
        - patterns_final.png: Final optimized patterns
        - training_log.csv: Loss history
        - checkpoints/: Pattern checkpoints
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import torch

from src.lab2.OpticalSGD import DecoderType, OptimizerConfig, OpticalSGDOptimizer, initialize_patterns
from src.lab2.scene_genertor import SCENE_PRESETS, build_runtime_scene_dict
from src.lab2.shader import Camera, Projector, StructuredLightRenderer


def load_scene_bundle(scene_dir: Path) -> dict[str, Any]:
    """Load scene bundle from directory."""
    scene_xml = scene_dir / "scene.xml"
    if not scene_xml.exists():
        raise FileNotFoundError(f"Scene XML not found: {scene_xml}")

    # Load scene info
    info_file = scene_dir / "info.json"
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
    else:
        info = {"name": scene_dir.name}

    return {"xml_path": str(scene_xml), "info": info}


def create_output_dir(base_dir: Path, scene: str, decoder: str, penalty: str) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{scene}_{decoder}_{penalty}"
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_config(output_dir: Path, config: OptimizerConfig, scene: str) -> None:
    """Save training configuration to JSON."""
    config_dict = asdict(config)
    config_dict["scene"] = scene

    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    print(f"Saved config to {config_file}")


def save_scene_info(output_dir: Path, scene_info: dict[str, Any]) -> None:
    """Save scene information."""
    info_file = output_dir / "scene_info.txt"
    with open(info_file, "w") as f:
        f.write(f"Scene: {scene_info.get('name', 'unknown')}\n")
        f.write(f"Description: {scene_info.get('description', 'N/A')}\n")
        f.write(f"XML Path: {scene_info.get('xml_path', 'N/A')}\n")

    print(f"Saved scene info to {info_file}")


@click.command()
@click.option("--scene", type=click.Choice(list(SCENE_PRESETS.keys())), required=True,
              help="Scene preset to use for training")
@click.option("--decoder", type=click.Choice(["zncc", "zncc_nn"]), default="zncc",
              help="Decoder type")
@click.option("--penalty", type=click.Choice(["l1", "zero_tolerance", "one_tolerance"]), default="l1",
              help="Penalty function")
@click.option("--iterations", type=int, default=100,
              help="Number of optimization iterations")
@click.option("--learning-rate", type=float, default=0.01,
              help="Learning rate for pattern optimization")
@click.option("--decoder-lr", type=float, default=0.01,
              help="Learning rate for decoder parameters (ZNCC-NN only)")
@click.option("--num-patterns", type=int, default=4,
              help="Number of projection patterns (K)")
@click.option("--neighborhood", type=int, default=1,
              help="Neighborhood size for decoder (1, 3, or 5)")
@click.option("--tau", type=float, default=50.0,
              help="Softmax temperature")
@click.option("--init-mode", type=click.Choice(["random", "sine", "constant", "stripe"]), default="random",
              help="Pattern initialization mode")
@click.option("--output-dir", type=str, default="results/lab2",
              help="Base output directory")
@click.option("--device", type=str, default="cpu",
              help="PyTorch device (cpu or cuda)")
def main(
    scene: str,
    decoder: str,
    penalty: str,
    iterations: int,
    learning_rate: float,
    decoder_lr: float,
    num_patterns: int,
    neighborhood: int,
    tau: float,
    init_mode: str,
    output_dir: str,
    device: str,
) -> None:
    """Run Optical SGD training with specified configuration."""

    # Setup device
    device = torch.device(device)

    # Load scene
    scene_dir = Path("assets/scenes") / scene
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    scene_bundle = load_scene_bundle(scene_dir)
    scene_info = scene_bundle["info"]

    # Create output directory
    output_path = Path(output_dir)
    output_path = create_output_dir(output_path, scene, decoder, penalty)
    print(f"Output directory: {output_path}")

    # Create configuration
    config = OptimizerConfig(
        num_iterations=iterations,
        learning_rate=learning_rate,
        decoder_learning_rate=decoder_lr,
        tau=tau,
        decoder_type=DecoderType.ZNCC if decoder == "zncc" else DecoderType.ZNCC_NN,
        neighborhood_size=neighborhood,
        penalty=penalty,
        output_dir=str(output_path),
    )

    # Save configuration
    save_config(output_path, config, scene)
    save_scene_info(output_path, scene_info)

    # Setup camera and projector
    camera = Camera(
        width=640,
        height=480,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        R=torch.eye(3),
        t=torch.tensor([0.0, 0.0, 0.0]),
    )

    projector = Projector(
        width=640,
        height=480,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        R=torch.eye(3),
        t=torch.tensor([0.1, 0.0, 0.0]),  # 10cm baseline
    )

    # Create renderer
    renderer = StructuredLightRenderer(
        scene_xml=scene_bundle["xml_path"],
        camera=camera,
        projector=projector,
        num_patterns=num_patterns,
        device=device,
    )

    # Create optimizer
    optimizer = OpticalSGDOptimizer(renderer, config)

    # Initialize patterns
    patterns = optimizer.initialize_patterns(
        num_patterns=num_patterns,
        projector_width=projector.width,
        init_mode=init_mode,
    )

    print(f"\nStarting training:")
    print(f"  Scene: {scene} ({scene_info.get('description', 'N/A')})")
    print(f"  Decoder: {decoder} (neighborhood={neighborhood})")
    print(f"  Penalty: {penalty}")
    print(f"  Patterns: {num_patterns}x{projector.width}")
    print(f"  Iterations: {iterations}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print()

    # Training loop
    start_time = time.time()

    for i in range(iterations):
        loss = optimizer.step()
        optimizer.iteration = i + 1

        if (i + 1) % config.log_interval == 0 or i == 0:
            elapsed = time.time() - start_time
            print(f"Iter {i+1}/{iterations}: loss={loss:.6f} ({elapsed:.1f}s)")

        if (i + 1) % config.save_interval == 0 or (i + 1) == iterations:
            checkpoint_dir = output_path / "checkpoints" / f"iter_{i+1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save patterns visualization
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(num_patterns, 1, figsize=(12, 2 * num_patterns))
            for k in range(num_patterns):
                axes[k].imshow(patterns[k].detach().cpu().numpy()[None, :], cmap="gray", aspect="auto")
                axes[k].set_title(f"Pattern {k+1}")
                axes[k].set_xlabel("Projector Column")
                axes[k].set_yticks([])
            plt.tight_layout()
            plt.savefig(checkpoint_dir / "patterns.png", dpi=150)
            plt.close()

            # Save pattern data
            torch.save(patterns.detach().cpu(), checkpoint_dir / "patterns.pt")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
