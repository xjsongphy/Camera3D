#!/usr/bin/env python3
"""Re-render held-out NeuS views and export geometry visualizations.

This script works around a nerfstudio NeuS-facto checkpoint loading failure that
silently turned every NeuS evaluation into a "render skipped" row.

Root cause
----------
``lab3`` evaluates NeuS on a held-out split by pointing the nerfstudio data
parser at a separate ``processed/test`` SDFStudio dataset (N_test images) while
the checkpoint was trained on ``processed/train`` (N_train images). At load time
``ns-render`` rebuilds ``NeuSFactoModel`` from that dataset, and the SDF field
allocates ``embedding_appearance`` with one row per dataset image. The trained
checkpoint stores ``N_train`` rows, so ``load_state_dict`` aborts with::

    RuntimeError: size mismatch for field.embedding_appearance.embedding.weight:
      copying a param with shape torch.Size([N_train, 32]) from checkpoint,
      the shape in current model is torch.Size([N_test, 32]).

Note the serialized ``use_appearance_embedding: false`` in ``config.yml`` is
misleading: the field still materializes the embedding at runtime, so the flag
does not prevent the mismatch.

Fix
---
Strip every ``*embedding_appearance*`` tensor from the checkpoint's ``pipeline``
state dict and write a patched checkpoint. The eval model then either has no
such parameter (flag honored) or keeps its freshly initialized one; either way
``strict=False`` loading succeeds because the offending key is gone. Appearance
is a minor view-dependent aid, so dropping it barely affects geometry-driven
NeuS renders.

What this script produces
-------------------------
- ``results/neus/renders/``  held-out RGB renders (so qualitative + metrics work)
- ``results/neus/normals/``  surface-normal renders from the SDF gradient
- ``results/neus/depth/``    depth renders
- ``results/neus/patched_checkpoint.ckpt``  the sanitized checkpoint

Run on the Ubuntu training box (CUDA), not on the report-authoring machine::

    python scripts/lab3/render_neus_heldout.py \
        --run-dir outputs/lab3/20260620_200014_dormitory_fps4

After it finishes, re-run the lab3 post-processing so ``metrics.csv`` and the
qualitative figures pick up the NeuS column::

    lab3 --run-dir outputs/lab3/20260620_200014_dormitory_fps4

This script only reads/writes inside the run directory; it does not modify the
shared virtual environment.
"""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import subprocess
import sys
from pathlib import Path

EMBEDDING_RE = re.compile(r"embedding_appearance")


def _latest_checkpoint(model_dir: Path) -> Path | None:
    ckpts = sorted(model_dir.glob("*.ckpt")) if model_dir.is_dir() else []
    # step-000040000.ckpt sorts last lexically; fall back to mtime if needed.
    return ckpts[-1] if ckpts else None


def strip_appearance_embedding(checkpoint_path: Path, output_path: Path) -> Path:
    """Write a copy of ``checkpoint_path`` without appearance-embedding tensors.

    Imports torch lazily so the script can be imported on machines without the
    CUDA training stack. Returns the patched checkpoint path.
    """
    import torch  # noqa: PLC0415 -- optional dependency on the authoring host

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pipeline_state = state.get("pipeline", state)
    pytorch_state = (
        pipeline_state.get("model", pipeline_state)
        if isinstance(pipeline_state, dict)
        else pipeline_state
    )
    removed = [
        key for key in list(pytorch_state.keys()) if EMBEDDING_RE.search(key)
    ]
    for key in removed:
        pytorch_state.pop(key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    print(f"[neus-fix] stripped {len(removed)} appearance-embedding keys -> {output_path}")
    return output_path


def _set_load_dir(eval_config: Path, patched_dir: Path) -> None:
    """Point an eval config's ``load_dir`` at the patched checkpoint folder."""
    import yaml  # noqa: PLC0415

    text = eval_config.read_text(encoding="utf-8")
    config = yaml.load(text, Loader=yaml.Loader)
    config.load_dir = str(patched_dir)
    config.load_step = None
    eval_config.write_text(yaml.dump(config), encoding="utf-8")


def render_outputs(eval_config: Path, output_root: Path, kinds: list[str]) -> None:
    """Invoke ``ns-render dataset`` for each requested output kind."""
    for kind in kinds:
        out_dir = output_root / kind
        cmd = [
            "ns-render", "dataset",
            "--load-config", str(eval_config),
            "--output-path", str(out_dir),
            "--split", "test",
            "--rendered-output-names", kind,
        ]
        print(f"[neus-fix] $ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="lab3 run directory (outputs/lab3/<run>)")
    parser.add_argument("--neus-subdir", type=Path, default=Path("results/neus"), help="NeuS result subdir inside the run")
    parser.add_argument("--skip-rgb", action="store_true", help="do not re-render RGB")
    parser.add_argument("--skip-normals", action="store_true", help="do not render normal maps")
    parser.add_argument("--skip-depth", action="store_true", help="do not render depth maps")
    args = parser.parse_args(argv)

    neus_dir = (args.run_dir / args.neus_subdir).resolve()
    eval_config = neus_dir / "eval_config.yml"
    if not eval_config.is_file():
        raise SystemExit(f"missing {eval_config}; run the pipeline's evaluate stage first")

    model_dir = None
    for candidate in (neus_dir / "train").rglob("nerfstudio_models"):
        model_dir = candidate
        break
    ckpt = _latest_checkpoint(model_dir) if model_dir else None
    if ckpt is None:
        raise SystemExit(f"no NeuS checkpoint under {neus_dir / 'train'}")

    patched_dir = neus_dir / "patched_models"
    patched_dir.mkdir(parents=True, exist_ok=True)
    patched = patched_dir / ckpt.name
    if not patched.exists():
        # Keep the original checkpoint intact; copy first so the strip is non-destructive.
        shutil.copy2(ckpt, patched)
    strip_appearance_embedding(patched, patched)
    _set_load_dir(eval_config, patched_dir)

    kinds = []
    if not args.skip_rgb:
        kinds.append("rgb")
    if not args.skip_normals:
        kinds.append("normal")
    if not args.skip_depth:
        kinds.append("depth")
    render_outputs(eval_config, neus_dir / "renders", kinds)
    print("[neus-fix] done. Re-run `lab3 --run-dir <run>` to refresh metrics + qualitative figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
