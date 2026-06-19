from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lab3.common import Lab3Error


@dataclass(frozen=True)
class ScalarRecord:
    tag: str
    step: int
    value: float
    wall_time: float


EVENT_FILE_GLOB = "events.out.tfevents.*"


def find_latest_event_file(root: Path) -> Path | None:
    matches = sorted(root.rglob(EVENT_FILE_GLOB))
    return matches[-1] if matches else None


def load_tensorboard_scalars(event_file: Path) -> list[ScalarRecord]:
    try:
        from tensorboard.backend.event_processing import event_file_loader
        from tensorboard.util import tensor_util
    except Exception as exc:  # pragma: no cover - depends on optional runtime env
        raise Lab3Error("TensorBoard is required to export training scalar artifacts.") from exc

    records: list[ScalarRecord] = []
    loader = event_file_loader.EventFileLoader(str(event_file))
    for event in loader.Load():
        summary = getattr(event, "summary", None)
        if summary is None:
            continue
        for value in summary.value:
            scalar_value: float | None = None
            if getattr(value, "HasField", None) is not None and value.HasField("simple_value"):
                scalar_value = float(value.simple_value)
            elif getattr(value, "HasField", None) is not None and value.HasField("tensor"):
                try:
                    ndarray = tensor_util.make_ndarray(value.tensor)
                except Exception:
                    continue
                if getattr(ndarray, "size", 0) == 1:
                    try:
                        scalar_value = float(ndarray.reshape(-1)[0])
                    except (TypeError, ValueError):
                        scalar_value = None
            if scalar_value is None:
                continue
            records.append(
                ScalarRecord(
                    tag=value.tag,
                    step=int(event.step),
                    value=scalar_value,
                    wall_time=float(event.wall_time),
                )
            )
    records.sort(key=lambda item: (item.step, item.tag, item.wall_time))
    return records


def write_scalar_records_csv(records: Iterable[ScalarRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "step", "value", "wall_time"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "tag": record.tag,
                    "step": record.step,
                    "value": record.value,
                    "wall_time": record.wall_time,
                }
            )


def select_loss_records(records: Iterable[ScalarRecord]) -> list[ScalarRecord]:
    return [record for record in records if "loss" in record.tag.lower()]


def write_loss_curve_csv(records: Iterable[ScalarRecord], path: Path) -> list[str]:
    grouped: dict[str, dict[int, float]] = {}
    steps: set[int] = set()
    for record in records:
        grouped.setdefault(record.tag, {})[record.step] = record.value
        steps.add(record.step)

    tags = sorted(grouped)
    ordered_steps = sorted(steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *tags])
        for step in ordered_steps:
            writer.writerow([step, *[grouped[tag].get(step, "") for tag in tags]])
    return tags


def plot_scalar_records(records: Iterable[ScalarRecord], path: Path, *, title: str) -> None:
    grouped: dict[str, list[ScalarRecord]] = {}
    for record in records:
        grouped.setdefault(record.tag, []).append(record)
    if not grouped:
        return

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for tag in sorted(grouped):
        series = sorted(grouped[tag], key=lambda item: item.step)
        ax.plot([item.step for item in series], [item.value for item in series], label=tag)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export_training_scalar_artifacts(method: str, event_root: Path, output_dir: Path) -> dict[str, Path]:
    event_file = find_latest_event_file(event_root)
    if event_file is None:
        return {}

    records = load_tensorboard_scalars(event_file)
    if not records:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(method)
    artifacts = {
        "event_file": event_file,
        "scalars_csv": output_dir / f"{prefix}_train_scalars.csv",
    }
    write_scalar_records_csv(records, artifacts["scalars_csv"])

    loss_records = select_loss_records(records)
    if loss_records:
        artifacts["loss_csv"] = output_dir / f"{prefix}_train_loss_curve.csv"
        artifacts["loss_png"] = output_dir / f"{prefix}_train_loss_curve.png"
        write_loss_curve_csv(loss_records, artifacts["loss_csv"])
        plot_scalar_records(loss_records, artifacts["loss_png"], title=f"{method} training losses")

    return artifacts


def _safe_prefix(method: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", method.strip().lower()).strip("_") or "train"
