from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from lab3.training_artifacts import (
    ScalarRecord,
    select_loss_records,
    write_loss_curve_csv,
    write_scalar_records_csv,
)


def test_select_loss_records_filters_case_insensitively() -> None:
    records = [
        ScalarRecord("train/loss", 1, 0.5, 0.0),
        ScalarRecord("Train PSNR", 1, 20.0, 0.0),
        ScalarRecord("eval_total_LOSS", 2, 0.4, 0.1),
    ]

    filtered = select_loss_records(records)

    assert [record.tag for record in filtered] == ["train/loss", "eval_total_LOSS"]


def test_write_scalar_records_csv_writes_long_format() -> None:
    records = [
        ScalarRecord("train/loss", 1, 0.5, 10.0),
        ScalarRecord("train/psnr", 1, 20.0, 10.0),
    ]

    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scalars.csv"
        write_scalar_records_csv(records, path)
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    assert rows == [
        {"tag": "train/loss", "step": "1", "value": "0.5", "wall_time": "10.0"},
        {"tag": "train/psnr", "step": "1", "value": "20.0", "wall_time": "10.0"},
    ]


def test_write_loss_curve_csv_writes_wide_step_table() -> None:
    records = [
        ScalarRecord("train/loss", 1, 0.5, 0.0),
        ScalarRecord("train/loss", 2, 0.4, 0.1),
        ScalarRecord("eval/loss", 2, 0.6, 0.1),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "loss.csv"
        tags = write_loss_curve_csv(records, path)
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))

    assert tags == ["eval/loss", "train/loss"]
    assert rows == [
        ["step", "eval/loss", "train/loss"],
        ["1", "", "0.5"],
        ["2", "0.6", "0.4"],
    ]
