from __future__ import annotations

import csv
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO


class _TeeStream:
    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()


class RunLogSession:
    """Mirror stdout/stderr to a run-scoped log file."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._fp = log_path.open("a", encoding="utf-8")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeStream(self._orig_stdout, self._fp)  # type: ignore[assignment]
        sys.stderr = _TeeStream(self._orig_stderr, self._fp)  # type: ignore[assignment]

    def close(self) -> None:
        if sys.stdout is not self._orig_stdout:
            sys.stdout = self._orig_stdout
        if sys.stderr is not self._orig_stderr:
            sys.stderr = self._orig_stderr
        self._fp.flush()
        self._fp.close()


def start_run_logging(output_dir: Path, prefix: str = "train") -> RunLogSession:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{prefix}_{ts}.log"
    return RunLogSession(log_path)


@dataclass
class TimingRow:
    stage: str
    elapsed_sec: float


class TimingTracker:
    def __init__(self) -> None:
        self._rows: list[TimingRow] = []

    def add(self, stage: str, elapsed_sec: float) -> None:
        self._rows.append(TimingRow(stage=stage, elapsed_sec=float(elapsed_sec)))

    @contextmanager
    def phase(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - t0)

    def save_csv(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["stage", "elapsed_sec"])
            writer.writeheader()
            for row in self._rows:
                writer.writerow({"stage": row.stage, "elapsed_sec": round(row.elapsed_sec, 6)})

