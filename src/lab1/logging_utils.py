from __future__ import annotations

import csv
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterator, TextIO


class _TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def build_timestamped_log_path(log_dir: Path, stem: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem).strip("_") or "run"
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = log_dir / f"{safe_stem}_{timestamp}.log"
    index = 1
    while candidate.exists():
        candidate = log_dir / f"{safe_stem}_{timestamp}_{index}.log"
        index += 1
    return candidate


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{rem:05.2f}s"
    hours, rem = divmod(minutes, 60)
    return f"{int(hours)}h{int(rem):02d}m{seconds % 60:05.2f}s"


@contextmanager
def timed_block(label: str, timings: dict[str, float]) -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    finally:
        timings[label] = timings.get(label, 0.0) + (perf_counter() - started)


def print_timing_summary(title: str, timings: dict[str, float]) -> None:
    print(f"\n{title}")
    for label, seconds in timings.items():
        print(f"  {label}: {format_duration(seconds)}")


def write_timing_csv(path: Path, timings: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "seconds", "human"])
        writer.writeheader()
        for label, seconds in timings.items():
            writer.writerow({"stage": label, "seconds": f"{seconds:.6f}", "human": format_duration(seconds)})


@contextmanager
def tee_console_output(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started_perf = perf_counter()
    status = "completed"

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[{started_at}] Run started\n")
        log_file.flush()

        stdout_tee = _TeeStream(sys.stdout, log_file)
        stderr_tee = _TeeStream(sys.stderr, log_file)

        try:
            with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                yield
        except SystemExit as exc:
            exit_code = exc.code
            if exit_code in (None, 0):
                status = "completed"
            else:
                status = f"exited: {exit_code}"
            raise
        except BaseException as exc:
            status = f"failed: {exc.__class__.__name__}: {exc}"
            raise
        finally:
            finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
            elapsed = perf_counter() - started_perf
            log_file.write(f"\n[{finished_at}] Run {status} (elapsed={format_duration(elapsed)})\n")
            log_file.flush()
