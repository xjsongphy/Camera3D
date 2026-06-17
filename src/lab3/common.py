from __future__ import annotations

import json
import shutil
import subprocess
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, Protocol, TypeVar

E = TypeVar("E", bound=Exception)


class Lab3Error(RuntimeError):
    pass


class JsonLikeDataclass(Protocol):
    pass


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "scene"


def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def build_run_dir(output_root: Path, scene_name: str, timestamp: str | None = None) -> Path:
    stamp = timestamp or timestamp_tag()
    return output_root / f"{slugify(scene_name)}_{stamp}"


def require_tool(tool_name: str, *, error_cls: type[E] = Lab3Error) -> None:
    if shutil.which(tool_name) is None:
        raise error_cls(f"Required tool not found in PATH: {tool_name}")


def run_cmd(
    cmd: list[str],
    *,
    dry_run: bool = False,
    cwd: Path | None = None,
    log_path: Path | None = None,
    error_cls: type[E] = Lab3Error,
) -> None:
    print("$", " ".join(cmd))
    header = f"$ {' '.join(cmd)}\n"
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        log_file.write(header)
    if dry_run:
        if log_file is not None:
            log_file.write("[dry-run] command not executed\n")
            log_file.close()
        return
    proc = subprocess.Popen(
        cmd,
        cwd=None if cwd is None else str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=120)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="")
            tail.append(line.rstrip("\n"))
            if log_file is not None:
                log_file.write(line)
    finally:
        if log_file is not None:
            log_file.close()
    return_code = proc.wait()
    if return_code != 0:
        raise error_cls(
            f"Command failed with exit code {return_code}: {' '.join(cmd)}\n"
            f"Last output lines:\n{chr(10).join(tail)}"
        )


@contextmanager
def timed_block(label: str, timings: dict[str, float]) -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    finally:
        timings[label] = timings.get(label, 0.0) + perf_counter() - started


# --------------------------------------------------------------------------- #
# GPU memory peak tracking (assignment §5.2: record VRAM/memory peak)          #
# --------------------------------------------------------------------------- #
# Training happens in external processes (3DGS ``train.py``, ``ns-train``), so
# we cannot read ``torch.cuda.max_memory_allocated()`` directly. Instead we poll
# ``nvidia-smi`` from a background thread while a command runs and keep the max
# sample. Best-effort: if ``nvidia-smi`` is absent (CPU box, no NVIDIA driver)
# every sample is ``None`` and no peak is recorded — the column stays blank and
# the report notes the limit, as the assignment allows ("若无法精确记录，可给出近似观察").
NVIDIA_SMI_FORMAT = "--query-gpu=memory.used --format=csv,noheader,nounits"


def parse_nvidia_smi_memory(output: str) -> float | None:
    """First parseable ``memory.used`` value (MiB) from nvidia-smi, converted to GiB."""
    for line in output.splitlines():
        token = line.strip().split(",")[0].strip()
        try:
            mib = float(token)
        except ValueError:
            continue
        return mib / 1024.0
    return None


def query_nvidia_smi_gb(nvidia_smi: str = "nvidia-smi") -> float | None:
    """One best-effort sample of currently-used GPU memory in GiB; ``None`` on any failure."""
    try:
        completed = subprocess.run(
            [nvidia_smi] + NVIDIA_SMI_FORMAT.split(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    return parse_nvidia_smi_memory(completed.stdout)


def peak_from_samples(samples: Iterable[float | None]) -> float | None:
    """Max of the non-``None`` samples (GiB); ``None`` if none sampled."""
    values = [s for s in samples if s is not None]
    return max(values) if values else None


@contextmanager
def monitored_block(
    label: str,
    timings: dict[str, float],
    peaks: dict[str, float] | None = None,
    *,
    sampler: Callable[[], float | None] = query_nvidia_smi_gb,
    interval: float = 0.5,
    enabled: bool = True,
) -> Iterator[None]:
    """Time a block and sample GPU memory in a background thread, recording the peak.

    Mirrors :func:`timed_block` but additionally polls ``sampler`` (default:
    nvidia-smi used-memory in GiB) every ``interval`` seconds. The running max is
    written to ``peaks[label]`` only when at least one sample succeeded. When
    ``enabled`` is False (e.g. dry-run, where no command actually runs) it just
    times the block without polling, so no ambient GPU reading is recorded.
    """
    if not enabled:
        with timed_block(label, timings):
            yield
        return

    started = perf_counter()
    stop = threading.Event()
    samples: list[float | None] = []

    def _poll() -> None:
        while not stop.is_set():
            try:
                samples.append(sampler())
            except Exception:
                samples.append(None)
            stop.wait(interval)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval + 1.0)
        timings[label] = timings.get(label, 0.0) + perf_counter() - started
        peak = peak_from_samples(samples)
        if peak is not None and peaks is not None:
            peaks[label] = max(peaks.get(label, 0.0), peak)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_default(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _json_default(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_default(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise Lab3Error(f"Expected JSON object in {path}")
    return value


def copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

