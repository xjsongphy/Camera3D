from __future__ import annotations

import json
import shutil
import subprocess
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Protocol, TypeVar

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


def run_cmd(cmd: list[str], *, dry_run: bool = False, cwd: Path | None = None, error_cls: type[E] = Lab3Error) -> None:
    print("$", " ".join(cmd))
    if dry_run:
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
    for line in proc.stdout:
        print(line, end="")
        tail.append(line.rstrip("\n"))
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

