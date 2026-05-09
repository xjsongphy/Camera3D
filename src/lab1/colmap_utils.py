from __future__ import annotations

import shutil
import subprocess
from collections import deque
from pathlib import Path
from typing import TypeVar

E = TypeVar("E", bound=Exception)


def run_cmd(cmd: list[str], *, dry_run: bool, error_cls: type[E]) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    proc = subprocess.Popen(
        cmd,
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
        log_tail = "\n".join(tail)
        raise error_cls(
            f"Command failed with exit code {return_code}: {' '.join(cmd)}\n"
            f"Last output lines:\n{log_tail}"
        )


def require_tool(tool_name: str, *, error_cls: type[E]) -> None:
    if shutil.which(tool_name) is None:
        raise error_cls(f"Required tool not found in PATH: {tool_name}")


def run_feature_extractor(
    *,
    colmap_bin: str,
    db_path: Path,
    images_dir: Path,
    mask_path: Path | None = None,
    camera_mask_path: Path | None = None,
    dry_run: bool,
    error_cls: type[E],
) -> None:
    cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        "PINHOLE",
    ]
    if mask_path is not None:
        cmd.extend(["--ImageReader.mask_path", str(mask_path)])
    if camera_mask_path is not None:
        cmd.extend(["--ImageReader.camera_mask_path", str(camera_mask_path)])
    run_cmd(
        cmd,
        dry_run=dry_run,
        error_cls=error_cls,
    )


def run_sequential_matcher(
    *,
    colmap_bin: str,
    db_path: Path,
    dry_run: bool,
    error_cls: type[E],
) -> None:
    run_cmd(
        [
            colmap_bin,
            "sequential_matcher",
            "--database_path",
            str(db_path),
        ],
        dry_run=dry_run,
        error_cls=error_cls,
    )


def run_model_converter(
    *,
    colmap_bin: str,
    model_dir: Path,
    dry_run: bool,
    error_cls: type[E],
) -> None:
    run_cmd(
        [
            colmap_bin,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(model_dir),
            "--output_type",
            "TXT",
        ],
        dry_run=dry_run,
        error_cls=error_cls,
    )
