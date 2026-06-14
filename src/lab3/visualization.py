from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lab3.common import require_tool, run_cmd


@dataclass(frozen=True)
class ViewerConfig:
    open3d_bin: str = "python"
    script_path: Path | None = None
    dry_run: bool = False


def open_point_cloud(path: Path, cfg: ViewerConfig) -> None:
    """Launch an optional offline point-cloud viewer after reconstruction."""
    if cfg.script_path is None:
        raise RuntimeError("Viewer script is not configured.")
    if not cfg.dry_run:
        require_tool(cfg.open3d_bin)
    run_cmd([cfg.open3d_bin, str(cfg.script_path), str(path)], dry_run=cfg.dry_run)
