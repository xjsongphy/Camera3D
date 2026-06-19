from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

CAMERA_RE = re.compile(r"^Reading camera\s+(\d+)/(\d+)")
PROGRESS_RE = re.compile(
    r"^Training progress:\s+\d+%.*?\|\s*(\d+)/(\d+)\s*\[[^\]]*Loss=([0-9.]+),\s*Depth Loss=([0-9.]+)\]"
)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _safe_write(text: str) -> None:
    try:
        print(text, end="")
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
        sys.stdout.buffer.write(encoded)
        sys.stdout.flush()


class ProgressFilter:
    def __init__(self) -> None:
        self.active_inline = False
        self.last_train_iter: int | None = None
        self.curves: list[dict[str, object]] = []

    def _flush_inline(self) -> str:
        if self.active_inline:
            self.active_inline = False
            return "\n"
        return ""

    def handle(self, line: str) -> tuple[str | None, str | None]:
        clean_line = ANSI_RE.sub("", line)

        camera = CAMERA_RE.match(clean_line)
        if camera:
            cur = int(camera.group(1))
            total = int(camera.group(2))
            self.active_inline = True
            stdout = f"\rReading camera {cur}/{total}"
            return stdout, None

        progress = PROGRESS_RE.match(clean_line)
        if progress:
            iteration = int(progress.group(1))
            total = int(progress.group(2))
            loss = float(progress.group(3))
            depth_loss = float(progress.group(4))
            self.active_inline = True
            stdout = (
                f"\rTraining progress: {iteration}/{total} | "
                f"Loss={loss:.7f} | Depth Loss={depth_loss:.7f}"
            )
            if self.last_train_iter != iteration:
                self.last_train_iter = iteration
                self.curves.append(
                    {
                        "iteration": iteration,
                        "total_iterations": total,
                        "loss": loss,
                        "depth_loss": depth_loss,
                    }
                )
            return stdout, None

        prefix = self._flush_inline()
        return prefix + clean_line, None

    def finalize(self) -> tuple[str | None, str | None]:
        suffix = self._flush_inline()
        return (suffix or None), None


def write_curve_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "total_iterations", "loss", "depth_loss"])
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter 3DGS train output and write structured curves.")
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--curve-path", type=Path, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.cmd:
        raise SystemExit("gs_train_wrapper requires a command after '--'.")
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        raise SystemExit("gs_train_wrapper requires a non-empty command.")

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    filter_ = ProgressFilter()
    tail: deque[str] = deque(maxlen=120)

    with args.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(args.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            stdout_text, log_text = filter_.handle(raw_line)
            log_file.write(raw_line)
            tail.append(ANSI_RE.sub("", raw_line).rstrip("\n"))
            if stdout_text:
                _safe_write(stdout_text)
            if log_text:
                log_file.write(log_text)
                tail.append(log_text.rstrip("\n"))
        stdout_text, log_text = filter_.finalize()
        if stdout_text:
            _safe_write(stdout_text)
        if log_text:
            log_file.write(log_text)
            tail.append(log_text.rstrip("\n"))
        return_code = proc.wait()

    write_curve_csv(args.curve_path, filter_.curves)
    if return_code != 0:
        raise SystemExit(
            "Command failed with exit code "
            f"{return_code}: {' '.join(cmd)}\nLast output lines:\n" + "\n".join(tail)
        )


if __name__ == "__main__":
    main()
