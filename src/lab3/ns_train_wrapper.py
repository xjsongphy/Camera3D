from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_PREFIX_RE = re.compile(r"^(\d+)\s+\(([\d.]+)%\)$")


def _safe_write(text: str) -> None:
    try:
        print(text, end="")
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
        sys.stdout.buffer.write(encoded)
        sys.stdout.flush()


class NerfstudioProgressFilter:
    def __init__(self) -> None:
        self.active_inline = False
        self.last_step: int | None = None

    def _flush_inline(self) -> str:
        if self.active_inline:
            self.active_inline = False
            return "\n"
        return ""

    def handle(self, line: str) -> str | None:
        clean_line = ANSI_RE.sub("", line)
        stripped = clean_line.strip()

        if not stripped:
            return None
        if stripped.startswith("Step (% Done)") or set(stripped) == {"-"}:
            return None

        parts = re.split(r"\s{2,}", stripped)
        if len(parts) >= 4:
            step_match = STEP_PREFIX_RE.match(parts[0])
            if step_match:
                step = int(step_match.group(1))
                percent = step_match.group(2)
                train_iter = parts[1]
                eta = parts[2]
                train_rays = parts[3]
                test_rays = parts[4] if len(parts) >= 5 else ""
                if self.last_step != step:
                    self.last_step = step
                    self.active_inline = True
                    summary = (
                        f"\rNeRF train: step {step} ({percent}%) | "
                        f"iter {train_iter} | ETA {eta} | train {train_rays}"
                    )
                    if test_rays:
                        summary += f" | test {test_rays}"
                    return summary
                return None

        prefix = self._flush_inline()
        return prefix + clean_line

    def finalize(self) -> str | None:
        suffix = self._flush_inline()
        return suffix or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter nerfstudio train stdout for terminal display.")
    parser.add_argument("--cwd", type=Path, required=False)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.cmd:
        raise SystemExit("ns_train_wrapper requires a command after '--'.")
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        raise SystemExit("ns_train_wrapper requires a non-empty command.")

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    filter_ = NerfstudioProgressFilter()
    tail: deque[str] = deque(maxlen=120)

    with args.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=None if args.cwd is None else str(args.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            log_file.write(raw_line)
            tail.append(ANSI_RE.sub("", raw_line).rstrip("\n"))
            stdout_text = filter_.handle(raw_line)
            if stdout_text:
                _safe_write(stdout_text)
        stdout_text = filter_.finalize()
        if stdout_text:
            _safe_write(stdout_text)
        return_code = proc.wait()

    if return_code != 0:
        raise SystemExit(
            "Command failed with exit code "
            f"{return_code}: {' '.join(cmd)}\nLast output lines:\n" + "\n".join(tail)
        )


if __name__ == "__main__":
    main()
