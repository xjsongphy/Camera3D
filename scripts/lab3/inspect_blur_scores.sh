#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

INPUT_DIR="${1:-input/lab3_dormitory_input}"
TOP_K="${TOP_K:-10}"

uv run python - <<'PY' "$INPUT_DIR" "$TOP_K"
from __future__ import annotations

import statistics
import sys
import tempfile
from pathlib import Path
from subprocess import run

from lab3.extract import compute_blur_score, discover_inputs

input_dir = Path(sys.argv[1])
top_k = int(sys.argv[2])

images, videos = discover_inputs(input_dir)

rows: list[tuple[float, str]] = []
for path in images:
    rows.append((compute_blur_score(path), str(path)))

if videos:
    with tempfile.TemporaryDirectory(prefix="lab3_blur_probe_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for video in videos:
            pattern = tmpdir_path / f"{video.stem}_%06d.jpg"
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video),
                    "-vf",
                    "fps=1",
                    "-q:v",
                    "2",
                    str(pattern),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            for frame in sorted(tmpdir_path.glob(f"{video.stem}_*.jpg")):
                rows.append((compute_blur_score(frame), f"{video}::{frame.name}"))

if not rows:
    raise SystemExit(f"No supported images or videos found under {input_dir}")

rows.sort(key=lambda item: item[0])
scores = [score for score, _ in rows]

def pct(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(max(int(round((len(sorted_values) - 1) * q)), 0), len(sorted_values) - 1)
    return sorted_values[idx]

print(f"Input dir: {input_dir}")
print(f"Image count: {len(rows)}")
print(f"Min score: {scores[0]:.3f}")
print(f"Median score: {statistics.median(scores):.3f}")
print(f"P10 score: {pct(scores, 0.10):.3f}")
print(f"P20 score: {pct(scores, 0.20):.3f}")
print(f"P25 score: {pct(scores, 0.25):.3f}")
print(f"P75 score: {pct(scores, 0.75):.3f}")
print(f"Max score: {scores[-1]:.3f}")
print()
print("Lowest-scoring images:")
for score, path in rows[:top_k]:
    print(f"  {score:10.3f}  {path}")
print()
print("Highest-scoring images:")
for score, path in rows[-top_k:]:
    print(f"  {score:10.3f}  {path}")
print()
print("Threshold hints:")
print(f"  Conservative: {pct(scores, 0.10):.3f}")
print(f"  Balanced:     {pct(scores, 0.20):.3f}")
print(f"  Aggressive:   {pct(scores, 0.25):.3f}")
PY
