from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select task2 subsequence candidates from existing task1 outputs without rerunning reconstruction."
    )
    parser.add_argument(
        "--task1-images-txt",
        default="outputs/lab1/task1/S1-2_fps30/sparse/0/images.txt",
        help="task1 full-sequence COLMAP images.txt path",
    )
    parser.add_argument(
        "--task2-summary",
        default="outputs/lab1/task2/S1-2_fps30/summary.csv",
        help="optional existing task2 summary.csv for filtering/annotating low-registration windows",
    )
    parser.add_argument("--step", type=int, default=30, help="candidate window start step in frames")
    parser.add_argument(
        "--output-csv",
        default="outputs/lab1/task2/S1-2_fps30/candidate_segments.csv",
        help="where to save the ranked candidate table",
    )
    return parser.parse_args()


def quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def parse_colmap_centers(images_txt: Path) -> tuple[np.ndarray, list[str]]:
    centers: list[np.ndarray] = []
    names: list[str] = []
    with images_txt.open("r", encoding="utf-8") as f:
        while True:
            pose_line = f.readline()
            if not pose_line:
                break
            pose_line = pose_line.strip()
            if not pose_line or pose_line.startswith("#"):
                continue
            parts = pose_line.split()
            if len(parts) < 10 or not parts[0].isdigit():
                raise ValueError(f"Malformed COLMAP images.txt line: {pose_line}")
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = parts[9]
            r = quat_to_rot(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=float)
            c = -r.T @ t
            centers.append(c)
            names.append(name)
            _ = f.readline()
    order = np.argsort(np.array(names))
    centers_arr = np.array(centers)[order]
    names_sorted = [names[i] for i in order]
    return centers_arr, names_sorted


@dataclass
class Candidate:
    start: int
    end: int
    length: int
    endpoint_ratio: float
    path_length: float
    endpoint_distance: float
    bbox_diag: float
    history_common_ratio: float | None
    history_ate: float | None


def load_task2_history(summary_csv: Path) -> dict[tuple[int, int], tuple[float, float]]:
    history: dict[tuple[int, int], tuple[float, float]] = {}
    if not summary_csv.exists():
        return history
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subseq = row["subseq"]
            try:
                range_part = subseq.split("_")[-1]
                start_s, end_s = range_part.split("-")
                start = int(start_s)
                end = int(end_s)
            except Exception:
                continue
            subset_frames = int(row["subset_frames"])
            common_registered = int(row["common_registered"])
            ate = float(row["ate"])
            history[(start, end)] = (common_registered / subset_frames if subset_frames else 0.0, ate)
    return history


def score_scan(c: Candidate) -> float:
    reg = c.history_common_ratio if c.history_common_ratio is not None else 1.0
    return (
        c.endpoint_ratio * 3.0
        + min(c.bbox_diag / 120.0, 1.0)
        + min(c.path_length / 140.0, 1.0)
        + reg * 2.0
        - abs(c.length - 300) / 500.0
    )


def score_return_mid(c: Candidate) -> float:
    reg = c.history_common_ratio if c.history_common_ratio is not None else 1.0
    return (
        (1.0 - c.endpoint_ratio) * 3.5
        + min(c.bbox_diag / 90.0, 1.0)
        + min(c.path_length / 170.0, 1.0)
        + reg * 2.0
        - abs(c.length - 420) / 600.0
    )


def score_return_long(c: Candidate) -> float:
    reg = c.history_common_ratio if c.history_common_ratio is not None else 1.0
    return (
        (1.0 - c.endpoint_ratio) * 4.0
        + min(c.bbox_diag / 180.0, 1.2)
        + min(c.path_length / 340.0, 1.2)
        + reg * 2.0
        - abs(c.length - 900) / 1000.0
    )


def build_candidates(centers: np.ndarray, step: int, history: dict[tuple[int, int], tuple[float, float]]) -> list[Candidate]:
    lengths = [240, 300, 360, 420, 480, 540, 600, 720, 900]
    candidates: list[Candidate] = []
    n = len(centers)
    for length in lengths:
        for start0 in range(0, n - length + 1, step):
            arr = centers[start0 : start0 + length]
            endpoint_distance = float(np.linalg.norm(arr[-1] - arr[0]))
            path_length = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
            endpoint_ratio = endpoint_distance / path_length if path_length > 0 else 0.0
            bbox_diag = float(np.linalg.norm(np.ptp(arr, axis=0)))
            start = start0 + 1
            end = start0 + length
            history_common_ratio, history_ate = history.get((start, end), (None, None))
            candidates.append(
                Candidate(
                    start=start,
                    end=end,
                    length=length,
                    endpoint_ratio=endpoint_ratio,
                    path_length=path_length,
                    endpoint_distance=endpoint_distance,
                    bbox_diag=bbox_diag,
                    history_common_ratio=history_common_ratio,
                    history_ate=history_ate,
                )
            )
    return candidates


def pick_best(candidates: list[Candidate], predicate, scorer) -> Candidate | None:
    pool = [c for c in candidates if predicate(c)]
    if not pool:
        return None
    return max(pool, key=scorer)


def write_candidates_csv(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "start",
                "end",
                "length",
                "endpoint_ratio",
                "path_length",
                "endpoint_distance",
                "bbox_diag",
                "history_common_ratio",
                "history_ate",
            ],
        )
        writer.writeheader()
        for c in sorted(candidates, key=lambda x: (x.length, x.start)):
            writer.writerow(
                {
                    "start": c.start,
                    "end": c.end,
                    "length": c.length,
                    "endpoint_ratio": f"{c.endpoint_ratio:.8f}",
                    "path_length": f"{c.path_length:.8f}",
                    "endpoint_distance": f"{c.endpoint_distance:.8f}",
                    "bbox_diag": f"{c.bbox_diag:.8f}",
                    "history_common_ratio": "" if c.history_common_ratio is None else f"{c.history_common_ratio:.8f}",
                    "history_ate": "" if c.history_ate is None else f"{c.history_ate:.8f}",
                }
            )


def print_candidate(label: str, name: str, c: Candidate) -> None:
    reg_text = "n/a" if c.history_common_ratio is None else f"{c.history_common_ratio:.3f}"
    print(
        f"{label}: {name} -> {c.start}:{c.end}:{name} | "
        f"len={c.length}, endpoint_ratio={c.endpoint_ratio:.4f}, "
        f"bbox_diag={c.bbox_diag:.1f}, path={c.path_length:.1f}, history_reg={reg_text}"
    )


def main() -> None:
    args = parse_args()
    centers, _ = parse_colmap_centers(Path(args.task1_images_txt))
    history = load_task2_history(Path(args.task2_summary))
    candidates = build_candidates(centers, step=args.step, history=history)

    scan = pick_best(
        candidates,
        lambda c: c.endpoint_ratio > 0.88 and c.bbox_diag > 90 and c.length <= 420 and (c.history_common_ratio is None or c.history_common_ratio >= 0.7),
        score_scan,
    )
    return_mid = pick_best(
        candidates,
        lambda c: 0.32 <= c.endpoint_ratio <= 0.50 and c.bbox_diag > 60 and 300 <= c.length <= 540 and (c.history_common_ratio is None or c.history_common_ratio >= 0.5),
        score_return_mid,
    )
    return_long = pick_best(
        candidates,
        lambda c: c.endpoint_ratio < 0.35 and c.bbox_diag > 150 and c.length >= 720 and (c.history_common_ratio is None or c.history_common_ratio >= 0.7),
        score_return_long,
    )

    print("Recommended task2 candidates (no reconstruction rerun involved):")
    if return_mid is not None:
        print_candidate("1", "return_mid", return_mid)
    if scan is not None:
        print_candidate("2", "scan_stable", scan)
    if return_long is not None:
        print_candidate("3", "return_long", return_long)

    print("\nSuggested command:")
    parts = []
    if return_mid is not None:
        parts.append(f"--subseq {return_mid.start}:{return_mid.end}:return_mid")
    if scan is not None:
        parts.append(f"--subseq {scan.start}:{scan.end}:scan_stable")
    if return_long is not None:
        parts.append(f"--subseq {return_long.start}:{return_long.end}:return_long")
    print("uv run lab1 task2 --source-fps 30 --stage all " + " ".join(parts))

    write_candidates_csv(Path(args.output_csv), candidates)
    print(f"\nSaved candidate table: {args.output_csv}")


if __name__ == "__main__":
    main()
