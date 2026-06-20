from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path


def _image_names(images_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    )


def _sequential_pairs(names: list[str], overlap: int, quadratic_overlap: bool) -> list[tuple[str, str]]:
    offsets = set(range(1, overlap + 1))
    if quadratic_overlap:
        step = 1
        while step <= overlap:
            offsets.add(step)
            step *= 2
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(names):
        for offset in sorted(offsets):
            right_index = index + offset
            if right_index < len(names):
                pairs.append((left, names[right_index]))
    return pairs


def _write_pairs(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for left, right in pairs:
            handle.write(f"{left} {right}\n")


def _run_build_sfm(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--matcher", required=True)
    parser.add_argument("--pairing", choices=("sequential", "exhaustive"), required=True)
    parser.add_argument("--pair-overlap", type=int, default=10)
    parser.add_argument("--quadratic-overlap", type=int, default=1)
    parser.add_argument("--camera-model", required=True)
    parser.add_argument("--single-camera", type=int, choices=(0, 1), required=True)
    args = parser.parse_args(argv)

    try:
        import pycolmap
        from hloc import extract_features, match_features, reconstruction
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "HLOC-based SfM requires Python packages 'hloc', 'pycolmap', and their dependencies "
            f"in the selected interpreter. Missing import: {exc.name}"
        ) from exc

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    names = _image_names(args.images)
    if not names:
        raise SystemExit(f"No supported images found in {args.images}")

    if args.pairing == "exhaustive":
        pairs = list(combinations(names, 2))
    else:
        pairs = _sequential_pairs(
            names,
            overlap=args.pair_overlap,
            quadratic_overlap=bool(args.quadratic_overlap),
        )
    if not pairs:
        raise SystemExit("No image pairs were generated for HLOC SfM")

    pairs_path = args.work_dir / f"pairs-{args.pairing}.txt"
    _write_pairs(pairs_path, pairs)

    try:
        feature_conf = extract_features.confs[args.extractor]
    except KeyError as exc:
        raise SystemExit(
            f"Unknown HLOC extractor preset '{args.extractor}'. "
            f"Available presets: {sorted(extract_features.confs)}"
        ) from exc
    try:
        matcher_conf = match_features.confs[args.matcher]
    except KeyError as exc:
        raise SystemExit(
            f"Unknown HLOC matcher preset '{args.matcher}'. "
            f"Available presets: {sorted(match_features.confs)}"
        ) from exc
    feature_path = extract_features.main(
        feature_conf,
        args.images,
        args.work_dir,
        image_list=names,
    )
    match_path = match_features.main(
        matcher_conf,
        pairs_path,
        feature_conf["output"],
        args.work_dir,
    )

    camera_mode = (
        pycolmap.CameraMode.SINGLE
        if args.single_camera
        else pycolmap.CameraMode.PER_IMAGE
    )
    reconstruction.main(
        args.model_dir,
        args.images,
        pairs_path,
        feature_path,
        match_path,
        image_list=names,
        camera_mode=camera_mode,
        image_options={"camera_model": args.camera_model},
        verbose=True,
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "build-sfm":
        raise SystemExit("usage: hloc_bridge.py build-sfm ...")
    _run_build_sfm(sys.argv[2:])


if __name__ == "__main__":
    main()
