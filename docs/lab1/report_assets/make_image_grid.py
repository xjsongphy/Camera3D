from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a labeled image grid.")
    parser.add_argument("--inputs", nargs="+", required=True, help="input image paths")
    parser.add_argument("--labels", nargs="*", default=None, help="label for each image")
    parser.add_argument("--output", required=True, help="output image path")
    parser.add_argument("--title", default="", help="optional figure title")
    parser.add_argument("--cols", type=int, default=2, help="number of columns")
    parser.add_argument("--figscale", type=float, default=5.0, help="base subplot size")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_paths = [Path(p) for p in args.inputs]
    labels = args.labels or [path.stem for path in input_paths]
    if len(labels) != len(input_paths):
        raise SystemExit("labels count must match inputs count")

    cols = max(1, args.cols)
    rows = (len(input_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * args.figscale, rows * args.figscale))
    axes_list = axes.flat if hasattr(axes, "flat") else [axes]

    for ax, path, label in zip(axes_list, input_paths, labels, strict=True):
        image = mpimg.imread(path)
        ax.imshow(image)
        ax.set_title(label, fontsize=12)
        ax.axis("off")

    for ax in list(axes_list)[len(input_paths):]:
        ax.axis("off")

    if args.title:
        fig.suptitle(args.title, fontsize=18)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
