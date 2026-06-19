from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a NeuS SDF zero-level mesh")
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--bound", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.resolution < 512 or args.resolution % 512 != 0:
        raise SystemExit("resolution must be a positive multiple of 512")

    from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
    from nerfstudio.fields.sdf_field import SDFField
    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, _, _ = eval_setup(args.load_config, test_mode="inference")
    if not hasattr(pipeline.model, "field"):
        raise RuntimeError("Loaded model has no SDF field; expected NeuS or NeuS-facto")
    field = pipeline.model.field
    if not isinstance(field, SDFField):
        raise RuntimeError(f"Loaded field is {type(field).__name__}, expected SDFField")

    bound = float(args.bound)
    mesh = generate_mesh_with_multires_marching_cubes(
        geometry_callable_field=lambda points: field.forward_geonetwork(points)[:, 0].contiguous(),
        resolution=args.resolution,
        bounding_box_min=(-bound, -bound, -bound),
        bounding_box_max=(bound, bound, bound),
        isosurface_threshold=0.0,
        coarse_mask=None,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output_path)
    print(f"NeuS mesh: {args.output_path}")


if __name__ == "__main__":
    main()
