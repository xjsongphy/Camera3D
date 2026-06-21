#!/usr/bin/env python3
"""Build docs/lab3/report_assets from existing lab3 dormitory runs.

Pure numpy + PIL (no nerfstudio / venv). Produces:
  report_assets/cross_fps/grid_<view>.png   GT | 3DGS | NeRF across fps2/4/5/8
  report_assets/methods/*.png               GT / render / error images for report.html
  report_assets/pointcloud/*.json           downsampled point clouds for three.js
  report_assets/metrics.json                metrics summary for client-side charts

Pairing for report panels prefers backend-native held-out order when a render
directory is a positional bundle (e.g. 3DGS writes ``00000.png``, ``00001.png``
... for the 1st/2nd/... held-out view, not for canonical frame numbers).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "lab3" / "report_assets"
RUNS = {
    "fps2": "20260620_160035_dormitory_fps2",
    "fps4": "20260620_200014_dormitory_fps4",
    "fps5": "20260619_114307_dormitory_fps5",
    "fps8": "20260621_110244_dormitory_fps8",
    "boya_close": "20260621_172005_boya_close",
    "boya_far": "20260621_192805_boya_far",
}
FPS_ORDER = ["fps2", "fps4", "fps5", "fps8"]
THUMB_H = 300  # target row height for montages


def run_dir(tag: str) -> Path:
    return ROOT / "outputs" / "lab3" / RUNS[tag]


def num_suffix(stem: str) -> int | None:
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


def load_img(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_h(img: np.ndarray, h: int) -> np.ndarray:
    H, W = img.shape[:2]
    w = max(1, int(round(W * h / H)))
    return np.asarray(Image.fromarray(img).resize((w, h), Image.Resampling.BILINEAR))


def find_render(render_root: Path, test_name: str) -> Path | None:
    """Match a test image to its render by numeric suffix value."""
    if not render_root.is_dir():
        return None
    target = num_suffix(Path(test_name).stem)
    candidates = [p for p in render_root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if target is None:
        return candidates[0] if candidates else None
    by_val = {num_suffix(p.stem): p for p in candidates if num_suffix(p.stem) is not None}
    return by_val.get(target)


def list_render_images(render_root: Path) -> list[Path]:
    if not render_root.is_dir():
        return []
    return sorted(
        path for path in render_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def is_positional_render_bundle(render_root: Path, ordered: list[Path]) -> bool:
    if not ordered or not all(path.stem.isdigit() for path in ordered):
        return False
    gt_dir = render_root.parent / "gt"
    if not gt_dir.is_dir():
        return False
    gt_images = list_render_images(gt_dir)
    return len(gt_images) == len(ordered)


def resolve_render(render_root: Path, test_name: str, index: int) -> Path | None:
    ordered = list_render_images(render_root)
    if is_positional_render_bundle(render_root, ordered):
        return ordered[index] if index < len(ordered) else None
    return find_render(render_root, test_name) or (ordered[index] if index < len(ordered) else None)


def dgs_render_dir(tag: str) -> Path:
    base = run_dir(tag) / "results" / "3dgs" / "test"
    dirs = sorted(base.glob("ours_*/renders"))
    return dirs[-1] if dirs else base


def nerf_render_dir(tag: str) -> Path:
    return run_dir(tag) / "results" / "nerf" / "renders" / "test" / "rgb"


def hstack_white(tiles: list[np.ndarray], h: int, pad: int = 6) -> np.ndarray:
    rows = [resize_h(t, h) for t in tiles]
    total_w = sum(t.shape[1] for t in rows) + pad * (len(rows) + 1)
    canvas = np.full((h + 2 * pad, total_w, 3), 255, dtype=np.uint8)
    x = pad
    for t in rows:
        canvas[pad:pad + h, x:x + t.shape[1]] = t
        x += t.shape[1] + pad
    return canvas


def labeled_row(label: str, tiles: list[np.ndarray], h: int) -> np.ndarray:
    row = hstack_white(tiles, h)
    lw = 150
    bar = np.full((row.shape[0], lw, 3), 245, dtype=np.uint8)
    img = Image.fromarray(bar)
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.text((12, row.shape[0] // 2 - 10), label, fill=(20, 20, 20))
    return np.hstack([np.asarray(img), row])


def build_cross_fps(test_view: str, tags: list[str], outname: str) -> None:
    tiles_by_fps: list[np.ndarray] = []
    used_tags: list[str] = []
    for tag in tags:
        rd = run_dir(tag)
        gt_path = rd / "prepared" / "images" / test_view
        if not gt_path.exists():
            continue
        gt = load_img(gt_path)
        dgs = find_render(dgs_render_dir(tag), test_view)
        nerf = find_render(nerf_render_dir(tag), test_view)
        # Skip the whole row unless BOTH renders resolved (pairing is only
        # reliable for runs whose 3DGS renders use full-set image indexing;
        # fps2/fps5 used eval-split positional indexing and are excluded here).
        if dgs is None or nerf is None:
            continue
        row_tiles = [gt, load_img(dgs), load_img(nerf)]
        tiles_by_fps.append(labeled_row(tag.upper(), row_tiles, THUMB_H))
        used_tags.append(tag)
    if not tiles_by_fps:
        return
    # equalize widths
    w = max(t.shape[1] for t in tiles_by_fps)
    stack = np.full((sum(t.shape[0] for t in tiles_by_fps), w, 3), 255, dtype=np.uint8)
    y = 0
    for t in tiles_by_fps:
        stack[y:y + t.shape[0], :t.shape[1]] = t
        y += t.shape[0]
    col_hdr = header_bar(["", "GT", "3DGS", "NeRF"], stack.shape[1], 150)
    final = np.vstack([col_hdr, stack])
    OUT.joinpath("cross_fps").mkdir(parents=True, exist_ok=True)
    out_path = OUT / "cross_fps" / f"{outname}_{Path(test_view).stem}.png"
    Image.fromarray(final).save(out_path)
    print(f"wrote {out_path}  (rows: {used_tags})")


def placeholder(shape: tuple[int, ...], text: str) -> np.ndarray:
    h, w = shape[0], shape[1]
    img = Image.fromarray(np.full((h, w, 3), 235, dtype=np.uint8))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.text((w // 2 - 60, h // 2 - 10), text, fill=(120, 120, 120))
    return np.asarray(img)


def header_bar(labels: list[str], width: int, first_col_w: int) -> np.ndarray:
    bar = np.full((34, width, 3), 250, dtype=np.uint8)
    img = Image.fromarray(bar)
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    # Distribute labels evenly. If first_col_w>0, the first label sits over the
    # tag column and the rest span the remaining width; otherwise all labels
    # share the full width equally.
    n = len(labels)
    if first_col_w > 0 and n >= 2:
        d.text((10, 8), labels[0], fill=(40, 40, 40))
        remain = width - first_col_w
        col_w = remain / (n - 1)
        for i, label in enumerate(labels[1:]):
            d.text((int(first_col_w + col_w * (i + 0.5)) - 18, 8), label, fill=(40, 40, 40))
    else:
        col_w = width / n
        for i, label in enumerate(labels):
            d.text((int(col_w * (i + 0.5)) - 18, 8), label, fill=(40, 40, 40))
    return np.asarray(img)


def _error_map_magma(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    h, w = gt.shape[:2]
    p = np.asarray(Image.fromarray(pred).resize((w, h), Image.Resampling.BILINEAR)).astype(np.float32)
    g = gt.astype(np.float32)
    diff = np.mean(np.abs(g - p), axis=-1)
    peak = float(diff.max())
    if peak <= 0:
        return np.zeros_like(diff)
    return (cm.magma(diff / peak)[:, :, :3] * 255).astype(np.uint8)


def save_report_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    print("wrote", path)


def build_qualitative(tag: str, views: list[str]) -> None:
    """Export correctly-paired GT / render / error images for one fps."""
    dest = OUT / "methods"
    dest.mkdir(parents=True, exist_ok=True)
    for idx, view in enumerate(views):
        gt_path = run_dir(tag) / "prepared" / "images" / view
        if not gt_path.exists():
            continue
        gt = load_img(gt_path)
        dgs = resolve_render(dgs_render_dir(tag), view, index=idx)
        nerf = resolve_render(nerf_render_dir(tag), view, index=idx)
        if dgs is None or nerf is None:
            print(f"skip {tag} {view}: missing render (3dgs={dgs}, nerf={nerf})")
            continue
        dgs_im = load_img(dgs)
        nerf_im = load_img(nerf)
        stem = f"{tag}_{Path(view).stem}"
        save_report_image(dest / f"{stem}_gt.png", gt)
        save_report_image(dest / f"{stem}_3dgs.png", dgs_im)
        save_report_image(dest / f"{stem}_3dgs_err.png", _error_map_magma(gt, dgs_im))
        save_report_image(dest / f"{stem}_nerf.png", nerf_im)
        save_report_image(dest / f"{stem}_nerf_err.png", _error_map_magma(gt, nerf_im))


def build_3dgs_only_qualitative(tag: str, views: list[str]) -> None:
    """Export GT / 3DGS / error images for appendix-only scenes such as Boya.

    Uses 3DGS test-set GT images (guaranteed to match renders pixel-perfectly)
    instead of the original prepared images which may differ in resolution/crop.
    """
    dest = OUT / "methods"
    dest.mkdir(parents=True, exist_ok=True)
    render_root = dgs_render_dir(tag)
    gt_dir = render_root.parent / "gt"  # 3DGS test-set GT, paired with renders
    for idx, view in enumerate(views):
        # Use 3DGS test GT (same resolution/processing as renders)
        gt_path = gt_dir / f"{idx:05d}.png"
        if not gt_path.exists():
            print(f"skip {tag} {view}: missing gt {gt_path}")
            continue
        gt = load_img(gt_path)
        dgs = render_root / f"{idx:05d}.png"
        if not dgs.exists():
            print(f"skip {tag} {view}: missing 3dgs render {dgs}")
            continue
        dgs_im = load_img(dgs)
        stem = f"{tag}_{Path(view).stem}"
        save_report_image(dest / f"{stem}_gt.png", gt)
        save_report_image(dest / f"{stem}_3dgs.png", dgs_im)
        save_report_image(dest / f"{stem}_3dgs_err.png", _error_map_magma(gt, dgs_im))


def copy_qualitative() -> None:
    dest = OUT / "methods"
    dest.mkdir(parents=True, exist_ok=True)
    for tag in FPS_ORDER:
        qdir = run_dir(tag) / "qualitative"
        if not qdir.is_dir():
            continue
        for img in sorted(qdir.glob("comparison_*.png")):
            target = dest / f"{tag}_{img.name}"
            Image.open(img).convert("RGB").save(target)
            print("copied", target)


# ---------------------------------------------------------------- point clouds

def parse_binary_vertex_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz[N,3] float32, rgb[N,3] uint8) for our binary PLYs."""
    with open(path, "rb") as f:
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if line.strip() == b"end_header":
                break
        header = header_bytes.decode("ascii", errors="replace")
    # parse properties for the vertex element up to the first non-vertex element
    lines = header.splitlines()
    props: list[tuple[str, str]] = []
    in_vertex = False
    vertex_count = 0
    for line in lines:
        toks = line.split()
        if toks[:1] == ["element"]:
            if toks[1] == "vertex":
                in_vertex = True
                vertex_count = int(toks[2])
            else:
                in_vertex = False
        elif toks[:1] == ["property"] and in_vertex:
            props.append((toks[1], toks[2]))  # (type, name)
    # build numpy structured dtype
    np_types = {"float": "f4", "double": "f8", "uchar": "u1", "uint8": "u1", "int": "i4"}
    dtype = [(name, np_types.get(ptype, "f4")) for ptype, name in props]
    data_offset = len(header_bytes)
    raw = np.memmap(str(path), dtype=np.dtype(dtype), mode="r", offset=data_offset,
                    shape=(vertex_count,))
    xyz = np.stack([np.asarray(raw["x"]), np.asarray(raw["y"]), np.asarray(raw["z"])], axis=1).astype(np.float32)
    rgb = np.zeros((vertex_count, 3), dtype=np.uint8)
    if "red" in raw.dtype.names and "green" in raw.dtype.names and "blue" in raw.dtype.names:
        rgb[:, 0] = np.asarray(raw["red"])
        rgb[:, 1] = np.asarray(raw["green"])
        rgb[:, 2] = np.asarray(raw["blue"])
    elif "f_dc_0" in raw.dtype.names:
        # 3DGS DC color: C = 0.5 + 0.28209479 * f_dc
        dc = np.stack([np.asarray(raw["f_dc_0"]), np.asarray(raw["f_dc_1"]),
                       np.asarray(raw["f_dc_2"])], axis=1).astype(np.float32)
        rgb = np.clip(0.5 + 0.28209479177387814 * dc, 0.0, 1.0)
        rgb = (rgb * 255).astype(np.uint8)
    return xyz, rgb


def parse_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", errors="replace") as f:
        header = []
        while True:
            line = f.readline()
            header.append(line)
            if "end_header" in line:
                break
        count = 0
        for line in header:
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
        arr = np.loadtxt(f, max_rows=count)
    xyz = arr[:, :3].astype(np.float32)
    rgb = arr[:, 3:6].astype(np.uint8) if arr.shape[1] >= 6 else np.zeros((count, 3), np.uint8)
    return xyz, rgb


def downsample(xyz: np.ndarray, rgb: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    n = xyz.shape[0]
    if n <= target:
        return xyz, rgb
    # deterministic stride + center-biased random via np default (seeded) -> use systematic stride
    idx = np.linspace(0, n - 1, target).astype(np.int64)
    return xyz[idx], rgb[idx]


def write_pointcloud_json(xyz: np.ndarray, rgb: np.ndarray, out: Path, label: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "n": int(xyz.shape[0]),
        "center": [float(c) for c in xyz.mean(axis=0)],
        "extent": [float(e) for e in (xyz.max(axis=0) - xyz.min(axis=0))],
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    # binary: float32 xyz interleaved, then uint8 rgb, raw bytes
    (out.with_suffix(".bin")).write_bytes(
        xyz.astype(np.float32).tobytes() + rgb.astype(np.uint8).tobytes()
    )
    print(f"wrote {out} ({payload['n']} pts, {out.with_suffix('.bin').stat().st_size//1024} KB)")


def build_pointclouds() -> None:
    pcdir = OUT / "pointcloud"
    # 3DGS fps4
    p = run_dir("fps4") / "geometry" / "3dgs" / "gaussians.ply"
    if p.exists():
        xyz, rgb = parse_binary_vertex_ply(p)
        xyz, rgb = downsample(xyz, rgb, 50000)
        write_pointcloud_json(xyz, rgb, pcdir / "3dgs_fps4.json", "3DGS gaussians (fps4, 100k subsample)")
    # NeuS mesh fps4 -> vertices
    p = run_dir("fps4") / "geometry" / "neus" / "sdf_mesh.ply"
    if p.exists():
        xyz, rgb = parse_binary_vertex_ply(p)
        rgb[:] = 180  # uniform grey: mesh has no vertex color
        xyz, rgb = downsample(xyz, rgb, 50000)
        write_pointcloud_json(xyz, rgb, pcdir / "neus_fps4.json", "NeuS SDF mesh vertices (fps4)")
    # SfM dense fps5
    p = run_dir("fps5") / "geometry" / "sfm" / "dense.ply"
    if p.exists():
        xyz, rgb = parse_binary_vertex_ply(p)
        xyz, rgb = downsample(xyz, rgb, 50000)
        write_pointcloud_json(xyz, rgb, pcdir / "sfm_dense_fps5.json", "COLMAP dense cloud (fps5, 100k subsample)")
    # NeRF sparse fps4 (ascii)
    p = run_dir("fps4") / "geometry" / "nerf" / "sparse_pc.ply"
    if p.exists():
        xyz, rgb = parse_ascii_ply(p)
        xyz, rgb = downsample(xyz, rgb, 50000)
        write_pointcloud_json(xyz, rgb, pcdir / "nerf_sparse_fps4.json", "Nerfacto sparse point cloud (fps4)")
    # 3DGS fps8 for cross-fps geometry artifact comparison
    p = run_dir("fps8") / "geometry" / "3dgs" / "gaussians.ply"
    if p.exists():
        xyz, rgb = parse_binary_vertex_ply(p)
        xyz, rgb = downsample(xyz, rgb, 50000)
        write_pointcloud_json(xyz, rgb, pcdir / "3dgs_fps8.json", "3DGS gaussians (fps8, 80k subsample)")


def build_metrics_json() -> None:
    rows = {}
    for tag in FPS_ORDER:
        mpath = run_dir(tag) / "metrics.csv"
        if not mpath.exists():
            continue
        lines = mpath.read_text().strip().splitlines()
        header = lines[0].split(",")
        for line in lines[1:]:
            cells = line.split(",")
            rec = dict(zip(header, cells))
            rows.setdefault(tag, {})[rec["method"]] = rec
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT / "metrics.json")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Correctly-paired per-fps qualitative panels (3DGS & NeRF same scene as GT).
    # fps2 3DGS only saved 16 eval-split renders (no full-set index), so it is
    # skipped here; its story is carried by the metrics chart.
    views = ["vid_001_dormitory_000010.jpg", "vid_001_dormitory_000020.jpg", "vid_001_dormitory_000030.jpg"]
    for tag in ("fps4", "fps8"):
        build_qualitative(tag, views)
    build_3dgs_only_qualitative("boya_close", ["img_000010.jpg"])
    build_3dgs_only_qualitative("boya_far", ["img_000020.jpg"])
    build_pointclouds()
    build_metrics_json()


if __name__ == "__main__":
    main()
