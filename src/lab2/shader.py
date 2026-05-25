from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .scene_genertor import build_runtime_scene_dict


@dataclass
class CameraConfig:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: torch.Tensor
    t: torch.Tensor


@dataclass
class ProjectorConfig:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: torch.Tensor
    t: torch.Tensor


@dataclass
class LightConfig:
    ambient: float = 0.05


class StructuredLightRenderer:
    """Single shader: Mitsuba rendering backend + torch tensor data API."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        spp: int = 64,
        mi_variant: str = "cuda_ad_rgb",
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.spp = int(spp)
        self.mi_variant = mi_variant

        self.camera: CameraConfig | None = None
        self.projector: ProjectorConfig | None = None
        self.lights = LightConfig()
        self.patterns: torch.Tensor | None = None

        self._scene_loaded = False
        self._mi_scene_file: Path | None = None
        self._scene_name: str = "sl_plane_diffuse"

        self._depth: torch.Tensor | None = None
        self._gt_corr: torch.Tensor | None = None
        self._autodiff_gain: torch.Tensor | None = None

    def _require_mitsuba(self):
        try:
            import mitsuba as mi  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Mitsuba is required for rendering but is not available.") from exc
        try:
            mi.set_variant(self.mi_variant)
        except Exception:
            # Fallback for environments without CUDA-enabled Mitsuba.
            mi.set_variant("llvm_ad_rgb")
        return mi

    def _to_tensor(self, x: Any, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if shape is not None and tuple(t.shape) != shape:
            raise ValueError(f"Expected shape {shape}, got {tuple(t.shape)}")
        return t

    def _parse_intrinsics_extrinsics(self, config: dict[str, Any], is_camera: bool) -> CameraConfig | ProjectorConfig:
        w = int(config["width"])
        h = int(config["height"])

        if "K" in config:
            K = self._to_tensor(config["K"], (3, 3))
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
        else:
            fx = float(config["fx"])
            fy = float(config["fy"])
            cx = float(config.get("cx", (w - 1) * 0.5))
            cy = float(config.get("cy", (h - 1) * 0.5))

        R = self._to_tensor(config.get("R", torch.eye(3)), (3, 3))
        t = self._to_tensor(config.get("t", torch.zeros(3)), (3,))

        if is_camera:
            return CameraConfig(w, h, fx, fy, cx, cy, R, t)
        return ProjectorConfig(w, h, fx, fy, cx, cy, R, t)

    def set_camera(self, camera_config: dict[str, Any]) -> None:
        self.camera = self._parse_intrinsics_extrinsics(camera_config, is_camera=True)  # type: ignore[assignment]
        self._gt_corr = None

    def set_projector(self, projector_config: dict[str, Any]) -> None:
        self.projector = self._parse_intrinsics_extrinsics(projector_config, is_camera=False)  # type: ignore[assignment]
        self._gt_corr = None

    def set_lights(self, light_config: dict[str, Any]) -> None:
        self.lights = LightConfig(ambient=float(light_config.get("ambient", 0.05)))

    def set_scene_name(self, scene_name: str) -> None:
        self._scene_name = scene_name

    def set_patterns(self, patterns: torch.Tensor) -> None:
        if patterns.ndim != 2:
            raise ValueError("patterns must be [K, Wp]")
        self.patterns = patterns.to(device=self.device, dtype=self.dtype)

    def update_patterns(self, patterns: torch.Tensor) -> None:
        self.set_patterns(patterns)

    def load_scene(self, scene_path: str | Path | None = None, cache_path: str | Path | None = None) -> None:
        """
        Load Mitsuba scene for rendering.

        - `scene_path` .xml: use as Mitsuba scene file. Must pair with cache_path.
        - `scene_path` .npz: load depth/gt_corr cache only.
        - `None`: use internal default scene.
        """
        if self.camera is None:
            raise RuntimeError("set_camera() must be called before load_scene().")

        self._mi_scene_file = None
        self._depth = None
        self._gt_corr = None

        if scene_path is not None:
            p = Path(scene_path)
            if p.suffix.lower() == ".xml":
                self._mi_scene_file = p
                if cache_path is None:
                    raise ValueError(
                        "When loading an XML render scene, cache_path (.npz with depth/gt_corr) is required "
                        "to keep images and supervision geometry aligned."
                    )
            elif p.suffix.lower() == ".npz":
                cache_path = p
            else:
                raise ValueError("scene_path must be .xml or .npz or None")

        if cache_path is not None:
            c = Path(cache_path)
            if c.suffix.lower() != ".npz":
                raise ValueError("cache_path must be a .npz file")
            data = np.load(c, allow_pickle=False)
            H, W = self.camera.height, self.camera.width
            if "depth" in data.files:
                self._depth = self._to_tensor(data["depth"], (H, W))
            if "gt_corr" in data.files:
                self._gt_corr = self._to_tensor(data["gt_corr"], (H, W))

        if self._depth is None and self._gt_corr is None:
            if self._mi_scene_file is not None:
                raise ValueError(
                    "XML scene provided but no usable depth/gt_corr cache found. "
                    "Pass cache_path with depth or gt_corr from the same geometry."
                )
            H, W = self.camera.height, self.camera.width
            v, u = torch.meshgrid(
                torch.linspace(0.0, 1.0, H, device=self.device, dtype=self.dtype),
                torch.linspace(0.0, 1.0, W, device=self.device, dtype=self.dtype),
                indexing="ij",
            )
            self._depth = 1.0 + 0.2 * (u - 0.5) + 0.1 * torch.sin(2 * np.pi * v)

        self._scene_loaded = True

    def _pixel_rays_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.camera is None:
            raise RuntimeError("Camera is not configured.")
        cam = self.camera

        v, u = torch.meshgrid(
            torch.arange(cam.height, device=self.device, dtype=self.dtype),
            torch.arange(cam.width, device=self.device, dtype=self.dtype),
            indexing="ij",
        )

        x = (u - cam.cx) / cam.fx
        y = (v - cam.cy) / cam.fy
        dirs_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)

        R_cw = cam.R
        Cw = -(R_cw.T @ cam.t)
        dirs_w = dirs_cam @ R_cw
        dirs_w = dirs_w / torch.norm(dirs_w, dim=-1, keepdim=True)
        rays_o = Cw.view(1, 1, 3).expand(cam.height, cam.width, 3)
        return rays_o, dirs_w

    def compute_gt_corr(self) -> torch.Tensor:
        if self.camera is None or self.projector is None:
            raise RuntimeError("Camera and projector must be configured.")
        if self._depth is None and self._gt_corr is None:
            raise RuntimeError("Scene not loaded.")

        if self._gt_corr is not None:
            return self._gt_corr

        if self._depth is None:
            raise RuntimeError("Depth cache is required to compute gt_corr.")

        rays_o, rays_d = self._pixel_rays_world()
        pts_world = rays_o + rays_d * self._depth[..., None]

        proj = self.projector
        pts_proj = pts_world @ proj.R.T + proj.t.view(1, 1, 3)
        z = pts_proj[..., 2]

        valid = z > 1e-6
        x = proj.fx * (pts_proj[..., 0] / z) + proj.cx
        valid = valid & (x >= 0) & (x <= (proj.width - 1))

        clamped = x.clamp(0, proj.width - 1)
        self._gt_corr = torch.where(valid, clamped, torch.full_like(clamped, float("nan")))
        return self._gt_corr

    @property
    def gt_corr(self) -> torch.Tensor:
        return self.compute_gt_corr()

    def _prepare_autodiff_sampling(self, gt_corr: torch.Tensor, wp: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare linear sampling indices/weights for differentiable pattern lookup.
        Returns (x0, x1, w1) with shape [H, W], and invalid pixels get zeroed.
        """
        valid = torch.isfinite(gt_corr)
        x = torch.where(valid, gt_corr, torch.zeros_like(gt_corr))
        x0 = torch.floor(x).long().clamp(0, wp - 1)
        x1 = (x0 + 1).clamp(0, wp - 1)
        w1 = (x - x0.to(x.dtype)).clamp(0.0, 1.0)
        w1 = torch.where(valid, w1, torch.zeros_like(w1))
        return x0, x1, w1

    def calibrate_autodiff_gain(self, pattern_value: float = 1.0) -> torch.Tensor:
        """
        Estimate a per-pixel gain map from Mitsuba by rendering a constant pattern.
        This aligns the differentiable torch path with current scene brightness.
        """
        if self.projector is None:
            raise RuntimeError("Projector is not configured.")
        wp = self.projector.width
        p = torch.full((1, wp), float(pattern_value), device=self.device, dtype=self.dtype)
        img = self.render_images(p)[0]
        gain = (img - self.lights.ambient).clamp(min=0.0)
        self._autodiff_gain = gain.detach()
        return self._autodiff_gain

    def render_images_autodiff(
        self,
        patterns: torch.Tensor | None = None,
        gain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Differentiable torch renderer for patterns -> captured images.
        Uses gt_corr sampling + optional gain map, enabling autograd on patterns.
        """
        patterns = self._prepare_patterns(patterns)
        gt = self.gt_corr
        k, wp = patterns.shape
        x0, x1, w1 = self._prepare_autodiff_sampling(gt, wp)

        # [K, H, W] linear sample from 1D projector patterns
        p0 = patterns[:, x0]
        p1 = patterns[:, x1]
        sampled = (1.0 - w1).unsqueeze(0) * p0 + w1.unsqueeze(0) * p1

        if gain is None:
            if self._autodiff_gain is None:
                gain_t = torch.ones_like(gt, dtype=self.dtype, device=self.device)
            else:
                gain_t = self._autodiff_gain.to(device=self.device, dtype=self.dtype)
        else:
            gain_t = gain.to(device=self.device, dtype=self.dtype)
        if gain_t.shape != gt.shape:
            raise ValueError(f"gain must be shape {tuple(gt.shape)}, got {tuple(gain_t.shape)}")

        images = sampled * gain_t.unsqueeze(0) + self.lights.ambient
        return images.clamp(0.0, 1.0)

    def patterns_image_autodiff_grad(
        self,
        patterns: torch.Tensor,
        grad_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute autograd gradient d(<images, grad_images>)/d(patterns) using torch path.
        """
        p = patterns.to(device=self.device, dtype=self.dtype).detach().requires_grad_(True)
        imgs = self.render_images_autodiff(p)
        if grad_images is None:
            loss = imgs.sum()
        else:
            g = grad_images.to(device=self.device, dtype=self.dtype)
            if g.shape != imgs.shape:
                raise ValueError(f"grad_images must be shape {tuple(imgs.shape)}, got {tuple(g.shape)}")
            loss = (imgs * g).sum()
        (grad_p,) = torch.autograd.grad(loss, p, retain_graph=False, create_graph=False)
        return grad_p

    def _pattern_to_image(self, pattern_1d: torch.Tensor, hp: int) -> np.ndarray:
        p = pattern_1d.detach().to("cpu", torch.float32).numpy()
        p = np.clip(p, 0.0, 1.0)
        img = np.tile(p[None, :], (hp, 1))
        rgb = np.stack([img, img, img], axis=-1)
        return rgb

    def _make_scene_with_pattern(self, pattern_path: str):
        mi = self._require_mitsuba()
        if self._mi_scene_file is not None:
            scene = mi.load_file(str(self._mi_scene_file))
        else:
            if self.camera is None or self.projector is None:
                raise RuntimeError("Camera and projector must be configured.")
            scene_dict = build_runtime_scene_dict(
                mi=mi,
                camera=self.camera,
                projector=self.projector,
                ambient=self.lights.ambient,
                pattern_path=pattern_path,
                scene_name=self._scene_name,
            )
            scene_dict["sensor"]["sampler"]["sample_count"] = self.spp
            scene = mi.load_dict(scene_dict)
        return mi, scene

    def _render_single_pattern(self, pattern_1d: torch.Tensor) -> torch.Tensor:
        if self.projector is None:
            raise RuntimeError("Projector is not configured.")

        with TemporaryDirectory(prefix="lab2_pattern_") as td:
            td_path = Path(td)
            pattern_rgb = self._pattern_to_image(pattern_1d, self.projector.height)
            pattern_file = td_path / "pattern.exr"

            try:
                import imageio.v3 as iio

                iio.imwrite(pattern_file, pattern_rgb.astype(np.float32))
            except Exception as exc:
                raise RuntimeError("imageio is required to write temporary pattern textures.") from exc

            mi, scene = self._make_scene_with_pattern(str(pattern_file))
            img = mi.render(scene, spp=self.spp)
            img_np = np.array(img, dtype=np.float32)
            if img_np.ndim == 3 and img_np.shape[-1] == 3:
                img_np = img_np.mean(axis=-1)

            out = torch.from_numpy(img_np).to(device=self.device, dtype=self.dtype)
            out = out.clamp(0.0, 1.0)
            return out

    def _prepare_patterns(self, patterns: torch.Tensor | None = None) -> torch.Tensor:
        if not self._scene_loaded:
            raise RuntimeError("Scene not loaded. Call load_scene() first.")
        if patterns is None:
            if self.patterns is None:
                raise RuntimeError("Patterns are not set.")
            patterns = self.patterns
        patterns = patterns.to(device=self.device, dtype=self.dtype)
        if patterns.ndim != 2:
            raise ValueError("patterns must be [K, Wp]")
        return patterns

    def render_images_and_gt(self, patterns: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Unified observation API that computes both rendered images and gt correspondence
        with shared preparation logic.
        """
        patterns = self._prepare_patterns(patterns)
        imgs = [self._render_single_pattern(patterns[k]) for k in range(patterns.shape[0])]
        images = torch.stack(imgs, dim=0)
        return {
            "images": images,
            "gt_corr": self.gt_corr,
        }

    def render_images(self, patterns: torch.Tensor | None = None) -> torch.Tensor:
        return self.render_images_and_gt(patterns)["images"]

    def render_images_batch(self, patterns_batch: torch.Tensor) -> torch.Tensor:
        if patterns_batch.ndim != 3:
            raise ValueError("patterns_batch must be [B, K, Wp]")
        patterns_batch = patterns_batch.to(device=self.device, dtype=self.dtype)

        out = [self.render_images(patterns_batch[b]) for b in range(patterns_batch.shape[0])]
        return torch.stack(out, dim=0)

    def render_train_batch(self, mode: str = "mitsuba") -> dict[str, torch.Tensor]:
        if mode == "mitsuba":
            return self.render_images_and_gt()
        if mode == "autodiff":
            return {
                "images": self.render_images_autodiff(),
                "gt_corr": self.gt_corr,
            }
        raise ValueError("mode must be 'mitsuba' or 'autodiff'")

    def finite_difference(self, patterns: torch.Tensor, direction: torch.Tensor, eps: float) -> torch.Tensor:
        patterns = patterns.to(device=self.device, dtype=self.dtype)
        direction = direction.to(device=self.device, dtype=self.dtype)
        images0 = self.render_images(patterns)
        images1 = self.render_images(patterns + eps * direction)
        return (images1 - images0) / eps

    def finite_difference_column(self, patterns: torch.Tensor, k: int, x: int, eps: float) -> torch.Tensor:
        direction = torch.zeros_like(patterns)
        direction[k, x] = 1.0
        return self.finite_difference(patterns, direction, eps)

    def finite_difference_batch(self, patterns: torch.Tensor, directions: torch.Tensor, eps: float) -> torch.Tensor:
        patterns = patterns.to(device=self.device, dtype=self.dtype)
        directions = directions.to(device=self.device, dtype=self.dtype)
        if directions.ndim != 3:
            raise ValueError("directions must be [B, K, Wp]")
        images0 = self.render_images(patterns)
        perturbed = patterns.unsqueeze(0) + eps * directions
        images1 = self.render_images_batch(perturbed)
        return (images1 - images0.unsqueeze(0)) / eps

    def finite_difference_batch_chunked(
        self,
        patterns: torch.Tensor,
        directions: torch.Tensor,
        eps: float,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        if directions.ndim != 3:
            raise ValueError("directions must be [B, K, Wp]")
        chunks = []
        for s in range(0, directions.shape[0], chunk_size):
            d = directions[s : s + chunk_size]
            chunks.append(self.finite_difference_batch(patterns, d, eps))
        return torch.cat(chunks, dim=0)

    def render_depth_for_visualization(self) -> torch.Tensor:
        if self._depth is None:
            raise RuntimeError("Scene not loaded.")
        return self._depth

    def save_visualization(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        obs = self.render_images_and_gt()
        images = obs["images"].detach().cpu().numpy()
        corr = obs["gt_corr"].detach().cpu().numpy()
        if images.shape[0] < 2:
            raise RuntimeError("self-check visualization expects at least two patterns (constant + stripe).")
        if self.patterns is None or self.patterns.shape[0] < 2:
            raise RuntimeError("self-check visualization expects at least two stored patterns.")

        patterns_np = self.patterns.detach().cpu().numpy()

        plt.figure(figsize=(8, 2.5))
        plt.plot(patterns_np[0], linewidth=1.5)
        plt.ylim(0.0, 1.0)
        plt.title("Constant Pattern")
        plt.tight_layout()
        plt.savefig(out / "constant_pattern.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 2.5))
        plt.plot(patterns_np[1], linewidth=1.5)
        plt.ylim(0.0, 1.0)
        plt.title("Stripe Pattern")
        plt.tight_layout()
        plt.savefig(out / "stripe_pattern.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.imshow(images[0], cmap="gray", vmin=0.0, vmax=1.0)
        plt.title("Constant Pattern Render")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(out / "constant_pattern_render.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.imshow(images[1], cmap="gray", vmin=0.0, vmax=1.0)
        plt.title("Stripe Pattern Render")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(out / "stripe_pattern_render.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.imshow(corr, cmap="viridis")
        plt.title("GT Correspondence")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(out / "gt_corr_vis.png", dpi=150)
        plt.close()
