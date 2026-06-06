from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .shader import CameraConfig, LightConfig, ProjectorConfig


@dataclass
class MaterialModel:
    albedo: torch.Tensor
    specular_strength: float = 0.0
    shininess: float = 32.0


class PytorchStructuredLightRenderer:
    """
    Lightweight differentiable structured-light renderer implemented in PyTorch.

    This renderer analytically intersects camera rays with the small set of
    project scenes used in Lab 2 and computes an approximate shading model.
    It is designed as a practical fallback when Mitsuba autodiff is too memory
    expensive for end-to-end training.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        spp: int = 1,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.spp = int(spp)

        self.camera: CameraConfig | None = None
        self.projector: ProjectorConfig | None = None
        self.lights = LightConfig()
        self.patterns: torch.Tensor | None = None
        self._scene_name: str = "sl_diffuse_objects"
        self._scene_loaded = False

        self._depth: torch.Tensor | None = None
        self._gt_corr: torch.Tensor | None = None
        self._world_points: torch.Tensor | None = None
        self._normals: torch.Tensor | None = None
        self._albedo: torch.Tensor | None = None
        self._specular_strength: torch.Tensor | None = None
        self._shininess: torch.Tensor | None = None
        self._object_id: torch.Tensor | None = None
        self._valid_mask: torch.Tensor | None = None

    def _to_cpu_tensor(self, x: Any, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=self.dtype, device="cpu")
        if shape is not None and tuple(t.shape) != shape:
            raise ValueError(f"Expected shape {shape}, got {tuple(t.shape)}")
        return t

    def _parse_intrinsics_extrinsics(self, config: dict[str, Any], is_camera: bool) -> CameraConfig | ProjectorConfig:
        w = int(config["width"])
        h = int(config["height"])

        if "K" in config:
            K = self._to_cpu_tensor(config["K"], (3, 3))
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
        else:
            fx = float(config["fx"])
            fy = float(config["fy"])
            cx = float(config.get("cx", (w - 1) * 0.5))
            cy = float(config.get("cy", (h - 1) * 0.5))

        R = self._to_cpu_tensor(config.get("R", torch.eye(3)), (3, 3))
        t = self._to_cpu_tensor(config.get("t", torch.zeros(3)), (3,))
        if is_camera:
            return CameraConfig(w, h, fx, fy, cx, cy, R, t)
        return ProjectorConfig(w, h, fx, fy, cx, cy, R, t)

    def set_camera(self, camera_config: dict[str, Any]) -> None:
        self.camera = self._parse_intrinsics_extrinsics(camera_config, True)  # type: ignore[assignment]
        self._invalidate_scene_cache()

    def set_projector(self, projector_config: dict[str, Any]) -> None:
        self.projector = self._parse_intrinsics_extrinsics(projector_config, False)  # type: ignore[assignment]
        self._invalidate_scene_cache()

    def set_lights(self, light_config: dict[str, Any]) -> None:
        self.lights = LightConfig(ambient=float(light_config.get("ambient", 0.12)))

    def set_scene_name(self, scene_name: str) -> None:
        self._scene_name = scene_name
        self._invalidate_scene_cache()

    def set_patterns(self, patterns: torch.Tensor) -> None:
        if patterns.ndim != 2:
            raise ValueError("patterns must be [K, Wp]")
        self.patterns = patterns.to(device=self.device, dtype=self.dtype)

    def update_patterns(self, patterns: torch.Tensor) -> None:
        self.set_patterns(patterns)

    def _invalidate_scene_cache(self) -> None:
        self._depth = None
        self._gt_corr = None
        self._world_points = None
        self._normals = None
        self._albedo = None
        self._specular_strength = None
        self._shininess = None
        self._object_id = None
        self._valid_mask = None

    def load_scene(self, scene_path: str | None = None, cache_path: str | None = None) -> None:
        if scene_path is not None or cache_path is not None:
            raise ValueError("PytorchStructuredLightRenderer does not support external scene/cache files.")
        if self.camera is None or self.projector is None:
            raise RuntimeError("Camera and projector must be configured before load_scene().")
        self._build_scene_buffers()
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
        # Image rows increase downward, while the camera's +Y axis points upward.
        y = -(v - cam.cy) / cam.fy
        dirs_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)

        R_cw = cam.R.to(device=self.device, dtype=self.dtype)
        t_cw = cam.t.to(device=self.device, dtype=self.dtype)
        origin = -(R_cw.T @ t_cw)
        dirs_world = dirs_cam @ R_cw
        dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)
        rays_o = origin.view(1, 1, 3).expand(cam.height, cam.width, 3)
        return rays_o, dirs_world

    def _camera_origin(self) -> torch.Tensor:
        if self.camera is None:
            raise RuntimeError("Camera is not configured.")
        R_cw = self.camera.R.to(device=self.device, dtype=self.dtype)
        t_cw = self.camera.t.to(device=self.device, dtype=self.dtype)
        return -(R_cw.T @ t_cw)

    def _projector_origin(self) -> torch.Tensor:
        if self.projector is None:
            raise RuntimeError("Projector is not configured.")
        R_pw = self.projector.R.to(device=self.device, dtype=self.dtype)
        t_pw = self.projector.t.to(device=self.device, dtype=self.dtype)
        return -(R_pw.T @ t_pw)

    def _ground_intersection(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        plane_y = torch.tensor(-0.45, device=self.device, dtype=self.dtype)
        denom = rays_d[..., 1]
        valid = denom.abs() > 1e-6
        t = (plane_y - rays_o[..., 1]) / torch.where(valid, denom, torch.ones_like(denom))
        pts = rays_o + t[..., None] * rays_d
        inside = (
            (pts[..., 0] >= -1.2) & (pts[..., 0] <= 1.2) &
            (pts[..., 2] >= 0.8) & (pts[..., 2] <= 2.4)
        )
        valid = valid & (t > 0) & inside
        normal = torch.zeros_like(pts)
        normal[..., 1] = 1.0
        return t, pts, normal, valid

    def _sphere_intersection(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        center = torch.tensor([-0.35, -0.15, 1.9], device=self.device, dtype=self.dtype)
        radius = torch.tensor(0.24, device=self.device, dtype=self.dtype)
        oc = rays_o - center
        a = (rays_d * rays_d).sum(dim=-1)
        b = 2.0 * (oc * rays_d).sum(dim=-1)
        c = (oc * oc).sum(dim=-1) - radius * radius
        disc = b * b - 4.0 * a * c
        valid = disc >= 0
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        t0 = (-b - sqrt_disc) / (2.0 * a)
        t1 = (-b + sqrt_disc) / (2.0 * a)
        inf = torch.full_like(t0, float("inf"))
        t0 = torch.where(t0 > 0, t0, inf)
        t1 = torch.where(t1 > 0, t1, inf)
        t = torch.minimum(t0, t1)
        valid = valid & torch.isfinite(t)
        pts = rays_o + t[..., None] * rays_d
        normal = (pts - center) / radius
        return t, pts, normal, valid

    def _cube_intersection(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        center = torch.tensor([0.35, -0.2, 2.05], device=self.device, dtype=self.dtype)
        half_extent = torch.tensor([0.24, 0.24, 0.24], device=self.device, dtype=self.dtype)
        bounds_min = center - half_extent
        bounds_max = center + half_extent

        inv_d = torch.where(rays_d.abs() > 1e-6, 1.0 / rays_d, torch.full_like(rays_d, 1e6))
        t0 = (bounds_min - rays_o) * inv_d
        t1 = (bounds_max - rays_o) * inv_d
        tmin = torch.minimum(t0, t1)
        tmax = torch.maximum(t0, t1)
        t_enter = tmin.max(dim=-1).values
        t_exit = tmax.min(dim=-1).values
        valid = (t_exit >= t_enter) & (t_exit > 0)
        t = torch.where(t_enter > 0, t_enter, t_exit)
        valid = valid & torch.isfinite(t)
        pts = rays_o + t[..., None] * rays_d

        normal = torch.zeros_like(pts)
        rel = pts - center
        abs_rel = rel.abs()
        axis = abs_rel.argmax(dim=-1)
        for i in range(3):
            mask = axis == i
            normal[..., i] = torch.where(mask, torch.sign(rel[..., i]), normal[..., i])
        return t, pts, normal, valid

    def _material_library(self) -> dict[str, MaterialModel]:
        if self._scene_name == "sl_diffuse_objects":
            return {
                "ground": MaterialModel(torch.tensor([0.68, 0.66, 0.62], device=self.device, dtype=self.dtype)),
                "sphere": MaterialModel(torch.tensor([0.78, 0.84, 0.92], device=self.device, dtype=self.dtype)),
                "cube": MaterialModel(torch.tensor([0.70, 0.77, 0.60], device=self.device, dtype=self.dtype)),
            }
        if self._scene_name == "sl_marble_objects":
            return {
                "ground": MaterialModel(torch.tensor([0.74, 0.71, 0.67], device=self.device, dtype=self.dtype)),
                "sphere": MaterialModel(
                    torch.tensor([0.90, 0.84, 0.78], device=self.device, dtype=self.dtype),
                    specular_strength=0.12,
                    shininess=48.0,
                ),
                "cube": MaterialModel(
                    torch.tensor([0.80, 0.84, 0.90], device=self.device, dtype=self.dtype),
                    specular_strength=0.10,
                    shininess=36.0,
                ),
            }
        raise ValueError(f"Unsupported scene for PytorchStructuredLightRenderer: {self._scene_name}")

    def _build_scene_buffers(self) -> None:
        rays_o, rays_d = self._pixel_rays_world()
        cam_origin = self._camera_origin()
        materials = self._material_library()

        candidates: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = [
            ("ground", *self._ground_intersection(rays_o, rays_d)),
            ("sphere", *self._sphere_intersection(rays_o, rays_d)),
            ("cube", *self._cube_intersection(rays_o, rays_d)),
        ]

        H, W = rays_o.shape[:2]
        inf = torch.full((H, W), float("inf"), device=self.device, dtype=self.dtype)
        best_t = inf.clone()
        best_pts = torch.zeros((H, W, 3), device=self.device, dtype=self.dtype)
        best_normals = torch.zeros((H, W, 3), device=self.device, dtype=self.dtype)
        best_albedo = torch.zeros((H, W, 3), device=self.device, dtype=self.dtype)
        best_specular = torch.zeros((H, W), device=self.device, dtype=self.dtype)
        best_shininess = torch.zeros((H, W), device=self.device, dtype=self.dtype)
        best_object_id = torch.full((H, W), -1, device=self.device, dtype=torch.long)
        best_valid = torch.zeros((H, W), device=self.device, dtype=torch.bool)

        for object_id, (name, t, pts, normals, valid) in enumerate(candidates):
            take = valid & (t < best_t)
            mat = materials[name]
            best_t = torch.where(take, t, best_t)
            best_pts = torch.where(take[..., None], pts, best_pts)
            best_normals = torch.where(take[..., None], normals, best_normals)
            best_albedo = torch.where(take[..., None], mat.albedo.view(1, 1, 3), best_albedo)
            best_specular = torch.where(take, torch.full_like(best_specular, mat.specular_strength), best_specular)
            best_shininess = torch.where(take, torch.full_like(best_shininess, mat.shininess), best_shininess)
            best_object_id = torch.where(take, torch.full_like(best_object_id, object_id), best_object_id)
            best_valid = best_valid | take

        depth = torch.linalg.norm(best_pts - cam_origin.view(1, 1, 3), dim=-1)
        depth = torch.where(best_valid, depth, torch.full_like(depth, float("nan")))

        self._depth = depth
        self._world_points = best_pts
        self._normals = torch.nn.functional.normalize(best_normals, dim=-1)
        self._albedo = best_albedo
        self._specular_strength = best_specular
        self._shininess = best_shininess
        self._object_id = best_object_id
        self._valid_mask = best_valid
        self._gt_corr = None

    def compute_gt_corr(self) -> torch.Tensor:
        if self.projector is None:
            raise RuntimeError("Projector is not configured.")
        if self._world_points is None or self._valid_mask is None:
            raise RuntimeError("Scene not loaded.")
        proj = self.projector
        R = proj.R.to(device=self.device, dtype=self.dtype)
        t = proj.t.to(device=self.device, dtype=self.dtype)
        pts_proj = self._world_points @ R.T + t.view(1, 1, 3)
        z = pts_proj[..., 2]
        x = proj.fx * (pts_proj[..., 0] / torch.clamp(z, min=1e-6)) + proj.cx
        valid = self._valid_mask & (z > 1e-6) & (x >= 0) & (x <= (proj.width - 1))
        gt = x.clamp(0, proj.width - 1)
        self._gt_corr = torch.where(valid, gt, torch.full_like(gt, float("nan")))
        return self._gt_corr

    @property
    def gt_corr(self) -> torch.Tensor:
        if self._gt_corr is None:
            return self.compute_gt_corr()
        return self._gt_corr

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

    def _sample_pattern_columns(self, patterns: torch.Tensor) -> torch.Tensor:
        gt = self.gt_corr
        valid = torch.isfinite(gt)
        x = torch.where(valid, gt, torch.zeros_like(gt))
        _, wp = patterns.shape
        x0 = torch.floor(x).long().clamp(0, wp - 1)
        x1 = (x0 + 1).clamp(0, wp - 1)
        w1 = (x - x0.to(dtype=self.dtype)).clamp(0.0, 1.0)
        p0 = patterns[:, x0]
        p1 = patterns[:, x1]
        sampled = (1.0 - w1).unsqueeze(0) * p0 + w1.unsqueeze(0) * p1
        return torch.where(valid.unsqueeze(0), sampled, torch.zeros_like(sampled))

    def render_images_autodiff(self, patterns: torch.Tensor | None = None) -> torch.Tensor:
        patterns = self._prepare_patterns(patterns)
        if any(v is None for v in (self._world_points, self._normals, self._albedo, self._specular_strength, self._shininess, self._valid_mask)):
            raise RuntimeError("Scene buffers are not initialized.")

        sampled = self._sample_pattern_columns(patterns)  # [K, H, W]
        pts = self._world_points
        normals = self._normals
        albedo = self._albedo
        specular_strength = self._specular_strength
        shininess = self._shininess
        object_id = self._object_id
        valid = self._valid_mask

        proj_origin = self._projector_origin()
        cam_origin = self._camera_origin()
        fill_pos = torch.tensor([0.0, 0.45, 1.05], device=self.device, dtype=self.dtype)

        l_proj = torch.nn.functional.normalize(proj_origin.view(1, 1, 3) - pts, dim=-1)
        l_fill = torch.nn.functional.normalize(fill_pos.view(1, 1, 3) - pts, dim=-1)
        v_dir = torch.nn.functional.normalize(cam_origin.view(1, 1, 3) - pts, dim=-1)
        h_proj = torch.nn.functional.normalize(l_proj + v_dir, dim=-1)

        proj_lambert = torch.clamp((normals * l_proj).sum(dim=-1), min=0.0)
        fill_lambert = torch.clamp((normals * l_fill).sum(dim=-1), min=0.0)
        proj_spec = torch.clamp((normals * h_proj).sum(dim=-1), min=0.0) ** torch.clamp(shininess, min=1.0)
        proj_spec = proj_spec * specular_strength

        base_projector = (proj_lambert + proj_spec).clamp(0.0, 1.5)
        fill_term = 0.55 * fill_lambert

        if object_id is not None:
            marble_mask = object_id == 1
            cube_mask = object_id == 2
            marble_vein = 0.08 * torch.sin(18.0 * pts[..., 0] + 11.0 * pts[..., 2])
            cube_vein = 0.06 * torch.cos(14.0 * pts[..., 1] - 9.0 * pts[..., 2])
            one = torch.ones_like(marble_vein)
            marble_tint = torch.stack([one + marble_vein, one, one - marble_vein], dim=-1)
            cube_tint = torch.stack([one - cube_vein, one, one + cube_vein], dim=-1)
            albedo = torch.where(marble_mask[..., None], (albedo * marble_tint).clamp(0.0, 1.0), albedo)
            albedo = torch.where(cube_mask[..., None], (albedo * cube_tint).clamp(0.0, 1.0), albedo)

        rgb = []
        for k in range(patterns.shape[0]):
            proj_rgb = sampled[k][..., None] * base_projector[..., None]
            color = self.lights.ambient + albedo * (proj_rgb + fill_term[..., None])
            color = torch.where(valid[..., None], color, torch.full_like(color, self.lights.ambient))
            rgb.append(color.clamp(0.0, 1.0))
        return torch.stack(rgb, dim=0)

    def render_images(self, patterns: torch.Tensor | None = None) -> torch.Tensor:
        with torch.no_grad():
            return self.render_images_autodiff(patterns)

    def render_images_and_gt(self, patterns: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return {"images": self.render_images(patterns), "gt_corr": self.gt_corr}

    def render_images_batch(self, patterns_batch: torch.Tensor) -> torch.Tensor:
        if patterns_batch.ndim != 3:
            raise ValueError("patterns_batch must be [B, K, Wp]")
        return torch.stack([self.render_images(patterns_batch[b]) for b in range(patterns_batch.shape[0])], dim=0)

    def render_train_batch(self, mode: str = "mitsuba") -> dict[str, torch.Tensor]:
        if mode not in {"mitsuba", "autodiff"}:
            raise ValueError("mode must be 'mitsuba' or 'autodiff'")
        return {"images": self.render_images_autodiff(), "gt_corr": self.gt_corr}

    def finite_difference(self, patterns: torch.Tensor, direction: torch.Tensor, eps: float) -> torch.Tensor:
        images0 = self.render_images(patterns)
        images1 = self.render_images(patterns + eps * direction)
        return (images1 - images0) / eps

    def finite_difference_column(self, patterns: torch.Tensor, k: int, x: int, eps: float) -> torch.Tensor:
        direction = torch.zeros_like(patterns)
        direction[k, x] = 1.0
        return self.finite_difference(patterns, direction, eps)

    def finite_difference_batch(self, patterns: torch.Tensor, directions: torch.Tensor, eps: float) -> torch.Tensor:
        perturbed = patterns.unsqueeze(0) + eps * directions
        images0 = self.render_images(patterns)
        images1 = self.render_images_batch(perturbed)
        return (images1 - images0.unsqueeze(0)) / eps

    def finite_difference_batch_chunked(self, patterns: torch.Tensor, directions: torch.Tensor, eps: float, chunk_size: int = 64) -> torch.Tensor:
        chunks = []
        for s in range(0, directions.shape[0], chunk_size):
            chunks.append(self.finite_difference_batch(patterns, directions[s : s + chunk_size], eps))
        return torch.cat(chunks, dim=0)

    def render_depth_for_visualization(self) -> torch.Tensor:
        if self._depth is None:
            if not self._scene_loaded:
                raise RuntimeError("Scene not loaded.")
            self._build_scene_buffers()
        return self._depth
