"""
Renderer self-check test for Lab 2.

Validates the Mitsuba projector rendering pipeline:
1. Renders a stripe pattern projected onto a scene with objects
2. Verifies the projector is actually contributing (not just ambient)
3. Saves visualizations showing scene geometry via projected pattern

Run with:
    pytest tests/lab2/test_shader_self_check.py -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    from lab2.shader import StructuredLightRenderer
except ImportError:
    torch = None  # type: ignore


class TestStructuredLightRendererSelfCheck(unittest.TestCase):
    """Self-check test for structured light renderer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = Path("outputs/lab2/self_check")
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def test_self_check_outputs(self) -> None:
        if torch is None:  # pragma: no cover
            self.skipTest("PyTorch not available")

        try:
            renderer = StructuredLightRenderer(device="cpu")
        except RuntimeError as exc:  # pragma: no cover
            if "Mitsuba is required" in str(exc):
                self.skipTest("Mitsuba not available")
            raise

        renderer.set_camera({
            "width": 640, "height": 480,
            "fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
            "R": torch.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
        })
        renderer.set_projector({
            "width": 640, "height": 480,
            "fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
            "R": torch.eye(3).tolist(), "t": [0.1, 0.0, 0.0],
        })

        renderer.set_scene_name("sl_marble_objects")
        renderer.load_scene()

        # Sine stripe pattern: clear spatial structure to reveal object shapes
        wp = 640
        x = torch.linspace(0.0, 1.0, wp)
        pattern = (0.5 + 0.5 * torch.sin(2 * np.pi * 8 * x)).unsqueeze(0)
        renderer.set_patterns(pattern)

        try:
            images = renderer.render_images()
        except RuntimeError as exc:  # pragma: no cover
            if "Mitsuba is required" in str(exc):
                self.skipTest(str(exc))
            raise

        gt_corr = renderer.gt_corr
        depth = renderer.render_depth_for_visualization()

        # Validate shapes
        self.assertEqual(tuple(images.shape), (1, 480, 640))
        self.assertEqual(tuple(gt_corr.shape), (480, 640))
        self.assertEqual(tuple(depth.shape), (480, 640))
        self.assertTrue(torch.isfinite(images).all().item())
        self.assertTrue((images >= 0.0).all().item())
        self.assertTrue((images <= 1.0).all().item())

        # Verify projector is contributing (not just ambient)
        img = images[0]
        ambient_only = renderer.lights.ambient
        projector_contribution = (img - ambient_only).clamp(min=0.0)
        self.assertGreater(projector_contribution.mean().item(), 0.0,
                           "Projector should contribute illumination beyond ambient")

        self._save_visualizations(renderer, self.output_dir)

        for fname in ("pattern.png", "render.png", "depth_map.png", "gt_corr.png"):
            self.assertTrue((self.output_dir / fname).exists())

    def _save_visualizations(self, renderer: StructuredLightRenderer, d: Path) -> None:
        import matplotlib.pyplot as plt

        images = renderer.render_images()
        gt_corr = renderer.gt_corr
        depth = renderer.render_depth_for_visualization()
        img_np = images[0].detach().cpu().numpy()
        depth_np = depth.cpu().numpy()

        # Pattern
        pat_np = renderer.patterns[0].detach().cpu().numpy()
        plt.figure(figsize=(8, 2.5))
        plt.plot(pat_np, linewidth=1.5)
        plt.ylim(0, 1)
        plt.title("Stripe Pattern")
        plt.tight_layout()
        plt.savefig(d / "pattern.png", dpi=150)
        plt.close()

        # Render — auto-scale so object brightness variation is visible
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        im1 = ax1.imshow(img_np, cmap="gray")
        ax1.set_title(f"Projected Pattern (range {img_np.min():.3f}-{img_np.max():.3f})")
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
        ax2.set_title("Projected Pattern (stretched)")
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.savefig(d / "render.png", dpi=150)
        plt.close()

        # Depth
        _, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
        im3 = ax3.imshow(depth_np, cmap="plasma")
        ax3.set_title(f"Depth Map ({depth_np.min():.2f}-{depth_np.max():.2f}m)")
        plt.colorbar(im3, ax=ax3)

        ax4.hist(depth_np.flatten(), bins=50, edgecolor="black")
        ax4.set_title("Depth Distribution")
        ax4.set_xlabel("Depth (m)")
        plt.tight_layout()
        plt.savefig(d / "depth_map.png", dpi=150)
        plt.close()

        # GT correspondence
        plt.figure(figsize=(6, 4))
        plt.imshow(gt_corr.cpu().numpy(), cmap="viridis")
        plt.title("Ground Truth Correspondence")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(d / "gt_corr.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    unittest.main()
