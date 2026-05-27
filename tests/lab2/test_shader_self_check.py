"""
Renderer self-check test for Lab 2.

This test validates the renderer by:
1. Rendering constant and stripe patterns
2. Verifying output shapes and value ranges
3. Generating visualizations saved to outputs/lab2/self_check/

Run with:
    pytest tests/lab2/test_shader_self_check.py -v
    python -m unittest tests.lab2.test_shader_self_check -v

Environment setup is automatic when importing lab2 modules.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

# Import lab2 to trigger automatic environment setup
# Note: Must use 'lab2.shader' not 'src.lab2.shader' to trigger __init__.py
try:
    import torch
    from lab2.shader import StructuredLightRenderer
except ImportError:
    torch = None  # type: ignore


class TestStructuredLightRendererSelfCheck(unittest.TestCase):
    """Self-check test for structured light renderer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup output directory once for all tests."""
        cls.output_dir = Path("outputs/lab2/self_check")
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def test_self_check_outputs(self) -> None:
        """Test renderer with constant and stripe patterns, save visualizations."""
        if torch is None:  # pragma: no cover
            self.skipTest("PyTorch not available")

        # Create renderer
        try:
            renderer = StructuredLightRenderer(device="cpu")
        except RuntimeError as exc:  # pragma: no cover
            if "Mitsuba is required" in str(exc):
                self.skipTest("Mitsuba not available - install with: uv sync --group lab2")
            raise

        # Setup camera
        renderer.set_camera({
            "width": 640,
            "height": 480,
            "fx": 600.0,
            "fy": 600.0,
            "cx": 320.0,
            "cy": 240.0,
            "R": torch.eye(3).tolist(),
            "t": [0.0, 0.0, 0.0],
        })

        # Setup projector
        renderer.set_projector({
            "width": 640,
            "height": 480,
            "fx": 600.0,
            "fy": 600.0,
            "cx": 320.0,
            "cy": 240.0,
            "R": torch.eye(3).tolist(),
            "t": [0.1, 0.0, 0.0],  # 10cm baseline
        })

        # Use scene with objects to show geometry
        renderer.set_scene_name("sl_marble_objects")  # Has sphere + cube
        renderer.load_scene()  # Uses internal virtual geometry

        # Create stripe pattern to better visualize scene geometry
        wp = 640
        # Stripe pattern: vertical stripes to show object shapes
        pattern = torch.zeros((1, wp), dtype=torch.float32)
        pattern[:, ::16] = 1.0  # Every 16th pixel is white
        patterns = pattern  # Single pattern

        renderer.set_patterns(patterns)

        # Render
        try:
            images = renderer.render_images()
        except RuntimeError as exc:  # pragma: no cover
            if "Mitsuba is required" in str(exc):
                self.skipTest(str(exc))
            raise

        gt_corr = renderer.gt_corr
        depth = renderer.render_depth_for_visualization()

        # Validate shapes
        self.assertEqual(tuple(images.shape), (1, 480, 640))  # Single pattern
        self.assertEqual(tuple(gt_corr.shape), (480, 640))
        self.assertEqual(tuple(depth.shape), (480, 640))

        # Validate pixel values
        self.assertTrue(torch.isfinite(images).all().item())
        self.assertTrue((images >= 0.0).all().item())
        self.assertTrue((images <= 1.0).all().item())

        # Save visualizations to persistent directory
        self._save_visualization_simple(renderer, output_dir=self.output_dir)

        # Verify files were created
        self.assertTrue((self.output_dir / "stripe_pattern.png").exists())
        self.assertTrue((self.output_dir / "pattern_render.png").exists())
        self.assertTrue((self.output_dir / "gt_corr_vis.png").exists())

    def _save_visualization_simple(self, renderer: StructuredLightRenderer, output_dir: Path) -> None:
        """Simple visualization for single pattern self-check."""
        import matplotlib.pyplot as plt

        # Get rendered images
        images = renderer.render_images()
        gt_corr = renderer.gt_corr

        # Save pattern visualization
        pattern_np = renderer.patterns.detach().cpu().numpy()
        plt.figure(figsize=(8, 2.5))
        plt.plot(pattern_np[0], linewidth=1.5, color='blue')
        plt.ylim(0.0, 1.0)
        plt.title("Stripe Pattern (Every 16th pixel white)")
        plt.xlabel("Projector Column")
        plt.ylabel("Intensity")
        plt.tight_layout()
        plt.savefig(output_dir / "stripe_pattern.png", dpi=150)
        plt.close()

        # Save rendered image (with gamma correction for visibility)
        img_np = images[0].detach().cpu().numpy()
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Original render
        im1 = ax1.imshow(img_np, cmap="gray", vmin=0.0, vmax=1.0)
        ax1.set_title("Pattern Render (Original) - Scene: sl_marble_objects")
        plt.colorbar(im1, ax=ax1)

        # Gamma corrected for better visibility
        gamma_corrected = np.power(img_np.clip(0, 1), 0.5)
        im2 = ax2.imshow(gamma_corrected, cmap="gray", vmin=0.0, vmax=1.0)
        ax2.set_title("Pattern Render (Gamma Corrected)")
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.savefig(output_dir / "pattern_render.png", dpi=150)
        plt.close()

        # Save GT correspondence
        plt.figure(figsize=(6, 4))
        plt.imshow(gt_corr.cpu().numpy(), cmap="viridis")
        plt.title("Ground Truth Correspondence")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_dir / "gt_corr_vis.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    unittest.main()
