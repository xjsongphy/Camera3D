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

# Import lab2 to trigger automatic environment setup
try:
    import torch
    from src.lab2.shader import StructuredLightRenderer
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

        # Load default scene
        renderer.load_scene()

        # Create test patterns
        wp = 640
        k_const = torch.full((1, wp), 0.5, dtype=torch.float32)
        k_stripe = torch.zeros((1, wp), dtype=torch.float32)
        k_stripe[:, ::8] = 1.0
        patterns = torch.cat([k_const, k_stripe], dim=0)

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
        self.assertEqual(tuple(images.shape), (2, 480, 640))
        self.assertEqual(tuple(gt_corr.shape), (480, 640))
        self.assertEqual(tuple(depth.shape), (480, 640))

        # Validate pixel values
        self.assertTrue(torch.isfinite(images).all().item())
        self.assertTrue((images >= 0.0).all().item())
        self.assertTrue((images <= 1.0).all().item())

        # Save visualizations to persistent directory
        renderer.save_visualization(output_dir=self.output_dir)

        # Verify files were created
        self.assertTrue((self.output_dir / "constant_pattern.png").exists())
        self.assertTrue((self.output_dir / "stripe_pattern.png").exists())
        self.assertTrue((self.output_dir / "constant_pattern_render.png").exists())
        self.assertTrue((self.output_dir / "stripe_pattern_render.png").exists())
        self.assertTrue((self.output_dir / "gt_corr_vis.png").exists())


if __name__ == "__main__":
    unittest.main()
