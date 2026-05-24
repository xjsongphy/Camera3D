from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestStructuredLightRendererSelfCheck(unittest.TestCase):
    def test_self_check_outputs(self) -> None:
        try:
            import torch
            from lab2 import StructuredLightRenderer
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"Required deps unavailable: {exc}")

        renderer = StructuredLightRenderer(device="cpu")
        renderer.set_camera(
            {
                "width": 64,
                "height": 48,
                "fx": 70.0,
                "fy": 70.0,
                "cx": 31.5,
                "cy": 23.5,
            }
        )
        renderer.set_projector(
            {
                "width": 80,
                "height": 60,
                "fx": 75.0,
                "fy": 75.0,
                "cx": 39.5,
                "cy": 29.5,
            }
        )
        renderer.load_scene()

        wp = 80
        k_const = torch.full((1, wp), 0.5, dtype=torch.float32)
        k_stripe = torch.zeros((1, wp), dtype=torch.float32)
        k_stripe[:, ::8] = 1.0
        patterns = torch.cat([k_const, k_stripe], dim=0)
        renderer.set_patterns(patterns)

        try:
            images = renderer.render_images()
        except RuntimeError as exc:  # pragma: no cover
            if "Mitsuba is required" in str(exc):
                self.skipTest(str(exc))
            raise
        gt_corr = renderer.gt_corr
        depth = renderer.render_depth_for_visualization()

        self.assertEqual(tuple(images.shape), (2, 48, 64))
        self.assertEqual(tuple(gt_corr.shape), (48, 64))
        self.assertEqual(tuple(depth.shape), (48, 64))

        self.assertTrue(torch.isfinite(images).all().item())
        self.assertTrue((images >= 0.0).all().item())
        self.assertTrue((images <= 1.0).all().item())

        with TemporaryDirectory(prefix="lab2_self_check_") as td:
            renderer.save_visualization(output_dir=td)
            out_dir = Path(td)
            self.assertTrue((out_dir / "constant_pattern.png").exists())
            self.assertTrue((out_dir / "stripe_pattern.png").exists())
            self.assertTrue((out_dir / "constant_pattern_render.png").exists())
            self.assertTrue((out_dir / "stripe_pattern_render.png").exists())
            self.assertTrue((out_dir / "gt_corr_vis.png").exists())


if __name__ == "__main__":
    unittest.main()
