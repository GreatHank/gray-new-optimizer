import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coding" / "python"))
import evaluate_source_image_channels as evaluator


class SourceImageChannelEvaluationTests(unittest.TestCase):
    def test_channel_target_mean_includes_black_background(self):
        tile = np.asarray([[0, 85], [170, 255]], dtype=np.uint8)
        metrics = evaluator.channel_metrics(tile)
        self.assertAlmostEqual(metrics["active_fraction"], 0.75)
        self.assertAlmostEqual(metrics["foreground_gray_mean"], 2 / 3)
        self.assertAlmostEqual(metrics["channel_target_mean"], 0.5)
        self.assertEqual(metrics["present_gray_levels"], 3)
        self.assertAlmostEqual(metrics["gray_entropy"], 1.0)

    def test_balanced_channels_have_zero_brightness_cv(self):
        tile = np.asarray([[0, 85], [170, 255]], dtype=np.uint8)
        image = np.tile(tile, (2, 2))
        rows, summary = evaluator.evaluate_image(image, grid_size=2)
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(summary["channel_target_mean_cv"], 0.0)
        self.assertEqual(summary["all_three_gray_levels_count"], 4)
        self.assertEqual(summary["black_channel_count"], 0)

    def test_quantization_uses_fixed_four_levels(self):
        source = np.asarray([0, 40, 90, 150, 210, 250], dtype=np.uint8)
        actual = evaluator.quantize_four_levels(source)
        np.testing.assert_array_equal(actual, [0, 0, 85, 170, 170, 255])


if __name__ == "__main__":
    unittest.main()
