import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "coding" / "python"))

from simulate_global_illumination import illumination_metrics


class GlobalIlluminationTests(unittest.TestCase):
    def test_one_global_multiplier_scales_brightness_not_relative_metrics(self):
        targets = np.zeros((2, 6, 6), dtype=np.float64)
        targets[:, 1, 1:5] = 1 / 3
        targets[:, 3, 1:5] = 2 / 3
        targets[:, 5, 1:5] = 1
        raw = 2 + targets * np.asarray([9.0, 12.0])[:, None, None]

        baseline = illumination_metrics(raw, targets, 1.0)
        quadrupled = illumination_metrics(raw, targets, 4.0)

        np.testing.assert_allclose(
            quadrupled["level_means"], baseline["level_means"] * 4
        )
        np.testing.assert_allclose(quadrupled["plane_mean"], baseline["plane_mean"] * 4)
        np.testing.assert_allclose(quadrupled["level_cvs"], baseline["level_cvs"])
        np.testing.assert_allclose(
            quadrupled["level_ratios"], baseline["level_ratios"]
        )


if __name__ == "__main__":
    unittest.main()
