import unittest
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "coding" / "python"))
import create_three_level_fruit_target as fruit_target


class ThreeLevelFruitTargetTests(unittest.TestCase):
    def test_each_channel_keeps_shape_and_has_three_levels(self):
        binary = np.zeros((23, 12, 12), dtype=np.uint8)
        binary[:, 2, 3:9] = 1
        binary[:, 9, 3:9] = 1
        binary[:, 2:10, 3] = 1
        binary[:, 2:10, 8] = 1
        targets, counts = fruit_target.create_three_level_targets(binary)

        expected_filled = np.stack(
            [ndimage.binary_fill_holes(channel) for channel in binary]
        )
        np.testing.assert_array_equal(targets > 0, expected_filled > 0)
        np.testing.assert_array_equal(
            np.unique(targets), np.array([0, 1 / 3, 2 / 3, 1], dtype=np.float32)
        )
        self.assertTrue(np.all(counts[:, 2:5] > 0))
        np.testing.assert_array_equal(counts[:, 1], counts[:, 2:5].sum(axis=1))


if __name__ == "__main__":
    unittest.main()
