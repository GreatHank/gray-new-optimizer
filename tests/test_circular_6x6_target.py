import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "coding" / "python"))

from create_circular_6x6_target import TARGET_LEVELS, build_targets


class Circular6x6TargetTests(unittest.TestCase):
    def test_whole_image_is_split_row_major_without_regrading(self):
        source = np.zeros((12, 12), dtype=np.uint8)
        source[:2, :2] = 85
        source[2:4, :2] = 170
        source[4:6, :2] = 255

        targets, positions, bounds = build_targets(source, grid_size=6, tile_size=2)

        self.assertEqual(targets.shape, (36, 2, 2))
        np.testing.assert_array_equal(
            positions, np.asarray([divmod(i, 6) for i in range(36)])
        )
        self.assertEqual(bounds[0], (0, 2, 0, 2))
        self.assertEqual(bounds[-1], (10, 12, 10, 12))
        self.assertTrue(np.all(targets[0] == TARGET_LEVELS[1]))
        self.assertTrue(np.all(targets[6] == TARGET_LEVELS[2]))
        self.assertTrue(np.all(targets[12] == TARGET_LEVELS[3]))

    def test_unknown_gray_level_is_rejected(self):
        source = np.zeros((6, 6), dtype=np.uint8)
        source[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "0、85、170、255"):
            build_targets(source, grid_size=6, tile_size=1)


if __name__ == "__main__":
    unittest.main()
