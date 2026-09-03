import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "coding" / "python"
sys.path.insert(0, str(PYTHON_DIR))

import create_thin_letter_target as target_generator


class ThinLetterTargetTests(unittest.TestCase):
    def test_quantize_tile_centers_thin_lines_and_keeps_three_levels(self):
        tile = np.full((40, 60), 28, dtype=np.uint8)
        tile[10, 5:55] = 110
        tile[20, 5:55] = 170
        tile[30, 5:55] = 240

        target = target_generator.quantize_tile(tile, image_size=100)

        self.assertEqual(target.shape, (100, 100))
        np.testing.assert_allclose(
            np.unique(target), np.array([0.0, 1 / 3, 2 / 3, 1.0]), atol=1e-7
        )
        # Resizing the whole 40 x 60 crop to 100 x 100 scales its content too;
        # the old implementation incorrectly kept only 150 pixels in a centered
        # sub-image and created an artificial guard zone.
        self.assertGreater(np.count_nonzero(target), 150)

    def test_grid_produces_requested_channel_count(self):
        gray = np.full((360, 360), 28, dtype=np.uint8)
        for row in range(6):
            for column in range(6):
                row0 = row * 60
                column0 = column * 60
                gray[row0 + 10 : row0 + 50 : 10, column0 + 5 : column0 + 55] = np.array(
                    [70, 130, 190, 250], dtype=np.uint8
                )[:, None]

        targets, tiles, selected = target_generator.build_targets(gray, channel_count=36)

        self.assertEqual(targets.shape, (36, 500, 500))
        self.assertEqual(len(tiles), 36)
        self.assertEqual(len(selected), 36)

    def test_spatial_assignment_keeps_three_contiguous_gray_groups(self):
        tile = np.full((60, 60), 1, dtype=np.uint8)
        tile[30, 3:57] = 150

        target = target_generator.quantize_tile(
            tile,
            image_size=60,
            min_line_intensity=30,
            level_assignment="spatial",
        )

        line = target[30, 3:57]
        changes = np.flatnonzero(np.diff(line) != 0)
        np.testing.assert_allclose(np.unique(line), target_generator.LEVELS)
        self.assertEqual(len(changes), 2)


if __name__ == "__main__":
    unittest.main()
