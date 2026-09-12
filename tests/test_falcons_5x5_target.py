import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "coding" / "python"
sys.path.insert(0, str(PYTHON_DIR))

import create_falcons_5x5_target as target_generator
import order_decoupling_grayscale as forward_model


class Falcons5x5TargetTests(unittest.TestCase):
    def test_identity_mapping_and_fixed_levels(self):
        source = np.zeros((10, 10), dtype=np.uint8)
        for row in range(5):
            for column in range(5):
                source[row * 2 : (row + 1) * 2, column * 2 : (column + 1) * 2] = (
                    (row * 5 + column) % 4
                ) * 85

        targets, positions = target_generator.build_targets(source, tile_size=4)

        self.assertEqual(targets.shape, (25, 4, 4))
        np.testing.assert_array_equal(
            positions, np.array([(r, c) for r in range(5) for c in range(5)])
        )
        expected_levels = target_generator.TARGET_LEVELS[
            np.arange(25) % len(target_generator.TARGET_LEVELS)
        ]
        np.testing.assert_allclose(targets[:, 0, 0], expected_levels)

    def test_requested_orders_are_exact_and_m_priority(self):
        positions = np.array([(r, c) for r in range(5) for c in range(5)])
        pair_mat = forward_model.build_order_pairs(positions, -4, 1)
        expected = np.array([(m, n) for m in range(-4, 1) for n in range(1, 6)])
        np.testing.assert_array_equal(pair_mat, expected)
        self.assertFalse(np.any(np.all(pair_mat == (0, 0), axis=1)))

    def test_three_by_three_global_mirror_mapping(self):
        source = np.zeros((6, 6), dtype=np.uint8)
        source[:2, :2] = 85
        source[:2, 4:] = 255

        targets, positions = target_generator.build_targets(
            source, tile_size=2, grid_size=3, transform="mirror_lr"
        )

        self.assertEqual(targets.shape, (9, 2, 2))
        np.testing.assert_array_equal(
            positions, np.array([(r, c) for r in range(3) for c in range(3)])
        )
        self.assertTrue(np.all(targets[0] == target_generator.TARGET_LEVELS[3]))
        self.assertTrue(np.all(targets[2] == target_generator.TARGET_LEVELS[1]))

        pair_mat = forward_model.build_order_pairs(positions, -3, 2)
        expected = np.array([(m, n) for m in range(-3, 0) for n in range(2, 5)])
        np.testing.assert_array_equal(pair_mat, expected)
        self.assertFalse(np.any(np.all(pair_mat == (0, 0), axis=1)))


if __name__ == "__main__":
    unittest.main()
