import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "coding" / "python"))

from compare_full_scene_results import restore_assembled_scene, rotate_assembled_scene
from create_butterfly_4x4_target import TARGET_LEVELS, build_targets


class Butterfly4x4TargetTests(unittest.TestCase):
    def test_rot180_mapping_is_global_and_row_major(self):
        source = np.zeros((8, 8), dtype=np.uint8)
        source[:2, :2] = 85
        source[6:, 6:] = 255

        targets, positions, source_positions = build_targets(source, tile_size=2)

        self.assertEqual(targets.shape, (16, 2, 2))
        np.testing.assert_array_equal(
            positions, np.asarray([divmod(index, 4) for index in range(16)])
        )
        np.testing.assert_array_equal(source_positions[0], [4, 4])
        np.testing.assert_array_equal(source_positions[-1], [1, 1])
        self.assertTrue(np.all(targets[0] == TARGET_LEVELS[3]))
        self.assertTrue(np.all(targets[-1] == TARGET_LEVELS[1]))

    def test_final_global_rotation_restores_orientation(self):
        source = np.arange(16).reshape(4, 4)
        np.testing.assert_array_equal(
            rotate_assembled_scene(np.rot90(source, 2), 180), source
        )

    def test_final_global_mirror_restores_orientation(self):
        source = np.arange(16).reshape(4, 4)
        np.testing.assert_array_equal(
            restore_assembled_scene(np.fliplr(source), mirror_lr=True), source
        )


if __name__ == "__main__":
    unittest.main()
