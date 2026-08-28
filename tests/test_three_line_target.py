import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "coding" / "python"
sys.path.insert(0, str(PYTHON_DIR))

import create_three_line_target as target_generator
import order_decoupling_grayscale as forward_model


class ThreeLineTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targets = target_generator.build_targets()

    def test_shape_labels_and_equal_pixel_counts(self):
        self.assertEqual(
            self.targets.shape,
            (
                target_generator.CHANNEL_COUNT,
                target_generator.IMAGE_SIZE,
                target_generator.IMAGE_SIZE,
            ),
        )
        np.testing.assert_array_equal(
            np.unique(self.targets),
            np.array([0, 1 / 3, 2 / 3, 1], dtype=np.float32),
        )
        self.assertTrue(np.all(self.targets == self.targets[0]))

        for channel in self.targets:
            counts = [
                np.count_nonzero(np.isclose(channel, level))
                for level in (0.0, *target_generator.LEVELS)
            ]
            self.assertEqual(counts, [247000, 1000, 1000, 1000])

    def test_lines_are_horizontal_separate_and_centered(self):
        channel = self.targets[0]
        nonzero_rows, nonzero_columns = np.nonzero(channel)
        self.assertEqual(
            tuple(np.unique(nonzero_rows)),
            tuple(
                row
                for center in (190, 250, 310)
                for row in range(center - 2, center + 3)
            ),
        )
        self.assertEqual(nonzero_columns.min(), 150)
        self.assertEqual(nonzero_columns.max(), 349)

    def test_loader_accepts_four_level_mat_data(self):
        mat_file = PROJECT_ROOT / "output" / "test_three_line_loader.mat"
        try:
            sio.savemat(mat_file, {"bw_all": self.targets})
            loaded = forward_model.load_targets(mat_file)
        finally:
            mat_file.unlink(missing_ok=True)

        self.assertEqual(loaded.dtype, np.float32)
        np.testing.assert_array_equal(loaded, self.targets)

    def test_stage_d_losses_are_finite_and_differentiable(self):
        targets = torch.zeros((23, 16, 16), dtype=torch.float32)
        targets[:, 3, 4:12] = 1 / 3
        targets[:, 7, 4:12] = 2 / 3
        targets[:, 11, 4:12] = 1
        dx = torch.rand((16, 16), dtype=torch.float32, requires_grad=True)
        dy = torch.rand((16, 16), dtype=torch.float32, requires_grad=True)
        pair_mat = torch.tensor(forward_model.PAIR_MAT, dtype=torch.float32)
        weights = torch.tensor(forward_model.CUSTOM_WEIGHTS, dtype=torch.float32)

        outputs = forward_model.total_cost(
            dx,
            dy,
            targets,
            pair_mat,
            weights,
            level_weight=1.0,
            cross_level_weight=0.5,
            gap_weight=0.2,
            line_uniformity_weight=0.1,
            visibility_weight=0.2,
            worst_level_weight=0.5,
            background_uniformity_weight=0.05,
            background_row_uniformity_weight=0.8,
        )
        outputs[0].backward()

        self.assertTrue(all(torch.all(torch.isfinite(output)) for output in outputs))
        self.assertIsNotNone(dx.grad)
        self.assertIsNotNone(dy.grad)
        self.assertTrue(torch.all(torch.isfinite(dx.grad)))
        self.assertTrue(torch.all(torch.isfinite(dy.grad)))


if __name__ == "__main__":
    unittest.main()
