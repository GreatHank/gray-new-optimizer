import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "coding" / "python"))

import order_decoupling_grayscale as forward_model


class PhysicalForwardRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20260828)
        cls.dx = rng.uniform(-np.pi, np.pi, size=(8, 8))
        cls.dy = rng.uniform(-np.pi, np.pi, size=(8, 8))

    def _reference_intensity(self):
        intensities = []
        for m, n in forward_model.PAIR_MAT:
            phase = m * self.dx + n * self.dy
            field = np.exp(1j * phase)
            intensities.append(np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2)
        return np.stack(intensities)

    def test_reconstruct_matches_independent_reference(self):
        dx = torch.tensor(self.dx, dtype=torch.float64)
        dy = torch.tensor(self.dy, dtype=torch.float64)
        pair_mat = torch.tensor(forward_model.PAIR_MAT, dtype=torch.float64)

        raw, _, _ = forward_model.reconstruct(dx, dy, pair_mat)

        np.testing.assert_allclose(
            raw,
            self._reference_intensity(),
            rtol=2e-12,
            atol=2e-10,
        )

    def test_parseval_energy_relation_holds_for_all_channels(self):
        intensities = self._reference_intensity()
        expected_field_energy = self.dx.size
        expected_intensity_energy = self.dx.size * expected_field_energy

        np.testing.assert_allclose(
            np.sum(intensities, axis=(1, 2)),
            expected_intensity_energy,
            rtol=2e-12,
            atol=2e-10,
        )

    def test_all_36_channels_use_the_same_two_phase_arrays(self):
        self.assertEqual(forward_model.PAIR_MAT.shape, (36, 2))
        self.assertEqual(len(forward_model.CUSTOM_WEIGHTS), 36)
        self.assertEqual(len(np.unique(forward_model.PAIR_MAT, axis=0)), 36)
        self.assertTrue(np.all((forward_model.PAIR_MAT >= 1) & (forward_model.PAIR_MAT <= 6)))
        np.testing.assert_array_equal(forward_model.CUSTOM_WEIGHTS, np.ones(36))

        raw = self._reference_intensity()
        self.assertEqual(raw.shape, (36, 8, 8))


if __name__ == "__main__":
    unittest.main()
