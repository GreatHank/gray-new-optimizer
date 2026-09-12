import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "coding" / "python"))

import validate_diffraction_orders as validation


class DiffractionOrderValidationTests(unittest.TestCase):
    def test_normal_incidence_matches_two_dimensional_grating_equation(self):
        orders = np.array([[0, 0], [1, -1]], dtype=np.int16)
        result = validation.order_directions(
            orders,
            wavelength_nm=500,
            period_x_nm=1000,
            period_y_nm=2000,
            theta_deg=0,
            azimuth_deg=0,
            n_in=1,
            n_out=1,
            numerical_aperture=1,
        )

        np.testing.assert_allclose(result["ux"], [0, 0.5])
        np.testing.assert_allclose(result["uy"], [0, -0.25])
        self.assertTrue(np.all(result["propagating"]))

    def test_order_outside_air_cone_is_evanescent(self):
        result = validation.order_directions(
            np.array([[2, 0]], dtype=np.int16),
            wavelength_nm=600,
            period_x_nm=1000,
            period_y_nm=1000,
            theta_deg=0,
            azimuth_deg=0,
            n_in=1,
            n_out=1,
            numerical_aperture=1,
        )

        self.assertFalse(result["propagating"][0])
        self.assertTrue(np.isnan(result["theta_out_deg"][0]))

    def test_na_can_reject_a_propagating_order(self):
        result = validation.order_directions(
            np.array([[1, 0]], dtype=np.int16),
            wavelength_nm=500,
            period_x_nm=2000,
            period_y_nm=2000,
            theta_deg=0,
            azimuth_deg=0,
            n_in=1,
            n_out=1,
            numerical_aperture=0.2,
        )

        self.assertTrue(result["propagating"][0])
        self.assertFalse(result["captured"][0])

    def test_broadcast_parameter_accepts_one_or_per_wavelength_values(self):
        np.testing.assert_array_equal(
            validation.broadcast_parameter([3], 3, "angle"), [3, 3, 3]
        )
        np.testing.assert_array_equal(
            validation.broadcast_parameter([1, 2, 3], 3, "angle"), [1, 2, 3]
        )
        with self.assertRaises(ValueError):
            validation.broadcast_parameter([1, 2], 3, "angle")

    def test_center_incidence_reproduces_known_grating_angles(self):
        result = validation.center_incidence_for_orders(
            np.array([[-1, 0], [-2, 0], [-3, 0]], dtype=np.int16),
            wavelength_nm=480,
            period_x_nm=1200,
            period_y_nm=1200,
            n_in=1,
        )

        np.testing.assert_allclose(
            result["incident_theta_deg"][:2],
            [np.degrees(np.arcsin(0.4)), np.degrees(np.arcsin(0.8))],
        )
        np.testing.assert_array_equal(result["center_feasible"], [True, True, False])

    def test_phase_relations_identify_zero_conjugates_and_harmonics(self):
        rows = validation.phase_relation_rows(
            np.array([[0, 0], [1, 1], [-1, -1], [2, 2]], dtype=np.int16)
        )
        by_order = {(row["m"], row["n"]): row for row in rows}

        self.assertFalse(by_order[(0, 0)]["phase_controllable"])
        self.assertFalse(by_order[(0, 0)]["conjugate_order_in_selection"])
        self.assertTrue(by_order[(1, 1)]["conjugate_order_in_selection"])
        self.assertTrue(by_order[(-1, -1)]["conjugate_order_in_selection"])
        self.assertTrue(by_order[(1, 1)]["independent_conjugate_representative"])
        self.assertFalse(by_order[(-1, -1)]["independent_conjugate_representative"])
        self.assertFalse(by_order[(2, 2)]["is_primitive"])
        self.assertEqual(by_order[(2, 2)]["harmonic_multiplier"], 2)

    def test_pb_phase_can_release_conjugate_pair_constraint(self):
        rows = validation.phase_relation_rows(
            np.array([[0, 0], [1, 0], [-1, 0]], dtype=np.int16),
            conjugate_encoding="detour-pb",
        )
        self.assertTrue(all(row["independent_conjugate_representative"] for row in rows))

    def test_paper_23_preset_matches_supporting_information(self):
        orders = {tuple(order) for order in validation.PAPER_23_ORDERS.tolist()}
        self.assertEqual(len(orders), 23)
        self.assertIn((-3, 3), orders)
        self.assertIn((-1, -3), orders)
        self.assertIn((0, 1), orders)
        self.assertIn((0, 2), orders)
        self.assertNotIn((0, 0), orders)


if __name__ == "__main__":
    unittest.main()
