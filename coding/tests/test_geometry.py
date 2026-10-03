import unittest

import numpy as np


from coding.geometry.preflight import validate_physics

BASE = {
    "wavelength_nm": 500,
    "period_x_nm": 2000,
    "period_y_nm": 2000,
    "n_in": 1,
    "n_out": 1,
    "na": 1,
    "incident_theta_deg": 0,
    "incident_azimuth_deg": 0,
    "common_incidence": True,
}


class PhysicsPreflightTests(unittest.TestCase):
    def test_distinguishes_center_and_actual_support_coverage(self):
        target = np.zeros((1, 8, 8))
        target[0, 4, 7] = 1
        result = validate_physics(np.asarray([[4, 0]]), BASE, target)
        self.assertEqual(result["center_propagating"], 1)
        self.assertEqual(result["coverage"]["order_centers"], "pass")
        self.assertEqual(
            result["coverage"]["actual_target_support"], "partial_or_outside"
        )

    def test_zero_order_and_axis_orders_are_eligible(self):
        axis_result = validate_physics(np.asarray([[0, 1], [1, 0]]), BASE)
        zero_result = validate_physics(np.asarray([[0, 0]]), BASE)
        self.assertEqual(axis_result["zero_order_count"], 0)
        self.assertEqual(zero_result["zero_order_count"], 1)
        self.assertEqual(zero_result["coverage"]["order_centers"], "pass")
        self.assertEqual(zero_result["failures"], [])

    def test_illegal_propagation_and_na_are_reported_separately(self):
        outside = validate_physics(np.asarray([[5, 0]]), BASE)
        low_na = validate_physics(np.asarray([[1, 0]]), {**BASE, "na": 0.1})
        self.assertIn("nonpropagating_centers", outside["failures"])
        self.assertIn("centers_outside_na", low_na["failures"])

    def test_missing_parameters_cannot_be_called_validated(self):
        result = validate_physics(np.asarray([[1, 0]]), {"wavelength_nm": 500})
        self.assertEqual(result["status"], "invalid")
        self.assertIn("missing_parameters", result["failures"])

    def test_partial_cell_can_encode_propagating_pixels_with_nonpropagating_center(self):
        config = {**BASE, "wavelength_nm": 532, "period_x_nm": 1333.458,
                  "period_y_nm": 1333.458,
                  "strict_require": ["all_cells_intersect_propagation", "actual_target_support"]}
        targets = np.zeros((1, 8, 8))
        targets[0, -1, 0] = 1
        result = validate_physics(np.array([[2, 2]]), config, targets)
        self.assertEqual(result["center_propagating"], 0)
        self.assertEqual(result["coverage"]["actual_target_support"], "pass")
        self.assertEqual(result["strict_failures"], [])
        self.assertEqual(result["status"], "layout_pass_with_partial_cells")
        targets[0, 0, -1] = 1
        outside = validate_physics(np.array([[2, 2]]), config, targets)
        self.assertEqual(outside["layout"]["outside_target_pixels"], 1)
        self.assertIn("actual_target_support", outside["strict_failures"])
        self.assertEqual(outside["status"], "fail")

    def test_symmetric_25_covers_84_degree_ring_and_respects_count(self):
        orders = np.array([(m, n) for m in range(-2, 3) for n in range(-2, 3)])
        config = {**BASE, "wavelength_nm": 532, "period_x_nm": 1333.458,
                  "period_y_nm": 1333.458,
                  "layout_requirements": {"max_order_count": 25, "ring_theta_deg": 84,
                                          "minimum_ring_coverage": 1.0},
                  "strict_require": ["order_count", "ring_coverage", "all_cells_intersect_propagation"]}
        result = validate_physics(orders, config)
        self.assertEqual(result["layout"]["ring_coverage_fraction"], 1.0)
        self.assertEqual(result["layout"]["fully_visible_cells"], 9)
        self.assertEqual(result["layout"]["partly_visible_cells"], 16)
        self.assertEqual(result["center_propagating"], 21)
        self.assertEqual(result["selected_conjugate_pair_count"], 12)
        self.assertEqual(result["strict_failures"], [])
        missing_cell = validate_physics(orders[:-1], config)
        self.assertIn("ring_coverage", missing_cell["strict_failures"])
        too_many = validate_physics(np.vstack([orders, [3, 0]]), config)
        self.assertIn("order_count", too_many["strict_failures"])


if __name__ == "__main__":
    unittest.main()
