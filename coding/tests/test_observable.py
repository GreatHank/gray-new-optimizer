import numpy as np
import torch

from coding.geometry.observable import observable_mask
from coding.optimization.model import common_scale_rmse, gray_level_consistency_mse, shared_intensity
from coding.results.observable_metrics import evaluate
from coding.results.compare_observable import central_orders


def test_geometry_mask_uses_fftshift_axes_propagation_and_na():
    physics = {"wavelength_nm": 1, "period_x_nm": 1, "period_y_nm": 1,
               "n_in": 1, "n_out": 1, "na": 0.25,
               "incident_theta_deg": 0, "incident_azimuth_deg": 0}
    mask = observable_mask(np.array([[0, 0]]), (4, 4), physics)[0]
    expected = np.zeros((4, 4), dtype=bool)
    expected[1:4, 2] = True
    expected[2, 1:4] = True
    np.testing.assert_array_equal(mask, expected)


def test_outside_error_has_no_direct_image_gradient_but_inside_black_does():
    target = torch.tensor([[[0., 1.], [0., 0.]]], dtype=torch.float64)
    mask = torch.tensor([[[True, True], [False, False]]])
    raw = torch.tensor([[[3., 4.], [50., 60.]]], dtype=torch.float64, requires_grad=True)
    value = common_scale_rmse(raw, target, torch.ones(1), mask)
    value.backward()
    assert raw.grad[0, 0, 0] != 0  # Original black background inside the circle.
    assert torch.all(raw.grad[0, 1] == 0)
    changed = raw.detach().clone()
    changed[0, 1] += 1000
    torch.testing.assert_close(common_scale_rmse(changed, target, torch.ones(1), mask), value)
    expected = np.sqrt(np.mean(np.array([3/4, 4/4 - 4])**2) + 1e-9)
    np.testing.assert_allclose(value.item(), expected)


def test_masked_gray_consistency_ignores_outside_nonblack_label():
    raw = torch.tensor([[[0., 4.]], [[0., 8.]]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[0, 1], [0, 0]])  # Second channel's nonblack pixel is outside.
    counts = torch.tensor([[1., 1.], [2., 0.]], dtype=torch.float64)
    loss = gray_level_consistency_mse(raw, labels, counts)
    assert loss.item() == 0
    loss.backward()
    assert raw.grad[1, 0, 1] == 0


def test_observable_metrics_ignore_outside_and_report_full_fft_energy():
    raw = np.array([[[2., 4.], [8., 100.]]])
    target = np.array([[[0., 0.], [1., 0.]]], dtype=np.float32)
    labels = target.astype(np.uint8)
    mask = np.array([[[True, True], [True, False]]])
    rows, grays, summary = evaluate(raw, target, labels, np.array([[0, 0]]), mask)
    np.testing.assert_allclose(rows[0]["raw_background_mean"], 3)
    np.testing.assert_allclose(rows[0]["snr_db"], 20*np.log10(8))
    np.testing.assert_allclose(summary["full_fft_total_intensity"], 114)
    np.testing.assert_allclose(summary["observable_energy_fraction"], 14/114)
    assert grays[0]["pixels"] == 1
    changed = raw.copy()
    changed[0, 1, 1] = 1000
    _, _, changed_summary = evaluate(changed, target, labels, np.array([[0, 0]]), mask)
    assert changed_summary["masked_rmse"] == summary["masked_rmse"]
    assert changed_summary["observable_energy_fraction"] != summary["observable_energy_fraction"]


def test_missing_background_has_undefined_snr_and_cnr():
    raw = np.array([[[2., 4.]]])
    target = np.array([[[1., 0.]]], dtype=np.float32)
    mask = np.array([[[True, False]]])
    rows, _, summary = evaluate(raw, target, target.astype(np.uint8), np.array([[0, 0]]), mask)
    assert np.isnan(rows[0]["snr_db"])
    assert np.isnan(rows[0]["cnr"])
    assert summary["snr_defined_channels"] == 0


def test_full_fft_parseval_unchanged_with_observable_mask():
    rng = np.random.default_rng(7)
    phases = rng.normal(size=(3, 8, 8))
    orders = np.array([[0, 0], [1, -1]])
    raw = shared_intensity(*[torch.tensor(p) for p in phases], torch.tensor(orders)).numpy()
    np.testing.assert_allclose(raw.sum(axis=(1, 2)), 8**4, rtol=1e-12)


def test_visual_grid_center_has_one_order_for_3x3_and_four_for_4x4():
    for size, expected in ((3, {(-1, -1)}),
                           (4, {(-1, -1), (0, -1), (-1, 0), (0, 0)})):
        positions = np.array([(r, c) for r in range(size) for c in range(size)])
        orders = np.array([(c - 2, size - 3 - r) for r, c in positions])
        actual_size, pairs = central_orders(orders, positions)
        assert actual_size == size
        assert pairs == expected
