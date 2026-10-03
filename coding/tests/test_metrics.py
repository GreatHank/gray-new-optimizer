import numpy as np

from coding.results.metrics import channel_metrics, evaluation_metrics, grayscale_metrics
from coding.optimization.diagnostics import measure_step
from coding.results.storage import json_value
import torch


def test_fixed_window_snr_cnr_response_and_energy_use_raw_intensity():
    target = np.array([[0, 0, 0], [1/3, 2/3, 1]], dtype=np.float32)
    image = np.array([[2, 4, 6], [18, 28, 38]], dtype=np.float64)
    targets = np.stack([target, target])
    raw = np.stack([image, image*2])
    orders = np.array([[0, 0], [1, 0]])
    rows, summary = evaluation_metrics(raw, targets)
    sigma = np.sqrt(8/3)
    np.testing.assert_allclose(rows[:, -1], 20*np.log10(28/sigma), rtol=1e-11)
    np.testing.assert_allclose(rows[:, -2], (28-4)/sigma, rtol=1e-11)
    np.testing.assert_allclose(rows[:, 6:9], [[14/16, 24/16, 34/16]]*2, rtol=1e-11)
    assert summary['grayscale_monotonic_channels'] == 2
    brightness = channel_metrics(raw, targets, orders)
    np.testing.assert_array_equal(brightness[:, 6], [28, 56])
    np.testing.assert_allclose(brightness[:, 5], 84/96, rtol=1e-11)
    gray = grayscale_metrics(raw, targets, orders)
    np.testing.assert_allclose(gray[:, 8:11], [[14/16, 24/16, 34/16]]*2, rtol=1e-11)


def test_missing_highest_gray_keeps_ratio_undefined():
    targets = np.array([[[0, 0], [1/3, 2/3]]], dtype=np.float32)
    raw = np.array([[[2, 4], [18, 28]]], dtype=np.float64)
    rows, _ = evaluation_metrics(raw, targets)
    assert np.isnan(rows[0, 4])
    assert np.isnan(rows[0, 8])


def test_black_only_order_retains_energy_and_background_without_inventing_snr():
    targets = np.array([[[0, 0], [1/3, 1]], [[0, 0], [0, 0]]], dtype=np.float32)
    raw = np.array([[[2, 4], [18, 38]], [[1, 3], [5, 7]]], dtype=np.float64)
    orders = np.array([[0, 0], [1, 0]])
    brightness = channel_metrics(raw, targets, orders)
    assert brightness[1, 4] == 16
    assert brightness[1, 5] == 0
    assert np.isnan(brightness[1, 6])
    assert brightness[1, 7] == 4
    rows, summary = evaluation_metrics(raw, targets)
    assert np.isnan(rows[1, 1])
    assert np.isnan(rows[1, -1])
    assert rows[1, 3] == 0
    assert summary['foreground_metric_channel_count'] == 1
    assert summary['background_only_channel_count'] == 1
    np.testing.assert_allclose(summary['foreground_background_snr_db_mean'],20*np.log10(28))
    assert json_value(float(rows[1, -1])) is None
    gray = grayscale_metrics(raw, targets, orders)
    assert gray[1, 4] == 4
    assert gray[1, 5] == 5
    measured = measure_step(torch.tensor(raw),torch.tensor(targets))
    assert torch.isnan(measured['foreground_level_history'][1])
    np.testing.assert_allclose(measured['snr_mean_db_history'].item(),20*np.log10(28),atol=1e-7)
    # The background-only order still contributes to the all-channel background statistic.
    np.testing.assert_allclose(measured['normalized_background_variance_history'].item(),
                               (1/15.5**2+5/4**2)/2)
