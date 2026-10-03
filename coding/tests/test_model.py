import numpy as np
import torch

from coding.optimization.model import shared_intensity, common_scale_rmse, gray_level_consistency_mse


def numpy_forward(phases, orders):
    return np.stack([np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j*(
        m*phases[0]+n*phases[1]+phases[2])))))**2 for m, n in orders])


def test_all_25_orders_match_independent_numpy_and_parseval():
    phases = np.random.default_rng(21).uniform(-np.pi, np.pi, (3, 8, 8))
    orders = np.array([(m, n) for m in range(-2, 3) for n in range(-2, 3)])
    actual = shared_intensity(*[torch.tensor(p) for p in phases], torch.tensor(orders)).numpy()
    np.testing.assert_allclose(actual, numpy_forward(phases, orders), rtol=2e-12, atol=2e-10)
    np.testing.assert_allclose(actual.sum((1, 2)), 8**4, rtol=2e-12)


def test_zero_order_depends_only_on_pp():
    phases = np.random.default_rng(20).normal(size=(3, 8, 8))
    actual = shared_intensity(*[torch.tensor(p) for p in phases], torch.tensor([[0, 0]])).numpy()
    expected = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j*phases[2]))))**2
    np.testing.assert_allclose(actual[0], expected, rtol=2e-12, atol=2e-10)


def test_spatial_pp_breaks_forced_opposite_mirror_but_constant_pp_keeps_it():
    rng = np.random.default_rng(702)
    a, b = rng.uniform(-np.pi, np.pi, (2, 8, 8))
    phases = [(a-b)/2, np.zeros_like(a), (a+b)/2]
    orders = torch.tensor([[1, 0], [-1, 0]])
    actual = shared_intensity(*[torch.tensor(p) for p in phases], orders).numpy()
    expected = np.stack([np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j*p))))**2 for p in (a, b)])
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-10)
    negative = (-np.arange(8)) % 8
    assert not np.allclose(actual[1], actual[0][np.ix_(negative, negative)])
    phases[2] = np.full_like(a, 0.73)
    constant = shared_intensity(*[torch.tensor(p) for p in phases], orders).numpy()
    np.testing.assert_allclose(constant[1], constant[0][np.ix_(negative, negative)], rtol=2e-12, atol=2e-10)


def test_rmse_and_analytic_gradient_match_independent_reference():
    rng = np.random.default_rng(31)
    phases = rng.normal(size=(3, 4, 4))
    orders = np.array([[0, 0], [-2, 1], [1, -1]])
    target = rng.uniform(0, 1, (3, 4, 4))
    parameters = [torch.tensor(p, requires_grad=True) for p in phases]
    intensity = shared_intensity(*parameters, torch.tensor(orders))
    loss = common_scale_rmse(intensity, torch.tensor(target), torch.ones(3, dtype=torch.float64))
    raw = numpy_forward(phases, orders)
    pixels = 16
    residual = raw/pixels - target/target.mean()
    expected_loss = np.sqrt(np.mean(residual**2)+1e-9)
    np.testing.assert_allclose(loss.item(), expected_loss, rtol=2e-12)
    loss.backward()
    intensity_gradient = residual/(len(orders)*pixels**2*expected_loss)
    phase_gradient = []
    for (m, n), g in zip(orders, intensity_gradient):
        field = np.exp(1j*(m*phases[0]+n*phases[1]+phases[2]))
        amplitude = np.fft.fftshift(np.fft.fft2(field))
        field_gradient = 2*pixels*np.fft.ifft2(np.fft.ifftshift(g*amplitude))
        phase_gradient.append(np.imag(np.conj(field)*field_gradient))
    phase_gradient = np.stack(phase_gradient)
    gradients = [(phase_gradient*orders[:, 0, None, None]).sum(0),
                 (phase_gradient*orders[:, 1, None, None]).sum(0), phase_gradient.sum(0)]
    for actual, expected in zip(parameters, gradients):
        np.testing.assert_allclose(actual.grad.numpy(), expected, rtol=2e-11, atol=2e-12)


def test_pp_wrap_preserves_complex_field_and_affine_order_relations():
    pp = torch.tensor([-np.pi-0.2, np.pi+0.3, 3*np.pi+0.1], dtype=torch.float64)
    wrapped = torch.remainder(pp+np.pi, 2*np.pi)-np.pi
    torch.testing.assert_close(torch.exp(1j*pp), torch.exp(1j*wrapped), rtol=1e-14, atol=1e-14)
    dx, dy, p = np.random.default_rng(5).normal(size=(3, 8, 8))
    phase = lambda m, n: m*dx+n*dy+p
    np.testing.assert_allclose(phase(1, 0)+phase(-1, 0), 2*phase(0, 0), atol=1e-14)
    np.testing.assert_allclose(phase(1, 0)+phase(0, 1), phase(1, 1)+phase(0, 0), atol=1e-14)


def test_gray_consistency_matches_weighted_variance_and_finite_difference_gradient():
    raw = torch.tensor([[[0., 4.], [8., 12.]], [[100., 8.], [8., 12.]]],
                       dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 3], [0, 1, 1, 3]])
    counts = torch.tensor([[1., 1., 1., 1.], [1., 2., 0., 1.]], dtype=torch.float64)
    # Label 1 has means 1 and 2, sample counts 1 and 2, pooled mean 5/3.
    # Its weighted squared deviation is 2/3; there are six foreground pixels.
    np.testing.assert_allclose(gray_level_consistency_mse(raw, labels, counts).item(), 1/9)
    assert torch.autograd.gradcheck(lambda x: gray_level_consistency_mse(x, labels, counts), (raw,))
    raw.grad = None
    gray_level_consistency_mse(raw, labels, counts).backward()
    assert torch.all(raw.grad.flatten(1)[:, 0] == 0)


def test_gray_consistency_does_not_equalize_genuinely_different_gray_values():
    labels = torch.tensor([[0, 1, 2, 3], [0, 1, 1, 3]])
    counts = torch.tensor([[1., 1., 1., 1.], [1., 2., 0., 1.]], dtype=torch.float64)
    raw = torch.tensor([[[0., 4.], [8., 12.]], [[100., 4.], [4., 12.]]], dtype=torch.float64)
    assert gray_level_consistency_mse(raw, labels, counts).item() == 0
