"""Fixed shared dx/dy/PP forward, image RMSE and optional gray consistency."""
import torch
import torch.fft as fft


def fftshift2(x):
    return torch.roll(
        torch.roll(x, shifts=(x.shape[-2] // 2,), dims=(-2,)),
        shifts=(x.shape[-1] // 2,),
        dims=(-1,),
    )


def shared_intensity(dx, dy, pp, pair_mat):
    phase = pair_mat[:, 0, None, None] * dx + pair_mat[:, 1, None, None] * dy + pp
    return torch.abs(fftshift2(fft.fft2(torch.exp(1j * phase)))) ** 2


def common_scale_rmse(intensity, targets, weights, mask=None):
    pixels = targets.shape[-1] * targets.shape[-2]
    difference = intensity / pixels - targets / targets.mean()
    if mask is not None:
        return torch.sqrt(difference.square().masked_select(mask).mean() + 1e-9)
    return torch.sqrt(torch.sum(weights * difference.square().mean((-2, -1)))
                      / weights.sum() + 1e-9)


def gray_level_consistency_mse(intensity, labels, counts):
    """Penalize different raw brightness for the same nonblack target label."""
    pixels = intensity.shape[-1] * intensity.shape[-2]
    sums = torch.zeros_like(counts).scatter_add(1, labels, (intensity / pixels).flatten(1))
    foreground_counts = counts[:, 1:]
    means = sums[:, 1:] / foreground_counts.clamp_min(1)
    pooled = sums[:, 1:].sum(0) / foreground_counts.sum(0).clamp_min(1)
    return (foreground_counts * (means - pooled).square()).sum() / foreground_counts.sum()
