"""Per-iteration measurements, detached from the loss and its gradients."""
import torch


@torch.no_grad()
def measure_step(intensity, targets, mask=None):
    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    foreground = (targets > 0) & mask
    background = (targets == 0) & mask
    foreground_counts = foreground.sum((-2, -1))
    present = foreground_counts > 0
    background_counts = background.sum((-2, -1))
    foreground_energy = (intensity * foreground).sum((-2, -1))
    foreground_mean = (foreground_energy / foreground_counts.clamp_min(1)).masked_fill(~present, float("nan"))
    background_mean = (intensity * background).sum((-2, -1)) / background_counts.clamp_min(1)
    background_variance = ((intensity-background_mean[:, None, None]).square()
                           * background).sum((-2, -1)) / background_counts.clamp_min(1)
    plane_mean = (intensity * mask).sum((-2, -1)) / mask.sum((-2, -1))
    foreground_level = foreground_mean / plane_mean
    snr = 20 * torch.log10((foreground_mean / (torch.sqrt(background_variance + 1e-9) + 1e-9)).clamp_min(1e-9))
    snr = snr.masked_fill(background_counts == 0, float("nan"))
    snr_present = present & (background_counts > 0)
    return {
        "eta_history": foreground_energy / intensity.sum((-2, -1)),
        "foreground_level_history": foreground_level,
        "brightness_cv_history": foreground_level[present].std(unbiased=False) / foreground_level[present].mean(),
        "snr_mean_db_history": snr[snr_present].mean() if snr_present.any() else intensity.new_tensor(float("nan")),
        "snr_min_db_history": snr[snr_present].min() if snr_present.any() else intensity.new_tensor(float("nan")),
        "normalized_background_variance_history": (background_variance / plane_mean.square()).mean(),
    }
