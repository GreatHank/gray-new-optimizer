"""Fixed observable FFT bins from incidence, propagation and numerical aperture."""

import numpy as np


def observable_mask(orders, shape, physics):
    height, width = shape
    yy, xx = np.indices((height, width))
    local_x = (xx - width // 2) / width
    local_y = -(yy - height // 2) / height
    theta = np.deg2rad(physics["incident_theta_deg"])
    azimuth = np.deg2rad(physics["incident_azimuth_deg"])
    incident = physics["n_in"] / physics["n_out"] * np.sin(theta)
    incident_x = incident * np.cos(azimuth)
    incident_y = incident * np.sin(azimuth)
    pitch_x = physics["wavelength_nm"] / (physics["n_out"] * physics["period_x_nm"])
    pitch_y = physics["wavelength_nm"] / (physics["n_out"] * physics["period_y_nm"])
    radius = min(1.0, physics["na"] / physics["n_out"])
    orders = np.asarray(orders)
    ux = incident_x + (orders[:, 0, None, None] + local_x) * pitch_x
    uy = incident_y + (orders[:, 1, None, None] + local_y) * pitch_y
    mask = ux**2 + uy**2 <= radius**2
    if not np.all(mask.any(axis=(1, 2))):
        raise ValueError("An order has no observable FFT bins")
    return mask
