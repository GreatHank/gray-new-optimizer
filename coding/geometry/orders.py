"""Common-incidence diffraction directions from the grating equation."""
import numpy as np


def order_directions(
    orders,
    wavelength_nm,
    period_x_nm,
    period_y_nm,
    theta_deg,
    azimuth_deg,
    n_in,
    n_out,
    numerical_aperture,
):
    theta = np.deg2rad(theta_deg)
    azimuth = np.deg2rad(azimuth_deg)
    incident_x = n_in * np.sin(theta) * np.cos(azimuth)
    incident_y = n_in * np.sin(theta) * np.sin(azimuth)
    transverse_x = incident_x + orders[:, 0] * wavelength_nm / period_x_nm
    transverse_y = incident_y + orders[:, 1] * wavelength_nm / period_y_nm
    ux = transverse_x / n_out
    uy = transverse_y / n_out
    radius = np.hypot(ux, uy)
    propagating = radius <= 1.0 + 1e-12
    captured = propagating & (np.hypot(transverse_x, transverse_y) <= numerical_aperture + 1e-12)
    theta_out = np.full(radius.shape, np.nan)
    theta_out[propagating] = np.rad2deg(np.arcsin(np.clip(radius[propagating], 0, 1)))
    azimuth_out = np.rad2deg(np.arctan2(uy, ux))
    return {
        "ux": ux,
        "uy": uy,
        "radius": radius,
        "propagating": propagating,
        "captured": captured,
        "air_margin": 1.0 - radius,
        "theta_out_deg": theta_out,
        "azimuth_out_deg": azimuth_out,
    }
