"""Evaluate fixed observable FFT bins on the original common intensity scale."""

import csv

import numpy as np


def evaluate(raw, targets, labels, orders, mask):
    if raw.shape != targets.shape or mask.shape != raw.shape or labels.shape != raw.shape:
        raise ValueError("Raw, target, labels and mask must have the same shape")
    pixels = raw.shape[-1] * raw.shape[-2]
    common_gray = raw.astype(np.float64) * float(targets.mean()) / pixels
    residual = raw.astype(np.float64) / pixels - targets.astype(np.float64) / float(targets.mean())
    rows, gray_rows = [], []
    for c, (order, visible) in enumerate(zip(orders, mask)):
        foreground = visible & (labels[c] > 0)
        background = visible & (labels[c] == 0)
        fg = raw[c][foreground]
        bg = raw[c][background]
        fg_mean = float(fg.mean()) if fg.size else np.nan
        bg_mean = float(bg.mean()) if bg.size else np.nan
        bg_std = float(bg.std()) if bg.size else np.nan
        total = float(raw[c].sum(dtype=np.float64))
        inside = float(raw[c][visible].sum(dtype=np.float64))
        snr = 20 * np.log10(fg_mean / bg_std) if fg.size and bg.size and bg_std > 0 and fg_mean > 0 else np.nan
        cnr = (fg_mean - bg_mean) / bg_std if fg.size and bg.size and bg_std > 0 else np.nan
        rows.append({
            "channel": c + 1, "m": int(order[0]), "n": int(order[1]),
            "observable_pixels": int(visible.sum()), "foreground_pixels": int(fg.size),
            "background_pixels": int(bg.size), "masked_rmse": float(np.sqrt(np.mean(residual[c][visible]**2))),
            "raw_foreground_mean": fg_mean, "raw_background_mean": bg_mean,
            "raw_background_std": bg_std, "snr_db": snr, "cnr": cnr,
            "full_fft_total_intensity": total, "observable_intensity": inside,
            "observable_energy_fraction": inside / total,
            "common_scale_foreground_mean": float(common_gray[c][foreground].mean()) if fg.size else np.nan,
            "common_scale_background_mean": float(common_gray[c][background].mean()) if bg.size else np.nan,
        })
        inside_mean = float(raw[c][visible].mean())
        for code in np.unique(labels[c][foreground]):
            selected = foreground & (labels[c] == code)
            level_mean = float(raw[c][selected].mean())
            response = (level_mean - bg_mean) / inside_mean if bg.size else np.nan
            gray_rows.append({"channel": c + 1, "m": int(order[0]), "n": int(order[1]),
                              "gray_label": int(code), "pixels": int(selected.sum()),
                              "raw_mean": level_mean, "effective_response": response})
    valid_snr = [row["snr_db"] for row in rows if np.isfinite(row["snr_db"])]
    valid_cnr = [row["cnr"] for row in rows if np.isfinite(row["cnr"])]
    summary = {
        "scope": "Fixed observable FFT bins; original target black pixels inside the mask are background",
        "masked_rmse": float(np.sqrt(np.mean(residual[mask]**2))),
        "foreground_background_snr_db_mean": float(np.mean(valid_snr)) if valid_snr else np.nan,
        "foreground_background_snr_db_min": float(np.min(valid_snr)) if valid_snr else np.nan,
        "foreground_background_cnr_mean": float(np.mean(valid_cnr)) if valid_cnr else np.nan,
        "snr_defined_channels": len(valid_snr), "cnr_defined_channels": len(valid_cnr),
        "observable_pixels": int(mask.sum()), "full_fft_total_intensity": float(raw.sum(dtype=np.float64)),
        "observable_intensity": float(raw[mask].sum(dtype=np.float64)),
        "observable_energy_fraction": float(raw[mask].sum(dtype=np.float64) / raw.sum(dtype=np.float64)),
        "common_display_white_raw": float(pixels / targets.mean()),
    }
    return rows, gray_rows, summary


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
