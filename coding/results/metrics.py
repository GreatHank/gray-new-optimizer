"""Raw-scale evaluation and common-exposure figures; no optimization penalties."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def channel_metrics(raw, targets, pair_mat):
    rows = []
    epsilon = 1e-12
    for channel in range(targets.shape[0]):
        mask = targets[channel] > 0
        foreground = raw[channel][mask]
        background = raw[channel][~mask]
        total_energy = float(np.sum(raw[channel]))
        foreground_mean = float(np.mean(foreground)) if foreground.size else np.nan
        background_mean = float(np.mean(background))
        rows.append([
            channel + 1,
            pair_mat[channel, 0],
            pair_mat[channel, 1],
            np.count_nonzero(mask),
            total_energy,
            float(np.sum(foreground) / (total_energy + epsilon)),
            foreground_mean,
            background_mean,
            foreground_mean / (background_mean + epsilon),
        ])
    return np.asarray(rows, dtype=np.float64)


def grayscale_metrics(raw, targets, pair_mat, gray_level_count=4):
    rows = []
    epsilon = 1e-12
    levels = np.arange(1, gray_level_count, dtype=np.uint8)
    for channel in range(targets.shape[0]):
        target = targets[channel]
        target_codes = np.rint(target * (gray_level_count - 1)).astype(np.uint8)
        background = raw[channel][target == 0]
        background_mask = target == 0
        row_weights = background_mask.astype(np.float64)
        row_counts = np.sum(row_weights, axis=1)
        background_row_means = np.sum(
            raw[channel] * row_weights, axis=1
        ) / (row_counts + epsilon)
        present = np.asarray([np.any(target_codes == level) for level in levels])
        line_means = np.asarray(
            [
                raw[channel][target_codes == level].mean() if is_present else np.nan
                for level, is_present in zip(levels, present)
            ]
        )
        plane_mean = float(np.mean(raw[channel]))
        background_mean = float(np.mean(background))
        responses = (line_means - background_mean) / (plane_mean + epsilon)
        ratios = (
            responses / (responses[-1] + epsilon)
            if present[-1]
            else np.full(len(levels), np.nan)
        )
        present_responses = responses[present]
        present_gaps = np.diff(present_responses)
        line_variances = [
            float(np.var(raw[channel][target_codes == level])) if is_present else np.nan
            for level, is_present in zip(levels, present)
        ]
        rows.append(
            [
                channel + 1,
                pair_mat[channel, 0],
                pair_mat[channel, 1],
                plane_mean,
                background_mean,
                float(np.var(background)),
                float(np.var(background_row_means)),
                float(np.percentile(background, 95)),
                *responses,
                *ratios,
                int(present.any() and np.all(present_gaps > 0)),
                float(np.min(present_gaps)) if present_gaps.size else np.nan,
                *line_variances,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def gray_names(gray_level_count):
    return tuple(f"S_{i}_{gray_level_count-1}" for i in range(1, gray_level_count-1)) + ("S_1",)


def evaluation_channel_headers(gray_level_count=4):
    return (
    "channel",
    "structure_cosine",
    "foreground_coverage_above_background_p95",
    "grayscale_monotonic",
    "grayscale_ratio_rmse",
    "normalized_min_level_gap",
    *gray_names(gray_level_count),
    "background_cv",
    "background_p95_ratio",
    "background_row_cv",
    "foreground_background_cnr",
    "foreground_background_snr_db",
    )


EVALUATION_SUMMARY_HEADERS = (
    "structure_cosine_mean",
    "structure_cosine_min",
    "foreground_coverage_mean",
    "foreground_coverage_min",
    "grayscale_monotonic_channels",
    "grayscale_ratio_rmse_mean",
    "grayscale_ratio_rmse_max",
    "S_1_3_mean",
    "S_1_3_min",
    "S_1_3_max",
    "S_1_3_cv",
    "S_2_3_mean",
    "S_2_3_min",
    "S_2_3_max",
    "S_2_3_cv",
    "S_1_mean",
    "S_1_min",
    "S_1_max",
    "S_1_cv",
    "background_cv_mean",
    "background_cv_max",
    "background_p95_ratio_mean",
    "background_p95_ratio_max",
    "background_row_cv_mean",
    "background_row_cv_max",
    "foreground_background_cnr_mean",
    "foreground_background_cnr_min",
    "foreground_background_snr_db_mean",
    "foreground_background_snr_db_min",
    "foreground_background_snr_db_max",
)


def evaluation_summary_headers(gray_level_count=4):
    return (
        *EVALUATION_SUMMARY_HEADERS[:7],
        *(f"{name}_{metric}" for name in gray_names(gray_level_count)
          for metric in ("mean", "min", "max", "cv")),
        *EVALUATION_SUMMARY_HEADERS[-11:],
    )


def evaluation_metrics(
    raw,
    targets,
    desired_gray_ratios=(1 / 3, 2 / 3, 1.0),
):
    rows = []
    epsilon = 1e-12
    desired_gray_ratios = np.asarray(desired_gray_ratios, dtype=np.float64)
    levels = np.linspace(0, 1, len(desired_gray_ratios) + 1)[1:]

    for channel in range(targets.shape[0]):
        target = targets[channel]
        foreground_mask = target > 0
        background_mask = target == 0
        image = raw[channel].astype(np.float64, copy=False)
        background = image[background_mask]
        background_mean = float(np.mean(background))
        background_std = float(np.std(background))
        background_p95 = float(np.percentile(background, 95))
        plane_mean = float(np.mean(image))
        foreground_mean = float(np.mean(image[foreground_mask])) if foreground_mask.any() else np.nan
        foreground_background_cnr = (
            foreground_mean - background_mean
        ) / (background_std + epsilon)
        foreground_background_snr_db = 20.0 * np.log10(
            max(foreground_mean / (background_std + epsilon), epsilon)
        )

        positive_contrast = np.maximum(image - background_mean, 0.0)
        target_shape = foreground_mask.astype(np.float64)
        desired_target = np.zeros_like(target, dtype=np.float64)
        for label, desired_ratio in zip(levels, desired_gray_ratios):
            desired_target[np.isclose(target, label, rtol=0, atol=1e-6)] = (
                desired_ratio
            )
        structure_contrast = positive_contrast.copy()
        structure_contrast[foreground_mask] /= desired_target[foreground_mask]
        structure_cosine = float(
            np.sum(structure_contrast * target_shape)
            / (
                np.linalg.norm(structure_contrast.ravel())
                * np.linalg.norm(target_shape.ravel())
                + epsilon
            )
        )
        if not foreground_mask.any():
            structure_cosine = np.nan
        foreground_coverage = float(
            np.mean(image[foreground_mask] > background_p95)
        ) if foreground_mask.any() else np.nan

        present = np.asarray(
            [
                np.any(np.isclose(target, level, rtol=0, atol=1e-6))
                for level in levels
            ]
        )
        level_means = np.asarray(
            [
                np.mean(image[np.isclose(target, level, rtol=0, atol=1e-6)])
                if is_present
                else np.nan
                for level, is_present in zip(levels, present)
            ]
        )
        responses = (level_means - background_mean) / (plane_mean + epsilon)
        present_responses = responses[present]
        gaps = np.diff(present_responses)
        grayscale_monotonic = int(present.any() and np.all(gaps > 0))
        if present[-1]:
            ratios = responses / (responses[-1] + epsilon)
            grayscale_ratio_rmse = float(
                np.sqrt(
                    np.mean(
                        (ratios[present] - desired_gray_ratios[present]) ** 2
                    )
                )
            )
            normalized_min_gap = (
                float(np.min(gaps) / (abs(responses[-1]) + epsilon))
                if gaps.size
                else np.nan
            )
        else:
            grayscale_ratio_rmse = np.nan
            normalized_min_gap = np.nan

        background_weights = background_mask.astype(np.float64)
        row_counts = np.sum(background_weights, axis=1)
        valid_rows = row_counts > 0
        background_row_means = np.sum(
            image * background_weights, axis=1
        )[valid_rows] / row_counts[valid_rows]

        rows.append(
            [
                channel + 1,
                structure_cosine,
                foreground_coverage,
                grayscale_monotonic,
                grayscale_ratio_rmse,
                normalized_min_gap,
                *responses,
                background_std / (background_mean + epsilon),
                background_p95 / (background_mean + epsilon),
                float(np.std(background_row_means))
                / (float(np.mean(background_row_means)) + epsilon),
                foreground_background_cnr,
                foreground_background_snr_db,
            ]
        )

    channel_rows = np.asarray(rows, dtype=np.float64)
    foreground_present = np.any(targets > 0, axis=(1, 2))
    foreground_rows = channel_rows[foreground_present]
    ratio_errors = channel_rows[:, 4]
    ratio_errors = ratio_errors[np.isfinite(ratio_errors)]
    summary = {
        "foreground_metric_channel_count": int(foreground_present.sum()),
        "background_only_channel_count": int((~foreground_present).sum()),
        "foreground_metrics_scope": "All original nonblack target pixels; foreground metrics undefined for black-only channels. Background and full-plane loss retain every channel.",
        "structure_cosine_mean": float(np.mean(foreground_rows[:, 1])),
        "structure_cosine_min": float(np.min(foreground_rows[:, 1])),
        "foreground_coverage_mean": float(np.mean(foreground_rows[:, 2])),
        "foreground_coverage_min": float(np.min(foreground_rows[:, 2])),
        "grayscale_monotonic_channels": int(np.sum(channel_rows[:, 3])),
        "grayscale_ratio_rmse_mean": float(np.mean(ratio_errors)) if ratio_errors.size else np.nan,
        "grayscale_ratio_rmse_max": float(np.max(ratio_errors)) if ratio_errors.size else np.nan,
    }
    for column, name in enumerate(gray_names(len(levels) + 1), 6):
        values = channel_rows[:, column]
        valid = values[np.isfinite(values)]
        if valid.size:
            mean = float(np.mean(valid))
            summary[f"{name}_mean"] = mean
            summary[f"{name}_min"] = float(np.min(valid))
            summary[f"{name}_max"] = float(np.max(valid))
            summary[f"{name}_cv"] = float(np.std(valid) / (abs(mean) + epsilon))
        else:
            summary[f"{name}_mean"] = np.nan
            summary[f"{name}_min"] = np.nan
            summary[f"{name}_max"] = np.nan
            summary[f"{name}_cv"] = np.nan
    first_tail = 6 + len(levels)
    for column, name in (
        (first_tail, "background_cv"),
        (first_tail + 1, "background_p95_ratio"),
        (first_tail + 2, "background_row_cv"),
    ):
        values = channel_rows[:, column]
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_max"] = float(np.max(values))
    cnr_values = foreground_rows[:, first_tail + 3]
    snr_db_values = foreground_rows[:, first_tail + 4]
    summary["foreground_background_cnr_mean"] = float(np.mean(cnr_values))
    summary["foreground_background_cnr_min"] = float(np.min(cnr_values))
    summary["foreground_background_snr_db_mean"] = float(np.mean(snr_db_values))
    summary["foreground_background_snr_db_min"] = float(np.min(snr_db_values))
    summary["foreground_background_snr_db_max"] = float(np.max(snr_db_values))
    return channel_rows, summary


def gray_level_cv_table(raw, targets, channel_rows, gray_level_count):
    """Summarize raw and background-adjusted gray means in fixed windows."""
    rows = []
    for level in range(1, gray_level_count):
        label = level / (gray_level_count - 1)
        masks = np.isclose(targets, label, rtol=0, atol=1e-6)
        pixel_counts = np.count_nonzero(masks, axis=(1, 2))
        present = pixel_counts > 0
        raw_means = np.asarray([np.mean(raw[channel][masks[channel]])
                                for channel in np.flatnonzero(present)])
        responses = channel_rows[present, 5 + level]
        raw_mean = float(np.mean(raw_means)) if raw_means.size else np.nan
        raw_std = float(np.std(raw_means)) if raw_means.size > 1 else np.nan
        response_mean = float(np.mean(responses)) if responses.size else np.nan
        response_std = float(np.std(responses)) if responses.size > 1 else np.nan
        rows.append((level, int(round(255 * label)), int(present.sum()),
                     int(pixel_counts.sum()),
                     int(pixel_counts[present].min()) if present.any() else 0,
                     raw_mean, raw_std,
                     raw_std / abs(raw_mean)
                     if raw_means.size > 1 and raw_mean != 0 else np.nan,
                     response_mean, response_std,
                     response_std / abs(response_mean)
                     if responses.size > 1 and response_mean != 0 else np.nan))
    return np.asarray(rows, dtype=np.float64)


def save_comparison(targets, optimized_01, output_file, title):
    pairs_per_row = 4
    rows = int(np.ceil(targets.shape[0] / pairs_per_row))
    figure, axes = plt.subplots(
        rows, pairs_per_row * 2, figsize=(16, rows * 3), squeeze=False
    )

    for channel in range(targets.shape[0]):
        row = channel // pairs_per_row
        column = (channel % pairs_per_row) * 2
        for axis, image, panel_title in (
            (axes[row, column], targets[channel], f"Ch{channel + 1} Target"),
            (axes[row, column + 1], optimized_01[channel], f"Ch{channel + 1} Optimized"),
        ):
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_title(panel_title, fontsize=9)
            axis.axis("off")

    used_axes = targets.shape[0] * 2
    for axis in axes.flat[used_axes:]:
        axis.axis("off")

    figure.suptitle(title, fontsize=16)
    figure.tight_layout()
    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_metrics_csv(path, brightness, grayscale, evaluation, gray_level_count=4):
    combined = np.column_stack(
        [brightness, grayscale[:, 3:], evaluation[:, 1:]]
    )
    names = gray_names(gray_level_count)
    header = ",".join((
        "channel", "m", "n", "target_pixels", "total_energy", "target_efficiency",
        "foreground_mean", "background_mean", "foreground_background_contrast",
        "plane_mean", "gray_background_mean", "background_variance",
        "background_row_variance", "background_p95",
        *(f"{name}_raw" for name in names),
        *(f"ratio_{name[2:]}" for name in names),
        "monotonic_raw", "min_gap",
        *(f"line_variance_{name[2:]}" for name in names),
        *evaluation_channel_headers(gray_level_count)[1:],
    ))
    np.savetxt(path, combined, delimiter=",", header=header, comments="")
