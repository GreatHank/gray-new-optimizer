import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEVELS = np.asarray([1 / 3, 2 / 3, 1.0], dtype=np.float64)


def illumination_metrics(raw, targets, factor):
    scaled = raw.astype(np.float64, copy=False) * factor
    responses = []
    for image, target in zip(scaled, targets):
        background_mean = float(np.mean(image[target == 0]))
        responses.append(
            [
                float(np.mean(image[np.isclose(target, level, atol=1e-6)]))
                - background_mean
                for level in LEVELS
            ]
        )
    responses = np.asarray(responses)
    means = np.mean(responses, axis=0)
    cvs = np.std(responses, axis=0) / np.abs(means)
    ratios = np.mean(responses / responses[:, 2:3], axis=0)
    return {
        "factor": float(factor),
        "plane_mean": float(np.mean(scaled)),
        "level_means": means,
        "level_cvs": cvs,
        "level_ratios": ratios,
    }


def write_metrics(rows, output_file):
    headers = [
        "illumination_factor",
        "plane_mean_raw",
        "level_1_3_mean_raw",
        "level_2_3_mean_raw",
        "level_1_mean_raw",
        "level_1_3_cv",
        "level_2_3_cv",
        "level_1_cv",
        "ratio_1_3_to_1",
        "ratio_2_3_to_1",
        "ratio_1_to_1",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [
                    row["factor"],
                    row["plane_mean"],
                    *row["level_means"],
                    *row["level_cvs"],
                    *row["level_ratios"],
                ]
            )


def channel_montage(raw, display_scale, tile_size=80):
    channel_count = raw.shape[0]
    columns = 6
    rows = int(np.ceil(channel_count / columns))
    canvas = np.zeros((rows * tile_size, columns * tile_size), dtype=np.uint8)
    for channel, image in enumerate(raw):
        shown = np.clip(image / display_scale, 0, 1)
        tile = Image.fromarray(np.uint8(np.round(shown * 255))).resize(
            (tile_size, tile_size), Image.Resampling.BILINEAR
        )
        row, column = divmod(channel, columns)
        canvas[
            row * tile_size : (row + 1) * tile_size,
            column * tile_size : (column + 1) * tile_size,
        ] = np.asarray(tile)
    return canvas


def save_figure(raw, rows, output_file):
    factors = [row["factor"] for row in rows]
    display_scale = float(np.percentile(raw, 99.9))
    figure = plt.figure(figsize=(15, 4.2 * len(rows)))
    grid = figure.add_gridspec(len(rows), 2, width_ratios=(1.15, 1.0))

    for index, factor in enumerate(factors):
        axis = figure.add_subplot(grid[index, 0])
        axis.imshow(
            channel_montage(raw * factor, display_scale),
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axis.set_title(f"All 36 channels, illumination {factor:g}x")
        axis.axis("off")

    right = grid[:, 1].subgridspec(3, 1)
    level_labels = ("1/3", "2/3", "1")
    for level, label in enumerate(level_labels):
        plt.subplot(right[0]).plot(
            factors,
            [row["level_means"][level] for row in rows],
            marker="o",
            label=label,
        )
        plt.subplot(right[1]).plot(
            factors,
            [row["level_cvs"][level] for row in rows],
            marker="o",
            label=label,
        )
        plt.subplot(right[2]).plot(
            factors,
            [row["level_ratios"][level] for row in rows],
            marker="o",
            label=label,
        )

    axes = [plt.subplot(right[index]) for index in range(3)]
    axes[0].set_title("Raw effective brightness scales with illumination")
    axes[0].set_ylabel("raw level response")
    axes[1].set_title("Cross-channel CV remains unchanged")
    axes[1].set_ylabel("CV")
    axes[2].set_title("Relative gray ratios remain unchanged")
    axes[2].set_ylabel("level / level 1")
    axes[2].set_xlabel("global illumination factor")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(
        "Global illumination test: one shared multiplier, fixed display exposure",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output_file, dpi=170, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(
        description="用单一全局倍率验证入射光强变化，不进行逐channel缩放。"
    )
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factors", nargs="+", type=float, default=(1.0, 2.0, 4.0))
    return parser.parse_args()


def main():
    args = parse_args()
    if any(factor <= 0 for factor in args.factors):
        raise ValueError("所有光强倍率必须大于0。")
    results = np.load(args.results_file)
    raw = results["optimized_raw"]
    targets = results["targets"]
    rows = [illumination_metrics(raw, targets, factor) for factor in args.factors]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics_file = args.output_dir / "global_illumination_metrics.csv"
    figure_file = args.output_dir / "global_illumination_test.png"
    write_metrics(rows, metrics_file)
    save_figure(raw, rows, figure_file)
    print(metrics_file)
    print(figure_file)


if __name__ == "__main__":
    main()
