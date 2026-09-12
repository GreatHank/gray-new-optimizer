import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEVELS = np.asarray([0, 85, 170, 255], dtype=np.uint8)


def quantize_four_levels(gray):
    gray = np.asarray(gray, dtype=np.float32)
    distances = np.abs(gray[..., None] - LEVELS.astype(np.float32))
    return LEVELS[np.argmin(distances, axis=-1)]


def split_grid(gray, grid_size):
    row_edges = np.linspace(0, gray.shape[0], grid_size + 1, dtype=int)
    column_edges = np.linspace(0, gray.shape[1], grid_size + 1, dtype=int)
    return [
        (
            row + 1,
            column + 1,
            gray[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ],
        )
        for row in range(grid_size)
        for column in range(grid_size)
    ]


def coefficient_of_variation(values, epsilon=1e-12):
    values = np.asarray(values, dtype=np.float64)
    return float(np.std(values) / (abs(np.mean(values)) + epsilon))


def channel_metrics(tile):
    counts = np.asarray([np.count_nonzero(tile == level) for level in LEVELS])
    pixel_count = int(tile.size)
    foreground_count = int(pixel_count - counts[0])
    level_fractions = counts / pixel_count
    normalized = tile.astype(np.float64) / 255.0
    channel_target_mean = float(np.mean(normalized))

    if foreground_count:
        foreground_fractions = counts[1:] / foreground_count
        foreground_gray_mean = float(np.mean(normalized[tile > 0]))
        nonzero = foreground_fractions > 0
        gray_entropy = float(
            -np.sum(foreground_fractions[nonzero] * np.log(foreground_fractions[nonzero]))
            / np.log(3.0)
        )
        present_gray_levels = int(np.count_nonzero(counts[1:]))
    else:
        foreground_gray_mean = np.nan
        gray_entropy = 0.0
        present_gray_levels = 0

    foreground = tile > 0
    horizontal = np.count_nonzero(foreground[:, 1:] != foreground[:, :-1])
    vertical = np.count_nonzero(foreground[1:, :] != foreground[:-1, :])
    neighbor_pairs = foreground.shape[0] * max(0, foreground.shape[1] - 1)
    neighbor_pairs += max(0, foreground.shape[0] - 1) * foreground.shape[1]
    boundary_density = float((horizontal + vertical) / max(1, neighbor_pairs))

    return {
        "pixel_count": pixel_count,
        "active_fraction": float(foreground_count / pixel_count),
        "foreground_gray_mean": foreground_gray_mean,
        "channel_target_mean": channel_target_mean,
        "present_gray_levels": present_gray_levels,
        "gray_entropy": gray_entropy,
        "boundary_density": boundary_density,
        "fraction_0": float(level_fractions[0]),
        "fraction_85": float(level_fractions[1]),
        "fraction_170": float(level_fractions[2]),
        "fraction_255": float(level_fractions[3]),
    }


def evaluate_image(gray, grid_size=6):
    rows = []
    for channel, (grid_row, grid_column, tile) in enumerate(
        split_grid(gray, grid_size), start=1
    ):
        rows.append(
            {
                "channel": channel,
                "grid_row": grid_row,
                "grid_column": grid_column,
                **channel_metrics(tile),
            }
        )

    target_means = np.asarray([row["channel_target_mean"] for row in rows])
    active_fractions = np.asarray([row["active_fraction"] for row in rows])
    boundary_densities = np.asarray([row["boundary_density"] for row in rows])
    foreground_means = np.asarray([row["foreground_gray_mean"] for row in rows])
    valid_foreground_means = foreground_means[np.isfinite(foreground_means)]

    summary = {
        "channel_count": len(rows),
        "black_channel_count": int(np.count_nonzero(active_fractions == 0)),
        "all_three_gray_levels_count": int(
            sum(row["present_gray_levels"] == 3 for row in rows)
        ),
        "channel_target_mean_mean": float(np.mean(target_means)),
        "channel_target_mean_min": float(np.min(target_means)),
        "channel_target_mean_max": float(np.max(target_means)),
        "channel_target_mean_cv": coefficient_of_variation(target_means),
        "active_fraction_cv": coefficient_of_variation(active_fractions),
        "foreground_gray_mean_cv": (
            coefficient_of_variation(valid_foreground_means)
            if valid_foreground_means.size
            else np.nan
        ),
        "boundary_density_cv": coefficient_of_variation(boundary_densities),
        "gray_entropy_mean": float(np.mean([row["gray_entropy"] for row in rows])),
        "gray_entropy_min": float(np.min([row["gray_entropy"] for row in rows])),
    }
    return rows, summary


def write_csv(rows, summary, output_dir):
    with (output_dir / "source_channel_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "source_channel_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())


def save_preview(gray, rows, grid_size, output_file):
    figure, axes = plt.subplots(grid_size, grid_size, figsize=(13, 13), squeeze=False)
    tiles = split_grid(gray, grid_size)
    for row, (_grid_row, _grid_column, tile), axis in zip(rows, tiles, axes.flat):
        axis.imshow(tile, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(
            f"Ch {row['channel']}  B={row['channel_target_mean']:.3f}\n"
            f"line={row['active_fraction']:.3f}  H={row['gray_entropy']:.2f}",
            fontsize=8,
        )
        axis.axis("off")
    figure.suptitle("Source-channel suitability (B includes the black background)")
    figure.tight_layout()
    figure.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="评价严格四灰度源图在规则网格中的逐channel亮度、灰阶信息和线条复杂度。"
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument(
        "--quantize-output",
        type=Path,
        help="可选：把输入按最近值量化为0/85/170/255并保存到该路径。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.grid_size < 1:
        raise ValueError("grid-size必须大于0。")
    source = np.asarray(Image.open(args.input_file).convert("L"))
    gray = quantize_four_levels(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.quantize_output is not None:
        args.quantize_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(gray).save(args.quantize_output)

    rows, summary = evaluate_image(gray, args.grid_size)
    write_csv(rows, summary, args.output_dir)
    save_preview(
        gray, rows, args.grid_size, args.output_dir / "source_channel_suitability.png"
    )
    print(
        "channel总亮度 mean/min/max/CV="
        f"{summary['channel_target_mean_mean']:.4f}/"
        f"{summary['channel_target_mean_min']:.4f}/"
        f"{summary['channel_target_mean_max']:.4f}/"
        f"{summary['channel_target_mean_cv']:.4f}"
    )
    print(
        "前景灰阶均值CV/线条覆盖率CV/边界复杂度CV="
        f"{summary['foreground_gray_mean_cv']:.4f}/"
        f"{summary['active_fraction_cv']:.4f}/"
        f"{summary['boundary_density_cv']:.4f}"
    )
    print(
        "三档齐全channel/黑channel="
        f"{summary['all_three_gray_levels_count']}/{summary['channel_count']} / "
        f"{summary['black_channel_count']}"
    )


if __name__ == "__main__":
    main()
