import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import scipy.io as sio
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRID_SIZE = 6
CHANNEL_COUNT = 36
IMAGE_SIZE = 500
LEVELS = np.array([1 / 3, 2 / 3, 1.0], dtype=np.float32)
BACKGROUND_DELTA = 50.0
MIN_LINE_INTENSITY = 96.0


def foreground_mask(
    gray,
    background_delta=BACKGROUND_DELTA,
    min_line_intensity=MIN_LINE_INTENSITY,
):
    background = float(np.percentile(gray, 50))
    threshold = max(background + background_delta, min_line_intensity)
    return gray >= threshold


def content_bounds(mask, padding=12):
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("输入图像中没有检测到可用线条。")
    return (
        max(0, int(rows.min()) - padding),
        min(mask.shape[0], int(rows.max()) + padding + 1),
        max(0, int(columns.min()) - padding),
        min(mask.shape[1], int(columns.max()) + padding + 1),
    )


def split_grid(
    gray,
    grid_size=GRID_SIZE,
    background_delta=BACKGROUND_DELTA,
    min_line_intensity=MIN_LINE_INTENSITY,
):
    mask = foreground_mask(gray, background_delta, min_line_intensity)
    row_min, row_max, column_min, column_max = content_bounds(mask)
    row_edges = np.linspace(row_min, row_max, grid_size + 1, dtype=int)
    column_edges = np.linspace(column_min, column_max, grid_size + 1, dtype=int)
    tiles = []
    for grid_row in range(grid_size):
        for grid_column in range(grid_size):
            bounds = (
                int(row_edges[grid_row]),
                int(row_edges[grid_row + 1]),
                int(column_edges[grid_column]),
                int(column_edges[grid_column + 1]),
            )
            row0, row1, column0, column1 = bounds
            tile = gray[row0:row1, column0:column1]
            score = int(
                np.count_nonzero(
                    foreground_mask(tile, background_delta, min_line_intensity)
                )
            )
            tiles.append((grid_row, grid_column, bounds, tile, score))
    return tiles


def quantize_tile(
    tile,
    image_size=IMAGE_SIZE,
    background_delta=BACKGROUND_DELTA,
    min_line_intensity=MIN_LINE_INTENSITY,
    level_assignment="source",
):
    # The complete grid cell is the channel image. Resize it directly to the
    # physical 500 x 500 plane; do not center it on a larger black canvas.
    resized = np.asarray(
        Image.fromarray(tile).resize(
            (image_size, image_size), resample=Image.Resampling.LANCZOS
        )
    )
    mask = foreground_mask(resized, background_delta, min_line_intensity)
    values = resized[mask]
    if values.size < 30:
        raise ValueError("候选区域中的线条像素过少。")

    quantized = np.zeros(resized.shape, dtype=np.float32)
    if level_assignment == "source":
        lower, upper = np.quantile(values, [1 / 3, 2 / 3])
        quantized[mask & (resized <= lower)] = LEVELS[0]
        quantized[mask & (resized > lower) & (resized <= upper)] = LEVELS[1]
        quantized[mask & (resized > upper)] = LEVELS[2]
    elif level_assignment == "spatial":
        rows, columns = np.nonzero(mask)
        spatial_key = rows / max(1, resized.shape[0] - 1) + columns / max(
            1, resized.shape[1] - 1
        )
        order = np.argsort(spatial_key, kind="stable")
        groups = np.array_split(order, 3)
        for level, group in zip(LEVELS, groups):
            quantized[rows[group], columns[group]] = level
    else:
        raise ValueError(f"未知灰度分配方式: {level_assignment}")
    return quantized


def build_targets(
    gray,
    channel_count=CHANNEL_COUNT,
    background_delta=BACKGROUND_DELTA,
    min_line_intensity=MIN_LINE_INTENSITY,
    level_assignment="source",
):
    tiles = split_grid(
        gray,
        background_delta=background_delta,
        min_line_intensity=min_line_intensity,
    )
    selected = sorted(tiles, key=lambda item: (-item[4], item[0], item[1]))[:channel_count]
    # Channel order follows the original image from top-left to bottom-right.
    selected.sort(key=lambda item: (item[0], item[1]))
    targets = np.stack(
        [
            quantize_tile(
                item[3],
                background_delta=background_delta,
                min_line_intensity=min_line_intensity,
                level_assignment=level_assignment,
            )
            for item in selected
        ]
    )
    return targets, tiles, selected


def save_grid_preview(tiles, selected, output_file):
    selected_cells = {(item[0], item[1]) for item in selected}
    figure, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(13, 13), squeeze=False)
    for grid_row, grid_column, _bounds, tile, score in tiles:
        axis = axes[grid_row, grid_column]
        axis.imshow(tile, cmap="gray", vmin=0, vmax=255)
        state = "selected" if (grid_row, grid_column) in selected_cells else "unused"
        axis.set_title(f"r{grid_row + 1}c{grid_column + 1} {state}\nline px={score}", fontsize=8)
        axis.axis("off")
    figure.suptitle("6 x 6 source crops; all 36 used as physical channels")
    figure.tight_layout()
    figure.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_target_preview(targets, output_file):
    columns = 5
    rows = int(np.ceil(targets.shape[0] / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3 * rows), squeeze=False)
    for channel, target in enumerate(targets):
        axis = axes.flat[channel]
        axis.imshow(target, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"Ch {channel + 1}")
        axis.axis("off")
    for axis in axes.flat[targets.shape[0] :]:
        axis.axis("off")
    figure.suptitle("Thin-line three-level channel targets")
    figure.tight_layout()
    figure.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_statistics(targets, selected, output_file):
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "channel", "grid_row", "grid_column", "source_row0", "source_row1",
            "source_column0", "source_column1", "source_line_pixels",
            "count_1_3", "count_2_3", "count_1",
        ])
        for channel, (target, item) in enumerate(zip(targets, selected), start=1):
            grid_row, grid_column, bounds, _tile, score = item
            writer.writerow([
                channel, grid_row + 1, grid_column + 1, *bounds, score,
                *(int(np.count_nonzero(np.isclose(target, level))) for level in LEVELS),
            ])


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="将细线字母图按6x6网格裁剪为36通道三灰度目标。")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=project_root / "input" / "thin_letter_reference.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output" / "thin_letter_6x6_target",
    )
    parser.add_argument(
        "--background-delta",
        type=float,
        default=BACKGROUND_DELTA,
        help="线条亮度至少高于图像中位背景的差值。",
    )
    parser.add_argument(
        "--min-line-intensity",
        type=float,
        default=MIN_LINE_INTENSITY,
        help="线条绝对灰度下限，范围0到255。",
    )
    parser.add_argument(
        "--level-assignment",
        choices=("source", "spatial"),
        default="source",
        help="source按原图亮度分级；spatial沿线条空间位置连续分成三档。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gray = np.asarray(Image.open(args.input_file).convert("L"))
    if args.background_delta < 0:
        raise ValueError("background-delta不能小于0。")
    if not 0 <= args.min_line_intensity <= 255:
        raise ValueError("min-line-intensity必须在0到255之间。")
    targets, tiles, selected = build_targets(
        gray,
        background_delta=args.background_delta,
        min_line_intensity=args.min_line_intensity,
        level_assignment=args.level_assignment,
    )

    mat_file = args.output_dir / "thin_letter_6x6_target.mat"
    sio.savemat(mat_file, {"bw_all": targets}, do_compression=True)
    save_grid_preview(tiles, selected, args.output_dir / "source_grid_selection.png")
    save_target_preview(targets, args.output_dir / "thin_letter_6x6_target_preview.png")
    write_statistics(targets, selected, args.output_dir / "thin_letter_6x6_target_stats.csv")

    print(f"目标数组: {targets.shape}, labels: 0, 1/3, 2/3, 1")
    print(f"MAT 文件: {mat_file}")
    print(f"候选网格: {args.output_dir / 'source_grid_selection.png'}")
    print(f"目标预览: {args.output_dir / 'thin_letter_6x6_target_preview.png'}")


if __name__ == "__main__":
    main()
