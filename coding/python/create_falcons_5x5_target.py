import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image


GRID_SIZE = 5
SOURCE_LEVELS = np.array([0, 85, 170, 255], dtype=np.uint8)
TARGET_LEVELS = np.array([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float32)


def build_targets(image, tile_size=500, grid_size=GRID_SIZE, transform="identity"):
    if image.ndim != 2:
        raise ValueError(f"输入必须是单通道灰度图，实际形状为 {image.shape}。")
    if not np.all(np.isin(image, SOURCE_LEVELS)):
        raise ValueError("输入图像必须严格只包含 0、85、170、255。")

    if grid_size < 1:
        raise ValueError("grid-size 必须大于0。")
    if transform == "mirror_lr":
        image = np.fliplr(image)
    elif transform != "identity":
        raise ValueError(f"不支持的整体映射: {transform}")
    row_edges = np.linspace(0, image.shape[0], grid_size + 1, dtype=int)
    column_edges = np.linspace(0, image.shape[1], grid_size + 1, dtype=int)
    targets = []
    positions = []
    for row in range(grid_size):
        for column in range(grid_size):
            tile = image[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ]
            resized = np.asarray(
                Image.fromarray(tile).resize(
                    (tile_size, tile_size), resample=Image.Resampling.NEAREST
                )
            )
            target = np.empty(resized.shape, dtype=np.float32)
            for source_level, target_level in zip(SOURCE_LEVELS, TARGET_LEVELS):
                target[resized == source_level] = target_level
            targets.append(target)
            positions.append((row, column))
    return np.stack(targets), np.asarray(positions, dtype=np.int16)


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="保持原图方向，将固定四灰度falcons图切成连续5x5目标。"
    )
    parser.add_argument(
        "--input-file", type=Path, default=project_root / "input" / "falcons.png"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument(
        "--transform", choices=("identity", "mirror_lr"), default="identity"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tile_size < 1:
        raise ValueError("tile-size 必须大于0。")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    source = np.asarray(Image.open(args.input_file).convert("L"))
    targets, positions = build_targets(
        source, args.tile_size, grid_size=args.grid_size, transform=args.transform
    )
    mat_file = args.output_dir / f"falcons_{args.grid_size}x{args.grid_size}_target.mat"
    sio.savemat(
        mat_file,
        {
            "bw_all": targets,
            "grid_positions": positions,
            "source_grid_positions_1based": positions + 1,
            "mapping_transform": args.transform,
        },
        do_compression=True,
    )

    with (args.output_dir / "tile_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "channel",
                "source_row",
                "source_column",
                "count_0",
                "count_85",
                "count_170",
                "count_255",
            ]
        )
        for channel, (target, (row, column)) in enumerate(
            zip(targets, positions), start=1
        ):
            counts = [
                int(np.count_nonzero(np.isclose(target, level)))
                for level in TARGET_LEVELS
            ]
            writer.writerow([channel, row + 1, column + 1, *counts])

    unique_values = np.unique(targets)
    if not np.array_equal(unique_values, TARGET_LEVELS):
        raise RuntimeError(f"输出标签异常: {unique_values}")
    print(f"目标数组: {targets.shape}")
    print(f"整体映射: {args.transform}")
    print(f"标签: {unique_values.tolist()}")
    print(f"MAT 文件: {mat_file}")


if __name__ == "__main__":
    main()
