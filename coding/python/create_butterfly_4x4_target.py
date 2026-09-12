import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image


GRID_SIZE = 4
SOURCE_LEVELS = np.array([0, 85, 170, 255], dtype=np.uint8)
TARGET_LEVELS = np.array([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float32)


def build_targets(image, tile_size=500):
    if image.ndim != 2:
        raise ValueError(f"输入必须是单通道灰度图，实际形状为 {image.shape}。")
    if not np.all(np.isin(image, SOURCE_LEVELS)):
        raise ValueError("输入图像必须严格只包含 0、85、170、255。")

    mapped_image = np.rot90(image, 2)
    row_edges = np.linspace(0, mapped_image.shape[0], GRID_SIZE + 1, dtype=int)
    column_edges = np.linspace(0, mapped_image.shape[1], GRID_SIZE + 1, dtype=int)
    targets = []
    positions = []
    source_positions = []
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            tile = mapped_image[
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
            source_positions.append((GRID_SIZE - row, GRID_SIZE - column))
    return (
        np.stack(targets),
        np.asarray(positions, dtype=np.int16),
        np.asarray(source_positions, dtype=np.int16),
    )


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="整图旋转180度后，将固定四灰度蝴蝶图切成连续4x4目标。"
    )
    parser.add_argument(
        "--input-file", type=Path, default=project_root / "input" / "butterfly.png"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tile_size < 1:
        raise ValueError("tile-size 必须大于0。")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    source = np.asarray(Image.open(args.input_file).convert("L"))
    targets, positions, source_positions = build_targets(source, args.tile_size)
    mat_file = args.output_dir / "butterfly_4x4_target.mat"
    sio.savemat(
        mat_file,
        {
            "bw_all": targets,
            "grid_positions": positions,
            "source_grid_positions_1based": source_positions,
            "mapping_transform": "rot180",
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
                "m",
                "n",
                "source_row",
                "source_column",
                "count_0",
                "count_85",
                "count_170",
                "count_255",
            ]
        )
        for channel, (target, (row, column), source_position) in enumerate(
            zip(targets, positions, source_positions), start=1
        ):
            counts = [
                int(np.count_nonzero(np.isclose(target, level)))
                for level in TARGET_LEVELS
            ]
            writer.writerow(
                [channel, row + 1, column + 1, *source_position, *counts]
            )

    unique_values = np.unique(targets)
    if not np.array_equal(unique_values, TARGET_LEVELS):
        raise RuntimeError(f"输出标签异常: {unique_values}")
    print(f"目标数组: {targets.shape}")
    print("整体映射: rot180")
    print("pairMat: [(m,n) for m in 1..4 for n in 1..4]")
    print(f"标签: {unique_values.tolist()}")
    print(f"MAT 文件: {mat_file}")


if __name__ == "__main__":
    main()
