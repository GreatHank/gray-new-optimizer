import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image


SOURCE_LEVELS = np.array([0, 85, 170, 255], dtype=np.uint8)
TARGET_LEVELS = np.array([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float32)


def build_targets(image, grid_size=6, tile_size=500):
    if image.ndim != 2:
        raise ValueError(f"输入必须是单通道灰度图，实际形状为 {image.shape}。")
    if not np.all(np.isin(image, SOURCE_LEVELS)):
        raise ValueError("输入图像必须严格只包含 0、85、170、255。")

    row_edges = np.linspace(0, image.shape[0], grid_size + 1, dtype=int)
    column_edges = np.linspace(0, image.shape[1], grid_size + 1, dtype=int)
    targets = []
    positions = []
    source_bounds = []
    for row in range(grid_size):
        for column in range(grid_size):
            bounds = (
                int(row_edges[row]),
                int(row_edges[row + 1]),
                int(column_edges[column]),
                int(column_edges[column + 1]),
            )
            row0, row1, column0, column1 = bounds
            source_tile = image[row0:row1, column0:column1]
            resized = np.asarray(
                Image.fromarray(source_tile).resize(
                    (tile_size, tile_size), resample=Image.Resampling.NEAREST
                )
            )
            target = np.empty(resized.shape, dtype=np.float32)
            for source_level, target_level in zip(SOURCE_LEVELS, TARGET_LEVELS):
                target[resized == source_level] = target_level
            targets.append(target)
            positions.append((row, column))
            source_bounds.append(bounds)
    return (
        np.stack(targets),
        np.asarray(positions, dtype=np.int16),
        source_bounds,
    )


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="将固定四灰度圆形场景整幅等分为完整6x6目标。"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=project_root
        / "output"
        / "circular_7x7_target_49"
        / "circular_scene_3500.png",
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
    targets, positions, source_bounds = build_targets(
        source, grid_size=6, tile_size=args.tile_size
    )
    mat_file = args.output_dir / "circular_6x6_target.mat"
    sio.savemat(
        mat_file,
        {"bw_all": targets, "grid_positions": positions},
        do_compression=True,
    )

    with (args.output_dir / "tile_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "channel",
                "grid_row",
                "grid_column",
                "source_row0",
                "source_row1",
                "source_column0",
                "source_column1",
                "count_0",
                "count_85",
                "count_170",
                "count_255",
            ]
        )
        for channel, (target, position, bounds) in enumerate(
            zip(targets, positions, source_bounds), start=1
        ):
            counts = [
                int(np.count_nonzero(np.isclose(target, level)))
                for level in TARGET_LEVELS
            ]
            writer.writerow([channel, *(position + 1), *bounds, *counts])

    unique_values = np.unique(targets)
    if not np.array_equal(unique_values, TARGET_LEVELS):
        raise RuntimeError(f"输出标签异常: {unique_values}")
    print(f"目标数组: {targets.shape}")
    print(f"grid_positions: {positions.tolist()}")
    print(f"标签: {unique_values.tolist()}")
    print(f"MAT 文件: {mat_file}")


if __name__ == "__main__":
    main()
