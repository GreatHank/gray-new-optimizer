import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image


SOURCE_LEVELS = np.array([0, 85, 170, 255], dtype=np.uint8)
TARGET_LEVELS = np.array([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float32)


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="将已切分的7×7四灰度图缩小并生成圆形49级次目标。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "input" / "output_blocksv4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output" / "circular_7x7_target_49",
    )
    parser.add_argument("--tile-size", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tile_size < 1:
        raise ValueError("tile-size必须大于0。")
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{args.output_dir}")

    files = [args.input_dir / f"block_{index:02d}.png" for index in range(1, 50)]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少切片：{missing[0]}")

    grid_size = 7
    scene_size = grid_size * args.tile_size
    scene = np.zeros((scene_size, scene_size), dtype=np.uint8)
    for index, path in enumerate(files):
        with Image.open(path) as image:
            gray = image.convert("L")
            values = np.unique(np.asarray(gray))
            if not np.all(np.isin(values, SOURCE_LEVELS)):
                raise ValueError(f"{path.name}包含四级灰度之外的像素：{values.tolist()}")
            tile = np.asarray(
                gray.resize(
                    (args.tile_size, args.tile_size),
                    resample=Image.Resampling.NEAREST,
                )
            )
        row, column = divmod(index, grid_size)
        row0 = row * args.tile_size
        column0 = column * args.tile_size
        scene[row0 : row0 + args.tile_size, column0 : column0 + args.tile_size] = tile

    coordinates = np.arange(scene_size, dtype=np.float32) - (scene_size - 1) / 2
    circle = coordinates[:, None] ** 2 + coordinates[None, :] ** 2 <= (scene_size / 2) ** 2
    scene[~circle] = 0

    args.output_dir.mkdir(parents=True)
    tile_dir = args.output_dir / "tiles_500"
    tile_dir.mkdir()
    Image.fromarray(scene).save(args.output_dir / "circular_scene_3500.png")

    targets = []
    positions = []
    statistics = []
    for index in range(49):
        row, column = divmod(index, grid_size)
        row0 = row * args.tile_size
        column0 = column * args.tile_size
        tile = scene[row0 : row0 + args.tile_size, column0 : column0 + args.tile_size]
        Image.fromarray(tile).save(tile_dir / f"block_{index + 1:02d}.png")
        counts = [int(np.count_nonzero(tile == value)) for value in SOURCE_LEVELS]
        statistics.append([index + 1, row + 1, column + 1, *counts])
        if np.any(tile):
            target = np.zeros(tile.shape, dtype=np.float32)
            for source_level, target_level in zip(SOURCE_LEVELS[1:], TARGET_LEVELS[1:]):
                target[tile == source_level] = target_level
            targets.append(target)
            positions.append((row, column))

    targets = np.stack(targets)
    positions = np.asarray(positions, dtype=np.int16)
    sio.savemat(
        args.output_dir / "circular_7x7_target.mat",
        {
            "bw_all": targets,
            "grid_positions": positions,
            "grid_size": np.asarray([[grid_size]], dtype=np.int16),
        },
        do_compression=True,
    )
    with (args.output_dir / "tile_statistics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_channel", "grid_row", "grid_column", "count_0", "count_85", "count_170", "count_255"])
        writer.writerows(statistics)

    print(f"场景尺寸: {scene.shape}, 灰度值: {np.unique(scene).tolist()}")
    print(f"有效通道: {targets.shape[0]}/49, 目标尺寸: {targets.shape}")
    print(f"圆形总图: {args.output_dir / 'circular_scene_3500.png'}")
    print(f"目标MAT: {args.output_dir / 'circular_7x7_target.mat'}")


if __name__ == "__main__":
    main()
