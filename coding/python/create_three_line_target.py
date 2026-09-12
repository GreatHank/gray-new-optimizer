import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import scipy.io as sio

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHANNEL_COUNT = 36
IMAGE_SIZE = 500
LINE_LENGTH = 200
LINE_WIDTH = 5
LINE_SPACING = 60
LEVELS = (1 / 3, 2 / 3, 1.0)


def build_targets():
    targets = np.zeros(
        (CHANNEL_COUNT, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
    )
    center = IMAGE_SIZE // 2
    column_start = center - LINE_LENGTH // 2
    column_end = column_start + LINE_LENGTH
    row_start = [
        center - LINE_SPACING - LINE_WIDTH // 2,
        center - LINE_WIDTH // 2,
        center + LINE_SPACING - LINE_WIDTH // 2,
    ]

    for level, start in zip(LEVELS, row_start):
        targets[:, start : start + LINE_WIDTH, column_start:column_end] = level

    return targets


def write_pixel_counts(targets, output_file):
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", "level", "pixel_count"])
        for channel in range(targets.shape[0]):
            for level in (0.0, *LEVELS):
                pixel_count = int(
                    np.count_nonzero(np.isclose(targets[channel], level))
                )
                writer.writerow([channel + 1, level, pixel_count])


def save_preview(targets, output_file):
    figure, axis = plt.subplots(figsize=(8, 8))
    image = axis.imshow(targets[0], cmap="gray", vmin=0, vmax=1)
    axis.set_title("Three-Line Four-Level Calibration Target")
    axis.set_xlabel("1/3, 2/3, 1 from top to bottom")
    axis.axis("off")
    figure.colorbar(image, ax=axis, ticks=[0, 1 / 3, 2 / 3, 1])
    figure.tight_layout()
    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成36通道三线四标签灰度校准目标。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "three_line_calibration",
        help="输出目录，必须尚不存在。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    targets = build_targets()
    mat_file = args.output_dir / "three_line_target.mat"
    preview_file = args.output_dir / "three_line_target_preview.png"
    counts_file = args.output_dir / "three_line_target_pixel_counts.csv"

    sio.savemat(mat_file, {"bw_all": targets}, do_compression=True)
    save_preview(targets, preview_file)
    write_pixel_counts(targets, counts_file)

    print(f"目标数组: {targets.shape}, labels: 0, 1/3, 2/3, 1")
    print(f"MAT 文件: {mat_file}")
    print(f"预览图: {preview_file}")
    print(f"像素检查: {counts_file}")


if __name__ == "__main__":
    main()
