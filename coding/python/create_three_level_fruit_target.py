from pathlib import Path

import argparse
import matplotlib
import numpy as np
import scipy.io as sio
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEVELS = np.array([1 / 3, 2 / 3, 1.0], dtype=np.float32)


def load_binary_targets(mat_file):
    data = sio.loadmat(mat_file)
    if "bw_all" not in data:
        raise KeyError(f"{mat_file} 中不存在变量 bw_all。")
    targets = np.asarray(data["bw_all"])
    if targets.shape != (23, 500, 500):
        raise ValueError(f"bw_all 必须是 (23, 500, 500)，实际为 {targets.shape}。")
    if not np.all(np.isin(targets, [0, 1])):
        raise ValueError("输入水果目标必须是二值数组。")
    return targets.astype(np.uint8, copy=False)


def create_three_level_targets(binary_targets):
    targets = np.zeros(binary_targets.shape, dtype=np.float32)
    counts = []
    for channel, binary in enumerate(binary_targets):
        filled = ndimage.binary_fill_holes(binary)
        rows, columns = np.nonzero(filled)
        if len(rows) < 3:
            raise ValueError(f"通道 {channel + 1} 的前景像素少于 3，无法分成三档。")
        distance = ndimage.distance_transform_edt(filled)
        values = distance[filled]
        thresholds = np.quantile(values, [1 / 3, 2 / 3])
        targets[channel][filled & (distance <= thresholds[0])] = LEVELS[0]
        targets[channel][
            filled & (distance > thresholds[0]) & (distance <= thresholds[1])
        ] = LEVELS[1]
        targets[channel][filled & (distance > thresholds[1])] = LEVELS[2]
        counts.append(
            [
                channel + 1,
                int(filled.sum()),
                *(int(np.count_nonzero(targets[channel] == level)) for level in LEVELS),
                int(rows.min()),
                int(rows.max()),
                int(columns.min()),
                int(columns.max()),
            ]
        )
    return targets, np.asarray(counts, dtype=np.int64)


def save_preview(targets, output_file):
    columns = 5
    rows = int(np.ceil(targets.shape[0] / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, rows * 3), squeeze=False)
    for channel in range(targets.shape[0]):
        axis = axes[channel // columns, channel % columns]
        axis.imshow(targets[channel], cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Ch {channel + 1}")
        axis.axis("off")
    for axis in axes.flat[targets.shape[0] :]:
        axis.axis("off")
    figure.suptitle("Filled Three-Level Fruit Targets: Outer, Middle, Core")
    figure.tight_layout()
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="从二值水果线稿生成填充三灰度目标")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=project_root / "input" / "grayscale_image.mat",
        help="包含二值 bw_all 的 MAT 文件",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output" / "fruit_three_level_filled",
        help="填充三灰度目标输出目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_targets = load_binary_targets(input_file)
    targets, counts = create_three_level_targets(binary_targets)
    mat_file = output_dir / "fruit_three_level_target.mat"
    sio.savemat(mat_file, {"bw_all": targets})
    preview_file = output_dir / "fruit_three_level_target_preview.png"
    save_preview(targets, preview_file)
    counts_file = output_dir / "fruit_three_level_pixel_counts.csv"
    np.savetxt(
        counts_file,
        counts,
        delimiter=",",
        fmt="%d",
        header="channel,foreground_pixels,count_1_3,count_2_3,count_1,row_min,row_max,column_min,column_max",
        comments="",
    )
    print(f"目标形状: {targets.shape}, labels: 0, 1/3, 2/3, 1")
    print(f"MAT 文件: {mat_file}")
    print(f"预览图: {preview_file}")
    print(f"像素统计: {counts_file}")


if __name__ == "__main__":
    main()
