import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def group_summary(rows):
    values = np.asarray(rows, dtype=np.float64)
    return {
        "channels": int(values.shape[0]),
        "structure_mean": float(np.mean(values[:, 0])),
        "structure_min": float(np.min(values[:, 0])),
        "coverage_mean": float(np.mean(values[:, 1])),
        "coverage_min": float(np.min(values[:, 1])),
        "brightness_mean": float(np.mean(values[:, 2])),
        "brightness_min": float(np.min(values[:, 2])),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="汇总n=0列并输出六块目标/重建对比。")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--illumination-factor", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.illumination_factor <= 0:
        raise ValueError("illumination-factor 必须大于0。")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    results = np.load(args.result_dir / "optimized_results.npz")
    pair_mat = results["pairMat"].astype(int)
    targets = results["targets"]
    raw = results["optimized_raw"].astype(np.float64)
    evaluation = np.loadtxt(
        args.result_dir / "evaluation_channel_metrics.csv",
        delimiter=",",
        skiprows=1,
    )
    brightness = np.loadtxt(
        args.result_dir / "channel_brightness_metrics.csv",
        delimiter=",",
        skiprows=1,
    )
    if pair_mat.shape != (36, 2):
        raise ValueError(f"必须是完整36级次，实际为 {pair_mat.shape}。")

    expected = np.asarray(
        [(m, n) for m in range(-6, 0) for n in range(-3, 3)], dtype=int
    )
    if not np.array_equal(pair_mat, expected):
        raise ValueError(f"pairMat 与指定级次不一致:\n{pair_mat}")

    rows = []
    for index, (m, n) in enumerate(pair_mat):
        rows.append(
            [
                index + 1,
                m,
                n,
                evaluation[index, 1],
                evaluation[index, 2],
                brightness[index, 6],
            ]
        )
    with (args.output_dir / "channel_metrics_with_orders.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["channel", "m", "n", "structure", "coverage", "foreground_brightness"]
        )
        writer.writerows(rows)

    n0_rows = [[row[3], row[4], row[5]] for row in rows if row[2] == 0]
    other_rows = [[row[3], row[4], row[5]] for row in rows if row[2] != 0]
    inner_rows = [
        [row[3], row[4], row[5]] for row in rows if row[2] in (-2, -1, 1)
    ]
    summaries = {
        "n=0": group_summary(n0_rows),
        "n!=0": group_summary(other_rows),
        "inner n=-2,-1,1": group_summary(inner_rows),
    }
    with (args.output_dir / "group_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = ["group", *next(iter(summaries.values())).keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, summary in summaries.items():
            writer.writerow({"group": group, **summary})

    n0_indices = np.flatnonzero(pair_mat[:, 1] == 0)
    display_scale = float(np.percentile(raw, 99.9))
    figure, axes = plt.subplots(6, 2, figsize=(8, 22), squeeze=False)
    for row_index, channel_index in enumerate(n0_indices):
        m, n = pair_mat[channel_index]
        axes[row_index, 0].imshow(
            targets[channel_index], cmap="gray", vmin=0, vmax=1, interpolation="nearest"
        )
        axes[row_index, 0].set_title(f"({m},{n}) target")
        axes[row_index, 1].imshow(
            np.clip(raw[channel_index] * args.illumination_factor / display_scale, 0, 1),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row_index, 1].set_title(
            f"({m},{n}) reconstruction, shared {args.illumination_factor:g}x"
        )
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle("n=0 integer-multiple column: targets vs reconstructions")
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    figure.savefig(args.output_dir / "n0_target_vs_reconstruction_2x.png", dpi=180)
    plt.close(figure)

    print(f"pairMat confirmed:\n{pair_mat}")
    print(f"n=0 summary: {summaries['n=0']}")
    print(f"n!=0 summary: {summaries['n!=0']}")
    print(f"shared_display_scale_1x={display_scale:.9g}")


if __name__ == "__main__":
    main()
