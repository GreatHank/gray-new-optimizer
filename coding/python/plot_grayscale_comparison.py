import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEVEL_KEYS = ("S_1_3", "S_2_3", "S_1")
LEVEL_LABELS = ("1/3", "2/3", "1")


def load_summary(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def load_channels(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def plot_comparison(before_dir, after_dir, output_file):
    before_summary = load_summary(before_dir / "evaluation_summary.csv")
    after_summary = load_summary(after_dir / "evaluation_summary.csv")
    before_channels = load_channels(before_dir / "evaluation_channel_metrics.csv")
    after_channels = load_channels(after_dir / "evaluation_channel_metrics.csv")

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(3)
    width = 0.36

    before_cv = [float(before_summary[f"{key}_cv"]) for key in LEVEL_KEYS]
    after_cv = [float(after_summary[f"{key}_cv"]) for key in LEVEL_KEYS]
    axes[0, 0].bar(x - width / 2, before_cv, width, label="before")
    axes[0, 0].bar(x + width / 2, after_cv, width, label="after")
    axes[0, 0].set_xticks(x, LEVEL_LABELS)
    axes[0, 0].set_title("Cross-channel CV by gray level (lower is better)")
    axes[0, 0].set_ylabel("CV")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.25)

    before_means = [float(before_summary[f"{key}_mean"]) for key in LEVEL_KEYS]
    after_means = [float(after_summary[f"{key}_mean"]) for key in LEVEL_KEYS]
    axes[0, 1].bar(x - width / 2, before_means, width, label="before")
    axes[0, 1].bar(x + width / 2, after_means, width, label="after")
    axes[0, 1].set_xticks(x, LEVEL_LABELS)
    axes[0, 1].set_title("Mean effective brightness by gray level")
    axes[0, 1].set_ylabel("response / plane mean")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.25)

    channel_x = np.arange(1, len(np.atleast_1d(before_channels)) + 1)
    colors = ("#4c78a8", "#f58518", "#54a24b")
    for key, label, color in zip(LEVEL_KEYS, LEVEL_LABELS, colors):
        axes[1, 0].plot(
            channel_x,
            np.atleast_1d(before_channels[key]),
            linestyle=":",
            color=color,
            alpha=0.65,
            label=f"{label} before",
        )
        axes[1, 0].plot(
            channel_x,
            np.atleast_1d(after_channels[key]),
            color=color,
            linewidth=2,
            label=f"{label} after",
        )
    axes[1, 0].set_title("Per-channel effective brightness")
    axes[1, 0].set_xlabel("channel")
    axes[1, 0].set_ylabel("response / plane mean")
    axes[1, 0].legend(ncol=2, fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    metric_labels = ("structure", "coverage", "ratio RMSE")
    before_metrics = (
        float(before_summary["structure_cosine_mean"]),
        float(before_summary["foreground_coverage_mean"]),
        float(before_summary["grayscale_ratio_rmse_mean"]),
    )
    after_metrics = (
        float(after_summary["structure_cosine_mean"]),
        float(after_summary["foreground_coverage_mean"]),
        float(after_summary["grayscale_ratio_rmse_mean"]),
    )
    metric_x = np.arange(len(metric_labels))
    axes[1, 1].bar(metric_x - width / 2, before_metrics, width, label="before")
    axes[1, 1].bar(metric_x + width / 2, after_metrics, width, label="after")
    axes[1, 1].set_xticks(metric_x, metric_labels)
    axes[1, 1].set_title("Structure and gray-ratio checks")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.suptitle("Thin-line grayscale loss: before vs final candidate")
    figure.tight_layout()
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare grayscale metrics between two optimization outputs.")
    parser.add_argument("--before-dir", type=Path, required=True)
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(args.before_dir, args.after_dir, args.output_file)
    print(args.output_file)


if __name__ == "__main__":
    main()
