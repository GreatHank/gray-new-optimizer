import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = ROOT / "output/easy_picture_3x3_final_report/final_channel_metrics.csv"
OUTPUT_FILE = ROOT / "output/easy_picture_3x3_final_report/evaluation_histogram.png"


def values(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def main():
    with CSV_FILE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"没有找到指标数据: {CSV_FILE}")

    metrics = [
        ("Structure cosine", "structure_cosine", "#2f6f9f"),
        ("Foreground coverage", "foreground_coverage_above_background_p95", "#3b8c6e"),
        ("Gray ratio RMSE", "grayscale_ratio_rmse", "#c58a2b"),
        ("Background CV", "background_cv", "#9b4d5c"),
        ("Background P95 / mean", "background_p95_ratio", "#7657a6"),
        ("CNR", "foreground_background_cnr", "#2d8c9e"),
        ("Paper SNR (dB)", "foreground_background_snr_db", "#c9573d"),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for axis, (title, field, color) in zip(axes, metrics):
        data = values(rows, field)
        bins = min(8, max(4, len(data) // 2))
        axis.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.9)
        axis.axvline(np.mean(data), color="#333333", linestyle="--", linewidth=1, label=f"mean {np.mean(data):.3g}")
        if field == "foreground_background_snr_db":
            axis.axvline(15.0, color="#b21f2d", linewidth=1.5, label="15 dB threshold")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)

    snr = values(rows, "foreground_background_snr_db")
    labels = [
        f"Ch{int(float(row['channel']))}\n({int(float(row['m']))},{int(float(row['n']))})"
        for row in rows
    ]
    axis = axes[-1]
    axis.clear()
    axis.bar(np.arange(len(snr)), snr, color="#c9573d")
    axis.axhline(15.0, color="#b21f2d", linestyle="--", linewidth=1.5, label="15 dB threshold")
    axis.set_xticks(np.arange(len(snr)), labels, fontsize=8, rotation=45, ha="right")
    axis.set_ylabel("dB")
    axis.set_title("Per-channel paper SNR")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    figure.suptitle("easy_picture 3x3 final evaluation", fontsize=16)
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
