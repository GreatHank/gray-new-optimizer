import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from order_decoupling_grayscale import (
    EVALUATION_CHANNEL_HEADERS,
    EVALUATION_SUMMARY_HEADERS,
    evaluation_metrics,
    print_evaluation_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="重新评价已有 optimized_results.npz，不执行优化。"
    )
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在: {args.output_dir}")

    results = np.load(args.results_file)
    required = {"optimized_raw", "targets", "pairMat"}
    missing = sorted(required.difference(results.files))
    if missing:
        raise KeyError(f"结果文件缺少字段: {', '.join(missing)}")

    raw = results["optimized_raw"]
    targets = results["targets"]
    pair_mat = results["pairMat"].astype(int)
    if pair_mat.shape != (targets.shape[0], 2):
        raise ValueError(
            f"pairMat形状{pair_mat.shape}与通道数{targets.shape[0]}不一致"
        )
    desired_gray_ratios = (
        results["desired_gray_ratios"]
        if "desired_gray_ratios" in results.files
        else (1 / 3, 2 / 3, 1.0)
    )
    channel_rows, summary = evaluation_metrics(
        raw, targets, desired_gray_ratios=desired_gray_ratios
    )

    args.output_dir.mkdir(parents=True)
    np.savetxt(
        args.output_dir / "evaluation_channel_metrics.csv",
        channel_rows,
        delimiter=",",
        header=",".join(EVALUATION_CHANNEL_HEADERS),
        comments="",
    )
    np.savetxt(
        args.output_dir / "evaluation_summary.csv",
        np.asarray([[summary[name] for name in EVALUATION_SUMMARY_HEADERS]]),
        delimiter=",",
        header=",".join(EVALUATION_SUMMARY_HEADERS),
        comments="",
    )

    cnr_index = EVALUATION_CHANNEL_HEADERS.index("foreground_background_cnr")
    snr_index = EVALUATION_CHANNEL_HEADERS.index("foreground_background_snr_db")
    order_rows = np.column_stack(
        [channel_rows[:, 0], pair_mat, channel_rows[:, cnr_index], channel_rows[:, snr_index]]
    )
    np.savetxt(
        args.output_dir / "snr_channel_metrics.csv",
        order_rows,
        delimiter=",",
        header="channel,m,n,foreground_background_cnr,foreground_background_snr_db",
        comments="",
    )

    labels = [f"({m},{n})" for m, n in pair_mat]
    snr_values = channel_rows[:, snr_index]
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(np.arange(len(snr_values)), snr_values, color="#3478b9")
    axis.axhline(np.mean(snr_values), color="#c43c39", linestyle="--", label="mean")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axis.set_ylabel("Paper-definition SNR (dB)")
    axis.set_title("Per-channel SNR = 20 log10(mean foreground / std background)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "snr_by_channel.png", dpi=180)
    plt.close(figure)

    print_evaluation_summary(summary, targets.shape[0])
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
