import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
REPORT = OUTPUT / "easy_picture_3x3_final_report"

STAGES = [
    ("A", "main structure", "easy_picture_3x3_stage_a_main_total_1000", 900, "1e-3", "easy_picture_3x3_stage_a_main_100"),
    ("B", "weak-channel structure", "easy_picture_3x3_stage_b_weak_structure_500", 500, "5e-4", "easy_picture_3x3_stage_a_main_total_1000"),
    ("C", "channel brightness CV", "easy_picture_3x3_stage_c_brightness_cv_300", 300, "2e-4", "easy_picture_3x3_stage_b_weak_structure_500"),
    ("D", "three-level CV", "easy_picture_3x3_stage_d_level_cv_300", 300, "1e-4", "easy_picture_3x3_stage_c_brightness_cv_300"),
    ("E", "background speckle and hotspots", "easy_picture_3x3_stage_e_background_300", 300, "1e-4", "easy_picture_3x3_stage_d_level_cv_300"),
    ("F", "balanced SNR threshold", "easy_picture_3x3_stage_f_snr15_500", 500, "1e-4", "easy_picture_3x3_stage_e_background_300"),
    ("F-limit-1", "pure SNR limit", "easy_picture_3x3_stage_f_snr_limit_1000", 1000, "2e-4", "easy_picture_3x3_stage_f_snr15_500"),
    ("F-limit-2", "pure SNR limit", "easy_picture_3x3_stage_f_snr_limit_total_2500", 1500, "2e-4", "easy_picture_3x3_stage_f_snr_limit_1000"),
    ("F-limit-3", "pure SNR limit", "easy_picture_3x3_stage_f_snr_limit_total_3500", 1000, "2e-4", "easy_picture_3x3_stage_f_snr_limit_total_2500"),
]


def read_single_row(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.DictReader(handle))


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def extra_metrics(npz, channel_rows):
    raw = npz["optimized_raw"].astype(np.float64)
    targets = npz["targets"]
    foreground_means = np.asarray(
        [raw[i][targets[i] > 0].mean() for i in range(len(raw))]
    )
    snr = np.asarray(
        [float(row["foreground_background_snr_db"]) for row in channel_rows]
    )
    return {
        "channel_brightness_cv": foreground_means.std() / foreground_means.mean(),
        "snr_median_db": np.median(snr),
        "snr_channels_gt_15_db": int(np.sum(snr > 15.0)),
    }


def command_for(stage, directory, epochs, lr, source):
    base = (
        "C:\\Users\\18441\\anaconda3\\python.exe "
        "coding\\python\\order_decoupling_grayscale.py "
        "--mat-file output\\easy_picture_3x3_order_analysis\\easy_picture_3x3_target.mat "
        f"--output-dir output\\{directory} --epochs {epochs} --lr {lr} --seed 42 "
        "--channel-count 9 --order-grid-size 3 --order-m-start -3 --order-n-start -3 "
        "--image-loss-mode energy --device cuda --log-interval 100"
    )
    if source:
        base += f" --initial-results output\\{source}\\optimized_results.npz"
    if stage in {"B", "C", "D", "E", "F"}:
        base += " --priority-channels 6 8 9 --priority-channel-weight 3"
    if stage in {"C", "D", "E", "F"}:
        base += " --brightness-consistency-weight 10 --worst-channel-weight 5"
    if stage in {"D", "E", "F"}:
        base += " --cross-level-weight 20"
    if stage in {"E", "F"}:
        base += (
            " --background-uniformity-weight 5 --background-cluster-weight 2"
            " --background-cluster-kernel 9 --background-cluster-upper 0.9"
        )
    if stage == "F":
        base += " --paper-snr-weight 20 --paper-snr-target-db 15"
    if stage.startswith("F-limit"):
        base += " --paper-snr-weight 100 --paper-snr-target-db 15"
    return base


def save_weak_channels(npz, output_file):
    raw = npz["optimized_raw"].astype(np.float64)
    targets = npz["targets"]
    pairs = npz["pairMat"].astype(int)
    selected = [5, 7, 8]
    scale = np.percentile(raw, 99.5)
    figure, axes = plt.subplots(len(selected), 2, figsize=(7, 10))
    for row_index, channel in enumerate(selected):
        axes[row_index, 0].imshow(targets[channel], cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].imshow(raw[channel], cmap="gray", vmin=0, vmax=scale)
        label = f"Ch{channel + 1} ({pairs[channel, 0]},{pairs[channel, 1]})"
        axes[row_index, 0].set_title(f"{label} target")
        axes[row_index, 1].set_title(f"{label} reconstruction")
        axes[row_index, 0].axis("off")
        axes[row_index, 1].axis("off")
    figure.tight_layout()
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def main():
    REPORT.mkdir(parents=True, exist_ok=False)
    stage_rows = []
    for stage, purpose, directory, epochs, lr, source in STAGES:
        stage_dir = OUTPUT / directory
        summary = read_single_row(stage_dir / "evaluation_summary.csv")
        channels = read_rows(stage_dir / "evaluation_channel_metrics.csv")
        with np.load(stage_dir / "optimized_results.npz") as npz:
            extra = extra_metrics(npz, channels)
        command = command_for(stage, directory, epochs, lr, source)
        (stage_dir / "run_command.txt").write_text(command + "\n", encoding="utf-8")
        stage_rows.append(
            {
                "stage": stage,
                "purpose": purpose,
                "source_npz": "random seed 42" if not source else f"output/{source}/optimized_results.npz",
                "epochs_this_run": epochs,
                "learning_rate": lr,
                "structure_mean": summary["structure_cosine_mean"],
                "structure_min": summary["structure_cosine_min"],
                "coverage_mean": summary["foreground_coverage_mean"],
                "coverage_min": summary["foreground_coverage_min"],
                "monotonic_channels": summary["grayscale_monotonic_channels"],
                "ratio_rmse_mean": summary["grayscale_ratio_rmse_mean"],
                "ratio_rmse_max": summary["grayscale_ratio_rmse_max"],
                "cv_1_3": summary["S_1_3_cv"],
                "cv_2_3": summary["S_2_3_cv"],
                "cv_1": summary["S_1_cv"],
                "channel_brightness_cv": extra["channel_brightness_cv"],
                "background_cv": summary["background_cv_mean"],
                "background_p95_ratio": summary["background_p95_ratio_mean"],
                "cnr_mean": summary["foreground_background_cnr_mean"],
                "cnr_min": summary["foreground_background_cnr_min"],
                "snr_mean_db": summary["foreground_background_snr_db_mean"],
                "snr_median_db": extra["snr_median_db"],
                "snr_min_db": summary["foreground_background_snr_db_min"],
                "snr_max_db": summary["foreground_background_snr_db_max"],
                "snr_channels_gt_15_db": extra["snr_channels_gt_15_db"],
                "command": command,
            }
        )

    with (REPORT / "stage_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)

    final_dir = OUTPUT / STAGES[-1][2]
    final_channels = read_rows(final_dir / "evaluation_channel_metrics.csv")
    final_summary = read_single_row(final_dir / "evaluation_summary.csv")
    with np.load(final_dir / "optimized_results.npz") as final:
        pairs = final["pairMat"].astype(int)
        extra = extra_metrics(final, final_channels)
        save_weak_channels(final, REPORT / "weak_channels_target_vs_reconstruction.png")

    combined = []
    for index, row in enumerate(final_channels):
        combined.append({"channel": index + 1, "m": pairs[index, 0], "n": pairs[index, 1], **row})
    with (REPORT / "final_channel_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    sorted_rows = sorted(combined, key=lambda row: float(row["foreground_background_snr_db"]))
    with (REPORT / "snr_sorted_channels.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["channel", "m", "n", "foreground_background_snr_db", "structure_cosine", "foreground_coverage_above_background_p95", "grayscale_monotonic", "grayscale_ratio_rmse"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_rows)

    final_summary.update(extra)
    with (REPORT / "final_summary_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_summary))
        writer.writeheader()
        writer.writerow(final_summary)


if __name__ == "__main__":
    main()
