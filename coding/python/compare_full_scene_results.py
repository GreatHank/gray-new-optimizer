import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRID_SIZE = 6


def assemble_channels(raw, grid_positions=None, grid_size=GRID_SIZE):
    if raw.ndim != 3 or not 1 <= raw.shape[0] <= grid_size**2:
        raise ValueError(
            f"optimized_raw 应为1到{grid_size ** 2}通道三维数组，实际形状为 {raw.shape}。"
        )
    if grid_positions is None:
        grid_positions = np.asarray(
            [divmod(index, grid_size) for index in range(raw.shape[0])]
        )
    grid_positions = np.asarray(grid_positions, dtype=int)
    if grid_positions.shape != (raw.shape[0], 2):
        raise ValueError("grid_positions 必须为与结果通道数一致的 N×2 数组。")
    tiles = np.zeros(
        (grid_size, grid_size, raw.shape[1], raw.shape[2]), dtype=raw.dtype
    )
    for image, (row, column) in zip(raw, grid_positions):
        tiles[row, column] = image
    return (
        tiles.transpose(0, 2, 1, 3)
        .reshape(grid_size * raw.shape[1], grid_size * raw.shape[2])
    )


def rotate_assembled_scene(scene, degrees=0):
    if degrees not in (0, 90, 180, 270):
        raise ValueError("assembly-rotation-deg 只允许 0、90、180、270。")
    return np.rot90(scene, degrees // 90)


def restore_assembled_scene(scene, degrees=0, mirror_lr=False):
    restored = rotate_assembled_scene(scene, degrees)
    return np.fliplr(restored) if mirror_lr else restored


def load_summary(result_dir):
    with (result_dir / "evaluation_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return next(csv.DictReader(handle))


def metric_title(result_dir, summary, ratios):
    return (
        f"{result_dir.name}\n"
        f"structure={float(summary['structure_cosine_mean']):.3f}, "
        f"CV={float(summary['S_1_3_cv']):.3f}/"
        f"{float(summary['S_2_3_cv']):.3f}/"
        f"{float(summary['S_1_cv']):.3f}, "
        f"ratios={ratios[0]:.3f}:{ratios[1]:.3f}:1"
    )


def save_individual(original, optimized, title, output_file, factor):
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[1].imshow(optimized, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Optimized, shared illumination {factor:g}x")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_overview(original, results, output_file, factor):
    rows = int(np.ceil(len(results) / 2))
    figure, axes = plt.subplots(rows, 2, figsize=(16, 7.5 * rows), squeeze=False)
    for axis, (optimized, title) in zip(axes.flat, results):
        comparison = np.concatenate((original / 255.0, optimized), axis=1)
        axis.imshow(comparison, cmap="gray", vmin=0, vmax=1)
        axis.axvline(original.shape[1], color="white", linewidth=1)
        axis.text(
            0.25,
            1.01,
            "Original",
            transform=axis.transAxes,
            ha="center",
            fontsize=10,
        )
        axis.text(
            0.75,
            1.01,
            f"Optimized {factor:g}x",
            transform=axis.transAxes,
            ha="center",
            fontsize=10,
        )
        axis.set_title(title, fontsize=9, pad=22)
        axis.axis("off")
    for axis in axes.flat[len(results) :]:
        axis.axis("off")
    figure.suptitle(
        "Full-scene comparison with one shared physical exposure",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(
        description="将方阵多通道结果重排为完整场景，并生成原图/优化图并排对比。"
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--illumination-factor", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument(
        "--assembly-rotation-deg",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="通道重排成完整场景后统一逆向旋转；不会交换单个图块。",
    )
    parser.add_argument(
        "--assembly-mirror-lr",
        action="store_true",
        help="通道重排后整体左右镜像，以恢复目标生成前的原图方向。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.illumination_factor <= 0:
        raise ValueError("illumination-factor 必须大于0。")
    if args.grid_size < 1:
        raise ValueError("grid-size 必须大于0。")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    loaded = []
    for result_dir in args.result_dirs:
        results = np.load(result_dir / "optimized_results.npz")
        raw = results["optimized_raw"].astype(np.float64, copy=False)
        summary = load_summary(result_dir)
        means = np.asarray(
            [
                float(summary["S_1_3_mean"]),
                float(summary["S_2_3_mean"]),
                float(summary["S_1_mean"]),
            ]
        )
        ratios = means / means[2]
        grid_positions = (
            results["grid_positions"]
            if "grid_positions" in results.files
            else None
        )
        loaded.append((result_dir, raw, summary, ratios, grid_positions))

    shared_display_scale = max(
        float(np.percentile(raw, 99.9))
        for _dir, raw, _summary, _ratios, _positions in loaded
    )
    scene_size = args.grid_size * loaded[0][1].shape[-1]
    original = np.asarray(
        Image.open(args.input_file)
        .convert("L")
        .resize((scene_size, scene_size), Image.Resampling.LANCZOS)
    )

    overview_results = []
    for result_dir, raw, summary, ratios, grid_positions in loaded:
        optimized = np.clip(
            restore_assembled_scene(
                assemble_channels(raw, grid_positions, args.grid_size),
                args.assembly_rotation_deg,
                args.assembly_mirror_lr,
            )
            * args.illumination_factor
            / shared_display_scale,
            0,
            1,
        )
        title = metric_title(result_dir, summary, ratios)
        output_file = args.output_dir / f"{result_dir.name}_original_vs_optimized.png"
        save_individual(
            original,
            optimized,
            title,
            output_file,
            args.illumination_factor,
        )
        overview_results.append((optimized, title))
        print(output_file)

    overview_file = args.output_dir / "all_stages_original_vs_optimized.png"
    save_overview(
        original,
        overview_results,
        overview_file,
        args.illumination_factor,
    )
    print(overview_file)
    print(f"shared_display_scale_1x={shared_display_scale:.9g}")


if __name__ == "__main__":
    main()
