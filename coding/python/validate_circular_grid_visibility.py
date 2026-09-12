import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="比较不同连续级次方阵在固定周期下对圆形目标的空气锥裁切。"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=project_root
        / "output"
        / "circular_7x7_target_49"
        / "circular_scene_3500.png",
    )
    parser.add_argument("--grid-sizes", nargs="+", type=int, default=[6, 7])
    parser.add_argument("--wavelength-nm", type=float, default=480.0)
    parser.add_argument("--period-nm", type=float, default=850.0)
    parser.add_argument("--analysis-size", type=int, default=1400)
    parser.add_argument(
        "--overscan-factor",
        type=float,
        default=1.0,
        help="相对严格空气锥内接尺寸的放大倍数；大于1允许圆图外缘被部分裁切。",
    )
    parser.add_argument("--order-m-start", type=int)
    parser.add_argument("--order-n-start", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def circle_coordinates(size):
    axis = (np.arange(size, dtype=np.float32) + 0.5 - size / 2) / (size / 2)
    return np.hypot(axis[:, None], axis[None, :])


def fit_scene_to_fraction(scene, fraction):
    size = scene.shape[0]
    fitted_size = max(1, int(round(size * fraction)))
    resized = np.asarray(
        Image.fromarray(scene).resize(
            (fitted_size, fitted_size), resample=Image.Resampling.NEAREST
        )
    )
    canvas = np.zeros_like(scene)
    offset = (size - fitted_size) // 2
    canvas[offset : offset + fitted_size, offset : offset + fitted_size] = resized
    return canvas


def foreground_detail(scene):
    normalized = scene.astype(np.float32) / 255.0
    dx = np.abs(np.diff(normalized, axis=1, prepend=normalized[:, :1]))
    dy = np.abs(np.diff(normalized, axis=0, prepend=normalized[:1, :]))
    return normalized + dx + dy


def tile_rows(scene, visible_mask, grid_size):
    detail = foreground_detail(scene)
    size = scene.shape[0]
    edges = np.linspace(0, size, grid_size + 1, dtype=int)
    rows = []
    for row in range(grid_size):
        for column in range(grid_size):
            rs = slice(edges[row], edges[row + 1])
            cs = slice(edges[column], edges[column + 1])
            local_visible = visible_mask[rs, cs]
            local_detail = detail[rs, cs]
            total_detail = float(local_detail.sum())
            visible_detail = float((local_detail * local_visible).sum())
            rows.append(
                {
                    "grid_size": grid_size,
                    "grid_row": row + 1,
                    "grid_column": column + 1,
                    "air_cone_area_fraction": float(local_visible.mean()),
                    "detail_total": total_detail,
                    "detail_visible": visible_detail,
                    "detail_retained_fraction": (
                        visible_detail / total_detail if total_detail > 0 else 1.0
                    ),
                }
            )
    return rows


def add_grid(image, grid_size):
    rendered = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(rendered)
    size = image.shape[0]
    for index in range(1, grid_size):
        coordinate = round(index * size / grid_size)
        draw.line((coordinate, 0, coordinate, size), fill=(255, 80, 80), width=2)
        draw.line((0, coordinate, size, coordinate), fill=(255, 80, 80), width=2)
    return np.asarray(rendered)


def save_fitted_target(
    scene,
    grid_size,
    output_file,
    tile_size=500,
    order_m_start=None,
    order_n_start=None,
):
    size = scene.shape[0]
    edges = np.linspace(0, size, grid_size + 1, dtype=int)
    targets = []
    positions = []
    for row in range(grid_size):
        for column in range(grid_size):
            if (
                order_m_start is not None
                and row + order_m_start == 0
                and column + order_n_start == 0
            ):
                continue
            tile = scene[
                edges[row] : edges[row + 1],
                edges[column] : edges[column + 1],
            ]
            if not np.any(tile):
                continue
            resized = np.asarray(
                Image.fromarray(tile).resize(
                    (tile_size, tile_size), resample=Image.Resampling.NEAREST
                )
            )
            target = resized.astype(np.float32) / 255.0
            targets.append(target)
            positions.append((row, column))
    sio.savemat(
        output_file,
        {
            "bw_all": np.stack(targets),
            "grid_positions": np.asarray(positions, dtype=np.int16),
            "grid_size": np.asarray([[grid_size]], dtype=np.int16),
        },
        do_compression=True,
    )
    return len(targets)


def main():
    args = parse_args()
    if not args.input_file.is_file():
        raise FileNotFoundError(f"输入图不存在：{args.input_file}")
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{args.output_dir}")
    if args.wavelength_nm <= 0 or args.period_nm <= 0:
        raise ValueError("波长和周期必须大于0。")
    if args.analysis_size < 100:
        raise ValueError("analysis-size必须至少为100。")
    if any(grid_size < 1 for grid_size in args.grid_sizes):
        raise ValueError("grid-size必须大于0。")
    if args.overscan_factor <= 0:
        raise ValueError("overscan-factor必须大于0。")
    if (args.order_m_start is None) != (args.order_n_start is None):
        raise ValueError("order-m-start和order-n-start必须同时提供。")

    with Image.open(args.input_file) as image:
        scene = np.asarray(
            image.convert("L").resize(
                (args.analysis_size, args.analysis_size),
                resample=Image.Resampling.NEAREST,
            )
        )
    radius = circle_coordinates(args.analysis_size)
    detail = foreground_detail(scene)
    total_detail = float(detail.sum())
    args.output_dir.mkdir(parents=True)

    summaries = []
    all_tile_rows = []
    for grid_size in args.grid_sizes:
        order_cell_width = args.wavelength_nm / args.period_nm
        source_circle_radius_in_direction_cosine = grid_size * order_cell_width / 2
        strict_fit_fraction = min(
            1.0, 1.0 / source_circle_radius_in_direction_cosine
        )
        fit_fraction = min(1.0, strict_fit_fraction * args.overscan_factor)
        visible_mask = radius <= strict_fit_fraction
        native_visible = scene.copy()
        native_visible[~visible_mask] = 0
        fitted_scene = fit_scene_to_fraction(scene, fit_fraction)
        fitted_target_file = args.output_dir / f"fitted_target_{grid_size}x{grid_size}.mat"

        rows = tile_rows(scene, visible_mask, grid_size)
        all_tile_rows.extend(rows)
        visible_tiles = sum(row["air_cone_area_fraction"] > 0 for row in rows)
        mostly_visible_tiles = sum(row["air_cone_area_fraction"] >= 0.5 for row in rows)
        retained_detail = float((detail * visible_mask).sum() / total_detail)
        active_fitted_tiles = save_fitted_target(
            fitted_scene,
            grid_size,
            fitted_target_file,
            order_m_start=args.order_m_start,
            order_n_start=args.order_n_start,
        )

        summary = {
            "grid_size": grid_size,
            "order_cell_width_lambda_over_period": order_cell_width,
            "source_circle_radius_in_direction_cosine": source_circle_radius_in_direction_cosine,
            "native_fill_visible_source_radius_fraction": strict_fit_fraction,
            "native_fill_retained_detail_fraction": retained_detail,
            "overscan_factor": args.overscan_factor,
            "target_scene_scale_fraction": fit_fraction,
            "overscan_visible_image_radius_fraction": min(
                1.0, 1.0 / args.overscan_factor
            ),
            "tiles_intersecting_air_cone": visible_tiles,
            "tiles_at_least_half_visible": mostly_visible_tiles,
            "fit_mode_active_nonblack_tiles": active_fitted_tiles,
            "fit_mode_target_mat": str(fitted_target_file),
            "omitted_zero_order": (
                args.order_m_start is not None
                and 0 <= -args.order_m_start < grid_size
                and 0 <= -args.order_n_start < grid_size
            ),
            "interpretation": (
                "native_fill quantifies strict air-cone clipping; target mode scales the "
                "supplied image relative to the strict fit, and overscan greater than one "
                "allows partial outer-edge clipping"
            ),
        }
        summaries.append(summary)

        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(add_grid(scene, grid_size))
        axes[0].set_title(f"{grid_size}x{grid_size} source fill")
        axes[1].imshow(add_grid(native_visible, grid_size))
        axes[1].set_title(
            f"Native fill: detail kept {retained_detail:.1%}\n"
            f"visible radius {strict_fit_fraction:.1%}"
        )
        axes[2].imshow(add_grid(fitted_scene, grid_size))
        axes[2].set_title(
            f"Target scale {fit_fraction:.1%}, overscan {args.overscan_factor:.2f}x\n"
            f"active tiles {active_fitted_tiles}/{grid_size**2}"
        )
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(
            args.output_dir / f"circular_visibility_{grid_size}x{grid_size}.png",
            dpi=180,
        )
        plt.close(figure)
        Image.fromarray(native_visible).save(
            args.output_dir / f"native_visible_{grid_size}x{grid_size}.png"
        )
        Image.fromarray(fitted_scene).save(
            args.output_dir / f"fitted_full_scene_{grid_size}x{grid_size}.png"
        )

    with (args.output_dir / "tile_visibility.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_tile_rows[0]))
        writer.writeheader()
        writer.writerows(all_tile_rows)
    (args.output_dir / "visibility_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
