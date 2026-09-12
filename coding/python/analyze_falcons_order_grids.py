import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_butterfly_4x4_orders import tile_complexity  # noqa: E402
from search_circular_6x6_orders import (  # noqa: E402
    conjugate_pair_count,
    group_metadata,
    incidence_angles,
    transform_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(description="比较falcons图的连续5x5/6x6无零级级次。")
    parser.add_argument("--input-file", type=Path, default=ROOT / "input/falcons.png")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wavelength-nm", type=float, default=480.0)
    parser.add_argument("--period-nm", type=float, default=2000.0)
    parser.add_argument("--start-min", type=int, default=-8)
    parser.add_argument("--start-max", type=int, default=8)
    return parser.parse_args()


def generic_order_grid(m_start, n_start, size):
    return np.asarray(
        [[(m_start + row, n_start + column) for column in range(size)] for row in range(size)],
        dtype=np.int16,
    )


def physical_metrics(orders, wavelength_nm, period_nm):
    step = wavelength_nm / period_nm
    center_order = orders.reshape(-1, 2).mean(axis=0)
    spectral_center = center_order * step
    size = orders.shape[0]

    def minimum_incidence_shift(half_extent):
        offsets = np.asarray(
            [[sx * half_extent, sy * half_extent] for sx in (-1, 1) for sy in (-1, 1)]
        )
        desired = -spectral_center
        initial = desired / max(1.0, float(np.linalg.norm(desired)))
        constraints = [
            {"type": "ineq", "fun": lambda value: 1.0 - float(np.dot(value, value))}
        ]
        for offset in offsets:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda value, offset=offset: 1.0
                    - float(np.dot(value + spectral_center + offset, value + spectral_center + offset)),
                }
            )
        result = minimize(
            lambda value: float(np.dot(value, value)),
            initial,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            return None
        value = np.asarray(result.x, dtype=float)
        if np.linalg.norm(value) > 1 + 1e-7:
            return None
        if any(np.linalg.norm(value + spectral_center + offset) > 1 + 1e-7 for offset in offsets):
            return None
        return value

    full_shift = minimum_incidence_shift(size * step / 2)
    center_shift = minimum_incidence_shift((size - 1) * step / 2)
    if center_shift is None:
        desired = -spectral_center
        shift = desired / max(1.0, float(np.linalg.norm(desired)))
        shift_mode = "maximum_count_fallback"
    else:
        shift = full_shift if full_shift is not None else center_shift
        shift_mode = "minimum_incidence_full_cells" if full_shift is not None else "minimum_incidence_all_centers"
    points = orders.reshape(-1, 2) * step + shift
    radii = np.linalg.norm(points, axis=1)
    theta_i, phi_i = incidence_angles(shift)
    low = orders[0, 0].astype(float) * step + shift - step / 2
    high = orders[-1, -1].astype(float) * step + shift + step / 2
    corners = np.asarray([[x, y] for x in (low[0], high[0]) for y in (low[1], high[1])])
    corner_radii = np.linalg.norm(corners, axis=1)
    return {
        "shift": shift,
        "shift_mode": shift_mode,
        "theta_i_deg": theta_i,
        "phi_i_deg": phi_i,
        "propagating_centers": int(np.sum(radii <= 1 + 1e-12)),
        "center_margin": float(1 - radii.max()),
        "full_cells_inside": bool(corner_radii.max() <= 1 + 1e-12),
        "cell_corner_margin": float(1 - corner_radii.max()),
        "nominal_circular_fov_deg": 2 * math.degrees(math.asin(min(1.0, size * step / 2))),
    }


def best_mapping(complexity, orders):
    size = orders.shape[0]
    rank = np.argsort(complexity.ravel())[::-1]
    top_count = size
    top = np.zeros(size * size, dtype=bool)
    top[rank[:top_count]] = True
    top = top.reshape(size, size)
    _groups, group_sizes, descriptions = group_metadata(orders)
    axes = (orders[:, :, 0] == 0) | (orders[:, :, 1] == 0)
    choices = []
    for name, index_map in transform_indices(size).items():
        assigned = complexity.ravel()[index_map]
        assigned_top = top.ravel()[index_map]
        partners = group_sizes - 1
        weighted = float(np.sum(assigned * partners))
        axis_weighted = float(np.sum(assigned[axes] * partners[axes]))
        top_weighted = float(np.sum(assigned[assigned_top] * partners[assigned_top]))
        score = weighted + 4 * axis_weighted + top_weighted
        choices.append(
            {
                "transform": name,
                "score": score,
                "weighted_burden": weighted,
                "axis_weighted_burden": axis_weighted,
                "top_without_multiple": int(np.sum(assigned_top & (group_sizes == 1))),
                "top_on_axis": int(np.sum(assigned_top & axes)),
                "index_map": index_map,
                "group_sizes": group_sizes,
                "descriptions": descriptions,
                "assigned_top": assigned_top,
            }
        )
    return min(choices, key=lambda item: (item["score"], -item["top_without_multiple"]))


def analyze_size(gray, size, args):
    complexity = tile_complexity(gray, size)[-1]
    rows = []
    for m_start in range(args.start_min, args.start_max + 1):
        for n_start in range(args.start_min, args.start_max + 1):
            orders = generic_order_grid(m_start, n_start, size)
            if np.any(np.all(orders == 0, axis=2)):
                continue
            physical = physical_metrics(orders, args.wavelength_nm, args.period_nm)
            mapping = best_mapping(complexity, orders)
            axes = (orders[:, :, 0] == 0) | (orders[:, :, 1] == 0)
            center = size // 2
            center_flat = int(mapping["index_map"][center, center])
            source_center = divmod(center_flat, size)
            row = {
                "size": size,
                "m_start": m_start,
                "m_end": m_start + size - 1,
                "n_start": n_start,
                "n_end": n_start + size - 1,
                "transform": mapping["transform"],
                "zero_order_count": 0,
                "axis_order_count": int(axes.sum()),
                "conjugate_pair_count": conjugate_pair_count(orders),
                "propagating_centers": physical["propagating_centers"],
                "full_cells_inside": physical["full_cells_inside"],
                "theta_i_deg": physical["theta_i_deg"],
                "phi_i_deg": physical["phi_i_deg"],
                "shift_mode": physical["shift_mode"],
                "center_margin": physical["center_margin"],
                "cell_corner_margin": physical["cell_corner_margin"],
                "nominal_circular_fov_deg": physical["nominal_circular_fov_deg"],
                "coupling_score": mapping["score"],
                "weighted_burden": mapping["weighted_burden"],
                "top_without_multiple": mapping["top_without_multiple"],
                "top_on_axis": mapping["top_on_axis"],
                "central_order_m": int(orders[center, center, 0]),
                "central_order_n": int(orders[center, center, 1]),
                "central_order_group_size": int(mapping["group_sizes"][center, center]),
                "order_center_receives_source_row": source_center[0] + 1,
                "order_center_receives_source_column": source_center[1] + 1,
            }
            rows.append(row)
    rows.sort(
        key=lambda row: (
            -row["propagating_centers"],
            not row["full_cells_inside"],
            row["axis_order_count"] > 0,
            row["central_order_group_size"] > 1,
            row["coupling_score"],
            row["theta_i_deg"],
        )
    )
    return rows, complexity


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_complexities(gray, results, output_file):
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for axis, (size, complexity) in zip(axes, results):
        axis.imshow(gray, cmap="gray", vmin=0, vmax=255)
        height, width = gray.shape
        for index in range(1, size):
            axis.axhline(index * height / size, color="tab:red", linewidth=0.8)
            axis.axvline(index * width / size, color="tab:red", linewidth=0.8)
        for row in range(size):
            for column in range(size):
                axis.text(
                    (column + 0.5) * width / size,
                    (row + 0.5) * height / size,
                    f"{row+1},{column+1}\n{complexity[row,column]:.2f}",
                    color="yellow",
                    fontsize=7,
                    ha="center",
                    va="center",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
                )
        axis.set_title(f"{size}x{size} complexity")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    gray = np.asarray(Image.open(args.input_file).convert("L"))
    summaries = {}
    complexity_results = []
    for size in (5, 6):
        rows, complexity = analyze_size(gray, size, args)
        write_csv(args.output_dir / f"ranking_{size}x{size}.csv", rows)
        summaries[f"{size}x{size}"] = rows[:10]
        complexity_results.append((size, complexity))
    plot_complexities(gray, complexity_results, args.output_dir / "falcons_complexity_5x5_6x6.png")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
