import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_circular_6x6_orders import (  # noqa: E402
    conjugate_pair_count,
    group_metadata,
    incidence_angles,
    primitive_direction,
    propagation,
    transform_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="为蝴蝶图搜索连续无零级4x4衍射级次并输出完整角度数据。"
    )
    parser.add_argument("--input-file", type=Path, default=ROOT / "input/butterfly.png")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wavelength-nm", type=float, default=480.0)
    parser.add_argument("--period-nm", type=float, default=850.0)
    parser.add_argument("--theory-period-nm", type=float, default=2000.0)
    parser.add_argument("--search-start-min", type=int, default=-10)
    parser.add_argument("--search-start-max", type=int, default=10)
    return parser.parse_args()


def order_grid(m_start, n_start, size=4):
    return np.asarray(
        [[(m_start + row, n_start + column) for column in range(size)] for row in range(size)],
        dtype=np.int16,
    )


def tile_complexity(gray, size=4):
    image = gray.astype(np.float64) / 255.0
    y_edges = np.linspace(0, image.shape[0], size + 1, dtype=int)
    x_edges = np.linspace(0, image.shape[1], size + 1, dtype=int)
    density = np.zeros((size, size), dtype=float)
    edge_tv = np.zeros((size, size), dtype=float)
    for row in range(size):
        for column in range(size):
            tile = image[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ]
            density[row, column] = float(tile.mean())
            edge_tv[row, column] = float(
                (
                    np.abs(np.diff(tile, axis=0)).sum()
                    + np.abs(np.diff(tile, axis=1)).sum()
                )
                / tile.size
            )
    density_n = density / density.max()
    edge_tv_n = edge_tv / edge_tv.max()
    complexity = 0.5 * density_n + 0.5 * edge_tv_n
    return density, edge_tv, density_n, edge_tv_n, complexity


def candidate_shifts(step, refinement_center=None):
    if refinement_center is None:
        axis = np.linspace(-1.0, 1.0, 101)
    else:
        axis = np.linspace(-0.03, 0.03, 121)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    shifts = np.column_stack((xx.ravel(), yy.ravel()))
    if refinement_center is not None:
        shifts += refinement_center
    return shifts[np.linalg.norm(shifts, axis=1) <= 1 + 1e-12]


def choose_shift(points, shifts):
    radii = np.linalg.norm(shifts[:, None, :] + points[None, :, :], axis=2)
    counts = (radii <= 1 + 1e-12).sum(axis=1)
    sorted_radii = np.sort(radii, axis=1)
    kth = sorted_radii[np.arange(len(shifts)), np.maximum(counts - 1, 0)]
    means = np.asarray(
        [sorted_radii[index, : max(1, int(count))].mean() for index, count in enumerate(counts)]
    )
    keys = np.lexsort((means, kth, -counts))
    return shifts[int(keys[0])]


def best_common_incidence(orders, wavelength_nm, period_nm):
    points = orders.reshape(-1, 2).astype(float) * (wavelength_nm / period_nm)
    coarse = choose_shift(points, candidate_shifts(wavelength_nm / period_nm))
    fine = choose_shift(points, candidate_shifts(wavelength_nm / period_nm, coarse))
    output, radii, captured = propagation(orders, fine, wavelength_nm, period_nm)
    theta_i, phi_i = incidence_angles(fine)
    return {
        "shift": fine,
        "output": output,
        "radii": radii,
        "captured": captured,
        "theta_i_deg": theta_i,
        "phi_i_deg": phi_i,
        "propagating_count": int(captured.sum()),
        "minimum_captured_margin": (
            float(1 - radii[captured].max()) if captured.any() else math.nan
        ),
    }


def theory_full_propagation(orders, wavelength_nm, period_nm):
    step = wavelength_nm / period_nm
    center_order = orders.reshape(-1, 2).mean(axis=0)
    desired = -center_order * step
    shift = desired / max(1.0, float(np.linalg.norm(desired)))
    output, radii, captured = propagation(orders, shift, wavelength_nm, period_nm)
    theta_i, phi_i = incidence_angles(shift)
    return {
        "shift": shift,
        "output": output,
        "radii": radii,
        "captured": captured,
        "theta_i_deg": theta_i,
        "phi_i_deg": phi_i,
        "propagating_count": int(captured.sum()),
        "minimum_margin": float(1 - radii.max()),
    }


def mapping_metrics(complexity, top8, top4, orders, index_map, transform):
    _groups, group_sizes, descriptions = group_metadata(orders)
    assigned = complexity.ravel()[index_map]
    assigned_top8 = top8.ravel()[index_map]
    assigned_top4 = top4.ravel()[index_map]
    partners = group_sizes - 1
    axes = (orders[:, :, 0] == 0) | (orders[:, :, 1] == 0)
    weighted = float(np.sum(assigned * partners))
    axis_weighted = float(np.sum(assigned[axes] * partners[axes]))
    top8_weighted = float(np.sum(assigned[assigned_top8] * partners[assigned_top8]))
    score = weighted + 4 * axis_weighted + top8_weighted
    return {
        "transform": transform,
        "coupling_score": score,
        "weighted_multiple_burden": weighted,
        "axis_weighted_burden": axis_weighted,
        "top8_weighted_burden": top8_weighted,
        "top8_without_multiple": int(np.sum(assigned_top8 & (group_sizes == 1))),
        "top4_without_multiple": int(np.sum(assigned_top4 & (group_sizes == 1))),
        "high_complexity_axis_group4": int(
            np.sum(assigned_top8 & axes & (group_sizes == 4))
        ),
        "largest_group_size": int(group_sizes.max()),
        "assigned": assigned,
        "assigned_top8": assigned_top8,
        "assigned_top4": assigned_top4,
        "group_sizes": group_sizes,
        "group_descriptions": descriptions,
        "index_map": index_map,
    }


def symmetry_class(m_start, n_start):
    canonical_m = min(int(m_start), -int(m_start) - 3)
    canonical_n = min(int(n_start), -int(n_start) - 3)
    return tuple(sorted((canonical_m, canonical_n)))


def search(complexity, wavelength_nm, period_nm, theory_period_nm, start_min, start_max):
    flat_rank = np.argsort(complexity.ravel())[::-1]
    top8 = np.zeros(16, dtype=bool)
    top8[flat_rank[:8]] = True
    top8 = top8.reshape(4, 4)
    top4 = np.zeros(16, dtype=bool)
    top4[flat_rank[:4]] = True
    top4 = top4.reshape(4, 4)
    transforms = transform_indices(4)
    rows = []
    details = {}
    for m_start in range(start_min, start_max + 1):
        for n_start in range(start_min, start_max + 1):
            orders = order_grid(m_start, n_start)
            if np.any(np.all(orders == 0, axis=2)):
                continue
            theory = theory_full_propagation(orders, wavelength_nm, theory_period_nm)
            if theory["propagating_count"] != 16:
                continue
            physical = best_common_incidence(orders, wavelength_nm, period_nm)
            mappings = [
                mapping_metrics(complexity, top8, top4, orders, index_map, name)
                for name, index_map in transforms.items()
            ]
            best_map = min(
                mappings,
                key=lambda item: (
                    item["coupling_score"],
                    -item["top4_without_multiple"],
                    -item["top8_without_multiple"],
                ),
            )
            rows.append(
                {
                    "m_start": m_start,
                    "m_end": m_start + 3,
                    "n_start": n_start,
                    "n_end": n_start + 3,
                    "zero_order_count": 0,
                    "conjugate_pair_count": conjugate_pair_count(orders),
                    "transform": best_map["transform"],
                    "p850_propagating_count": physical["propagating_count"],
                    "p850_theta_i_deg": physical["theta_i_deg"],
                    "p850_phi_i_deg": physical["phi_i_deg"],
                    "p850_minimum_captured_margin": physical[
                        "minimum_captured_margin"
                    ],
                    "p2000_propagating_count": theory["propagating_count"],
                    "p2000_theta_i_deg": theory["theta_i_deg"],
                    "p2000_phi_i_deg": theory["phi_i_deg"],
                    "p2000_minimum_margin": theory["minimum_margin"],
                    "top8_without_multiple": best_map["top8_without_multiple"],
                    "top4_without_multiple": best_map["top4_without_multiple"],
                    "high_complexity_axis_group4": best_map[
                        "high_complexity_axis_group4"
                    ],
                    "weighted_multiple_burden": best_map[
                        "weighted_multiple_burden"
                    ],
                    "axis_weighted_burden": best_map["axis_weighted_burden"],
                    "coupling_score": best_map["coupling_score"],
                    "largest_group_size": best_map["largest_group_size"],
                    "symmetry_class": str(symmetry_class(m_start, n_start)),
                }
            )
            details[(m_start, n_start)] = (orders, best_map, physical, theory)
    rows.sort(
        key=lambda item: (
            -item["p850_propagating_count"],
            item["coupling_score"],
            -item["top4_without_multiple"],
            item["p850_theta_i_deg"],
        )
    )
    representatives = []
    classes = set()
    for row in rows:
        if row["symmetry_class"] in classes:
            continue
        classes.add(row["symmetry_class"])
        representatives.append(row)
    return rows, representatives, details, flat_rank


def cell_overlap(center, step, samples=301):
    axis = np.linspace(-step / 2, step / 2, samples)
    xx, yy = np.meshgrid(axis + center[0], axis + center[1], indexing="xy")
    radius = np.hypot(xx, yy)
    inside = radius <= 1
    if not inside.any():
        return 0.0, math.nan, math.nan
    theta = np.degrees(np.arcsin(np.clip(radius[inside], 0, 1)))
    return float(inside.mean()), float(theta.min()), float(theta.max())


def selected_order_rows(
    recommended, details, complexity, wavelength_nm, period_nm, detail_index=2
):
    key = (recommended["m_start"], recommended["n_start"])
    selected_details = details[key]
    orders, mapping, physical = (
        selected_details[0],
        selected_details[1],
        selected_details[detail_index],
    )
    step = wavelength_nm / period_nm
    rows = []
    for grid_row in range(4):
        for grid_column in range(4):
            m, n = map(int, orders[grid_row, grid_column])
            flat = int(mapping["index_map"][grid_row, grid_column])
            source_row, source_column = divmod(flat, 4)
            ux, uy = map(float, physical["output"][grid_row * 4 + grid_column])
            radius = float(physical["radii"][grid_row * 4 + grid_column])
            propagating = bool(physical["captured"][grid_row * 4 + grid_column])
            overlap, theta_min, theta_max = cell_overlap((ux, uy), step)
            primitive_m, primitive_n = primitive_direction(m, n)
            rows.append(
                {
                    "order_grid_row": grid_row + 1,
                    "order_grid_column": grid_column + 1,
                    "source_tile_row": source_row + 1,
                    "source_tile_column": source_column + 1,
                    "m": m,
                    "n": n,
                    "complexity": float(complexity[source_row, source_column]),
                    "primitive_m": primitive_m,
                    "primitive_n": primitive_n,
                    "multiple_group_size": int(mapping["group_sizes"][grid_row, grid_column]),
                    "multiple_coupling_degree": int(
                        mapping["group_sizes"][grid_row, grid_column] - 1
                    ),
                    "multiple_group": mapping["group_descriptions"][grid_row, grid_column],
                    "on_axis": m == 0 or n == 0,
                    "ux": ux,
                    "uy": uy,
                    "u_radius": radius,
                    "uz": math.sqrt(max(0.0, 1 - radius * radius)) if propagating else math.nan,
                    "theta_out_deg": math.degrees(math.asin(radius)) if propagating else math.nan,
                    "phi_out_deg": math.degrees(math.atan2(uy, ux)),
                    "propagating": propagating,
                    "air_margin": 1 - radius,
                    "cell_ux_min": ux - step / 2,
                    "cell_ux_max": ux + step / 2,
                    "cell_uy_min": uy - step / 2,
                    "cell_uy_max": uy + step / 2,
                    "cell_air_circle_overlap_fraction": overlap,
                    "cell_theta_min_deg": theta_min,
                    "cell_theta_max_deg": theta_max,
                }
            )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_complexity(gray, complexity, flat_rank, output_file):
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(gray, cmap="gray", vmin=0, vmax=255)
    height, width = gray.shape
    ranks = {int(flat): rank + 1 for rank, flat in enumerate(flat_rank)}
    for index in range(1, 4):
        axis.axhline(index * height / 4, color="tab:red", linewidth=1)
        axis.axvline(index * width / 4, color="tab:red", linewidth=1)
    for row in range(4):
        for column in range(4):
            flat = row * 4 + column
            axis.text(
                (column + 0.5) * width / 4,
                (row + 0.5) * height / 4,
                f"tile {row+1},{column+1}\nC={complexity[row,column]:.3f}\nrank {ranks[flat]}",
                ha="center",
                va="center",
                color="yellow",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
            )
    axis.axis("off")
    axis.set_title("Butterfly 4x4 complexity")
    figure.tight_layout()
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_order_map(recommended, details, order_rows, wavelength_nm, period_nm, output_file):
    key = (recommended["m_start"], recommended["n_start"])
    _orders, _mapping, physical, _theory = details[key]
    step = wavelength_nm / period_nm
    figure, axis = plt.subplots(figsize=(8, 8))
    circle = plt.Circle((0, 0), 1, fill=False, color="black", linewidth=2)
    axis.add_patch(circle)
    for item in order_rows:
        color = "tab:green" if item["propagating"] else "tab:red"
        rectangle = plt.Rectangle(
            (item["cell_ux_min"], item["cell_uy_min"]),
            step,
            step,
            facecolor=color,
            alpha=0.14,
            edgecolor=color,
            linewidth=1,
        )
        axis.add_patch(rectangle)
        axis.scatter(item["ux"], item["uy"], color=color, s=42, zorder=3)
        axis.text(
            item["ux"],
            item["uy"],
            f"({item['m']},{item['n']})\n{item['theta_out_deg']:.1f}°" if item["propagating"] else f"({item['m']},{item['n']})\nevan.",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axis.scatter(
        physical["shift"][0],
        physical["shift"][1],
        marker="*",
        color="tab:blue",
        s=130,
        label="incident transverse vector",
    )
    axis.axhline(0, color="0.75", linewidth=0.8)
    axis.axvline(0, color="0.75", linewidth=0.8)
    axis.set_xlim(-1.65, 1.65)
    axis.set_ylim(-1.65, 1.65)
    axis.set_aspect("equal")
    axis.set_xlabel("u_x")
    axis.set_ylabel("u_y")
    axis.set_title(
        f"Recommended 4x4 order cells, {wavelength_nm:.0f}/{period_nm:.0f} nm\n"
        f"theta_i={physical['theta_i_deg']:.3f}°, phi_i={physical['phi_i_deg']:.3f}°, "
        f"centers={physical['propagating_count']}/16"
    )
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output_file, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{args.output_dir}")
    with Image.open(args.input_file) as image:
        gray = np.asarray(image.convert("L"))
    density, edge_tv, density_n, edge_tv_n, complexity = tile_complexity(gray)
    ranking, representatives, details, flat_rank = search(
        complexity,
        args.wavelength_nm,
        args.period_nm,
        args.theory_period_nm,
        args.search_start_min,
        args.search_start_max,
    )
    recommended = representatives[0]
    order_rows = selected_order_rows(
        recommended, details, complexity, args.wavelength_nm, args.period_nm
    )
    theory_order_rows = selected_order_rows(
        recommended,
        details,
        complexity,
        args.wavelength_nm,
        args.theory_period_nm,
        detail_index=3,
    )
    args.output_dir.mkdir(parents=True)
    complexity_rows = []
    ranks = {int(flat): rank + 1 for rank, flat in enumerate(flat_rank)}
    for row in range(4):
        for column in range(4):
            flat = row * 4 + column
            complexity_rows.append(
                {
                    "tile_row": row + 1,
                    "tile_column": column + 1,
                    "bright_line_density": float(density[row, column]),
                    "edge_tv_per_pixel": float(edge_tv[row, column]),
                    "density_normalized": float(density_n[row, column]),
                    "edge_tv_normalized": float(edge_tv_n[row, column]),
                    "complexity": float(complexity[row, column]),
                    "complexity_rank": ranks[flat],
                }
            )
    write_csv(args.output_dir / "tile_complexity.csv", complexity_rows)
    write_csv(args.output_dir / "all_candidate_ranking.csv", ranking)
    write_csv(args.output_dir / "symmetry_class_ranking.csv", representatives)
    write_csv(args.output_dir / "recommended_order_data.csv", order_rows)
    write_csv(
        args.output_dir / "recommended_order_data_P2000_theory.csv",
        theory_order_rows,
    )
    plot_complexity(gray, complexity, flat_rank, args.output_dir / "butterfly_4x4_complexity.png")
    plot_order_map(
        recommended,
        details,
        order_rows,
        args.wavelength_nm,
        args.period_nm,
        args.output_dir / "recommended_order_angular_map.png",
    )
    visible_cells = [item for item in order_rows if item["cell_air_circle_overlap_fraction"] > 0]
    block_visible_fraction = float(
        np.mean([item["cell_air_circle_overlap_fraction"] for item in order_rows])
    )
    complexity_weighted_visible_fraction = float(
        sum(
            item["complexity"] * item["cell_air_circle_overlap_fraction"]
            for item in order_rows
        )
        / sum(item["complexity"] for item in order_rows)
    )
    nominal_radius = 2 * args.wavelength_nm / args.period_nm
    summary = {
        "input_file": str(args.input_file.resolve()),
        "image_size": list(gray.shape[::-1]),
        "complexity_definition": "0.5*density/max(density)+0.5*edgeTV/max(edgeTV)",
        "search": {
            "start_range": [args.search_start_min, args.search_start_max],
            "zero_order_forbidden": True,
            "theory_filter": f"16/16 centers propagate for P={args.theory_period_nm:g} nm",
            "candidate_count": len(ranking),
            "symmetry_class_count": len(representatives),
            "ranking": "maximize P850 center count, then minimize complexity-weighted multiple burden with 4x axis penalty",
        },
        "recommended": recommended,
        "angular_summary": {
            "wavelength_nm": args.wavelength_nm,
            "period_nm": args.period_nm,
            "order_spacing_lambda_over_p": args.wavelength_nm / args.period_nm,
            "incident_ux": details[(recommended["m_start"], recommended["n_start"])][2]["shift"][0],
            "incident_uy": details[(recommended["m_start"], recommended["n_start"])][2]["shift"][1],
            "theta_i_deg": recommended["p850_theta_i_deg"],
            "phi_i_deg": recommended["p850_phi_i_deg"],
            "propagating_centers": recommended["p850_propagating_count"],
            "local_cells_intersecting_air_circle": len(visible_cells),
            "full_4x4_block_area_inside_air_circle_fraction": block_visible_fraction,
            "complexity_weighted_area_inside_air_circle_fraction": complexity_weighted_visible_fraction,
            "top4_complexity_centers_propagating": sum(
                item["propagating"]
                for item in sorted(
                    order_rows, key=lambda row: row["complexity"], reverse=True
                )[:4]
            ),
            "nominal_4x4_inscribed_circle_radius": nominal_radius,
            "air_clipped_circular_fov_deg": 2
            * math.degrees(math.asin(min(1.0, nominal_radius))),
            "fully_visible_radius_fraction": min(1.0, 1 / nominal_radius),
        },
        "recommended_orders": order_rows,
        "recommended_orders_P2000_theory": theory_order_rows,
        "top5_symmetry_classes": representatives[:5],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
