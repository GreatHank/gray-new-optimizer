import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER_23_ORDERS = np.asarray(
    [(m, n) for n in range(3, -4, -1) for m in (-3, -2, -1)]
    + [(0, 2), (0, 1)],
    dtype=np.int16,
)


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="搜索连续、无(0,0)的6x6级次方阵并按图块倍频负担排名。"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=root / "output/circular_7x7_target_49/circular_scene_3500.png",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-start-min", type=int, default=-12)
    parser.add_argument("--search-start-max", type=int, default=12)
    parser.add_argument("--top-count", type=int, default=5)
    return parser.parse_args()


def transform_indices(size):
    base = np.arange(size * size).reshape(size, size)
    return {
        "identity": base,
        "rot90_ccw": np.rot90(base, 1),
        "rot180": np.rot90(base, 2),
        "rot270_ccw": np.rot90(base, 3),
        "mirror_lr": np.fliplr(base),
        "mirror_ud": np.flipud(base),
        "transpose": base.T,
        "anti_transpose": np.fliplr(np.flipud(base)).T,
    }


def tile_complexity(gray, size=6):
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
            tv_sum = np.abs(np.diff(tile, axis=0)).sum() + np.abs(
                np.diff(tile, axis=1)
            ).sum()
            edge_tv[row, column] = float(tv_sum / tile.size)
    density_normalized = density / density.max()
    tv_normalized = edge_tv / edge_tv.max()
    complexity = 0.5 * density_normalized + 0.5 * tv_normalized
    return density, edge_tv, density_normalized, tv_normalized, complexity


def primitive_direction(m, n):
    divisor = math.gcd(abs(int(m)), abs(int(n)))
    if divisor == 0:
        return 0, 0
    return int(m) // divisor, int(n) // divisor


def order_grid(m_start, n_start, size=6):
    return np.asarray(
        [[(m_start + row, n_start + column) for column in range(size)] for row in range(size)],
        dtype=np.int16,
    )


def group_metadata(orders):
    groups = defaultdict(list)
    for row in range(orders.shape[0]):
        for column in range(orders.shape[1]):
            m, n = map(int, orders[row, column])
            groups[primitive_direction(m, n)].append((row, column, m, n))
    sizes = np.zeros(orders.shape[:2], dtype=np.int16)
    descriptions = np.empty(orders.shape[:2], dtype=object)
    for members in groups.values():
        text = ";".join(f"({m},{n})" for _r, _c, m, n in members)
        for row, column, _m, _n in members:
            sizes[row, column] = len(members)
            descriptions[row, column] = text
    return groups, sizes, descriptions


def conjugate_pair_count(orders):
    flat = {tuple(map(int, item)) for item in orders.reshape(-1, 2)}
    return sum((m, n) != (0, 0) and (-m, -n) in flat for m, n in flat) // 2


def centered_incidence(m_start, n_start, wavelength_nm, period_nm):
    spacing = wavelength_nm / period_nm
    shift = -spacing * np.asarray([m_start + 2.5, n_start + 2.5], dtype=float)
    return shift


def propagation(orders, shift, wavelength_nm, period_nm):
    points = orders.reshape(-1, 2).astype(float) * (wavelength_nm / period_nm)
    output = points + np.asarray(shift, dtype=float)
    radii = np.linalg.norm(output, axis=1)
    captured = radii <= 1.0 + 1e-12
    return output, radii, captured


def incidence_angles(shift):
    radius = float(np.linalg.norm(shift))
    if radius > 1 + 1e-10:
        return float("nan"), float("nan")
    theta = math.degrees(math.asin(min(1.0, radius)))
    phi = math.degrees(math.atan2(float(shift[1]), float(shift[0])))
    return theta, phi


def best_common_incidence(orders, wavelength_nm, period_nm):
    centered = centered_incidence(
        int(orders[0, 0, 0]), int(orders[0, 0, 1]), wavelength_nm, period_nm
    )
    if np.linalg.norm(centered) <= 1 + 1e-12:
        _out, radii, captured = propagation(orders, centered, wavelength_nm, period_nm)
        if captured.all():
            theta, phi = incidence_angles(centered)
            return {
                "shift_x": float(centered[0]),
                "shift_y": float(centered[1]),
                "theta_i_deg": theta,
                "phi_i_deg": phi,
                "propagating_count": int(captured.sum()),
                "minimum_captured_margin": float(1 - radii.max()),
            }

    projected = centered / max(1.0, float(np.linalg.norm(centered)))
    _out, radii, captured = propagation(
        orders, projected, wavelength_nm, period_nm
    )
    if captured.all():
        theta, phi = incidence_angles(projected)
        return {
            "shift_x": float(projected[0]),
            "shift_y": float(projected[1]),
            "theta_i_deg": theta,
            "phi_i_deg": phi,
            "propagating_count": int(captured.sum()),
            "minimum_captured_margin": float(1 - radii.max()),
        }

    points = orders.reshape(-1, 2).astype(float) * (wavelength_nm / period_nm)

    def choose(shifts):
        best = None
        chunk_size = 20000
        for start in range(0, len(shifts), chunk_size):
            block = shifts[start : start + chunk_size]
            radii = np.linalg.norm(block[:, None, :] + points[None, :, :], axis=2)
            counts = (radii <= 1 + 1e-12).sum(axis=1)
            sorted_radii = np.sort(radii, axis=1)
            kth = sorted_radii[np.arange(len(block)), np.maximum(counts - 1, 0)]
            means = np.asarray(
                [sorted_radii[i, : max(int(counts[i]), 1)].mean() for i in range(len(block))]
            )
            for index in range(len(block)):
                key = (int(counts[index]), -float(kth[index]), -float(means[index]))
                if best is None or key > best[0]:
                    best = (key, block[index].copy())
        return best

    axis = np.linspace(-1, 1, 201)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coarse = np.column_stack((xx.ravel(), yy.ravel()))
    coarse = coarse[np.linalg.norm(coarse, axis=1) <= 1 + 1e-12]
    coarse_best = choose(coarse)
    center = coarse_best[1]
    fine_axis = np.linspace(-0.02, 0.02, 161)
    fx, fy = np.meshgrid(fine_axis, fine_axis, indexing="xy")
    fine = center + np.column_stack((fx.ravel(), fy.ravel()))
    fine = fine[np.linalg.norm(fine, axis=1) <= 1 + 1e-12]
    best = choose(fine)
    shift = best[1]
    _out, radii, captured = propagation(orders, shift, wavelength_nm, period_nm)
    theta, phi = incidence_angles(shift)
    return {
        "shift_x": float(shift[0]),
        "shift_y": float(shift[1]),
        "theta_i_deg": theta,
        "phi_i_deg": phi,
        "propagating_count": int(captured.sum()),
        "minimum_captured_margin": float(1 - radii[captured].max()),
    }


def evaluate_mapping(complexity, top12_mask, top4_mask, orders, index_map, transform_name):
    groups, group_sizes, descriptions = group_metadata(orders)
    assigned = complexity.ravel()[index_map]
    assigned_top12 = top12_mask.ravel()[index_map]
    assigned_top4 = top4_mask.ravel()[index_map]
    partners = group_sizes - 1
    axes = (orders[:, :, 0] == 0) | (orders[:, :, 1] == 0)
    weighted_burden = float(np.sum(assigned * partners))
    axis_burden = float(np.sum(assigned[axes] * partners[axes]))
    top12_burden = float(np.sum(assigned[assigned_top12] * partners[assigned_top12]))
    score = weighted_burden + 3.0 * axis_burden + top12_burden
    return {
        "transform": transform_name,
        "score": score,
        "weighted_multiple_burden": weighted_burden,
        "axis_weighted_burden": axis_burden,
        "top12_weighted_burden": top12_burden,
        "top12_without_multiple": int(np.sum(assigned_top12 & (group_sizes == 1))),
        "top4_without_multiple": int(np.sum(assigned_top4 & (group_sizes == 1))),
        "axis_tiles": int(axes.sum()),
        "axis_tiles_group5_or6": int(np.sum(axes & (group_sizes >= 5))),
        "high_complex_axis_group5_or6": int(
            np.sum(axes & assigned_top12 & (group_sizes >= 5))
        ),
        "group_count": len(groups),
        "coupled_group_count": sum(len(items) > 1 for items in groups.values()),
        "largest_group_size": max(len(items) for items in groups.values()),
        "assigned_complexity": assigned,
        "assigned_top12": assigned_top12,
        "assigned_top4": assigned_top4,
        "group_sizes": group_sizes,
        "group_descriptions": descriptions,
        "index_map": index_map,
    }


def candidate_rows(complexity, search_min, search_max):
    size = 6
    flat_order = np.argsort(complexity.ravel())[::-1]
    top12_mask = np.zeros(size * size, dtype=bool)
    top12_mask[flat_order[:12]] = True
    top12_mask = top12_mask.reshape(size, size)
    top4_mask = np.zeros(size * size, dtype=bool)
    top4_mask[flat_order[:4]] = True
    top4_mask = top4_mask.reshape(size, size)
    transformations = transform_indices(size)
    rows = []
    details = {}
    for m_start in range(search_min, search_max + 1):
        for n_start in range(search_min, search_max + 1):
            orders = order_grid(m_start, n_start, size)
            if np.any(np.all(orders == 0, axis=2)):
                continue
            theory_shift = centered_incidence(m_start, n_start, 480.0, 2000.0)
            theory_shift = theory_shift / max(1.0, float(np.linalg.norm(theory_shift)))
            _out, _radii, captured = propagation(orders, theory_shift, 480.0, 2000.0)
            if not captured.all():
                continue
            options = [
                evaluate_mapping(complexity, top12_mask, top4_mask, orders, index_map, name)
                for name, index_map in transformations.items()
            ]
            best = min(
                options,
                key=lambda item: (
                    item["score"],
                    -item["top12_without_multiple"],
                    -item["top4_without_multiple"],
                ),
            )
            row = {
                "m_start": m_start,
                "m_end": m_start + 5,
                "n_start": n_start,
                "n_end": n_start + 5,
                "zero_order_count": 0,
                "conjugate_pair_count": conjugate_pair_count(orders),
                "transform": best["transform"],
                "score": best["score"],
                "weighted_multiple_burden": best["weighted_multiple_burden"],
                "axis_weighted_burden": best["axis_weighted_burden"],
                "top12_weighted_burden": best["top12_weighted_burden"],
                "top12_without_multiple": best["top12_without_multiple"],
                "top4_without_multiple": best["top4_without_multiple"],
                "coupled_group_count": best["coupled_group_count"],
                "largest_group_size": best["largest_group_size"],
                "high_complex_axis_group5_or6": best[
                    "high_complex_axis_group5_or6"
                ],
            }
            rows.append(row)
            details[(m_start, n_start)] = (orders, best)
    rows.sort(
        key=lambda item: (
            item["score"],
            -item["top12_without_multiple"],
            -item["top4_without_multiple"],
            item["weighted_multiple_burden"],
            abs(item["m_start"] + 2.5) + abs(item["n_start"] + 2.5),
        )
    )
    return rows, details, flat_order


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_complexity(gray, density_n, tv_n, complexity, top_order, output_file):
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    height, width = gray.shape
    for index in range(1, 6):
        axes[0].axhline(index * height / 6, color="tab:red", linewidth=0.8)
        axes[0].axvline(index * width / 6, color="tab:red", linewidth=0.8)
    rank = {int(flat): position + 1 for position, flat in enumerate(top_order[:12])}
    for row in range(6):
        for column in range(6):
            flat = row * 6 + column
            label = f"C={complexity[row, column]:.3f}"
            if flat in rank:
                label += f"\nTOP {rank[flat]}"
            axes[0].text(
                (column + 0.5) * width / 6,
                (row + 0.5) * height / 6,
                label,
                ha="center",
                va="center",
                color="yellow",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
            )
    axes[0].axis("off")
    axes[0].set_title("6x6 image complexity (equal-weight density + edge TV)")
    heat = axes[1].imshow(complexity, cmap="magma", vmin=0, vmax=1)
    for row in range(6):
        for column in range(6):
            axes[1].text(
                column,
                row,
                f"D {density_n[row,column]:.2f}\nTV {tv_n[row,column]:.2f}\nC {complexity[row,column]:.2f}",
                ha="center",
                va="center",
                color="white" if complexity[row, column] < 0.7 else "black",
                fontsize=8,
            )
    axes[1].set_xticks(range(6), labels=range(1, 7))
    axes[1].set_yticks(range(6), labels=range(1, 7))
    axes[1].set_xlabel("source tile column")
    axes[1].set_ylabel("source tile row")
    axes[1].set_title("Normalized components")
    figure.colorbar(heat, ax=axes[1], fraction=0.046)
    figure.tight_layout()
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_candidates(top_rows, details, output_file):
    figure, axes = plt.subplots(1, len(top_rows), figsize=(4.4 * len(top_rows), 5.8), squeeze=False)
    for rank, row in enumerate(top_rows, start=1):
        axis = axes[0, rank - 1]
        orders, detail = details[(row["m_start"], row["n_start"])]
        for grid_row in range(6):
            for grid_column in range(6):
                m, n = map(int, orders[grid_row, grid_column])
                source_flat = int(detail["index_map"][grid_row, grid_column])
                source_row, source_column = divmod(source_flat, 6)
                group_size = int(detail["group_sizes"][grid_row, grid_column])
                color = plt.cm.RdYlGn_r((group_size - 1) / 5)
                rectangle = plt.Rectangle(
                    (grid_column, grid_row),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="tab:blue" if detail["assigned_top4"][grid_row, grid_column] else "black",
                    linewidth=3 if detail["assigned_top4"][grid_row, grid_column] else (2 if detail["assigned_top12"][grid_row, grid_column] else 0.6),
                )
                axis.add_patch(rectangle)
                axis.text(grid_column + 0.5, grid_row + 0.30, f"({m},{n})", ha="center", fontsize=8)
                axis.text(grid_column + 0.5, grid_row + 0.58, f"src {source_row+1},{source_column+1}", ha="center", fontsize=7)
                axis.text(grid_column + 0.5, grid_row + 0.82, f"group {group_size}", ha="center", fontsize=7)
        axis.set_xlim(0, 6)
        axis.set_ylim(6, 0)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"#{rank} m={row['m_start']}..{row['m_end']}\nn={row['n_start']}..{row['n_end']}\n{row['transform']} | top12 free {row['top12_without_multiple']}/12",
            fontsize=10,
        )
    figure.suptitle(
        "Top 5 continuous zero-free 6x6 mappings (blue: top 4; thick black: top 12)",
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)


def tile_rows(density, edge_tv, density_n, tv_n, complexity, top_order):
    ranks = {int(flat): rank + 1 for rank, flat in enumerate(top_order)}
    rows = []
    for row in range(6):
        for column in range(6):
            flat = row * 6 + column
            rows.append(
                {
                    "grid_row": row + 1,
                    "grid_column": column + 1,
                    "bright_line_density": float(density[row, column]),
                    "edge_total_variation_per_pixel": float(edge_tv[row, column]),
                    "density_normalized": float(density_n[row, column]),
                    "edge_tv_normalized": float(tv_n[row, column]),
                    "complexity": float(complexity[row, column]),
                    "complexity_rank": ranks[flat],
                    "top12": ranks[flat] <= 12,
                }
            )
    return rows


def mapped_order_rows(rank, candidate, details, complexity):
    orders, detail = details[(candidate["m_start"], candidate["n_start"])]
    rows = []
    for grid_row in range(6):
        for grid_column in range(6):
            source_flat = int(detail["index_map"][grid_row, grid_column])
            source_row, source_column = divmod(source_flat, 6)
            m, n = map(int, orders[grid_row, grid_column])
            rows.append(
                {
                    "candidate_rank": rank,
                    "m_start": candidate["m_start"],
                    "n_start": candidate["n_start"],
                    "transform": candidate["transform"],
                    "order_grid_row": grid_row + 1,
                    "order_grid_column": grid_column + 1,
                    "source_grid_row": source_row + 1,
                    "source_grid_column": source_column + 1,
                    "m": m,
                    "n": n,
                    "complexity": float(complexity[source_row, source_column]),
                    "top12": bool(detail["assigned_top12"][grid_row, grid_column]),
                    "top4": bool(detail["assigned_top4"][grid_row, grid_column]),
                    "primitive_m": primitive_direction(m, n)[0],
                    "primitive_n": primitive_direction(m, n)[1],
                    "multiple_group_size": int(detail["group_sizes"][grid_row, grid_column]),
                    "multiple_coupling_degree": int(detail["group_sizes"][grid_row, grid_column] - 1),
                    "multiple_group": detail["group_descriptions"][grid_row, grid_column],
                    "on_axis": m == 0 or n == 0,
                }
            )
    return rows


def circular_fov(wavelength_nm, period_nm):
    nominal_radius = 3 * wavelength_nm / period_nm
    return {
        "nominal_direction_cosine_radius": nominal_radius,
        "air_clipped_direction_cosine_radius": min(1.0, nominal_radius),
        "circular_fov_deg": 2 * math.degrees(math.asin(min(1.0, nominal_radius))),
        "fully_supported_by_6x6_centers": nominal_radius <= 1,
    }


def symmetry_class(m_start, n_start):
    canonical_m = min(int(m_start), -int(m_start) - 5)
    canonical_n = min(int(n_start), -int(n_start) - 5)
    first, second = sorted((canonical_m, canonical_n))
    return f"{first},{second}"


def paper23_summary():
    orders = PAPER_23_ORDERS.reshape(1, -1, 2)
    searched = best_common_incidence(orders, 488.0, 2000.0)
    theta = math.radians(46.0)
    published_shift = np.asarray([math.sin(theta), 0.0])
    output, radii, captured = propagation(orders, published_shift, 488.0, 2000.0)
    angles = {}
    flat_orders = orders.reshape(-1, 2)
    for target in [(-3, 3), (-2, 0), (-1, 0)]:
        index = np.flatnonzero(np.all(flat_orders == target, axis=1))[0]
        angles[f"{target[0]},{target[1]}"] = math.degrees(math.asin(float(radii[index])))
    return {
        "orders": flat_orders.tolist(),
        "zero_order_count": int(np.sum(np.all(flat_orders == 0, axis=1))),
        "conjugate_pair_count": conjugate_pair_count(orders),
        "published_condition": {
            "wavelength_nm": 488.0,
            "period_nm": 2000.0,
            "theta_i_deg": 46.0,
            "phi_i_deg": 0.0,
            "propagating_count": int(captured.sum()),
            "selected_output_theta_deg": angles,
        },
        "searched_common_incidence": searched,
    }


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{args.output_dir}")
    if args.search_start_min > args.search_start_max:
        raise ValueError("search-start-min不能大于search-start-max。")
    with Image.open(args.input_file) as image:
        gray = np.asarray(image.convert("L"))
    density, edge_tv, density_n, tv_n, complexity = tile_complexity(gray)
    ranking, details, top_order = candidate_rows(
        complexity, args.search_start_min, args.search_start_max
    )
    if len(ranking) < args.top_count:
        raise RuntimeError("满足P=2000 nm下36/36传播的无零级候选不足。")
    for candidate in ranking:
        orders, _detail = details[(candidate["m_start"], candidate["n_start"])]
        for period in (850.0, 2000.0):
            physical = best_common_incidence(orders, 480.0, period)
            suffix = f"p{int(period)}"
            for key, value in physical.items():
                candidate[f"{suffix}_{key}"] = value
            fov = circular_fov(480.0, period)
            candidate[f"{suffix}_circular_fov_deg"] = fov["circular_fov_deg"]
            candidate[f"{suffix}_nominal_radius"] = fov[
                "nominal_direction_cosine_radius"
            ]
        candidate["symmetry_class"] = symmetry_class(
            candidate["m_start"], candidate["n_start"]
        )

    ranking.sort(
        key=lambda item: (
            -item["p850_propagating_count"],
            item["score"],
            item["p850_theta_i_deg"],
            -item["top4_without_multiple"],
        )
    )
    top = []
    seen_classes = set()
    for candidate in ranking:
        if candidate["symmetry_class"] in seen_classes:
            continue
        seen_classes.add(candidate["symmetry_class"])
        top.append(candidate)
        if len(top) == args.top_count:
            break
    if len(top) < args.top_count:
        raise RuntimeError("对称归并后候选类别不足。")

    args.output_dir.mkdir(parents=True)
    tiles = tile_rows(density, edge_tv, density_n, tv_n, complexity, top_order)
    write_csv(args.output_dir / "tile_complexity.csv", tiles)
    write_csv(args.output_dir / "all_candidate_ranking.csv", ranking)
    write_csv(args.output_dir / "top5_candidates.csv", top)
    mapped = []
    for rank, candidate in enumerate(top, start=1):
        mapped.extend(mapped_order_rows(rank, candidate, details, complexity))
    write_csv(args.output_dir / "top5_order_tile_metrics.csv", mapped)
    plot_complexity(
        gray,
        density_n,
        tv_n,
        complexity,
        top_order,
        args.output_dir / "tile_complexity_map.png",
    )
    plot_candidates(top, details, args.output_dir / "top5_candidate_maps.png")

    top12 = sorted((item for item in tiles if item["top12"]), key=lambda item: item["complexity_rank"])
    manual_review = []
    for rank, candidate in enumerate(top[:2], start=1):
        candidate_rows_mapped = [item for item in mapped if item["candidate_rank"] == rank]
        top4 = sorted(
            (item for item in candidate_rows_mapped if item["top4"]),
            key=lambda item: -item["complexity"],
        )
        risky_axis = [
            item
            for item in candidate_rows_mapped
            if item["top12"] and item["on_axis"] and item["multiple_group_size"] >= 5
        ]
        manual_review.append(
            {
                "candidate_rank": rank,
                "m_range": [candidate["m_start"], candidate["m_end"]],
                "n_range": [candidate["n_start"], candidate["n_end"]],
                "transform": candidate["transform"],
                "top4": [
                    {
                        "source_tile": [item["source_grid_row"], item["source_grid_column"]],
                        "order": [item["m"], item["n"]],
                        "group_size": item["multiple_group_size"],
                        "free_of_multiple": item["multiple_group_size"] == 1,
                    }
                    for item in top4
                ],
                "high_complexity_in_axis_group5_or6": [
                    {
                        "source_tile": [item["source_grid_row"], item["source_grid_column"]],
                        "order": [item["m"], item["n"]],
                        "complexity": item["complexity"],
                        "group_size": item["multiple_group_size"],
                    }
                    for item in risky_axis
                ],
            }
        )
    summary = {
        "input_file": str(args.input_file.resolve()),
        "complexity_definition": "0.5*(bright-line density/max)+0.5*(edge TV per pixel/max)",
        "candidate_domain": {
            "enumerated_starts": [args.search_start_min, args.search_start_max],
            "hard_zero_order_exclusion": True,
            "reasonable_candidate_filter": "lambda=480 nm, P=2000 nm theory: 36/36 centers propagating under a centered or incident-disk-boundary common free-space incidence",
            "retained_candidate_count": len(ranking),
        },
        "ranking_score": "first maximize P=850 nm propagating centers, then minimize weighted burden + 3*axis weighted burden + top-12 weighted burden; D4-symmetric ranges are one class",
        "top12_tiles": top12,
        "top5_candidates": top,
        "manual_review_top2": manual_review,
        "paper23": paper23_summary(),
        "fov_definition": "2*asin(min(1,3*lambda/P)); at P=850 the 180-deg value is air-cone clipping, not full 36-center support",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
