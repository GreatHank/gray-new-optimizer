import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import scipy.io as sio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PAPER_23_ORDERS = np.asarray(
    [(m, n) for n in range(3, -4, -1) for m in (-3, -2, -1)]
    + [(0, 2), (0, 1)],
    dtype=np.int16,
)


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="用二维光栅方程验证连续衍射级次的传播与收集可行性。"
    )
    parser.add_argument("--wavelengths-nm", nargs="+", type=float, default=[480.0])
    parser.add_argument(
        "--mode",
        choices=("fixed-incidence", "center-each-order"),
        default="fixed-incidence",
        help=(
            "fixed-incidence按共同入射条件检查大视场中的出射级次；"
            "center-each-order为每个级次分别反算垂直出射所需入射角"
        ),
    )
    parser.add_argument(
        "--order-preset",
        choices=("continuous", "paper-23"),
        default="continuous",
        help="continuous使用连续方阵；paper-23使用论文补充材料Figure S5的23级次",
    )
    parser.add_argument(
        "--conjugate-encoding",
        choices=("detour-only", "detour-pb"),
        default="detour-only",
        help=(
            "detour-only按当前dx/dy模型处理共轭关系；"
            "detour-pb表示另有PB相位自由度可独立编码共轭级次"
        ),
    )
    parser.add_argument("--period-x-nm", type=float, default=850.0)
    parser.add_argument("--period-y-nm", type=float, default=850.0)
    parser.add_argument(
        "--incident-theta-deg",
        nargs="+",
        type=float,
        default=[0.0],
        help="相对表面法线的入射极角；可给一个公共值或每个波长一个值",
    )
    parser.add_argument(
        "--incident-azimuth-deg",
        nargs="+",
        type=float,
        default=[0.0],
        help="入射方位角；可给一个公共值或每个波长一个值",
    )
    parser.add_argument("--n-in", type=float, default=1.0)
    parser.add_argument("--n-out", type=float, default=1.0)
    parser.add_argument(
        "--na",
        type=float,
        default=1.0,
        help="输出侧收集数值孔径；空气中最大为1",
    )
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--order-m-start", type=int, default=-9)
    parser.add_argument("--order-n-start", type=int, default=3)
    parser.add_argument(
        "--target-mat",
        type=Path,
        default=project_root
        / "output"
        / "circular_7x7_target_49"
        / "circular_7x7_target.mat",
        help="可选MAT；存在grid_positions时只验证实际有效channel",
    )
    parser.add_argument(
        "--all-grid-positions",
        action="store_true",
        help="忽略target-mat，验证完整grid-size×grid-size级次网格",
    )
    parser.add_argument("--search-start-min", type=int, default=-12)
    parser.add_argument("--search-start-max", type=int, default=12)
    parser.add_argument("--ranking-limit", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def broadcast_parameter(values, count, name):
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.repeat(values, count)
    if values.size != count:
        raise ValueError(f"{name}必须给1个值或与波长数量相同的{count}个值。")
    return values


def load_grid_positions(target_mat, grid_size):
    if target_mat is None:
        return np.asarray(
            [divmod(index, grid_size) for index in range(grid_size**2)],
            dtype=np.int16,
        )
    if not target_mat.is_file():
        raise FileNotFoundError(f"目标MAT不存在：{target_mat}")
    data = sio.loadmat(target_mat)
    if "grid_positions" not in data:
        if "bw_all" not in data:
            raise KeyError("目标MAT必须包含grid_positions或bw_all。")
        count = int(np.asarray(data["bw_all"]).shape[0])
        return np.asarray(
            [divmod(index, grid_size) for index in range(count)], dtype=np.int16
        )
    positions = np.asarray(data["grid_positions"], dtype=np.int16)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("grid_positions必须为N×2数组。")
    if np.any(positions < 0) or np.any(positions >= grid_size):
        raise ValueError("grid_positions超出连续级次网格范围。")
    return positions


def order_directions(
    orders,
    wavelength_nm,
    period_x_nm,
    period_y_nm,
    theta_deg,
    azimuth_deg,
    n_in,
    n_out,
    numerical_aperture,
):
    theta = np.deg2rad(theta_deg)
    azimuth = np.deg2rad(azimuth_deg)
    incident_x = n_in * np.sin(theta) * np.cos(azimuth)
    incident_y = n_in * np.sin(theta) * np.sin(azimuth)
    transverse_x = incident_x + orders[:, 0] * wavelength_nm / period_x_nm
    transverse_y = incident_y + orders[:, 1] * wavelength_nm / period_y_nm
    ux = transverse_x / n_out
    uy = transverse_y / n_out
    radius = np.hypot(ux, uy)
    propagating = radius <= 1.0 + 1e-12
    captured = propagating & (np.hypot(transverse_x, transverse_y) <= numerical_aperture + 1e-12)
    theta_out = np.full(radius.shape, np.nan)
    theta_out[propagating] = np.rad2deg(np.arcsin(np.clip(radius[propagating], 0, 1)))
    azimuth_out = np.rad2deg(np.arctan2(uy, ux))
    return {
        "ux": ux,
        "uy": uy,
        "radius": radius,
        "propagating": propagating,
        "captured": captured,
        "air_margin": 1.0 - radius,
        "theta_out_deg": theta_out,
        "azimuth_out_deg": azimuth_out,
    }


def center_incidence_for_orders(
    orders,
    wavelength_nm,
    period_x_nm,
    period_y_nm,
    n_in,
):
    """反算让每个级次分别垂直出射所需的自由空间入射方向。"""
    required_x = -orders[:, 0] * wavelength_nm / period_x_nm
    required_y = -orders[:, 1] * wavelength_nm / period_y_nm
    transverse_norm = np.hypot(required_x, required_y)
    feasible = transverse_norm <= n_in + 1e-12
    theta = np.full(transverse_norm.shape, np.nan)
    theta[feasible] = np.rad2deg(
        np.arcsin(np.clip(transverse_norm[feasible] / n_in, 0, 1))
    )
    azimuth = np.rad2deg(np.arctan2(required_y, required_x))
    return {
        "incident_transverse_norm": transverse_norm,
        "center_feasible": feasible,
        "incident_theta_deg": theta,
        "incident_azimuth_deg": azimuth,
    }


def primitive_order(order):
    m, n = (int(order[0]), int(order[1]))
    divisor = int(np.gcd(abs(m), abs(n)))
    if divisor == 0:
        return 0, 0, 0
    primitive_m, primitive_n = m // divisor, n // divisor
    if primitive_m < 0 or (primitive_m == 0 and primitive_n < 0):
        primitive_m, primitive_n = -primitive_m, -primitive_n
        divisor = -divisor
    return primitive_m, primitive_n, divisor


def phase_relation_rows(orders, conjugate_encoding="detour-only"):
    """报告严格相位关系；运动学可行并不等于已实现独立解耦。"""
    order_set = {tuple(map(int, order)) for order in orders}
    rows = []
    for order in orders:
        m, n = map(int, order)
        primitive_m, primitive_n, harmonic_multiplier = primitive_order(order)
        conjugate = (-m, -n)
        phase_controllable = (m, n) != (0, 0) or conjugate_encoding == "detour-pb"
        conjugate_in_selection = (m, n) != (0, 0) and conjugate in order_set
        canonical_half_plane = m > 0 or (m == 0 and n > 0)
        independent_representative = phase_controllable and (
            conjugate_encoding == "detour-pb"
            or not conjugate_in_selection
            or canonical_half_plane
        )
        rows.append(
            {
                "m": m,
                "n": n,
                "phase_controllable": phase_controllable,
                "conjugate_order_in_selection": conjugate_in_selection,
                "conjugate_m": conjugate[0],
                "conjugate_n": conjugate[1],
                "canonical_half_plane": canonical_half_plane,
                "independent_conjugate_representative": independent_representative,
                "primitive_m": primitive_m,
                "primitive_n": primitive_n,
                "harmonic_multiplier": harmonic_multiplier,
                "is_primitive": abs(harmonic_multiplier) == 1,
            }
        )
    return rows


def relation_counts(orders):
    collinear = 0
    for first, second in combinations(orders, 2):
        collinear += int(first[0] * second[1] == first[1] * second[0])

    order_set = {tuple(order) for order in orders}
    additive = 0
    for first_index, first in enumerate(orders):
        for second in orders[first_index:]:
            additive += int(tuple(first + second) in order_set)
    return collinear, additive


def minimum_separation(ux, uy):
    if len(ux) < 2:
        return np.nan
    points = np.column_stack((ux, uy))
    differences = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def evaluate_block(
    positions,
    m_start,
    n_start,
    wavelengths,
    theta_values,
    azimuth_values,
    args,
):
    orders = positions + np.asarray([m_start, n_start], dtype=np.int16)
    zero_order_count = int(np.count_nonzero(np.all(orders == 0, axis=1)))
    collinear, additive = relation_counts(orders)
    propagation_count = 0
    captured_count = 0
    margins = []
    separations = []
    for wavelength, theta, azimuth in zip(wavelengths, theta_values, azimuth_values):
        result = order_directions(
            orders,
            wavelength,
            args.period_x_nm,
            args.period_y_nm,
            theta,
            azimuth,
            args.n_in,
            args.n_out,
            args.na,
        )
        propagation_count += int(np.count_nonzero(result["propagating"]))
        captured_count += int(np.count_nonzero(result["captured"]))
        margins.extend(result["air_margin"].tolist())
        separations.append(minimum_separation(result["ux"], result["uy"]))
    return {
        "m_start": m_start,
        "m_end": m_start + args.grid_size - 1,
        "n_start": n_start,
        "n_end": n_start + args.grid_size - 1,
        "propagating_count": propagation_count,
        "captured_count": captured_count,
        "total_tests": len(orders) * len(wavelengths),
        "zero_order_count": zero_order_count,
        "minimum_air_margin": float(np.min(margins)),
        "minimum_same_wavelength_separation": float(np.min(separations)),
        "collinear_pair_count": collinear,
        "additive_relation_count": additive,
    }


def rank_blocks(positions, wavelengths, theta_values, azimuth_values, args):
    rows = []
    for m_start in range(args.search_start_min, args.search_start_max + 1):
        for n_start in range(args.search_start_min, args.search_start_max + 1):
            row = evaluate_block(
                positions,
                m_start,
                n_start,
                wavelengths,
                theta_values,
                azimuth_values,
                args,
            )
            if row is not None:
                rows.append(row)
    rows.sort(
        key=lambda row: (
            -row["captured_count"],
            -row["propagating_count"],
            row["zero_order_count"],
            -row["minimum_air_margin"],
            row["collinear_pair_count"],
            row["additive_relation_count"],
        )
    )
    return rows


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_order_map(
    output_file,
    positions,
    orders,
    wavelengths,
    theta_values,
    azimuth_values,
    args,
):
    figure, axes = plt.subplots(
        1, len(wavelengths), figsize=(7 * len(wavelengths), 6.4), squeeze=False
    )
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:green", label="Independent conjugate representative"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.7", label="Captured conjugate partner"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:orange", label="Selected: propagating outside NA"),
        Line2D([0], [0], marker="x", color="tab:red", label="Evanescent or uncontrollable zero order"),
    ]
    relation_by_order = {
        (row["m"], row["n"]): row
        for row in phase_relation_rows(orders, args.conjugate_encoding)
    }
    display_min = args.search_start_min
    display_max = args.search_start_max + args.grid_size - 1
    display_orders = np.asarray(
        [
            (m, n)
            for m in range(display_min, display_max + 1)
            for n in range(display_min, display_max + 1)
            if (m, n) != (0, 0)
        ],
        dtype=np.int16,
    )

    for axis, wavelength, theta, azimuth in zip(
        axes[0], wavelengths, theta_values, azimuth_values
    ):
        background = order_directions(
            display_orders,
            wavelength,
            args.period_x_nm,
            args.period_y_nm,
            theta,
            azimuth,
            args.n_in,
            args.n_out,
            args.na,
        )
        inside = background["propagating"]
        axis.scatter(
            background["ux"][inside],
            background["uy"][inside],
            s=9,
            color="0.75",
            zorder=1,
        )

        selected = order_directions(
            orders,
            wavelength,
            args.period_x_nm,
            args.period_y_nm,
            theta,
            azimuth,
            args.n_in,
            args.n_out,
            args.na,
        )
        for order, ux, uy, propagating, captured in zip(
            orders,
            selected["ux"],
            selected["uy"],
            selected["propagating"],
            selected["captured"],
        ):
            relation = relation_by_order[tuple(map(int, order))]
            if not relation["phase_controllable"]:
                color, marker = "tab:red", "x"
            elif captured and relation["independent_conjugate_representative"]:
                color, marker = "tab:green", "o"
            elif captured:
                color, marker = "0.7", "o"
            elif propagating:
                color, marker = "tab:orange", "o"
            else:
                color, marker = "tab:red", "x"
            axis.scatter(ux, uy, s=28, color=color, marker=marker, zorder=3)
            axis.annotate(
                f"({order[0]},{order[1]})",
                (ux, uy),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
                color=color,
            )

        air_circle = plt.Circle((0, 0), 1, fill=False, linestyle="--", color="black")
        axis.add_patch(air_circle)
        na_radius = args.na / args.n_out
        if na_radius < 1 - 1e-12:
            axis.add_patch(
                plt.Circle((0, 0), na_radius, fill=False, color="tab:blue", linewidth=1.5)
            )
        selected_limit = max(
            1.12,
            float(np.max(np.abs(selected["ux"]))) * 1.08,
            float(np.max(np.abs(selected["uy"]))) * 1.08,
        )
        axis.set_xlim(-selected_limit, selected_limit)
        axis.set_ylim(-selected_limit, selected_limit)
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)
        axis.set_xlabel("u_x")
        axis.set_ylabel("u_y")
        axis.set_title(
            f"{wavelength:g} nm, incidence theta={theta:g} deg, phi={azimuth:g} deg\n"
            f"propagating {np.count_nonzero(selected['propagating'])}/{len(orders)}, "
            f"captured {np.count_nonzero(selected['captured'])}/{len(orders)}"
        )
        axis.legend(handles=legend, loc="best", fontsize=8)

    figure.suptitle(
        "2D diffraction-order feasibility\n"
        f"P=({args.period_x_nm:g}, {args.period_y_nm:g}) nm, "
        f"n_in={args.n_in:g}, n_out={args.n_out:g}, NA={args.na:g}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_file, dpi=180)
    plt.close(figure)


def plot_coupling_grid(
    output_file,
    positions,
    orders,
    wavelength,
    theta,
    azimuth,
    args,
):
    result = order_directions(
        orders,
        wavelength,
        args.period_x_nm,
        args.period_y_nm,
        theta,
        azimuth,
        args.n_in,
        args.n_out,
        args.na,
    )
    relation_rows = phase_relation_rows(orders, args.conjugate_encoding)
    figure, axis = plt.subplots(figsize=(10, 9))
    axis.set_xlim(0, args.grid_size)
    axis.set_ylim(args.grid_size, 0)
    axis.set_aspect("equal")
    axis.set_xticks(np.arange(args.grid_size) + 0.5)
    axis.set_yticks(np.arange(args.grid_size) + 0.5)
    axis.set_xticklabels(
        [str(args.order_n_start + index) for index in range(args.grid_size)]
    )
    axis.set_yticklabels(
        [str(args.order_m_start + index) for index in range(args.grid_size)]
    )
    axis.set_xlabel("n (grid column)")
    axis.set_ylabel("m (grid row)")

    colors = {
        "weak": "#79c67a",
        "conditional": "#f4d35e",
        "harmonic": "#f29e4c",
        "zero": "#e45756",
    }
    captured_by_order = {
        tuple(map(int, order)): bool(captured)
        for order, captured in zip(orders, result["captured"])
    }
    for index, (position, order, relation) in enumerate(
        zip(positions, orders, relation_rows)
    ):
        row, column = map(int, position)
        if not relation["phase_controllable"]:
            category, code = "zero", "Z"
        elif not relation["is_primitive"]:
            category, code = "harmonic", "H"
        elif captured_by_order.get(
            (-int(order[0]), -int(order[1])), False
        ):
            category, code = "conditional", "C"
        else:
            category, code = "weak", "W"
        captured = bool(result["captured"][index])
        rectangle = plt.Rectangle(
            (column, row),
            1,
            1,
            facecolor=colors[category],
            alpha=0.95 if captured else 0.28,
            edgecolor="black" if captured else "0.55",
            linewidth=2.2 if captured else 0.8,
            linestyle="-" if captured else "--",
        )
        axis.add_patch(rectangle)
        axis.text(
            column + 0.5,
            row + 0.39,
            f"({int(order[0])},{int(order[1])})",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold" if captured else "normal",
        )
        axis.text(
            column + 0.5,
            row + 0.70,
            f"{code}  r={result['radius'][index]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

    legend = [
        plt.Rectangle((0, 0), 1, 1, color=colors["weak"], label="W: primitive, conjugate partner not visible"),
        plt.Rectangle((0, 0), 1, 1, color=colors["conditional"], label="C: conjugate partner is also visible"),
        plt.Rectangle((0, 0), 1, 1, color=colors["harmonic"], label="H: integer harmonic / non-primitive"),
        plt.Rectangle((0, 0), 1, 1, color=colors["zero"], label="Z: uncontrollable zero order"),
        Line2D([0], [0], color="black", linewidth=2.2, label="solid border: center inside air cone"),
        Line2D([0], [0], color="0.55", linestyle="--", label="dashed/faded: center outside air cone"),
    ]
    axis.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    axis.set_title(
        f"{args.grid_size}x{args.grid_size} order coupling map\n"
        f"lambda={wavelength:g} nm, P={args.period_x_nm:g} nm, "
        f"theta={theta:g} deg, phi={azimuth:g} deg"
    )
    figure.tight_layout()
    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)


def validate_args(args):
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{args.output_dir}")
    if args.grid_size < 1:
        raise ValueError("grid-size必须大于0。")
    if args.period_x_nm <= 0 or args.period_y_nm <= 0:
        raise ValueError("二维周期必须大于0。")
    if any(wavelength <= 0 for wavelength in args.wavelengths_nm):
        raise ValueError("波长必须大于0。")
    if args.n_in <= 0 or args.n_out <= 0:
        raise ValueError("折射率必须大于0。")
    if not 0 < args.na <= args.n_out:
        raise ValueError("NA必须满足0 < NA <= n-out。")
    if args.search_start_min > args.search_start_max:
        raise ValueError("search-start-min不能大于search-start-max。")
    if args.ranking_limit < 1:
        raise ValueError("ranking-limit必须大于0。")
    if args.mode == "center-each-order" and len(args.wavelengths_nm) != 1:
        raise ValueError("center-each-order当前一次只接受一个波长。")


def main():
    args = parse_args()
    validate_args(args)
    wavelengths = np.asarray(args.wavelengths_nm, dtype=float)
    theta_values = broadcast_parameter(
        args.incident_theta_deg, len(wavelengths), "incident-theta-deg"
    )
    azimuth_values = broadcast_parameter(
        args.incident_azimuth_deg, len(wavelengths), "incident-azimuth-deg"
    )
    using_preset = args.order_preset == "paper-23"
    if using_preset:
        orders = PAPER_23_ORDERS.copy()
        positions = np.asarray(
            [divmod(index, args.grid_size) for index in range(len(orders))],
            dtype=np.int16,
        )
    else:
        positions = load_grid_positions(
            None if args.all_grid_positions else args.target_mat,
            args.grid_size,
        )
        orders = positions + np.asarray(
            [args.order_m_start, args.order_n_start], dtype=np.int16
        )
    args.output_dir.mkdir(parents=True)
    relation_by_order = {
        (row["m"], row["n"]): row
        for row in phase_relation_rows(orders, args.conjugate_encoding)
    }
    selected_rows = []
    for wavelength, theta, azimuth in zip(
        wavelengths, theta_values, azimuth_values
    ):
        result = order_directions(
            orders,
            wavelength,
            args.period_x_nm,
            args.period_y_nm,
            theta,
            azimuth,
            args.n_in,
            args.n_out,
            args.na,
        )
        for order, position, index in zip(orders, positions, range(len(orders))):
            relation = relation_by_order[tuple(map(int, order))]
            selected_rows.append(
                {
                    "wavelength_nm": wavelength,
                    "selection_index": index + 1,
                    "grid_row": "" if using_preset else int(position[0]) + 1,
                    "grid_column": "" if using_preset else int(position[1]) + 1,
                    "m": int(order[0]),
                    "n": int(order[1]),
                    "ux": result["ux"][index],
                    "uy": result["uy"][index],
                    "radius": result["radius"][index],
                    "propagating": bool(result["propagating"][index]),
                    "captured_by_na": bool(result["captured"][index]),
                    "air_margin": result["air_margin"][index],
                    "theta_out_deg": result["theta_out_deg"][index],
                    "azimuth_out_deg": result["azimuth_out_deg"][index],
                    "large_fov_candidate": bool(result["captured"][index])
                    and relation["independent_conjugate_representative"],
                    "conservative_primitive_candidate": bool(result["captured"][index])
                    and relation["independent_conjugate_representative"]
                    and relation["is_primitive"],
                    **{
                        key: value
                        for key, value in relation.items()
                        if key not in {"m", "n"}
                    },
                }
            )

    center_rows = []
    if args.mode == "center-each-order":
        wavelength = float(wavelengths[0])
        centered = center_incidence_for_orders(
            orders,
            wavelength,
            args.period_x_nm,
            args.period_y_nm,
            args.n_in,
        )
        for order, position, index in zip(orders, positions, range(len(orders))):
            relation = relation_by_order[tuple(map(int, order))]
            center_rows.append(
                {
                    "wavelength_nm": wavelength,
                    "selection_index": index + 1,
                    "grid_row": "" if using_preset else int(position[0]) + 1,
                    "grid_column": "" if using_preset else int(position[1]) + 1,
                    "m": int(order[0]),
                    "n": int(order[1]),
                    "required_incident_theta_deg": centered["incident_theta_deg"][index],
                    "required_incident_azimuth_deg": centered["incident_azimuth_deg"][index],
                    "required_incident_transverse_norm": centered[
                        "incident_transverse_norm"
                    ][index],
                    "center_feasible_free_space": bool(centered["center_feasible"][index]),
                    "centered_candidate": bool(centered["center_feasible"][index])
                    and relation["independent_conjugate_representative"],
                    "centered_conservative_primitive_candidate": bool(
                        centered["center_feasible"][index]
                    )
                    and relation["independent_conjugate_representative"]
                    and relation["is_primitive"],
                    **{
                        key: value
                        for key, value in relation.items()
                        if key not in {"m", "n"}
                    },
                }
            )

    selected_fields = list(selected_rows[0])
    write_csv(
        args.output_dir / "selected_order_metrics.csv", selected_fields, selected_rows
    )
    independent_rows = [row for row in selected_rows if row["large_fov_candidate"]]
    conservative_rows = [
        row for row in selected_rows if row["conservative_primitive_candidate"]
    ]
    write_csv(
        args.output_dir / "independent_candidate_orders.csv",
        selected_fields,
        independent_rows,
    )
    write_csv(
        args.output_dir / "conservative_primitive_orders.csv",
        selected_fields,
        conservative_rows,
    )
    if center_rows:
        write_csv(
            args.output_dir / "center_incidence_requirements.csv",
            list(center_rows[0]),
            center_rows,
        )
    ranking = []
    if not using_preset:
        ranking = rank_blocks(
            positions, wavelengths, theta_values, azimuth_values, args
        )
        write_csv(
            args.output_dir / "continuous_block_ranking.csv",
            list(ranking[0]),
            ranking[: args.ranking_limit],
        )
    plot_order_map(
        args.output_dir / "diffraction_order_map.png",
        positions,
        orders,
        wavelengths,
        theta_values,
        azimuth_values,
        args,
    )
    if not using_preset and len(wavelengths) == 1:
        plot_coupling_grid(
            args.output_dir / "coupling_grid.png",
            positions,
            orders,
            float(wavelengths[0]),
            float(theta_values[0]),
            float(azimuth_values[0]),
            args,
        )

    if using_preset:
        selected_summary = {
            "order_preset": "paper-23",
            "propagating_count": sum(row["propagating"] for row in selected_rows),
            "captured_count": sum(row["captured_by_na"] for row in selected_rows),
            "total_tests": len(selected_rows),
            "minimum_air_margin": min(row["air_margin"] for row in selected_rows),
            "minimum_same_wavelength_separation": min(
                minimum_separation(
                    np.asarray([row["ux"] for row in selected_rows if row["wavelength_nm"] == wavelength]),
                    np.asarray([row["uy"] for row in selected_rows if row["wavelength_nm"] == wavelength]),
                )
                for wavelength in wavelengths
            ),
        }
    else:
        selected_summary = evaluate_block(
            positions,
            args.order_m_start,
            args.order_n_start,
            wavelengths,
            theta_values,
            azimuth_values,
            args,
        )
    config = {
        "mode": args.mode,
        "order_preset": args.order_preset,
        "conjugate_encoding": args.conjugate_encoding,
        "assumption": "period_x_nm and period_y_nm are the physical diffraction periods",
        "angle_convention": "incident theta is measured from the surface normal",
        "interpretation": (
            "fixed-incidence tests angular tiling under one illumination; "
            "center-each-order solves a separate incidence condition for every order"
        ),
        "wavelengths_nm": wavelengths.tolist(),
        "incident_theta_deg": theta_values.tolist(),
        "incident_azimuth_deg": azimuth_values.tolist(),
        "period_x_nm": args.period_x_nm,
        "period_y_nm": args.period_y_nm,
        "n_in": args.n_in,
        "n_out": args.n_out,
        "na": args.na,
        "active_channel_count": len(positions),
        "selected_block": selected_summary,
        "best_ranked_block": ranking[0] if ranking else None,
        "phase_relation_summary": {
            "zero_order_count": sum(
                not row["phase_controllable"] for row in relation_by_order.values()
            ),
            "orders_with_selected_conjugate": sum(
                row["conjugate_order_in_selection"]
                for row in relation_by_order.values()
            ),
            "nonprimitive_harmonic_orders": sum(
                row["phase_controllable"] and not row["is_primitive"]
                for row in relation_by_order.values()
            ),
            "large_fov_independent_candidate_count": sum(
                row["large_fov_candidate"] for row in selected_rows
            ),
            "conservative_primitive_candidate_count": sum(
                row["conservative_primitive_candidate"] for row in selected_rows
            ),
        },
    }
    if center_rows:
        config["center_each_order_summary"] = {
            "free_space_center_feasible_count": sum(
                row["center_feasible_free_space"] for row in center_rows
            ),
            "total_orders": len(center_rows),
            "independent_centered_candidate_count": sum(
                row["centered_candidate"] for row in center_rows
            ),
            "conservative_primitive_centered_candidate_count": sum(
                row["centered_conservative_primitive_candidate"]
                for row in center_rows
            ),
            "note": "each feasible row requires its own incidence direction",
        }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"级次图: {args.output_dir / 'diffraction_order_map.png'}")
    if not using_preset and len(wavelengths) == 1:
        print(f"耦合矩阵图: {args.output_dir / 'coupling_grid.png'}")
    print(f"逐级次指标: {args.output_dir / 'selected_order_metrics.csv'}")
    print(f"独立共轭代表候选: {args.output_dir / 'independent_candidate_orders.csv'}")
    print(f"保守本原级次候选: {args.output_dir / 'conservative_primitive_orders.csv'}")
    if ranking:
        print(f"连续方阵排名: {args.output_dir / 'continuous_block_ranking.csv'}")
    if center_rows:
        print(f"逐级次居中入射条件: {args.output_dir / 'center_incidence_requirements.csv'}")


if __name__ == "__main__":
    main()
