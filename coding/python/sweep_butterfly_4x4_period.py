import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_butterfly_4x4_orders import (  # noqa: E402
    mapping_metrics,
    order_grid,
    selected_order_rows,
    theory_full_propagation,
    tile_complexity,
    transform_indices,
    write_csv,
)


PERIODS_NM = [850, 1000, 1100, 1200, 1224, 1250, 1300, 1358, 1400,
              1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400]


def parse_args():
    parser = argparse.ArgumentParser(description="扫描蝴蝶4x4级次的有效衍射周期。")
    parser.add_argument("--input-file", type=Path, default=ROOT / "input/butterfly.png")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wavelength-nm", type=float, default=480.0)
    return parser.parse_args()


def centered_metrics(m_start, n_start, period_nm, wavelength_nm):
    orders = order_grid(m_start, n_start)
    result = theory_full_propagation(orders, wavelength_nm, period_nm)
    step = wavelength_nm / period_nm
    desired_shift = -orders.reshape(-1, 2).mean(axis=0) * step
    centered_feasible = float(np.linalg.norm(desired_shift)) <= 1 + 1e-12
    # A local cell extends half a period beyond each outer order center.  When
    # the order square is exactly centered, the full 4x4 spectral square has
    # corner radius 2*sqrt(2)*lambda/P.
    if centered_feasible:
        incident_mode = "exactly_centered"
    else:
        incident_mode = "radially_projected_to_grazing"
    cell_corner_offsets = np.asarray(
        [[sx * step / 2, sy * step / 2] for sx in (-1, 1) for sy in (-1, 1)]
    )
    centers = result["output"]
    maximum_cell_corner_radius = max(
        float(np.linalg.norm(center + offset))
        for center in centers
        for offset in cell_corner_offsets
    )
    full_cell_square_inside = maximum_cell_corner_radius <= 1 + 1e-12
    fov = 2 * math.degrees(math.asin(min(1.0, 2 * step)))
    return {
        "period_nm": period_nm,
        "order_m_range": f"{m_start}..{m_start + 3}",
        "order_n_range": f"{n_start}..{n_start + 3}",
        "step_lambda_over_P": step,
        "centered_incident_norm": float(np.linalg.norm(desired_shift)),
        "centered_incidence_feasible": centered_feasible,
        "incident_mode": incident_mode,
        "theta_i_deg": result["theta_i_deg"],
        "phi_i_deg": result["phi_i_deg"],
        "propagating_centers": result["propagating_count"],
        "minimum_center_margin": result["minimum_margin"],
        "full_4x4_cell_square_inside_air_circle": full_cell_square_inside,
        "maximum_cell_corner_radius": maximum_cell_corner_radius,
        "circular_fov_deg": fov,
    }, result


def mapping_for(complexity, orders, transform):
    rank = np.argsort(complexity.ravel())[::-1]
    top8 = np.zeros(16, dtype=bool)
    top4 = np.zeros(16, dtype=bool)
    top8[rank[:8]] = True
    top4[rank[:4]] = True
    return mapping_metrics(
        complexity,
        top8.reshape(4, 4),
        top4.reshape(4, 4),
        orders,
        transform_indices(4)[transform],
        transform,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    gray = np.asarray(Image.open(args.input_file).convert("L"))
    complexity = tile_complexity(gray)[-1]
    strategies = [
        ("high_fov_one_axis", -2, 1, "rot90_ccw"),
        ("axis_free", 1, 1, "rot180"),
    ]
    sweep_rows = []
    p2000_summary = []
    for name, m_start, n_start, transform in strategies:
        orders = order_grid(m_start, n_start)
        mapping = mapping_for(complexity, orders, transform)
        details = {(m_start, n_start): (orders, mapping, None, None)}
        for period in PERIODS_NM:
            row, result = centered_metrics(m_start, n_start, period, args.wavelength_nm)
            row["strategy"] = name
            row["transform"] = transform
            row["axis_order_count"] = int(np.sum((orders[:, :, 0] == 0) | (orders[:, :, 1] == 0)))
            row["top4_without_multiple"] = mapping["top4_without_multiple"]
            row["largest_multiple_group"] = mapping["largest_group_size"]
            sweep_rows.append(row)
            if period == 2000:
                details[(m_start, n_start)] = (orders, mapping, result, result)
                recommended = {"m_start": m_start, "n_start": n_start}
                order_rows = selected_order_rows(
                    recommended, details, complexity, args.wavelength_nm, period, detail_index=2
                )
                write_csv(args.output_dir / f"{name}_P2000_order_data.csv", order_rows)
                p2000_summary.append({**row, "coupling_score": mapping["coupling_score"]})
    write_csv(args.output_dir / "period_sweep.csv", sweep_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "wavelength_nm": args.wavelength_nm,
                "periods_nm": PERIODS_NM,
                "p2000": p2000_summary,
                "thresholds_nm": {
                    "one_axis_exact_centering": args.wavelength_nm * math.sqrt(6.5),
                    "one_axis_full_cells_when_centered": 2 * math.sqrt(2) * args.wavelength_nm,
                    "axis_free_all_centers_at_grazing": 2 * math.sqrt(2) * args.wavelength_nm,
                    "axis_free_full_cells_at_grazing": 2.25 * math.sqrt(2) * args.wavelength_nm,
                    "axis_free_exact_centering": args.wavelength_nm * math.sqrt(12.5),
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
