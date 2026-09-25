import argparse
import json
import math
from pathlib import Path

import numpy as np


PAPER_23_ORDERS = np.asarray(
    [(m, n) for n in range(3, -4, -1) for m in (-3, -2, -1)]
    + [(0, 2), (0, 1)],
    dtype=np.int16,
)


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="复算李仲阳团队大视场论文中公开参数足够完整的实例。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output" / "li_zhongyang_paper_instances",
    )
    parser.add_argument("--edge-samples", type=int, default=121)
    return parser.parse_args()


def direction_rays(points_xy):
    radii2 = np.sum(points_xy**2, axis=1)
    inside = radii2 <= 1 + 1e-12
    points_xy = points_xy[inside]
    radii2 = radii2[inside]
    z = np.sqrt(np.maximum(0, 1 - radii2))
    return np.column_stack((points_xy, z))


def maximum_ray_separation_deg(rays):
    minimum_dot = 1.0
    for start in range(0, len(rays), 256):
        dots = rays[start : start + 256] @ rays.T
        minimum_dot = min(minimum_dot, float(np.min(dots)))
    return math.degrees(math.acos(np.clip(minimum_dot, -1, 1)))


def square_boundary_points(center_x, center_y, half_width, samples):
    offsets = np.linspace(-half_width, half_width, samples)
    left = np.column_stack(
        (np.full(samples, center_x - half_width), center_y + offsets)
    )
    right = np.column_stack(
        (np.full(samples, center_x + half_width), center_y + offsets)
    )
    bottom = np.column_stack(
        (center_x + offsets, np.full(samples, center_y - half_width))
    )
    top = np.column_stack(
        (center_x + offsets, np.full(samples, center_y + half_width))
    )
    intersections = []
    y_min, y_max = center_y - half_width, center_y + half_width
    x_min, x_max = center_x - half_width, center_x + half_width
    for x_value in (x_min, x_max):
        if abs(x_value) <= 1:
            y_circle = math.sqrt(max(0, 1 - x_value**2))
            for y_value in (-y_circle, y_circle):
                if y_min - 1e-12 <= y_value <= y_max + 1e-12:
                    intersections.append((x_value, y_value))
    for y_value in (y_min, y_max):
        if abs(y_value) <= 1:
            x_circle = math.sqrt(max(0, 1 - y_value**2))
            for x_value in (-x_circle, x_circle):
                if x_min - 1e-12 <= x_value <= x_max + 1e-12:
                    intersections.append((x_value, y_value))
    parts = [left, right, bottom, top]
    if intersections:
        parts.append(np.asarray(intersections, dtype=float))
    return np.vstack(parts)


def paper_23_case(edge_samples):
    wavelength_nm = 488.0
    period_nm = 2000.0
    incident_theta_deg = 46.0
    delta_u = wavelength_nm / period_nm
    incident_ux = math.sin(math.radians(incident_theta_deg))
    centers = np.column_stack(
        (
            incident_ux + PAPER_23_ORDERS[:, 0] * delta_u,
            PAPER_23_ORDERS[:, 1] * delta_u,
        )
    )
    center_rays = direction_rays(centers)
    boundaries = np.vstack(
        [
            square_boundary_points(x, y, delta_u / 2, edge_samples)
            for x, y in centers
        ]
    )
    boundary_rays = direction_rays(boundaries)
    radii = np.linalg.norm(centers, axis=1)
    theta_out = np.degrees(np.arcsin(radii))
    lookup = {
        tuple(order): float(theta)
        for order, theta in zip(PAPER_23_ORDERS.tolist(), theta_out.tolist())
    }
    return {
        "name": "23-channel diffraction-order decoupling",
        "source_status": "公开参数足够完整，可严格复算级次中心",
        "parameters": {
            "wavelength_nm": wavelength_nm,
            "period_x_nm": period_nm,
            "period_y_nm": period_nm,
            "incident_theta_deg": incident_theta_deg,
            "incident_phi_deg": 0.0,
            "n_in": 1.0,
            "n_out": 1.0,
            "na": 1.0,
            "orders": PAPER_23_ORDERS.tolist(),
        },
        "paper_reported_fov_deg": 137.0,
        "reproduced": {
            "delta_u": delta_u,
            "propagating_centers": int(np.count_nonzero(radii <= 1 + 1e-12)),
            "total_centers": int(len(radii)),
            "minimum_air_margin": float(np.min(1 - radii)),
            "maximum_center_polar_angle_deg": float(np.max(theta_out)),
            "maximum_center_ray_separation_deg": maximum_ray_separation_deg(
                center_rays
            ),
            "maximum_local_square_boundary_separation_deg_approx": maximum_ray_separation_deg(
                boundary_rays
            ),
            "table_s3_angle_checks_deg": {
                "(-3,3)": lookup[(-3, 3)],
                "(-2,0)": lookup[(-2, 0)],
                "(-1,0)": lookup[(-1, 0)],
                "(0,2)": lookup[(0, 2)],
            },
        },
        "fov_note": (
            "论文Table S1的约137°不是23个级次中心的最大夹角；"
            "它采用论文自己的全息角谱覆盖口径，不能用中心点外包角替代。"
        ),
    }


def sample_3_case(wavelength_nm):
    period_nm = 1000.0
    grid_size = 3
    delta_u = wavelength_nm / period_nm
    span_u = grid_size * delta_u
    theoretical_fov = (
        2 * math.degrees(math.asin(span_u / 2)) if span_u <= 2 else None
    )
    used_span_u = 2 * math.sin(math.radians(87.0 / 2))
    return {
        "assumed_wavelength_nm": wavelength_nm,
        "delta_u": delta_u,
        "three_order_span_u": span_u,
        "theoretical_full_circular_fov_deg": theoretical_fov,
        "paper_used_fov_deg": 87.0,
        "paper_used_span_u": used_span_u,
        "used_fraction_of_full_diameter": used_span_u / span_u,
    }


def build_report(edge_samples):
    return {
        "paper": {
            "title": "Breaking the Diffraction-Encoding Limit for High-Capacity Meta-Holography via Multiorder Decoupling",
            "doi": "10.1021/acsnano.6c04285",
            "supporting_information_doi": "10.1021/acsnano.6c04285.s001",
        },
        "paper_23_channel": paper_23_case(edge_samples),
        "sample_3_clock": {
            "source_status": (
                "SI Figure S9明确P=1000 nm、9个相邻级次和87° FoV；"
                "该节未给出波长、绝对级次编号和共同入射角。"
            ),
            "paper_reported_fov_deg": 87.0,
            "relative_geometry_runs": [
                sample_3_case(488.0),
                sample_3_case(480.0),
            ],
        },
        "large_fov_168": {
            "paper_reported_fov_deg": 168.0,
            "source_status": (
                "正文摘要给出168°；公开SI未同时给齐该实例的波长、"
                "绝对级次集合和入射角，不能严格复跑。"
            ),
        },
        "figure_s2_near_180": {
            "paper_reported_fov_deg": 180.0,
            "orders": [[m, n] for n in (1, 0, -1) for m in (-1, 0, 1)],
            "encoding": "detour phase + PB phase",
            "source_status": (
                "SI给出3×3级次与混合相位机制，但未给齐P、波长和入射角；"
                "只能确认接近180°的数值结果，不能严格复跑参数。"
            ),
        },
    }


def main():
    args = parse_args()
    if args.edge_samples < 3:
        raise ValueError("edge-samples必须至少为3。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.edge_samples)
    output_path = args.output_dir / "fov_reproduction_summary.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"结果: {output_path}")


if __name__ == "__main__":
    main()
