"""Read-only physical checks used before a long optimizer run."""

import argparse
import json
from pathlib import Path

from math import gcd

import numpy as np

from coding.geometry.orders import order_directions


REQUIRED_FIELDS = (
    "wavelength_nm",
    "period_x_nm",
    "period_y_nm",
    "n_in",
    "n_out",
    "na",
    "incident_theta_deg",
    "incident_azimuth_deg",
)


def _harmonic(order):
    m, n = (abs(int(value)) for value in order)
    return gcd(m, n) > 1


def validate_target_layout(orders, targets, config):
    """Check the diagram's cells and actual target support at fftshift bins."""
    wavelength = float(config["wavelength_nm"])
    n_out = float(config["n_out"])
    pitch = wavelength / n_out / np.asarray(
        [config["period_x_nm"], config["period_y_nm"]], dtype=float)
    theta, azimuth = np.deg2rad([config["incident_theta_deg"], config["incident_azimuth_deg"]])
    incident = float(config["n_in"]) / n_out * np.sin(theta) * np.array(
        [np.cos(azimuth), np.sin(azimuth)])
    centers = incident + orders * pitch
    half = pitch / 2
    near = np.linalg.norm(np.maximum(np.abs(centers) - half, 0), axis=1)
    far = np.linalg.norm(np.abs(centers) + half, axis=1)
    capture_radius = min(1.0, float(config["na"]) / n_out)
    requirements = config.get("layout_requirements", {})
    report = {
        "fully_visible_cells": int(np.sum(far <= capture_radius)),
        "partly_visible_cells": int(np.sum((near < capture_radius) & (far > capture_radius))),
        "fully_outside_cells": int(np.sum(near >= capture_radius)),
        "all_cells_intersect_propagation": bool(np.all(near < capture_radius)),
        "order_count_pass": len(orders) <= int(requirements.get("max_order_count", len(orders))),
    }
    if "ring_theta_deg" in requirements:
        angles = np.arange(72000) * (2 * np.pi / 72000)
        radius = np.sin(np.deg2rad(float(requirements["ring_theta_deg"])))
        points = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        nearest_orders = np.floor((points - incident) / pitch + 0.5).astype(int)
        selected = set(map(tuple, orders.tolist()))
        fraction = float(np.mean([tuple(pair) in selected for pair in nearest_orders]))
        if radius > capture_radius:
            fraction = 0.0
        report.update(ring_theta_deg=float(requirements["ring_theta_deg"]),
                      ring_coverage_fraction=fraction,
                      ring_coverage_pass=fraction + 1e-12 >= float(requirements["minimum_ring_coverage"]))
    if targets is not None:
        targets = np.asarray(targets)
        if targets.ndim != 3 or len(targets) != len(orders):
            raise ValueError("Target support must have one H×W plane per order")
        height, width = targets.shape[1:]
        yy, xx = np.indices((height, width))
        local = np.stack([(xx - width // 2) / width,
                          -(yy - height // 2) / height], axis=-1)
        per_order, outside_pixels, outside_mass, total_mass = [], 0, 0.0, float(targets.sum())
        for order, center, target in zip(orders, centers, targets):
            radius = np.linalg.norm(center + local * pitch, axis=-1)
            support = target > 0
            outside = support & (radius > capture_radius + 1e-12)
            count = int(outside.sum())
            outside_pixels += count
            outside_mass += float(target[outside].sum())
            per_order.append({"m": int(order[0]), "n": int(order[1]),
                              "target_pixels": int(support.sum()), "outside_target_pixels": count,
                              "maximum_target_radius_u": float(radius[support].max()) if support.any() else None})
        report.update(actual_target_support_pass=outside_pixels == 0,
                      outside_target_pixels=outside_pixels,
                      outside_target_mass_fraction=outside_mass / total_mass if total_mass else 0.0,
                      target_support_by_order=per_order,
                      support_axes="fftshift bins: x right/u+, rows down/v-; no half-bin offset")
    return report


def validate_physics(pair_mat, config, targets=None):
    """Validate one common illumination without changing the optical forward model."""
    if not config:
        return {
            "status": "not_configured",
            "scope": "software_forward_only",
            "messages": ["未提供物理配置；不能宣称论文或器件配置已验证。"],
        }
    missing = [name for name in REQUIRED_FIELDS if name not in config]
    if missing:
        return {
            "status": "invalid",
            "scope": config.get("scope", "unspecified"),
            "messages": [f"缺少物理参数：{', '.join(missing)}"],
            "failures": ["missing_parameters"],
        }
    if config.get("common_incidence", True) is not True:
        return {
            "status": "invalid",
            "scope": config.get("scope", "unspecified"),
            "messages": ["优化器要求所有级次使用同一次共同入射。"],
            "failures": ["not_common_incidence"],
        }
    values = {name: float(config[name]) for name in REQUIRED_FIELDS}
    if min(values["wavelength_nm"], values["period_x_nm"], values["period_y_nm"], values["n_in"], values["n_out"], values["na"]) <= 0:
        return {
            "status": "invalid",
            "scope": config.get("scope", "unspecified"),
            "messages": ["波长、周期、折射率和 NA 必须为正数。"],
            "failures": ["nonpositive_parameter"],
        }
    if values["na"] > values["n_out"]:
        return {
            "status": "invalid",
            "scope": config.get("scope", "unspecified"),
            "messages": ["NA 必须不大于输出介质折射率。"],
            "failures": ["invalid_na"],
        }

    orders = np.asarray(pair_mat, dtype=np.int16)
    directions = order_directions(
        orders,
        values["wavelength_nm"],
        values["period_x_nm"],
        values["period_y_nm"],
        values["incident_theta_deg"],
        values["incident_azimuth_deg"],
        values["n_in"],
        values["n_out"],
        values["na"],
    )
    zero_count = int(np.sum(np.all(orders == 0, axis=1)))
    selected = {tuple(map(int, order)) for order in orders}
    conjugate_pairs = sum(tuple(map(int, -order)) in selected for order in orders) // 2
    failures = []
    if not np.all(directions["propagating"]):
        failures.append("nonpropagating_centers")
    if not np.all(directions["captured"]):
        failures.append("centers_outside_na")

    coverage = {"order_centers": "pass" if not failures else "fail"}
    required = set(config.get("strict_require", ["all_centers_propagating", "all_centers_captured"]))
    layout = validate_target_layout(orders, targets, values | {
        "layout_requirements": config.get("layout_requirements", {})})
    coverage["nominal_square"] = ("pass" if layout["fully_visible_cells"] == len(orders)
                                  else "partial_or_outside")
    coverage["actual_target_support"] = "not_evaluated"
    if "actual_target_support_pass" in layout:
        coverage["actual_target_support"] = "pass" if layout["actual_target_support_pass"] else "partial_or_outside"
    strict_failures = []
    if "all_centers_propagating" in required and not np.all(directions["propagating"]):
        strict_failures.append("all_centers_propagating")
    if "all_centers_captured" in required and not np.all(directions["captured"]):
        strict_failures.append("all_centers_captured")
    if "actual_target_support" in required and coverage["actual_target_support"] != "pass":
        strict_failures.append("actual_target_support")
    for requirement, field in (("order_count", "order_count_pass"),
                               ("ring_coverage", "ring_coverage_pass"),
                               ("all_cells_intersect_propagation", "all_cells_intersect_propagation")):
        if requirement in required and not layout.get(field, False):
            strict_failures.append(requirement)

    status = "pass" if not failures and not strict_failures else "fail"
    if failures and not strict_failures and required.intersection(
            {"order_count", "ring_coverage", "all_cells_intersect_propagation"}):
        status = "layout_pass_with_partial_cells"
    if not config.get("device_configuration_verified", False) and status == "pass":
        status = "pass_with_engineering_assumptions"
    return {
        "status": status,
        "scope": config.get("scope", "unspecified"),
        "common_incidence": True,
        "center_propagating": int(np.sum(directions["propagating"])),
        "center_captured_by_na": int(np.sum(directions["captured"])),
        "channel_count": int(len(orders)),
        "minimum_air_margin": float(np.min(directions["air_margin"])),
        "zero_order_count": zero_count,
        "selected_conjugate_pair_count": int(conjugate_pairs),
        "nonprimitive_harmonic_count": int(sum(_harmonic(order) for order in orders if np.any(order))),
        "coverage": coverage,
        "layout": layout,
        "failures": failures,
        "strict_failures": strict_failures,
        "parameters": values,
        "source": config.get("source", "unspecified"),
        "messages": [
            "本检查只覆盖级次几何/NA；不验证单元响应、联合分光效率、全波仿真或实验。"
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="检查共同入射下的级次中心和可选支撑范围。")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.config.read_text(encoding="utf-8"))
    if "optimizer" in data:
        from coding import ROOT
        from coding.optimization.data import load_target
        targets, orders, _ = load_target(ROOT/data["optimizer"]["mat_file"], data["optimizer"]["gray_level_count"])
        result = validate_physics(orders, data.get("physics"), targets)
    else:
        result = validate_physics(data["orders"], data.get("physics", data))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("strict_failures"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
