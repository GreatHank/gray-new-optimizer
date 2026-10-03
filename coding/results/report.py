"""Report the configured full scene with shared exposure and untouched raw metrics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from coding.results.storage import load_result, json_value
from coding.geometry.orders import order_directions
from coding.geometry.preflight import validate_physics
from coding import ROOT

TARGET = ROOT / "output/targets/circle_ocean_strict9_3x3_p800_gray256_500"


def assemble(planes, positions):
    size = planes.shape[-1]
    rows, cols = np.max(positions, axis=0) + 1
    canvas = np.zeros((rows * size, cols * size), dtype=planes.dtype)
    for plane, (r, c) in zip(planes, positions):
        canvas[r*size:(r+1)*size, c*size:(c+1)*size] = plane
    return canvas


def panels(images, titles, path):
    figure, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 6.5), squeeze=False)
    gray = plt.get_cmap("gray").copy()
    gray.set_bad("#b8b8b8")
    for axis, im, title in zip(axes.flat, images, titles):
        axis.imshow(im, cmap=gray, vmin=0, vmax=1)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=TARGET)
    args = parser.parse_args()
    args.output_dir = (ROOT/args.output_dir).resolve()
    if (ROOT/"output/data").resolve() not in args.output_dir.parents:
        raise ValueError("Detailed reports must use a new directory under output/data")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    target_dir = args.target_dir
    manifest = json.loads((target_dir / "summary.json").read_text(encoding="utf-8"))
    level_count = manifest.get("gray_level_count", 4)
    positions = np.asarray(manifest["canvas_positions"])
    originals = np.asarray(Image.open(target_dir / "source_resized.png")) / 255
    target_canvas = np.asarray(Image.open(target_dir / ("target.png" if (target_dir / "target.png").exists() else "target_fourgray.png"))) / 255
    runs, rows, checks, per_order_rows, pair_rows = [], [], [], [], []
    for directory in args.result_dirs:
        data = load_result(directory)
        orders = np.asarray(data["pairMat"])
        np.testing.assert_array_equal(orders, manifest["orders"])
        labels = np.asarray(data["target_labels"])
        targets = labels.astype(np.float64) / (level_count - 1)
        np.testing.assert_array_equal(assemble(labels, positions), np.rint(target_canvas * (level_count - 1)).astype(np.uint8))
        raw = np.asarray(data["optimized_raw"], dtype=np.float64)
        pixel_count = raw.shape[-1] ** 2
        reference = np.stack([np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * (
            m * data["phdx"].astype(np.float64) + n * data["phdy"].astype(np.float64)
            + data["pp"].astype(np.float64)))))) ** 2 for m, n in orders])
        relative_l2 = float(np.linalg.norm(raw - reference) / np.linalg.norm(reference))
        parseval_error = float(np.max(np.abs(raw.sum(axis=(1, 2)) / pixel_count ** 2 - 1)))
        if relative_l2 > 2e-5 or parseval_error > 2e-6:
            raise ValueError(f"Forward verification failed: {directory}, {relative_l2}, {parseval_error}")
        checks.append({"run": directory.name, "numpy_relative_l2_error": relative_l2,
                       "parseval_max_relative_error": parseval_error})
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        physics = config["physics"]
        physical_check = validate_physics(orders, physics, targets)
        if physical_check["strict_failures"]:
            raise ValueError(f"Physical layout requirements failed: {physical_check['strict_failures']}")
        theta, azimuth = np.deg2rad([physics["incident_theta_deg"], physics["incident_azimuth_deg"]])
        incident = physics["n_in"] / physics["n_out"] * np.sin(theta) * np.array([np.cos(azimuth), np.sin(azimuth)])
        pitch = physics["wavelength_nm"] / physics["n_out"] / np.array([physics["period_x_nm"], physics["period_y_nm"]])
        yy, xx = np.indices(raw.shape[1:])
        local = np.stack([(xx-raw.shape[2]//2)/raw.shape[2], -(yy-raw.shape[1]//2)/raw.shape[1]], axis=-1)
        visible = np.stack([np.linalg.norm(incident + (order + local) * pitch, axis=-1)
                            <= min(1, physics["na"] / physics["n_out"]) for order in orders])
        visible_fractions = (raw * visible).sum(axis=(1, 2)) / raw.sum(axis=(1, 2))
        scale = pixel_count / targets.mean()
        common_gray = raw / scale
        zero = int(np.flatnonzero(np.all(orders == 0, axis=1))[0])
        with (directory / "metrics.csv").open(encoding="utf-8") as handle:
            channel_metrics = list(csv.DictReader(handle))
        foreground_means = np.asarray([float(item["foreground_mean"]) for item in channel_metrics])
        foreground_means = foreground_means[np.isfinite(foreground_means)]
        row = {
            "run": directory.name, "loss": config["optimizer"]["image_loss_mode"],
            "optimizer_name": config["optimizer"].get("optimizer_name", "adam"),
            "stage_iterations": config["optimizer"]["epochs"],
            "common_scale_gray_rmse": float(np.sqrt(np.mean((common_gray - targets) ** 2))),
            "raw_target_cosine": float(np.sum(raw * targets) / (np.linalg.norm(raw) * np.linalg.norm(targets))),
            "structure_cosine_mean": summary["metrics"]["structure_cosine_mean"],
            "structure_cosine_min": summary["metrics"]["structure_cosine_min"],
            "snr_mean_db": summary["metrics"]["foreground_background_snr_db_mean"],
            "snr_min_db": summary["metrics"]["foreground_background_snr_db_min"],
            "cnr_mean": summary["metrics"]["foreground_background_cnr_mean"],
            "foreground_mean_cv": float(np.std(foreground_means) / np.mean(foreground_means)),
            "foreground_metric_channel_count": int(len(foreground_means)),
            "background_only_channel_count": int(len(orders)-len(foreground_means)),
            "target_region_energy_fraction_mean": float(np.mean([float(x["target_efficiency"]) for x in channel_metrics])),
            "gray_monotonic_channels": summary["metrics"]["grayscale_monotonic_channels"],
            "gray_ratio_rmse_mean_present_white": summary["metrics"]["grayscale_ratio_rmse_mean"],
            "zero_order_snr_db": float(channel_metrics[zero]["foreground_background_snr_db"]),
            "zero_order_structure_cosine": float(channel_metrics[zero]["structure_cosine"]),
            "zero_order_image_cosine": float(np.sum(raw[zero] * targets[zero])
                / (np.linalg.norm(raw[zero]) * np.linalg.norm(targets[zero]))),
            "zero_order_target_region_energy_fraction": float(channel_metrics[zero]["target_efficiency"]),
            "raw_display_white_1x": float(scale),
            "propagating_scalar_fraction_mean": float(visible_fractions.mean()),
            "propagating_centers": physical_check["center_propagating"],
            "ring_coverage_fraction": physical_check["layout"].get("ring_coverage_fraction"),
            "outside_target_pixels": physical_check["layout"]["outside_target_pixels"],
            "outside_target_mass_fraction": physical_check["layout"]["outside_target_mass_fraction"],
        }
        for c, (order, metrics) in enumerate(zip(orders, channel_metrics)):
            per_order_rows.append({
                "run": directory.name, "channel": c+1, "m": int(order[0]), "n": int(order[1]),
                "total_raw_energy": float(metrics["total_energy"]),
                "raw_plane_mean": float(metrics["plane_mean"]),
                "raw_foreground_mean": float(metrics["foreground_mean"]),
                "raw_background_mean": float(metrics["background_mean"]),
                "raw_background_std": float(np.sqrt(float(metrics["background_variance"]))),
                "snr_db": float(metrics["foreground_background_snr_db"]),
                "cnr": float(metrics["foreground_background_cnr"]),
                "target_region_energy_fraction": float(metrics["target_efficiency"]),
                "image_cosine": (float(np.sum(raw[c] * targets[c])
                    / (np.linalg.norm(raw[c]) * np.linalg.norm(targets[c])))
                    if np.any(targets[c]) else None),
                "common_scale_gray_rmse": float(np.sqrt(np.mean((common_gray[c] - targets[c])**2))),
                "propagating_scalar_fraction": float(visible_fractions[c]),
            })
        order_indices = {tuple(order): index for index, order in enumerate(orders)}
        mirror_indices = (-np.arange(raw.shape[-1])) % raw.shape[-1]
        for order, c in order_indices.items():
            opposite = tuple(-np.array(order))
            if opposite not in order_indices or order >= opposite:
                continue
            other = order_indices[opposite]
            mirrored = raw[c][np.ix_(mirror_indices, mirror_indices)]
            pair_rows.append({"run": directory.name, "m": int(order[0]), "n": int(order[1]),
                              "opposite_m": int(opposite[0]), "opposite_n": int(opposite[1]),
                              "mirror_cosine": float(np.sum(mirrored * raw[other]) / (np.linalg.norm(mirrored) * np.linalg.norm(raw[other]))),
                              "mirror_relative_l2": float(np.linalg.norm(mirrored - raw[other]) / np.linalg.norm(raw[other]))})
        rows.append(row)
        scene = assemble(common_gray, positions)
        visible_canvas = assemble(visible, positions)
        runs.append((directory.name, scene, row, common_gray, targets, zero))
        for exposure in (1, 2):
            display = np.clip(scene * exposure, 0, 1)
            Image.fromarray(np.rint(display * 255).astype(np.uint8)).save(
                args.output_dir / f"{directory.name}_full_scene_{exposure}x.png")
            observable = np.ma.array(display, mask=~visible_canvas)
            panels([target_canvas, observable],
                   ["Complete target canvas",
                    f"Propagating scalar reconstruction, common {exposure}x\nGrey area outside propagation cone; no rescaling"],
                   args.output_dir / f"{directory.name}_propagating_comparison_{exposure}x.png")
            target_panels = [originals] if level_count == 256 else [originals, target_canvas]
            target_titles = ["Full-gray circular target, selected grid" if manifest.get("circular_field_mapping") else "Full-gray square target"] if level_count == 256 else ["Resampled source", f"{level_count}-gray target"]
            panels(target_panels + [display],
                   target_titles + [f"{row['loss'].upper()} + shared PP, {exposure}x exposure\nSNR {row['snr_mean_db']:.2f} dB; image cosine {row['raw_target_cosine']:.3f}"],
                   args.output_dir / f"{directory.name}_comparison_{exposure}x.png")
        panels([targets[zero], np.clip(common_gray[zero], 0, 1)],
               ["(0,0) target", f"(0,0) reconstruction, common 1x\nSNR {row['zero_order_snr_db']:.2f} dB"],
               args.output_dir / f"{directory.name}_zero_order.png")
        if (-1, 1) in order_indices and (1, -1) in order_indices:
            c, other = order_indices[(-1, 1)], order_indices[(1, -1)]
            panels([targets[c], np.clip(common_gray[c]*2, 0, 1), targets[other], np.clip(common_gray[other]*2, 0, 1)],
                   ["(-1,1) target", "(-1,1) reconstruction, common 2x",
                    "(1,-1) target", "(1,-1) reconstruction, common 2x"],
                   args.output_dir / f"{directory.name}_opposite_orders.png")
    panels([target_canvas] + [np.clip(scene, 0, 1) for _, scene, *_ in runs],
           [f"{level_count}-gray target"] + [f"{name}\nSNR {row['snr_mean_db']:.2f} dB, image cosine {row['raw_target_cosine']:.3f}" for name, _, row, *_ in runs],
           args.output_dir / "all_runs_shared_1x.png")
    with (args.output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = min(rows, key=lambda item: item["common_scale_gray_rmse"])
    with (args.output_dir / "best_per_order_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_order_rows[0]))
        writer.writeheader()
        writer.writerows(item for item in per_order_rows if item["run"] == best["run"])
    if pair_rows:
        with (args.output_dir / "opposite_pair_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
            writer.writeheader()
            writer.writerows(pair_rows)
    report = {
        "source": manifest["source"], "channel_count": len(manifest["orders"]), "gray_level_count": level_count,
        "physical_layout_check": physical_check,
        "forward": "Phi=m*dx+n*dy+PP; U=exp(i*Phi); I=abs(fftshift(fft2(U)))**2",
        "target_preprocessing": manifest["preprocessing"],
        "display": "All channels and runs share raw white = tile_pixel_count / global_target_mean; only stated exposure multiplies the displayed image",
        "metrics_scope": "Full, unmasked raw scalar FFT and original target regions. Foreground metrics are undefined for black-only channels; averages state the foreground channel count. Background and full-plane loss include all channels. Propagation masks are display only, not device efficiencies.",
        "physical_scope": "Paper detour-phase and scalar FFT plus shared PP extension; unit-cell PP realization and full device remain unverified",
        "best_run_by_common_scale_rmse": best["run"],
        "forward_checks": checks, "runs": rows,
    }
    (args.output_dir / "report.json").write_text(json.dumps(json_value(report), indent=2, allow_nan=False), encoding="utf-8")
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    physics = config["physics"]
    ring_angle = physics.get("layout_requirements", {}).get("ring_theta_deg")
    half_x = physics["wavelength_nm"] / physics["n_out"] / physics["period_x_nm"] / 2
    half_y = physics["wavelength_nm"] / physics["n_out"] / physics["period_y_nm"] / 2
    for axis, theta in zip(axes, (physics["incident_theta_deg"], 5.0, 50.0)):
        directions = order_directions(orders, physics["wavelength_nm"],
            physics["period_x_nm"], physics["period_y_nm"], theta, physics["incident_azimuth_deg"],
            physics["n_in"], physics["n_out"], physics["na"])
        ux, uy = directions["ux"], directions["uy"]
        axis.add_patch(plt.Circle((0, 0), 1, fill=False, color="black"))
        if ring_angle is not None:
            axis.add_patch(plt.Circle((0, 0), np.sin(np.deg2rad(ring_angle)), fill=False, color="#c53932", ls="--", lw=0.8))
        angle_check = validate_physics(orders, {**physics, "incident_theta_deg": theta}, targets)
        colors = np.where(directions["propagating"], "#1479a8", "#c53932")
        for x, y, order, color in zip(ux, uy, orders, colors):
            axis.add_patch(plt.Rectangle((x-half_x, y-half_y), 2*half_x, 2*half_y,
                                        fill=False, edgecolor=color, linewidth=0.6))
            axis.scatter(x, y, color=color, s=10)
            axis.text(x, y+0.025, f"{order[0]},{order[1]}", ha="center", fontsize=6)
        coverage_title = (f"{ring_angle:g}-degree ring coverage: {angle_check['layout']['ring_coverage_fraction']:.1%}"
                          if ring_angle is not None else
                          f"Full / partial cells: {angle_check['layout']['fully_visible_cells']} / {angle_check['layout']['partly_visible_cells']}")
        axis.set(xlim=(min(-1.2, float(ux.min()-half_x)), max(1.2, float(ux.max()+half_x))),
                 ylim=(min(-1.2, float(uy.min()-half_y)), max(1.2, float(uy.max()+half_y))), aspect="equal",
                 xlabel="outgoing direction cosine u_x", ylabel="u_y",
                 title=f"Incidence {theta:.2f} deg; {directions['propagating'].sum()}/{len(orders)} centers\n{coverage_title}")
        axis.grid(alpha=0.2)
    figure.suptitle(f"Same phase maps; {physics['wavelength_nm']:g} nm wavelength, {physics['period_x_nm']:g} nm period")
    figure.tight_layout()
    figure.savefig(args.output_dir / "angle_geometry_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"output": str(args.output_dir),
                      "best_run": best["run"], "forward_checks": checks, "runs": rows},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
