"""Compare different order grids at one raw exposure and angular scale."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coding import ROOT
from coding.results.report import assemble
from coding.results.storage import load_result, json_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dirs", nargs=2, type=Path, required=True)
    parser.add_argument("--report-dirs", nargs=2, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--common-orders", action="store_true",
                        help="Compare metrics on identical targets of shared orders, using their fixed common raw white")
    args = parser.parse_args()
    output = (ROOT/args.output_dir).resolve()
    if (ROOT/"output/data").resolve() not in output.parents:
        raise ValueError("Comparison must use a new directory under output/data")
    cases = []
    for directory, report_dir in zip(args.result_dirs, args.report_dirs):
        directory, report_dir = ROOT/directory, ROOT/report_dir
        report = json.loads((report_dir/"report.json").read_text(encoding="utf-8"))
        if len(report["runs"]) != 1 or report["runs"][0]["run"] != directory.name:
            raise ValueError("Each report must verify its corresponding single run")
        data = load_result(directory)
        config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
        summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((Path(config["input"]["path"]).parent/"summary.json").read_text(encoding="utf-8"))
        target = data["target_labels"].astype(np.float64)/(manifest["gray_level_count"]-1)
        cases.append((data, config, manifest, target, report["runs"][0], summary))
    if cases[0][2]["source_sha256"] != cases[1][2]["source_sha256"]:
        raise ValueError("Grid comparison requires the same complete source image")
    if cases[0][0]["phdx"].shape != cases[1][0]["phdx"].shape:
        raise ValueError("Grid comparison requires the same shared array size")
    white = max(case[4]["raw_display_white_1x"] for case in cases)
    common_orders, common_indices = None, None
    if args.common_orders:
        other_orders = set(map(tuple, cases[1][0]["pairMat"]))
        common_orders = [tuple(order) for order in cases[0][0]["pairMat"] if tuple(order) in other_orders]
        if not common_orders:
            raise ValueError("No shared orders to compare")
        common_indices = [[next(i for i, order in enumerate(case[0]["pairMat"]) if tuple(order)==pair)
                           for pair in common_orders] for case in cases]
        np.testing.assert_array_equal(cases[0][3][common_indices[0]], cases[1][3][common_indices[1]])
        geometry_keys = ("wavelength_nm", "period_x_nm", "period_y_nm", "incident_theta_deg",
                         "incident_azimuth_deg", "n_in", "n_out", "na")
        np.testing.assert_array_equal([cases[0][1]["physics"][key] for key in geometry_keys],
                                      [cases[1][1]["physics"][key] for key in geometry_keys])
        white = float(np.prod(cases[0][3].shape[-2:])/cases[0][3][common_indices[0]].mean())
    output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(2, 2, figsize=(12, 12))
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#b8b8b8")
    records = []
    for row, (data, config, manifest, target, metrics, summary) in enumerate(cases):
        positions = np.asarray(manifest["canvas_positions"])
        target_scene = assemble(target, positions)
        raw_scene = assemble(data["optimized_raw"], positions).astype(np.float64)
        orders = np.asarray(manifest["orders"])
        physics = config["physics"]
        theta, azimuth = np.deg2rad([physics["incident_theta_deg"], physics["incident_azimuth_deg"]])
        incident = physics["n_in"]/physics["n_out"]*np.sin(theta)*np.array([np.cos(azimuth), np.sin(azimuth)])
        pitch = physics["wavelength_nm"]/physics["n_out"]/np.array([physics["period_x_nm"], physics["period_y_nm"]])
        size = manifest["tile_size"]
        yy, xx = np.indices(target_scene.shape)
        u = incident[0]+(orders[:, 0].min()-0.5+xx/size)*pitch[0]
        v = incident[1]+(orders[:, 1].max()+0.5-yy/size)*pitch[1]
        visible = u*u+v*v <= min(1, physics["na"]/physics["n_out"])**2
        extent = (u[0, 0]-pitch[0]/(2*size), u[0, -1]+pitch[0]/(2*size),
                  v[-1, 0]-pitch[1]/(2*size), v[0, 0]+pitch[1]/(2*size))
        grid = manifest["grid_size"]
        layout_label = manifest.get("layout_label", f"{grid} x {grid}")
        with (ROOT/args.result_dirs[row]/"metrics.csv").open(encoding="utf-8") as handle:
            snr_rows = list(csv.DictReader(handle))
        titles = [f"{layout_label} target; P={physics['period_x_nm']:g} nm",
                  f"{layout_label} reconstruction; common 1x\n"
                  f"SNR mean/min ({metrics['foreground_metric_channel_count']}/{len(orders)}): "
                  f"{metrics['snr_mean_db']:.2f}/{metrics['snr_min_db']:.2f} dB"]
        common_metrics = None
        if args.common_orders:
            indices = common_indices[row]
            shared_raw = np.asarray(data["optimized_raw"][indices],dtype=np.float64)
            shared_target = target[indices]
            snr = np.array([float(snr_rows[index]["foreground_background_snr_db"]) for index in indices])
            valid_snr = snr[np.isfinite(snr)]
            common_metrics = {"orders": common_orders, "targets_identical": True,
                "order_count": len(indices), "foreground_metric_channel_count": len(valid_snr),
                "snr_mean_db": float(valid_snr.mean()), "snr_min_db": float(valid_snr.min()),
                "raw_target_cosine": float(np.sum(shared_raw*shared_target)/(np.linalg.norm(shared_raw)*np.linalg.norm(shared_target))),
                "common_raw_white_gray_rmse": float(np.sqrt(np.mean((shared_raw/white-shared_target)**2)))}
            titles[1] = (f"{layout_label} reconstruction; common 1x\n"
                         f"Same {len(indices)} orders SNR mean/min: {valid_snr.mean():.2f}/{valid_snr.min():.2f} dB")
        for axis, plane, title in zip(axes[row], (target_scene, raw_scene/white), titles):
            axis.set_facecolor("#b8b8b8")
            axis.imshow(np.ma.array(plane, mask=~visible), extent=extent, cmap=cmap, vmin=0, vmax=1)
            axis.add_patch(plt.Circle((0, 0), 1, fill=False, color="#48b7b1", lw=0.8))
            axis.set(xlim=(-1.04, 1.04), ylim=(-1.04, 1.04), aspect="equal", title=title,
                     xlabel="u (x right)", ylabel="v (y up)")
        records.append({**metrics, "grid_size": grid, "layout_label": layout_label, "period_nm": physics["period_x_nm"],
                        "common_orders_metrics": common_metrics,
                        "circular_field_mapping": manifest.get("circular_field_mapping"),
                        "incident_theta_deg": physics["incident_theta_deg"],
                        "iterations": config["optimizer"]["epochs"],
                        "runtime_seconds": summary["runtime_seconds"],
                        "angular_fft_bin_spacing_u": pitch[0]/size,
                        "maximum_target_radius_u": max(item["maximum_target_radius_u"] or 0
                            for item in manifest["physics_check"]["layout"]["target_support_by_order"]),
                        "foreground_snr_by_order": [{"m": int(order[0]), "n": int(order[1]),
                            "target_pixels": int(float(value["target_pixels"])),
                            "snr_db": float(value["foreground_background_snr_db"])}
                            for order, value in zip(data["pairMat"], snr_rows)]})
    source_name = Path(cases[0][2]["source"]).stem
    height, width = cases[0][0]["phdx"].shape
    figure.suptitle(f"{source_name}; shared {width} x {height} array; raw white={white:.3f}\n"
                   "Common angular axes and exposure; gray is outside propagation/selected grid", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output/"targets_and_results_shared_1x.png", dpi=180)
    plt.close(figure)
    (output/"comparison.json").write_text(json.dumps(json_value({
        "common_raw_white_1x": white, "display_scope": "One fixed raw white for both grids; no per-channel gains. Propagation masks used only for display.",
        "common_orders_comparison": args.common_orders,
        "common_raw_white_definition": "HW / mean(identical common-order targets)" if args.common_orders else "Maximum of native run raw whites",
        "metrics_scope": ("Original full-run metrics retained; common_orders_metrics use identical targets and fixed foreground/background on the same shared orders."
                          if args.common_orders else "Original full-plane raw and fixed target foreground/background from each verified report; targets differ in sampling and order partition."),
        "runs": records}), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "common_raw_white_1x": white}, indent=2))


if __name__ == "__main__":
    main()
