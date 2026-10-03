"""Compare array sizes at fixed orders, source mapping and common exposure."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coding import ROOT
from coding.results.report import assemble
from coding.results.storage import json_value, load_result, result_file


def snr_by_order(raw, labels):
    """Same original nonblack/black target definition, without propagation cuts."""
    values = []
    for plane, label in zip(raw, labels):
        foreground = plane[label > 0]
        background = plane[label == 0]
        values.append(20*np.log10(max(foreground.mean()/(background.std()+1e-12), 1e-12)))
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--report-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.result_dirs) != len(args.report_dirs) or len(args.result_dirs) < 2:
        raise ValueError("Provide corresponding verified reports for at least two runs")
    output = (ROOT/args.output_dir).resolve()
    if (ROOT/"output/data").resolve() not in output.parents:
        raise ValueError("Array comparison must use a new directory under output/data")
    cases = []
    for directory, report_dir in zip(args.result_dirs, args.report_dirs):
        directory, report_dir = ROOT/directory, ROOT/report_dir
        report = json.loads((report_dir/"report.json").read_text(encoding="utf-8"))
        if len(report["runs"]) != 1 or report["runs"][0]["run"] != directory.name:
            raise ValueError("Each report must verify its corresponding run")
        data = load_result(directory)
        config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
        summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((Path(config["input"]["path"]).parent/"summary.json").read_text(encoding="utf-8"))
        if hashlib.sha256((ROOT/manifest["source"]).read_bytes()).hexdigest() != manifest["source_sha256"]:
            raise ValueError("Original source hash changed")
        cases.append((data, config, summary, manifest, report["runs"][0]))
    reference = min(cases, key=lambda case: case[0]["target_labels"].shape[-1])
    common_size = reference[0]["target_labels"].shape[-1]
    labels_ref = reference[0]["target_labels"]
    physics_keys = ("wavelength_nm", "period_x_nm", "period_y_nm", "incident_theta_deg",
                    "incident_azimuth_deg", "n_in", "n_out", "na")
    for data, config, _, manifest, _ in cases:
        size = data["target_labels"].shape[-1]
        if size % common_size or data["target_labels"].shape[-2] != size:
            raise ValueError("This comparison requires square arrays with nested integer sampling")
        np.testing.assert_array_equal(data["pairMat"], reference[0]["pairMat"])
        np.testing.assert_array_equal(manifest["canvas_positions"], reference[3]["canvas_positions"])
        np.testing.assert_array_equal([config["physics"][key] for key in physics_keys],
                                      [reference[1]["physics"][key] for key in physics_keys])
        if manifest["source_sha256"] != reference[3]["source_sha256"] or manifest["circular_field_mapping"] != reference[3]["circular_field_mapping"]:
            raise ValueError("Array sizes must retain the same source and circle mapping")
        if manifest["gray_level_count"] != reference[3]["gray_level_count"]:
            raise ValueError("Gray levels must agree")
        factor = size//common_size
        np.testing.assert_array_equal(data["target_labels"][:, ::factor, ::factor], labels_ref)
    white = 1/(labels_ref.astype(np.float64)/(reference[3]["gray_level_count"]-1)).mean()
    figure, axes = plt.subplots(len(cases), 2, figsize=(12, 6*len(cases)), squeeze=False)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#b8b8b8")
    records, order_rows = [], []
    for row, (data, config, summary, manifest, metrics) in enumerate(cases):
        labels = data["target_labels"]
        target = labels.astype(np.float64)/(manifest["gray_level_count"]-1)
        raw = data["optimized_raw"].astype(np.float64)
        size = target.shape[-1]
        pixels = size**2
        snr = snr_by_order(raw, labels)
        np.testing.assert_allclose(snr.mean(), metrics["snr_mean_db"], rtol=1e-12)
        np.testing.assert_allclose(snr.min(), metrics["snr_min_db"], rtol=1e-12)
        factor = size//common_size
        common_snr = snr_by_order(raw[:, ::factor, ::factor], labels_ref)
        gray = raw/pixels/white
        record = {"run": metrics["run"], "array_size": size, "optimized_order_count": len(snr),
                  "iterations": config["optimizer"]["epochs"], "seed": config["optimizer"]["seed"],
                  "warm_start_source": config["warm_start"]["source"],
                  "warm_start_semantics": summary["warm_start_semantics"],
                  "snr_mean_db": float(snr.mean()), "snr_min_db": float(snr.min()),
                  "common_bins_snr_mean_db": float(common_snr.mean()),
                  "common_bins_snr_min_db": float(common_snr.min()),
                  "raw_target_cosine": metrics["raw_target_cosine"],
                  "common_exposure_gray_rmse": float(np.sqrt(np.mean((gray-target)**2))),
                  "gray_consistency_mse": summary["final_gray_consistency_mse"],
                  "runtime_seconds": summary["runtime_seconds"],
                  "raw_white_1x": pixels*white,
                  "angular_fft_bin_spacing_u": config["physics"]["wavelength_nm"]/config["physics"]["n_out"]/config["physics"]["period_x_nm"]/size}
        source_iterations = 0
        if record["warm_start_source"]:
            source_summary = json.loads(result_file(record["warm_start_source"]).with_name("summary.json").read_text(encoding="utf-8"))
            source_iterations = source_summary["optimizer_work"]["iterations"]
        record["source_stage_iterations"] = source_iterations
        records.append(record)
        for order, value, matched in zip(data["pairMat"], snr, common_snr):
            order_rows.append({"run": metrics["run"], "m": int(order[0]), "n": int(order[1]),
                               "native_snr_db": float(value), "common_bins_snr_db": float(matched)})
        physics = config["physics"]
        theta, phi = np.deg2rad([physics["incident_theta_deg"], physics["incident_azimuth_deg"]])
        incident = physics["n_in"]/physics["n_out"]*np.sin(theta)*np.array([np.cos(phi), np.sin(phi)])
        pitch = physics["wavelength_nm"]/physics["n_out"]/np.array([physics["period_x_nm"], physics["period_y_nm"]])
        positions = np.asarray(manifest["canvas_positions"])
        orders = data["pairMat"]
        scenes = (assemble(target, positions), assemble(gray, positions))
        yy, xx = np.indices(scenes[0].shape)
        u = incident[0]+(orders[:, 0].min()-0.5+xx/size)*pitch[0]
        v = incident[1]+(orders[:, 1].max()+0.5-yy/size)*pitch[1]
        mask = u*u+v*v > min(1, physics["na"]/physics["n_out"])**2
        extent = (u[0, 0]-pitch[0]/(2*size), u[0, -1]+pitch[0]/(2*size),
                  v[-1, 0]-pitch[1]/(2*size), v[0, 0]+pitch[1]/(2*size))
        steps = (f"Adam {source_iterations} + {record['iterations']} steps (phase warm start)" if record["warm_start_source"]
                 else f"Adam {record['iterations']} steps")
        titles = [f"{size} x {size} target; {len(snr)} orders",
                  f"{size} x {size}, {steps}\nSNR mean/min: {snr.mean():.2f}/{snr.min():.2f} dB"]
        for axis, scene, title in zip(axes[row], scenes, titles):
            axis.set_facecolor("#b8b8b8")
            axis.imshow(np.ma.array(scene, mask=mask), cmap=cmap, vmin=0, vmax=1, extent=extent)
            axis.add_patch(plt.Circle((0, 0), 1, fill=False, color="#48b7b1", lw=0.8))
            axis.set(xlim=(-1.04, 1.04), ylim=(-1.04, 1.04), aspect="equal", title=title,
                     xlabel="u (x right)", ylabel="v (y up)")
    output.mkdir(parents=True, exist_ok=False)
    figure.suptitle(f"{Path(reference[3]['source']).stem}; same {len(reference[0]['pairMat'])} orders and source circle\n"
                    f"Common exposure: raw / (HW); white={white:.6f}; no channel gains", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output/"targets_and_results_shared_1x.png", dpi=180)
    plt.close(figure)
    for name, rows in (("metrics.csv", records), ("per_order_metrics.csv", order_rows)):
        with (output/name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output/"comparison.json").write_text(json.dumps(json_value({
        "display_scope": "Common white of I/(HW), fixed by coarsest target; raw evidence untouched. Propagation mask used only for display.",
        "common_normalized_white_1x": white, "common_angular_bin_size": common_size,
        "nested_targets_identical": True, "source_sha256": reference[3]["source_sha256"],
        "circular_field_mapping": reference[3]["circular_field_mapping"],
        "metrics_scope": "Native full-plane SNR uses original T>0/T=0 at each resolution; common-bin SNR is separately labeled and uses the same coarse angular bins and identical labels, without averaging raw.",
        "interpretation": "Single-seed size experiment. Initialization dimensions and target sampling differ; numerical improvements do not establish statistical significance. Warm-start rows label additional stage updates and restore phases with a new optimizer, not full training state.",
        "runs": records}), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": records}, indent=2))


if __name__ == "__main__":
    main()
