"""Reevaluate a full-plane baseline and an observable-only run on one fixed mask."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from coding import ROOT
from coding.geometry.observable import observable_mask
from coding.results.observable_metrics import evaluate, write_rows
from coding.results.report import assemble
from coding.results.storage import json_value, load_result


def load(directory):
    directory = (ROOT / directory).resolve()
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    return config, load_result(directory)


def independent_forward_check(data):
    raw = data["optimized_raw"]
    height, width = raw.shape[-2:]
    expected_total = (height * width)**2
    relative_parseval_error = float(np.max(np.abs(raw.sum((1, 2), dtype=np.float64) / expected_total - 1)))
    if relative_parseval_error > 2e-6:
        raise AssertionError(f"Full-plane Parseval failed: {relative_parseval_error}")
    channel = int(np.flatnonzero(np.all(data["pairMat"] == (0, 0), axis=1))[0])
    m, n = data["pairMat"][channel]
    phase = m * data["phdx"].astype(np.float64) + n * data["phdy"].astype(np.float64) + data["pp"].astype(np.float64)
    reference = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * phase))))**2
    relative_reference_error = float(np.linalg.norm(raw[channel] - reference) / np.linalg.norm(reference))
    if relative_reference_error > 2e-5:
        raise AssertionError(f"Independent NumPy forward failed: {relative_reference_error}")
    return {"max_full_plane_parseval_relative_error": relative_parseval_error,
            "zero_order_numpy_reference_relative_l2": relative_reference_error}


def region_summary(rows, center_orders, central):
    selected = [row for row in rows if ((row["m"], row["n"]) in center_orders) == central]
    return {"channel_count": len(selected),
            "masked_rmse_mean": float(np.mean([row["masked_rmse"] for row in selected])),
            "raw_foreground_mean": float(np.nanmean([row["raw_foreground_mean"] for row in selected])),
            "raw_background_mean": float(np.nanmean([row["raw_background_mean"] for row in selected])),
            "snr_db_mean": float(np.nanmean([row["snr_db"] for row in selected])),
            "observable_energy_fraction_mean": float(np.mean([row["observable_energy_fraction"] for row in selected]))}


def central_orders(orders, positions):
    grid_size = int(positions.max()) + 1
    center_cells = {grid_size // 2} if grid_size % 2 else {grid_size // 2 - 1, grid_size // 2}
    pairs = {tuple(map(int, order)) for order, (row, col) in zip(orders, positions)
             if row in center_cells and col in center_cells}
    return grid_size, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if (ROOT / "output/data").resolve() not in output.parents or output.exists():
        raise ValueError("Choose a new output/data subdirectory")
    old_config, old = load(args.baseline)
    new_config, new = load(args.candidate)
    for field in ("epochs", "lr", "optimizer_name", "gray_consistency_weight", "seed", "gray_level_count", "initial_results"):
        if old_config["optimizer"][field] != new_config["optimizer"][field]:
            raise ValueError(f"Unmatched optimizer parameter: {field}")
    if old_config["input"]["sha256"] != new_config["input"]["sha256"]:
        raise ValueError("Baseline and candidate must have the same target MAT")
    if old_config["physics"] != new_config["physics"]:
        physics_old = {k: v for k, v in old_config["physics"].items() if k != "square_target_scope"}
        physics_new = {k: v for k, v in new_config["physics"].items() if k != "square_target_scope"}
        if physics_old != physics_new:
            raise ValueError("Baseline and candidate physical parameters differ")
    for key in ("pairMat", "grid_positions", "target_labels"):
        np.testing.assert_array_equal(old[key], new[key])
    mask = observable_mask(old["pairMat"], old["targets"].shape[-2:], old_config["physics"])
    np.testing.assert_array_equal(mask, new["observable_mask"])
    old_rows, old_gray, old_summary = evaluate(old["optimized_raw"], old["targets"], old["target_labels"], old["pairMat"], mask)
    new_rows, new_gray, new_summary = evaluate(new["optimized_raw"], new["targets"], new["target_labels"], new["pairMat"], mask)
    target_dir = Path(old_config["input"]["path"]).parent
    manifest = json.loads((target_dir / "summary.json").read_text(encoding="utf-8"))
    positions = np.asarray(manifest["canvas_positions"])
    np.testing.assert_array_equal(np.rint(assemble(old["targets"], positions) * 255).astype(np.uint8),
                                  np.asarray(Image.open(target_dir / "target.png")))
    grid_size, center_orders = central_orders(old["pairMat"], positions)
    output.mkdir(parents=True)
    write_rows(output / "baseline_masked_metrics.csv", old_rows)
    write_rows(output / "candidate_masked_metrics.csv", new_rows)
    write_rows(output / "baseline_gray_levels.csv", old_gray)
    write_rows(output / "candidate_gray_levels.csv", new_gray)
    report = {"scope": "Same fixed propagation/NA mask, same target and common raw display scale; full FFT is retained",
              "baseline": str(args.baseline), "candidate": str(args.candidate),
              "mask_pixels": int(mask.sum()), "mask_fraction": float(mask.mean()),
              "baseline_summary": old_summary, "candidate_summary": new_summary,
              "center_orders": sorted(center_orders),
              "central": {"baseline": region_summary(old_rows, center_orders, True),
                          "candidate": region_summary(new_rows, center_orders, True)},
              "peripheral": {"baseline": region_summary(old_rows, center_orders, False),
                             "candidate": region_summary(new_rows, center_orders, False)},
              "forward_checks": {"baseline": independent_forward_check(old), "candidate": independent_forward_check(new)},
              "physical_boundary": "Masked energy fraction is a software FFT distribution measure, not device efficiency; outside-mask spectrum is not a reservoir of propagating power."}
    (output / "comparison.json").write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    mask_canvas = ~assemble(mask, positions)
    white = old["targets"].mean() / np.prod(old["targets"].shape[-2:])
    images = (old["targets"], old["optimized_raw"] * white, new["optimized_raw"] * white)
    labels = ("Target", "Full-plane loss", "Observable-only loss")
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#b8b8b8")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, image, title in zip(axes, images, labels):
        axis.imshow(np.ma.array(assemble(image, positions), mask=mask_canvas), cmap=cmap,
                    vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{Path(manifest['source']).stem} {grid_size}x{grid_size}, "
                 f"{old['targets'].shape[-1]}x{old['targets'].shape[-2]}; "
                 "fixed observable mask; shared 1x exposure")
    fig.tight_layout()
    fig.savefig(output / "comparison_1x.png", dpi=220)
    plt.close(fig)
    print(json.dumps(json_value(report), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
