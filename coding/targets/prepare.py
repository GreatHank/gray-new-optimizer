"""Map source images into the configured PP-enabled diffraction-order grid."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.integrate import quad
from scipy.ndimage import map_coordinates

from coding.geometry.preflight import validate_physics
from coding import ROOT


def circle_rectangle_area(bounds, radius):
    """Continuous circle area inside one diffraction-order rectangle."""
    left, right, bottom, top = bounds
    start, end = max(left, -radius), min(right, radius)
    if start >= end:
        return 0.0
    points = [x for y in (bottom, top) if abs(y) < radius
              for x in (-np.sqrt(radius**2-y**2), np.sqrt(radius**2-y**2))
              if start < x < end]
    def height(x):
        edge = np.sqrt(max(0, radius**2-x*x))
        return max(0, min(top, edge)-max(bottom, -edge))
    return quad(height, start, end, points=points, epsabs=1e-10, epsrel=1e-10)[0]



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/circle-ocean-3x3-p800-500-strict9.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    level_count = config["optimizer"]["gray_level_count"]
    tile_size = config["target"]["tile_size"]
    grid_size = config["target"]["grid_size"]
    m_start = config["target"]["order_m_start"]
    n_start = config["target"]["order_n_start"]
    if level_count not in (4, 16, 256) or not isinstance(tile_size, int) or tile_size < 8:
        raise ValueError("gray_level_count must be 4, 16 or 256; tile_size must be an integer >= 8")
    if not isinstance(grid_size, int) or grid_size < 1 or any(
            not isinstance(value, int) for value in (m_start, n_start)):
        raise ValueError("grid_size must be a positive integer; order starts must be integers")
    output = (ROOT/(args.output_dir or Path(config["optimizer"]["mat_file"]).parent)).resolve()
    if (ROOT/"output/targets").resolve() not in output.parents:
        raise ValueError("Derived targets must use a new directory under output/targets")
    source = ROOT/config["target"]["source"]
    original = Image.open(source).convert("L")
    physics = config["physics"]
    pitch = physics["wavelength_nm"]/physics["n_out"]/physics["period_x_nm"]
    pitch_y = physics["wavelength_nm"]/physics["n_out"]/physics["period_y_nm"]
    theta, azimuth = np.deg2rad([physics["incident_theta_deg"], physics["incident_azimuth_deg"]])
    incident_u = physics["n_in"]/physics["n_out"]*np.sin(theta)*np.cos(azimuth)
    incident_v = physics["n_in"]/physics["n_out"]*np.sin(theta)*np.sin(azimuth)
    canvas_size = grid_size * tile_size
    circle = config["target"].get("source_circle")
    require_full_circle = config["target"].get("require_full_field_coverage", True)
    if not isinstance(require_full_circle, bool):
        raise ValueError("require_full_field_coverage must be a boolean")
    circle_mapping = None
    if circle is None:
        resized_size = config["target"].get("resized_size", [canvas_size, canvas_size])
        offset = config["target"].get("canvas_offset", [(canvas_size-value)//2 for value in resized_size])
        if len(resized_size) != 2 or len(offset) != 2 or any(
                not isinstance(value, int) for value in [*resized_size, *offset]) or any(
                size <= 0 or start < 0 or start+size > canvas_size
                for size, start in zip(resized_size, offset)):
            raise ValueError("The complete resized source must fit inside the canvas without cropping")
        resized = original.resize(tuple(resized_size), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (canvas_size, canvas_size))
        canvas.paste(resized, tuple(offset))
        preprocessing = f"Complete source globally resized from {original.width}x{original.height} to {resized_size[0]}x{resized_size[1]}, placed at {offset} on a {canvas_size}x{canvas_size} black canvas. Excluded cells contain only black background. No source cropping, brightness balancing, per-tile gains, rotation, or mirroring."
    else:
        center = np.asarray(circle["center_px"], dtype=np.float64)
        radius = float(circle["radius_px"])
        background_max = circle["outside_background_max_gray"]
        if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
            raise ValueError("source_circle requires a finite center and positive radius")
        if not isinstance(background_max, int) or not 0 <= background_max <= 255:
            raise ValueError("outside_background_max_gray must be an 8-bit integer")
        source_labels = np.asarray(original).copy()
        sy, sx = np.indices(source_labels.shape)
        source_outside = (sx-center[0])**2+(sy-center[1])**2 > radius**2
        removed_max = int(source_labels[source_outside].max(initial=0))
        if removed_max > background_max:
            raise ValueError("source_circle would remove pixels brighter than the declared outside background")
        removed_count = int(np.count_nonzero(source_labels[source_outside]))
        source_labels[source_outside] = 0
        field_radius = min(1, physics["na"]/physics["n_out"])
        bounds = (incident_u+(m_start-0.5)*pitch, incident_u+(m_start+grid_size-0.5)*pitch,
                  incident_v+(n_start-0.5)*pitch_y, incident_v+(n_start+grid_size-0.5)*pitch_y)
        if require_full_circle and (bounds[0] > -field_radius or bounds[1] < field_radius or bounds[2] > -field_radius or bounds[3] < field_radius):
            raise ValueError("Selected canvas cannot contain the complete field circle; include the rim orders")
        yy, xx = np.indices((canvas_size, canvas_size))
        u = incident_u+(m_start-0.5+xx/tile_size)*pitch
        v = incident_v+(n_start+grid_size-0.5-yy/tile_size)*pitch_y
        field_mask = u*u+v*v <= field_radius**2
        mapped = map_coordinates(source_labels,
            [center[1]-radius*v/field_radius, center[0]+radius*u/field_radius],
            order=1, mode="constant", cval=0, prefilter=False)
        mapped[~field_mask] = 0
        canvas = Image.fromarray(mapped)
        resized_size, offset = None, None
        circle_mapping = {"source_center_px": center.tolist(), "source_radius_px": radius,
                          "field_radius_u": field_radius, "outside_source_nonblack_pixels_removed": removed_count,
                          "require_full_field_coverage": require_full_circle,
                          "outside_source_removed_max_gray": removed_max,
                          "sampling": "bilinear at exact FFT bins; source x=cx+R*u/field_radius, y=cy-R*v/field_radius"}
        preprocessing = "User-requested source circle mapped exactly to the full physical field circle; outside-source residual background removed up to the declared gray limit. Circle interior brightness preserved, with global bilinear sampling; no per-channel gains or target balancing. Outside-field target is black by definition; metrics use this explicitly new target."
        if not require_full_circle:
            preprocessing += " Only the requested order window is retained; directions outside that grid are unconstrained, not zero-valued optimized channels. Full field-circle coverage is not required."
    labels = np.rint(np.asarray(canvas).astype(np.float64) / 255 * (level_count - 1)).astype(np.uint8)
    excluded = config["target"].get("excluded_canvas_positions", [])
    if any(len(pair) != 2 or any(not isinstance(value, int) or not 0 <= value < grid_size
                               for value in pair) for pair in excluded):
        raise ValueError("Excluded canvas positions must be integer row/column pairs in the grid")
    excluded = set(map(tuple, excluded))
    if circle is not None and require_full_circle:
        for r, c in excluded:
            left = bounds[0]+c*pitch
            bottom = bounds[3]-(r+1)*pitch_y
            nearest_u = np.clip(0, left, left+pitch)
            nearest_v = np.clip(0, bottom, bottom+pitch_y)
            if nearest_u**2+nearest_v**2 <= field_radius**2:
                raise ValueError("Excluded orders leave part of the field circle uncovered")
    if any(np.any(labels[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size])
           for r, c in excluded):
        raise ValueError("Excluded orders contain source foreground; excluding them would crop the image")
    canvas_positions = np.asarray([(r, c) for r in range(grid_size) for c in range(grid_size)
                                   if (r, c) not in excluded])
    if not len(canvas_positions):
        raise ValueError("At least one diffraction order must be selected")
    if circle is not None:
        coverage = 1.0 if require_full_circle else sum(circle_rectangle_area(
            (bounds[0]+c*pitch, bounds[0]+(c+1)*pitch,
             bounds[3]-(r+1)*pitch_y, bounds[3]-r*pitch_y), field_radius)
            for r, c in canvas_positions)/(np.pi*field_radius**2)
        circle_mapping["field_circle_grid_coverage_fraction"] = float(coverage)
        circle_mapping["field_circle_uncovered_area_fraction"] = float(1-coverage)
    orders = np.asarray([(m_start + c, n_start + grid_size - 1 - r)
                         for r, c in canvas_positions], dtype=np.int16)
    zero_indices = np.flatnonzero(np.all(orders == 0, axis=1))
    zero_channel = int(zero_indices[0] + 1) if zero_indices.size else None
    tiles = np.stack([labels[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size]
                      for r, c in canvas_positions])
    if not np.any(tiles) or any(np.all(tile > 0) for tile in tiles):
        raise ValueError("The scene needs foreground and every tile needs black background")
    targets = tiles.astype(np.float32) / (level_count - 1)
    check = validate_physics(orders, physics, targets=targets)
    if check["strict_failures"]:
        raise ValueError(check)
    output.mkdir(parents=True, exist_ok=False)
    sio.savemat(output / "target.mat", {
        "bw_all": targets, "grid_positions": orders - [m_start, n_start],
        "order_pairs": orders, "canvas_positions": canvas_positions,
        "gray_level_count": level_count,
        "mapping_transform": preprocessing,
    }, do_compression=True)
    canvas.save(output / "source_resized.png")
    Image.fromarray(np.rint(labels.astype(np.float64) * 255 / (level_count - 1)).astype(np.uint8)).save(output / "target.png")
    manifest = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_size": list(original.size), "canvas_size": list(canvas.size),
        "resized_size": list(resized_size) if resized_size is not None else None,
        "source_box": [*offset, offset[0]+resized_size[0], offset[1]+resized_size[1]] if offset is not None else None,
        "tile_size": tile_size, "grid_size": grid_size, "channel_count": len(orders),
        "phase_array_shape": [tile_size, tile_size],
        "shared_phase_parameter_count": 3 * tile_size ** 2,
        "gray_level_count": level_count, "levels": np.rint(np.linspace(0, 255, level_count)).astype(int).tolist(),
        "canvas_positions": canvas_positions.tolist(), "orders": orders.tolist(),
        "background_only_channels_1based": (np.flatnonzero(~np.any(tiles, axis=(1, 2)))+1).tolist(),
        "zero_order_channel_1based": zero_channel,
        "layout_label": config["target"].get("layout_label", f"{grid_size} x {grid_size}"),
        "circular_field_mapping": circle_mapping,
        "preprocessing": preprocessing,
        "physics_check": check,
    }
    (output / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    extent = (incident_u+(m_start-0.5)*pitch, incident_u+(m_start+grid_size-0.5)*pitch,
              incident_v+(n_start-0.5)*pitch_y, incident_v+(n_start+grid_size-0.5)*pitch_y)
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(labels, extent=extent, cmap="gray", vmin=0, vmax=level_count-1)
    for value in range(grid_size + 1):
        axis.axhline(incident_v+(n_start-0.5+value)*pitch_y, color="#63bed2", lw=0.5)
        axis.axvline(incident_u+(m_start-0.5+value)*pitch, color="#63bed2", lw=0.5)
    for m, n in orders:
        axis.text(incident_u+m*pitch, incident_v+n*pitch_y, f"({m},{n})", color="#ffdf59", ha="center", fontsize=9)
    axis.add_patch(plt.Circle((0, 0), 1, fill=False, color="#c44c4c", lw=1.5))
    ring_angle = physics.get("layout_requirements", {}).get("ring_theta_deg")
    if ring_angle is not None:
        axis.add_patch(plt.Circle((0, 0), np.sin(np.deg2rad(ring_angle)), fill=False, color="#c44c4c", ls="--"))
    axis.set(xlabel="u (x right)", ylabel="v (y up)", aspect="equal",
             title=f"{'Full field-circle' if circle_mapping and require_full_circle else 'Selected circular grid' if circle_mapping else 'Full square'} target; {len(orders)} orders; phase array {tile_size} x {tile_size}\n"
                   f"{check['center_propagating']}/{len(orders)} centers propagate; red circle is propagation boundary")
    figure.tight_layout()
    figure.savefig(output / "layout_preview.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"target": str(output), "config": str(args.config),
                      "zero_order_channel": zero_channel,
                      "layout": {key: value for key, value in check["layout"].items()
                                 if key != "target_support_by_order"}}, indent=2))


if __name__ == "__main__":
    main()
