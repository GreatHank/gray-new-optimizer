"""Publish one target/reconstruction image per optimization, grouped by source."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from coding import ROOT
from coding.geometry.observable import observable_mask
from coding.results.report import assemble
from coding.results.storage import load_result, json_value, result_file


SOURCE_NAMES = {
    "tiger_detailed": "老虎", "circle_cosmic_gray": "圆形星空",
    "circle_ocean_gray": "乌龟", "circle_forest_gray": "山涧",
    "circle_peony_gray": "牡丹", "circle_dragon_gray": "盘龙",
    "circle_phoenix_gray": "凤凰",
}


def total_updates(directory):
    """Count the recorded phase warm-start stages, with a new optimizer per stage."""
    config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
    count = config["optimizer"]["epochs"]
    parent = config["warm_start"]["source"]
    if parent:
        count += total_updates(result_file(ROOT/parent).parent)
    return count


def publish(directory):
    directory = (ROOT/directory).resolve()
    if (ROOT/"output/data").resolve() not in directory.parents:
        raise ValueError("Publish an optimization stored under output/data")
    config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
    manifest = json.loads((Path(config["input"]["path"]).parent/"summary.json").read_text(encoding="utf-8"))
    summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    data = load_result(directory)
    target = data["targets"]
    orders = data["pairMat"]
    np.testing.assert_array_equal(orders, manifest["orders"])
    positions = np.asarray(manifest["canvas_positions"])
    height, width = target.shape[-2:]
    gray = data["optimized_raw"].astype(np.float64)*float(target.mean())/(height*width)
    physics = config["physics"]
    visible = observable_mask(orders, (height, width), physics)
    if "observable_mask" in data:
        np.testing.assert_array_equal(visible, data["observable_mask"])
    mask = ~assemble(visible, positions)
    source = Path(manifest["source"])
    name = SOURCE_NAMES.get(source.stem, source.stem)
    channel_count = len(orders)
    layout = {9: "严格3×3（9级次）", 13: "3×3加外围四级次（共13级次）",
              16: "4×4（16级次）"}.get(channel_count, f"{channel_count}级次")
    mode = "视场圆映射" if manifest.get("circular_field_mapping") else "完整原图"
    updates = total_updates(directory)
    steps = (f"累计{updates}步（续跑{config['optimizer']['epochs']}步）"
             if config["warm_start"]["source"] else f"{updates}步")
    optimizer = config["optimizer"]
    timestamp = datetime.fromtimestamp(result_file(directory).stat().st_mtime,
                                       timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    constraint = "_圆内约束" if optimizer.get("spatial_constraint") == "observable" else ""
    label = f"{layout}_{width}×{height}_{steps}_P{physics['period_x_nm']:g}_{mode}{constraint}_{optimizer['optimizer_name'].upper()}_seed{optimizer['seed']}"
    output = ROOT/"output/runs"/name/f"{label}_{timestamp}.png"
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    snr = summary["metrics"]["foreground_background_snr_db_mean"]
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#b8b8b8")
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.8))
    for axis, plane, title in zip(axes, (assemble(target, positions), assemble(gray, positions)),
                                  ("目标 Target", f"优化效果｜{'圆内' if constraint else '全平面'}SNR {snr:.2f} dB｜统一曝光 1×")):
        axis.imshow(np.ma.array(plane, mask=mask), cmap=cmap, vmin=0, vmax=1,
                    interpolation="nearest")
        axis.set_title(title, fontproperties=font, fontsize=13)
        axis.axis("off")
    figure.suptitle(f"{name}｜{layout}｜{width}×{height}｜{steps}\n灰色区域不属于所选级次的可传播视场",
                   fontproperties=font, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .92))
    figure.savefig(output, dpi=250)
    plt.close(figure)
    index_path = ROOT/"configs/selected-results.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index[output.relative_to(ROOT/"output/runs").as_posix()] = {
        "data_dir": directory.relative_to(ROOT).as_posix(), "source": source.as_posix(),
        "description": label, "snr_mean_db": snr,
        "metric_scope": "observable" if constraint else "full_plane",
    }
    index_path.write_text(json.dumps(json_value(index), ensure_ascii=False, indent=2,
                                    allow_nan=False)+"\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dirs", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    for directory in args.result_dirs:
        print(publish(directory))


if __name__ == "__main__":
    main()
