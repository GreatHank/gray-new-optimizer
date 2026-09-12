import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="按(m,n)共同本原方向分析图块复杂度与整数倍级次耦合。"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=project_root
        / "output"
        / "circular_7x7_target_49"
        / "circular_scene_3500.png",
    )
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--order-m-start", type=int, default=-6)
    parser.add_argument("--order-n-start", type=int, default=-3)
    parser.add_argument("--complex-count", type=int, default=12)
    parser.add_argument("--analysis-size", type=int, default=1200)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def primitive_direction(m, n):
    divisor = math.gcd(abs(m), abs(n))
    if divisor == 0:
        return 0, 0
    return m // divisor, n // divisor


def tile_complexities(gray, grid_size):
    normalized = gray.astype(np.float32) / 255.0
    dx = np.abs(np.diff(normalized, axis=1, prepend=normalized[:, :1]))
    dy = np.abs(np.diff(normalized, axis=0, prepend=normalized[:1, :]))
    detail = normalized + dx + dy
    edges = np.linspace(0, gray.shape[0], grid_size + 1, dtype=int)
    values = np.zeros((grid_size, grid_size), dtype=np.float64)
    for row in range(grid_size):
        for column in range(grid_size):
            values[row, column] = detail[
                edges[row] : edges[row + 1],
                edges[column] : edges[column + 1],
            ].sum()
    return values / values.max()


def analyze(complexities, m_start, n_start, complex_count):
    grid_size = complexities.shape[0]
    groups = defaultdict(list)
    for row in range(grid_size):
        for column in range(grid_size):
            m, n = m_start + row, n_start + column
            groups[primitive_direction(m, n)].append((row, column, m, n))

    complex_indices = set(
        np.argsort(complexities.ravel())[::-1][:complex_count].tolist()
    )
    rows = []
    for direction, members in groups.items():
        member_orders = ";".join(f"({m},{n})" for _r, _c, m, n in members)
        for row, column, m, n in members:
            flat_index = row * grid_size + column
            rows.append(
                {
                    "grid_row": row + 1,
                    "grid_column": column + 1,
                    "m": m,
                    "n": n,
                    "complexity": float(complexities[row, column]),
                    "high_complexity": flat_index in complex_indices,
                    "primitive_m": direction[0],
                    "primitive_n": direction[1],
                    "multiple_coupling_degree": len(members) - 1,
                    "multiple_group": member_orders,
                }
            )
    rows.sort(key=lambda item: (item["grid_row"], item["grid_column"]))
    return rows


def plot_analysis(gray, complexities, rows, grid_size, output_file):
    figure, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Supplied circular image: 6x6 complexity")
    height, width = gray.shape
    for index in range(1, grid_size):
        axes[0].axhline(index * height / grid_size, color="red", linewidth=0.8)
        axes[0].axvline(index * width / grid_size, color="red", linewidth=0.8)
    for row in range(grid_size):
        for column in range(grid_size):
            axes[0].text(
                (column + 0.5) * width / grid_size,
                (row + 0.5) * height / grid_size,
                f"{complexities[row, column]:.2f}",
                color="yellow",
                ha="center",
                va="center",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
            )
    axes[0].axis("off")

    colors = ["#72c472", "#f4d35e", "#f29e4c", "#e45756"]
    for item in rows:
        row = item["grid_row"] - 1
        column = item["grid_column"] - 1
        degree = item["multiple_coupling_degree"]
        color = colors[min(degree, len(colors) - 1)]
        rectangle = plt.Rectangle(
            (column, row),
            1,
            1,
            facecolor=color,
            edgecolor="black",
            linewidth=3 if item["high_complexity"] else 0.8,
        )
        axes[1].add_patch(rectangle)
        axes[1].text(
            column + 0.5,
            row + 0.36,
            f"({item['m']},{item['n']})",
            ha="center",
            va="center",
            fontweight="bold" if item["high_complexity"] else "normal",
        )
        axes[1].text(
            column + 0.5,
            row + 0.68,
            f"complex={item['complexity']:.2f}  degree={degree}",
            ha="center",
            va="center",
            fontsize=7.5,
        )
    axes[1].set_xlim(0, grid_size)
    axes[1].set_ylim(grid_size, 0)
    axes[1].set_aspect("equal")
    axes[1].set_xticks(np.arange(grid_size) + 0.5)
    axes[1].set_yticks(np.arange(grid_size) + 0.5)
    axes[1].set_xticklabels([str(item) for item in range(1, grid_size + 1)])
    axes[1].set_yticklabels([str(item) for item in range(1, grid_size + 1)])
    axes[1].set_xlabel("image grid column")
    axes[1].set_ylabel("image grid row")
    axes[1].set_title(
        "Integer-multiple coupling\nblack border = top-complexity tile; degree 0 = no multiple partner"
    )
    figure.tight_layout()
    figure.savefig(output_file, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{args.output_dir}")
    if args.grid_size < 1 or not 1 <= args.complex_count <= args.grid_size**2:
        raise ValueError("grid-size或complex-count不合法。")
    with Image.open(args.input_file) as image:
        gray = np.asarray(
            image.convert("L").resize(
                (args.analysis_size, args.analysis_size),
                Image.Resampling.NEAREST,
            )
        )
    complexities = tile_complexities(gray, args.grid_size)
    rows = analyze(
        complexities,
        args.order_m_start,
        args.order_n_start,
        args.complex_count,
    )
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "image_order_multiple_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    high = [item for item in rows if item["high_complexity"]]
    summary = {
        "grid_size": args.grid_size,
        "order_m_range": [args.order_m_start, args.order_m_start + args.grid_size - 1],
        "order_n_range": [args.order_n_start, args.order_n_start + args.grid_size - 1],
        "high_complexity_count": len(high),
        "high_complexity_without_multiple_partner": sum(
            item["multiple_coupling_degree"] == 0 for item in high
        ),
        "high_complexity_coupled_orders": [
            [item["m"], item["n"]]
            for item in high
            if item["multiple_coupling_degree"] > 0
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_analysis(
        gray,
        complexities,
        rows,
        args.grid_size,
        args.output_dir / "image_order_multiple_map.png",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
