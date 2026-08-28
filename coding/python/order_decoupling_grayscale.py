import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import scipy.io as sio
import torch
import torch.fft as fft
import torch.optim as optim

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CUSTOM_WEIGHTS = [
    2.31, 1.77, 2.61, 2.0, 2.56, 3.0, 3.78, 3.58, 6.23, 5.36,
    10.71, 2.44, 3.37, 10.82, 2.41, 3.39, 9.35, 2.36, 4.48, 2.9,
    1.63, 1.93, 4.37,
]

TARGET_LEVELS = np.array([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float32)

PAIR_MAT = np.array([
    [3, -3], [2, -3], [1, -3],
    [3, -2], [2, -2], [1, -2], [0, -2],
    [3, -1], [2, -1], [1, -1], [0, -1],
    [3, 0], [2, 0], [1, 0],
    [3, 1], [2, 1], [1, 1],
    [3, 2], [2, 2], [1, 2],
    [3, 3], [2, 3], [1, 3],
], dtype=int)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="优化 23 通道二值或四标签目标，并输出效果图与目标图对比。"
    )
    parser.add_argument(
        "--mat-file",
        type=Path,
        default=script_dir / "grayscale_image.mat",
        help="包含 bw_all 的 MAT 文件",
    )
    parser.add_argument("--epochs", type=int, default=30000, help="优化轮数")
    parser.add_argument("--lr", type=float, default=5e-4, help="Adam 学习率")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="运行设备",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录；默认在脚本目录下按时间创建 results_23channels_*",
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--brightness-consistency-weight",
        type=float,
        default=0.0,
        help="目标白色像素原始平均强度的跨通道一致性权重；0 保持原算法",
    )
    parser.add_argument(
        "--worst-channel-weight",
        type=float,
        default=0.0,
        help="最差通道亮区平均强度落后于通道均值的惩罚权重；0 表示关闭",
    )
    return parser.parse_args()


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA。")
    return torch.device(name)


def load_targets(mat_file):
    mat = sio.loadmat(mat_file)
    if "bw_all" not in mat:
        raise KeyError(f"{mat_file} 中不存在变量 bw_all。")

    targets = np.asarray(mat["bw_all"])
    if targets.ndim != 3:
        raise ValueError(f"bw_all 应为三维数组，实际形状为 {targets.shape}。")
    if targets.shape[0] != len(CUSTOM_WEIGHTS):
        raise ValueError(
            f"通道数应为 {len(CUSTOM_WEIGHTS)}，实际为 {targets.shape[0]}。"
        )
    if targets.shape[1] != targets.shape[2]:
        raise ValueError(f"目标图应为正方形，实际形状为 {targets.shape[1:]}。")
    if not np.all(
        np.any(
            np.isclose(
                targets[..., None], TARGET_LEVELS, rtol=0, atol=1e-6
            ),
            axis=-1,
        )
    ):
        raise ValueError("bw_all 必须只包含 0、1/3、2/3、1 四个标签。")
    return targets.astype(np.float32, copy=False)


def fftshift2(x):
    return torch.roll(
        torch.roll(x, shifts=(x.shape[-2] // 2,), dims=(-2,)),
        shifts=(x.shape[-1] // 2,),
        dims=(-1,),
    )


def total_cost(
    dx,
    dy,
    targets,
    pair_mat,
    weights,
    brightness_consistency_weight=0.0,
    worst_channel_weight=0.0,
):
    base_total = torch.zeros((), device=dx.device)
    efficiencies = []
    foreground_levels = []
    epsilon = 1e-9

    for channel in range(targets.shape[0]):
        m, n = pair_mat[channel]
        field = torch.exp(1j * (m * dx + n * dy))
        intensity = torch.abs(fftshift2(fft.fft2(field))) ** 2
        efficiency = torch.sum(intensity[targets[channel] > 0]) / (
            torch.sum(intensity) + epsilon
        )
        foreground_mean = torch.mean(intensity[targets[channel] > 0])
        full_plane_mean = torch.mean(intensity)
        foreground_levels.append(foreground_mean / (full_plane_mean + epsilon))
        intensity_01 = intensity / (torch.max(intensity) + epsilon)
        difference = intensity_01 - targets[channel]
        base_total = base_total + (
            weights[channel]
            * (((1 - efficiency) * 8) ** 3)
            * torch.sum(difference**2)
        )
        efficiencies.append(efficiency)

    foreground_levels = torch.stack(foreground_levels)
    foreground_mean = torch.mean(foreground_levels)
    brightness_cv_squared = torch.mean(
        ((foreground_levels - foreground_mean) / (foreground_mean + epsilon)) ** 2
    )
    worst_channel_ratio = torch.min(foreground_levels) / (foreground_mean + epsilon)
    worst_channel_gap_squared = (1 - worst_channel_ratio) ** 2

    penalty_scale = base_total.detach()
    total = base_total + penalty_scale * (
        brightness_consistency_weight * brightness_cv_squared
        + worst_channel_weight * worst_channel_gap_squared
    )

    return (
        total,
        torch.stack(efficiencies).detach(),
        foreground_levels.detach(),
        torch.sqrt(brightness_cv_squared).detach(),
        worst_channel_ratio.detach(),
    )


def reconstruct(phdx, phdy, pair_mat):
    images = []
    with torch.no_grad():
        for m, n in pair_mat:
            field = torch.exp(1j * (m * phdx + n * phdy))
            intensity = torch.abs(fftshift2(fft.fft2(field))) ** 2
            images.append(intensity)
    raw = torch.stack(images)
    per_channel_01 = raw / (torch.amax(raw, dim=(-2, -1), keepdim=True) + 1e-9)
    global_01 = raw / (torch.max(raw) + 1e-9)
    return raw.cpu().numpy(), per_channel_01.cpu().numpy(), global_01.cpu().numpy()


def channel_metrics(raw, targets):
    rows = []
    epsilon = 1e-12
    for channel in range(targets.shape[0]):
        mask = targets[channel] > 0
        foreground = raw[channel][mask]
        background = raw[channel][~mask]
        total_energy = float(np.sum(raw[channel]))
        foreground_mean = float(np.mean(foreground))
        background_mean = float(np.mean(background))
        rows.append([
            channel + 1,
            PAIR_MAT[channel, 0],
            PAIR_MAT[channel, 1],
            np.count_nonzero(mask),
            total_energy,
            float(np.sum(foreground) / (total_energy + epsilon)),
            foreground_mean,
            background_mean,
            foreground_mean / (background_mean + epsilon),
        ])
    return np.asarray(rows, dtype=np.float64)


def save_comparison(targets, optimized_01, output_file, title):
    pairs_per_row = 4
    rows = int(np.ceil(targets.shape[0] / pairs_per_row))
    figure, axes = plt.subplots(
        rows, pairs_per_row * 2, figsize=(16, rows * 3), squeeze=False
    )

    for channel in range(targets.shape[0]):
        row = channel // pairs_per_row
        column = (channel % pairs_per_row) * 2
        for axis, image, title in (
            (axes[row, column], targets[channel], f"Ch{channel + 1} Target"),
            (axes[row, column + 1], optimized_01[channel], f"Ch{channel + 1} Optimized"),
        ):
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_title(title, fontsize=9)
            axis.axis("off")

    used_axes = targets.shape[0] * 2
    for axis in axes.flat[used_axes:]:
        axis.axis("off")

    figure.suptitle(title, fontsize=16)
    figure.tight_layout()
    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs 必须大于 0。")
    if args.log_interval < 1:
        raise ValueError("log-interval 必须大于 0。")
    if args.brightness_consistency_weight < 0:
        raise ValueError("brightness-consistency-weight 不能小于 0。")
    if args.worst_channel_weight < 0:
        raise ValueError("worst-channel-weight 不能小于 0。")

    device = select_device(args.device)
    targets_np = load_targets(args.mat_file)
    targets = torch.tensor(targets_np, device=device)
    pair_mat = torch.tensor(PAIR_MAT, dtype=torch.float32, device=device)
    weights = torch.tensor(CUSTOM_WEIGHTS, dtype=torch.float32, device=device)

    print(f"Device: {device}")
    print(f"Targets: {targets_np.shape}, values: 0, 1/3, 2/3, 1")

    torch.manual_seed(42)
    size = targets_np.shape[-1]
    phdx = (torch.rand((size, size), device=device) * 2 * np.pi).requires_grad_()
    phdy = (torch.rand((size, size), device=device) * 2 * np.pi).requires_grad_()
    optimizer = optim.Adam([phdx, phdy], lr=args.lr)

    costs = []
    efficiency_history = []
    foreground_level_history = []
    brightness_cv_history = []
    worst_channel_ratio_history = []
    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        (
            loss,
            efficiencies,
            foreground_levels,
            brightness_cv,
            worst_channel_ratio,
        ) = total_cost(
            phdx,
            phdy,
            targets,
            pair_mat,
            weights,
            args.brightness_consistency_weight,
            args.worst_channel_weight,
        )
        loss.backward()
        optimizer.step()

        costs.append(loss.item())
        efficiency_history.append(efficiencies.cpu().numpy())
        foreground_level_history.append(foreground_levels.cpu().numpy())
        brightness_cv_history.append(brightness_cv.item())
        worst_channel_ratio_history.append(worst_channel_ratio.item())
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"[Epoch {epoch}/{args.epochs}] Loss={loss.item():.6e} "
                f"BrightCV={brightness_cv.item():.4f} "
                f"Worst/Mean={worst_channel_ratio.item():.4f}"
            )

    print(f"优化完成，用时 {time.time() - started_at:.1f} 秒")

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent / f"results_23channels_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    optimized_raw, optimized_01, optimized_global_01 = reconstruct(
        phdx, phdy, pair_mat
    )
    metrics = channel_metrics(optimized_raw, targets_np)
    np.savez_compressed(
        output_dir / "optimized_results.npz",
        phdx=phdx.detach().cpu().numpy(),
        phdy=phdy.detach().cpu().numpy(),
        costs=np.asarray(costs),
        pairMat=PAIR_MAT,
        weights=np.asarray(CUSTOM_WEIGHTS),
        targets=targets_np,
        optimized_raw=optimized_raw,
        optimized_01=optimized_01,
        optimized_global_01=optimized_global_01,
        eta_history=np.asarray(efficiency_history),
        foreground_level_history=np.asarray(foreground_level_history),
        brightness_cv_history=np.asarray(brightness_cv_history),
        worst_channel_ratio_history=np.asarray(worst_channel_ratio_history),
        brightness_consistency_weight=args.brightness_consistency_weight,
        worst_channel_weight=args.worst_channel_weight,
    )
    np.savetxt(
        output_dir / "channel_brightness_metrics.csv",
        metrics,
        delimiter=",",
        header=(
            "channel,m,n,target_pixels,total_energy,target_efficiency,"
            "foreground_mean,background_mean,foreground_background_contrast"
        ),
        comments="",
    )
    comparison_file = output_dir / "target_vs_optimized_0_1.png"
    save_comparison(
        targets_np,
        optimized_01,
        comparison_file,
        "0-1 Target vs Per-Channel Normalized Result",
    )
    global_comparison_file = output_dir / "target_vs_optimized_global_scale.png"
    save_comparison(
        targets_np,
        optimized_global_01,
        global_comparison_file,
        "0-1 Target vs Globally Scaled Result",
    )

    print(f"对比图: {comparison_file}")
    print(f"统一曝光对比图: {global_comparison_file}")
    print(f"逐通道亮度指标: {output_dir / 'channel_brightness_metrics.csv'}")
    print("流程已结束：未生成相位 CSV、CIF 或后续仿真内容。")


if __name__ == "__main__":
    main()
