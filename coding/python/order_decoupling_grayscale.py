import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import scipy.io as sio
import torch
import torch.fft as fft
import torch.nn.functional as F
import torch.optim as optim

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CUSTOM_WEIGHTS = np.ones(36, dtype=np.float32)

TARGET_LEVELS = np.array([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float32)

PAIR_MAT = np.array(
    [(m, n) for m in range(1, 7) for n in range(1, 7)], dtype=int
)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="优化最多36通道二值或四标签目标，并输出效果图与目标图对比。"
    )
    parser.add_argument(
        "--mat-file",
        type=Path,
        default=script_dir / "grayscale_image.mat",
        help="包含 bw_all 的 MAT 文件",
    )
    parser.add_argument("--epochs", type=int, default=30000, help="优化轮数")
    parser.add_argument("--lr", type=float, default=5e-4, help="Adam 学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机初始化种子")
    parser.add_argument(
        "--channel-count",
        type=int,
        default=len(PAIR_MAT),
        help="只优化前 N 个通道，用于逐步检查多通道可行性",
    )
    parser.add_argument(
        "--image-loss-mode",
        choices=("energy", "sum", "balanced"),
        default="energy",
        help="图案损失模式；energy 按目标灰度分配总衍射能量，适合稀疏灰度目标",
    )
    parser.add_argument(
        "--foreground-loss-weight",
        type=float,
        default=1.0,
        help="balanced 图案损失中的前景权重",
    )
    parser.add_argument(
        "--background-loss-weight",
        type=float,
        default=1.0,
        help="balanced 图案损失中的背景权重",
    )
    parser.add_argument(
        "--foreground-efficiency-weight",
        type=float,
        default=0.0,
        help="直接提高目标前景能量占比的损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--structure-completeness-weight",
        type=float,
        default=0.0,
        help="逐目标像素最低灰度响应损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--gray-ratio-weight",
        type=float,
        default=0.0,
        help="每通道三档相对比例损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="运行设备",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录；默认在脚本目录下按时间创建 results_36channels_*",
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
    parser.add_argument(
        "--level-weight",
        type=float,
        default=0.0,
        help="三档公共比例损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--cross-level-weight",
        type=float,
        default=0.0,
        help="同档跨通道有效亮度一致性损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--gap-weight",
        type=float,
        default=0.0,
        help="三档最小间隔损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--line-uniformity-weight",
        type=float,
        default=0.0,
        help="线条内部强度方差损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--visibility-weight",
        type=float,
        default=0.0,
        help="最低灰度可见性损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--worst-level-weight",
        type=float,
        default=0.0,
        help="最差通道/灰度档低于公共目标的损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--background-uniformity-weight",
        type=float,
        default=0.0,
        help="背景空间方差损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--reference-q",
        type=float,
        default=0.0,
        help="固定公共最高档有效响应尺度；0 表示使用当轮统计量",
    )
    parser.add_argument(
        "--background-row-uniformity-weight",
        type=float,
        default=0.0,
        help="背景逐行均值方差损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--background-band-weight",
        type=float,
        default=0.0,
        help="所有背景像素亮度带损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--background-band-lower",
        type=float,
        default=0.0,
        help="背景亮度带下限，按 I/全平面均值归一化",
    )
    parser.add_argument(
        "--background-band-upper",
        type=float,
        default=0.0,
        help="背景亮度带上限，按 I/全平面均值归一化；0 表示关闭亮度带",
    )
    parser.add_argument(
        "--background-cluster-weight",
        type=float,
        default=0.0,
        help="背景局部亮度集中损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--background-cluster-kernel",
        type=int,
        default=9,
        help="背景局部亮度集中损失的邻域边长，必须为大于等于 3 的奇数",
    )
    parser.add_argument(
        "--background-cluster-upper",
        type=float,
        default=0.0,
        help="背景邻域平均亮度上限，按 I/全平面均值归一化；0 表示关闭",
    )
    parser.add_argument(
        "--initial-results",
        type=Path,
        help="包含 phdx/phdy 的 NPZ 初始结果；不指定时使用随机初始化",
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
    if not 1 <= targets.shape[0] <= len(CUSTOM_WEIGHTS):
        raise ValueError(
            f"通道数应在1到{len(CUSTOM_WEIGHTS)}之间，实际为{targets.shape[0]}。"
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


def energy_distribution_loss(intensity, target, epsilon=1e-9):
    foreground_mask = target > 0
    desired = target[foreground_mask] / torch.sum(target)
    actual = intensity[foreground_mask] / torch.sum(intensity)
    return torch.sum(
        desired * (torch.log(desired + epsilon) - torch.log(actual + epsilon))
    )


def total_cost(
    dx,
    dy,
    targets,
    pair_mat,
    weights,
    brightness_consistency_weight=0.0,
    worst_channel_weight=0.0,
    level_weight=0.0,
    cross_level_weight=0.0,
    gap_weight=0.0,
    line_uniformity_weight=0.0,
    visibility_weight=0.0,
    worst_level_weight=0.0,
    background_uniformity_weight=0.0,
    reference_q=0.0,
    background_row_uniformity_weight=0.0,
    background_band_weight=0.0,
    background_band_lower=0.0,
    background_band_upper=0.0,
    background_cluster_weight=0.0,
    background_cluster_kernel=9,
    background_cluster_upper=0.0,
    image_loss_mode="energy",
    foreground_loss_weight=1.0,
    background_loss_weight=1.0,
    foreground_efficiency_weight=0.0,
    structure_completeness_weight=0.0,
    gray_ratio_weight=0.0,
):
    base_total = torch.zeros((), device=dx.device)
    efficiencies = []
    foreground_levels = []
    level_responses = []
    line_uniformities = []
    background_uniformities = []
    background_row_uniformities = []
    background_band_losses = []
    background_cluster_losses = []
    foreground_contrasts = []
    foreground_targets = []
    epsilon = 1e-9
    target_levels = torch.tensor(
        [1 / 3, 2 / 3, 1.0], dtype=targets.dtype, device=targets.device
    )
    has_three_levels = all(
        bool(torch.any(targets == level).item()) for level in target_levels
    )

    for channel in range(targets.shape[0]):
        m, n = pair_mat[channel]
        field = torch.exp(1j * (m * dx + n * dy))
        intensity = torch.abs(fftshift2(fft.fft2(field))) ** 2
        target = targets[channel]
        foreground_mask = target > 0
        background_mask = target == 0
        efficiency = torch.sum(intensity[foreground_mask]) / (
            torch.sum(intensity) + epsilon
        )
        foreground_mean = torch.mean(intensity[foreground_mask])
        full_plane_mean = torch.mean(intensity)
        foreground_levels.append(foreground_mean / (full_plane_mean + epsilon))
        if has_three_levels:
            line_means = torch.stack(
                [torch.mean(intensity[target == level]) for level in target_levels]
            )
        else:
            line_mean = torch.mean(intensity[foreground_mask])
            line_means = torch.stack([line_mean, line_mean, line_mean])
        background_mean = torch.mean(intensity[background_mask])
        responses = (line_means - background_mean) / (
            full_plane_mean.detach() + epsilon
        )
        level_responses.append(responses)
        foreground_contrasts.append(
            torch.relu(
                (intensity[foreground_mask] - background_mean)
                / (full_plane_mean.detach() + epsilon)
            )
        )
        foreground_targets.append(target[foreground_mask])
        if has_three_levels:
            level_variances = torch.stack(
                [
                    torch.var(intensity[target == level], unbiased=False)
                    / (line_mean.detach() ** 2 + epsilon)
                    for level, line_mean in zip(target_levels, line_means)
                ]
            )
        else:
            foreground_variance = torch.var(
                intensity[foreground_mask], unbiased=False
            ) / (foreground_mean.detach() ** 2 + epsilon)
            level_variances = torch.stack(
                [foreground_variance, foreground_variance, foreground_variance]
            )
        line_uniformities.append(torch.mean(level_variances))
        background_uniformities.append(
            torch.var(intensity[background_mask], unbiased=False)
            / (full_plane_mean.detach() ** 2 + epsilon)
        )
        background_weights = background_mask.to(intensity.dtype)
        row_counts = torch.sum(background_weights, dim=-1)
        background_row_means = torch.sum(
            intensity * background_weights, dim=-1
        ) / (row_counts + epsilon)
        background_row_uniformities.append(
            torch.var(background_row_means, unbiased=False)
            / (full_plane_mean.detach() ** 2 + epsilon)
        )
        normalized_intensity = intensity / (full_plane_mean.detach() + epsilon)
        upper_violation = torch.relu(normalized_intensity - background_band_upper)
        lower_violation = torch.relu(background_band_lower - normalized_intensity)
        background_band_losses.append(
            torch.mean(
                upper_violation[background_mask] ** 2
                + lower_violation[background_mask] ** 2
            )
        )
        kernel = background_cluster_kernel
        pooled_intensity = F.avg_pool2d(
            (normalized_intensity * background_mask).unsqueeze(0).unsqueeze(0),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )
        pooled_mask = F.avg_pool2d(
            background_mask.to(intensity.dtype).unsqueeze(0).unsqueeze(0),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )
        local_background_mean = pooled_intensity / (pooled_mask + epsilon)
        cluster_violation = torch.relu(
            local_background_mean.squeeze(0).squeeze(0)
            - background_cluster_upper
        )
        background_cluster_losses.append(
            torch.mean(cluster_violation[background_mask] ** 2)
        )
        if image_loss_mode == "energy":
            image_error = energy_distribution_loss(intensity, target, epsilon)
            channel_cost = weights[channel] * image_error
        else:
            intensity_01 = intensity / (torch.max(intensity) + epsilon)
            difference = intensity_01 - target
            if image_loss_mode == "balanced":
                foreground_error = torch.mean(difference[foreground_mask] ** 2)
                background_error = torch.mean(difference[background_mask] ** 2)
                image_error = (
                    foreground_loss_weight * foreground_error
                    + background_loss_weight * background_error
                ) / (foreground_loss_weight + background_loss_weight)
            else:
                image_error = torch.sum(difference**2)
            channel_cost = (
                weights[channel]
                * (((1 - efficiency) * 8) ** 3)
                * image_error
            )
        base_total = base_total + channel_cost
        efficiencies.append(efficiency)

    foreground_levels = torch.stack(foreground_levels)
    efficiencies = torch.stack(efficiencies)
    level_responses = torch.stack(level_responses)
    line_uniformity_loss = torch.mean(torch.stack(line_uniformities))
    background_uniformity_loss = torch.mean(torch.stack(background_uniformities))
    background_row_uniformity_loss = torch.mean(
        torch.stack(background_row_uniformities)
    )
    background_band_loss = torch.mean(torch.stack(background_band_losses))
    background_cluster_loss = torch.mean(torch.stack(background_cluster_losses))
    foreground_mean = torch.mean(foreground_levels)
    brightness_cv_squared = torch.mean(
        ((foreground_levels - foreground_mean) / (foreground_mean + epsilon)) ** 2
    )
    worst_channel_ratio = torch.min(foreground_levels) / (foreground_mean + epsilon)
    worst_channel_gap_squared = (1 - worst_channel_ratio) ** 2

    if reference_q > 0:
        q = torch.tensor(reference_q, dtype=targets.dtype, device=targets.device)
    else:
        q = torch.clamp(
            torch.mean(level_responses[:, 2]).detach(), min=epsilon
        )
    level_targets = q * target_levels
    structure_completeness_loss = torch.mean(
        torch.stack(
            [
                torch.mean(
                    (
                        torch.relu(q * target_values - contrasts)
                        / (q + epsilon)
                    )
                    ** 2
                )
                for contrasts, target_values in zip(
                    foreground_contrasts, foreground_targets
                )
            ]
        )
    )
    if has_three_levels:
        channel_q = torch.clamp(
            torch.abs(level_responses[:, 2].detach()), min=epsilon
        )
        gray_ratio_loss = torch.mean(
            (
                level_responses / channel_q.unsqueeze(1)
                - target_levels.unsqueeze(0)
            )
            ** 2
        )
        level_loss = torch.mean(
            ((level_responses - level_targets) / (q + epsilon)) ** 2
        )
        level_means = torch.mean(level_responses, dim=0).detach()
        cross_level_loss = torch.mean(
            ((level_responses - level_means) / (torch.abs(level_means) + epsilon)) ** 2
        )
        gaps = torch.diff(level_responses, dim=1)
        minimum_gap = q / 6
        gap_loss = torch.mean(
            (torch.relu(minimum_gap - gaps) / (q + epsilon)) ** 2
        )
        visibility_margin = q / 10
        visibility_loss = torch.mean(
            (torch.relu(visibility_margin - level_responses[:, 0]) / (q + epsilon))
            ** 2
        )
        worst_level_loss = torch.mean(
            (torch.relu(level_targets - level_responses) / (q + epsilon)) ** 2
        )
    else:
        zero = torch.zeros((), dtype=targets.dtype, device=targets.device)
        gray_ratio_loss = zero
        level_loss = zero
        cross_level_loss = zero
        gap_loss = zero
        visibility_loss = zero
        worst_level_loss = zero

    penalty_scale = base_total.detach()
    foreground_efficiency_loss = torch.mean((1 - efficiencies) ** 2)
    total = base_total + penalty_scale * (
        brightness_consistency_weight * brightness_cv_squared
        + worst_channel_weight * worst_channel_gap_squared
        + level_weight * level_loss
        + cross_level_weight * cross_level_loss
        + gap_weight * gap_loss
        + line_uniformity_weight * line_uniformity_loss
        + visibility_weight * visibility_loss
        + worst_level_weight * worst_level_loss
        + background_uniformity_weight * background_uniformity_loss
        + background_row_uniformity_weight * background_row_uniformity_loss
        + background_band_weight * background_band_loss
        + background_cluster_weight * background_cluster_loss
        + foreground_efficiency_weight * foreground_efficiency_loss
        + structure_completeness_weight * structure_completeness_loss
        + gray_ratio_weight * gray_ratio_loss
    )

    return (
        total,
        efficiencies.detach(),
        foreground_levels.detach(),
        torch.sqrt(brightness_cv_squared).detach(),
        worst_channel_ratio.detach(),
        level_responses.detach(),
        q.detach(),
        level_loss.detach(),
        cross_level_loss.detach(),
        gap_loss.detach(),
        line_uniformity_loss.detach(),
        visibility_loss.detach(),
        worst_level_loss.detach(),
        background_uniformity_loss.detach(),
        background_row_uniformity_loss.detach(),
        background_band_loss.detach(),
        background_cluster_loss.detach(),
        structure_completeness_loss.detach(),
        gray_ratio_loss.detach(),
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


def grayscale_metrics(raw, targets):
    rows = []
    epsilon = 1e-12
    levels = (1 / 3, 2 / 3, 1.0)
    for channel in range(targets.shape[0]):
        target = targets[channel]
        background = raw[channel][target == 0]
        background_mask = target == 0
        row_weights = background_mask.astype(np.float64)
        row_counts = np.sum(row_weights, axis=1)
        background_row_means = np.sum(
            raw[channel] * row_weights, axis=1
        ) / (row_counts + epsilon)
        if all(np.any(target == level) for level in levels):
            line_means = np.asarray(
                [raw[channel][target == level].mean() for level in levels]
            )
        else:
            line_mean = raw[channel][target > 0].mean()
            line_means = np.full(3, line_mean, dtype=np.float64)
        plane_mean = float(np.mean(raw[channel]))
        background_mean = float(np.mean(background))
        responses = (line_means - background_mean) / (plane_mean + epsilon)
        ratios = responses / (responses[2] + epsilon)
        gaps = np.diff(responses)
        if all(np.any(target == level) for level in levels):
            line_variances = [
                float(np.var(raw[channel][target == level])) for level in levels
            ]
        else:
            line_variance = float(np.var(raw[channel][target > 0]))
            line_variances = [line_variance] * 3
        rows.append(
            [
                channel + 1,
                PAIR_MAT[channel, 0],
                PAIR_MAT[channel, 1],
                plane_mean,
                background_mean,
                float(np.var(background)),
                float(np.var(background_row_means)),
                float(np.percentile(background, 95)),
                *responses,
                *ratios,
                int(np.all(gaps > 0)),
                float(np.min(gaps)),
                *line_variances,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


EVALUATION_CHANNEL_HEADERS = (
    "channel",
    "structure_cosine",
    "foreground_coverage_above_background_p95",
    "grayscale_monotonic",
    "grayscale_ratio_rmse",
    "normalized_min_level_gap",
    "S_1_3",
    "S_2_3",
    "S_1",
    "background_cv",
    "background_p95_ratio",
    "background_row_cv",
)

EVALUATION_SUMMARY_HEADERS = (
    "structure_cosine_mean",
    "structure_cosine_min",
    "foreground_coverage_mean",
    "foreground_coverage_min",
    "grayscale_monotonic_channels",
    "grayscale_ratio_rmse_mean",
    "grayscale_ratio_rmse_max",
    "S_1_3_mean",
    "S_1_3_min",
    "S_1_3_max",
    "S_1_3_cv",
    "S_2_3_mean",
    "S_2_3_min",
    "S_2_3_max",
    "S_2_3_cv",
    "S_1_mean",
    "S_1_min",
    "S_1_max",
    "S_1_cv",
    "background_cv_mean",
    "background_cv_max",
    "background_p95_ratio_mean",
    "background_p95_ratio_max",
    "background_row_cv_mean",
    "background_row_cv_max",
)


def evaluation_metrics(raw, targets):
    rows = []
    epsilon = 1e-12
    levels = np.asarray([1 / 3, 2 / 3, 1.0], dtype=np.float64)

    for channel in range(targets.shape[0]):
        target = targets[channel]
        foreground_mask = target > 0
        background_mask = target == 0
        image = raw[channel].astype(np.float64, copy=False)
        background = image[background_mask]
        background_mean = float(np.mean(background))
        background_std = float(np.std(background))
        background_p95 = float(np.percentile(background, 95))
        plane_mean = float(np.mean(image))

        positive_contrast = np.maximum(image - background_mean, 0.0)
        target_shape = foreground_mask.astype(np.float64)
        structure_contrast = positive_contrast.copy()
        structure_contrast[foreground_mask] /= target[foreground_mask]
        structure_cosine = float(
            np.sum(structure_contrast * target_shape)
            / (
                np.linalg.norm(structure_contrast.ravel())
                * np.linalg.norm(target_shape.ravel())
                + epsilon
            )
        )
        foreground_coverage = float(
            np.mean(image[foreground_mask] > background_p95)
        )

        has_three_levels = all(
            np.any(np.isclose(target, level, rtol=0, atol=1e-6))
            for level in levels
        )
        if has_three_levels:
            level_means = np.asarray(
                [
                    np.mean(
                        image[np.isclose(target, level, rtol=0, atol=1e-6)]
                    )
                    for level in levels
                ]
            )
            responses = (level_means - background_mean) / (plane_mean + epsilon)
            ratios = responses / (responses[2] + epsilon)
            gaps = np.diff(responses)
            grayscale_monotonic = int(np.all(gaps > 0))
            grayscale_ratio_rmse = float(np.sqrt(np.mean((ratios - levels) ** 2)))
            normalized_min_gap = float(np.min(gaps) / (abs(responses[2]) + epsilon))
        else:
            response = (
                float(np.mean(image[foreground_mask])) - background_mean
            ) / (plane_mean + epsilon)
            responses = np.full(3, response, dtype=np.float64)
            grayscale_monotonic = 1
            grayscale_ratio_rmse = 0.0
            normalized_min_gap = 0.0

        background_weights = background_mask.astype(np.float64)
        row_counts = np.sum(background_weights, axis=1)
        valid_rows = row_counts > 0
        background_row_means = np.sum(
            image * background_weights, axis=1
        )[valid_rows] / row_counts[valid_rows]

        rows.append(
            [
                channel + 1,
                structure_cosine,
                foreground_coverage,
                grayscale_monotonic,
                grayscale_ratio_rmse,
                normalized_min_gap,
                *responses,
                background_std / (background_mean + epsilon),
                background_p95 / (background_mean + epsilon),
                float(np.std(background_row_means))
                / (float(np.mean(background_row_means)) + epsilon),
            ]
        )

    channel_rows = np.asarray(rows, dtype=np.float64)
    summary = {
        "structure_cosine_mean": float(np.mean(channel_rows[:, 1])),
        "structure_cosine_min": float(np.min(channel_rows[:, 1])),
        "foreground_coverage_mean": float(np.mean(channel_rows[:, 2])),
        "foreground_coverage_min": float(np.min(channel_rows[:, 2])),
        "grayscale_monotonic_channels": int(np.sum(channel_rows[:, 3])),
        "grayscale_ratio_rmse_mean": float(np.mean(channel_rows[:, 4])),
        "grayscale_ratio_rmse_max": float(np.max(channel_rows[:, 4])),
    }
    for column, name in zip((6, 7, 8), ("S_1_3", "S_2_3", "S_1")):
        values = channel_rows[:, column]
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_min"] = float(np.min(values))
        summary[f"{name}_max"] = float(np.max(values))
        summary[f"{name}_cv"] = float(
            np.std(values) / (abs(np.mean(values)) + epsilon)
        )
    for column, name in (
        (9, "background_cv"),
        (10, "background_p95_ratio"),
        (11, "background_row_cv"),
    ):
        values = channel_rows[:, column]
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_max"] = float(np.max(values))
    return channel_rows, summary


def print_evaluation_summary(summary, channel_count):
    print("四项结果评价:")
    print(
        "  结构完整: "
        f"余弦相似度 mean/min={summary['structure_cosine_mean']:.4f}/"
        f"{summary['structure_cosine_min']:.4f}, "
        f"前景覆盖率 mean/min={summary['foreground_coverage_mean']:.4f}/"
        f"{summary['foreground_coverage_min']:.4f}"
    )
    print(
        "  灰度分级: "
        f"单调通道={summary['grayscale_monotonic_channels']}/{channel_count}, "
        f"比例RMSE mean/max={summary['grayscale_ratio_rmse_mean']:.4f}/"
        f"{summary['grayscale_ratio_rmse_max']:.4f}"
    )
    print(
        "  同灰度跨通道CV: "
        f"1/3={summary['S_1_3_cv']:.4f}, "
        f"2/3={summary['S_2_3_cv']:.4f}, 1={summary['S_1_cv']:.4f}"
    )
    print(
        "  背景均匀: "
        f"像素CV mean/max={summary['background_cv_mean']:.4f}/"
        f"{summary['background_cv_max']:.4f}, "
        f"P95/均值 mean/max={summary['background_p95_ratio_mean']:.4f}/"
        f"{summary['background_p95_ratio_max']:.4f}, "
        f"逐行CV mean/max={summary['background_row_cv_mean']:.4f}/"
        f"{summary['background_row_cv_max']:.4f}"
    )


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
    if not 1 <= args.channel_count <= len(PAIR_MAT):
        raise ValueError(f"channel-count 必须在 1 到 {len(PAIR_MAT)} 之间。")
    if args.log_interval < 1:
        raise ValueError("log-interval 必须大于 0。")
    if args.brightness_consistency_weight < 0:
        raise ValueError("brightness-consistency-weight 不能小于 0。")
    if args.worst_channel_weight < 0:
        raise ValueError("worst-channel-weight 不能小于 0。")
    for option_name in (
        "level_weight",
        "cross_level_weight",
        "gap_weight",
        "line_uniformity_weight",
        "visibility_weight",
        "worst_level_weight",
        "background_uniformity_weight",
        "reference_q",
        "background_row_uniformity_weight",
        "background_band_weight",
        "background_band_lower",
        "background_band_upper",
        "background_cluster_weight",
        "background_cluster_kernel",
        "background_cluster_upper",
        "foreground_efficiency_weight",
        "structure_completeness_weight",
        "gray_ratio_weight",
    ):
        if getattr(args, option_name) < 0:
            raise ValueError(f"{option_name} 不能小于 0。")
    if args.background_band_upper > 0 and args.background_band_lower > args.background_band_upper:
        raise ValueError("background-band-lower 不能大于 background-band-upper。")
    if args.background_band_weight > 0 and args.background_band_upper <= 0:
        raise ValueError("启用 background-band-weight 时必须提供正的 background-band-upper。")
    if args.background_cluster_weight > 0 and args.background_cluster_upper <= 0:
        raise ValueError("启用 background-cluster-weight 时必须提供正的 background-cluster-upper。")
    if args.background_cluster_kernel < 3 or args.background_cluster_kernel % 2 == 0:
        raise ValueError("background-cluster-kernel 必须为大于等于 3 的奇数。")
    if args.image_loss_mode == "balanced":
        if args.foreground_loss_weight <= 0 or args.background_loss_weight <= 0:
            raise ValueError("balanced 图案损失的前景和背景权重必须为正数。")

    device = select_device(args.device)
    targets_np = load_targets(args.mat_file)
    if args.channel_count > targets_np.shape[0]:
        raise ValueError(
            f"channel-count={args.channel_count} 超过输入目标通道数{targets_np.shape[0]}。"
        )
    targets_np = targets_np[: args.channel_count]
    targets = torch.tensor(targets_np, device=device)
    pair_mat_np = PAIR_MAT[: args.channel_count]
    pair_mat = torch.tensor(pair_mat_np, dtype=torch.float32, device=device)
    weights = torch.tensor(
        CUSTOM_WEIGHTS[: args.channel_count], dtype=torch.float32, device=device
    )

    print(f"Device: {device}")
    print(f"Targets: {targets_np.shape}, values: 0, 1/3, 2/3, 1")

    size = targets_np.shape[-1]
    if args.initial_results is None:
        torch.manual_seed(args.seed)
        phdx = (
            torch.rand((size, size), device=device) * 2 * np.pi
        ).requires_grad_()
        phdy = (
            torch.rand((size, size), device=device) * 2 * np.pi
        ).requires_grad_()
    else:
        initial = np.load(args.initial_results)
        if "phdx" not in initial or "phdy" not in initial:
            raise KeyError("initial-results 必须包含 phdx 和 phdy。")
        if initial["phdx"].shape != (size, size) or initial["phdy"].shape != (
            size,
            size,
        ):
            raise ValueError("initial-results 中 phdx/phdy 尺寸必须与目标图一致。")
        phdx = torch.tensor(initial["phdx"], device=device).requires_grad_()
        phdy = torch.tensor(initial["phdy"], device=device).requires_grad_()
    optimizer = optim.Adam([phdx, phdy], lr=args.lr)

    costs = []
    efficiency_history = []
    foreground_level_history = []
    brightness_cv_history = []
    worst_channel_ratio_history = []
    level_response_history = []
    q_history = []
    level_loss_history = []
    cross_level_loss_history = []
    gap_loss_history = []
    line_uniformity_loss_history = []
    visibility_loss_history = []
    worst_level_loss_history = []
    background_uniformity_loss_history = []
    background_row_uniformity_loss_history = []
    background_band_loss_history = []
    background_cluster_loss_history = []
    structure_completeness_loss_history = []
    gray_ratio_loss_history = []
    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        (
            loss,
            efficiencies,
            foreground_levels,
            brightness_cv,
            worst_channel_ratio,
            level_responses,
            q,
            level_loss,
            cross_level_loss,
            gap_loss,
            line_uniformity_loss,
            visibility_loss,
            worst_level_loss,
            background_uniformity_loss,
            background_row_uniformity_loss,
            background_band_loss,
            background_cluster_loss,
            structure_completeness_loss,
            gray_ratio_loss,
        ) = total_cost(
            phdx,
            phdy,
            targets,
            pair_mat,
            weights,
            brightness_consistency_weight=args.brightness_consistency_weight,
            worst_channel_weight=args.worst_channel_weight,
            level_weight=args.level_weight,
            cross_level_weight=args.cross_level_weight,
            gap_weight=args.gap_weight,
            line_uniformity_weight=args.line_uniformity_weight,
            visibility_weight=args.visibility_weight,
            worst_level_weight=args.worst_level_weight,
            background_uniformity_weight=args.background_uniformity_weight,
            reference_q=args.reference_q,
            background_row_uniformity_weight=args.background_row_uniformity_weight,
            background_band_weight=args.background_band_weight,
            background_band_lower=args.background_band_lower,
            background_band_upper=args.background_band_upper,
            background_cluster_weight=args.background_cluster_weight,
            background_cluster_kernel=args.background_cluster_kernel,
            background_cluster_upper=args.background_cluster_upper,
            image_loss_mode=args.image_loss_mode,
            foreground_loss_weight=args.foreground_loss_weight,
            background_loss_weight=args.background_loss_weight,
            foreground_efficiency_weight=args.foreground_efficiency_weight,
            structure_completeness_weight=args.structure_completeness_weight,
            gray_ratio_weight=args.gray_ratio_weight,
        )
        loss.backward()
        optimizer.step()

        costs.append(loss.item())
        efficiency_history.append(efficiencies.cpu().numpy())
        foreground_level_history.append(foreground_levels.cpu().numpy())
        brightness_cv_history.append(brightness_cv.item())
        worst_channel_ratio_history.append(worst_channel_ratio.item())
        level_response_history.append(level_responses.cpu().numpy())
        q_history.append(q.item())
        level_loss_history.append(level_loss.item())
        cross_level_loss_history.append(cross_level_loss.item())
        gap_loss_history.append(gap_loss.item())
        line_uniformity_loss_history.append(line_uniformity_loss.item())
        visibility_loss_history.append(visibility_loss.item())
        worst_level_loss_history.append(worst_level_loss.item())
        background_uniformity_loss_history.append(background_uniformity_loss.item())
        background_row_uniformity_loss_history.append(
            background_row_uniformity_loss.item()
        )
        background_band_loss_history.append(background_band_loss.item())
        background_cluster_loss_history.append(background_cluster_loss.item())
        structure_completeness_loss_history.append(
            structure_completeness_loss.item()
        )
        gray_ratio_loss_history.append(gray_ratio_loss.item())
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"[Epoch {epoch}/{args.epochs}] Loss={loss.item():.6e} "
                f"BrightCV={brightness_cv.item():.4f} "
                f"Worst/Mean={worst_channel_ratio.item():.4f} "
                f"BgVar={background_uniformity_loss.item():.4f} "
                f"BgRowVar={background_row_uniformity_loss.item():.4f} "
                f"BgBand={background_band_loss.item():.4f}"
                f" BgCluster={background_cluster_loss.item():.4f}"
            )

    print(f"优化完成，用时 {time.time() - started_at:.1f} 秒")

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent / f"results_36channels_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    optimized_raw, optimized_01, optimized_global_01 = reconstruct(
        phdx, phdy, pair_mat
    )
    metrics = channel_metrics(optimized_raw, targets_np)
    grayscale_level_metrics = grayscale_metrics(optimized_raw, targets_np)
    evaluation_channel_metrics, evaluation_summary = evaluation_metrics(
        optimized_raw, targets_np
    )
    np.savez_compressed(
        output_dir / "optimized_results.npz",
        phdx=phdx.detach().cpu().numpy(),
        phdy=phdy.detach().cpu().numpy(),
        costs=np.asarray(costs),
        pairMat=pair_mat_np,
        weights=weights.cpu().numpy(),
        targets=targets_np,
        optimized_raw=optimized_raw,
        optimized_01=optimized_01,
        optimized_global_01=optimized_global_01,
        eta_history=np.asarray(efficiency_history),
        foreground_level_history=np.asarray(foreground_level_history),
        brightness_cv_history=np.asarray(brightness_cv_history),
        worst_channel_ratio_history=np.asarray(worst_channel_ratio_history),
        level_response_history=np.asarray(level_response_history),
        q_history=np.asarray(q_history),
        level_loss_history=np.asarray(level_loss_history),
        cross_level_loss_history=np.asarray(cross_level_loss_history),
        gap_loss_history=np.asarray(gap_loss_history),
        line_uniformity_loss_history=np.asarray(line_uniformity_loss_history),
        visibility_loss_history=np.asarray(visibility_loss_history),
        worst_level_loss_history=np.asarray(worst_level_loss_history),
        background_uniformity_loss_history=np.asarray(
            background_uniformity_loss_history
        ),
        background_row_uniformity_loss_history=np.asarray(
            background_row_uniformity_loss_history
        ),
        background_band_loss_history=np.asarray(background_band_loss_history),
        background_cluster_loss_history=np.asarray(background_cluster_loss_history),
        structure_completeness_loss_history=np.asarray(
            structure_completeness_loss_history
        ),
        gray_ratio_loss_history=np.asarray(gray_ratio_loss_history),
        brightness_consistency_weight=args.brightness_consistency_weight,
        worst_channel_weight=args.worst_channel_weight,
        level_weight=args.level_weight,
        cross_level_weight=args.cross_level_weight,
        gap_weight=args.gap_weight,
        line_uniformity_weight=args.line_uniformity_weight,
        visibility_weight=args.visibility_weight,
        worst_level_weight=args.worst_level_weight,
        background_uniformity_weight=args.background_uniformity_weight,
        reference_q=args.reference_q,
        background_row_uniformity_weight=args.background_row_uniformity_weight,
        background_band_weight=args.background_band_weight,
        background_band_lower=args.background_band_lower,
        background_band_upper=args.background_band_upper,
        background_cluster_weight=args.background_cluster_weight,
        background_cluster_kernel=args.background_cluster_kernel,
        background_cluster_upper=args.background_cluster_upper,
        image_loss_mode=args.image_loss_mode,
        foreground_loss_weight=args.foreground_loss_weight,
        background_loss_weight=args.background_loss_weight,
        foreground_efficiency_weight=args.foreground_efficiency_weight,
        structure_completeness_weight=args.structure_completeness_weight,
        gray_ratio_weight=args.gray_ratio_weight,
        channel_count=args.channel_count,
        seed=args.seed,
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
    np.savetxt(
        output_dir / "grayscale_level_metrics.csv",
        grayscale_level_metrics,
        delimiter=",",
        header=(
            "channel,m,n,plane_mean,background_mean,background_variance,"
            "background_row_variance,background_p95,S_1_3,S_2_3,S_1,"
            "ratio_1_3,ratio_2_3,ratio_1,"
            "monotonic,min_gap,line_variance_1_3,line_variance_2_3,"
            "line_variance_1"
        ),
        comments="",
    )
    np.savetxt(
        output_dir / "evaluation_channel_metrics.csv",
        evaluation_channel_metrics,
        delimiter=",",
        header=",".join(EVALUATION_CHANNEL_HEADERS),
        comments="",
    )
    np.savetxt(
        output_dir / "evaluation_summary.csv",
        np.asarray([[evaluation_summary[name] for name in EVALUATION_SUMMARY_HEADERS]]),
        delimiter=",",
        header=",".join(EVALUATION_SUMMARY_HEADERS),
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
    print_evaluation_summary(evaluation_summary, targets_np.shape[0])
    print(f"四项评价汇总: {output_dir / 'evaluation_summary.csv'}")
    print("流程已结束：未生成相位 CSV、CIF 或后续仿真内容。")


if __name__ == "__main__":
    main()
