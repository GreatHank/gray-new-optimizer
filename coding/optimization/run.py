"""Optimize shared phase maps with image RMSE and optional gray consistency."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from coding import ROOT
from coding.geometry.preflight import validate_physics
from coding.geometry.observable import observable_mask
from coding.optimization.data import load_target, target_labels
from coding.optimization.diagnostics import measure_step
from coding.optimization.model import common_scale_rmse, gray_level_consistency_mse, shared_intensity
from coding.results.metrics import (channel_metrics, evaluation_metrics, grayscale_metrics,
                                   gray_level_cv_table, save_comparison, write_metrics_csv)
from coding.results.observable_metrics import evaluate as evaluate_observable, write_rows
from coding.results.storage import FORMAT_VERSION, json_value, load_warm_start, save_result


def parse_args(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=ROOT/"configs/circle-ocean-3x3-p800-500-strict9.json")
    config_args, _ = preliminary.parse_known_args(argv)
    configuration = json.loads(config_args.config.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--mat-file", type=Path)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--optimizer-name", choices=("adam", "lbfgs"), default="adam")
    parser.add_argument("--gray-consistency-weight", type=float, default=0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--gray-level-count", type=int, choices=(4, 16, 256), default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--strict-physics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spatial-constraint", choices=("full_plane", "observable"), default="full_plane")
    parser.add_argument("--initial-results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    defaults = configuration["optimizer"].copy()
    unknown = set(defaults) - {action.dest for action in parser._actions}
    if unknown:
        raise ValueError(f"Unknown optimizer settings: {sorted(unknown)}")
    for key in ("mat_file", "initial_results", "output_dir"):
        if defaults.get(key) is not None:
            defaults[key] = Path(defaults[key])
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.lr <= 0 or args.log_interval < 1:
        raise ValueError("epochs, lr and log_interval must be positive")
    if args.gray_consistency_weight < 0:
        raise ValueError("gray_consistency_weight must be nonnegative")
    if args.mat_file is None:
        raise ValueError("The target MAT file is required")
    args.mat_file = (ROOT / args.mat_file).resolve()
    if args.initial_results is not None:
        args.initial_results = (ROOT / args.initial_results).resolve()
    return args, configuration


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(args, shape):
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.config.stem}-{shape[-1]}x{shape[-2]}-{args.optimizer_name}-{args.epochs}steps-{timestamp}"
    directory = args.output_dir or ROOT/"output/data"/run_name
    directory = (ROOT/directory).resolve()
    allowed = (ROOT/"output/data").resolve()
    if directory == allowed or allowed not in directory.parents:
        raise ValueError("Optimization data must use a new directory under output/data")
    if directory.exists():
        raise FileExistsError(f"Output directory already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = int(np.prod(shape))*8 + shape[1]*shape[2]*12 + args.epochs*(shape[0]*2+5)*8
    free_bytes = shutil.disk_usage(directory.parent).free
    if free_bytes < estimated_bytes:
        raise OSError(f"Insufficient disk space: need approximately {estimated_bytes}, have {free_bytes}")
    directory.mkdir()
    (directory/".active").write_text(f"pid={os.getpid()}\n", encoding="utf-8")
    return directory


def initialize_phases(size, seed, initial_results, device):
    if initial_results is None:
        torch.manual_seed(seed)
        dx = torch.rand((size, size), device=device)*2*np.pi
        dy = torch.rand((size, size), device=device)*2*np.pi
        pp = torch.tensor(np.random.default_rng(seed).uniform(-np.pi, np.pi, (size, size))
                          .astype(np.float32), device=device)
    else:
        phases = load_warm_start(initial_results)
        if any(phase.shape != (size, size) or not np.isfinite(phase).all() for phase in phases):
            raise ValueError("Warm-start phase arrays must be finite and match the target tile size")
        if np.any(np.abs(phases[2]) > np.pi):
            raise ValueError("Saved PP must be in [-pi, pi]")
        dx, dy, pp = [torch.tensor(phase, dtype=torch.float32, device=device) for phase in phases]
    return [phase.requires_grad_() for phase in (dx, dy, pp)]


def main(argv=None):
    args, configuration = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot use CUDA")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    targets_np, orders_np, positions = load_target(args.mat_file, args.gray_level_count)
    physics = configuration.get("physics")
    physics_check = validate_physics(orders_np, physics, targets_np)
    if args.strict_physics and (physics_check["status"] in ("not_configured", "invalid")
                                or physics_check.get("strict_failures")):
        raise ValueError(f"Physical preflight failed: {physics_check}")
    targets = torch.tensor(targets_np, device=device)
    visible_np = (observable_mask(orders_np, targets_np.shape[-2:], physics)
                  if args.spatial_constraint == "observable" else None)
    visible = torch.tensor(visible_np, device=device) if visible_np is not None else None
    orders = torch.tensor(orders_np, dtype=torch.float32, device=device)
    weights = torch.ones(len(orders_np), dtype=torch.float32, device=device)
    if args.gray_consistency_weight:
        label_codes = target_labels(targets_np, args.gray_level_count)
        if visible_np is not None:
            label_codes = np.where(visible_np, label_codes, 0)
        label_codes = label_codes.reshape(len(orders_np), -1)
        gray_labels = torch.tensor(label_codes.astype(np.int64), device=device)
        gray_counts = torch.tensor(np.stack([np.bincount(row, minlength=args.gray_level_count)
                                            for row in label_codes]), dtype=torch.float32, device=device)

    def calculate_loss(intensity):
        image_rmse = common_scale_rmse(intensity, targets, weights, visible)
        if args.gray_consistency_weight:
            consistency = gray_level_consistency_mse(intensity, gray_labels, gray_counts)
            return torch.sqrt(image_rmse.square() + args.gray_consistency_weight * consistency)
        return image_rmse

    dx, dy, pp = initialize_phases(targets.shape[-1], args.seed, args.initial_results, device)
    parameters = [dx, dy, pp]
    if args.optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
    else:
        optimizer = torch.optim.LBFGS(parameters, lr=args.lr, max_iter=1, history_size=20,
                                     line_search_fn="strong_wolfe", tolerance_grad=0, tolerance_change=0)
    directory = prepare_output(args, targets_np.shape)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    effective_optimizer = {**vars(args), "image_loss_mode": "rmse_gray_consistency"
                           if args.gray_consistency_weight else "rmse", "output_level": "standard"}
    run_config = {
        "data_format_version": FORMAT_VERSION,
        "forward_version": "shared-dxdy-pp-fft2-fftshift-intensity-v1",
        "optimizer": json_value(effective_optimizer), "actual_device": str(device), "dtype": "float32",
        "input": {"path": str(args.mat_file), "sha256": sha256(args.mat_file),
                  "target_shape": list(targets_np.shape), "preprocessing": "MAT loaded without resizing or reordering"},
        "source": {"git_head": git.stdout.strip(), "python": sys.version.split()[0],
                   "source_sha256": {name: sha256(ROOT/name) for name in
                                     ("coding/optimization/run.py", "coding/optimization/model.py")}},
        "warm_start": {"kind": "phase_only_optimizer_reinitialized" if args.initial_results else None,
                       "source": str(args.initial_results) if args.initial_results else None},
        "physics": physics, "physics_check": physics_check,
    }
    (directory/"config.json").write_text(json.dumps(json_value(run_config), ensure_ascii=False,
                                                   indent=2, allow_nan=False), encoding="utf-8")
    costs, history = [], {}
    closure_evaluations = 0

    def closure():
        nonlocal closure_evaluations
        optimizer.zero_grad()
        value = calculate_loss(shared_intensity(dx, dy, pp, orders))
        value.backward()
        closure_evaluations += 1
        return value

    print(f"Device: {device}; targets: {tuple(targets.shape)}; shared phase variables: {3*targets.shape[-1]**2}")
    started = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        optimizer.zero_grad()
        intensity = shared_intensity(dx, dy, pp, orders)
        loss = calculate_loss(intensity)
        measurements = measure_step(intensity, targets, visible)
        if args.optimizer_name == "adam":
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pp.copy_(torch.remainder(pp+np.pi, 2*np.pi)-np.pi)
        else:
            optimizer.step(closure)
        costs.append(loss.item())
        for name, value in measurements.items():
            history.setdefault(name, []).append(value.cpu().numpy())
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(f"[Epoch {epoch}/{args.epochs}] Objective={costs[-1]:.8f}; "
                  f"SNR mean/min={measurements['snr_mean_db_history'].item():.2f}/"
                  f"{measurements['snr_min_db_history'].item():.2f} dB", flush=True)
    with torch.no_grad():
        pp.copy_(torch.remainder(pp+np.pi, 2*np.pi)-np.pi)
        final_intensity = shared_intensity(dx, dy, pp, orders)
        final_loss = common_scale_rmse(final_intensity, targets, weights, visible).item()
        final_objective = calculate_loss(final_intensity).item()
        final_consistency = (gray_level_consistency_mse(final_intensity, gray_labels, gray_counts).item()
                             if args.gray_consistency_weight else None)
        raw = final_intensity.cpu().numpy()
    if not np.isfinite(raw).all() or any(not torch.isfinite(phase).all().item() for phase in parameters):
        raise FloatingPointError("Optimization produced nonfinite phases or intensities")
    desired = np.linspace(0, 1, args.gray_level_count, dtype=np.float32)[1:]
    labels = target_labels(targets_np, args.gray_level_count)
    if visible_np is None:
        evaluation, evaluation_summary = evaluation_metrics(raw, targets_np, desired)
        brightness = channel_metrics(raw, targets_np, orders_np)
        grayscale = grayscale_metrics(raw, targets_np, orders_np, args.gray_level_count)
    else:
        channels, gray_rows, evaluation_summary = evaluate_observable(raw, targets_np, labels, orders_np, visible_np)
    arrays = {
        "phdx": dx.detach().cpu().numpy(), "phdy": dy.detach().cpu().numpy(), "pp": pp.detach().cpu().numpy(),
        "pairMat": orders_np, "grid_positions": positions, "weights": weights.cpu().numpy(),
        "target_labels": labels, "gray_level_count": np.asarray(args.gray_level_count, dtype=np.int16),
        "desired_gray_ratios": desired, "optimized_raw": raw, "costs": np.asarray(costs),
        **({"observable_mask": visible_np} if visible_np is not None else {}),
        **{name: np.asarray(values) for name, values in history.items()},
    }
    save_result(directory/"optimized_results.npz", arrays)
    if visible_np is None:
        write_metrics_csv(directory/"metrics.csv", brightness, grayscale, evaluation, args.gray_level_count)
        np.savetxt(directory/"gray_level_cv.csv", gray_level_cv_table(raw, targets_np, evaluation, args.gray_level_count),
                   delimiter=",", fmt=["%d"]*5+["%.8g"]*6,
                   header="level,gray_value,present_channels,total_target_pixels,min_pixels_per_present_channel,raw_mean,raw_std,raw_cv,response_mean,response_std,response_cv",
                   comments="")
        foreground_means = brightness[:, 6]
    else:
        write_rows(directory/"metrics.csv", channels)
        write_rows(directory/"gray_levels.csv", gray_rows)
        foreground_means = np.asarray([row["raw_foreground_mean"] for row in channels])
    common_gray = raw*float(targets_np.mean())/targets_np.shape[-1]**2
    save_comparison(targets_np, common_gray, directory/"overview.png", "Targets vs common-scale reconstruction (1x)")
    foreground_means = foreground_means[np.isfinite(foreground_means)]
    summary = {
        "run_id": directory.name, "status": "complete", "output_level": "standard", "channel_count": len(orders_np),
        "order_scheme": orders_np.tolist(), "parent_run": str(args.initial_results) if args.initial_results else None,
        "warm_start_semantics": "phase_only_optimizer_reinitialized" if args.initial_results else None,
        "metrics": evaluation_summary, "physical_validation": physics_check,
        "claim_scope": physics_check.get("scope", "software_forward_only"), "finite_numeric_result": True,
        "final_image_loss": final_loss, "final_objective": final_objective,
        "final_gray_consistency_mse": final_consistency, "runtime_seconds": time.perf_counter()-started,
        "optimizer_work": {"optimizer_name": args.optimizer_name, "iterations": args.epochs,
                           "closure_evaluations": closure_evaluations,
                           "loss_function_evaluations": args.epochs+closure_evaluations+1},
        "brightness_consistency": {"mode": "foreground_mean", "final_optimized_brightness_cv": float(np.std(foreground_means)/np.mean(foreground_means))},
        "history_semantics": "Each stored loss and diagnostic measures the state before that iteration's parameter update. final_image_loss and optimized_raw use the final updated phases.",
        "files": ["config.json", "optimized_results.npz", "metrics.csv",
                  "gray_levels.csv" if visible_np is not None else "gray_level_cv.csv", "summary.json", "overview.png"],
    }
    (directory/"summary.json").write_text(json.dumps(json_value(summary), ensure_ascii=False,
                                                    indent=2, allow_nan=False), encoding="utf-8")
    if "target" in configuration:
        from coding.results.publish import publish
        print(f"Target / reconstruction: {publish(directory)}")
    (directory/".active").unlink()
    print(f"Complete: {directory}\nFinal RMSE={final_loss:.8f}; mean SNR={evaluation_summary['foreground_background_snr_db_mean']:.2f} dB")


if __name__ == "__main__":
    main()
