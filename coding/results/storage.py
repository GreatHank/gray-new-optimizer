"""Versioned optimizer result I/O without duplicating derived image arrays."""

from pathlib import Path

import numpy as np


FORMAT_VERSION = 4


def json_value(value):
    """Represent undefined numeric metrics explicitly as JSON null."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def result_file(path):
    path = Path(path)
    return path / "optimized_results.npz" if path.is_dir() else path


def load_result(path):
    """Read PP results and reconstruct targets from lossless labels."""
    path = result_file(path)
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    missing = {"optimized_raw", "target_labels", "gray_level_count", "pp"} - result.keys()
    if missing:
        raise KeyError(f"结果缺少 {sorted(missing)}：{path}")
    level_count = int(np.asarray(result["gray_level_count"]).item())
    result["targets"] = result["target_labels"].astype(np.float32) / (level_count - 1)
    return result


def load_warm_start(path):
    """Load the three shared phase maps for phase-only warm starts."""
    path = result_file(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = {"phdx", "phdy", "pp"} - set(archive.files)
        if missing:
            raise KeyError(f"热启动结果缺少 {sorted(missing)}：{path}")
        return archive["phdx"], archive["phdy"], archive["pp"]


def save_result(path, arrays):
    """Write raw evidence without duplicating derived display arrays."""
    payload = dict(arrays)
    if "pp" not in payload:
        raise KeyError("新结果必须包含 pp 相位图。")
    payload["format_version"] = np.asarray(FORMAT_VERSION, dtype=np.int16)
    np.savez_compressed(Path(path), **payload)
