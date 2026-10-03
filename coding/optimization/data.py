"""Read immutable target images and their explicit diffraction-order sequence."""
from pathlib import Path

import numpy as np
import scipy.io as sio


def load_target(mat_file, gray_level_count):
    data = sio.loadmat(Path(mat_file))
    targets = np.asarray(data["bw_all"])
    orders = np.asarray(data["order_pairs"])
    positions = np.asarray(data["grid_positions"])
    if targets.ndim != 3 or targets.shape[1] != targets.shape[2]:
        raise ValueError("bw_all must have shape channels × size × size")
    if orders.shape != (len(targets), 2) or positions.shape != orders.shape:
        raise ValueError("order_pairs and grid_positions must match the target channel sequence")
    for name, array in (("order_pairs", orders), ("grid_positions", positions)):
        if not np.isfinite(array).all() or not np.equal(array, np.rint(array)).all():
            raise ValueError(f"{name} must contain integer coordinates")
        if len(np.unique(array, axis=0)) != len(targets):
            raise ValueError(f"{name} contains duplicate coordinates")
    if int(np.asarray(data["gray_level_count"]).item()) != gray_level_count:
        raise ValueError("MAT gray_level_count disagrees with the experiment configuration")
    codes = np.rint(targets * (gray_level_count - 1))
    if not (np.isfinite(targets).all() and np.all((codes >= 0) & (codes < gray_level_count))
            and np.allclose(targets, codes / (gray_level_count-1), rtol=0, atol=1e-6)):
        raise ValueError(f"Targets must lie on the {gray_level_count}-level grayscale lattice")
    if not np.any(targets > 0) or np.any(np.sum(targets == 0, axis=(1, 2)) == 0):
        raise ValueError("The scene needs nonzero target pixels and every channel needs black background")
    return targets.astype(np.float32, copy=False), orders.astype(np.int16), positions.astype(np.int16)


def target_labels(targets, gray_level_count):
    labels = np.rint(targets * (gray_level_count-1)).astype(np.uint8)
    if not np.allclose(labels.astype(np.float32)/(gray_level_count-1), targets, rtol=0, atol=1e-6):
        raise ValueError("Target labels cannot be stored losslessly")
    return labels
