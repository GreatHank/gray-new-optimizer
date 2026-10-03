import numpy as np
import pytest
import scipy.io as sio

from coding.optimization.data import load_target
from coding.results.storage import load_result, load_warm_start, save_result


def test_loader_preserves_all_256_levels_and_exact_order_sequence(workspace_tmp):
    labels = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)
    target = labels.astype(np.float32)/255
    path = workspace_tmp/"target.mat"
    fields = {"bw_all": target, "gray_level_count": 256,
              "order_pairs": [[-2, 1]], "grid_positions": [[0, 3]]}
    sio.savemat(path, fields)
    loaded, orders, _ = load_target(path, 256)
    np.testing.assert_array_equal(loaded, target)
    np.testing.assert_array_equal(orders, [[-2, 1]])
    target[0, 2, 2] += 0.001
    sio.savemat(path, fields)
    with pytest.raises(ValueError, match="256"):
        load_target(path, 256)


def test_storage_preserves_raw_labels_and_three_warm_start_phases(workspace_tmp):
    raw = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    arrays = {"optimized_raw": raw, "target_labels": raw.astype(np.uint8), "gray_level_count": 256,
              "phdx": np.ones((4, 4), dtype=np.float32), "phdy": np.zeros((4, 4), dtype=np.float32),
              "pp": np.full((4, 4), 0.25, dtype=np.float32)}
    path = workspace_tmp/"optimized_results.npz"
    save_result(path, arrays)
    loaded = load_result(path)
    np.testing.assert_array_equal(loaded["optimized_raw"], raw)
    np.testing.assert_array_equal(loaded["targets"], raw/255)
    for phase, name in zip(load_warm_start(path), ("phdx", "phdy", "pp")):
        np.testing.assert_array_equal(phase, arrays[name])
    del arrays["pp"]
    with pytest.raises(KeyError, match="pp"):
        save_result(path, arrays)


def test_results_without_pp_are_rejected(workspace_tmp):
    phases = {"phdx": np.ones((4, 4), dtype=np.float32), "phdy": np.zeros((4, 4), dtype=np.float32)}
    path = workspace_tmp/"old.npz"
    np.savez_compressed(path, optimized_raw=np.ones((1, 4, 4)),
                        target_labels=np.zeros((1, 4, 4), dtype=np.uint8),
                        gray_level_count=256, **phases)
    with pytest.raises(KeyError, match="pp"):
        load_result(path)
    with pytest.raises(KeyError, match="pp"):
        load_warm_start(path)
