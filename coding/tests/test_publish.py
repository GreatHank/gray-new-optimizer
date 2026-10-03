"""Check that a warm-start result becomes one clearly named image, without data edits."""

import json

import numpy as np
from PIL import Image
import pytest

from coding.results import publish as publisher


def test_publish_groups_one_image_and_labels_cumulative_steps(workspace_tmp, monkeypatch):
    monkeypatch.setattr(publisher, "ROOT", workspace_tmp)
    (workspace_tmp/"configs").mkdir()
    index_path = workspace_tmp/"configs/selected-results.json"
    index_path.write_text("{}", encoding="utf-8")
    parent = workspace_tmp/"output/data/parent"
    run = workspace_tmp/"output/data/continued"
    target_dir = workspace_tmp/"output/targets/test"
    for directory in (parent, run, target_dir):
        directory.mkdir(parents=True)
    parent_config = {"optimizer": {"epochs": 5000}, "warm_start": {"source": None}}
    (parent/"config.json").write_text(json.dumps(parent_config), encoding="utf-8")
    orders = np.array([(m, n) for n in (0, -1, -2) for m in (-2, -1, 0)])
    positions = [[row, col] for row in range(3) for col in range(3)]
    manifest = {"source": "input/circle_ocean_gray.png", "orders": orders.tolist(),
                "canvas_positions": positions, "circular_field_mapping": {"field_radius_u": 1}}
    (target_dir/"summary.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {"optimizer": {"epochs": 20000, "optimizer_name": "adam", "seed": 43},
              "warm_start": {"source": "output/data/parent"},
              "input": {"path": str(target_dir/"target.mat")},
              "physics": {"incident_theta_deg": 70.1276, "incident_azimuth_deg": 45,
                          "n_in": 1, "n_out": 1, "wavelength_nm": 532,
                          "period_x_nm": 800, "period_y_nm": 800, "na": 1}}
    (run/"config.json").write_text(json.dumps(config), encoding="utf-8")
    (run/"summary.json").write_text(json.dumps({"metrics": {
        "foreground_background_snr_db_mean": 13.43}}), encoding="utf-8")
    path = run/"optimized_results.npz"
    np.savez_compressed(path, target_labels=np.full((9, 4, 4), 128, dtype=np.uint8),
                        gray_level_count=256, pairMat=orders, pp=np.zeros((4, 4)),
                        optimized_raw=np.full((9, 4, 4), 32.))
    evidence = path.read_bytes()
    preview = publisher.publish(run)
    assert preview.parent == workspace_tmp/"output/runs/乌龟"
    assert "严格3×3（9级次）" in preview.name
    assert "累计25000步（续跑20000步）" in preview.name
    assert list(preview.parent.iterdir()) == [preview]
    with Image.open(preview) as image:
        assert image.format == "PNG" and image.width > image.height
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert next(iter(index.values()))["data_dir"] == "output/data/continued"
    assert path.read_bytes() == evidence
    with pytest.raises(FileExistsError):
        publisher.publish(run)
