"""Check image preservation and order mapping for a 500 x 500 shared array."""
import json
import hashlib
import shutil

import numpy as np
import pytest
from PIL import Image

from coding import ROOT
from coding.optimization.data import load_target
from coding.targets.prepare import main, circle_rectangle_area
from coding.results.report import assemble


def test_500_array_preserves_image_and_maps_zero_order_to_top_right(workspace_tmp):
    source = np.zeros((1500, 1500), dtype=np.uint8)
    for index, (row, column) in enumerate(
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]):
        source[row*500+100:row*500+400, column*500+100:column*500+400] = (index+1)*25
    image_path = workspace_tmp/"source.png"
    Image.fromarray(source).save(image_path)
    config = json.loads((ROOT/"configs/tiger-3x3-p800-500.json").read_text(encoding="utf-8"))
    config["target"]["source"] = str(image_path)
    # Isolate exact tile mapping from the source image's placement and support checks.
    config["target"].pop("resized_size")
    config["target"].pop("canvas_offset")
    config["physics"]["strict_require"] = ["order_count", "all_centers_propagating"]
    config_path = workspace_tmp/"config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    target_parent = (ROOT/"output/targets").resolve()
    output = target_parent/workspace_tmp.name
    try:
        main(["--config", str(config_path), "--output-dir", str(output)])
        targets, orders, grid_positions = load_target(output/"target.mat", 256)
        assert targets.shape == (9, 500, 500)
        np.testing.assert_array_equal(orders, [
            [-2, 0], [-1, 0], [0, 0], [-2, -1], [-1, -1], [0, -1],
            [-2, -2], [-1, -2], [0, -2]])
        np.testing.assert_array_equal(grid_positions, [
            [0, 2], [1, 2], [2, 2], [0, 1], [1, 1], [2, 1], [0, 0], [1, 0], [2, 0]])
        np.testing.assert_array_equal(np.asarray(Image.open(output/"target.png")), source)
        for index, target in enumerate(targets):
            assert target[250, 250] == np.float32((index+1)*25/255)
            assert target[0, 0] == 0
        manifest = json.loads((output/"summary.json").read_text(encoding="utf-8"))
        assert manifest["zero_order_channel_1based"] == 3
        assert manifest["phase_array_shape"] == [500, 500]
        assert manifest["shared_phase_parameter_count"] == 750000
        assert manifest["physics_check"]["center_propagating"] == 9
    finally:
        if output.exists():
            resolved = output.resolve()
            assert target_parent in resolved.parents
            shutil.rmtree(resolved)


@pytest.mark.parametrize("layout,period,channels", [("4x4",1050,16)])
def test_circular_art_covers_full_field_without_changing_interior_brightness(
        workspace_tmp, layout, period, channels):
    config_path = ROOT/f"configs/circle-cosmic-{layout}-p{period}-500-fullcircle.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path = ROOT/config["target"]["source"]
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = np.asarray(Image.open(source_path).convert("L"))
    parent = (ROOT/"output/targets").resolve()
    output = parent/workspace_tmp.name
    try:
        main(["--config",str(config_path),"--output-dir",str(output)])
        targets, orders, _ = load_target(output/"target.mat",256)
        assert targets.shape == (channels,500,500)
        settings = config["target"]
        positions = np.array([(settings["order_n_start"]+settings["grid_size"]-1-n,
                               m-settings["order_m_start"]) for m,n in orders])
        labels = assemble(np.rint(targets*255).astype(np.uint8),positions)
        np.testing.assert_array_equal(labels,np.asarray(Image.open(output/"target.png")))
        yy,xx = np.indices(labels.shape)
        theta = np.deg2rad(config["physics"]["incident_theta_deg"])
        incident = np.sin(theta)/np.sqrt(2)
        pitch = 532/period
        u = incident+(settings["order_m_start"]-.5+xx/500)*pitch
        v = incident+(settings["order_n_start"]+settings["grid_size"]-.5-yy/500)*pitch
        disk = u*u+v*v <= 1
        covered = np.zeros(labels.shape,dtype=bool)
        for r,c in positions:
            covered[r*500:(r+1)*500,c*500:(c+1)*500] = True
        assert covered[disk].all()
        assert not np.any(labels[~disk])
        assert np.max(np.hypot(u,v)[labels>0]) > .9999
        cx,cy = settings["source_circle"]["center_px"]
        radius = settings["source_circle"]["radius_px"]
        # Independently interpolate asymmetric interior samples to check axes and gray values.
        for expected_u,expected_v in [(.25,.38),(-.45,.2),(.1,-.5),(-.33,-.12)]:
            r,c = np.unravel_index(np.argmin((u-expected_u)**2+(v-expected_v)**2),u.shape)
            sx,sy = cx+radius*u[r,c],cy-radius*v[r,c]
            x,y = int(np.floor(sx)),int(np.floor(sy))
            wx,wy = sx-x,sy-y
            expected = ((1-wy)*((1-wx)*source[y,x]+wx*source[y,x+1])
                        +wy*((1-wx)*source[y+1,x]+wx*source[y+1,x+1]))
            assert abs(int(labels[r,c])-expected) <= .501
        manifest = json.loads((output/"summary.json").read_text(encoding="utf-8"))
        mapping = manifest["circular_field_mapping"]
        assert mapping["field_circle_grid_coverage_fraction"] == 1
        assert mapping["outside_source_removed_max_gray"] == 2
        assert manifest["physics_check"]["strict_failures"] == []
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
    finally:
        if output.exists():
            assert parent in output.resolve().parents
            shutil.rmtree(output.resolve())


@pytest.mark.parametrize("failure",["missing_rim","excluded_rim","bright_exterior"])
def test_circle_mapping_rejects_incomplete_field_or_bright_source_removal(workspace_tmp,failure):
    config = json.loads((ROOT/"configs/circle-cosmic-4x4-p1050-500-fullcircle.json").read_text(encoding="utf-8"))
    if failure == "missing_rim":
        config["target"].update(grid_size=3,order_m_start=-2,order_n_start=-2,
                                 excluded_canvas_positions=[])
        message = "cannot contain the complete field circle"
    elif failure == "excluded_rim":
        config["target"]["excluded_canvas_positions"].append([2,0])
        message = "part of the field circle uncovered"
    else:
        config["target"]["source_circle"]["radius_px"] = 500
        message = "brighter than the declared outside background"
    path = workspace_tmp/"config.json"
    path.write_text(json.dumps(config),encoding="utf-8")
    with pytest.raises(ValueError,match=message):
        main(["--config",str(path),"--output-dir",str(ROOT/"output/targets"/workspace_tmp.name)])


def test_partial_circle_area_matches_four_analytic_caps():
    edge = .9975
    cap = np.arccos(edge)-edge*np.sqrt(1-edge**2)
    expected = np.pi-4*cap
    assert abs(circle_rectangle_area((-edge,edge,-edge,edge),1)-expected) < 1e-10


@pytest.mark.parametrize("subject",["ocean","forest"])
def test_strict_nine_orders_keep_saved_target_labels_exactly(workspace_tmp,subject):
    parent = (ROOT/"output/targets").resolve()
    output = parent/workspace_tmp.name
    config_path = ROOT/f"configs/circle-{subject}-3x3-p800-500-strict9.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = ROOT/config["target"]["source"]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        main(["--config",str(config_path),"--output-dir",str(output)])
        new,orders,_ = load_target(output/"target.mat",256)
        old,old_orders,_ = load_target(ROOT/f"output/targets/circle_{subject}_strict9_3x3_p800_gray256_500/target.mat",256)
        assert new.shape == (9,500,500)
        np.testing.assert_array_equal(orders,[[m,n] for n in [0,-1,-2] for m in [-2,-1,0]])
        indices = [np.flatnonzero(np.all(old_orders==order,axis=1)).item() for order in orders]
        np.testing.assert_array_equal(new,old[indices])
        manifest = json.loads((output/"summary.json").read_text(encoding="utf-8"))
        assert manifest["zero_order_channel_1based"] == 3
        assert not manifest["circular_field_mapping"]["require_full_field_coverage"]
        assert .99969 < manifest["circular_field_mapping"]["field_circle_grid_coverage_fraction"] < .99971
        assert manifest["physics_check"]["strict_failures"] == []
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    finally:
        if output.exists():
            assert parent in output.resolve().parents
            shutil.rmtree(output.resolve())


@pytest.mark.parametrize("grid,period,theta,size,offset,zero_channel", [
    (3,800,70.1276,1402,(48,85),3),
    (4,1050,20.9938,1840,(79,127),7),
])
def test_screenshot_grids_keep_complete_tiger_near_circle_boundary(
        workspace_tmp,grid,period,theta,size,offset,zero_channel):
    target_parent=(ROOT/"output/targets").resolve()
    output=target_parent/workspace_tmp.name
    try:
        main(["--config",str(ROOT/f"configs/tiger-{grid}x{grid}-p{period}-500.json"),
              "--output-dir",str(output)])
        targets,orders,_=load_target(output/"target.mat",256)
        positions=np.array([(r,c) for r in range(grid) for c in range(grid)])
        np.testing.assert_array_equal(orders,[[-2+c,grid-3-r] for r,c in positions])
        expected=Image.new("L",(grid*500,grid*500))
        source=Image.open(ROOT/"input/tiger_detailed.png").convert("L")
        expected.paste(source.resize((size,size),Image.Resampling.LANCZOS),offset)
        np.testing.assert_array_equal(assemble(np.rint(targets*255).astype(np.uint8),positions),np.array(expected))
        assert np.all(np.any(targets>0,axis=(1,2)))
        yy,xx=np.indices((500,500))
        pitch=532/period
        incident=np.sin(np.deg2rad(theta))/np.sqrt(2)
        radii=np.stack([np.hypot(incident+(m+(xx-250)/500)*pitch,
                                 incident+(n-(yy-250)/500)*pitch) for m,n in orders])
        assert np.max(radii[targets>0])<=1
        assert np.max(radii[targets>0])>.99
        manifest=json.loads((output/"summary.json").read_text(encoding="utf-8"))
        assert manifest["zero_order_channel_1based"]==zero_channel
        assert manifest["physics_check"]["center_propagating"]==(9 if grid==3 else 12)
        assert manifest["physics_check"]["strict_failures"]==[]
    finally:
        if output.exists():
            resolved=output.resolve()
            assert target_parent in resolved.parents
            shutil.rmtree(resolved)
