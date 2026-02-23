import numpy as np
import open3d as o3d
import pytest
from measurement.engine import measure_asset

def create_box_pcd(L, W, H, num_points=1000, seed=7):
    """Creates a synthetic box point cloud."""
    rng = np.random.default_rng(seed)
    pts = rng.random((num_points, 3))
    pts[:, 0] *= L
    pts[:, 1] *= W
    pts[:, 2] *= H
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def create_cylinder_pcd(radius, length, num_points=1000, seed=11):
    """Creates a synthetic cylinder point cloud along Z axis."""
    rng = np.random.default_rng(seed)
    theta = rng.random(num_points) * 2 * np.pi
    z = rng.random(num_points) * length
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    pts = np.stack([x, y, z], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def translate_pcd(pcd, offset):
    shifted = o3d.geometry.PointCloud(pcd)
    shifted.translate(offset)
    return shifted


def merge_pcds(*pcds):
    all_pts = []
    for p in pcds:
        all_pts.append(np.asarray(p.points))
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    return merged


def test_trench_measurement():
    L, W, H = 2.0, 0.5, 1.2
    pcd = create_box_pcd(W, L, H) # Width along X, Length along Y
    res = measure_asset(pcd, "trench")

    assert res["category"] == "trench"
    assert pytest.approx(res["depth_m"], abs=0.05) == H
    assert pytest.approx(res["width_m"], abs=0.05) == W

def test_manhole_measurement():
    radius = 0.4
    depth = 1.5
    pcd = create_cylinder_pcd(radius, depth)
    res = measure_asset(pcd, "manhole")

    assert res["category"] == "manhole"
    assert pytest.approx(res["diameter_m"], abs=0.05) == 2 * radius
    assert pytest.approx(res["depth_m"], abs=0.05) == depth

def test_duct_measurement():
    radius = 0.15
    length = 3.0
    pcd = create_cylinder_pcd(radius, length)
    # Rotate cylinder to some arbitrary axis
    R = o3d.geometry.get_rotation_matrix_from_xyz([0.5, 0.5, 0.5])
    pcd.rotate(R, center=(0, 0, 0))

    res = measure_asset(pcd, "duct")

    assert res["category"] == "duct"
    assert pytest.approx(res["diameter_m"], abs=0.05) == 2 * radius
    assert pytest.approx(res["length_m"], abs=0.1) == length

def test_handhole_measurement():
    L, W, H = 1.0, 0.8, 0.6
    pcd = create_box_pcd(L, W, H)
    # Rotate box
    R = o3d.geometry.get_rotation_matrix_from_xyz([0.3, 0.4, 0.5])
    pcd.rotate(R, center=(0, 0, 0))

    res = measure_asset(pcd, "handhole")

    assert res["category"] == "handhole"
    # Extent is sorted L > W > H
    assert pytest.approx(res["L_m"], abs=0.05) == L
    assert pytest.approx(res["W_m"], abs=0.05) == W
    assert pytest.approx(res["H_m"], abs=0.05) == H


def test_trench_measurement_with_clutter():
    trench = create_box_pcd(0.55, 2.1, 1.25, num_points=1800)
    clutter = create_cylinder_pcd(0.35, 1.0, num_points=1600)
    clutter = translate_pcd(clutter, (3.0, 2.5, 0.2))
    scene = merge_pcds(trench, clutter)

    res = measure_asset(scene, "trench")
    assert pytest.approx(res["depth_m"], abs=0.10) == 1.25
    assert pytest.approx(res["width_m"], abs=0.10) == 0.55


def test_manhole_measurement_with_clutter():
    manhole = create_cylinder_pcd(0.42, 1.4, num_points=1800)
    trench_like = create_box_pcd(0.6, 2.5, 1.0, num_points=1700)
    trench_like = translate_pcd(trench_like, (3.2, 2.4, 0.0))
    scene = merge_pcds(manhole, trench_like)

    res = measure_asset(scene, "manhole")
    assert pytest.approx(res["diameter_m"], abs=0.10) == 0.84
    assert pytest.approx(res["depth_m"], abs=0.12) == 1.4


def test_duct_measurement_with_clutter():
    duct = create_cylinder_pcd(0.15, 3.2, num_points=2000)
    R = o3d.geometry.get_rotation_matrix_from_xyz([0.4, 0.2, 0.3])
    duct.rotate(R, center=(0, 0, 0))
    handhole_like = create_box_pcd(1.0, 0.8, 0.6, num_points=1600)
    handhole_like = translate_pcd(handhole_like, (3.1, 2.8, 0.1))
    scene = merge_pcds(duct, handhole_like)

    res = measure_asset(scene, "duct")
    assert pytest.approx(res["diameter_m"], abs=0.10) == 0.30
    assert pytest.approx(res["length_m"], abs=0.20) == 3.2


def test_handhole_measurement_with_clutter():
    handhole = create_box_pcd(1.1, 0.75, 0.5, num_points=1800)
    R = o3d.geometry.get_rotation_matrix_from_xyz([0.2, 0.3, 0.4])
    handhole.rotate(R, center=(0, 0, 0))
    duct_like = create_cylinder_pcd(0.18, 3.0, num_points=1800)
    duct_like = translate_pcd(duct_like, (3.2, 2.2, 0.3))
    scene = merge_pcds(handhole, duct_like)

    res = measure_asset(scene, "handhole")
    assert pytest.approx(res["L_m"], abs=0.12) == 1.1
    assert pytest.approx(res["W_m"], abs=0.12) == 0.75
    assert pytest.approx(res["H_m"], abs=0.12) == 0.5


def test_returns_not_detected_when_category_mismatch():
    # Pure cylinder should not pass trench ROI validation.
    pcd = create_cylinder_pcd(radius=0.35, length=1.0, num_points=1800)
    res = measure_asset(pcd, "trench")

    assert res["detected"] is False
    assert res["message"] == "trench not detected"
    assert res["depth_m"] is None
    assert res["width_m"] is None
