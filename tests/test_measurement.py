import numpy as np
import open3d as o3d
import pytest
from measurement.engine import measure_asset

def create_box_pcd(L, W, H, num_points=1000):
    """Creates a synthetic box point cloud."""
    pts = np.random.rand(num_points, 3)
    pts[:, 0] *= L
    pts[:, 1] *= W
    pts[:, 2] *= H
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def create_cylinder_pcd(radius, length, num_points=1000):
    """Creates a synthetic cylinder point cloud along Z axis."""
    theta = np.random.rand(num_points) * 2 * np.pi
    z = np.random.rand(num_points) * length
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    pts = np.stack([x, y, z], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

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
