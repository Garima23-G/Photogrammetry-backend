import numpy as np
import open3d as o3d
import pytest
from measurement.engine import measure_asset

def create_box_pcd(L, W, H, num_points=10000, noise=0.001):
    """Creates a synthetic box point cloud with noise."""
    pts = np.random.rand(num_points, 3)
    pts[:, 0] *= W
    pts[:, 1] *= L
    pts[:, 2] *= H
    # Add noise
    pts += np.random.normal(0, noise, pts.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def create_cylinder_pcd(radius, length, num_points=10000, noise=0.001, partial=1.0):
    """Creates a synthetic cylinder point cloud with noise and partial visibility."""
    theta = np.random.rand(num_points) * 2 * np.pi * partial
    z = np.random.rand(num_points) * length
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    pts = np.stack([x, y, z], axis=1)
    # Add noise
    pts += np.random.normal(0, noise, pts.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def check_accuracy(category, measured, ground_truth, target_pct):
    error_pct = abs(measured - ground_truth) / ground_truth * 100
    status = "PASS" if error_pct <= target_pct else "FAIL"
    return measured, ground_truth, error_pct, status

def test_accuracy_report():
    report = []

    # 1. Trench Width Accuracy (Target: 1-3%)
    W_gt = 0.6
    pcd = create_box_pcd(L=2.0, W=W_gt, H=1.0, num_points=20000, noise=0.005)
    res = measure_asset(pcd, "trench")
    m, gt, err, status = check_accuracy("Trench Width", res["width_m"], W_gt, 3.0)
    report.append(["Trench Width", gt, m, err, status])

    # 2. Manhole Diameter Accuracy (Target: < 1%)
    D_gt = 0.8
    pcd = create_cylinder_pcd(radius=D_gt/2, length=1.5, num_points=20000, noise=0.001)
    res = measure_asset(pcd, "manhole")
    m, gt, err, status = check_accuracy("Manhole Diameter", res["diameter_m"], D_gt, 1.0)
    report.append(["Manhole Diameter", gt, m, err, status])

    # 3. Duct Diameter Accuracy (Target: 2-4%, Partial Arc)
    D_gt_duct = 0.3
    pcd = create_cylinder_pcd(radius=D_gt_duct/2, length=3.0, num_points=10000, noise=0.002, partial=0.6)
    res = measure_asset(pcd, "duct")
    m, gt, err, status = check_accuracy("Duct Diameter", res["diameter_m"], D_gt_duct, 4.0)
    report.append(["Duct Diameter", gt, m, err, status])

    # 4. Handhole Dimensions Accuracy (Target: 1-2%)
    L, W, H = 1.2, 0.9, 0.7
    pcd = create_box_pcd(L, W, H, num_points=15000, noise=0.002)
    # Rotate
    R = o3d.geometry.get_rotation_matrix_from_xyz([0.1, 0.2, 0.3])
    pcd.rotate(R, center=(0, 0, 0))
    res = measure_asset(pcd, "handhole")

    # Check L
    m, gt, err, status = check_accuracy("Handhole L", res["L_m"], L, 2.0)
    report.append(["Handhole L", gt, m, err, status])
    # Check W
    m, gt, err, status = check_accuracy("Handhole W", res["W_m"], W, 2.0)
    report.append(["Handhole W", gt, m, err, status])

    # Print Summary Table
    print("\n" + "="*80)
    print(f"{'Category':<20} | {'GT (m)':<10} | {'Measured':<10} | {'Error %':<10} | {'Status':<10}")
    print("-"*80)
    for row in report:
        print(f"{row[0]:<20} | {row[1]:<10.4f} | {row[2]:<10.4f} | {row[3]:<10.2f} | {row[4]:<10}")
    print("="*80 + "\n")

    for row in report:
        assert row[4] == "PASS", f"{row[0]} failed accuracy target"

def test_confidence_correlation():
    # High: > 50000
    pcd_high = create_box_pcd(1, 1, 1, num_points=50001)
    assert measure_asset(pcd_high, "handhole")["confidence"] == "high"

    # Medium: 5001 - 50000
    pcd_med = create_box_pcd(1, 1, 1, num_points=10000)
    assert measure_asset(pcd_med, "handhole")["confidence"] == "medium"

    # Low: <= 5000
    pcd_low = create_box_pcd(1, 1, 1, num_points=1000)
    assert measure_asset(pcd_low, "handhole")["confidence"] == "low"
