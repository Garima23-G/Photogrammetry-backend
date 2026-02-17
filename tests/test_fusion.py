import numpy as np
import open3d as o3d
import pytest
from reconstruction.fusion import reconstruct_scene

def test_reconstruct_scene():
    # Setup synthetic data
    h, w = 480, 640
    intrinsic_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    # Create a simple scene: a plane at z=2m
    rgb_frames = []
    depth_frames = []
    poses = []

    for i in range(3):
        # RGB: all white
        rgb = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Depth: plane at 2m in front of camera (2000mm)
        depth = np.ones((h, w), dtype=np.uint16) * 2000

        # Pose: Camera moving along X axis
        pose = np.eye(4)
        pose[0, 3] = i * 0.1 # Move 10cm along X

        rgb_frames.append(rgb)
        depth_frames.append(depth)
        poses.append(pose)

    category_tag = "test_asset"

    # Run reconstruction
    result = reconstruct_scene(rgb_frames, depth_frames, intrinsic_matrix, poses, category_tag)

    # Assertions
    assert "points" in result
    assert "pcd" in result
    assert "mesh" in result
    assert result["category_tag"] == category_tag

    assert isinstance(result["points"], np.ndarray)
    assert isinstance(result["pcd"], o3d.geometry.PointCloud)
    assert isinstance(result["mesh"], o3d.geometry.TriangleMesh)

    # Check if we have some points
    assert len(result["points"]) > 0
    # The mesh should not be empty if enough points are integrated
    assert not result["pcd"].is_empty()
    assert not result["mesh"].is_empty()

def test_estimate_bounding_box():
    from reconstruction.fusion import estimate_bounding_box

    h, w = 480, 640
    intrinsic_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])

    # Single frame, depth at 2m
    depth_frames = [np.ones((h, w), dtype=np.uint16) * 2000]
    poses = [np.eye(4)]

    bbox_min, bbox_max = estimate_bounding_box(depth_frames, intrinsic_matrix, poses)

    assert bbox_min is not None
    assert bbox_max is not None
    # Depth at 2m, so Z should be around 2.0
    assert np.allclose(bbox_min[2], 2.0, atol=0.1)
    assert np.allclose(bbox_max[2], 2.0, atol=0.1)
