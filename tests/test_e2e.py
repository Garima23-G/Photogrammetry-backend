import io
import json
import numpy as np
import cv2
import pytest
import open3d as o3d
from api.app import app
from measurement.classifier import detect_asset_category_hybrid

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_api_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    res_json = response.get_json()
    assert res_json["status"] == "ok"
    assert "endpoints" in res_json


def test_e2e_pipeline(client):
    # 1. Prepare synthetic data (Handhole)
    L_gt, W_gt, H_gt = 1.0, 0.8, 0.6
    intrinsic_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    # Simple one-frame "reconstruction" simulation
    # We'll create a single RGB and Depth frame that when integrated
    # will produce something measurable.
    # Actually, Stage 04 needs a sequence. Let's provide 2 frames.

    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = 0.1 # Move 10cm

    data = {
        'intrinsics': json.dumps(intrinsic_matrix.tolist()),
        'poses': json.dumps([p.tolist() for p in poses]),
        'category': 'handhole'
    }

    # Create synthetic images
    # For a handhole at 2m depth
    # Depth 16-bit in mm
    depth_img = np.zeros((480, 640), dtype=np.uint16)
    depth_img[100:380, 200:440] = 2000

    rgb_img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    _, rgb_encoded = cv2.imencode('.png', rgb_img)
    _, depth_encoded = cv2.imencode('.png', depth_img)

    for i in range(2):
        data[f'rgb_{i}'] = (io.BytesIO(rgb_encoded.tobytes()), f'rgb_{i}.png')
        data[f'depth_{i}'] = (io.BytesIO(depth_encoded.tobytes()), f'depth_{i}.png')

    # 2. POST to API
    response = client.post('/process', data=data, content_type='multipart/form-data')

    # 3. Assertions
    assert response.status_code == 200
    res_json = response.get_json()

    assert res_json['category'] == 'handhole'
    assert 'L_m' in res_json
    assert 'W_m' in res_json
    assert 'H_m' in res_json
    assert res_json['point_count'] > 0
    assert res_json['confidence'] in ['low', 'medium', 'high']

    # Check if dimensions are reasonable (Handhole OBB sorted L > W > H)
    # The box projected onto 2m depth will have some dimensions
    assert res_json['L_m'] > 0
    assert res_json['W_m'] > 0
    assert res_json['H_m'] >= 0 # Depth of the plane might be small


def test_e2e_pipeline_with_auto_category(client):
    intrinsic_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = 0.1

    data = {
        'intrinsics': json.dumps(intrinsic_matrix.tolist()),
        'poses': json.dumps([p.tolist() for p in poses]),
    }

    depth_img = np.zeros((480, 640), dtype=np.uint16)
    depth_img[100:380, 200:440] = 2000
    rgb_img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    _, rgb_encoded = cv2.imencode('.png', rgb_img)
    _, depth_encoded = cv2.imencode('.png', depth_img)

    for i in range(2):
        data[f'rgb_{i}'] = (io.BytesIO(rgb_encoded.tobytes()), f'rgb_{i}.png')
        data[f'depth_{i}'] = (io.BytesIO(depth_encoded.tobytes()), f'depth_{i}.png')

    response = client.post('/process', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    res_json = response.get_json()

    assert res_json['category_source'] == 'auto_hybrid'
    assert res_json['category'] in ['trench', 'manhole', 'duct', 'handhole']
    assert 'category_confidence' in res_json
    assert 'category_score' in res_json
    assert 'category_scores' not in res_json
    assert 'category_details' not in res_json


def test_hybrid_classifier_scores_are_normalized():
    rgb = np.ones((240, 320, 3), dtype=np.uint8) * 255
    pts = np.random.rand(2000, 3).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    category, confidence, scores, details = detect_asset_category_hybrid([rgb], pcd)

    assert category in ['trench', 'manhole', 'duct', 'handhole']
    assert confidence in ['low', 'medium', 'high']
    assert abs(sum(scores.values()) - 1.0) < 1e-6
    assert "rgb_scores" in details
    assert "geometry_scores" in details


def _create_rgb_depth_payload(include_metadata=True):
    intrinsic_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = 0.1

    data = {}
    if include_metadata:
        data["intrinsics"] = json.dumps(intrinsic_matrix.tolist())
        data["poses"] = json.dumps([p.tolist() for p in poses])

    depth_img = np.zeros((480, 640), dtype=np.uint16)
    depth_img[100:380, 200:440] = 2000
    rgb_img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    _, rgb_encoded = cv2.imencode(".png", rgb_img)
    _, depth_encoded = cv2.imencode(".png", depth_img)

    for i in range(2):
        data[f"rgb_{i}"] = (io.BytesIO(rgb_encoded.tobytes()), f"rgb_{i}.png")
        data[f"depth_{i}"] = (io.BytesIO(depth_encoded.tobytes()), f"depth_{i}.png")

    return data


def test_classify_rgb_only(client):
    data = _create_rgb_depth_payload(include_metadata=False)
    # RGB-only mode should ignore depth payload and classify from images.
    data.pop("depth_0")
    data.pop("depth_1")

    response = client.post("/classify", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_json = response.get_json()

    assert res_json["category_source"] == "auto_rgb"
    assert res_json["category"] in ["trench", "manhole", "duct", "handhole"]
    assert "category_scores" in res_json
    assert "category_confidence" in res_json
    assert "detection" in res_json


def test_classify_hybrid(client):
    data = _create_rgb_depth_payload(include_metadata=True)
    response = client.post("/classify", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_json = response.get_json()

    assert res_json["category_source"] == "auto_hybrid"
    assert res_json["category"] in ["trench", "manhole", "duct", "handhole"]
    assert "category_details" in res_json
    assert "detection" in res_json
    assert "reconstruction_summary" in res_json
    assert res_json["reconstruction_summary"]["point_count"] > 0


def test_reconstruct_only_endpoint(client):
    data = _create_rgb_depth_payload(include_metadata=True)
    response = client.post("/reconstruct", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_json = response.get_json()

    assert "point_count" in res_json
    assert "triangle_count" in res_json
    assert "pcd_empty" in res_json
    assert "mesh_empty" in res_json


def test_pose_sanity_valid_payload(client):
    intrinsic_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = 0.1

    data = {
        "intrinsics": json.dumps(intrinsic_matrix.tolist()),
        "poses": json.dumps([p.tolist() for p in poses]),
    }

    response = client.post("/pose_sanity", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_json = response.get_json()

    assert res_json["ok"] is True
    assert res_json["issues"] == []
    assert res_json["summary"]["pose_count"] == 2


def test_pose_sanity_detects_transposed_pose_layout(client):
    intrinsic_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    pose_a = np.eye(4)
    pose_b = np.eye(4)
    # Simulate transposed serialization where translation lands in last row.
    pose_a[3, 0] = 0.1
    pose_b[3, 0] = 0.2

    data = {
        "intrinsics": json.dumps(intrinsic_matrix.tolist()),
        "poses": json.dumps([pose_a.tolist(), pose_b.tolist()]),
    }

    response = client.post("/pose_sanity", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_json = response.get_json()

    assert res_json["ok"] is False
    assert any("column-major" in issue for issue in res_json["issues"])
