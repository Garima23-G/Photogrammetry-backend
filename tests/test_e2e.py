import io
import json
import numpy as np
import cv2
import pytest
from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

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
