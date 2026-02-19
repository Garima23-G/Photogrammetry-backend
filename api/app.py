import os
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from reconstruction.fusion import reconstruct_scene
from measurement.engine import measure_asset

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_reconstruction():
    """
    Endpoint for end-to-end reconstruction and measurement.
    Expects multipart/form-data:
    - rgb_0, rgb_1, ... : RGB image files
    - depth_0, depth_1, ... : 16-bit depth image files
    - intrinsics: JSON string of 3x3 matrix
    - poses: JSON string of list of 4x4 matrices
    - category: asset category tag
    """
    try:
        # 1. Parse JSON metadata
        intrinsics_json = request.form.get('intrinsics')
        poses_json = request.form.get('poses')
        category = request.form.get('category')

        if not all([intrinsics_json, poses_json, category]):
            return jsonify({"error": "Missing metadata"}), 400

        intrinsic_matrix = np.array(json.loads(intrinsics_json), dtype=np.float64)
        poses = [np.array(p, dtype=np.float64) for p in json.loads(poses_json)]

        # 2. Parse image files
        rgb_frames = []
        depth_frames = []

        # We expect indices 0 to len(poses)-1
        for i in range(len(poses)):
            rgb_file = request.files.get(f'rgb_{i}')
            depth_file = request.files.get(f'depth_{i}')

            if not rgb_file or not depth_file:
                return jsonify({"error": f"Missing frame {i}"}), 400

            # Read RGB
            rgb_bytes = rgb_file.read()
            rgb_arr = np.frombuffer(rgb_bytes, np.uint8)
            rgb = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # Read Depth (16-bit)
            depth_bytes = depth_file.read()
            depth_arr = np.frombuffer(depth_bytes, np.uint8)
            depth = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)

            if rgb is None or depth is None:
                return jsonify({"error": f"Failed to decode frame {i}"}), 400

            rgb_frames.append(rgb)
            depth_frames.append(depth)

        # 3. Stage 04: Reconstruction
        reconstruction_result = reconstruct_scene(
            rgb_frames, depth_frames, intrinsic_matrix, poses, category
        )

        # 4. Stage 05: Measurement
        measurement_result = measure_asset(
            reconstruction_result['pcd'], category
        )

        # 5. Return JSON result
        return jsonify(measurement_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
