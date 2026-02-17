import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from reconstruction.fusion import reconstruct_scene
from measurement.engine import measure_asset
from measurement.classifier import (
    detect_asset_category_hybrid,
    detect_asset_category_from_images,
)
from measurement.categories import normalize_category_name

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "service": "Photogrammetry Backend API",
            "status": "ok",
            "endpoints": {
                "POST /classify": "Classify asset (RGB-only or hybrid RGBD).",
                "POST /reconstruct": "Run Stage-04 reconstruction only.",
                "POST /process": "Run end-to-end reconstruction + classification + measurement.",
            },
        }
    )


def _decode_rgb_file(rgb_file):
    rgb_bytes = rgb_file.read()
    rgb_arr = np.frombuffer(rgb_bytes, np.uint8)
    rgb = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def _decode_depth_file(depth_file):
    depth_bytes = depth_file.read()
    depth_arr = np.frombuffer(depth_bytes, np.uint8)
    return cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)


def _parse_intrinsics_and_poses():
    intrinsics_json = request.form.get("intrinsics")
    poses_json = request.form.get("poses")
    if not intrinsics_json or not poses_json:
        raise ValueError("Missing metadata")

    intrinsic_matrix = np.array(json.loads(intrinsics_json), dtype=np.float64)
    poses = [np.array(p, dtype=np.float64) for p in json.loads(poses_json)]
    return intrinsic_matrix, poses


def _parse_rgbd_frames_from_poses(poses):
    rgb_frames = []
    depth_frames = []
    for i in range(len(poses)):
        rgb_file = request.files.get(f"rgb_{i}")
        depth_file = request.files.get(f"depth_{i}")
        if not rgb_file or not depth_file:
            raise ValueError(f"Missing frame {i}")

        rgb = _decode_rgb_file(rgb_file)
        depth = _decode_depth_file(depth_file)
        if rgb is None or depth is None:
            raise ValueError(f"Failed to decode frame {i}")

        rgb_frames.append(rgb)
        depth_frames.append(depth)
    return rgb_frames, depth_frames


def _parse_rgb_frames_only():
    rgb_indices = []
    for key in request.files.keys():
        if key.startswith("rgb_"):
            suffix = key.split("_", 1)[1]
            if suffix.isdigit():
                rgb_indices.append(int(suffix))

    rgb_indices = sorted(set(rgb_indices))
    if not rgb_indices:
        raise ValueError("Missing RGB frames")

    rgb_frames = []
    for idx in rgb_indices:
        rgb_file = request.files.get(f"rgb_{idx}")
        if not rgb_file:
            raise ValueError(f"Missing RGB frame {idx}")
        rgb = _decode_rgb_file(rgb_file)
        if rgb is None:
            raise ValueError(f"Failed to decode RGB frame {idx}")
        rgb_frames.append(rgb)
    return rgb_frames


def _build_reconstruction_summary(reconstruction_result):
    points_np = reconstruction_result["points"]
    pcd = reconstruction_result["pcd"]
    mesh = reconstruction_result["mesh"]
    point_count = int(len(points_np))

    bbox_min = None
    bbox_max = None
    if point_count > 0:
        bbox_min = np.min(points_np, axis=0).astype(float).tolist()
        bbox_max = np.max(points_np, axis=0).astype(float).tolist()

    return {
        "category_tag": reconstruction_result["category_tag"],
        "point_count": point_count,
        "triangle_count": int(len(mesh.triangles)),
        "pcd_empty": bool(pcd.is_empty()),
        "mesh_empty": bool(mesh.is_empty()),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }


@app.route("/classify", methods=["POST"])
def classify_asset():
    """
    Classification endpoint.

    Modes:
    - RGB-only: send rgb_0..rgb_n
    - Hybrid RGB+3D: send rgb_i + depth_i + intrinsics + poses
    """
    try:
        has_metadata = bool(request.form.get("intrinsics") and request.form.get("poses"))
        if has_metadata:
            intrinsic_matrix, poses = _parse_intrinsics_and_poses()
            rgb_frames, depth_frames = _parse_rgbd_frames_from_poses(poses)
            reconstruction_result = reconstruct_scene(
                rgb_frames, depth_frames, intrinsic_matrix, poses, "unclassified"
            )
            category, confidence, scores, details = detect_asset_category_hybrid(
                rgb_frames, reconstruction_result["pcd"]
            )
            return jsonify(
                {
                    "category": category,
                    "category_source": "auto_hybrid",
                    "category_confidence": confidence,
                    "category_scores": scores,
                    "category_details": details,
                    "reconstruction_summary": _build_reconstruction_summary(reconstruction_result),
                }
            )

        rgb_frames = _parse_rgb_frames_only()
        category, confidence, scores = detect_asset_category_from_images(rgb_frames)
        return jsonify(
            {
                "category": category,
                "category_source": "auto_rgb",
                "category_confidence": confidence,
                "category_scores": scores,
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reconstruct", methods=["POST"])
def reconstruct_only():
    """
    Reconstruction-only endpoint for Stage 04.
    """
    try:
        intrinsic_matrix, poses = _parse_intrinsics_and_poses()
        rgb_frames, depth_frames = _parse_rgbd_frames_from_poses(poses)
        category = normalize_category_name(request.form.get("category")) or "unclassified"

        reconstruction_result = reconstruct_scene(
            rgb_frames, depth_frames, intrinsic_matrix, poses, category
        )
        return jsonify(_build_reconstruction_summary(reconstruction_result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process', methods=['POST'])
def process_reconstruction():
    """
    Endpoint for end-to-end reconstruction + classification + measurement.
    Expects multipart/form-data:
    - rgb_0, rgb_1, ... : RGB image files
    - depth_0, depth_1, ... : 16-bit depth image files
    - intrinsics: JSON string of 3x3 matrix
    - poses: JSON string of list of 4x4 matrices
    - category: optional asset category tag
    """
    try:
        intrinsic_matrix, poses = _parse_intrinsics_and_poses()
        rgb_frames, depth_frames = _parse_rgbd_frames_from_poses(poses)
        category = normalize_category_name(request.form.get("category"))

        category_for_recon = category if category else "unclassified"
        reconstruction_result = reconstruct_scene(
            rgb_frames, depth_frames, intrinsic_matrix, poses, category_for_recon
        )

        category_source = "request"
        category_confidence = None
        category_score = None
        if not category:
            category, category_confidence, fused_scores, _ = detect_asset_category_hybrid(
                rgb_frames, reconstruction_result["pcd"]
            )
            category_score = float(fused_scores.get(category, 0.0))
            category_source = "auto_hybrid"

        measurement_result = measure_asset(reconstruction_result["pcd"], category)
        measurement_result["category_source"] = category_source
        if category_confidence is not None:
            measurement_result["category_confidence"] = category_confidence
            measurement_result["category_score"] = category_score
        return jsonify(measurement_result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
