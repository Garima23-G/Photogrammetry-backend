import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from reconstruction.fusion import reconstruct_scene
from measurement.engine import measure_asset
from measurement.classifier import (
    detect_asset_category_hybrid,
    detect_asset_category_from_images,
    detect_asset_region_from_images,
)
from measurement.categories import normalize_category_name

app = Flask(__name__)

# Frontend fallback depth scale used when depth is sent as 8-bit PNG.
# If depth pixel is in [0,255], we map back to millimeters in [0, 8000].
FALLBACK_DEPTH_MM_MAX = 8000


def _log_request_overview(endpoint):
    form_keys = sorted(list(request.form.keys()))
    file_keys = sorted(list(request.files.keys()))
    rgb_keys = sorted([k for k in file_keys if k.startswith("rgb_")])
    depth_keys = sorted([k for k in file_keys if k.startswith("depth_")])
    app.logger.info(
        "[%s] content_type=%s form_keys=%s file_keys=%s rgb_count=%d depth_count=%d",
        endpoint,
        request.content_type,
        form_keys,
        file_keys,
        len(rgb_keys),
        len(depth_keys),
    )


def _log_bad_request(endpoint, err):
    intrinsics_present = bool(request.form.get("intrinsics"))
    poses_raw = request.form.get("poses")
    poses_present = bool(poses_raw)
    poses_count_hint = None
    if poses_raw:
        try:
            poses_count_hint = len(json.loads(poses_raw))
        except Exception:
            poses_count_hint = "parse_error"

    app.logger.warning(
        "[%s] bad_request error=%s intrinsics_present=%s poses_present=%s poses_count_hint=%s",
        endpoint,
        str(err),
        intrinsics_present,
        poses_present,
        poses_count_hint,
    )

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
                "POST /pose_sanity": "Validate intrinsics/poses metadata before reconstruction.",
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
    depth = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None

    # Preferred format: uint16 single-channel depth in millimeters.
    if depth.ndim == 2 and depth.dtype == np.uint16:
        return depth

    # Fallback: grayscale uint8 depth -> up-scale back to millimeters.
    if depth.ndim == 2 and depth.dtype == np.uint8:
        return (depth.astype(np.uint16) * FALLBACK_DEPTH_MM_MAX // 255)

    # Fallback: color PNG (e.g. ARGB_8888 export) -> grayscale -> millimeters.
    if depth.ndim == 3:
        if depth.shape[2] == 4:
            gray = cv2.cvtColor(depth, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        return (gray.astype(np.uint16) * FALLBACK_DEPTH_MM_MAX // 255)

    return None


def _parse_intrinsics_and_poses():
    intrinsics_json = request.form.get("intrinsics")
    poses_json = request.form.get("poses")
    if not intrinsics_json or not poses_json:
        raise ValueError("Missing metadata")

    intrinsic_matrix = np.array(json.loads(intrinsics_json), dtype=np.float64)
    poses = [np.array(p, dtype=np.float64) for p in json.loads(poses_json)]
    return intrinsic_matrix, poses


def _analyze_intrinsics_and_poses(intrinsic_matrix, poses):
    issues = []
    warnings = []

    if intrinsic_matrix.shape != (3, 3):
        issues.append("Intrinsics must be a 3x3 matrix.")
    elif not np.all(np.isfinite(intrinsic_matrix)):
        issues.append("Intrinsics contain non-finite values.")
    else:
        fx = float(intrinsic_matrix[0, 0])
        fy = float(intrinsic_matrix[1, 1])
        if fx <= 0 or fy <= 0:
            issues.append("Intrinsics focal lengths (fx, fy) must be positive.")

    if not poses:
        issues.append("At least one pose is required.")

    translation_vectors = []
    nonzero_last_row_xyz = 0
    near_zero_translation_column = 0
    invalid_pose_count = 0

    for idx, pose in enumerate(poses):
        if pose.shape != (4, 4):
            issues.append(f"Pose {idx} is not 4x4.")
            invalid_pose_count += 1
            continue
        if not np.all(np.isfinite(pose)):
            issues.append(f"Pose {idx} contains non-finite values.")
            invalid_pose_count += 1
            continue

        translation = pose[:3, 3]
        translation_vectors.append(translation)
        if np.linalg.norm(translation) < 1e-4:
            near_zero_translation_column += 1

        last_row_xyz_norm = float(np.linalg.norm(pose[3, :3]))
        if last_row_xyz_norm > 1e-4:
            nonzero_last_row_xyz += 1

        last_row_err = float(np.max(np.abs(pose[3] - np.array([0.0, 0.0, 0.0, 1.0]))))
        if last_row_err > 1e-2:
            warnings.append(
                f"Pose {idx} has unusual last row; expected close to [0, 0, 0, 1]."
            )

        rotation = pose[:3, :3]
        ortho_err = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
        det = float(np.linalg.det(rotation))
        if ortho_err > 0.1 or abs(det) < 0.5 or abs(det) > 1.5:
            issues.append(
                f"Pose {idx} rotation block is not a valid rotation (det={det:.4f}, ortho_err={ortho_err:.4f})."
            )
        elif ortho_err > 0.01 or abs(det - 1.0) > 0.05:
            warnings.append(
                f"Pose {idx} rotation block is noisy (det={det:.4f}, ortho_err={ortho_err:.4f})."
            )

    pose_count = len(poses)
    step_count = 0
    median_step_m = 0.0
    max_step_m = 0.0
    if len(translation_vectors) >= 2:
        translation_np = np.asarray(translation_vectors, dtype=np.float64)
        steps = np.linalg.norm(np.diff(translation_np, axis=0), axis=1)
        step_count = int(len(steps))
        median_step_m = float(np.median(steps))
        max_step_m = float(np.max(steps))
        if median_step_m < 1e-5:
            warnings.append("Pose sequence has near-zero camera motion.")
        if max_step_m > 5.0:
            warnings.append("Pose sequence has very large frame-to-frame motion (>5m).")

    if pose_count > 0:
        likely_transposed = (
            nonzero_last_row_xyz >= max(1, pose_count // 2)
            and near_zero_translation_column >= max(1, pose_count // 2)
        )
        if likely_transposed:
            issues.append(
                "Poses look transposed/column-major in JSON (translation appears in last row, not last column)."
            )

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "summary": {
            "pose_count": pose_count,
            "invalid_pose_count": invalid_pose_count,
            "step_count": step_count,
            "median_step_m": median_step_m,
            "max_step_m": max_step_m,
        },
    }


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
        _log_request_overview("classify")
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
            detection, detection_preview_b64 = detect_asset_region_from_images(rgb_frames, include_preview=True)
            return jsonify(
                {
                    "category": category,
                    "category_source": "auto_hybrid",
                    "category_confidence": confidence,
                    "category_scores": scores,
                    "category_details": details,
                    "detection": detection,
                    "detection_preview_jpeg_base64": detection_preview_b64,
                    "reconstruction_summary": _build_reconstruction_summary(reconstruction_result),
                }
            )

        rgb_frames = _parse_rgb_frames_only()
        category, confidence, scores = detect_asset_category_from_images(rgb_frames)
        detection, detection_preview_b64 = detect_asset_region_from_images(rgb_frames, include_preview=True)
        return jsonify(
            {
                "category": category,
                "category_source": "auto_rgb",
                "category_confidence": confidence,
                "category_scores": scores,
                "detection": detection,
                "detection_preview_jpeg_base64": detection_preview_b64,
            }
        )
    except ValueError as e:
        _log_bad_request("classify", e)
        return jsonify({"error": str(e)}), 400
    except MemoryError as e:
        app.logger.exception("[classify] memory error")
        return jsonify(
            {
                "error": "Server out of memory during reconstruction/classification.",
                "details": str(e),
            }
        ), 507
    except Exception as e:
        app.logger.exception("[classify] unhandled error")
        return jsonify({"error": str(e)}), 500


@app.route("/reconstruct", methods=["POST"])
def reconstruct_only():
    """
    Reconstruction-only endpoint for Stage 04.
    """
    try:
        _log_request_overview("reconstruct")
        intrinsic_matrix, poses = _parse_intrinsics_and_poses()
        rgb_frames, depth_frames = _parse_rgbd_frames_from_poses(poses)
        category = normalize_category_name(request.form.get("category")) or "unclassified"

        reconstruction_result = reconstruct_scene(
            rgb_frames, depth_frames, intrinsic_matrix, poses, category
        )
        return jsonify(_build_reconstruction_summary(reconstruction_result))
    except ValueError as e:
        _log_bad_request("reconstruct", e)
        return jsonify({"error": str(e)}), 400
    except MemoryError as e:
        app.logger.exception("[reconstruct] memory error")
        return jsonify(
            {
                "error": "Server out of memory during reconstruction.",
                "details": str(e),
            }
        ), 507
    except Exception as e:
        app.logger.exception("[reconstruct] unhandled error")
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
        _log_request_overview("process")
        intrinsic_matrix, poses = _parse_intrinsics_and_poses()
        rgb_frames, depth_frames = _parse_rgbd_frames_from_poses(poses)
        requested_category = normalize_category_name(request.form.get("category"))
        category_for_recon = requested_category if requested_category else "unclassified"
        reconstruction_result = reconstruct_scene(
            rgb_frames, depth_frames, intrinsic_matrix, poses, category_for_recon
        )

        # Always compute predicted category so UI can surface mismatch warnings.
        predicted_category, predicted_confidence, fused_scores, _ = detect_asset_category_hybrid(
            rgb_frames, reconstruction_result["pcd"]
        )
        predicted_score = float(fused_scores.get(predicted_category, 0.0))

        category_source = "request"
        category = requested_category
        if not category:
            category = predicted_category
            category_source = "auto_hybrid"

        measurement_result = measure_asset(reconstruction_result["pcd"], category)
        measurement_result["category_source"] = category_source
        measurement_result["measurement_confidence"] = measurement_result.get("confidence")
        measurement_result["predicted_category"] = predicted_category
        measurement_result["predicted_category_confidence"] = predicted_confidence
        measurement_result["predicted_category_score"] = predicted_score
        measurement_result["category_mismatch"] = bool(
            requested_category and predicted_category != requested_category
        )
        if measurement_result["category_mismatch"]:
            measurement_result["category_mismatch_message"] = (
                f"Requested '{requested_category}', but model predicts '{predicted_category}'."
            )
        else:
            measurement_result["category_mismatch_message"] = ""
        return jsonify(measurement_result)
    except ValueError as e:
        _log_bad_request("process", e)
        return jsonify({"error": str(e)}), 400
    except MemoryError as e:
        app.logger.exception("[process] memory error")
        return jsonify(
            {
                "error": "Server out of memory during end-to-end processing.",
                "details": str(e),
            }
        ), 507
    except Exception as e:
        app.logger.exception("[process] unhandled error")
        return jsonify({"error": str(e)}), 500


@app.route("/pose_sanity", methods=["POST"])
def pose_sanity():
    """
    Validate capture metadata only (intrinsics + poses), without processing images.
    """
    try:
        _log_request_overview("pose_sanity")
        intrinsic_matrix, poses = _parse_intrinsics_and_poses()
        analysis = _analyze_intrinsics_and_poses(intrinsic_matrix, poses)
        return jsonify(analysis)
    except ValueError as e:
        _log_bad_request("pose_sanity", e)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("[pose_sanity] unhandled error")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
