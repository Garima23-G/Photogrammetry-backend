import numpy as np
import open3d as o3d

from measurement.utils import fit_circle_ransac


def _pca_shape_features(points):
    if len(points) < 20:
        return None

    center = np.mean(points, axis=0)
    centered = points - center
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    proj = centered @ eigvecs
    mins = np.min(proj, axis=0)
    maxs = np.max(proj, axis=0)
    ext = np.maximum(maxs - mins, 1e-6)
    ext_sorted = np.sort(ext)[::-1]
    l_dim, w_dim, h_dim = ext_sorted

    elongation = float(l_dim / max(w_dim, 1e-6))
    cross_symmetry = 1.0 - abs(w_dim - h_dim) / max(w_dim, h_dim, 1e-6)
    cross_symmetry = float(np.clip(cross_symmetry, 0.0, 1.0))

    z_axis = int(np.argmin(ext))
    other_axes = [i for i in range(3) if i != z_axis]
    z_vals = proj[:, z_axis]
    z_span = max(float(np.max(z_vals) - np.min(z_vals)), 1e-6)
    top_mask = z_vals > (np.max(z_vals) - 0.15 * z_span)

    circle_quality = 0.0
    rectangularity = 0.0
    if np.sum(top_mask) >= 20:
        top_xy = proj[top_mask][:, other_axes]
        # Keep ROI scoring deterministic even though RANSAC sampling is random.
        rng_state = np.random.get_state()
        np.random.seed(int((len(top_xy) * 2654435761) & 0xFFFFFFFF))
        try:
            _, _, radius = fit_circle_ransac(top_xy, iterations=120, threshold=0.02)
        finally:
            np.random.set_state(rng_state)
        if radius > 1e-6:
            d = np.sqrt((top_xy[:, 0] - np.mean(top_xy[:, 0])) ** 2 + (top_xy[:, 1] - np.mean(top_xy[:, 1])) ** 2)
            residual = float(np.mean(np.abs(d - np.median(d))) / max(radius, 1e-6))
            circle_quality = float(np.clip(np.exp(-4.0 * residual), 0.0, 1.0))

        hull = o3d.geometry.PointCloud()
        hull.points = o3d.utility.Vector3dVector(
            np.hstack([top_xy, np.zeros((len(top_xy), 1), dtype=np.float64)])
        )
        aabb = hull.get_axis_aligned_bounding_box()
        ext2 = aabb.get_extent()
        bbox_area = max(float(ext2[0] * ext2[1]), 1e-6)
        hull2d = top_xy.astype(np.float32)
        if len(hull2d) >= 3:
            hull_poly = cv2_convex_hull_area(hull2d)
            rectangularity = float(np.clip(hull_poly / bbox_area, 0.0, 1.0))

    return {
        "elongation": elongation,
        "cross_symmetry": cross_symmetry,
        "circle_quality": float(circle_quality),
        "rectangularity": float(rectangularity),
    }


def cv2_convex_hull_area(points_2d):
    # Local import avoids hard dependency for non-segmentation paths.
    import cv2

    hull = cv2.convexHull(points_2d)
    return float(cv2.contourArea(hull))


def _cluster_score(features, category):
    elongated = float(np.clip((features["elongation"] - 1.2) / 2.5, 0.0, 1.0))
    compact = float(np.clip(1.0 - elongated, 0.0, 1.0))
    circle_quality = features["circle_quality"]
    rectangularity = features["rectangularity"]
    cross_symmetry = features["cross_symmetry"]

    if category == "trench":
        return 3.0 * elongated + 2.4 * rectangularity + 2.6 * (1.0 - cross_symmetry) + 1.6 * (1.0 - circle_quality)
    if category == "manhole":
        return 2.5 * compact + 2.2 * cross_symmetry + 2.0 * circle_quality + 0.4 * (1.0 - elongated)
    if category == "duct":
        return 2.8 * elongated + 2.0 * cross_symmetry + 1.9 * circle_quality + 0.5 * (1.0 - rectangularity)
    if category == "handhole":
        return 2.6 * compact + 2.4 * rectangularity + 1.0 * (1.0 - circle_quality)
    return 0.0


def _extract_clusters(pcd):
    points = np.asarray(pcd.points)
    if len(points) < 80:
        return []

    extent = np.max(points, axis=0) - np.min(points, axis=0)
    diag = float(np.linalg.norm(extent))
    eps = float(np.clip(0.04 * max(diag, 1.0), 0.04, 0.25))
    min_points = max(10, int(min(40, len(points) * 0.003)))

    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0:
        return []

    clusters = []
    unique_labels = [lb for lb in np.unique(labels) if lb >= 0]
    for lb in unique_labels:
        idx = np.where(labels == lb)[0]
        if len(idx) < 20:
            continue
        clusters.append((lb, idx))
    return clusters


def _pick_best_cluster(pcd, category):
    clusters = _extract_clusters(pcd)
    if not clusters:
        return None

    total_points = len(pcd.points)
    min_cluster_points = max(60, int(total_points * 0.08))

    best_idx = None
    best_score = -1e9
    for _, idx in clusters:
        if len(idx) < min_cluster_points:
            continue
        pts = np.asarray(pcd.points)[idx]
        features = _pca_shape_features(pts)
        if features is None:
            continue

        score = _cluster_score(features, category)
        # Mild preference for substantial clusters.
        score += 0.25 * np.log1p(float(len(idx)))
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:
        return None
    return pcd.select_by_index(best_idx)


def segment_asset_roi(pcd, category_tag):
    """
    Category-aware geometric localization for measurement.
    Returns a focused ROI point cloud; falls back to the input cloud when uncertain.
    """
    if pcd is None or pcd.is_empty():
        return pcd

    points = np.asarray(pcd.points)
    if len(points) < 120:
        return pcd

    filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if filtered.is_empty() or len(filtered.points) < 80:
        return pcd

    candidates = [filtered]

    try:
        plane_model, inliers = filtered.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=150)
        outlier_cloud = filtered.select_by_index(inliers, invert=True)
        if len(outlier_cloud.points) >= 80:
            candidates.append(outlier_cloud)
    except Exception:
        pass

    best_candidate = None
    best_candidate_size = -1
    for candidate in candidates:
        roi = _pick_best_cluster(candidate, category_tag)
        if roi is None or roi.is_empty():
            continue
        size = len(roi.points)
        if size > best_candidate_size:
            best_candidate = roi
            best_candidate_size = size

    if best_candidate is None or best_candidate_size < 60:
        return filtered

    # Rehydrate ROI from the original cloud to preserve boundary points
    # that may be trimmed by outlier filtering.
    try:
        bbox = best_candidate.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()
        pad = max(0.01, 0.04 * float(np.linalg.norm(ext)))
        expanded = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bbox.min_bound - pad,
            max_bound=bbox.max_bound + pad,
        )
        original_idx = expanded.get_point_indices_within_bounding_box(pcd.points)
        if len(original_idx) >= 60:
            return pcd.select_by_index(original_idx)
    except Exception:
        pass

    return best_candidate


def score_roi_against_category(pcd, category_tag):
    if pcd is None or pcd.is_empty():
        return 0.0, None
    points = np.asarray(pcd.points)
    features = _pca_shape_features(points)
    if features is None:
        return 0.0, None
    return float(_cluster_score(features, category_tag)), features


def roi_matches_category(pcd, category_tag):
    score, features = score_roi_against_category(pcd, category_tag)
    min_scores = {
        "trench": 2.8,
        "manhole": 3.0,
        "duct": 3.0,
        "handhole": 2.8,
    }
    threshold = float(min_scores.get(category_tag, 0.0))
    if threshold <= 0.0:
        return True, score, features

    if features is None:
        return False, score, features

    # Hard gates to reduce cross-category false positives.
    if category_tag == "trench":
        gate_ok = (
            features["elongation"] >= 1.40
            and features["cross_symmetry"] <= 0.75
            and features["rectangularity"] >= 0.70
            and features["circle_quality"] <= 0.80
        )
    elif category_tag == "manhole":
        gate_ok = (
            features["elongation"] <= 1.9
            and features["cross_symmetry"] >= 0.60
            and features["rectangularity"] >= 0.45
        )
    elif category_tag == "duct":
        gate_ok = (
            features["elongation"] >= 1.35
            and features["cross_symmetry"] >= 0.45
        )
    elif category_tag == "handhole":
        gate_ok = (
            features["elongation"] <= 2.2
            and features["rectangularity"] >= 0.55
            and features["circle_quality"] <= 0.85
        )
    else:
        gate_ok = True

    return bool(gate_ok and score >= threshold), score, features
