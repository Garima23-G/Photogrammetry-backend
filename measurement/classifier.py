import cv2
import numpy as np
from measurement.utils import fit_circle_ransac
from measurement.categories import get_registered_categories


def _normalize_scores(scores, categories):
    total = float(sum(max(v, 0.0) for v in scores.values()))
    if total <= 1e-9:
        uniform = 1.0 / max(1, len(categories))
        return {k: uniform for k in categories}
    return {k: float(max(scores.get(k, 0.0), 0.0) / total) for k in categories}


def _confidence_from_top(top_score):
    return "high" if top_score >= 0.58 else "medium" if top_score >= 0.4 else "low"


def _frame_features(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min(rgb.shape[:2]) // 8),
        param1=100,
        param2=28,
        minRadius=10,
        maxRadius=max(15, min(rgb.shape[:2]) // 3),
    )
    circle_count = 0 if circles is None else int(circles.shape[1])

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=50,
        minLineLength=max(20, min(rgb.shape[:2]) // 8),
        maxLineGap=8,
    )
    line_count = 0
    vertical_lines = 0
    horizontal_lines = 0
    if lines is not None:
        for ln in lines[:, 0, :]:
            x1, y1, x2, y2 = ln
            line_count += 1
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx <= 4 and dy > dx:
                vertical_lines += 1
            elif dy <= 4 and dx > dy:
                horizontal_lines += 1

    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangularity = 0.0
    aspect_ratio = 1.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = max(cv2.contourArea(largest), 1.0)
        x, y, w, h = cv2.boundingRect(largest)
        bbox_area = max(float(w * h), 1.0)
        rectangularity = float(area / bbox_area)
        aspect_ratio = float(max(w, h) / max(1.0, min(w, h)))

    return {
        "circles": float(circle_count),
        "lines": float(line_count),
        "vlines": float(vertical_lines),
        "hlines": float(horizontal_lines),
        "rectangularity": rectangularity,
        "aspect_ratio": aspect_ratio,
    }


def detect_asset_category_from_images(rgb_frames):
    """
    Heuristic image-based classifier for routing measurements.
    Returns: (category, confidence, normalized_scores)
    """
    categories = get_registered_categories()
    if not rgb_frames:
        scores = _normalize_scores({}, categories)
        fallback = "handhole" if "handhole" in categories else categories[0]
        return fallback, "low", scores

    agg = {
        "circles": 0.0,
        "lines": 0.0,
        "vlines": 0.0,
        "hlines": 0.0,
        "rectangularity": 0.0,
        "aspect_ratio": 1.0,
    }
    for rgb in rgb_frames:
        f = _frame_features(rgb)
        for k in ("circles", "lines", "vlines", "hlines", "rectangularity"):
            agg[k] += f[k]
        agg["aspect_ratio"] += f["aspect_ratio"]

    n = float(len(rgb_frames))
    agg["circles"] /= n
    agg["lines"] /= n
    agg["vlines"] /= n
    agg["hlines"] /= n
    agg["rectangularity"] /= n
    agg["aspect_ratio"] /= n

    elongated = max(0.0, agg["aspect_ratio"] - 1.3)
    orthogonal_lines = min(agg["vlines"], agg["hlines"])

    rule_scores = {
        "manhole": 2.8 * agg["circles"] + 0.5 * (1.0 - abs(agg["aspect_ratio"] - 1.0)),
        "duct": 1.7 * agg["circles"] + 1.3 * elongated + 0.3 * agg["lines"],
        "trench": 2.2 * elongated + 1.1 * agg["lines"] + 0.6 * agg["hlines"],
        "handhole": 2.4 * agg["rectangularity"] + 1.2 * orthogonal_lines,
    }
    raw_scores = {cat: rule_scores.get(cat, 0.1) for cat in categories}

    scores = _normalize_scores(raw_scores, categories)
    category = max(scores, key=scores.get)
    top = scores[category]
    confidence = _confidence_from_top(top)
    return category, confidence, scores


def detect_asset_category_from_geometry(pcd):
    """
    Geometry-only classifier from reconstructed point cloud.
    Returns: (category, confidence, normalized_scores)
    """
    categories = get_registered_categories()
    points = np.asarray(pcd.points)
    if len(points) < 50:
        scores = _normalize_scores({}, categories)
        fallback = "handhole" if "handhole" in categories else categories[0]
        return fallback, "low", scores

    center = np.mean(points, axis=0)
    centered = points - center

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    proj = centered @ eigvecs
    mins = np.min(proj, axis=0)
    maxs = np.max(proj, axis=0)
    ext = np.maximum(maxs - mins, 1e-6)
    ext_sorted = np.sort(ext)[::-1]
    l_dim, w_dim, h_dim = ext_sorted

    elongation = l_dim / max(w_dim, 1e-6)
    cross_symmetry = 1.0 - abs(w_dim - h_dim) / max(w_dim, h_dim, 1e-6)
    cross_symmetry = float(np.clip(cross_symmetry, 0.0, 1.0))

    # Circle quality from top slice in PCA frame.
    z_axis = np.argmin(ext)
    other_axes = [i for i in range(3) if i != z_axis]
    z_vals = proj[:, z_axis]
    z_span = max(np.max(z_vals) - np.min(z_vals), 1e-6)
    top_mask = z_vals > (np.max(z_vals) - 0.15 * z_span)
    circle_quality = 0.0
    rectangularity = 0.0
    if np.sum(top_mask) >= 40:
        top_xy = proj[top_mask][:, other_axes]
        _, _, radius = fit_circle_ransac(top_xy, iterations=150, threshold=0.02)
        if radius > 1e-6:
            d = np.sqrt((top_xy[:, 0] - np.mean(top_xy[:, 0])) ** 2 + (top_xy[:, 1] - np.mean(top_xy[:, 1])) ** 2)
            residual = float(np.mean(np.abs(d - np.median(d))) / max(radius, 1e-6))
            circle_quality = float(np.clip(np.exp(-4.0 * residual), 0.0, 1.0))

        hull = cv2.convexHull(top_xy.astype(np.float32))
        hull_area = max(cv2.contourArea(hull), 1e-6)
        x, y, w, h = cv2.boundingRect(hull)
        bbox_area = max(float(w * h), 1e-6)
        rectangularity = float(np.clip(hull_area / bbox_area, 0.0, 1.0))

    elongated_score = float(np.clip((elongation - 1.2) / 2.5, 0.0, 1.0))
    compact_score = float(np.clip(1.0 - elongated_score, 0.0, 1.0))

    rule_scores = {
        "manhole": 2.3 * compact_score + 2.1 * cross_symmetry + 2.0 * circle_quality,
        "duct": 2.4 * elongated_score + 2.0 * cross_symmetry + 1.7 * circle_quality,
        "trench": 2.3 * elongated_score + 2.1 * rectangularity + 1.2 * (1.0 - cross_symmetry),
        "handhole": 2.4 * compact_score + 2.5 * rectangularity + 1.1 * (1.0 - circle_quality),
    }
    raw_scores = {cat: rule_scores.get(cat, 0.1) for cat in categories}
    scores = _normalize_scores(raw_scores, categories)

    category = max(scores, key=scores.get)
    confidence = _confidence_from_top(scores[category])
    return category, confidence, scores


def detect_asset_category_hybrid(rgb_frames, pcd, rgb_weight=0.4, geom_weight=0.6):
    """
    Hybrid classifier that fuses RGB and geometry class scores.
    Returns: (category, confidence, fused_scores, details)
    """
    categories = get_registered_categories()
    _, _, rgb_scores = detect_asset_category_from_images(rgb_frames)
    _, _, geom_scores = detect_asset_category_from_geometry(pcd)

    raw_fused = {}
    for cat in categories:
        raw_fused[cat] = (
            float(rgb_weight) * float(rgb_scores.get(cat, 0.0))
            + float(geom_weight) * float(geom_scores.get(cat, 0.0))
        )

    fused_scores = _normalize_scores(raw_fused, categories)
    category = max(fused_scores, key=fused_scores.get)
    confidence = _confidence_from_top(fused_scores[category])
    details = {
        "rgb_scores": rgb_scores,
        "geometry_scores": geom_scores,
        "weights": {"rgb": float(rgb_weight), "geometry": float(geom_weight)},
    }
    return category, confidence, fused_scores, details
