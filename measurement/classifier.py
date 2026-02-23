import base64
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


def _extract_primary_contour(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    image_area = float(h * w)

    edge = cv2.Canny(blur, 60, 160)
    edge = cv2.dilate(edge, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=2)

    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_otsu = cv2.morphologyEx(th_otsu, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=2)
    th_inv = cv2.bitwise_not(th_otsu)

    candidates = []
    for mask in (edge, th_otsu, th_inv):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(contours)

    best = None
    best_score = -1.0
    for cnt in candidates:
        area = float(cv2.contourArea(cnt))
        if area < 0.005 * image_area or area > 0.95 * image_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        bbox_area = float(max(1, bw * bh))
        rectangularity = float(np.clip(area / bbox_area, 0.0, 1.0))
        area_ratio = float(area / image_area)
        # Prefer substantial, coherent foreground contours.
        score = 3.2 * area_ratio + 1.2 * rectangularity
        if score > best_score:
            best_score = score
            best = (cnt, x, y, bw, bh, area_ratio, rectangularity, score)

    if best is None:
        return None
    return best


def detect_asset_region_from_images(rgb_frames, include_preview=True):
    """
    Finds a dominant object contour from RGB frames and returns:
    - detection dictionary (bbox + polygon in pixel and normalized coords)
    - optional annotated preview image as base64-encoded JPEG.
    """
    if not rgb_frames:
        return None, None

    best = None
    best_frame_idx = -1
    for idx, rgb in enumerate(rgb_frames):
        contour_info = _extract_primary_contour(rgb)
        if contour_info is None:
            continue
        if best is None or contour_info[-1] > best[-1]:
            best = contour_info
            best_frame_idx = idx

    if best is None:
        return None, None

    cnt, x, y, bw, bh, area_ratio, rectangularity, score = best
    frame = rgb_frames[best_frame_idx]
    h, w = frame.shape[:2]
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
    polygon = [[int(p[0][0]), int(p[0][1])] for p in approx]

    detection = {
        "frame_index": int(best_frame_idx),
        "score": float(score),
        "area_ratio": float(area_ratio),
        "rectangularity": float(rectangularity),
        "bbox": {
            "x": int(x),
            "y": int(y),
            "w": int(bw),
            "h": int(bh),
        },
        "bbox_norm": {
            "x": float(x / max(1.0, w)),
            "y": float(y / max(1.0, h)),
            "w": float(bw / max(1.0, w)),
            "h": float(bh / max(1.0, h)),
        },
        "polygon": polygon,
        "polygon_norm": [
            [float(px / max(1.0, w)), float(py / max(1.0, h))]
            for px, py in polygon
        ],
        "image_size": {"width": int(w), "height": int(h)},
    }

    preview_b64 = None
    if include_preview:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.drawContours(bgr, [approx], -1, (50, 220, 50), 3)
        cv2.rectangle(bgr, (x, y), (x + bw, y + bh), (60, 60, 240), 2)
        cv2.putText(
            bgr,
            f"score={score:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            preview_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    return detection, preview_b64


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
