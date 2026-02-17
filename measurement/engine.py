import numpy as np
import open3d as o3d
from measurement.utils import fit_circle_ransac, estimate_axis_pca

def measure_trench(points):
    """
    Trench algorithms:
    - Depth: z_max - z_min
    - Width: multi-slice X-projection averaging
    """
    if len(points) == 0:
        return {"depth_m": 0.0, "width_m": 0.0}

    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    depth = z_max - z_min

    # Width calculation: slice horizontally at multiple depths
    z_levels = np.linspace(z_min + 0.1 * depth, z_max - 0.1 * depth, 5)
    widths = []
    for z in z_levels:
        mask = np.abs(points[:, 2] - z) < 0.02
        slice_pts = points[mask]
        if len(slice_pts) > 10:
            # Multi-slice averaging correctly handles tapered walls
            w = np.max(slice_pts[:, 0]) - np.min(slice_pts[:, 0])
            widths.append(w)

    width = np.mean(widths) if widths else 0.0
    return {"depth_m": float(depth), "width_m": float(width)}

def measure_manhole(points):
    """
    Manhole algorithms:
    - Diameter: RANSAC circle fit at top slice
    - Depth: z_max - z_min
    """
    if len(points) == 0:
        return {"diameter_m": 0.0, "depth_m": 0.0}

    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    depth = z_max - z_min

    # Extract vertical slice near top opening (top 5cm)
    top_mask = points[:, 2] > (z_max - 0.05)
    top_pts = points[top_mask]

    if len(top_pts) >= 3:
        # Fit circle — Least Squares or RANSAC
        _, _, r = fit_circle_ransac(top_pts[:, :2])
        diameter = 2 * r
    else:
        diameter = 0.0

    return {"diameter_m": float(diameter), "depth_m": float(depth)}

def measure_duct(points):
    """
    Duct algorithms:
    - Diameter: Multi-section robust arc fitting -> median radius
    - Length: Project points onto fitted cylinder axis
    """
    if len(points) == 0:
        return {"diameter_m": 0.0, "length_m": 0.0}

    # Estimate cylinder axis
    axis = estimate_axis_pca(points)
    projections = points @ axis
    length = np.max(projections) - np.min(projections)

    # Multiple cross-sections along axis -> median radius
    p_min, p_max = np.min(projections), np.max(projections)
    sections = np.linspace(p_min + 0.1*length, p_max - 0.1*length, 3)
    radii = []

    # Find orthogonal vectors for 2D projection
    if abs(axis[0]) < 0.9:
        v1 = np.cross(axis, [1, 0, 0])
    else:
        v1 = np.cross(axis, [0, 1, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(axis, v1)

    for s in sections:
        mask = np.abs(projections - s) < 0.05
        sec_pts = points[mask]
        if len(sec_pts) >= 3:
            pts_2d = np.stack([sec_pts @ v1, sec_pts @ v2], axis=1)
            _, _, r = fit_circle_ransac(pts_2d)
            radii.append(r)

    diameter = 2 * np.median(radii) if radii else 0.0
    return {"diameter_m": float(diameter), "length_m": float(length)}

def measure_handhole(pcd):
    """
    Handhole algorithms:
    - Oriented Bounding Box (OBB)
    """
    if pcd is None or pcd.is_empty():
        return {"L_m": 0.0, "W_m": 0.0, "H_m": 0.0}

    # Pre-process: remove outliers for robust OBB
    pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # OBB finds minimal enclosing box at any angle
    # Using get_minimal_oriented_bounding_box for higher accuracy
    obb = pcd_filtered.get_minimal_oriented_bounding_box()
    # Extent sorted: longest -> shortest
    extent = sorted(obb.extent, reverse=True)

    return {
        "L_m": float(extent[0]),
        "W_m": float(extent[1]),
        "H_m": float(extent[2])
    }

def measure_asset(pcd, category_tag):
    """
    Main router for Stage 05 Measurement Engine.

    Args:
        pcd: o3d.geometry.PointCloud object.
        category_tag: String identifying the asset category.

    Returns:
        Structured dictionary following Stage 05 JSON response schema.
    """
    points = np.asarray(pcd.points)
    point_count = len(points)

    # Initialize default response structure
    res = {
        "category": category_tag,
        "depth_m": None,
        "width_m": None,
        "diameter_m": None,
        "length_m": None,
        "L_m": None,
        "W_m": None,
        "H_m": None,
        "confidence": "high" if point_count > 50000 else "medium" if point_count > 5000 else "low",
        "point_count": int(point_count)
    }

    cat = category_tag.lower()
    if cat == "trench":
        m = measure_trench(points)
        res.update(m)
    elif cat == "manhole":
        m = measure_manhole(points)
        res.update(m)
    elif cat == "duct":
        m = measure_duct(points)
        res.update(m)
    elif cat == "handhole":
        m = measure_handhole(pcd)
        res.update(m)

    return res
