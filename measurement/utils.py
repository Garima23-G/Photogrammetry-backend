import numpy as np

def fit_circle_3_points(p1, p2, p3):
    """Fits a circle through three 2D points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-9:
        return None, None, None # Collinear

    a = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    b = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    r = np.sqrt((x1 - a)**2 + (y1 - b)**2)

    return a, b, r

def fit_circle_ransac(points, iterations=200, threshold=0.01):
    """Fits a circle to 2D points using RANSAC."""
    best_circle = (0.0, 0.0, 0.0)
    max_inliers = -1

    if len(points) < 3:
        return 0.0, 0.0, 0.0

    for _ in range(iterations):
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]

        a, b, r = fit_circle_3_points(p1, p2, p3)
        if a is None or r > 5.0: # Sanity check for radius
            continue

        distances = np.abs(np.sqrt((points[:, 0] - a)**2 + (points[:, 1] - b)**2) - r)
        inliers = np.sum(distances < threshold)

        if inliers > max_inliers:
            max_inliers = inliers
            best_circle = (a, b, r)

    return best_circle

def estimate_axis_pca(points):
    """Estimates the principal axis of a 3D point cloud using PCA."""
    if len(points) < 3:
        return np.array([0, 0, 1], dtype=np.float64)

    mean = np.mean(points, axis=0)
    centered = points - mean
    # We use SVD for robustness
    _, _, vh = np.linalg.svd(centered)
    # The first row of V^H (first column of V) is the principal component
    return vh[0]
