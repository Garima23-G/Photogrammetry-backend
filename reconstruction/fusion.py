import numpy as np
import open3d as o3d
import cv2

def estimate_bounding_box(depth_frames, intrinsic_matrix, poses, sample_rate=10, step=20):
    """
    Estimates the 3D bounding box of the scene from depth frames.

    Args:
        depth_frames: List of 16-bit depth maps (in mm).
        intrinsic_matrix: (3, 3) intrinsic matrix.
        poses: List of (4, 4) camera-to-world matrices.
        sample_rate: Rate at which to sample frames.
        step: Pixel step for subsampling depth maps.

    Returns:
        bbox_min: (3,) numpy array.
        bbox_max: (3,) numpy array.
    """
    all_points = []

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    for i in range(0, len(depth_frames), max(1, len(depth_frames) // sample_rate)):
        depth = depth_frames[i]
        pose = poses[i]

        # Subsample depth map for speed
        d_sub = depth[::step, ::step]
        rows, cols = d_sub.shape
        v, u = np.indices((rows, cols))
        u = (u * step).flatten()
        v = (v * step).flatten()
        z = d_sub.flatten() / 1000.0  # mm to meters

        mask = (z > 0.1) & (z < 10.0) # valid depth range
        z = z[mask]
        u = u[mask]
        v = v[mask]

        if len(z) == 0:
            continue

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pts_cam = np.vstack((x, y, z, np.ones_like(z)))
        pts_world = pose @ pts_cam
        all_points.append(pts_world[:3, :].T)

    if not all_points:
        return None, None

    all_points = np.concatenate(all_points, axis=0)
    bbox_min = np.min(all_points, axis=0)
    bbox_max = np.max(all_points, axis=0)

    return bbox_min, bbox_max

def invert_poses(poses):
    """
    Inverts camera-to-world poses to world-to-camera extrinsic matrices.
    """
    return [np.linalg.inv(p) for p in poses]


def _uniform_frame_indices(frame_count, max_frames):
    if frame_count <= max_frames:
        return list(range(frame_count))
    return np.linspace(0, frame_count - 1, max_frames, dtype=int).tolist()

def reconstruct_scene(rgb_frames, depth_frames, intrinsic_matrix, poses, category_tag,
                      voxel_length=0.004, sdf_trunc=0.012, max_frames=30, max_image_dim=640):
    """
    Performs RGBD volumetric fusion using ScalableTSDFVolume.

    Args:
        rgb_frames: List of (H, W, 3) uint8 numpy arrays.
        depth_frames: List of (H, W) uint16 numpy arrays (in mm).
        intrinsic_matrix: (3, 3) intrinsic matrix.
        poses: List of (4, 4) camera-to-world matrices.
        category_tag: String tag for the asset category.
        voxel_length: Voxel size in meters.
        sdf_trunc: Truncation distance in meters.

    Returns:
        A dictionary containing:
            'points': (N, 3) float32 numpy array.
            'pcd': o3d.geometry.PointCloud object.
            'mesh': o3d.geometry.TriangleMesh object.
            'category_tag': The input category tag.
    """
    if not rgb_frames or not depth_frames or not poses:
        raise ValueError("No frames/poses available for reconstruction.")

    # Keep fusion bounded in memory for mobile uploads:
    # - uniform frame subsampling when too many frames are sent
    # - optional image downscale before TSDF integration
    frame_indices = _uniform_frame_indices(len(rgb_frames), max_frames)
    rgb_frames = [rgb_frames[i] for i in frame_indices]
    depth_frames = [depth_frames[i] for i in frame_indices]
    poses = [poses[i] for i in frame_indices]

    # 1. Estimate bounding box (used later for cropping/padding requirements)
    bbox_min, bbox_max = estimate_bounding_box(depth_frames, intrinsic_matrix, poses)

    # 2. Invert poses (C2W to W2C)
    extrinsics = invert_poses(poses)

    # 3. Initialize ScalableTSDFVolume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # 4. Create Open3D intrinsic object
    h0, w0 = depth_frames[0].shape
    scale = 1.0
    intrinsic_matrix = intrinsic_matrix.copy()
    if max(h0, w0) > max_image_dim:
        scale = max_image_dim / float(max(h0, w0))
    h = int(round(h0 * scale))
    w = int(round(w0 * scale))

    if scale != 1.0:
        intrinsic_matrix[0, 0] *= scale  # fx
        intrinsic_matrix[1, 1] *= scale  # fy
        intrinsic_matrix[0, 2] *= scale  # cx
        intrinsic_matrix[1, 2] *= scale  # cy

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        w, h,
        intrinsic_matrix[0, 0], intrinsic_matrix[1, 1],
        intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    )

    # 5. Integrate frames
    for i in range(len(rgb_frames)):
        rgb_np = rgb_frames[i]
        depth_np = depth_frames[i]

        if scale != 1.0:
            rgb_np = cv2.resize(rgb_np, (w, h), interpolation=cv2.INTER_AREA)
            depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_NEAREST)

        rgb = o3d.geometry.Image(rgb_np)
        depth = o3d.geometry.Image(depth_np)

        # Note: depth_scale=1000.0 because depth is in mm
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth,
            depth_scale=1000.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False
        )

        try:
            volume.integrate(rgbd, intrinsic, extrinsics[i])
        except MemoryError as e:
            raise MemoryError(
                f"TSDF integrate failed at frame {i} "
                f"(frames={len(rgb_frames)}, size={w}x{h}, voxel={voxel_length})."
            ) from e

    # 6. Extraction
    pcd = volume.extract_point_cloud()
    mesh = volume.extract_triangle_mesh()

    # 7. Post-processing
    mesh.compute_vertex_normals()

    # Apply bounding box padding/cropping to satisfy "Volume extent" requirement
    # "Volume extent padded ±0.5 m around point cloud bounds."
    if bbox_min is not None and bbox_max is not None:
        padded_min = bbox_min - 0.5
        padded_max = bbox_max + 0.5

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=padded_min, max_bound=padded_max)
        pcd = pcd.crop(bbox)
        mesh = mesh.crop(bbox)

    # 8. Format output
    points_np = np.asarray(pcd.points, dtype=np.float32)

    return {
        'points': points_np,
        'pcd': pcd,
        'mesh': mesh,
        'category_tag': category_tag
    }
