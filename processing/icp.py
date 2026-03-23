import open3d as o3d
import numpy as np


def color_icp(source, target, max_iter=50, voxel_size=0.005):
    """
    Colour-assisted ICP registration between two point clouds.
    Both source and target must have colour data.

    Returns: (result, transformation, fitness, inlier_rmse)
    Falls back to point_to_plane_icp if colour ICP fails.
    """
    if source.is_empty() or target.is_empty():
        print('[icp] WARNING: Empty point cloud passed to color_icp, skipping.')
        identity = np.eye(4)
        return None, identity, 0.0, 0.0

    radius = voxel_size * 2

    try:
        result = o3d.pipelines.registration.registration_colored_icp(
            source, target,
            radius,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter
            )
        )
        fitness = result.fitness
        inlier_rmse = result.inlier_rmse

        # If colour ICP gives no result, fall back
        if fitness == 0.0:
            print('[icp] colour_icp fitness=0, falling back to point-to-plane ICP')
            return point_to_plane_icp(source, target, max_iter, voxel_size)

        return result, result.transformation, fitness, inlier_rmse

    except Exception as e:
        print(f'[icp] colour_icp failed ({e}), falling back to point-to-plane ICP')
        return point_to_plane_icp(source, target, max_iter, voxel_size)


def point_to_plane_icp(source, target, max_iter=50, voxel_size=0.005):
    """
    Point-to-plane ICP fallback. Estimates normals if not present.

    Returns: (result, transformation, fitness, inlier_rmse)
    """
    if source.is_empty() or target.is_empty():
        identity = np.eye(4)
        return None, identity, 0.0, 0.0

    radius = voxel_size * 2

    # Estimate normals if missing
    for pcd in [source, target]:
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius, max_nn=30
                )
            )

    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=radius,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter
        )
    )

    return result, result.transformation, result.fitness, result.inlier_rmse