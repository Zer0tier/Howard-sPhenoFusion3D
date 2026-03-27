import numpy as np
import cv2
import open3d as o3d

_INTRINSIC_CACHE = {}
_UNDISTORT_MAP_CACHE = {}


def _get_intrinsic(width, height, fx, fy, cx, cy):
    key = (int(width), int(height), float(fx), float(fy), float(cx), float(cy))
    intrinsic = _INTRINSIC_CACHE.get(key)
    if intrinsic is None:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy
        )
        _INTRINSIC_CACHE[key] = intrinsic
    return intrinsic


def _get_undistort_maps(width, height, K, dist_arr):
    key = (
        int(width), int(height),
        tuple(np.asarray(K, dtype=np.float64).reshape(-1).tolist()),
        tuple(np.asarray(dist_arr, dtype=np.float64).reshape(-1).tolist()),
    )
    cached = _UNDISTORT_MAP_CACHE.get(key)
    if cached is None:
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist_arr, None, K, (width, height), cv2.CV_32FC1
        )
        cached = (map1, map2)
        _UNDISTORT_MAP_CACHE[key] = cached
    return cached


def rgbd2pcd(color_img, depth_img, K, dist=None, bbox=None, depth_scale=1000.0, depth_trunc=3.0):
    """
    Convert an RGB image + depth image into an Open3D coloured PointCloud.

    Args:
        color_img   : np.ndarray (H, W, 3) in RGB order
        depth_img   : np.ndarray (H, W) uint16, depth in mm (divide by depth_scale -> metres)
        K           : 3x3 intrinsic matrix (np.ndarray or nested list)
        dist        : distortion coefficients (list of 5), or None
        bbox        : optional [x1, y1, x2, y2] crop on the colour image before projection
        depth_scale : divisor to convert raw depth to metres (1000 for RealSense mm, 1 for ICL-NUIM)
        depth_trunc : discard depth beyond this many metres (default 3.0 m)

    Returns:
        o3d.geometry.PointCloud with colour
    """
    K = np.array(K, dtype=np.float64)
    h, w = color_img.shape[:2]

    # Undistort if distortion coefficients provided and non-zero
    if dist is not None and any(d != 0.0 for d in dist):
        dist_arr = np.array(dist, dtype=np.float64)
        map1, map2 = _get_undistort_maps(w, h, K, dist_arr)
        color_img = cv2.remap(color_img, map1, map2, interpolation=cv2.INTER_LINEAR)
        depth_img = cv2.remap(depth_img, map1, map2, interpolation=cv2.INTER_NEAREST)

    # Optional bbox crop (applied before projection)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color_img = color_img[y1:y2, x1:x2]
        depth_img = depth_img[y1:y2, x1:x2]
        # Adjust principal point for the crop
        K_crop = K.copy()
        K_crop[0, 2] -= x1
        K_crop[1, 2] -= y1
        K = K_crop
        h, w = color_img.shape[:2]

    # Ensure colour is uint8 RGB
    if color_img.dtype != np.uint8:
        color_img = (color_img * 255).astype(np.uint8)

    # Ensure depth is uint16
    if depth_img.dtype != np.uint16:
        depth_img = depth_img.astype(np.uint16)

    # Create Open3D images
    o3d_color = o3d.geometry.Image(color_img)
    o3d_depth = o3d.geometry.Image(depth_img)

    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    # Build camera intrinsics from K matrix
    fx = float(K[0, 0])
    fy = float(abs(K[1, 1]))   # abs handles ICL-NUIM negative fy convention
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    intrinsic = _get_intrinsic(w, h, fx, fy, cx, cy)

    # Project to point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    return pcd