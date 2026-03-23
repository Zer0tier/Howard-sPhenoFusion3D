import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processing.rgbd import rgbd2pcd


def make_synthetic_pair(h=64, w=64):
    """Create a simple synthetic colour + depth pair for testing."""
    color = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    # Flat depth plane at 1 metre (1000mm in uint16)
    depth = np.full((h, w), 1000, dtype=np.uint16)
    K = [[525.0, 0, 32.0], [0, 525.0, 32.0], [0, 0, 1.0]]
    return color, depth, K


def test_rgbd2pcd_returns_pointcloud():
    color, depth, K = make_synthetic_pair()
    pcd = rgbd2pcd(color, depth, K, depth_scale=1000.0)
    import open3d as o3d
    assert isinstance(pcd, o3d.geometry.PointCloud)


def test_rgbd2pcd_not_empty():
    color, depth, K = make_synthetic_pair()
    pcd = rgbd2pcd(color, depth, K, depth_scale=1000.0)
    assert not pcd.is_empty(), 'Point cloud should not be empty for valid input'


def test_rgbd2pcd_has_colors():
    color, depth, K = make_synthetic_pair()
    pcd = rgbd2pcd(color, depth, K, depth_scale=1000.0)
    assert pcd.has_colors(), 'Point cloud should carry colour data'


def test_rgbd2pcd_zero_depth_gives_empty():
    color = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    depth = np.zeros((64, 64), dtype=np.uint16)
    K = [[525.0, 0, 32.0], [0, 525.0, 32.0], [0, 0, 1.0]]
    pcd = rgbd2pcd(color, depth, K, depth_scale=1000.0)
    assert pcd.is_empty(), 'Zero depth should produce an empty point cloud'