import os
import sys
import numpy as np
import pytest
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processing.icp import color_icp, point_to_plane_icp


def make_sphere_pcd(n=500, noise=0.001):
    """Synthetic coloured point cloud on a sphere surface."""
    phi = np.random.uniform(0, np.pi, n)
    theta = np.random.uniform(0, 2 * np.pi, n)
    x = np.sin(phi) * np.cos(theta) + np.random.randn(n) * noise
    y = np.sin(phi) * np.sin(theta) + np.random.randn(n) * noise
    z = np.cos(phi) + np.random.randn(n) * noise
    pts = np.stack([x, y, z], axis=1)
    colors = np.random.uniform(0, 1, (n, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_offset_pair(offset=0.01):
    """Source and target are the same cloud with a small translation."""
    source = make_sphere_pcd(n=500)
    target = o3d.geometry.PointCloud(source)
    # Apply a tiny known translation
    T = np.eye(4)
    T[0, 3] = offset
    target.transform(T)
    return source, target


def test_color_icp_returns_tuple():
    source, target = make_offset_pair()
    result, transform, fitness, rmse = color_icp(source, target)
    assert transform.shape == (4, 4)
    assert isinstance(fitness, float)
    assert isinstance(rmse, float)


def test_color_icp_fitness_positive():
    source, target = make_offset_pair(offset=0.005)
    _, _, fitness, _ = color_icp(source, target, voxel_size=0.01)
    assert fitness >= 0.0, 'Fitness must be non-negative'


def test_point_to_plane_icp_runs():
    source, target = make_offset_pair()
    result, transform, fitness, rmse = point_to_plane_icp(source, target)
    assert transform.shape == (4, 4)


def test_color_icp_empty_input():
    empty = o3d.geometry.PointCloud()
    source = make_sphere_pcd()
    _, transform, fitness, rmse = color_icp(empty, source)
    assert fitness == 0.0
    assert np.allclose(transform, np.eye(4))