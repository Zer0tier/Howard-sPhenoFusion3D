import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from file_io.loader import get_default_intrinsics, load_image_pairs, load_intrinsics
from file_io.exporter import save_ply
from processing.reconstructor import Reconstructor


@dataclass
class RunResult:
    label: str
    intrinsics_source: str
    output_ply: str
    mean_fitness: float
    mean_rmse: float
    success_frames: int
    failed_frames: int
    success_rate: float
    point_count: int
    plane_inlier_ratio: float
    frame_metrics: List[Dict]

    def to_dict(self):
        return {
            "label": self.label,
            "intrinsics_source": self.intrinsics_source,
            "output_ply": self.output_ply,
            "mean_fitness": self.mean_fitness,
            "mean_rmse": self.mean_rmse,
            "success_frames": self.success_frames,
            "failed_frames": self.failed_frames,
            "success_rate": self.success_rate,
            "point_count": self.point_count,
            "plane_inlier_ratio": self.plane_inlier_ratio,
            "frame_metrics": self.frame_metrics,
        }


def infer_depth_scale(path_hint: str, default: float = 1000.0) -> float:
    text = (path_hint or "").lower()
    if any(token in text for token in ("icl", "nuim", "tum", "freiburg")):
        return 1.0
    return default


def build_fx_intrinsics(width: int, height: int, fx: float, fy: Optional[float] = None):
    fy = fx if fy is None else fy
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = [0.0, 0.0, 0.0, 0.0, 0.0]
    return K, dist


def summarize_metrics(success_list: List[Dict], fail_list: List[Dict], point_count: int):
    fitness_vals = [float(item.get("fitness", 0.0)) for item in success_list if item.get("frame", -1) != 0]
    rmse_vals = [float(item.get("rmse", 0.0)) for item in success_list if item.get("frame", -1) != 0]

    mean_fitness = float(np.mean(fitness_vals)) if fitness_vals else 0.0
    mean_rmse = float(np.mean(rmse_vals)) if rmse_vals else 0.0
    total = len(success_list) + len(fail_list)
    success_rate = (len(success_list) / total) if total > 0 else 0.0

    frame_metrics = []
    for item in success_list:
        frame_metrics.append(
            {
                "frame": int(item.get("frame", -1)),
                "status": "OK",
                "fitness": float(item.get("fitness", 0.0)),
                "rmse": float(item.get("rmse", 0.0)),
            }
        )
    for item in fail_list:
        frame_metrics.append(
            {
                "frame": int(item.get("frame", -1)),
                "status": "FAILED",
                "fitness": 0.0,
                "rmse": 0.0,
                "note": str(item.get("reason", "")),
            }
        )
    frame_metrics.sort(key=lambda row: row["frame"])

    return {
        "mean_fitness": mean_fitness,
        "mean_rmse": mean_rmse,
        "success_frames": len(success_list),
        "failed_frames": len(fail_list),
        "success_rate": success_rate,
        "point_count": int(point_count),
        "frame_metrics": frame_metrics,
    }


def plane_inlier_ratio(pcd: o3d.geometry.PointCloud, distance_threshold: float = 0.003, ransac_n: int = 3, num_iters: int = 400):
    if pcd is None or pcd.is_empty():
        return 0.0
    if len(pcd.points) < 100:
        return 0.0
    try:
        _, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iters,
        )
        return len(inliers) / max(1, len(pcd.points))
    except Exception:
        return 0.0


def load_intrinsics_or_default(intrinsics_path: Optional[str], width: int = 640, height: int = 480):
    loaded = load_intrinsics(intrinsics_path) if intrinsics_path else None
    if loaded:
        K, dist, _, _ = loaded
        return K, dist, "file"
    K, dist = get_default_intrinsics(width=width, height=height)
    return K, dist, "default"


def run_reconstruction_once(
    label: str,
    rgb_dir: str,
    depth_dir: str,
    output_ply_path: str,
    intrinsics_path: Optional[str] = None,
    K_override=None,
    dist_override=None,
    step: int = 1,
    max_frames: Optional[int] = None,
    depth_scale: Optional[float] = None,
    voxel_size: float = 0.005,
    icp_max_points: int = 45000,
):
    pairs = load_image_pairs(rgb_dir, depth_dir, step=step)
    if max_frames is not None and max_frames > 0:
        pairs = pairs[: int(max_frames)]
    if not pairs:
        raise RuntimeError("No image pairs available after filtering.")

    K = K_override
    dist = dist_override
    source = "override"
    if K is None:
        K, dist, source = load_intrinsics_or_default(intrinsics_path)

    if depth_scale is None:
        depth_scale = infer_depth_scale(f"{rgb_dir} {depth_dir}")

    recon = Reconstructor(
        pairs=pairs,
        K=K,
        dist=dist,
        depth_scale=depth_scale,
        voxel_size=voxel_size,
        icp_max_points=icp_max_points,
        save_every_n_frames=0,
    )
    merged, succeed, fail = recon.run()
    save_ok = save_ply(merged, output_ply_path)
    if not save_ok:
        raise RuntimeError(f"Failed to save PLY to: {output_ply_path}")

    summary = summarize_metrics(succeed, fail, len(merged.points))
    inlier_ratio = plane_inlier_ratio(merged)
    return RunResult(
        label=label,
        intrinsics_source=source,
        output_ply=output_ply_path,
        mean_fitness=summary["mean_fitness"],
        mean_rmse=summary["mean_rmse"],
        success_frames=summary["success_frames"],
        failed_frames=summary["failed_frames"],
        success_rate=summary["success_rate"],
        point_count=summary["point_count"],
        plane_inlier_ratio=inlier_ratio,
        frame_metrics=summary["frame_metrics"],
    )


def dump_json(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

