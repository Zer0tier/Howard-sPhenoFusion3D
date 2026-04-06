import argparse
import csv
import json
import os
import sys
from typing import List

import numpy as np
import open3d as o3d


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import (
    build_fx_intrinsics,
    infer_depth_scale,
    run_reconstruction_once,
)


def parse_fx_values(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("No valid fx values provided.")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep candidate focal lengths and score reconstruction quality."
    )
    parser.add_argument("--rgb-dir", required=True, help="RGB folder path")
    parser.add_argument("--depth-dir", required=True, help="Depth folder path")
    parser.add_argument(
        "--fx-values",
        default="700,800,900,1000,1050,1108,1200",
        help="Comma-separated focal lengths to test",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of frames used per candidate (default: 5 = anchor + neighbors)",
    )
    parser.add_argument("--step", type=int, default=1, help="Frame step for loading data")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Depth scale override. If omitted, inferred from path.",
    )
    parser.add_argument("--width", type=int, default=640, help="Image width for principal point")
    parser.add_argument("--height", type=int, default=480, help="Image height for principal point")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="Voxel size")
    parser.add_argument("--icp-max-points", type=int, default=45000, help="ICP max points")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "experiments", "calibration", "outputs"),
        help="Output directory for sweep artifacts",
    )
    parser.add_argument(
        "--save-best-ply",
        action="store_true",
        help="If set, save a full reconstruction PLY for the best fx candidate.",
    )
    return parser.parse_args()


def write_csv(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fx", "fitness", "rmse", "plane_inlier_ratio", "success_rate", "point_count"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def best_candidate(rows: List[dict]):
    # Primary: plane inlier ratio, Secondary: fitness, Tertiary: lower rmse
    return max(rows, key=lambda r: (r["plane_inlier_ratio"], r["fitness"], -r["rmse"]))


def main():
    args = parse_args()
    fx_values = parse_fx_values(args.fx_values)
    os.makedirs(args.output_dir, exist_ok=True)

    depth_scale = args.depth_scale
    if depth_scale is None:
        depth_scale = infer_depth_scale(f"{args.rgb_dir} {args.depth_dir}")

    rows = []
    for fx in fx_values:
        label = f"sweep_fx{int(round(fx))}"
        ply_path = os.path.join(args.output_dir, f"{label}.ply")
        K, dist = build_fx_intrinsics(
            width=args.width,
            height=args.height,
            fx=fx,
            fy=fx,
        )
        print(f"[sweep] evaluating fx={fx:.2f} ...")
        result = run_reconstruction_once(
            label=label,
            rgb_dir=args.rgb_dir,
            depth_dir=args.depth_dir,
            output_ply_path=ply_path,
            K_override=K,
            dist_override=dist,
            step=args.step,
            max_frames=args.frames,
            depth_scale=depth_scale,
            voxel_size=args.voxel_size,
            icp_max_points=args.icp_max_points,
        )
        row = {
            "fx": float(fx),
            "fitness": float(result.mean_fitness),
            "rmse": float(result.mean_rmse),
            "plane_inlier_ratio": float(result.plane_inlier_ratio),
            "success_rate": float(result.success_rate),
            "point_count": int(result.point_count),
        }
        rows.append(row)
        print(
            "[sweep] "
            f"fx={row['fx']:.2f}, fitness={row['fitness']:.4f}, rmse={row['rmse']:.5f}, "
            f"plane={100.0 * row['plane_inlier_ratio']:.2f}%"
        )

    rows.sort(key=lambda r: r["fx"])
    csv_path = os.path.join(args.output_dir, "intrinsics_sweep.csv")
    write_csv(csv_path, rows)

    winner = best_candidate(rows)
    winner_fx = float(winner["fx"])
    K_best, dist_best = build_fx_intrinsics(
        width=args.width,
        height=args.height,
        fx=winner_fx,
        fy=winner_fx,
    )
    candidate = {
        "K": K_best.tolist(),
        "dist": dist_best,
        "width": args.width,
        "height": args.height,
        "selected_fx": winner_fx,
        "selection_rule": "max(plane_inlier_ratio, fitness, -rmse)",
        "sweep_csv": csv_path,
    }
    candidate_path = os.path.join(args.output_dir, "candidate_intrinsics.json")
    with open(candidate_path, "w", encoding="utf-8") as f:
        json.dump(candidate, f, indent=2)

    print(f"[sweep] wrote: {csv_path}")
    print(f"[sweep] wrote: {candidate_path}")
    print(
        f"[sweep] winner fx={winner_fx:.2f} "
        f"(fitness={winner['fitness']:.4f}, rmse={winner['rmse']:.5f}, "
        f"plane={100.0 * winner['plane_inlier_ratio']:.2f}%)"
    )

    if args.save_best_ply:
        best_label = f"best_fx{int(round(winner_fx))}"
        best_ply = os.path.join(args.output_dir, f"{best_label}_full.ply")
        run_reconstruction_once(
            label=best_label,
            rgb_dir=args.rgb_dir,
            depth_dir=args.depth_dir,
            output_ply_path=best_ply,
            K_override=K_best,
            dist_override=dist_best,
            step=args.step,
            max_frames=None,
            depth_scale=depth_scale,
            voxel_size=args.voxel_size,
            icp_max_points=args.icp_max_points,
        )
        print(f"[sweep] wrote: {best_ply}")


if __name__ == "__main__":
    main()

