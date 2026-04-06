import argparse
import csv
import json
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import run_reconstruction_once


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple intrinsics candidates on a fixed reconstruction slice."
    )
    parser.add_argument("--rgb-dir", required=True, help="RGB image directory")
    parser.add_argument("--depth-dir", required=True, help="Depth image directory")
    parser.add_argument(
        "intrinsics_paths",
        nargs="+",
        help="List of intrinsics JSON files to compare",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(ROOT, "evaluation", "comparison_results.csv"),
        help="CSV path for aggregate results",
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join(ROOT, "evaluation", "comparison_results.json"),
        help="JSON path for detailed results",
    )
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to evaluate")
    parser.add_argument("--step", type=int, default=1, help="Frame step")
    parser.add_argument("--depth-scale", type=float, default=None, help="Depth scale override")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="Voxel size")
    parser.add_argument("--icp-max-points", type=int, default=45000, help="ICP point cap")
    parser.add_argument(
        "--output-ply-dir",
        default=os.path.join(ROOT, "evaluation", "ply_outputs"),
        help="Directory for generated PLY files",
    )
    return parser.parse_args()


def path_label(path: str):
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    return stem


def write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = [
        "label",
        "intrinsics_path",
        "mean_fitness",
        "mean_rmse",
        "success_rate",
        "success_frames",
        "failed_frames",
        "point_count",
        "plane_inlier_ratio",
        "output_ply",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_markdown_table(rows: List[Dict]):
    print("")
    print("| Candidate | Mean Fitness | Mean RMSE | Success % | Points | Plane Inlier % |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['label']} | "
            f"{row['mean_fitness']:.4f} | "
            f"{row['mean_rmse']:.5f} | "
            f"{100.0 * row['success_rate']:.2f}% | "
            f"{int(row['point_count'])} | "
            f"{100.0 * row['plane_inlier_ratio']:.2f}% |"
        )
    print("")


def main():
    args = parse_args()
    os.makedirs(args.output_ply_dir, exist_ok=True)

    rows = []
    details = []
    for intrinsics_path in args.intrinsics_paths:
        label = path_label(intrinsics_path)
        output_ply = os.path.join(args.output_ply_dir, f"{label}.ply")
        print(f"[compare] Running {label} using {intrinsics_path}")
        run = run_reconstruction_once(
            label=label,
            rgb_dir=args.rgb_dir,
            depth_dir=args.depth_dir,
            output_ply_path=output_ply,
            intrinsics_path=intrinsics_path,
            step=args.step,
            max_frames=args.frames,
            depth_scale=args.depth_scale,
            voxel_size=args.voxel_size,
            icp_max_points=args.icp_max_points,
        )
        row = {
            "label": label,
            "intrinsics_path": intrinsics_path,
            "mean_fitness": float(run.mean_fitness),
            "mean_rmse": float(run.mean_rmse),
            "success_rate": float(run.success_rate),
            "success_frames": int(run.success_frames),
            "failed_frames": int(run.failed_frames),
            "point_count": int(run.point_count),
            "plane_inlier_ratio": float(run.plane_inlier_ratio),
            "output_ply": output_ply,
        }
        rows.append(row)
        details.append(run.to_dict())

    rows.sort(key=lambda item: (item["mean_fitness"], item["plane_inlier_ratio"]), reverse=True)
    write_csv(args.output_csv, rows)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "rgb_dir": args.rgb_dir,
                    "depth_dir": args.depth_dir,
                    "frames": args.frames,
                    "step": args.step,
                    "depth_scale": args.depth_scale,
                    "voxel_size": args.voxel_size,
                    "icp_max_points": args.icp_max_points,
                },
                "results": rows,
                "details": details,
            },
            f,
            indent=2,
        )

    print_markdown_table(rows)
    print(f"[compare] CSV:  {args.output_csv}")
    print(f"[compare] JSON: {args.output_json}")


if __name__ == "__main__":
    main()

