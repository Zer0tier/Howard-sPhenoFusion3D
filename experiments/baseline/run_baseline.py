import argparse
import os
import sys
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import build_fx_intrinsics, dump_json, infer_depth_scale, run_reconstruction_once


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline reconstruction with placeholder intrinsics (T-01)."
    )
    parser.add_argument("--rgb-dir", required=True, help="RGB images directory")
    parser.add_argument("--depth-dir", required=True, help="Depth images directory")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "baseline"),
        help="Output directory for baseline artifacts",
    )
    parser.add_argument(
        "--fx-values",
        nargs="+",
        type=float,
        default=[1108.0, 900.0],
        help="Focal lengths to test (default: 1108 900)",
    )
    parser.add_argument("--width", type=int, default=640, help="Image width for K matrix center")
    parser.add_argument("--height", type=int, default=480, help="Image height for K matrix center")
    parser.add_argument("--step", type=int, default=1, help="Frame step for loading pairs")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Depth scale override (if omitted, inferred from dataset path)",
    )
    parser.add_argument("--voxel-size", type=float, default=0.005, help="Voxel size for reconstruction")
    parser.add_argument("--icp-max-points", type=int, default=45000, help="Point cap used during ICP")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    depth_scale = args.depth_scale
    if depth_scale is None:
        depth_scale = infer_depth_scale(f"{args.rgb_dir} {args.depth_dir}")

    run_results = []
    for fx in args.fx_values:
        label = f"baseline_fx{int(round(fx))}"
        ply_path = os.path.join(args.output_dir, f"{label}.ply")
        K, dist = build_fx_intrinsics(
            width=args.width,
            height=args.height,
            fx=fx,
            fy=fx,
        )
        print(f"[baseline] Running reconstruction for {label}...")
        result = run_reconstruction_once(
            label=label,
            rgb_dir=args.rgb_dir,
            depth_dir=args.depth_dir,
            output_ply_path=ply_path,
            K_override=K,
            dist_override=dist,
            step=args.step,
            depth_scale=depth_scale,
            voxel_size=args.voxel_size,
            icp_max_points=args.icp_max_points,
        )
        run_results.append(result.to_dict())
        print(
            f"[baseline] {label}: "
            f"fitness={result.mean_fitness:.4f}, rmse={result.mean_rmse:.5f}, "
            f"success_rate={result.success_rate:.2%}, points={result.point_count}"
        )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset": {
            "rgb_dir": args.rgb_dir,
            "depth_dir": args.depth_dir,
            "step": args.step,
            "depth_scale": depth_scale,
            "width": args.width,
            "height": args.height,
        },
        "runs": run_results,
        "notes": [
            "Inspect generated PLYs in Open3D/MeshLab/CloudCompare for tilt/folding/noise.",
            "This file is the Phase 0 baseline anchor for Sprint experiments.",
        ],
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    dump_json(metrics_path, payload)
    print(f"[baseline] Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()

