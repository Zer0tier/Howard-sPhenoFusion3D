import argparse

import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="View one PLY in Open3D.")
    parser.add_argument("--ply", required=True, help="PLY path")
    parser.add_argument("--name", default="point_cloud", help="Window title")
    parser.add_argument("--voxel", type=float, default=0.0, help="Optional voxel downsample before viewing")
    parser.add_argument("--no-view", action="store_true", help="Only print stats, do not open viewer")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply)
    print(f"{args.name} points (raw): {len(pcd.points)}")
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
        print(f"{args.name} points (downsampled @ {args.voxel}): {len(pcd.points)}")

    if not args.no_view:
        o3d.visualization.draw_geometries([pcd], window_name=args.name)


if __name__ == "__main__":
    main()

