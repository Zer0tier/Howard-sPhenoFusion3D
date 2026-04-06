import argparse

import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Open two PLY files in separate Open3D windows.")
    parser.add_argument("--ply-a", required=True, help="First PLY path")
    parser.add_argument("--ply-b", required=True, help="Second PLY path")
    parser.add_argument("--name-a", default="cloud_a", help="Window title for first cloud")
    parser.add_argument("--name-b", default="cloud_b", help="Window title for second cloud")
    args = parser.parse_args()

    p1 = o3d.io.read_point_cloud(args.ply_a)
    p2 = o3d.io.read_point_cloud(args.ply_b)

    print(f"{args.name_a} points: {len(p1.points)}")
    print(f"{args.name_b} points: {len(p2.points)}")

    o3d.visualization.draw_geometries([p1], window_name=args.name_a)
    o3d.visualization.draw_geometries([p2], window_name=args.name_b)


if __name__ == "__main__":
    main()

