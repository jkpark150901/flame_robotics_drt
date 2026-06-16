"""
Remove statistical outliers from a point cloud file.

The filter removes points whose mean distance to neighboring points is farther
than the global mean distance by the configured standard deviation ratio.
"""

import argparse
import os
import sys

import open3d as o3d


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Remove statistical outliers from a point cloud."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Input point cloud file (*.pcd, *.ply, *.xyz, *.pts, ...).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output point cloud path. Defaults to '<input>_filtered.<ext>'.",
    )
    parser.add_argument(
        "--nb-neighbors",
        type=int,
        default=20,
        help="Number of nearest neighbors used to estimate each point distance.",
    )
    parser.add_argument(
        "--std-ratio",
        type=float,
        default=2.0,
        help="Standard deviation multiplier. Lower values remove more points.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Write output point cloud in ASCII format when supported.",
    )
    return parser.parse_args()


def _default_output_path(input_path):
    path_no_ext, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".pcd"
    return f"{path_no_ext}_filtered{ext}"


def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    if pcd.is_empty():
        raise ValueError("Input point cloud has no points.")
    if nb_neighbors < 2:
        raise ValueError("--nb-neighbors must be at least 2.")
    if std_ratio <= 0.0:
        raise ValueError("--std-ratio must be positive.")

    filtered_pcd, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return filtered_pcd, inlier_indices


def main():
    args = _parse_args()
    output_path = args.output or _default_output_path(args.file)

    pcd = o3d.io.read_point_cloud(args.file)
    original_count = len(pcd.points)

    filtered_pcd, inlier_indices = remove_statistical_outliers(
        pcd,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
    )

    filtered_count = len(filtered_pcd.points)
    removed_count = original_count - filtered_count
    removed_ratio = 0.0 if original_count == 0 else removed_count / original_count * 100.0

    if not o3d.io.write_point_cloud(output_path, filtered_pcd, write_ascii=args.ascii):
        raise RuntimeError(f"Failed to write point cloud: {output_path}")

    print(f"Input points: {original_count:,}")
    print(f"Inlier points: {filtered_count:,}")
    print(f"Removed points: {removed_count:,} ({removed_ratio:.2f}%)")
    print(f"Neighbor count: {args.nb_neighbors}")
    print(f"Std ratio: {args.std_ratio:g}")
    print(f"Saved point cloud: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
