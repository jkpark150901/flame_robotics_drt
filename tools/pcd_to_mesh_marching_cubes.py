"""
Convert a PCD point cloud to a triangle mesh with marching cubes.

The point cloud is rasterized into a voxel grid, converted to an unsigned
distance field, and meshed at a chosen distance iso-level.
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
from scipy import ndimage
from skimage import measure


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and save a mesh from a PCD file using marching cubes."
    )
    parser.add_argument("--file", required=True, help="Input point cloud file (*.pcd, *.ply, *.xyz, ...)")
    parser.add_argument(
        "--output",
        default="",
        help="Output mesh path (*.stl, *.ply, *.obj). Defaults to '<input>_mc.stl'.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Voxel size in the point cloud unit. If omitted, it is estimated from --grid-resolution.",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=160,
        help="Target grid resolution along the longest axis when --voxel-size is omitted.",
    )
    parser.add_argument(
        "--point-radius",
        type=float,
        default=0.0,
        help="Iso-surface radius around points. Defaults to 1.5 * voxel_size.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=4,
        help="Padding around the point cloud bounds in voxels.",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=20_000_000,
        help="Safety limit for total voxels in the generated grid.",
    )
    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        help="Remove statistical outliers before meshing.",
    )
    parser.add_argument(
        "--nb-neighbors",
        type=int,
        default=20,
        help="Neighbor count for --remove-outliers.",
    )
    parser.add_argument(
        "--std-ratio",
        type=float,
        default=2.0,
        help="Standard deviation ratio for --remove-outliers.",
    )
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=0,
        help="Number of Taubin smoothing iterations after marching cubes.",
    )
    parser.add_argument(
        "--simplify-triangles",
        type=int,
        default=0,
        help="Target triangle count for quadric simplification. 0 disables simplification.",
    )
    return parser.parse_args()


def _default_output_path(input_path):
    path_no_ext, _ = os.path.splitext(input_path)
    return f"{path_no_ext}_mc.stl"


def _load_points(path, remove_outliers, nb_neighbors, std_ratio):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Point cloud has no points: {path}")

    if remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        if pcd.is_empty():
            raise ValueError("All points were removed as outliers.")

    return np.asarray(pcd.points, dtype=np.float64)


def _estimate_voxel_size(points, grid_resolution):
    if grid_resolution < 8:
        raise ValueError("--grid-resolution must be at least 8.")

    extent = points.max(axis=0) - points.min(axis=0)
    longest_axis = float(np.max(extent))
    if longest_axis <= 0.0:
        raise ValueError("Point cloud bounds are degenerate.")

    return longest_axis / float(grid_resolution - 1)


def _build_distance_field(points, voxel_size, padding, max_voxels):
    min_bound = points.min(axis=0) - padding * voxel_size
    max_bound = points.max(axis=0) + padding * voxel_size
    grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(np.int64) + 1

    total_voxels = int(np.prod(grid_shape))
    if total_voxels > max_voxels:
        raise ValueError(
            f"Grid is too large: shape={tuple(grid_shape)}, voxels={total_voxels:,}. "
            "Increase --voxel-size or --max-voxels, or lower --grid-resolution."
        )

    indices = np.rint((points - min_bound) / voxel_size).astype(np.int64)
    indices = np.clip(indices, 0, grid_shape - 1)

    occupied = np.zeros(tuple(grid_shape), dtype=bool)
    occupied[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    distance = ndimage.distance_transform_edt(~occupied, sampling=voxel_size)
    return distance, min_bound, grid_shape


def _create_mesh_from_field(distance, min_bound, voxel_size, point_radius):
    if point_radius <= 0.0:
        raise ValueError("--point-radius must be positive.")
    if point_radius >= float(distance.max()):
        raise ValueError(
            "--point-radius is larger than the distance field range. "
            "Use a smaller value or add more padding."
        )

    vertices, faces, normals, _ = measure.marching_cubes(
        distance,
        level=point_radius,
        spacing=(voxel_size, voxel_size, voxel_size),
        allow_degenerate=False,
    )
    vertices += min_bound

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    return mesh


def _clean_mesh(mesh, smooth_iterations, simplify_triangles):
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    if smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)

    if simplify_triangles > 0 and len(mesh.triangles) > simplify_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=simplify_triangles)
        mesh.remove_degenerate_triangles()

    mesh.compute_vertex_normals()
    return mesh


def main():
    args = _parse_args()

    points = _load_points(
        args.file,
        remove_outliers=args.remove_outliers,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
    )

    voxel_size = args.voxel_size or _estimate_voxel_size(points, args.grid_resolution)
    if voxel_size <= 0.0:
        raise ValueError("--voxel-size must be positive.")

    point_radius = args.point_radius or (1.5 * voxel_size)
    output_path = args.output or _default_output_path(args.file)

    distance, min_bound, grid_shape = _build_distance_field(
        points,
        voxel_size=voxel_size,
        padding=args.padding,
        max_voxels=args.max_voxels,
    )
    mesh = _create_mesh_from_field(
        distance,
        min_bound=min_bound,
        voxel_size=voxel_size,
        point_radius=point_radius,
    )
    mesh = _clean_mesh(
        mesh,
        smooth_iterations=args.smooth_iterations,
        simplify_triangles=args.simplify_triangles,
    )

    if not o3d.io.write_triangle_mesh(output_path, mesh):
        raise RuntimeError(f"Failed to write mesh: {output_path}")

    print(f"Input points: {len(points):,}")
    print(f"Grid shape: {tuple(int(v) for v in grid_shape)}")
    print(f"Voxel size: {voxel_size:g}")
    print(f"Iso radius: {point_radius:g}")
    print(f"Output vertices: {len(mesh.vertices):,}")
    print(f"Output triangles: {len(mesh.triangles):,}")
    print(f"Saved mesh: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
