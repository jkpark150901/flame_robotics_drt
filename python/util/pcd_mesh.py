"""
Point Cloud to Mesh via Marching Cubes
Supports PCD and PLY input files.

Method:
  1. Load point cloud
  2. Voxelize into a 3D occupancy grid
  3. Apply Gaussian smoothing to create a scalar field
  4. Run marching cubes at isovalue `level` to extract surface mesh
  5. Save mesh and smoothed point cloud (original world scale)

Level (isovalue):
  Gaussian smoothing produces a scalar field in range [0, 1].
  `level` defines the isosurface threshold — the mesh is extracted exactly
  where the field equals this value.
    - level=0.5  : surface at 50% density (default, balanced)
    - level < 0.5: surface expands outward (captures sparser regions)
    - level > 0.5: surface shrinks inward (only dense core)

Usage:
  python pcd_ply_to_mesh.py input.pcd -o output.ply
  python pcd_ply_to_mesh.py input.ply --resolution 256 --sigma 2.0 --level 0.4
"""

import argparse
import pathlib
import numpy as np
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import trimesh
import open3d as o3d


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_point_cloud(path: str) -> np.ndarray:
    """Load PCD or PLY file via open3d, return (N, 3) float32 array."""
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError(f"No points loaded from: {path}")
    return pts


# ---------------------------------------------------------------------------
# Marching Cubes pipeline
# ---------------------------------------------------------------------------

def build_scalar_field(
    points: np.ndarray,
    resolution: int,
    sigma: float,
    padding: float,
):
    """
    Voxelize point cloud and apply Gaussian smoothing.

    Returns:
        smoothed:  (R, R, R) float32 scalar field, normalized to [0, 1]
        bb_min:    world-space origin of the voxel grid
        extent:    world-space size of the voxel grid per axis
    """
    pts = np.asarray(points, dtype=np.float64)
    bb_min = pts.min(axis=0)
    bb_max = pts.max(axis=0)
    extent = bb_max - bb_min
    pad = extent * padding
    bb_min -= pad
    bb_max += pad
    extent = bb_max - bb_min

    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    norm = (pts - bb_min) / extent
    idx = (norm * (resolution - 1)).astype(int)
    idx = np.clip(idx, 0, resolution - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    smoothed = gaussian_filter(grid, sigma=sigma)

    s_min, s_max = smoothed.min(), smoothed.max()
    if s_max - s_min < 1e-10:
        raise ValueError("Smoothed field is flat — check point cloud data")
    smoothed = (smoothed - s_min) / (s_max - s_min)

    return smoothed, bb_min, extent, grid


def scalar_field_to_pcd(
    smoothed: np.ndarray,
    bb_min: np.ndarray,
    extent: np.ndarray,
    threshold: float = 0.05,
) -> trimesh.PointCloud:
    """
    Convert non-zero voxels of the smoothed scalar field back to a point cloud
    in original world coordinates. Points are colored by their scalar value
    (blue = low, red = high).

    Args:
        smoothed:  normalized scalar field [0, 1]
        bb_min:    world-space origin
        extent:    world-space extent per axis
        threshold: minimum scalar value to include as a point

    Returns:
        trimesh.PointCloud with RGBA colors
    """
    resolution = smoothed.shape[0]
    xi, yi, zi = np.where(smoothed > threshold)
    values = smoothed[xi, yi, zi]

    # Voxel index → world coords (original scale, no normalization applied)
    voxel_coords = np.stack([xi, yi, zi], axis=1).astype(np.float64)
    world_coords = voxel_coords / (resolution - 1) * extent + bb_min

    # Color: blue (low value) → red (high value)
    colors = np.zeros((len(values), 4), dtype=np.uint8)
    colors[:, 0] = (values * 255).astype(np.uint8)          # R
    colors[:, 2] = ((1.0 - values) * 255).astype(np.uint8)  # B
    colors[:, 3] = 255                                        # A

    return trimesh.PointCloud(vertices=world_coords, colors=colors)


def points_to_mesh(
    points: np.ndarray,
    resolution: int = 128,
    sigma: float = 1.5,
    padding: float = 0.05,
    level: float = 0.5,
) -> tuple:
    """
    Convert point cloud to mesh via marching cubes, and also return
    the smoothed scalar field as a point cloud in world coordinates.

    Args:
        points:     (N, 3) point cloud
        resolution: voxel grid resolution per axis
        sigma:      Gaussian smoothing sigma in voxels
        padding:    fractional padding around bounding box
        level:      marching cubes isovalue in [0, 1]
                    0.5 = surface at 50% density (default)
                    lower → expands outward, higher → shrinks inward

    Returns:
        (mesh: trimesh.Trimesh, smoothed_pcd: trimesh.PointCloud)
    """
    smoothed, bb_min, extent, grid = build_scalar_field(points, resolution, sigma, padding)

    verts, faces, normals, _ = marching_cubes(grid, level=level)

    # Map voxel coords → world coords (original scale)
    verts_world = verts / (resolution - 1) * extent + bb_min

    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, vertex_normals=normals)
    smoothed_pcd = scalar_field_to_pcd(smoothed, bb_min, extent)

    return mesh, smoothed_pcd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Point cloud (PCD/PLY) → mesh via marching cubes"
    )
    parser.add_argument("input", help="Input PCD or PLY file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output mesh file (PLY/OBJ/STL). Default: <input>_mesh.ply")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Voxel grid resolution (default: 128)")
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Gaussian smoothing sigma in voxels (default: 1.5)")
    parser.add_argument("--padding", type=float, default=0.05,
                        help="Bounding box padding fraction (default: 0.05)")
    parser.add_argument("--level", type=float, default=0.5,
                        help=(
                            "Marching cubes isovalue in [0,1] (default: 0.5). "
                            "Lower = surface expands outward, Higher = shrinks inward."
                        ))
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        return

    output_path = pathlib.Path(args.output) if args.output else \
        input_path.with_name(input_path.stem + "_mesh.ply")
    smoothed_path = output_path.with_name(output_path.stem.replace("_mesh", "") + "_smoothed.ply")

    print(f"Loading: {input_path}")
    points = load_point_cloud(str(input_path))
    print(f"  {len(points)} points loaded")
    print(f"  Bounding box: {points.min(axis=0)} ~ {points.max(axis=0)}")

    print(f"Running marching cubes "
          f"(resolution={args.resolution}, sigma={args.sigma}, level={args.level}) ...")
    mesh, smoothed_pcd = points_to_mesh(
        points,
        resolution=args.resolution,
        sigma=args.sigma,
        padding=args.padding,
        level=args.level,
    )
    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  Smoothed PCD: {len(smoothed_pcd.vertices)} points")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh.export(str(output_path))
    print(f"Saved mesh:         {output_path}")

    smoothed_pcd.export(str(smoothed_path))
    print(f"Saved smoothed PCD: {smoothed_path}")


if __name__ == "__main__":
    main()
