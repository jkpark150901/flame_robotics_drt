"""
STL Mesh to Point Cloud Data File (pcd, ply)
"""

import open3d as o3d
import trimesh
import argparse
import os

from util.logger.console import ConsoleLogger

if __name__ == "__main__":

    console = ConsoleLogger.get_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', nargs='?', required=True, help="Mesh File (*.stl)", default="")
    parser.add_argument('--sample', nargs='?', required=False, help="Number of Samples", default=10000)
    parser.add_argument('--ext', nargs='?', required=False, help="Converting File (PCD, PLY)", default="pcd")
    args = parser.parse_args()

    try:
        mesh = o3d.io.read_triangle_mesh(args.file)
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=int(args.sample))

        path_no_ext, ext = os.path.splitext(args.file)
        o3d.io.write_point_cloud(f"{path_no_ext}.{args.ext}", pcd)

    except Exception as e:
        console.error(f"{e}")