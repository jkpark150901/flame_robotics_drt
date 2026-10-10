'''
3D Pipe Inspection Pose Detection Tool
@auhtor Byunghun Hwang<bh.hwnag@iae.re.kr>
'''

import sys, os, platform
import pathlib
import json
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# root directory registration on system environment
ROOT_PATH = pathlib.Path(__file__).parent.parent
APP_NAME = pathlib.Path(__file__).stem
sys.path.append(ROOT_PATH.as_posix())

import argparse
from pdtool.window import AppWindow
from util.logger.console import ConsoleLogger


if __name__ == "__main__":
    
    console = ConsoleLogger.get_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', required=False, help="Configuration File(*.cfg)", default="default.cfg")
    parser.add_argument('--file', nargs='?', required=True, help="Polygon File Format(*.ply)", default="default.ply")
    parser.add_argument('--verbose', nargs='?', required=False, help="Enable/Disable verbose", default=True)
    parser.add_argument('--sample', nargs='?', required=False, help="Number of Samples", default=10000)
    args = parser.parse_args()

    app = None
    try:
        with open(args.config, "r") as cfile:
            configure = json.load(cfile)

            configure["root_path"] = ROOT_PATH
            configure["app_path"] = (pathlib.Path(__file__).parent / APP_NAME)
            configure["verbose"] = args.verbose

            if args.verbose:
                console.info(f"+ Root Path : {configure['root_path']}")
                console.info(f"+ Application Path : {configure['app_path']}")
                console.info(f"+ Verbose Level : {configure['verbose']}")
                
            if args.file:
                # pcd = o3d.io.read_point_cloud(args.file)
                # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # flip 
                
                # axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
                # axis_aligned_bounding_box.color = (1, 0, 0)
                
                mesh = o3d.io.read_triangle_mesh(args.file)
                mesh.compute_vertex_normals()
                pcl = mesh.sample_points_poisson_disk(number_of_points=args.sample, use_triangle_normal=True)
                hull, _ = pcl.compute_convex_hull()
                hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
                hull_ls.paint_uniform_color((1, 0, 0))
    
                # o3d.visualization.draw([pcd, axis_aligned_bounding_box, hull_ls])
                o3d.visualization.draw([mesh, hull_ls])
                

    except json.JSONDecodeError as e:
        console.critical(f"Configuration File Load Error : {e}")
    except Exception as e:
        console.critical(f"{e}")
        
    
        
    