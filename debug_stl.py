import open3d as o3d
import os

stl_path = 'sample/PIPE NO.1_fill.stl'
if os.path.exists(stl_path):
    mesh = o3d.io.read_triangle_mesh(stl_path)
    print(f"STL Loaded: {stl_path}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.triangles)}")
    print(f"Bounds: {mesh.get_axis_aligned_bounding_box()}")
    print(f"Min Bound: {mesh.get_min_bound()}")
    print(f"Max Bound: {mesh.get_max_bound()}")
else:
    print("STL file not found")
