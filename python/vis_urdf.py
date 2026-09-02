
"""
URDF Load & Visualizationn Test
"""
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from urdf_parser import URDF
import os

# Initialize Open3D GUI application
app = gui.Application.instance
app.initialize()

# URDF load
urdf_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../urdf")
robot = URDF.load(os.path.join(urdf_root, "ur5.urdf"), lazy_load_meshes=False)

joint_cfg = {"shoulder_lift_joint": 0.0}
fk = robot.visual_trimesh_fk(cfg=joint_cfg)


# show linkes and visual mesh file paths
links = []
for link in robot.links:
    for visual in link.visuals:
        if hasattr(visual.geometry, 'mesh') and visual.geometry.mesh is not None:
            mesh_file_path = os.path.join(urdf_root, visual.geometry.mesh.filename)
            print(f"{link.name}: {mesh_file_path}")
            links.append(link)

# create window
w = app.create_window("URDF Visualizer", 1920, 1080)
scene = rendering.Open3DScene(w.renderer)

meshes = []
a = 1
for tm, T in fk.items():
    
    tm_copy = tm.copy()
    tm_copy.apply_transform(T)

    # convert trimesh to Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tm_copy.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tm_copy.faces)
    o3d_mesh.compute_vertex_normals()

    
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    #material.base_color = [1, 0, 0, 1]  # RGBA

    scene.add_geometry(f"robot_link_{a}", o3d_mesh, material)
    a = a+1


widget = gui.SceneWidget()
widget.scene = scene
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1, -1, -1), max_bound=(1, 1, 1))
center = bbox.get_center()
widget.setup_camera(60.0,bbox, center)  # fov, lookat, eye, up
w.add_child(widget)

app.run()