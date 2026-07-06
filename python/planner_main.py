import open3d as o3d
import numpy as np
import importlib
import os
import sys
import argparse
import glob
import inspect
import csv
import datetime
import datetime
import itertools
import logging
import colorlog

# Global Config
GOAL_NORMAL_OFFSET = 10.0

# Add Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'core/pluginbase')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'core/algorithms')))



from plannerbase import PlannerBase

def load_plugins():
    plugins = {}
    algo_dir = os.path.join(os.path.dirname(__file__), 'core/algorithms')
    sys.path.append(algo_dir)
    
    for file in os.listdir(algo_dir):
        if file.endswith('.py') and not file == '__init__.py':
            module_name = file[:-3]
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, PlannerBase) and obj is not PlannerBase:
                        plugins[module_name] = obj
            except Exception as e:
                print(f"Error loading {module_name}: {e}")
                
    return plugins

from optimizerbase import OptimizerBase

def load_optimizers():
    optimizers = {}
    opt_dir = os.path.join(os.path.dirname(__file__), 'core/algorithms/optimization')
    sys.path.append(opt_dir)
    
    if os.path.exists(opt_dir):
        for file in os.listdir(opt_dir):
            if file.endswith('.py') and not file == '__init__.py':
                module_name = file[:-3]
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, OptimizerBase) and obj is not OptimizerBase:
                            optimizers[module_name] = obj
                            optimizers[name.lower()] = obj
                except Exception as e:
                    print(f"Error loading optimizer {module_name}: {e}")
    return optimizers

def create_coordinate_frame_mesh(pose, size=1.0):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    R = o3d.geometry.get_rotation_matrix_from_xyz(pose[3:])
    mesh.rotate(R, center=(0, 0, 0))
    mesh.translate(pose[:3])
    return mesh

def generate_random_pose(mesh, distance_offset):
    min_b = mesh.get_min_bound()
    max_b = mesh.get_max_bound()
    center = (min_b + max_b) / 2.0
    extent = np.linalg.norm(max_b - min_b) / 2.0
    
    while True:
        direction = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(direction) > 0.1:
            direction /= np.linalg.norm(direction)
            break
            
    pos = center + direction * (extent + distance_offset)
    orient = np.random.uniform(-np.pi, np.pi, 3)
    return np.concatenate((pos, orient))

def create_sphere_marker(pose, color, radius=0.5):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.paint_uniform_color(color)
    mesh.translate(pose[:3])
    return mesh

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s < 1e-6:
        # Same direction or opposite
        if c > 0:
            return np.eye(3)
        else:
            # Opposite direction
            if np.abs(a[0]) > 0.9:
                axis = np.array([0, 1, 0])
            else:
                axis = np.array([1, 0, 0])
            v_ortho = np.cross(a, axis)
            v_ortho /= np.linalg.norm(v_ortho)
            K = np.array([[0, -v_ortho[2], v_ortho[1]], [v_ortho[2], 0, -v_ortho[0]], [-v_ortho[1], v_ortho[0], 0]])
            return np.eye(3) + 2 * (K @ K)
            
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rotation_matrix_to_euler(R):
    # Calculates Rotation Matrix to Euler Angles (XYZ Convention)
    sy = np.sqrt(R[0,0] * R[0,0] + R[0,1] * R[0,1])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(-R[1,2], R[2,2])
        y = np.arctan2(R[0,2], sy)
        z = np.arctan2(-R[0,1], R[0,0])
    else:
        x = np.arctan2(R[1,0], R[1,1])
        y = np.arctan2(R[0,2], sy)
        z = 0

    return np.array([x, y, z])

    return np.array([x, y, z])

class PathPlannerApp:
    def __init__(self):
        self.setup_logging()
        self.path = None
        self.optimized_path = None
        self.parser = argparse.ArgumentParser(description="Path Planner Main")
        self.parser.add_argument('--algorithm', type=str, default='task_space_rrt', help='Algorithm to use')
        self.parser.add_argument('--stl', type=str, default='sample/PIPE NO.1_fill.stl', help='Path to STL file')
        self.parser.add_argument('--tool', type=str, default=None, help='Path to Tool STL file (relative to end-effector)')
        self.parser.add_argument('--start', nargs=6, type=float, default=None, help='Start pose x y z r p y')
        self.parser.add_argument('--goal', nargs=6, type=float, default=None, help='Goal pose x y z r p y')
        self.parser.add_argument('--show_coord', action='store_true', help='Show coordinate frames for waypoints')
        self.parser.add_argument('--show_history', action='store_true', help='Visualize planning process iteration by iteration')
        self.parser.add_argument('--optimize', type=str, default=None, help='Optimization method')
        self.args = self.parser.parse_args()
        
        
        self.stl_path = os.path.abspath(self.args.stl)
        if not os.path.exists(self.stl_path):
            print(f"STL file not found: {self.stl_path}")
            sys.exit(1)
            
        self.mesh = o3d.io.read_triangle_mesh(self.stl_path)
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color([0.5, 0.5, 0.5])
        self.mesh_obb = self.mesh.get_oriented_bounding_box()
        
        self.tool_mesh = None
        if self.args.tool:
            tool_path = os.path.abspath(self.args.tool)
            if os.path.exists(tool_path):
                self.tool_mesh = o3d.io.read_triangle_mesh(tool_path)
                self.tool_mesh.compute_vertex_normals()
                self.tool_mesh.paint_uniform_color([0.8, 0.4, 0.0]) # Orange-ish for tool
                print(f"Loaded tool mesh: {tool_path}")
            else:
                print(f"Tool file not found: {tool_path}")
        
        self.min_b = self.mesh.get_min_bound()
        self.max_b = self.mesh.get_max_bound()
        self.center = (self.min_b + self.max_b) / 2.0
        self.extent = self.max_b - self.min_b
        
        self._init_collision_checker()

        # Initialize random Start/Goal if not provided
        if self.args.start is None:
            print("Start pose not provided, randomizing...")
            self.args.start = self._sample_start_pose()
        else:
             self.args.start = np.array(self.args.start)

        if self.args.goal is None:
             print("Goal pose not provided, sampling from mesh surface...")
             self.args.goal = self._sample_goal_pose_on_surface()
        else:
             self.args.goal = np.array(self.args.goal)
             
        self.plugins = load_plugins()
        self.setup_algorithm()
        
        self.dynamic_geometries = []
        self.vis = None
        
    def setup_logging(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)s] %(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        logger = colorlog.getLogger()
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def _sample_start_pose(self):
        max_attempts = 1000
        for _ in range(max_attempts):
            rand_pos = np.zeros(3)
            for i in range(3):
                rand_pos[i] = np.random.uniform(self.center[i] - self.extent[i], self.center[i] + self.extent[i])
            
            # Check if point is inside OBB
            indices = self.mesh_obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector([rand_pos]))
            if len(indices) == 0:
                # Outside OBB
                rand_ori = np.random.uniform(-np.pi, np.pi, 3)
                pose = np.concatenate((rand_pos, rand_ori))
                if not self._check_collision_pose(pose):
                    return pose
                
        print("Warning: Could not sample start pose outside OBB after many attempts.")
        # Fallback
        rand_ori = np.random.uniform(-np.pi, np.pi, 3)
        return np.concatenate((rand_pos, rand_ori))

    def _sample_goal_pose_on_surface(self):
        # Sample points uniformly. If mesh has normals, sampled pcd inherits them?
        # We need to rely on Open3D behavior. `sample_points_uniformly` usually preserves normals.
        # To be safe, we sample, then finding nearest point on mesh is redundant if sampling *from* mesh.
        
        max_attempts = 100
        for _ in range(max_attempts):
            pcd = self.mesh.sample_points_uniformly(number_of_points=100) 
            idx = np.random.randint(0, len(pcd.points))
            point = np.asarray(pcd.points)[idx]
            
            if pcd.has_normals():
                normal = np.asarray(pcd.normals)[idx]
            else:
                normal = np.array([0, 0, 1])
                
            self.goal_surface_point = point
            self.goal_normal = normal
                
            target_pos = point + normal * GOAL_NORMAL_OFFSET
            
            # Calculate Orientation: Align Z-axis (0,0,1) with Normal
            z_axis = np.array([0, 0, 1])
            R = rotation_matrix_from_vectors(z_axis, normal)
            rpy = rotation_matrix_to_euler(R)
            
            pose = np.concatenate((target_pos, rpy))
            
            if not self._check_collision_pose(pose):
                return pose
        
        print("Warning: Could not find collision-free goal pose after many attempts.")
        return pose

    def setup_algorithm(self):
        if self.args.algorithm not in self.plugins and self.args.algorithm not in [p.split('.')[-1] for p in self.plugins]:
            print(f"Algorithm {self.args.algorithm} not found. Available: {list(self.plugins.keys())}")
            for k in self.plugins:
                if self.args.algorithm in k:
                    self.args.algorithm = k
                    break
        
        if self.args.algorithm in self.plugins:
            self.planner_class = self.plugins[self.args.algorithm]
            self.planner = self.planner_class()
            print(f"Loaded planner: {self.planner_class.__name__}")
            self.planner.add_collision_object(self.mesh)
            if self.tool_mesh:
                self.planner.set_tool_geometry(self.tool_mesh)
            self._configure_planner()
        else:
             print("Algorithm setup failed.")
             sys.exit(1)

    def _configure_planner(self):
        # Configuration is now handled entirely by the JSON files of the respective algorithms.
        # We process overrides here only if strictly necessary (e.g. from command line args).
        pass

    def run_planning(self):
        print(f"Generating path from {self.args.start} to {self.args.goal}...")
        
        cb = self.planning_callback if self.args.show_history and self.vis else None
        if cb:
             print("Visualizing history...")
             
        self.path = self.planner.generate(self.args.start, self.args.goal, step_callback=cb)
        
        self.optimized_path = None
        if self.path:
            print(f"Path found with {len(self.path)} waypoints")
            if self.args.optimize:
                optimizers = load_optimizers()
                if self.args.optimize in optimizers:
                    print(f"Optimizing using {self.args.optimize}...")
                    opt_class = optimizers[self.args.optimize]
                    optimizer = opt_class()
                    self.optimized_path = optimizer.optimize(self.path, self.planner)
                    print(f"Optimized path: {len(self.optimized_path)} waypoints")
                else:
                    print(f"Optimizer {self.args.optimize} not found.")
        else:
            print("No path found!")
            
        # Clear history (green wireframes) after planning
        if hasattr(self, 'history_geometries'):
            if self.vis:
                 for g in self.history_geometries:
                     self.vis.remove_geometry(g, reset_bounding_box=False)
            self.history_geometries = []
            
        if hasattr(self, 'last_tool_geoms'):
            if self.vis:
                 for g in self.last_tool_geoms:
                     self.vis.remove_geometry(g, reset_bounding_box=False)
            self.last_tool_geoms = []

        self.save_csv()
        return self.path, self.optimized_path
    
    def save_csv(self):
        if not self.path: return
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"generated_path_{timestamp}.csv"
        
        headers = []
        algo_name = self.args.algorithm
        headers.extend([f"{algo_name}_x", f"{algo_name}_y", f"{algo_name}_z", f"{algo_name}_r", f"{algo_name}_p", f"{algo_name}_y"])
        
        if self.optimized_path:
            opt_name = self.args.optimize
            headers.extend([f"{opt_name}_x", f"{opt_name}_y", f"{opt_name}_z", f"{opt_name}_r", f"{opt_name}_p", f"{opt_name}_y"])
            
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            iter_path = self.path
            iter_opt = self.optimized_path if self.optimized_path else []
            for p1, p2 in itertools.zip_longest(iter_path, iter_opt, fillvalue=None):
                row = []
                if p1 is not None:
                    row.extend(p1)
                else:
                    row.extend([''] * 6)
                if self.optimized_path:
                    if p2 is not None:
                        row.extend(p2)
                    else:
                        row.extend([''] * 6)
                writer.writerow(row)
        print(f"Saved {filename}")

    def randomize_start(self):
        self.args.start = self._sample_start_pose()
        self.args.goal = self._sample_goal_pose_on_surface()
        print(f"New Random Start: {self.args.start}")
        print(f"New Random Goal: {self.args.goal}")

    def update_geometries(self, vis):
        for geom in self.dynamic_geometries:
            vis.remove_geometry(geom, reset_bounding_box=False)
        self.dynamic_geometries.clear()
        
        bbox_diag = np.linalg.norm(self.extent)
        scale = bbox_diag * 0.2 if bbox_diag > 0 else 1.0
        
        g1 = create_coordinate_frame_mesh(self.args.start, size=scale*0.2)
        g2 = create_sphere_marker(self.args.start, [1, 0, 0], radius=scale*0.02)
        g3 = create_coordinate_frame_mesh(self.args.goal, size=scale*0.2)
        g4 = create_sphere_marker(self.args.goal, [0, 0, 1], radius=scale*0.02)
        
        self.dynamic_geometries.extend([g1, g2, g3, g4])
        
        # Tool visualization at Start/Goal
        if self.tool_mesh:
            # Start (Wireframe)
            start_tool_mesh = o3d.geometry.TriangleMesh(self.tool_mesh)
            R = o3d.geometry.get_rotation_matrix_from_xyz(self.args.start[3:])
            start_tool_mesh.rotate(R, center=(0, 0, 0))
            start_tool_mesh.translate(self.args.start[:3])
            
            start_tool_wire = o3d.geometry.LineSet.create_from_triangle_mesh(start_tool_mesh)
            start_tool_wire.paint_uniform_color([0.0, 0.0, 1.0]) # Blue
            self.dynamic_geometries.append(start_tool_wire)
            
            # Goal (Wireframe)
            goal_tool_mesh = o3d.geometry.TriangleMesh(self.tool_mesh)
            R = o3d.geometry.get_rotation_matrix_from_xyz(self.args.goal[3:])
            goal_tool_mesh.rotate(R, center=(0, 0, 0))
            goal_tool_mesh.translate(self.args.goal[:3])
            
            goal_tool_wire = o3d.geometry.LineSet.create_from_triangle_mesh(goal_tool_mesh)
            goal_tool_wire.paint_uniform_color([1.0, 0.0, 0.0]) # Red
            self.dynamic_geometries.append(goal_tool_wire)
            
        if self.path:
            points = [p[:3] for p in self.path]
            lines = [[i, i+1] for i in range(len(points)-1)]
            colors = [[0, 0, 1] for _ in range(len(lines))] # Blue
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            ls.colors = o3d.utility.Vector3dVector(colors)
            self.dynamic_geometries.append(ls)
            
            if self.args.show_coord and not self.optimized_path:
                for p in self.path:
                    self.dynamic_geometries.append(create_coordinate_frame_mesh(p, size=scale*0.3))

        if self.optimized_path:
            points = [p[:3] for p in self.optimized_path]
            lines = [[i, i+1] for i in range(len(points)-1)]
            colors = [[1, 0, 0] for _ in range(len(lines))] # Red
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            ls.colors = o3d.utility.Vector3dVector(colors)
            self.dynamic_geometries.append(ls)
             
            if self.args.show_coord:
                for p in self.optimized_path:
                    self.dynamic_geometries.append(create_coordinate_frame_mesh(p, size=scale*0.3))
            
            self.dynamic_geometries.append(create_coordinate_frame_mesh(self.optimized_path[0], size=scale*0.25))
            self.dynamic_geometries.append(create_coordinate_frame_mesh(self.optimized_path[-1], size=scale*0.25))

        if hasattr(self, 'goal_surface_point') and hasattr(self, 'goal_normal'):
            # Visualize Normal as a Cylinder (Thicker Line)
            # Length: 30, Radius: 0.2 (Double thickness approx)
            length = 30.0
            radius = 0.2
            normal_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
            normal_cylinder.paint_uniform_color([1, 0, 0]) # Red
            
            # Align Cylinder Z-axis to Normal
            z_axis = np.array([0, 0, 1])
            R = rotation_matrix_from_vectors(z_axis, self.goal_normal)
            normal_cylinder.rotate(R, center=(0, 0, 0))
            
            # Translate to Surface Point (Start) + Half Length * Normal (to center it correctly)
            # Cylinder origin is at its center.
            center_pos = self.goal_surface_point + self.goal_normal * (length / 2.0)
            normal_cylinder.translate(center_pos)
            
            self.dynamic_geometries.append(normal_cylinder)

        for geom in self.dynamic_geometries:
            vis.add_geometry(geom, reset_bounding_box=False)

    def planning_callback(self, *args):
        # args: (nodes, parents) for RRT/RRT*
        # args: (tree_a, parents_a, tree_b, parents_b) for Connect
        
        if not self.vis: return
        
        # Helper to draw tree
        def draw_tree(nodes, parents, color):
            points = [n[:3] for n in nodes]
            lines = []
            for i, parent_idx in parents.items():
                if parent_idx is not None:
                    lines.append([parent_idx, i])
            
            if lines:
                ls = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                ls.paint_uniform_color(color)
                return ls
            return None
            
        # Helper to draw tool
        def draw_tool(pose, color):
            if not self.tool_mesh: return None
            tm = o3d.geometry.LineSet.create_from_triangle_mesh(self.tool_mesh)
            R = o3d.geometry.get_rotation_matrix_from_xyz(pose[3:])
            tm.rotate(R, center=(0,0,0))
            tm.translate(pose[:3])
            tm.paint_uniform_color(color)
            return tm

        # Remove previous history geoms
        if hasattr(self, 'history_geometries'):
            for g in self.history_geometries:
                self.vis.remove_geometry(g, reset_bounding_box=False)
        self.history_geometries = []
        
        geoms_to_add = []
        self.last_tool_geoms = []
        
        if len(args) == 2: # RRT / RRT*
            nodes, parents = args
            tree_ls = draw_tree(nodes, parents, [0, 0, 0]) # Black
            if tree_ls: geoms_to_add.append(tree_ls)
            
            # Draw Tool at last node
            if nodes:
                tool_ls = draw_tool(nodes[-1], [0.0, 1.0, 0.0]) # Green tool
                if tool_ls: 
                    geoms_to_add.append(tool_ls)
                    self.last_tool_geom = tool_ls
                
        elif len(args) == 4: # Connect
            t1, p1, t2, p2 = args
            ls1 = draw_tree(t1, p1, [0, 0.5, 0]) # Green
            ls2 = draw_tree(t2, p2, [0, 0, 0.5]) # Blue
            if ls1: geoms_to_add.append(ls1)
            if ls2: geoms_to_add.append(ls2)
            
            if t1:
                tool1 = draw_tool(t1[-1], [0.0, 1.0, 0.0])
                if tool1:  
                    geoms_to_add.append(tool1)
                    # We might have 2 tools. Just track one as last? Or list?
                    # User likely wants both removed.
                    # Let's use a list self.last_tool_geoms = []
                    self.last_tool_geoms.append(tool1)
            if t2:
                tool2 = draw_tool(t2[-1], [0.0, 1.0, 0.0])
                if tool2: 
                    geoms_to_add.append(tool2)
                    self.last_tool_geoms.append(tool2)
                
        for g in geoms_to_add:
            self.vis.add_geometry(g, reset_bounding_box=False)
            self.history_geometries.append(g)
            
        self.vis.poll_events()
        self.vis.update_renderer()
        import time
        time.sleep(0.001)

    def run(self):
        # 1. Setup Visualizer Manually
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Path Planner", width=1600, height=1200, left=50, top=50)
        
        # 2. Prepare Static Geometries
        vis_init = [self.mesh]
        
        # Visualize Search Space (Black Box) from Planner Bounds
        if hasattr(self.planner, 'bounds'):
            b = self.planner.bounds
            x_min, x_max = b['x_min'], b['x_max']
            y_min, y_max = b['y_min'], b['y_max']
            z_min, z_max = b['z_min'], b['z_max']
            
            box_points = [
                [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min], # Bottom 0-3
                [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]  # Top 4-7
            ]
            box_lines = [
                [0, 1], [1, 2], [2, 3], [3, 0], # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4], # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical lines
            ]
            search_box = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(box_points),
                lines=o3d.utility.Vector2iVector(box_lines),
            )
            search_box.paint_uniform_color([0, 0, 0])
            vis_init.append(search_box)
            
        # 3. Add Static to Vis
        for g in vis_init:
            self.vis.add_geometry(g)
            
        # 4. Register Callbacks
        def view_x(vis):
            ctr = vis.get_view_control()
            ctr.set_front([-1.0, 0.0, 0.0])
            ctr.set_up([0.0, 0.0, 1.0])
            return False
            
        def view_y(vis):
            ctr = vis.get_view_control()
            ctr.set_front([0.0, -1.0, 0.0])
            ctr.set_up([0.0, 0.0, 1.0])
            return False
            
        def view_z(vis):
            ctr = vis.get_view_control()
            ctr.set_front([0.0, 0.0, 1.0])
            ctr.set_up([0.0, 1.0, 0.0])
            return False
            
        def perform_replan(vis):
            # 1. Clear History (Tree)
            if hasattr(self, 'history_geometries'):
                for g in self.history_geometries:
                    vis.remove_geometry(g, reset_bounding_box=False)
                self.history_geometries = []
            
            # 2. Reset Path and Show New Start/Goal (Before Planning)
            self.path = None
            self.optimized_path = None
            self.update_geometries(vis) # Removes old dynamic, adds new Start/Goal
            vis.poll_events()
            vis.update_renderer()
                
            self.run_planning()
            
            # Remove ghost tools
            if hasattr(self, 'last_tool_geoms'):
                for g in self.last_tool_geoms:
                    vis.remove_geometry(g, reset_bounding_box=False)
                self.last_tool_geoms = []
                
            self.update_geometries(vis) # Update to show Path
            
            vis.poll_events()
            vis.update_renderer()
            return True

        def randomize_and_replan(vis):
            print("\n[N] pressed. Replanning with NEW random start...")
            self.randomize_start()
            return perform_replan(vis)
            
        def replan_current(vis):
            print("\n[R] pressed. Replanning with SAME start/goal...")
            return perform_replan(vis)

        self.vis.register_key_callback(ord('1'), view_x)
        self.vis.register_key_callback(ord('2'), view_y)
        self.vis.register_key_callback(ord('3'), view_z)
        
        # New Bindings
        self.vis.register_key_callback(ord('N'), randomize_and_replan)
        self.vis.register_key_callback(ord('n'), randomize_and_replan)
        self.vis.register_key_callback(ord('R'), replan_current)
        self.vis.register_key_callback(ord('r'), replan_current)
        
        # 5. Initial Render (Warm-up for macOS window)
        print("Initializing Visualization Window...")
        
        # Draw Initial Start/Goal
        self.update_geometries(self.vis)
        
        import time
        for _ in range(10):
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.05)
        print("Window opened...")
        
        # 6. Run Planning (Will animate if show_history is True)
        self.run_planning()
        
        # Initialize Planneral result (Path, etc.)
        # Note: run_planning updates self.path. We need to refresh geoms to show path.
        # But we must clear the old 'Start/Goal' markers we just added? 
        # Actually update_geometries appends to list.
        # If we call it again, we get duplicates?
        # update_geometries resets the list: self.dynamic_geometries = []
        # So we must remove current dynamic_geometries from Vis before calling it again.
        
        for g in self.dynamic_geometries:
            self.vis.remove_geometry(g, reset_bounding_box=False)
            
        self.update_geometries(self.vis)
        
        # 8. Start Interaction Loop
        print("Press '1', '2', '3' for views. Press 'R' to randomize start and replan.")
        self.vis.run()
        self.vis.destroy_window()

    def _init_collision_checker(self):
        self.scene = o3d.t.geometry.RaycastingScene()
        # Add static mesh to scene
        try:
            # We need to convert legacy mesh to tensor mesh for RaycastingScene
            t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
            self.scene.add_triangles(t_mesh)
        except Exception as e:
            print(f"Error initializing collision scene: {e}")
            self.scene = None

        if self.tool_mesh:
             self.tool_pcd = self.tool_mesh.sample_points_poisson_disk(number_of_points=5000)
             self.tool_pts = np.asarray(self.tool_pcd.points)

    def _check_collision_pose(self, pose):
        # pose: [x,y,z, r,p,y]
        if self.scene is None: return False
        
        # Determine points to check
        if hasattr(self, 'tool_pts'):
             # Transform tool points
            R = o3d.geometry.get_rotation_matrix_from_xyz(pose[3:])
            # (N,3) @ (3,3) + (3,)
            check_pts = (R @ self.tool_pts.T).T + pose[:3]
        else:
            # Check just the center point (Start/Goal position)
            check_pts = np.array([pose[:3]])
        
        query = o3d.core.Tensor(check_pts, dtype=o3d.core.Dtype.Float32)
        dist = self.scene.compute_distance(query)
        min_dist = dist.min().item()
        
        # If any point is inside or too close (< 0.5mm)
        if min_dist < 0.5:
            return True
        return False

if __name__ == "__main__":
    app = PathPlannerApp()
    app.run()
