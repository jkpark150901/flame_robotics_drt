"""
3D Visualizer using Vedo
@note
- Vedo is a Python library for 3D visualization based on VTK (Visualization Toolkit).
- Visualizer is a class that renders 3D geometries
- All ZMQ communication is handled by Zapi (viewervedo/zapi.py)
"""

import threading
from collections import deque
import time
import importlib
import inspect
import sys
import types
import os
from pathlib import Path
import numpy as np
import vedo
try:
    import pinocchio as pin
except ImportError:
    pin = None

# Open3D core geometry is used here; the optional ML module can fail in this
# workspace because of NumPy/SciPy ABI mismatch.
sys.modules.setdefault("open3d.ml", types.ModuleType("open3d.ml"))
import open3d as _o3d
from util.logger.console import ConsoleLogger
from common.graphic_device import GraphicDevice
from viewervedo.robot import RobotModel, load_robots_from_config


class Visualizer:
    def __init__(self, config:dict=None):
        if config is None:
            config = {}
        self._config = config
    
        self.__console = ConsoleLogger.get_logger()

        # Thread-safe request queue (populated by Zapi)
        self._request_queue = deque(maxlen=100)
        self._queue_lock = threading.Lock()

        # Device Detection (Reusing GraphicDevice from common)
        self.gdevice = GraphicDevice()
        self.__console.info(f"Graphic Device: Running on {self.gdevice.get_device_name()}")
        
        # GPU Acceleration check for Vedo (VTK)
        if "cuda" in self.gdevice.get_device_name().lower() or "mps" in self.gdevice.get_device_name().lower():
             self.__console.info("GPU Acceleration enabled for Vedo/VTK (if available via drivers)")
             vedo.settings.use_depth_peeling = True # Better transparency on GPU
        else:
             self.__console.info("Running on CPU mode for Vedo/VTK")

        # Initialize Vedo Plotter
        window_title = config.get('window_title', f'Vedo Viewer (Optimized - {self.gdevice.get_device_name()})')
        window_size = config.get('window_size', [1920, 1080])
        bg_color = config.get('background_color', [1.0, 1.0, 1.0])
        
        # create plotter
        self.plotter = vedo.Plotter(title=window_title, size=window_size, bg=bg_color, interactive=False)

        # Setup scene elements
        self._setup_c_space(config)
        self._setup_robots(config)

        
        # Flag for external termination (set by Zapi)
        self._should_close = False

        # 매니퓰레이터 조인트 애니메이션(보간 이동) 상태
        # 각 항목: {"model", "joint", "target", "speed"}  speed 단위/프레임당 = unit/s
        self._joint_animations = []
        self._last_anim_time = None
        self._inspection_pick_enabled = False
        self._inspection_pick_identity = None
        self._inspection_point = None
        self._inspection_marker = None
        self._chuck_mount_pick_enabled = False
        self._chuck_mount_pick_identity = None
        self._chuck_mount_points = []
        self._chuck_mount_local_points = []
        self._chuck_mount_markers = []
        self._ef_pose_actors = []
        self._ef_target_poses = {}
        self._inspection_path_actor = None
        self._last_inspection_path = None
        self._last_inspection_q_path = None
        self._last_inspection_edge_collisions = []
        self._last_inspection_robot = None
        self._robot_path_playback = None
        self._path_playback = None
        self._path_playback_marker = None
        self._collision_highlight_original_colors = {}

        self.loop_count = 0
        self.last_log_time = time.time()
        self.last_frame_time = time.time()
        self.target_frequency_hz = 60
        self.fps_text = None

        display_options = config.get("display_options", {})
        if display_options.get("show_fps", False):
            self.fps_text = vedo.Text2D("FPS: 0.0", pos='top-left', s=1.0, c="black", bg="white", alpha=0.5)
            self.plotter.add(self.fps_text)

        # Register key callback
        self.plotter.add_callback("KeyPress", self._on_key_press)
        self.plotter.add_callback("mouse click", self._on_mouse_click)

    def _setup_c_space(self, config: dict):
        """Add C-Space bounding box and axes to the plotter."""
        display_options = config.get("display_options", {})
        if not display_options.get("show_c_space", False):
            return

        self.c_bounds = config.get("c_space_bound", [5.0, 8.0, 5.0])
        c_bounds = self.c_bounds
        self.c_center = [c_bounds[0]/2, c_bounds[1]/2, c_bounds[2]/2]

        c_space_box = vedo.Box(
            pos=(c_bounds[0]/2, c_bounds[1]/2, c_bounds[2]/2),
            length=c_bounds[0], width=c_bounds[1], height=c_bounds[2]
        )
        c_space_box.wireframe().c('gray').alpha(0.3)

        # Create custom axes with 1-unit intervals
        x_ticks = [(i, str(i)) for i in range(int(c_bounds[0]) + 1)]
        y_ticks = [(i, str(i)) for i in range(int(c_bounds[1]) + 1)]
        z_ticks = [(i, str(i)) for i in range(int(c_bounds[2]) + 1)]

        axes_config = dict(
            xtitle='X', x_values_and_labels=x_ticks,
            ytitle='Y', y_values_and_labels=y_ticks,
            ztitle='Z', z_values_and_labels=z_ticks,
            c='black'
        )
        c_space_axes = vedo.Axes(c_space_box, **axes_config)
        self.plotter.add(c_space_box, c_space_axes)

    def _setup_robots(self, config: dict):
        """Load robot URDF models and add their meshes to the plotter."""
        self._robot_models = []
        root_path = config.get("root_path", "")
        for entry in config.get("urdf", []):
            import os
            name = entry.get("name", "unknown")
            path = entry.get("path", "")
            base = entry.get("base", [0, 0, 0, 0, 0, 0])
            full_path = os.path.join(str(root_path), path) if root_path else path
            if not os.path.exists(full_path):
                self.__console.error(f"[Robot] URDF file not found: {full_path}")
                continue
            model = RobotModel(name=name, urdf_path=full_path, base_pose=base)
            model.load()
            self._robot_models.append(model)

        all_actors = [a for m in self._robot_models for a in m.actors]
        if all_actors:
            self.plotter.add(*all_actors)
            self.__console.info(f"Added {len(all_actors)} robot mesh actors to plotter")

    def _on_key_press(self, event):
        """Handle key press events for camera control"""
        if not event.keypress:
            return

        key = event.keypress
        
        if not hasattr(self, 'c_bounds'):
            return

        # Direction vectors for each view (will be normalized internally)
        if key == '1': # XY Plane (Top View)
            self._set_camera_view((0, 0, 1), (0, 1, 0), "XY Plane (Top View)")
        elif key == '2': # YZ Plane (Side View)
            self._set_camera_view((1, 0, 0), (0, 0, 1), "YZ Plane (Side View)")
        elif key == '3': # XZ Plane (Front View)
            self._set_camera_view((0, -1, 0), (0, 0, 1), "XZ Plane (Front View)")
        elif key == '4': # Isometric View
            self._set_camera_view((1, 1, 1), (0, 0, 1), "Isometric View")

    def _set_camera_view(self, direction, view_up, label=None):
        """Set camera view from a direction vector, preserving current zoom level.
        
        Args:
            direction: (x, y, z) direction vector from focal point to camera
            view_up: (x, y, z) camera up-direction tuple
            label: optional log label for the view
        """
        if not hasattr(self, 'c_center'):
            return
        cx, cy, cz = self.c_center

        # Get current camera distance (zoom level) from focal point
        cam_pos = np.array(self.plotter.camera.GetPosition())
        focal = np.array([cx, cy, cz])
        current_dist = np.linalg.norm(cam_pos - focal)
        if current_dist < 1e-6:
            current_dist = max(self.c_bounds) * 2.0  # fallback

        # Normalize direction and apply current distance
        d = np.array(direction, dtype=float)
        d = d / np.linalg.norm(d)
        new_pos = focal + d * current_dist

        self.plotter.camera.SetPosition(*new_pos)
        self.plotter.camera.SetFocalPoint(cx, cy, cz)
        self.plotter.camera.SetViewUp(*view_up)
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()
        if label:
            self.__console.info(f"Camera set to {label}")

    def _on_mouse_click(self, event):
        """Pick points on the currently loaded pipe when a viewer pick mode is armed."""
        if getattr(self, '_chuck_mount_pick_enabled', False):
            self._handle_chuck_mount_pick(event)
            return

        if not getattr(self, '_inspection_pick_enabled', False):
            return
        pts = self._get_spool_points()
        if pts is None or len(pts) == 0:
            self.__console.warning("inspection pick: 로드된 스풀이 없습니다")
            return

        picked = getattr(event, "picked3d", None)
        if picked is None:
            self.__console.warning("inspection pick: pipe 표면 클릭 좌표를 얻지 못했습니다")
            return

        picked = np.asarray(picked, dtype=float)
        # PCD/mesh 모두에서 실제 pipe 점으로 스냅해 저장한다.
        idx = int(np.argmin(np.linalg.norm(pts - picked, axis=1)))
        point = np.asarray(pts[idx], dtype=float)
        self._set_inspection_point(point)
        self._inspection_pick_enabled = False

        identity = getattr(self, '_inspection_pick_identity', None)
        if hasattr(self, 'zapi') and self.zapi and identity:
            self.zapi.update_inspection_point({"point": point.tolist()}, identity=identity)
        self.__console.info(f"inspection point picked: {np.round(point, 4)}")

    def _set_inspection_point(self, point):
        self._inspection_point = np.asarray(point, dtype=float)
        self._clear_ef_pose_visuals()
        old = getattr(self, '_inspection_marker', None)
        if old is not None:
            self.plotter.remove(old)
        marker = vedo.Sphere(pos=self._inspection_point, r=0.045, c="tomato")
        marker.pickable(False)
        self._inspection_marker = marker
        self.plotter.add(marker)
        self.plotter.render()

    def _handle_chuck_mount_pick(self, event):
        pts = self._get_spool_points()
        if pts is None or len(pts) == 0:
            self.__console.warning("chuck mount pick: no loaded spool points")
            return

        picked = getattr(event, "picked3d", None)
        if picked is None:
            self.__console.warning("chuck mount pick: click a pipe surface point")
            return

        picked = np.asarray(picked, dtype=float)
        idx = int(np.argmin(np.linalg.norm(pts - picked, axis=1)))
        point = np.asarray(pts[idx], dtype=float)
        local_point = self._spool_world_to_local(point)
        self._add_chuck_mount_point(point, local_point)

        count = len(self._chuck_mount_points)
        if count < 2:
            self.__console.info("chuck mount point 1 picked; click the opposite chuck mount point")
            return

        self._chuck_mount_pick_enabled = False
        identity = getattr(self, '_chuck_mount_pick_identity', None)
        payload = self._get_chuck_mount_points_payload()
        if hasattr(self, 'zapi') and self.zapi and identity:
            self.zapi.update_chuck_mount_points(payload, identity=identity)
        self.__console.info(f"chuck mount points picked: {np.round(payload['points'], 4)}")

    def _spool_world_to_local(self, point):
        world_T = getattr(self, '_spool_world_T', None)
        if world_T is None:
            return None
        point_h = np.ones(4, dtype=float)
        point_h[:3] = np.asarray(point, dtype=float)
        local = np.linalg.inv(world_T) @ point_h
        return local[:3]

    def _get_chuck_mount_points_payload(self):
        payload = {"points": [np.asarray(p, dtype=float).tolist() for p in self._chuck_mount_points]}
        if len(self._chuck_mount_local_points) == len(self._chuck_mount_points):
            payload["local_points"] = [
                None if p is None else np.asarray(p, dtype=float).tolist()
                for p in self._chuck_mount_local_points
            ]
        return payload

    def _clear_chuck_mount_points(self):
        for marker in getattr(self, '_chuck_mount_markers', []):
            if marker is not None:
                self.plotter.remove(marker)
        self._chuck_mount_points = []
        self._chuck_mount_local_points = []
        self._chuck_mount_markers = []
        self.plotter.render()

    def _add_chuck_mount_point(self, point, local_point=None):
        point = np.asarray(point, dtype=float)
        colors = ("dodgerblue", "orange")
        marker = vedo.Sphere(
            pos=point,
            r=0.055,
            c=colors[len(self._chuck_mount_points) % len(colors)],
        )
        marker.pickable(False)
        self._chuck_mount_points.append(point)
        self._chuck_mount_local_points.append(None if local_point is None else np.asarray(local_point, dtype=float))
        self._chuck_mount_markers.append(marker)
        self.plotter.add(marker)
        self.plotter.render()

    def _set_chuck_mount_points(self, points, local_points=None):
        self._clear_chuck_mount_points()
        if not points:
            return
        for i, point in enumerate(points[:2]):
            local_point = None
            if local_points and i < len(local_points):
                local_point = local_points[i]
            self._add_chuck_mount_point(point, local_point)

    def _clear_inspection_visuals(self, clear_point=True):
        if getattr(self, '_inspection_path_actor', None) is not None:
            self.plotter.remove(self._inspection_path_actor)
            self._inspection_path_actor = None
        if clear_point:
            self._clear_ef_pose_visuals()
        if clear_point and getattr(self, '_inspection_marker', None) is not None:
            self.plotter.remove(self._inspection_marker)
            self._inspection_marker = None
            self._inspection_point = None
        self.plotter.render()

    def _clear_ef_pose_visuals(self, clear_poses=True):
        for actor in getattr(self, '_ef_pose_actors', []) or []:
            try:
                self.plotter.remove(actor)
            except Exception:
                pass
        self._ef_pose_actors = []
        if clear_poses:
            self._ef_target_poses = {}

    def _robot_target_link_name(self, robot_name):
        if robot_name == "rb20_1900es":
            return "rt_link_end"
        if robot_name == "dda_rb10_1300e":
            return "dda_link_end"
        model = self._find_robot(robot_name)
        if model is not None and model._urdf is not None:
            for preferred in ("link_end", "end"):
                for link in model._urdf.links:
                    lname = getattr(link, "name", "")
                    if preferred in lname.lower() and "tcp" not in lname.lower():
                        return lname
        return self._robot_tcp_link_name(robot_name)

    def _robot_tcp_link_name(self, robot_name):
        if robot_name == "rb20_1900es":
            return "rt_tcp"
        if robot_name == "dda_rb10_1300e":
            return "dda_link_tcp"
        model = self._find_robot(robot_name)
        if model is not None and model._urdf is not None:
            for link in model._urdf.links:
                lname = getattr(link, "name", "")
                if "tcp" in lname.lower():
                    return lname
        return None

    def _get_robot_target_pose(self, robot_name):
        model = self._find_robot(robot_name)
        if model is None:
            return None
        link_name = self._robot_target_link_name(robot_name)
        if link_name is None:
            return None
        T = model.get_link_world_T(link_name)
        if T is None:
            return None
        pose = np.zeros(6, dtype=float)
        pose[:3] = T[:3, 3]
        return pose

    def _get_robot_tcp_pose(self, robot_name):
        return self._get_robot_target_pose(robot_name)

    def _rpy_matrix(self, rpy):
        roll, pitch, yaw = [float(v) for v in rpy]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
        return rz @ ry @ rx

    def _pose_to_T(self, pose):
        pose = np.asarray(pose, dtype=float)
        T = np.eye(4)
        T[:3, 3] = pose[:3]
        if pose.shape[0] >= 6:
            T[:3, :3] = self._rpy_matrix(pose[3:6])
        return T

    def _pose_frame_actors(self, pose, scale=0.18, axes=(0, 1, 2), show_origin=True):
        T = self._pose_to_T(pose)
        origin = T[:3, 3]
        actors = []
        for axis, color in ((0, "red"), (1, "green"), (2, "blue")):
            if axis not in axes:
                continue
            actor = vedo.Arrow(origin, origin + T[:3, axis] * scale, s=0.0008, c=color)
            actor.alpha(0.35)
            actor.pickable(False)
            actors.append(actor)
        if show_origin:
            marker = vedo.Sphere(pos=origin, r=scale * 0.055, c="yellow")
            marker.alpha(0.35)
            marker.pickable(False)
            actors.append(marker)
        return actors

    def _ef_pose_robot_name(self, pose_name):
        return "dda_rb10_1300e" if pose_name == "DDA" else "rb20_1900es"

    def _ef_pose_tcp_to_link_T(self, robot_name):
        model = self._find_robot(robot_name)
        if model is None or getattr(model, "_urdf", None) is None:
            return np.eye(4)
        joint_name = "dda_joint_tcp" if robot_name == "dda_rb10_1300e" else "rt_joint_tcp"
        joint = getattr(model._urdf, "joint_map", {}).get(joint_name)
        if joint is None or getattr(joint, "origin", None) is None:
            return np.eye(4)
        origin = joint.origin
        if isinstance(origin, np.ndarray):
            T = np.asarray(origin, dtype=float)
            if T.shape == (4, 4):
                return np.linalg.inv(T)
        T = np.eye(4)
        T[:3, :3] = self._rpy_matrix(origin.rpy)
        T[:3, 3] = origin.xyz
        return np.linalg.inv(T)

    def _ef_pose_mesh_actors(self, pose_name, pose):
        robot_name = self._ef_pose_robot_name(pose_name)
        model = self._find_robot(robot_name)
        if model is None:
            return []
        link_name = self._robot_target_link_name(robot_name)
        mesh_list = getattr(model, "_link_mesh_data", {}).get(link_name, [])
        if not mesh_list:
            self.__console.warning(f"EF pose mesh unavailable: robot={robot_name}, link={link_name}")
            return []
        T = self._pose_to_T(pose) @ self._ef_pose_tcp_to_link_T(robot_name)
        color = "gold" if pose_name == "DDA" else "deepskyblue"
        actors = []
        for local_verts, faces in mesh_list:
            verts = (T[:3, :3] @ np.asarray(local_verts, dtype=float).T).T + T[:3, 3]
            actor = vedo.Mesh([verts, np.asarray(faces, dtype=np.int32)])
            actor.c(color).alpha(0.28)
            actor.pickable(False)
            actors.append(actor)
        return actors

    def _show_ef_target_poses(self, poses):
        self._clear_ef_pose_visuals(clear_poses=False)
        actors = []
        for name, pose in poses.items():
            scale = 0.22 if name.startswith("RT") else 0.18
            actors.extend(self._ef_pose_mesh_actors(name, pose))
            actors.extend(self._pose_frame_actors(pose, scale=scale, axes=(0, 1, 2), show_origin=False))
        self._ef_pose_actors = actors
        if actors:
            self.plotter.add(*actors)
            self.plotter.render()

    def _candidate_x_axis_actors(self, candidate_poses, scale=0.12):
        actors = []
        if candidate_poses is None:
            return actors
        for pose in list(candidate_poses):
            T = self._pose_to_T(pose)
            origin = T[:3, 3]
            actor = vedo.Arrow(origin, origin + T[:3, 0] * scale, s=0.0005, c="red")
            actor.alpha(0.18)
            actor.pickable(False)
            actors.append(actor)
        return actors

    def _show_ef_pose_candidates(self, candidate_poses):
        actors = self._candidate_x_axis_actors(candidate_poses)
        self._ef_pose_actors.extend(actors)
        if actors:
            self.plotter.add(*actors)
            self.plotter.render()

    def _pose_determinator_point_cloud(self):
        pts = self._get_spool_points()
        if pts is None or len(pts) < 10:
            raise RuntimeError("loaded spool point cloud is not available")
        pcd = _o3d.geometry.PointCloud()
        pcd.points = _o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
        pcd.estimate_normals(
            search_param=_o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
        )
        pcd.normalize_normals()
        return pcd

    def _extract_ef_poses(self, pose_groups):
        if not pose_groups:
            raise RuntimeError("poseDeterminator returned no valid pose group")
        group = pose_groups[0]
        orientation_group = group.get("0") or group.get("90") or next(iter(group.values()), None)
        if not orientation_group:
            raise RuntimeError("pose group does not contain DDA/RT poses")
        poses = {}
        if orientation_group.get("DDA") is not None:
            poses["DDA"] = np.asarray(orientation_group["DDA"], dtype=float)
        if orientation_group.get("RT1") is not None:
            poses["RT1"] = np.asarray(orientation_group["RT1"], dtype=float)
        elif orientation_group.get("RT2") is not None:
            poses["RT2"] = np.asarray(orientation_group["RT2"], dtype=float)
        if not poses:
            raise RuntimeError("pose group is empty")
        return poses

    def _determine_ef_pose(self, request_data):
        identity = request_data.get("_identity")
        result = {"status": "failed"}
        try:
            target = getattr(self, '_inspection_point', None)
            if target is None:
                raise RuntimeError("inspection point is not selected")

            pose_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "plugins", "poseDeterminator")
            )
            if pose_dir not in sys.path:
                sys.path.insert(0, pose_dir)
            from EndEffectorPoseOptimizer import EndEffectorPoseOptimizer

            root_path = os.path.abspath(str(self._config.get("root_path", os.getcwd())))
            optimizer = EndEffectorPoseOptimizer(debug_mode=True)
            optimizer._scan_data = self._pose_determinator_point_cloud()
            optimizer.load_DDA_from_urdf(os.path.join(root_path, "urdf", "rb10_1300e_DDA.urdf"))
            optimizer.load_RT_from_urdf(os.path.join(root_path, "urdf", "rb10_1300e_RT.urdf"))

            params = self._config.get("ef_pose", {}) or {}
            target = np.asarray(target, dtype=float)
            optimizer.calculate_pipe_profile(
                target,
                sampling_size_for_calculating_normal=float(
                    params.get("sampling_size_for_calculating_normal", 0.01)),
                radius_offset_for_sampling_points_in_sphere=float(
                    params.get("radius_offset_for_sampling_points_in_sphere", 0.003)),
            )
            _, pose_groups = optimizer.calculate_DDA_RT_pose_for_taking_xray(
                target,
                num_candidates=int(params.get("num_candidates", 8)),
                distance_from_dda_to_surface=float(params.get("distance_from_dda_to_surface", 0.01)),
                distance_from_dda_to_rt=float(params.get("distance_from_dda_to_rt", 0.3)),
                angle_of_rt=float(params.get("angle_of_rt", 10.0)),
            )
            poses = self._extract_ef_poses(pose_groups)
            self._ef_target_poses = poses
            self._show_ef_target_poses(poses)
            debug_info = getattr(optimizer, "debuging_info", {}) or {}
            candidate_poses = debug_info.get("valid_base_dda_poses")
            if candidate_poses is None or len(candidate_poses) == 0:
                candidate_poses = debug_info.get("dda_base_candidates", [])
            self._show_ef_pose_candidates(candidate_poses)
            result = {
                "status": "success",
                "poses": {name: pose.tolist() for name, pose in poses.items()},
                "candidate_count": len(candidate_poses),
            }
            self.__console.info(f"EF pose determined: {list(poses.keys())}")
        except Exception as exc:
            result = {"status": "failed", "message": str(exc)}
            self.__console.error(f"EF pose determination failed: {exc}")
        if hasattr(self, 'zapi') and self.zapi:
            self.zapi.reply_ef_pose(result, identity=identity)

    def _load_path_planner(self, module_name):
        from plugins.pluginbase.plannerbase import PlannerBase
        module = importlib.import_module(f"plugins.pathplanner.{module_name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, PlannerBase) and obj is not PlannerBase:
                return obj()
        raise RuntimeError(f"Planner plugin class not found: {module_name}")

    def _inspection_q_space_planner_name(self, planner_name):
        q_space_planners = {"rrt_connect", "rrt_star"}
        planner_name = str(planner_name or "rrt_connect")
        if planner_name in q_space_planners:
            return planner_name
        self.__console.warning(
            f"inspection path: planner '{planner_name}' is task-space or does not support "
            "q-space Pinocchio collision; using rrt_connect")
        return "rrt_connect"

    def _current_spool_collision_mesh(self):
        """Build an Open3D mesh from the currently rendered pipe. Positioner/pipe are static here."""
        spool = getattr(self, '_loaded_spool_mesh', None)
        actors = spool if isinstance(spool, (list, tuple)) else [spool]
        mesh_actor = next(
            (actor for actor in actors
             if actor is not None
             and hasattr(actor, "vertices")
             and hasattr(actor, "cells")
             and len(getattr(actor, "cells", [])) > 0),
            None)
        if mesh_actor is not None:
            mesh = _o3d.geometry.TriangleMesh()
            mesh.vertices = _o3d.utility.Vector3dVector(np.asarray(mesh_actor.vertices, dtype=float))
            mesh.triangles = _o3d.utility.Vector3iVector(np.asarray(mesh_actor.cells, dtype=np.int32))
            mesh.compute_vertex_normals()
            return mesh

        pts = self._get_spool_points()
        if pts is None or len(pts) < 4:
            return None
        pts = self._spool_collision_points(pts)
        pcd = _o3d.geometry.PointCloud()
        pcd.points = _o3d.utility.Vector3dVector(np.asarray(pts, dtype=float))
        try:
            pcd.remove_non_finite_points()
            pcd.remove_duplicated_points()
        except Exception:
            pass
        try:
            # Alpha-shape often emits many "invalid tetra" warnings for noisy or
            # nearly co-planar pipe PCDs. They are not actionable for this
            # EF-only collision mesh, so suppress Open3D warning spam here.
            with _o3d.utility.VerbosityContextManager(_o3d.utility.VerbosityLevel.Error):
                mesh = _o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.06)
            if mesh.has_triangles():
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_unreferenced_vertices()
                mesh.compute_vertex_normals()
                return mesh
        except Exception as exc:
            self.__console.warning(
                "inspection path: alpha mesh failed; using AABB collision fallback "
                f"({self._short_exception(exc)})")
        mesh = self._aabb_collision_mesh(pcd)
        mesh.compute_vertex_normals()
        return mesh

    def _spool_collision_points(self, pts):
        pts = np.asarray(pts, dtype=float)
        load_cfg = self._config.get("spool_load", {}) or {}
        max_points = int(load_cfg.get("collision_max_points", 100000))
        if max_points <= 0 or len(pts) <= max_points:
            return pts
        step = int(np.ceil(len(pts) / max_points))
        reduced = pts[::step]
        self.__console.info(
            f"inspection path: collision point cloud downsampled "
            f"{len(pts)} -> {len(reduced)} points")
        return reduced

    def _aabb_collision_mesh(self, pcd):
        bbox = pcd.get_axis_aligned_bounding_box()
        mn = np.asarray(bbox.get_min_bound(), dtype=float)
        mx = np.asarray(bbox.get_max_bound(), dtype=float)
        ext = np.maximum(mx - mn, 0.01)
        pad = np.maximum(ext * 0.01, 0.005)
        mn = mn - pad
        ext = ext + 2.0 * pad
        mesh = _o3d.geometry.TriangleMesh.create_box(
            width=float(ext[0]),
            height=float(ext[1]),
            depth=float(ext[2]),
        )
        mesh.translate(mn)
        return mesh

    def _short_exception(self, exc, limit=220):
        msg = str(exc).replace("\r", " ").replace("\n", " ")
        if len(msg) > limit:
            return msg[:limit] + "..."
        return msg

    def _configure_inspection_planner(self, planner, obstacle_mesh, start, goal, step_size, max_iter, robot_name=None):
        mn = np.minimum(obstacle_mesh.get_min_bound(), np.minimum(start[:3], goal[:3]))
        mx = np.maximum(obstacle_mesh.get_max_bound(), np.maximum(start[:3], goal[:3]))
        ext = np.maximum(mx - mn, 1e-6)
        pad = np.maximum(ext * 0.5, 0.5)
        bounds = {
            "x_min": float(mn[0] - pad[0]), "x_max": float(mx[0] + pad[0]),
            "y_min": float(mn[1] - pad[1]), "y_max": float(mx[1] + pad[1]),
            "z_min": float(mn[2] - pad[2]), "z_max": float(mx[2] + pad[2]),
            "roll_min": -np.pi, "roll_max": np.pi,
            "pitch_min": -np.pi, "pitch_max": np.pi,
            "yaw_min": -np.pi, "yaw_max": np.pi,
        }
        if hasattr(planner, "bounds"):
            planner.bounds = bounds
        if hasattr(planner, "step_size"):
            planner.step_size = float(step_size)
        if hasattr(planner, "max_iter"):
            planner.max_iter = int(max_iter)
        if hasattr(planner, "max_iterations"):
            planner.max_iterations = int(max_iter)
        if hasattr(planner, "pin_collision_sample_resolution"):
            planner.pin_collision_sample_resolution = float(step_size)
        if robot_name is not None and hasattr(planner, "setup_pinocchio_collision"):
            model = self._find_robot(robot_name)
            urdf_path = getattr(model, "urdf_path", None) if model is not None else None
            if urdf_path:
                try:
                    planner.setup_pinocchio_collision(
                        urdf_path,
                        package_dirs=[os.path.dirname(urdf_path)],
                    )
                except Exception as exc:
                    self.__console.warning(f"inspection path: Pinocchio collision setup failed ({exc})")
        planner.add_collision_object(obstacle_mesh)
        return bounds

    def _probe_current_spool_pinocchio_collision(self, reason="spool"):
        """Add the currently loaded spool mesh to Pinocchio and check current robot q."""
        probe_enabled = bool(
            self._config.get("probe_collision_on_spool_update", False)
            or (self._config.get("spool_load", {}) or {}).get("probe_collision_on_update", False)
        )
        if not probe_enabled:
            self.__console.debug(f"{reason}: spool collision probe skipped")
            return []

        obstacle_mesh = self._current_spool_collision_mesh()
        if obstacle_mesh is None or not obstacle_mesh.has_triangles():
            self.__console.warning(f"{reason}: spool collision mesh is not available")
            return []

        from plugins.pluginbase.plannerbase import PlannerBase

        class _CollisionProbe(PlannerBase):
            def generate(self, current_pose, target_pose, step_callback=None):
                return []

        results = []
        for model in getattr(self, '_robot_models', []):
            urdf_path = getattr(model, "urdf_path", None)
            if not urdf_path:
                continue
            try:
                probe = _CollisionProbe()
                probe.setup_pinocchio_collision(
                    urdf_path,
                    package_dirs=[os.path.dirname(urdf_path)],
                )
                geom_id = probe.add_collision_object(obstacle_mesh)
                if geom_id is None:
                    self.__console.warning(
                        f"{reason}: failed to add spool mesh to Pinocchio for {model.name}")
                    continue
                q = self._current_robot_q(model, probe.pin_model)
                hit, pairs = probe.check_pinocchio_collision(q, return_pairs=True)
                result = {
                    "robot": getattr(model, "name", ""),
                    "object_geom_id": geom_id,
                    "collision": bool(hit),
                    "pairs": [list(pair) for pair in pairs],
                }
                results.append(result)
                if hit:
                    pair_text = ", ".join(f"{a} <-> {b}" for a, b in pairs)
                    self.__console.warning(
                        f"{reason}: current robot collision detected for {model.name}: {pair_text}")
                else:
                    self.__console.info(
                        f"{reason}: spool mesh added to Pinocchio for {model.name} "
                        f"(object_geom_id={geom_id}), no collision at current q")
            except Exception as exc:
                self.__console.warning(
                    f"{reason}: Pinocchio spool collision probe failed for "
                    f"{getattr(model, 'name', 'robot')} ({exc})")
        self._last_spool_collision_probe = results
        return results

    def _pin_joint_names(self, pin_model):
        return [str(name) for name in list(pin_model.names)[1:1 + pin_model.nq]]

    def _current_robot_q(self, model, pin_model):
        q = np.zeros(pin_model.nq, dtype=float)
        for i, joint_name in enumerate(self._pin_joint_names(pin_model)):
            q[i] = float(model._joint_cfg.get(joint_name, 0.0))
        return q

    def _apply_robot_q(self, model, pin_model, q):
        for i, joint_name in enumerate(self._pin_joint_names(pin_model)):
            model.set_joint(joint_name, float(q[i]))
        model.update_fk()

    def _pin_target_frame_id(self, pin_model, robot_name):
        link_name = self._robot_target_link_name(robot_name)
        if link_name:
            try:
                fid = pin_model.getFrameId(link_name)
                if fid < pin_model.nframes:
                    return fid
            except Exception:
                pass
        return pin_model.nframes - 1

    def _pin_target_world_T(self, model, pin_model, q, robot_name):
        if pin is None:
            return None
        data = pin_model.createData()
        pin.forwardKinematics(pin_model, data, q)
        pin.updateFramePlacements(pin_model, data)
        fid = self._pin_target_frame_id(pin_model, robot_name)
        local_T = data.oMf[fid].homogeneous
        return model._base_T @ local_T

    def _pin_tcp_world_T(self, model, pin_model, q, robot_name):
        return self._pin_target_world_T(model, pin_model, q, robot_name)

    def _solve_inspection_ik(self, model, planner, robot_name, target_world, q_init):
        if pin is None or planner.pin_model is None:
            return None

        pin_model = planner.pin_model
        data = pin_model.createData()
        fid = self._pin_target_frame_id(pin_model, robot_name)

        current_world_T = self._pin_target_world_T(model, pin_model, q_init, robot_name)
        if current_world_T is None:
            return None

        target_world_T = current_world_T.copy()
        target_world_T[:3, 3] = np.asarray(target_world, dtype=float)
        target_local_T = np.linalg.inv(model._base_T) @ target_world_T
        target_se3 = pin.SE3(target_local_T[:3, :3], target_local_T[:3, 3])

        q = np.asarray(q_init, dtype=float).copy()
        damping = 1e-3
        dt = 0.35
        tol = 1e-4
        max_iter = 600

        for _ in range(max_iter):
            pin.forwardKinematics(pin_model, data, q)
            pin.updateFramePlacements(pin_model, data)
            err = pin.log6(data.oMf[fid].inverse() * target_se3).vector
            if np.linalg.norm(err) < tol:
                return q
            J = pin.computeFrameJacobian(pin_model, data, q, fid, pin.ReferenceFrame.LOCAL)
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + damping * np.eye(6), err)
            q = pin.integrate(pin_model, q, dt * dq)
            q = np.minimum(np.maximum(q, pin_model.lowerPositionLimit), pin_model.upperPositionLimit)

        final_T = self._pin_target_world_T(model, pin_model, q, robot_name)
        final_err = np.linalg.norm(final_T[:3, 3] - np.asarray(target_world, dtype=float)) if final_T is not None else float("inf")
        if final_err < 0.01:
            return q
        self.__console.warning(f"inspection IK failed: position error={final_err:.4f} m")
        return None

    def _q_path_to_target_poses(self, model, pin_model, robot_name, q_path, sample_resolution=0.03):
        poses = []
        q_pts = [np.asarray(q, dtype=float) for q in q_path]
        if not q_pts:
            return poses
        resolution = max(float(sample_resolution), 1e-6)
        if len(q_pts) == 1:
            samples = q_pts
        else:
            samples = []
            for edge_idx, (qa, qb) in enumerate(zip(q_pts[:-1], q_pts[1:])):
                steps = max(1, int(np.ceil(np.linalg.norm(qb - qa) / resolution)))
                for step in range(steps + 1):
                    if edge_idx > 0 and step == 0:
                        continue
                    ratio = step / steps
                    samples.append(qa * (1.0 - ratio) + qb * ratio)
        for q in samples:
            T = self._pin_target_world_T(model, pin_model, q, robot_name)
            if T is not None:
                pose = np.zeros(6, dtype=float)
                pose[:3] = T[:3, 3]
                poses.append(pose)
        return poses

    def _q_path_to_tcp_poses(self, model, pin_model, robot_name, q_path, sample_resolution=0.03):
        return self._q_path_to_target_poses(model, pin_model, robot_name, q_path, sample_resolution)

    def _verify_planned_path(self, planner, path):
        colliding_edges = 0
        collision_pairs = []
        edge_collisions = []
        seen_pairs = set()
        poses = [np.asarray(p, dtype=float) for p in path]
        for edge_idx, (a_pose, b_pose) in enumerate(zip(poses[:-1], poses[1:])):
            pairs = (planner.collision_pairs_along_edge(a_pose, b_pose)
                     if hasattr(planner, "collision_pairs_along_edge") else [])
            if pairs or planner._check_collision(a_pose, b_pose):
                colliding_edges += 1
                edge_collisions.append({
                    "edge": edge_idx,
                    "from_waypoint": edge_idx,
                    "to_waypoint": edge_idx + 1,
                    "pairs": [list(pair) for pair in pairs],
                })
            for pair in pairs:
                key = tuple(pair)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                collision_pairs.append(list(pair))
        return {
            "colliding_edges": colliding_edges,
            "collision_pairs": collision_pairs,
            "edge_collisions": edge_collisions,
            "end_link_colliding": any(
                "link_end" in name
                for pair in collision_pairs
                for name in pair),
            "backend": "pinocchio" if getattr(planner, "pin_model", None) is not None else "none",
        }

    def _show_inspection_path(self, path):
        self._clear_inspection_visuals(clear_point=False)
        pts = np.asarray([np.asarray(p, dtype=float)[:3] for p in path], dtype=float)
        if len(pts) < 2:
            return
        actor = vedo.Line(pts).c("limegreen").lw(5)
        actor.pickable(False)
        self._inspection_path_actor = actor
        self.plotter.add(actor)
        self.plotter.render()

    def _plan_inspection_path(self, request_data):
        identity = request_data.get("_identity")
        result = {"status": "failed"}
        try:
            target = getattr(self, '_inspection_point', None)
            if target is None:
                raise RuntimeError("inspection point is not selected")
            obstacle_mesh = self._current_spool_collision_mesh()
            if obstacle_mesh is None:
                raise RuntimeError("loaded pipe is not available")
            robot_name = request_data.get("robot", "rb20_1900es")
            start = self._get_robot_tcp_pose(robot_name)
            if start is None:
                raise RuntimeError(f"robot TCP not found: {robot_name}")
            goal = np.zeros(6, dtype=float)
            goal[:3] = np.asarray(target, dtype=float)
            robot_model = self._find_robot(robot_name)
            if robot_model is None:
                raise RuntimeError(f"robot model not found: {robot_name}")

            planner_name = self._inspection_q_space_planner_name(request_data.get("planner", "rrt_connect"))
            planner = self._load_path_planner(planner_name)
            self._configure_inspection_planner(
                planner,
                obstacle_mesh,
                start,
                goal,
                float(request_data.get("step_size", 0.08)),
                int(request_data.get("max_iter", 3000)),
                robot_name=robot_name)
            if getattr(planner, "pin_model", None) is None:
                raise RuntimeError("Pinocchio collision model is not configured")

            start_q = self._current_robot_q(robot_model, planner.pin_model)
            goal_q = self._solve_inspection_ik(robot_model, planner, robot_name, target, start_q)
            if goal_q is None:
                raise RuntimeError("failed to solve robot IK for inspection point")

            t0 = time.time()
            q_path = planner.generate(start_q, goal_q)
            elapsed = time.time() - t0
            forced_collision_preview = False
            if not q_path:
                q_path = [start_q, goal_q]
                forced_collision_preview = True
                self.__console.warning(
                    f"inspection path: no collision-free path found ({elapsed:.2f}s); "
                    "using direct q path for collision preview")
            verification = self._verify_planned_path(planner, q_path)
            if verification["colliding_edges"] != 0:
                forced_collision_preview = True
                self.__console.warning(
                    f"inspection path collision detected; keeping path for preview: {verification}")
            display_resolution = float(request_data.get("display_step_size", request_data.get("step_size", 0.08)))
            path = self._q_path_to_tcp_poses(
                robot_model,
                planner.pin_model,
                robot_name,
                q_path,
                sample_resolution=display_resolution)
            if len(path) < 2:
                raise RuntimeError("planned q path could not be converted to TCP path")
            self._last_inspection_q_path = [np.asarray(q, dtype=float) for q in q_path]
            self._last_inspection_edge_collisions = verification.get("edge_collisions", [])
            self._last_inspection_robot = robot_name
            self._last_inspection_path = [np.asarray(p, dtype=float) for p in path]
            self._show_inspection_path(path)
            result = {
                "status": "success",
                "planner": planner_name,
                "robot": robot_name,
                "waypoints": len(q_path),
                "elapsed": elapsed,
                "start": start.tolist(),
                "goal": goal.tolist(),
                "verification": verification,
                "robot_links_considered": True,
                "collision_preview": forced_collision_preview,
            }
            if forced_collision_preview:
                self.__console.warning(
                    f"inspection q path kept for collision preview: {planner_name}, "
                    f"{len(q_path)} waypoints, {elapsed:.2f}s")
            else:
                self.__console.info(
                    f"inspection q path OK: {planner_name}, {len(q_path)} waypoints, {elapsed:.2f}s")
        except Exception as e:
            result = {"status": "failed", "message": str(e)}
            self.__console.error(f"inspection path failed: {e}")
        if hasattr(self, 'zapi') and self.zapi and identity:
            self.zapi.reply_inspection_path(result, identity=identity)

    def run(self, frequency_hz: int):
        self.target_frequency_hz = frequency_hz
        self.__console.debug(f"Starting Vedo GUI loop (target: {frequency_hz} Hz)")
        
        # shape initial view to 3D perspective if C-Space is defined
        if hasattr(self, 'c_bounds'):
             max_dim = max(self.c_bounds)
             self.plotter.show(interactive=False)
             if self.plotter.camera:
                 # Set initial distance, then apply isometric direction
                 cx, cy, cz = self.c_center
                 init_dist = max_dim * 2.0
                 iso_d = init_dist / np.sqrt(3)
                 self.plotter.camera.SetPosition(cx + iso_d, cy + iso_d, cz + iso_d)
                 self.plotter.camera.SetFocalPoint(cx, cy, cz)
                 self._set_camera_view((1, 1, 1), (0, 0, 1))
        else:
             self.plotter.show(interactive=False)
        
        while not self._should_close:
            if not self.plotter.interactor or self.plotter.interactor.GetDone():
                break
            start_time = time.time()
            
            # Logic step
            if not self._on_tick():
                break
                
            # Render step
            self.plotter.render()
            
            # Event processing (interactor)
            if self.plotter.interactor:
                self.plotter.interactor.ProcessEvents()
            
            # Timing control
            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.target_frequency_hz) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        self.on_close()
        self.plotter.close()
        self.__console.info("Visualizer closed")
    
    def _on_tick(self) -> bool:
        """Called every frame by the GUI event loop."""
        
        # 1. Log Frequency
        self.loop_count, self.last_log_time = self._log_rendering_frequency(self.loop_count, self.last_log_time)
        
        # 2. Process requests from ZApi queue
        processed_count = 0
        while processed_count < 10:
            with self._queue_lock:
                if not self._request_queue:
                    break
                request_data = self._request_queue.popleft()
            
            self._process_request(request_data)
            processed_count += 1

        # 3. Step manipulator joint animations (보간 이동)
        now = time.time()
        dt = 0.0 if self._last_anim_time is None else (now - self._last_anim_time)
        self._last_anim_time = now
        if self._joint_animations and dt > 0:
            self._step_joint_animations(min(dt, 0.1))   # 큰 dt는 클램프
        if (getattr(self, '_path_playback', None) is not None
                or getattr(self, '_robot_path_playback', None) is not None) and dt > 0:
            self._step_path_playback(min(dt, 0.1))

        return True

    def _find_robot(self, name):
        for m in getattr(self, '_robot_models', []):
            if getattr(m, 'name', None) == name:
                return m
        return None

    def _step_joint_animations(self, dt):
        """활성 조인트 애니메이션을 사다리꼴 속도 프로파일로 한 스텝 진행.
        가속(accel)으로 max_speed까지 올린 뒤 순항, target 도달 전 감속해 정지.
        """
        still = []
        for anim in self._joint_animations:
            model = anim["model"]; jn = anim["joint"]
            tgt = float(anim["target"])
            vmax = max(float(anim["speed"]), 1e-6)
            accel = max(float(anim["accel"]), 1e-6)
            cur = float(model._joint_cfg.get(jn, 0.0))
            vel = float(anim.get("vel", 0.0))

            d_rem = tgt - cur
            dist = abs(d_rem)
            direction = np.sign(d_rem) if d_rem != 0 else 0.0

            # 정지 임박: 남은 거리·속도가 충분히 작으면 스냅
            if dist <= 1e-6 and vel <= accel * dt:
                model.set_joint(jn, tgt); model.update_fk()
                continue

            # 감속에 필요한 거리 = v² / (2a). 그보다 가까우면 감속, 아니면 가속/순항
            stop_dist = (vel * vel) / (2.0 * accel)
            if dist <= stop_dist:
                vel = max(0.0, vel - accel * dt)      # 감속
            else:
                vel = min(vmax, vel + accel * dt)     # 가속 후 vmax 순항

            new_cur = cur + direction * vel * dt
            # target을 지나치면 스냅하고 종료
            if (tgt - new_cur) * direction <= 0:
                model.set_joint(jn, tgt); model.update_fk()
                continue

            anim["vel"] = vel
            model.set_joint(jn, new_cur); model.update_fk()
            still.append(anim)
        self._joint_animations = still

    def _set_joint_animation(self, robot_name, joint_name, target, speed, accel=None):
        """해당 로봇/조인트의 기존 애니메이션을 교체하고 사다리꼴 프로파일로 이동 시작.
        accel 미지정 시 speed의 2배(약 0.5s 가속)로 기본 설정.
        """
        model = self._find_robot(robot_name)
        if model is None or model._urdf is None:
            self.__console.warning(f"move_manipulator: 로봇 없음 '{robot_name}'")
            return
        if joint_name not in model._urdf._joint_map:
            self.__console.warning(f"move_manipulator: 조인트 없음 '{joint_name}'")
            return
        spd = float(speed)
        acc = float(accel) if accel is not None else max(spd * 2.0, 1e-6)
        # 같은 (robot, joint)의 현재 속도는 이어받아 부드럽게 재타게팅
        prev_vel = 0.0
        for a in self._joint_animations:
            if a["model"] is model and a["joint"] == joint_name:
                prev_vel = a.get("vel", 0.0)
        self._joint_animations = [
            a for a in self._joint_animations
            if not (a["model"] is model and a["joint"] == joint_name)
        ]
        self._joint_animations.append({
            "model": model, "joint": joint_name,
            "target": float(target), "speed": spd, "accel": acc, "vel": prev_vel,
        })
        self.__console.info(
            f"move_manipulator: {robot_name}.{joint_name} → {target} (vmax={spd}, accel={acc})")

    def _stop_joint_animation(self, robot_name, joint_name=None):
        """해당 로봇(또는 특정 조인트)의 애니메이션을 즉시 중지."""
        model = self._find_robot(robot_name)
        self._joint_animations = [
            a for a in self._joint_animations
            if not (a["model"] is model and (joint_name is None or a["joint"] == joint_name))
        ]
        self.__console.info(f"stop_manipulator: {robot_name} {joint_name or '(all)'}")

    def _start_path_playback(self, speed=0.2):
        """Replay the last planned inspection q path by moving the robot model."""
        q_path = getattr(self, '_last_inspection_q_path', None)
        robot_name = getattr(self, '_last_inspection_robot', None)
        model = self._find_robot(robot_name) if robot_name else None
        if q_path is None or len(q_path) < 2 or model is None:
            self.__console.warning("execute_inspection_path: planned path가 없습니다")
            return False

        q_pts = np.asarray([np.asarray(q, dtype=float) for q in q_path], dtype=float)
        seg_lengths = np.linalg.norm(np.diff(q_pts, axis=0), axis=1)
        if not np.any(seg_lengths > 1e-9):
            self.__console.warning("execute_inspection_path: path 길이가 0입니다")
            return False

        pin_model = self._build_pin_model_for_robot(model)
        if pin_model is None:
            self.__console.warning("execute_inspection_path: Pinocchio model 생성 실패")
            return False

        self._clear_collision_highlights()
        path = getattr(self, '_last_inspection_path', None)
        pts = np.asarray([np.asarray(p, dtype=float)[:3] for p in path], dtype=float) if path else None
        old = getattr(self, '_path_playback_marker', None)
        if old is not None:
            self.plotter.remove(old)
        if pts is not None and len(pts) > 0:
            marker = vedo.Sphere(pos=pts[0], r=0.055, c="dodgerblue")
            marker.pickable(False)
            self._path_playback_marker = marker
            self.plotter.add(marker)
        else:
            self._path_playback_marker = None

        self._robot_path_playback = {
            "model": model,
            "pin_model": pin_model,
            "robot_name": robot_name,
            "q_points": q_pts,
            "seg_lengths": seg_lengths,
            "seg_idx": 0,
            "seg_s": 0.0,
            "speed": max(float(speed), 1e-6),
            "edge_collisions": {
                int(item.get("edge", -1)): item.get("pairs", [])
                for item in getattr(self, '_last_inspection_edge_collisions', [])
            },
            "logged_collision_edges": set(),
        }
        if self._robot_path_playback["edge_collisions"]:
            edges = sorted(self._robot_path_playback["edge_collisions"].keys())
            self.__console.warning(
                f"execute_inspection_path: collision detected between waypoints {edges}")
            self._log_path_playback_collision(self._robot_path_playback, 0)
        self._path_playback = None
        self._apply_robot_q(model, pin_model, q_pts[0])
        self.plotter.render()
        self.__console.info(f"execute_inspection_path: robot playback started ({len(q_pts)} waypoints)")
        return True

    def _log_path_playback_collision(self, playback, edge_idx):
        edge_collisions = playback.get("edge_collisions", {})
        logged = playback.get("logged_collision_edges", set())
        if edge_idx not in edge_collisions or edge_idx in logged:
            return
        pairs = edge_collisions.get(edge_idx) or []
        self._highlight_collision_pairs(pairs)
        pair_text = ", ".join(f"{a} <-> {b}" for a, b in pairs) if pairs else "unknown pair"
        self.__console.warning(
            "execute_inspection_path: collision between "
            f"waypoint {edge_idx} -> {edge_idx + 1} ({pair_text})")
        logged.add(edge_idx)
        playback["logged_collision_edges"] = logged
        self.plotter.render()

    def _remember_actor_color(self, actor):
        key = id(actor)
        if key not in self._collision_highlight_original_colors:
            try:
                self._collision_highlight_original_colors[key] = (actor, tuple(actor.color()))
            except Exception:
                self._collision_highlight_original_colors[key] = (actor, None)

    def _highlight_actor_collision(self, actor):
        if actor is None:
            return
        self._remember_actor_color(actor)
        try:
            actor.c("red")
        except Exception:
            pass

    def _clear_collision_highlights(self):
        for actor, color in list(getattr(self, '_collision_highlight_original_colors', {}).values()):
            try:
                if color is not None:
                    actor.c(color)
            except Exception:
                pass
        self._collision_highlight_original_colors = {}

    def _highlight_spool_collision_object(self):
        spool = getattr(self, '_loaded_spool_mesh', None)
        actors = spool if isinstance(spool, (list, tuple)) else [spool]
        for actor in actors:
            self._highlight_actor_collision(actor)

    def _link_name_from_collision_geom(self, model, geom_name):
        link_actors = getattr(model, '_link_actors', {}) or {}
        candidates = sorted(link_actors.keys(), key=len, reverse=True)
        for link_name in candidates:
            if geom_name == link_name or geom_name.startswith(f"{link_name}_"):
                return link_name
        return None

    def _highlight_collision_geometry_name(self, geom_name):
        if not geom_name:
            return
        if str(geom_name).startswith("collision_object_"):
            self._highlight_spool_collision_object()
            return
        for model in getattr(self, '_robot_models', []):
            link_name = self._link_name_from_collision_geom(model, str(geom_name))
            if not link_name:
                continue
            for actor in getattr(model, '_link_actors', {}).get(link_name, []):
                self._highlight_actor_collision(actor)
            return

    def _highlight_collision_pairs(self, pairs):
        for pair in pairs or []:
            for geom_name in pair:
                self._highlight_collision_geometry_name(geom_name)

    def _step_path_playback(self, dt):
        if getattr(self, '_robot_path_playback', None) is not None:
            self._step_robot_path_playback(dt)
            return

        pb = getattr(self, '_path_playback', None)
        marker = getattr(self, '_path_playback_marker', None)
        if pb is None or marker is None:
            self._path_playback = None
            return

        pts = pb["points"]
        seg_lengths = pb["seg_lengths"]
        remaining = pb["speed"] * dt
        idx = int(pb["seg_idx"])
        seg_s = float(pb["seg_s"])

        while remaining > 0.0 and idx < len(seg_lengths):
            length = float(seg_lengths[idx])
            if length <= 1e-9:
                idx += 1
                seg_s = 0.0
                continue
            advance = min(remaining, length - seg_s)
            seg_s += advance
            remaining -= advance
            if seg_s >= length - 1e-9:
                idx += 1
                seg_s = 0.0

        if idx >= len(seg_lengths):
            marker.pos(pts[-1])
            self._path_playback = None
            self.__console.info("execute_inspection_path: playback finished")
        else:
            length = float(seg_lengths[idx])
            ratio = 0.0 if length <= 1e-9 else seg_s / length
            pos = pts[idx] * (1.0 - ratio) + pts[idx + 1] * ratio
            marker.pos(pos)
            pb["seg_idx"] = idx
            pb["seg_s"] = seg_s
        self.plotter.render()

    def _build_pin_model_for_robot(self, model):
        if pin is None:
            return None
        try:
            return pin.buildModelFromUrdf(model.urdf_path)
        except Exception:
            return None

    def _step_robot_path_playback(self, dt):
        rb = getattr(self, '_robot_path_playback', None)
        if rb is None:
            return
        model = rb["model"]
        pin_model = rb["pin_model"]
        q_pts = rb["q_points"]
        seg_lengths = rb["seg_lengths"]
        remaining = rb["speed"] * dt
        idx = int(rb["seg_idx"])
        seg_s = float(rb["seg_s"])
        self._log_path_playback_collision(rb, idx)

        while remaining > 0.0 and idx < len(seg_lengths):
            length = float(seg_lengths[idx])
            if length <= 1e-9:
                idx += 1
                seg_s = 0.0
                self._log_path_playback_collision(rb, idx)
                continue
            advance = min(remaining, length - seg_s)
            seg_s += advance
            remaining -= advance
            if seg_s >= length - 1e-9:
                idx += 1
                seg_s = 0.0
                self._log_path_playback_collision(rb, idx)

        if idx >= len(seg_lengths):
            q = q_pts[-1]
            self._apply_robot_q(model, pin_model, q)
            self._robot_path_playback = None
            self.__console.info("execute_inspection_path: robot playback finished")
        else:
            length = float(seg_lengths[idx])
            ratio = 0.0 if length <= 1e-9 else seg_s / length
            q = q_pts[idx] * (1.0 - ratio) + q_pts[idx + 1] * ratio
            self._apply_robot_q(model, pin_model, q)
            rb["seg_idx"] = idx
            rb["seg_s"] = seg_s

        marker = getattr(self, '_path_playback_marker', None)
        if marker is not None:
            tcp_T = self._pin_tcp_world_T(model, pin_model, q, rb["robot_name"])
            if tcp_T is not None:
                marker.pos(tcp_T[:3, 3])
        self.plotter.render()

    def _rotate_point_about_x(self, point, angle_deg, center):
        """Rotate a point around the global X axis."""
        point = np.array(point, dtype=float)
        center = np.array(center, dtype=float)
        rad = np.deg2rad(angle_deg)
        rel = point - center
        cos_v = np.cos(rad)
        sin_v = np.sin(rad)
        rotated = np.array([
            rel[0],
            rel[1] * cos_v - rel[2] * sin_v,
            rel[1] * sin_v + rel[2] * cos_v,
        ])
        return center + rotated

    def _get_spool_pose_payload(self):
        # 스풀 포즈 = chuck 기준 오프셋 (포지셔너가 움직여도 값은 그대로)
        x, y, z = getattr(self, '_spool_offset_xyz', (0.0, 0.0, 0.0))
        return {
            "x": float(x), "y": float(y), "z": float(z),
            "x_rotation": float(getattr(self, '_spool_offset_xrot', 0.0)),
            "z_rotation": float(getattr(self, '_spool_offset_zrot', 0.0)),
        }

    def _send_spool_pose_update(self, identity=None):
        if hasattr(self, 'zapi') and self.zapi and identity:
            self.zapi.update_spool_pose(self._get_spool_pose_payload(), identity=identity)

    def _get_spool_points(self):
        """Return full-resolution spool points in world coordinates when available."""
        full_local = getattr(self, '_spool_full_local_points', None)
        world_T = getattr(self, '_spool_world_T', None)
        if full_local is not None and world_T is not None:
            full_local = np.asarray(full_local, dtype=float)
            return (world_T[:3, :3] @ full_local.T).T + world_T[:3, 3]

        spool = getattr(self, '_loaded_spool_mesh', None)
        if spool is None:
            return None
        actors = spool if isinstance(spool, (list, tuple)) else [spool]
        all_verts = []
        for a in actors:
            if hasattr(a, "vertices"):
                v = np.asarray(a.vertices)
                if len(v):
                    all_verts.append(v)
        if not all_verts:
            return None
        return np.vstack(all_verts)

    def _replace_spool_points(self, new_pts):
        """스풀 actor를 새 점군으로 교체 (필터 결과 반영)."""
        old = getattr(self, '_loaded_spool_mesh', None)
        if old is not None:
            self.plotter.remove(old)
        recon = getattr(self, '_spool_recon_mesh', None)
        if recon is not None:
            self.plotter.remove(recon)
            self._spool_recon_mesh = None
        new_pts = np.asarray(new_pts, dtype=np.float64)
        new_actor = vedo.Points(new_pts)
        self.plotter.add(new_actor)
        self._loaded_spool_mesh = new_actor
        # 오프셋 모델 일관성: world 점을 현재 chuck@offset 기준 local로 환산
        Tc = self._chuck_world_T()
        if Tc is not None and getattr(self, '_spool_local_verts', None) is not None:
            Tinv = np.linalg.inv(Tc @ self._spool_offset_T())
            self._spool_local_verts = (Tinv[:3, :3] @ new_pts.T).T + Tinv[:3, 3]
        self.plotter.render()

    def _filter_loaded_spool(self, request_data):
        """현재 로드된 스풀에 직접 노이즈 필터(SOR/CCL)를 적용."""
        pts = self._get_spool_points()
        if pts is None:
            self.__console.warning("filter_spool: 로드된 스풀이 없습니다")
            return
        method = (request_data.get("method") or "").lower()
        params = request_data.get("params", {}) or {}
        n0 = len(pts)
        try:
            if method == "sor":
                pcd = _o3d.geometry.PointCloud()
                pcd.points = _o3d.utility.Vector3dVector(pts)
                clean, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=int(params.get("neighbors", 20)),
                    std_ratio=float(params.get("std_ratio", 2.0)))
                kept = np.asarray(clean.points)
            elif method == "ccl":
                from util.pcd_tool import voxel_ccl
                level = int(params.get("level", 7))
                min_points = int(params.get("min_points", 30))
                extent = float((pts.max(axis=0) - pts.min(axis=0)).max()) * 1.01
                voxel = extent / (2 ** level)
                _, labels = voxel_ccl(pts, voxel, min_points=min_points, connectivity=26)
                valid = labels[labels >= 0]
                if len(valid) == 0:
                    self.__console.warning("filter_spool(ccl): 연결요소 없음")
                    return
                uniq, cnts = np.unique(valid, return_counts=True)
                kept = pts[labels == uniq[np.argmax(cnts)]]
            else:
                self.__console.warning(f"filter_spool: 알 수 없는 method '{method}'")
                return
            self._replace_spool_points(kept)
            self.__console.info(f"filter_spool({method}): {n0} → {len(kept)} 점 (제거 {n0 - len(kept)})")
        except Exception as e:
            self.__console.error(f"filter_spool 실패: {e}")

    def _reconstruct_loaded_spool_mesh(self, request_data):
        """현재 로드된 스풀 점군으로 메시 재건(Marching Cubes) 후 표시."""
        pts = self._get_spool_points()
        if pts is None:
            self.__console.warning("reconstruct_mesh: 로드된 스풀이 없습니다")
            return
        params = request_data.get("params", {}) or {}
        try:
            from util.pcd_tool import reconstruct_mesh_marching_cubes
            pcd = _o3d.geometry.PointCloud()
            pcd.points = _o3d.utility.Vector3dVector(pts)
            mesh_o3d = reconstruct_mesh_marching_cubes(
                pcd,
                resolution=int(params.get("resolution", 128)),
                sigma=float(params.get("sigma", 1.5)),
                level=float(params.get("level", 0.5)))
            verts = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            if len(verts) == 0 or len(faces) == 0:
                self.__console.warning("reconstruct_mesh: 빈 메시")
                return
            vmesh = vedo.Mesh([verts, faces]).c("gray")

            # 기존 pcd 스풀 + 이전 재건 메시 제거
            old_pcd = getattr(self, '_loaded_spool_mesh', None)
            if old_pcd is not None:
                self.plotter.remove(old_pcd)
            old_recon = getattr(self, '_spool_recon_mesh', None)
            if old_recon is not None and old_recon is not old_pcd:
                self.plotter.remove(old_recon)

            self.plotter.add(vmesh)
            # 재건 메시를 새 스풀로 삼아 오프셋 모델에 연결 → 포지셔너 추종(같이 이동)
            self._loaded_spool_mesh = vmesh
            self._spool_recon_mesh = vmesh
            Tc = self._chuck_world_T()
            T = (getattr(self, '_spool_world_T', None)
                 if getattr(self, '_spool_world_T', None) is not None
                 else ((Tc @ self._spool_offset_T()) if Tc is not None else np.eye(4)))
            Tinv = np.linalg.inv(T)
            # verts(월드) → local 로 환산해 world = T @ local 유지 (현재 위치 보존 + 추종 가능)
            self._spool_local_verts = (Tinv[:3, :3] @ verts.T).T + Tinv[:3, 3]
            self._spool_world_T = T
            if Tc is not None:
                self._chuck_prev_T = Tc
            self.plotter.render()
            self._probe_current_spool_pinocchio_collision("reconstruct_mesh")
            self.__console.info(f"reconstruct_mesh: 정점 {len(verts)}, 면 {len(faces)} (pcd 제거, 메시가 스풀로 전환)")
        except Exception as e:
            self.__console.error(f"reconstruct_mesh 실패: {e}")

    def _save_loaded_spool(self, request_data):
        """현재 결과를 저장. 재건 메시가 있으면 메시를, 없으면 점군을 저장."""
        path = request_data.get("path")
        if not path:
            return
        try:
            recon = getattr(self, '_spool_recon_mesh', None)
            if recon is not None and hasattr(recon, "vertices") and hasattr(recon, "cells"):
                # 저장 mesh는 spool local frame으로 기록한다. 옆 JSON의 chuck 기준 offset을
                # 다시 적용하면 load 후 동일한 pose로 돌아갈 수 있다.
                verts = getattr(self, '_spool_local_verts', None)
                if verts is None:
                    verts = np.asarray(recon.vertices)
                m = _o3d.geometry.TriangleMesh()
                m.vertices = _o3d.utility.Vector3dVector(np.asarray(verts, dtype=float))
                m.triangles = _o3d.utility.Vector3iVector(np.asarray(recon.cells, dtype=np.int32))
                m.compute_vertex_normals()
                _o3d.io.write_triangle_mesh(path, m)
                self.__console.info(f"save_spool: local-frame 메시 저장 {path}")
            else:
                pts = self._get_spool_points()
                if pts is None:
                    self.__console.warning("save_spool: 저장할 스풀이 없습니다")
                    return
                pcd = _o3d.geometry.PointCloud()
                pcd.points = _o3d.utility.Vector3dVector(pts)
                _o3d.io.write_point_cloud(path, pcd)
                self.__console.info(f"save_spool: 점군 저장 {path} ({len(pts)} 점)")
        except Exception as e:
            self.__console.error(f"save_spool 실패: {e}")

    # --- 스풀 프레임/고정(강체 부착) 유틸 ---
    CHUCK_LINK_NAME = "m_column_passive_r"

    @staticmethod
    def _rotz(deg):
        r = np.deg2rad(deg); c, s = np.cos(r), np.sin(r)
        T = np.eye(4); T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c
        return T

    @staticmethod
    def _rotx(deg):
        r = np.deg2rad(deg); c, s = np.cos(r), np.sin(r)
        T = np.eye(4); T[1, 1] = c; T[1, 2] = -s; T[2, 1] = s; T[2, 2] = c
        return T

    @staticmethod
    def _transl(v):
        T = np.eye(4); T[:3, 3] = np.asarray(v, dtype=float)
        return T

    @staticmethod
    def _rot_about_axis(axis, center, deg):
        """center를 지나는 axis 둘레로 deg도 회전하는 4x4 (월드)."""
        axis = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return np.eye(4)
        x, y, z = axis / n
        th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th); C = 1 - c
        R = np.array([
            [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
        ])
        center = np.asarray(center, dtype=float)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = center - R @ center
        return T

    def _chuck_world_T(self):
        """column m chuck joint(m_column_passive_r) 링크의 4x4 월드 변환."""
        for model in getattr(self, '_robot_models', []):
            if hasattr(model, 'get_link_world_T'):
                T = model.get_link_world_T(self.CHUCK_LINK_NAME)
                if T is not None:
                    return np.asarray(T, dtype=float)
        return None

    def _spool_offset_T(self):
        """UI(=chuck 기준) 오프셋 포즈를 4x4 변환으로. spool_world = T_chuck @ T_offset @ local"""
        x, y, z = getattr(self, '_spool_offset_xyz', (0.0, 0.0, 0.0))
        xrot = getattr(self, '_spool_offset_xrot', 0.0)
        zrot = getattr(self, '_spool_offset_zrot', 0.0)
        return self._transl([x, y, z]) @ self._rotz(zrot) @ self._rotx(xrot)

    def _apply_spool_world_T(self):
        """현재 _spool_world_T 로 스풀 actor 정점 갱신 (world = T @ local)."""
        local = getattr(self, '_spool_local_verts', None)
        spool = getattr(self, '_loaded_spool_mesh', None)
        T = getattr(self, '_spool_world_T', None)
        if local is None or spool is None or T is None:
            return False
        world = (T[:3, :3] @ local.T).T + T[:3, 3]
        actors = spool if isinstance(spool, (list, tuple)) else [spool]
        if actors and hasattr(actors[0], 'vertices'):
            actors[0].vertices = world
            return True
        return False

    def _ensure_spool_frame_from_actor(self):
        """
        Mesh로 로드된 스풀처럼 local frame이 없는 경우, 현재 화면 좌표를
        현재 chuck@offset 기준 local frame으로 환산해 이후 fixation 이동을 가능하게 한다.
        """
        if getattr(self, '_spool_local_verts', None) is not None and getattr(self, '_spool_world_T', None) is not None:
            return True
        pts = self._get_spool_points()
        if pts is None or len(pts) == 0:
            return False
        Tc = self._chuck_world_T()
        T = (Tc @ self._spool_offset_T()) if Tc is not None else np.eye(4)
        Tinv = np.linalg.inv(T)
        self._spool_local_verts = (Tinv[:3, :3] @ np.asarray(pts, dtype=float).T).T + Tinv[:3, 3]
        self._spool_world_T = T
        if Tc is not None:
            self._chuck_prev_T = Tc
        self.__console.info("spool fixation frame initialized from current actor using chuck offset")
        return True

    def _render_spool_offset(self):
        """수동 배치: 현재 chuck 기준으로 스풀을 절대 배치 (spool_world = T_chuck @ T_offset)."""
        local = getattr(self, '_spool_local_verts', None)
        spool = getattr(self, '_loaded_spool_mesh', None)
        if local is None or spool is None:
            return False
        Tc = self._chuck_world_T()
        if Tc is None:
            return False
        self._spool_world_T = Tc @ self._spool_offset_T()
        self._chuck_prev_T = Tc
        return self._apply_spool_world_T()

    def _ensure_point_cloud_normals(self, pcd, source_path):
        """Ensure an Open3D point cloud has normals for pose determination."""
        if pcd is None or not pcd.has_points():
            raise RuntimeError(f"point cloud has no points: {source_path}")
        try:
            pcd.remove_non_finite_points()
            pcd.remove_duplicated_points()
        except Exception:
            pass
        if pcd.has_normals():
            self.__console.info(f"load_spool: normals included in point cloud: {source_path}")
        else:
            pcd.estimate_normals(
                search_param=_o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
            )
            pcd.normalize_normals()
            self.__console.info(f"load_spool: normals missing; estimated point cloud normals: {source_path}")
            self._save_estimated_normal_point_cloud(pcd, source_path)
        return pcd

    def _save_estimated_normal_point_cloud(self, pcd, source_path):
        source = Path(source_path)
        normal_path = source.with_name(f"{source.stem}_normal{source.suffix}")
        ok = _o3d.io.write_point_cloud(str(normal_path), pcd)
        if ok:
            self.__console.info(f"load_spool: saved estimated-normal point cloud: {normal_path}")
        else:
            self.__console.warning(f"load_spool: failed to save estimated-normal point cloud: {normal_path}")

    def _spool_load_scale(self, suffix):
        load_cfg = self._config.get("spool_load", {}) or {}
        scale_by_ext = load_cfg.get("scale_by_extension", {}) or {}
        if suffix in scale_by_ext:
            return float(scale_by_ext[suffix])
        if suffix == ".pcd":
            return float(load_cfg.get("pcd_scale", self._config.get("spool_pcd_scale", 1e-3)))
        if suffix == ".ply":
            return float(load_cfg.get("ply_scale", self._config.get("spool_ply_scale", 1.0)))
        return float(load_cfg.get("scale", self._config.get("spool_load_scale", 1.0)))

    def _apply_point_cloud_scale(self, pcd, scale, source_path):
        if scale == 1.0:
            return pcd
        pts = np.asarray(pcd.points, dtype=np.float64) * float(scale)
        pcd.points = _o3d.utility.Vector3dVector(pts)
        self.__console.info(f"load_spool: applied point cloud scale={scale:g}: {source_path}")
        return pcd

    def _apply_triangle_mesh_scale(self, mesh_o3d, scale, source_path):
        if scale == 1.0:
            return mesh_o3d
        verts = np.asarray(mesh_o3d.vertices, dtype=np.float64) * float(scale)
        mesh_o3d.vertices = _o3d.utility.Vector3dVector(verts)
        self.__console.info(f"load_spool: applied mesh scale={scale:g}: {source_path}")
        return mesh_o3d

    def _point_cloud_visual_points(self, pcd, source_path):
        pts = np.asarray(pcd.points, dtype=np.float64)
        load_cfg = self._config.get("spool_load", {}) or {}
        max_points = int(load_cfg.get("visual_max_points", 50000))
        if max_points <= 0 or len(pts) <= max_points:
            return pts
        step = int(np.ceil(len(pts) / max_points))
        visual_pts = pts[::step]
        self.__console.info(
            f"load_spool: visual point cloud downsampled {len(pts)} -> "
            f"{len(visual_pts)} points: {source_path}")
        return visual_pts

    def _load_spool_geometry_with_normals(self, path):
        """Load spool geometry and estimate normals when a point-cloud PLY/PCD has none."""
        suffix = os.path.splitext(path)[1].lower()
        scale = self._spool_load_scale(suffix)
        if Path(path).stem.endswith("_normal"):
            load_cfg = self._config.get("spool_load", {}) or {}
            scale = float(load_cfg.get("normal_scale", 1.0))
        if suffix in (".pcd", ".ply"):
            mesh_o3d = None
            if suffix == ".ply":
                try:
                    mesh_o3d = _o3d.io.read_triangle_mesh(path)
                except Exception:
                    mesh_o3d = None
                if mesh_o3d is not None and mesh_o3d.has_triangles():
                    mesh_o3d = self._apply_triangle_mesh_scale(mesh_o3d, scale, path)
                    if mesh_o3d.has_vertex_normals():
                        self.__console.info(f"load_spool: vertex normals included in mesh: {path}")
                    else:
                        mesh_o3d.compute_vertex_normals()
                        self.__console.info(f"load_spool: vertex normals missing; computed mesh normals: {path}")
                    verts = np.asarray(mesh_o3d.vertices, dtype=np.float64)
                    faces = np.asarray(mesh_o3d.triangles, dtype=np.int32)
                    return vedo.Mesh([verts, faces]), "mesh", mesh_o3d, None

            pcd = _o3d.io.read_point_cloud(path)
            pcd = self._apply_point_cloud_scale(pcd, scale, path)
            pcd = self._ensure_point_cloud_normals(pcd, path)
            pts = self._point_cloud_visual_points(pcd, path)
            return vedo.Points(pts), "point_cloud", None, pcd

        return vedo.load(path), "mesh", None, None

    def _process_request(self, request_data):
        """Process a request from the ZApi queue."""
        try:
            # Handle dictionary payload from zapi_* handlers
            if isinstance(request_data, dict):
                command = request_data.get("command")
                if command == "load_spool":
                    path = request_data.get("path")
                    if path:
                        self.__console.info(f"Loading Spool: {path}")
                        try:
                            self._clear_collision_highlights()
                            import pathlib as _pl
                            mesh, _geom_kind, _mesh_o3d, _pcd = self._load_spool_geometry_with_normals(path)
                            if mesh is not None:
                                # 스풀 위치는 chuck 조인트(m_column_passive_r)를 원점으로 본다.
                                # spool_world = T_chuck @ T_offset @ local
                                #  - local: 점군을 centroid로 중심화 + 기본정렬(chuck 기준 상수)
                                #    → 포지셔너 위치와 무관하게 reload 시 항상 현재 chuck 기준으로 배치
                                #  - T_offset: UI(chuck 기준) 위치/회전, 처음엔 0
                                _is_pcd = _pl.Path(path).suffix.lower() == ".pcd"
                                _is_point_cloud = _geom_kind == "point_cloud"
                                _default_x = -0.442  # 척 길이만큼 x로 (chuck 기준)

                                # 기존에 로드된 스풀/재건 메시 모두 제거 (새 파이프로 교체)
                                _old_sp = getattr(self, '_loaded_spool_mesh', None)
                                if _old_sp is not None:
                                    self.plotter.remove(_old_sp)
                                    self._loaded_spool_mesh = None
                                _old_rc = getattr(self, '_spool_recon_mesh', None)
                                if _old_rc is not None:
                                    if _old_rc is not _old_sp:
                                        self.plotter.remove(_old_rc)
                                    self._spool_recon_mesh = None

                                self._spool_offset_xyz = [0.0, 0.0, 0.0]
                                self._spool_offset_xrot = 0.0
                                self._spool_offset_zrot = 0.0
                                self._spool_fix_r = False
                                self._positioner_r_deg = 0.0
                                self._spool_world_T = None
                                self._chuck_prev_T = None
                                self._loaded_spool_x_flipped = False
                                self._loaded_spool_point_cloud = _pcd
                                self._loaded_spool_open3d_mesh = _mesh_o3d
                                self._spool_full_local_points = None

                                if _is_pcd:
                                    _pts = np.asarray(_pcd.points, dtype=np.float64)
                                    _visual_pts = np.asarray(mesh.vertices, dtype=np.float64)
                                    scaled = _pts
                                    visual_scaled = _visual_pts
                                    centroid = scaled.mean(axis=0)
                                    Rz = self._rotz(-90)[:3, :3]
                                    # centroid 중심화 → -90도 정렬 → chuck 기준 x 오프셋(상수)
                                    self._spool_full_local_points = (
                                        (Rz @ (scaled - centroid).T).T + np.array([_default_x, 0.0, 0.0]))
                                    self._spool_local_verts = (
                                        (Rz @ (visual_scaled - centroid).T).T + np.array([_default_x, 0.0, 0.0]))
                                    self.plotter.add(mesh)
                                    self._loaded_spool_mesh = mesh
                                    self._render_spool_offset()
                                elif _is_point_cloud:
                                    self._spool_full_local_points = np.asarray(_pcd.points, dtype=float).copy()
                                    self._spool_local_verts = np.asarray(mesh.vertices, dtype=float).copy()
                                    self.plotter.add(mesh)
                                    self._loaded_spool_mesh = mesh
                                    self._render_spool_offset()
                                else:
                                    # 저장된 PLY/mesh는 spool local frame(m)으로 간주한다.
                                    # 옆 JSON의 chuck 기준 offset을 적용하면 저장 시 pose로 복원된다.
                                    if hasattr(mesh, "vertices"):
                                        self._spool_local_verts = np.asarray(mesh.vertices, dtype=float).copy()
                                        self._spool_full_local_points = self._spool_local_verts.copy()
                                    else:
                                        self._spool_local_verts = None
                                        self._spool_full_local_points = None
                                    self.plotter.add(mesh)
                                    self._loaded_spool_mesh = mesh
                                    if hasattr(mesh, "cells"):
                                        self._spool_recon_mesh = mesh
                                    self._render_spool_offset()

                                self.plotter.render()
                                self._probe_current_spool_pinocchio_collision("load_spool")
                                self.__console.info(f"Successfully loaded {path}")
                                
                                # Send reply
                                if hasattr(self, 'zapi') and self.zapi:
                                    identity = request_data.get("_identity")
                                    self.zapi.reply_load_spool(path, True, identity=identity)
                            else:
                                self.__console.error(f"Failed to load mesh from {path}")
                                if hasattr(self, 'zapi') and self.zapi:
                                    identity = request_data.get("_identity")
                                    self.zapi.reply_load_spool(path, False, identity=identity)
                        except Exception as e:
                            self.__console.error(f"Exception loading mesh: {e}")
                            if hasattr(self, 'zapi') and self.zapi:
                                identity = request_data.get("_identity")
                                self.zapi.reply_load_spool(path, False, identity=identity)
                elif command == "flip_spool_x":
                    spool = getattr(self, '_loaded_spool_mesh', None)
                    if spool is None or (isinstance(spool, (list, tuple)) and len(spool) == 0):
                        self.__console.warning("Cannot flip spool X direction: no spool loaded")
                        return True

                    actors = spool if isinstance(spool, (list, tuple)) else [spool]

                    # 스풀의 현재 bounding box 중심을 mirror origin으로 사용
                    bounds_list = [a.bounds() for a in actors if hasattr(a, "bounds")]
                    if bounds_list:
                        x_min = min(b[0] for b in bounds_list)
                        x_max = max(b[1] for b in bounds_list)
                        y_min = min(b[2] for b in bounds_list)
                        y_max = max(b[3] for b in bounds_list)
                        z_min = min(b[4] for b in bounds_list)
                        z_max = max(b[5] for b in bounds_list)
                        center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
                    else:
                        center = [0, 0, 0]

                    for actor in actors:
                        if hasattr(actor, "mirror"):
                            actor.mirror(axis="x", origin=center)
                    if self._ensure_spool_frame_from_actor():
                        T = getattr(self, '_spool_world_T', None)
                        pts = self._get_spool_points()
                        if T is not None and pts is not None:
                            Tinv = np.linalg.inv(T)
                            self._spool_local_verts = (Tinv[:3, :3] @ pts.T).T + Tinv[:3, 3]

                    self._loaded_spool_x_flipped = not getattr(self, '_loaded_spool_x_flipped', False)
                    self.plotter.render()
                    self.__console.info(
                        f"Flipped spool X direction: {self._loaded_spool_x_flipped}")
                elif command == "move_spool":
                    # 스풀 위치/회전을 chuck 기준 오프셋으로 설정 (x,y,z,x_rot,z_rot)
                    spool = getattr(self, '_loaded_spool_mesh', None)
                    if spool is None:
                        self.__console.warning("move_spool: 로드된 스풀(PCD)이 없습니다")
                        return True
                    new_xyz = [
                        float(request_data.get("x", 0.0)),
                        float(request_data.get("y", 0.0)),
                        float(request_data.get("z", 0.0)),
                    ]
                    new_xrot = float(request_data.get("x_rotation", 0.0))
                    new_zrot = float(request_data.get("z_rotation", 0.0))

                    # 저장된 mesh/ply처럼 world 좌표로 로드되어 local frame이 없는 경우:
                    # 요청된 offset 기준으로 local을 역산해 현재 화면 위치를 보존한 채
                    # 이후 offset/positioner 추종 모델에 편입한다.
                    if getattr(self, '_spool_local_verts', None) is None:
                        pts = self._get_spool_points()
                        Tc = self._chuck_world_T()
                        if pts is None or Tc is None:
                            self.__console.warning("move_spool: 스풀 local frame 초기화 실패")
                            return True
                        old_xyz = getattr(self, '_spool_offset_xyz', [0.0, 0.0, 0.0])
                        old_xrot = getattr(self, '_spool_offset_xrot', 0.0)
                        old_zrot = getattr(self, '_spool_offset_zrot', 0.0)
                        self._spool_offset_xyz = new_xyz
                        self._spool_offset_xrot = new_xrot
                        self._spool_offset_zrot = new_zrot
                        Tnew = Tc @ self._spool_offset_T()
                        Tinv = np.linalg.inv(Tnew)
                        self._spool_local_verts = (Tinv[:3, :3] @ pts.T).T + Tinv[:3, 3]
                        self._spool_world_T = Tnew
                        self._spool_offset_xyz = old_xyz
                        self._spool_offset_xrot = old_xrot
                        self._spool_offset_zrot = old_zrot

                    self._spool_offset_xyz = new_xyz
                    self._spool_offset_xrot = new_xrot
                    self._spool_offset_zrot = new_zrot
                    self._render_spool_offset()
                    self.plotter.render()
                    self._probe_current_spool_pinocchio_collision("move_spool")
                    self.__console.info(
                        f"Spool offset set to xyz={self._spool_offset_xyz}, "
                        f"x_rot={self._spool_offset_xrot}, z_rot={self._spool_offset_zrot}")
                elif command == "set_spool_fixation":
                    fix_m_column_z = bool(request_data.get("fix_m_column_z", False))
                    fix_f_column_r = bool(request_data.get("fix_f_column_r", False))
                    self._spool_fix_r = fix_f_column_r
                    if fix_m_column_z or fix_f_column_r:
                        self._ensure_spool_frame_from_actor()
                    Tc_now = self._chuck_world_T()
                    if Tc_now is not None:
                        self._chuck_prev_T = Tc_now
                    self.__console.info(
                        f"Spool fixation set: fix_f={fix_f_column_r}, fix_z={fix_m_column_z}")
                elif command == "move_positioner":
                    import math
                    axis = request_data.get("axis")
                    position = float(request_data.get("position", 0.0))
                    velocity = float(request_data.get("velocity", 0.0))
                    fix_m_column_z = bool(request_data.get("fix_m_column_z", False))
                    fix_f_column_r = bool(request_data.get("fix_f_column_r", False))
                    self._spool_fix_r = fix_f_column_r
                    if fix_m_column_z or fix_f_column_r:
                        self._ensure_spool_frame_from_actor()

                    for model in getattr(self, '_robot_models', []):
                        joint_map = model._urdf._joint_map if model._urdf else {}
                        if axis == "x" and "base_to_m_column" in joint_map:
                            model.set_joint("base_to_m_column", -position)
                        elif axis == "z" and "base_to_f_column_z" in joint_map:
                            model.set_joint("base_to_f_column_z", position)
                            model.set_joint("m_column_to_m_column_z", position)
                        elif axis == "r" and "f_column_z_to_f_column_r" in joint_map:
                            model.set_joint("f_column_z_to_f_column_r", math.radians(position))
                        elif axis == "clamp" and "f_column_r_to_f_column_passive_clamp" in joint_map:
                            # prismatic y-axis, range -0.9~0; UI value 0~0.9 → joint = -position
                            model.set_joint("f_column_r_to_f_column_passive_clamp", -position)
                        else:
                            continue
                        model.update_fk()

                    # 스풀 추종은 "고정된 축"에 대해서만. (체크 안 하면 안 따라감)
                    Tc_now = self._chuck_world_T()
                    has_frame = (getattr(self, '_spool_world_T', None) is not None
                                 and getattr(self, '_spool_local_verts', None) is not None)
                    if has_frame and Tc_now is not None:
                        if axis in ("x", "z") and fix_m_column_z and getattr(self, '_chuck_prev_T', None) is not None:
                            # column m 고정: chuck 병진량만큼 스풀 평행이동
                            dt = Tc_now[:3, 3] - self._chuck_prev_T[:3, 3]
                            T = np.eye(4); T[:3, 3] = dt
                            self._spool_world_T = T @ self._spool_world_T
                            self._apply_spool_world_T()
                        elif axis == "r" and fix_f_column_r:
                            # column r 고정: chuck joint 중심·축(chuck x축) 기준으로 스풀 회전
                            r_prev = getattr(self, '_positioner_r_deg', 0.0)
                            delta_r = position - r_prev
                            center = Tc_now[:3, 3]
                            axis_w = Tc_now[:3, :3] @ np.array([1.0, 0.0, 0.0])
                            Rm = self._rot_about_axis(axis_w, center, delta_r)
                            self._spool_world_T = Rm @ self._spool_world_T
                            self._apply_spool_world_T()
                    if Tc_now is not None:
                        self._chuck_prev_T = Tc_now
                    if axis == "r":
                        self._positioner_r_deg = position

                    self.plotter.render()
                    self.__console.info(f"Positioner {axis} moved to {position} (vel={velocity})")
                elif command == "move_manipulator":
                    self._set_joint_animation(
                        request_data.get("robot"),
                        request_data.get("joint"),
                        request_data.get("target", 0.0),
                        request_data.get("speed", 1.0),
                        request_data.get("accel"))
                elif command == "stop_manipulator":
                    self._stop_joint_animation(
                        request_data.get("robot"),
                        request_data.get("joint"))
                elif command == "filter_spool":
                    self._filter_loaded_spool(request_data)
                elif command == "reconstruct_mesh":
                    self._reconstruct_loaded_spool_mesh(request_data)
                elif command == "save_spool":
                    self._save_loaded_spool(request_data)
                elif command == "pick_inspection_point":
                    self._inspection_pick_enabled = bool(request_data.get("enabled", True))
                    self._inspection_pick_identity = request_data.get("_identity")
                    if self._inspection_pick_enabled:
                        self._chuck_mount_pick_enabled = False
                    self.__console.info(
                        "inspection pick mode enabled" if self._inspection_pick_enabled
                        else "inspection pick mode disabled")
                elif command == "pick_chuck_mount_points":
                    enabled = bool(request_data.get("enabled", True))
                    self._chuck_mount_pick_enabled = enabled
                    self._chuck_mount_pick_identity = request_data.get("_identity")
                    if enabled:
                        self._inspection_pick_enabled = False
                        if bool(request_data.get("clear", True)):
                            self._clear_chuck_mount_points()
                    self.__console.info(
                        "chuck mount pick mode enabled: click fixed-side point, then moving-side point"
                        if enabled else "chuck mount pick mode disabled")
                elif command == "set_chuck_mount_points":
                    self._set_chuck_mount_points(
                        request_data.get("points", []),
                        request_data.get("local_points"))
                elif command == "clear_chuck_mount_points":
                    self._chuck_mount_pick_enabled = False
                    self._clear_chuck_mount_points()
                elif command == "plan_inspection_path":
                    self._plan_inspection_path(request_data)
                elif command == "determine_ef_pose":
                    self._determine_ef_pose(request_data)
                elif command == "clear_inspection_path":
                    self._inspection_pick_enabled = False
                    self._path_playback = None
                    self._robot_path_playback = None
                    self._clear_collision_highlights()
                    self._clear_inspection_visuals(clear_point=True)
                    if getattr(self, '_path_playback_marker', None) is not None:
                        self.plotter.remove(self._path_playback_marker)
                        self._path_playback_marker = None
                    self._last_inspection_path = None
                    self._last_inspection_q_path = None
                    self._last_inspection_edge_collisions = []
                    self._last_inspection_robot = None
                elif command == "execute_inspection_path":
                    self._start_path_playback(request_data.get("speed", 0.2))
                elif command == "load_test_weld_point":
                    path = request_data.get("path")
                    if path:
                        self.__console.info(f"Loading Test Weld Point from CSV: {path}")
                        # We will log it for now. Actual rendering can be implemented based on CSV format.
                        # Can be implemented further as needed.
                        self.__console.info(f"Successfully handled test weld point CSV path: {path}")
            
            # Legacy/Raw handling (list/tuple)
            elif isinstance(request_data, (list, tuple)) and len(request_data) >= 2:
                 pass # Handle raw messages if any
                 
        except Exception as e:
            self.__console.error(f"Error processing request: {e}")

    def set_zapi(self, zapi):
        """Set the ZApi instance for callbacks."""
        self.zapi = zapi

    def push_request(self, data):
        """Thread-safe method for ZApi to push requests into the visualizer queue."""
        with self._queue_lock:
            self._request_queue.append(data)

    def _log_rendering_frequency(self, loop_count, last_log_time):
        current_time = time.time()
        
        # Calculate Instantaneous FPS for Text Overlay
        frame_duration = current_time - self.last_frame_time
        if frame_duration > 0:
            inst_fps = 1.0 / frame_duration
            if self.fps_text:
                self.fps_text.text(f"FPS: {inst_fps:.1f}")
        
        self.last_frame_time = current_time
            
        return loop_count, last_log_time

    def on_close(self):
        """Cleanup on visualizer close. Socket cleanup is handled by Zapi."""
        self._should_close = True
        self.__console.debug("Visualizer on_close called")
