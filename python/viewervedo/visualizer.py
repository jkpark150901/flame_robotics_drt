"""
3D Visualizer using Vedo
@note
- Vedo is a Python library for 3D visualization based on VTK (Visualization Toolkit).
- Visualizer is a class that renders 3D geometries
- All ZMQ communication is handled by Zapi (viewervedo/zapi.py)
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from collections import deque
import time
import importlib
import inspect
import sys
import types
import os
import json
import copy
import csv
from pathlib import Path
import numpy as np
import vedo
try:
    import pinocchio as pin
except ImportError:
    pin = None
if pin is not None and not hasattr(pin, "forwardKinematics"):
    pin = None

# Open3D core geometry is used here; the optional ML module can fail in this
# workspace because of NumPy/SciPy ABI mismatch.
sys.modules.setdefault("open3d.ml", types.ModuleType("open3d.ml"))
import open3d as _o3d
from util.logger.console import ConsoleLogger
from common.graphic_device import GraphicDevice
from viewervedo.robot import RobotModel, load_robots_from_config
from plugins.pluginbase.plannerbase import PlannerBase
from plugins.robotics.backend import RobotDescription
from plugins.robotics.inspection_planning_base import InspectionIKRequest, InspectionPlanningBase
from plugins.robotics.pinocchio_backend import PinocchioRoboticsBackend


class InspectionIKFailure(RuntimeError):
    def __init__(self, message, failure_info=None):
        super().__init__(message)
        self.failure_info = failure_info or {}


class Visualizer:
    def __init__(self, config:dict=None):
        if config is None:
            config = {}
        self._config = config
    
        self.__console = ConsoleLogger.get_logger()
        experiment_root = Path(config.get("experiment_dir", "experiment")) / "inspection_ik"
        self._inspection_ik_experiment_dir = experiment_root / time.strftime("session_%Y%m%d_%H%M%S")
        self._inspection_ik_experiment_dir.mkdir(parents=True, exist_ok=True)
        self.__console.info(f"inspection IK experiment session: {self._inspection_ik_experiment_dir}")

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

        # 留ㅻ땲?곕젅?댄꽣 議곗씤???좊땲硫붿씠??蹂닿컙 ?대룞) ?곹깭
        # 媛???ぉ: {"model", "joint", "target", "speed"}  speed ?⑥쐞/?꾨젅?꾨떦 = unit/s
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
        self._chuck_profile_actors = []
        self._chuck_frame_actors = []
        self._ef_pose_actors = []
        self._inspection_goal_pose_actors = []
        self._inspection_goal_robot_actors = []
        self._ef_target_poses = {}
        self._ef_pose_groups = []
        self._inspection_path_actor = None
        self._ik_failure_actors = []
        self._robot_tcp_axis_actors = []
        self._last_inspection_path = None
        self._last_inspection_q_path = None
        self._last_inspection_edge_collisions = []
        self._last_inspection_robot = None
        self._last_inspection_plan_sequence = []
        self._inspection_sequence_playback = None
        self._robot_path_playback = None
        self._path_playback = None
        self._path_playback_marker = None
        self._collision_highlight_original_colors = {}
        self._robot_joint_state_identity = None
        self._last_robot_joint_state_sent = 0.0
        self._spool_source_path = None

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
        self._show_chuck_frames(render=False)
        self._show_robot_tcp_axes(render=False)

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
        self._pinocchio_robot_collision_cache = {}
        try:
            self._robotics_backend = PinocchioRoboticsBackend()
            self._inspection_planning_base = InspectionPlanningBase(self._robotics_backend)
        except Exception as exc:
            raise RuntimeError(f"robotics backend initialization failed: {exc}") from exc
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
            self._register_robotics_backend_model(name, full_path, base)
            self._cache_robot_pinocchio_collision_model(name, full_path)

        all_actors = [a for m in self._robot_models for a in m.actors]
        if all_actors:
            self.plotter.add(*all_actors)
            self.__console.info(f"Added {len(all_actors)} robot mesh actors to plotter")

    def _register_robotics_backend_model(self, robot_name, urdf_path, base_pose):
        backend = getattr(self, "_robotics_backend", None)
        if backend is None:
            raise RuntimeError("robotics backend is not initialized")
        try:
            description = RobotDescription(
                name=str(robot_name),
                urdf_path=os.path.abspath(urdf_path),
                base_T=self._pose6_to_T(base_pose or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                package_dirs=[os.path.dirname(os.path.abspath(urdf_path))],
                target_frame=self._robot_target_link_name(robot_name),
            )
            handle = backend.register_robot(description)
            self.__console.info(
                f"registered robotics backend model: robot={robot_name}, "
                f"backend={backend.name}, nq={handle.model.nq}")
            return handle
        except Exception as exc:
            raise RuntimeError(f"failed to register robotics backend model for {robot_name}: {exc}") from exc

    def _cache_robot_pinocchio_collision_model(self, robot_name, urdf_path):
        if pin is None:
            return
        cache = getattr(self, "_pinocchio_robot_collision_cache", None)
        if cache is None:
            self._pinocchio_robot_collision_cache = {}
            cache = self._pinocchio_robot_collision_cache
        if robot_name in cache:
            return
        try:
            t0 = time.perf_counter()
            backend = getattr(self, "_robotics_backend", None)
            if backend is None:
                raise RuntimeError("robotics backend is not initialized")
            backend.configure_collision(robot_name, static_meshes=None, sample_resolution=0.05)
            backend_cache = backend.collision_model_cache(robot_name)
            cache[robot_name] = {
                "urdf_path": os.path.abspath(urdf_path),
                "pin_model": backend_cache.get("pin_model"),
                "pin_geom_model": backend_cache.get("pin_geom_model"),
                "robot_geom_ids": list(backend_cache.get("robot_geom_ids", [])),
            }
            cache[robot_name]["ik_collision_probe"] = self._make_inspection_ik_collision_probe(cache[robot_name])
            geom_model = cache[robot_name].get("pin_geom_model")
            self.__console.info(
                "Cached Pinocchio collision model: "
                f"robot={robot_name}, urdf={urdf_path}, "
                f"geoms={len(geom_model.geometryObjects)}, "
                f"pairs={len(geom_model.collisionPairs)}, "
                f"elapsed={time.perf_counter() - t0:.3f}s")
        except Exception as exc:
            raise RuntimeError(f"failed to cache robotics collision model for {robot_name}: {exc}") from exc

    def _make_inspection_ik_collision_probe(self, pin_cache):
        if pin is None or not pin_cache:
            return None
        pin_model = pin_cache.get("pin_model")
        pin_geom_model = pin_cache.get("pin_geom_model")
        if pin_model is None or pin_geom_model is None:
            return None
        probe = PlannerBase()
        probe.pin_model = pin_model
        probe.pin_data = pin_model.createData()
        probe.pin_geom_model = copy.deepcopy(pin_geom_model)
        probe._pin_robot_geom_ids = list(pin_cache.get("robot_geom_ids", []))
        probe._pin_static_object_ids = []
        probe.pin_geom_data = pin.GeometryData(probe.pin_geom_model)
        return probe

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
            self.__console.warning("inspection pick: 濡쒕뱶???ㅽ????놁뒿?덈떎")
            return

        picked = getattr(event, "picked3d", None)
        if picked is None:
            self.__console.warning("inspection pick: no picked pipe surface point")
            return

        picked = np.asarray(picked, dtype=float)
        # PCD/mesh 紐⑤몢?먯꽌 ?ㅼ젣 pipe ?먯쑝濡??ㅻ깄????ν븳??
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
        self._clear_ik_failure_visuals(render=False)
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
        if bool(getattr(self, '_chuck_mount_align_on_pick', False)):
            identity = getattr(self, '_chuck_mount_pick_identity', None)
            self._chuck_mount_pick_enabled = False
            align_target = getattr(self, '_chuck_mount_align_target', "f")
            if align_target == "m":
                self._align_spool_profile_to_chuck(
                    point,
                    identity=identity,
                    link_name=self.M_CHUCK_LINK_NAME,
                    label="m-column")
            else:
                self._align_column_to_profile(
                    point,
                    identity=identity,
                    link_name=self.F_CHUCK_LINK_NAME,
                    label="f-column")
            if hasattr(self, 'zapi') and self.zapi and identity:
                self.zapi.update_chuck_mount_points(self._get_chuck_mount_points_payload(), identity=identity)
            return

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
        self._clear_chuck_profile_visuals(render=False)
        self.plotter.render()

    def _add_chuck_mount_point(self, point, local_point=None):
        point = np.asarray(point, dtype=float)
        colors = ("dodgerblue", "orange")
        marker = vedo.Sphere(
            pos=point,
            r=0.018,
            c=colors[len(self._chuck_mount_points) % len(colors)],
        )
        marker.pickable(False)
        self._chuck_mount_points.append(point)
        self._chuck_mount_local_points.append(None if local_point is None else np.asarray(local_point, dtype=float))
        self._chuck_mount_markers.append(marker)
        self.plotter.add(marker)
        self.plotter.render()

    def _refresh_chuck_mount_markers(self):
        for marker in getattr(self, '_chuck_mount_markers', []):
            if marker is not None:
                self.plotter.remove(marker)
        self._chuck_mount_markers = []
        colors = ("dodgerblue", "orange")
        for i, point in enumerate(getattr(self, '_chuck_mount_points', [])):
            marker = vedo.Sphere(
                pos=np.asarray(point, dtype=float),
                r=0.018,
                c=colors[i % len(colors)],
            )
            marker.pickable(False)
            self._chuck_mount_markers.append(marker)
            self.plotter.add(marker)

    def _set_chuck_mount_points(self, points, local_points=None):
        self._clear_chuck_mount_points()
        if not points:
            return
        for i, point in enumerate(points[:2]):
            local_point = None
            if local_points and i < len(local_points):
                local_point = local_points[i]
            self._add_chuck_mount_point(point, local_point)

    def _clear_chuck_profile_visuals(self, render=True):
        for actor in getattr(self, '_chuck_profile_actors', []) or []:
            try:
                self.plotter.remove(actor)
            except Exception:
                pass
        self._chuck_profile_actors = []
        if render:
            self.plotter.render()

    @staticmethod
    def _rotation_between_vectors(source, target):
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        source_norm = np.linalg.norm(source)
        target_norm = np.linalg.norm(target)
        if source_norm < 1e-12 or target_norm < 1e-12:
            return np.eye(3)
        a = source / source_norm
        b = target / target_norm
        cross = np.cross(a, b)
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if dot > 1.0 - 1e-12:
            return np.eye(3)
        if dot < -1.0 + 1e-12:
            basis = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(a, basis))) > 0.9:
                basis = np.array([0.0, 1.0, 0.0])
            axis = np.cross(a, basis)
            axis = axis / np.linalg.norm(axis)
            return -np.eye(3) + 2.0 * np.outer(axis, axis)
        skew = np.array([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ])
        return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))

    @staticmethod
    def _unit_vector(vector):
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm < 1e-12:
            return vector
        return vector / norm

    def _signed_angle_about_axis(self, source, target, axis):
        axis = self._unit_vector(axis)
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        source = source - np.dot(source, axis) * axis
        target = target - np.dot(target, axis) * axis
        if np.linalg.norm(source) < 1e-12 or np.linalg.norm(target) < 1e-12:
            return 0.0
        source = self._unit_vector(source)
        target = self._unit_vector(target)
        sin_v = float(np.dot(axis, np.cross(source, target)))
        cos_v = float(np.clip(np.dot(source, target), -1.0, 1.0))
        return float(np.arctan2(sin_v, cos_v))

    def _align_spool_profile_to_chuck(self, target_point, identity=None, link_name=None, label="chuck"):
        try:
            if not self._ensure_spool_frame_from_actor():
                raise RuntimeError("spool frame is not available")
            Tc = self._chuck_link_world_T(link_name or self.M_CHUCK_LINK_NAME)
            if Tc is None:
                raise RuntimeError("chuck link transform is not available")

            profile = self._profile_for_chuck_mount_point(target_point)
            pipe_axis = np.asarray(profile["axis"], dtype=float)
            pipe_center = np.asarray(profile["center"], dtype=float)
            pipe_radius = float(profile["radius"])
            chuck_center = self._chuck_center_world(link_name or self.M_CHUCK_LINK_NAME, Tc)
            chuck_axis = self._chuck_axis_world(link_name or self.M_CHUCK_LINK_NAME, Tc)
            alignment_axis = np.asarray(chuck_axis, dtype=float)
            if (link_name or self.M_CHUCK_LINK_NAME) == self.M_CHUCK_LINK_NAME:
                pipe_origin = np.asarray(target_point, dtype=float)
                pipe_axis, positive_count, negative_count = self._pipe_axis_toward_sparse_side(
                    pipe_axis,
                    pipe_origin,
                )
                profile["end_center"] = pipe_origin
                profile["far_end_center"] = None
                profile["axis"] = pipe_axis
                profile["sparse_side_counts"] = {
                    "positive": positive_count,
                    "negative": negative_count,
                }
                m_cfg = self._chuck_frame_config(self.M_CHUCK_LINK_NAME)
                alignment_axis = self._unit_vector(
                    chuck_axis * float(m_cfg.get("profile_align_axis_sign", -1.0))
                )
                self.__console.info(
                    "m-column profile direction: "
                    f"selected={np.round(pipe_origin, 4)}, "
                    f"positive_count={positive_count}, negative_count={negative_count}, "
                    f"axis={np.round(pipe_axis, 4)}, "
                    f"chuck_axis={np.round(chuck_axis, 4)}, "
                    f"alignment_axis={np.round(alignment_axis, 4)}")
            else:
                if float(np.dot(pipe_axis, chuck_axis)) < 0.0:
                    pipe_axis = -pipe_axis
                pipe_origin = np.asarray(self._pipe_profile_end_center(
                    profile.get("fit_points"),
                    pipe_axis,
                    pipe_center,
                    pipe_radius,
                    target_point,
                    self._profile_distance_threshold(target_point),
                ), dtype=float)

            R_align = self._rotation_between_vectors(pipe_axis, alignment_axis)
            T_align = np.eye(4)
            T_align[:3, :3] = R_align
            T_align[:3, 3] = chuck_center - R_align @ pipe_origin
            aligned_origin = R_align @ pipe_origin + T_align[:3, 3]
            aligned_profile = {
                "axis": R_align @ pipe_axis,
                "center": aligned_origin,
                "radius": pipe_radius,
            }
            aligned_profile["center_error"] = float(np.linalg.norm(aligned_origin - chuck_center))
            aligned_profile["axis_error_deg"] = float(np.rad2deg(np.arccos(np.clip(np.dot(
                self._unit_vector(aligned_profile["axis"]), self._unit_vector(alignment_axis)), -1.0, 1.0))))
            self._send_chuck_mount_profile_update(label, aligned_profile, identity=identity)

            self._spool_world_T = T_align @ getattr(self, '_spool_world_T')
            self._apply_spool_world_T()
            updated_points = []
            for point, local_point in zip(self._chuck_mount_points, self._chuck_mount_local_points):
                if local_point is not None:
                    local_h = np.ones(4, dtype=float)
                    local_h[:3] = np.asarray(local_point, dtype=float)
                    updated_points.append((self._spool_world_T @ local_h)[:3])
                else:
                    updated_points.append(T_align[:3, :3] @ np.asarray(point, dtype=float) + T_align[:3, 3])
            self._chuck_mount_points = updated_points
            self._refresh_chuck_mount_markers()
            self._send_spool_pose_update(identity=identity)
            self._save_spool_alignment_state(reason=f"{label} align")
            self._show_chuck_profile_alignment(
                pipe_center,
                pipe_axis,
                pipe_radius,
                chuck_center,
                alignment_axis,
                T_align,
                fit_points=profile.get("fit_points"),
                pipe_origin=pipe_origin,
            )
            self.plotter.render()
            self.__console.info(
                f"{label} mount aligned: profile_origin={np.round(pipe_origin, 4)}, "
                f"radius={pipe_radius:.6f}")
        except Exception as exc:
            self.__console.error(f"chuck mount profile alignment failed: {exc}")

    def _send_chuck_mount_profile_update(self, label, profile, identity=None):
        if not (hasattr(self, 'zapi') and self.zapi and identity):
            return
        if not hasattr(self.zapi, 'update_chuck_mount_profile'):
            return
        try:
            center = np.asarray(profile["center"], dtype=float)
            axis = self._unit_vector(profile["axis"])
            payload = {
                "target": str(label),
                "center": center.tolist(),
                "axis": np.asarray(axis, dtype=float).tolist(),
                "radius": float(profile["radius"]),
            }
            for key in ("center_error", "axis_error_deg"):
                if key in profile:
                    payload[key] = float(profile[key])
            self.zapi.update_chuck_mount_profile(payload, identity=identity)
        except Exception as exc:
            self.__console.warning(f"Failed to send chuck mount profile update: {exc}")

    def _profile_for_chuck_mount_point(self, target_point):
        pose_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "plugins", "poseDeterminator")
        )
        if pose_dir not in sys.path:
            sys.path.insert(0, pose_dir)
        import PipeEndProfileAnalyzer as pipe_analyzer

        points = np.asarray(self._get_spool_points(), dtype=float)
        if points is None or len(points) < 10:
            raise RuntimeError("loaded spool point cloud is not available")
        target_point = np.asarray(target_point, dtype=float)
        anchor_idx = int(np.argmin(np.linalg.norm(points - target_point, axis=1)))
        bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        distance_threshold = self._profile_distance_threshold(target_point, bbox_diag=bbox_diag)
        params = self._config.get("chuck_mount_profile", {}) or {}
        min_points = int(params.get("min_points", 20))
        print(f"PipeEndProfileAnalyzer: anchor_idx={anchor_idx}, distance_threshold={distance_threshold:.6f}, min_points={min_points}")
        sample, model, debug = pipe_analyzer._sample_profile_points_from_anchor(
            points,
            anchor_idx,
            distance_threshold,
            bbox_diag,
            min_points=min_points,
            log_timing=False,
        )
        if model is None:
            raise RuntimeError("PipeEndProfileAnalyzer profile sampling failed.")
        axis, center, radius = model
        end_center = self._pipe_profile_end_center(
            np.asarray(sample, dtype=float),
            np.asarray(axis, dtype=float),
            np.asarray(center, dtype=float),
            float(radius),
            target_point,
            distance_threshold,
        )
        return {
            "axis": np.asarray(axis, dtype=float),
            "center": np.asarray(center, dtype=float),
            "radius": float(radius),
            "end_center": end_center,
            "debug": debug or {},
            "fit_points": np.asarray(sample, dtype=float) if sample is not None else None,
        }

    def _profile_distance_threshold(self, target_point=None, bbox_diag=None):
        points = self._get_spool_points()
        if bbox_diag is None:
            bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) if points is not None else 0.0
        params = self._config.get("chuck_mount_profile", {}) or {}
        return float(params.get(
            "distance_threshold",
            max(float(bbox_diag) * 0.005, np.finfo(float).eps),
        ))

    def _pipe_profile_end_center(
        self,
        points,
        axis,
        axis_point,
        radius,
        target_point,
        distance_threshold,
        prefer_axis_min=False,
    ):
        axis = self._unit_vector(axis)
        source_points = self._get_spool_points()
        if source_points is None:
            source_points = points
        points = np.asarray(source_points, dtype=float)
        if points.size == 0:
            return np.asarray(axis_point, dtype=float)
        points = points.reshape((-1, 3))
        axis_point = np.asarray(axis_point, dtype=float)
        target_point = np.asarray(target_point, dtype=float)
        rel = points - axis_point
        projections = rel @ axis
        radial = rel - np.outer(projections, axis)
        residual = np.abs(np.linalg.norm(radial, axis=1) - float(radius))
        tolerance = max(float(distance_threshold) * 2.0, float(radius) * 0.25, np.finfo(float).eps)
        mask = residual <= tolerance
        if int(mask.sum()) < 10:
            mask = residual <= max(tolerance * 2.0, float(radius) * 0.5)
        candidate_proj = projections[mask] if int(mask.sum()) >= 2 else projections
        target_proj = float(np.dot(target_point - axis_point, axis))
        min_proj = float(np.min(candidate_proj))
        max_proj = float(np.max(candidate_proj))
        if prefer_axis_min:
            end_proj = min_proj
        else:
            end_proj = min_proj if abs(target_proj - min_proj) <= abs(target_proj - max_proj) else max_proj
        endpoint = np.asarray(axis_point + end_proj * axis, dtype=float)
        self.__console.info(
            "pipe endpoint from full PCD projection: "
            f"candidates={int(len(candidate_proj))}, "
            f"target_proj={target_proj:.5f}, min_proj={min_proj:.5f}, max_proj={max_proj:.5f}, "
            f"endpoint={np.round(endpoint, 5).tolist()}")
        return endpoint

    def _pipe_profile_near_far_centers(
        self,
        points,
        axis,
        axis_point,
        radius,
        target_point,
        distance_threshold,
    ):
        axis = self._unit_vector(axis)
        source_points = self._get_spool_points()
        if source_points is None:
            source_points = points
        points = np.asarray(source_points, dtype=float)
        if points.size == 0:
            center = np.asarray(axis_point, dtype=float)
            return center, center + axis, axis
        points = points.reshape((-1, 3))
        axis_point = np.asarray(axis_point, dtype=float)
        target_point = np.asarray(target_point, dtype=float)
        rel = points - axis_point
        projections = rel @ axis
        radial = rel - np.outer(projections, axis)
        residual = np.abs(np.linalg.norm(radial, axis=1) - float(radius))
        tolerance = max(float(distance_threshold) * 2.0, float(radius) * 0.25, np.finfo(float).eps)
        mask = residual <= tolerance
        if int(mask.sum()) < 10:
            mask = residual <= max(tolerance * 2.0, float(radius) * 0.5)
        candidate_proj = projections[mask] if int(mask.sum()) >= 2 else projections
        target_proj = float(np.dot(target_point - axis_point, axis))
        min_proj = float(np.min(candidate_proj))
        max_proj = float(np.max(candidate_proj))
        min_center = np.asarray(axis_point + min_proj * axis, dtype=float)
        max_center = np.asarray(axis_point + max_proj * axis, dtype=float)
        if abs(target_proj - min_proj) <= abs(target_proj - max_proj):
            near_center, far_center = min_center, max_center
        else:
            near_center, far_center = max_center, min_center
        near_to_far = self._unit_vector(far_center - near_center)
        if np.linalg.norm(near_to_far) < 1e-12:
            near_to_far = axis
        self.__console.info(
            "pipe near/far endpoints from full PCD projection: "
            f"candidates={int(len(candidate_proj))}, "
            f"target_proj={target_proj:.5f}, min_proj={min_proj:.5f}, max_proj={max_proj:.5f}, "
            f"near={np.round(near_center, 5).tolist()}, far={np.round(far_center, 5).tolist()}")
        return near_center, far_center, near_to_far

    def _pipe_axis_toward_sparse_side(self, axis, origin):
        axis = self._unit_vector(axis)
        points = self._get_spool_points()
        if points is None or len(points) == 0 or np.linalg.norm(axis) < 1e-12:
            return axis, 0, 0
        origin = np.asarray(origin, dtype=float)
        projections = (np.asarray(points, dtype=float).reshape((-1, 3)) - origin) @ axis
        eps = max(float(np.ptp(projections)) * 1e-4, 1e-9)
        positive_count = int(np.count_nonzero(projections > eps))
        negative_count = int(np.count_nonzero(projections < -eps))
        sparse_axis = axis if positive_count <= negative_count else -axis
        self.__console.info(
            "pipe sparse-side axis from full PCD: "
            f"positive_count={positive_count}, negative_count={negative_count}, "
            f"axis={np.round(sparse_axis, 5).tolist()}")
        return sparse_axis, positive_count, negative_count

    def _chuck_mount_profile_params(self):
        spool_points = self._get_spool_points()
        bbox_diag = float(np.linalg.norm(spool_points.max(axis=0) - spool_points.min(axis=0)))
        if bbox_diag > 10.0:
            defaults = {
                "sampling_size_for_calculating_normal": max(5.0, bbox_diag * 0.01),
                "radius_offset_for_sampling_points_in_sphere": 3.0,
                "sampling_cylinder_radius": 5.0,
                "sampling_cylinder_height_range": (-100.0, 300.0),
            }
        else:
            defaults = {
                "sampling_size_for_calculating_normal": 0.01,
                "radius_offset_for_sampling_points_in_sphere": 0.003,
                "sampling_cylinder_radius": 0.005,
                "sampling_cylinder_height_range": (-0.1, 0.3),
            }
        params = self._config.get("chuck_mount_profile", {}) or {}
        merged = defaults.copy()
        merged.update(params)
        merged["sampling_size_for_calculating_normal"] = float(merged["sampling_size_for_calculating_normal"])
        merged["radius_offset_for_sampling_points_in_sphere"] = float(merged["radius_offset_for_sampling_points_in_sphere"])
        merged["sampling_cylinder_radius"] = float(merged["sampling_cylinder_radius"])
        merged["sampling_cylinder_height_range"] = tuple(merged["sampling_cylinder_height_range"])
        return merged

    def _chuck_link_world_T(self, link_name):
        for model in getattr(self, '_robot_models', []):
            if hasattr(model, 'get_link_world_T'):
                T = model.get_link_world_T(link_name)
                if T is not None:
                    return np.asarray(T, dtype=float)
        return None

    def _positioner_robot_model(self):
        for model in getattr(self, '_robot_models', []):
            if getattr(model, "name", None) == "positioner":
                return model
        for model in getattr(self, '_robot_models', []):
            joint_map = model._urdf._joint_map if getattr(model, "_urdf", None) else {}
            if "base_to_m_column" in joint_map and "base_to_f_column_z" in joint_map:
                return model
        return None

    @staticmethod
    def _pose6_to_T(pose):
        x, y, z, roll, pitch, yaw = [float(v) for v in pose[:6]]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = [x, y, z]
        return T

    def _positioner_urdf_config(self):
        for item in self._config.get("urdf", []) or []:
            if item.get("name") == "positioner":
                return item
        return None

    def _positioner_pin_model_data(self):
        if pin is None:
            raise RuntimeError("Pinocchio is not available")
        cached = getattr(self, "_positioner_pin_cache", None)
        if cached is not None:
            return cached
        item = self._positioner_urdf_config()
        if item is None:
            raise RuntimeError("positioner URDF config is not available")
        root_path = self._config.get("root_path", "")
        urdf_path = item.get("path")
        if not urdf_path:
            raise RuntimeError("positioner URDF path is not available")
        full_path = urdf_path if os.path.isabs(urdf_path) else os.path.join(root_path, urdf_path)
        model = self._build_pin_model_from_urdf(full_path)
        data = model.createData()
        base_T = self._pose6_to_T(item.get("base", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        joint_names = [str(model.names[i]) for i in range(1, model.njoints)]
        cached = {
            "model": model,
            "data": data,
            "base_T": base_T,
            "joint_names": joint_names,
            "urdf_path": full_path,
        }
        self._positioner_pin_cache = cached
        self.__console.info(
            f"positioner Pinocchio model loaded: {full_path} "
            f"({len(joint_names)} joints: {joint_names})")
        return cached

    def _build_pin_model_from_urdf(self, urdf_path):
        if pin is None:
            raise RuntimeError("Pinocchio is not available")
        if hasattr(pin, "buildModelFromUrdf"):
            return pin.buildModelFromUrdf(urdf_path)
        if hasattr(pin, "buildModelFromURDF"):
            return pin.buildModelFromURDF(urdf_path)
        if hasattr(pin, "buildModelsFromUrdf"):
            models = pin.buildModelsFromUrdf(urdf_path)
            if isinstance(models, tuple) and models:
                return models[0]
        try:
            from pinocchio.robot_wrapper import RobotWrapper
            return RobotWrapper.BuildFromURDF(urdf_path).model
        except Exception as exc:
            available = ", ".join(name for name in (
                "buildModelFromUrdf",
                "buildModelFromURDF",
                "buildModelsFromUrdf",
                "robot_wrapper",
            ) if hasattr(pin, name))
            raise RuntimeError(
                "Pinocchio URDF model builder is not available "
                f"(module={getattr(pin, '__file__', None)}, available={available})"
            ) from exc

    def _positioner_pin_q_from_values(self, values):
        cache = self._positioner_pin_model_data()
        model = cache["model"]
        values = np.asarray(values, dtype=float)
        joint_values = {
            "base_to_m_column": -float(values[0]),
            "base_to_f_column_z": float(values[1]),
            "m_column_to_m_column_z": float(values[1]),
            "f_column_z_to_f_column_r": float(np.deg2rad(values[2])),
            "f_column_r_to_f_column_passive_clamp": -float(values[3]),
        }
        # Keep the same q packing convention as controller.manipulation.compute_fk:
        # pin_model.names excludes universe at index 0, and q[i] maps to names[i + 1].
        q = np.zeros(model.nq, dtype=float)
        for i, joint_name in enumerate(cache["joint_names"]):
            if i < len(q):
                q[i] = joint_values.get(joint_name, 0.0)
        return q

    def _positioner_pin_link_world_T(self, values, link_name):
        cache = self._positioner_pin_model_data()
        model = cache["model"]
        data = cache["data"]
        q = self._positioner_pin_q_from_values(values)
        pin.forwardKinematics(model, data, q)
        if hasattr(pin, "updateFramePlacements"):
            pin.updateFramePlacements(model, data)
        elif hasattr(pin, "framesForwardKinematics"):
            pin.framesForwardKinematics(model, data, q)
        else:
            raise RuntimeError("Pinocchio frame placement update API is not available")
        frame_id = model.getFrameId(link_name)
        if frame_id >= model.nframes:
            raise RuntimeError(f"Pinocchio frame is not available: {link_name}")
        placement = data.oMf[frame_id]
        T = np.eye(4)
        T[:3, :3] = np.asarray(placement.rotation, dtype=float)
        T[:3, 3] = np.asarray(placement.translation, dtype=float)
        return cache["base_T"] @ T

    def _positioner_chuck_center_axis_for_values(self, values, link_name):
        link_T = self._positioner_fk_link_world_T(values, link_name)
        return (
            self._chuck_center_world(link_name, link_T),
            self._unit_vector(self._chuck_axis_world(link_name, link_T)),
            link_T,
        )

    def _positioner_fk_link_world_T(self, values, link_name):
        if pin is not None:
            return self._positioner_pin_link_world_T(values, link_name)
        return self._positioner_robot_link_world_T(values, link_name)

    def _positioner_robot_link_world_T(self, values, link_name):
        model = self._positioner_robot_model()
        if model is None:
            raise RuntimeError("positioner RobotModel is not available")
        values = np.asarray(values, dtype=float)
        joint_map = model._urdf._joint_map if getattr(model, "_urdf", None) else {}
        joint_values = {
            "base_to_m_column": -float(values[0]),
            "base_to_f_column_z": float(values[1]),
            "m_column_to_m_column_z": float(values[1]),
            "f_column_z_to_f_column_r": float(np.deg2rad(values[2])),
            "f_column_r_to_f_column_passive_clamp": -float(values[3]),
        }
        for joint_name, value in joint_values.items():
            if joint_name in joint_map:
                model.set_joint(joint_name, value)
        model.update_fk()
        link_T = model.get_link_world_T(link_name)
        if link_T is None:
            raise RuntimeError(f"positioner link transform is not available: {link_name}")
        return np.asarray(link_T, dtype=float)

    def _chuck_frame_config(self, link_name):
        cfg = self._config.get("chuck_mount", {}) or {}
        if link_name == self.F_CHUCK_LINK_NAME:
            defaults = {
                "center_offset": [0.0, 0.0, 0.0],
                "axis": [1.0, 0.0, 0.0],
            }
            values = cfg.get("f_column", {}) or {}
        elif link_name == self.M_CHUCK_LINK_NAME:
            defaults = {
                "center_offset": [0.0, 0.0, 0.0],
                "axis": [-1.0, 0.0, 0.0],
            }
            values = cfg.get("m_column", {}) or {}
        else:
            defaults = {
                "center_offset": [0.0, 0.0, 0.0],
                "axis": [1.0, 0.0, 0.0],
            }
            values = {}
        merged = defaults.copy()
        merged.update(values)
        return merged

    def _chuck_center_world(self, link_name, link_T=None):
        if link_T is None:
            link_T = self._chuck_link_world_T(link_name)
        if link_T is None:
            raise RuntimeError(f"chuck link transform is not available: {link_name}")
        cfg = self._chuck_frame_config(link_name)
        offset = np.asarray(cfg.get("center_offset", [0.0, 0.0, 0.0]), dtype=float)
        return np.asarray(link_T[:3, :3] @ offset + link_T[:3, 3], dtype=float)

    def _chuck_axis_world(self, link_name, link_T=None):
        if link_T is None:
            link_T = self._chuck_link_world_T(link_name)
        if link_T is None:
            raise RuntimeError(f"chuck link transform is not available: {link_name}")
        cfg = self._chuck_frame_config(link_name)
        local_axis = np.asarray(cfg.get("axis", [1.0, 0.0, 0.0]), dtype=float)
        return np.asarray(link_T[:3, :3] @ local_axis, dtype=float)

    @staticmethod
    def _frame_from_primary_and_reference(primary, reference):
        x_axis = np.asarray(primary, dtype=float)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-12:
            raise RuntimeError("cannot build frame from zero-length primary vector")
        x_axis = x_axis / x_norm
        ref = np.asarray(reference, dtype=float)
        ref = ref - np.dot(ref, x_axis) * x_axis
        if np.linalg.norm(ref) < 1e-12:
            ref = np.array([0.0, 0.0, 1.0])
            if abs(float(np.dot(ref, x_axis))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            ref = ref - np.dot(ref, x_axis) * x_axis
        y_axis = ref / np.linalg.norm(ref)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack([x_axis, y_axis, z_axis])

    def _align_spool_profiles_to_chucks(self, target_points, identity=None):
        try:
            if not self._ensure_spool_frame_from_actor():
                raise RuntimeError("spool frame is not available")

            f_T = self._chuck_link_world_T(self.F_CHUCK_LINK_NAME)
            m_T = self._chuck_link_world_T(self.M_CHUCK_LINK_NAME)
            if f_T is None or m_T is None:
                raise RuntimeError("f/m chuck link transforms are not available")

            f_profile = self._profile_for_chuck_mount_point(target_points[0])
            m_profile = self._profile_for_chuck_mount_point(target_points[1])
            f_axis = np.asarray(f_profile["axis"], dtype=float)
            m_axis = np.asarray(m_profile["axis"], dtype=float)
            f_chuck_axis = self._chuck_axis_world(self.F_CHUCK_LINK_NAME, f_T)
            m_chuck_axis = self._chuck_axis_world(self.M_CHUCK_LINK_NAME, m_T)
            if np.dot(f_axis, f_chuck_axis) < 0.0:
                f_axis = -f_axis
            if np.dot(m_axis, m_chuck_axis) < 0.0:
                m_axis = -m_axis

            source_f = f_profile["center"]
            source_m = m_profile["center"]
            target_f = self._chuck_center_world(self.F_CHUCK_LINK_NAME, f_T)
            target_m = self._chuck_center_world(self.M_CHUCK_LINK_NAME, m_T)

            # First fixture: fix the selected f-column pipe profile to the f-column chuck.
            R_align = self._rotation_between_vectors(f_axis, f_chuck_axis)
            T_align = np.eye(4)
            T_align[:3, :3] = R_align
            T_align[:3, 3] = target_f - R_align @ source_f

            self._spool_world_T = T_align @ getattr(self, '_spool_world_T')
            self._apply_spool_world_T()
            self._update_chuck_mount_points_after_transform(T_align)
            self._send_spool_pose_update(identity=identity)
            self._show_two_chuck_profile_alignment(
                f_profile,
                m_profile,
                f_T,
                m_T,
                T_align,
            )
            self.plotter.render()

            aligned_m = R_align @ source_m + T_align[:3, 3]
            m_center_delta = aligned_m - target_m
            center_error = float(np.linalg.norm(m_center_delta))
            suggested_m_x_delta = float(np.dot(m_center_delta, m_T[:3, :3] @ np.array([1.0, 0.0, 0.0])))
            suggested_m_z_delta = float(m_center_delta[2])
            f_axis_error = float(np.rad2deg(np.arccos(np.clip(np.dot(
                self._unit_vector(R_align @ f_axis), self._unit_vector(f_chuck_axis)), -1.0, 1.0))))
            m_axis_error = float(np.rad2deg(np.arccos(np.clip(np.dot(
                self._unit_vector(R_align @ m_axis), self._unit_vector(m_chuck_axis)), -1.0, 1.0))))
            self.__console.info(
                "f-column fixed; m-column target measured: "
                f"m_center_error={center_error:.6f}, "
                f"suggested_m_x_delta={suggested_m_x_delta:.6f}, "
                f"suggested_m_z_delta={suggested_m_z_delta:.6f}, "
                f"f_axis_error={f_axis_error:.2f}deg, "
                f"m_axis_error={m_axis_error:.2f}deg")
        except Exception as exc:
            self.__console.error(f"two chuck profile alignment failed: {exc}")

    def _apply_positioner_pose_values(self, x=None, z=None, r=None, clamp=None, update_frames=True):
        if x is not None:
            self._positioner_x = float(x)
        if z is not None:
            self._positioner_z = float(z)
        if r is not None:
            self._positioner_r_deg = float(r)
        if clamp is not None:
            self._positioner_clamp = float(clamp)
        x_val = float(getattr(self, '_positioner_x', 0.0))
        z_val = float(getattr(self, '_positioner_z', 0.0))
        r_val = float(getattr(self, '_positioner_r_deg', 0.0))
        clamp_val = float(getattr(self, '_positioner_clamp', 0.0))
        import math
        for model in getattr(self, '_robot_models', []):
            joint_map = model._urdf._joint_map if model._urdf else {}
            if "base_to_m_column" in joint_map:
                model.set_joint("base_to_m_column", -x_val)
            if "base_to_f_column_z" in joint_map:
                model.set_joint("base_to_f_column_z", z_val)
            if "m_column_to_m_column_z" in joint_map:
                model.set_joint("m_column_to_m_column_z", z_val)
            if "f_column_z_to_f_column_r" in joint_map:
                model.set_joint("f_column_z_to_f_column_r", math.radians(r_val))
            if "f_column_r_to_f_column_passive_clamp" in joint_map:
                model.set_joint("f_column_r_to_f_column_passive_clamp", -clamp_val)
            model.update_fk()
        if update_frames:
            self._show_chuck_frames(render=False)

    def _send_positioner_pose_update(self, identity=None):
        if hasattr(self, 'zapi') and self.zapi and identity and hasattr(self.zapi, 'update_positioner_pose'):
            self.zapi.update_positioner_pose(
                {
                    "x": float(getattr(self, '_positioner_x', 0.0)),
                    "z": float(getattr(self, '_positioner_z', 0.0)),
                    "r": float(getattr(self, '_positioner_r_deg', 0.0)),
                    "clamp": float(getattr(self, '_positioner_clamp', 0.0)),
                },
                identity=identity,
            )

    def _align_column_to_profile(self, target_point, identity=None, link_name=None, label="column"):
        try:
            profile = self._profile_for_chuck_mount_point(target_point)
            link_name = link_name or self.M_CHUCK_LINK_NAME
            chuck_T = self._chuck_link_world_T(link_name)
            if chuck_T is None:
                raise RuntimeError(f"{label} chuck link transform is not available")
            profile_axis = np.asarray(profile["axis"], dtype=float)
            pipe_center = np.asarray(profile["center"], dtype=float)
            pipe_radius = float(profile["radius"])
            chuck_center = self._chuck_center_world(link_name, chuck_T)
            chuck_axis = self._chuck_axis_world(link_name, chuck_T)
            near_center, far_center, near_to_far_axis = self._pipe_profile_near_far_centers(
                profile.get("fit_points"),
                profile_axis,
                pipe_center,
                pipe_radius,
                target_point,
                self._profile_distance_threshold(target_point),
            )
            profile_center = np.asarray(near_center, dtype=float)
            profile_axis = np.asarray(near_to_far_axis, dtype=float)
            if np.dot(profile_axis, chuck_axis) < 0.0:
                profile_axis = -profile_axis
            profile["axis"] = profile_axis
            profile["end_center"] = profile_center
            profile["far_end_center"] = np.asarray(far_center, dtype=float)
            self.__console.info(
                f"{label} profile endpoint selected: "
                f"clicked={np.round(np.asarray(target_point, dtype=float), 4)}, "
                f"near={np.round(profile_center, 4)}, "
                f"far={np.round(np.asarray(far_center, dtype=float), 4)}, "
                f"axis={np.round(profile_axis, 4)}")

            if link_name == self.F_CHUCK_LINK_NAME:
                self._align_f_column_positioner_to_profile(
                    profile,
                    profile_center,
                    profile_axis,
                    identity=identity,
                    label=label,
                )
                return

            delta = profile_center - chuck_center
            current_x = float(getattr(self, '_positioner_x', 0.0))
            current_z = float(getattr(self, '_positioner_z', 0.0))
            current_r = float(getattr(self, '_positioner_r_deg', 0.0))
            current_clamp = float(getattr(self, '_positioner_clamp', 0.0))
            if link_name == self.M_CHUCK_LINK_NAME:
                # base_to_m_column joint is set as -UI x in the existing positioner command path.
                suggested_x = current_x - float(delta[0])
                suggested_r = current_r
                suggested_clamp = current_clamp
            else:
                suggested_x = current_x
                rotation_axis = chuck_T[:3, :3] @ np.array([1.0, 0.0, 0.0])
                r_delta = self._signed_angle_about_axis(chuck_axis, profile_axis, rotation_axis)
                suggested_r = current_r + float(np.rad2deg(r_delta))
                suggested_clamp = current_clamp
            suggested_z = current_z + float(delta[2])

            self._apply_positioner_pose_values(x=suggested_x, z=suggested_z, r=suggested_r, clamp=suggested_clamp)
            if link_name == self.F_CHUCK_LINK_NAME:
                after_zr_T = self._chuck_link_world_T(link_name)
                if after_zr_T is not None:
                    after_zr_center = self._chuck_center_world(link_name, after_zr_T)
                    clamp_axis = self._unit_vector(after_zr_T[:3, 1])
                    residual = profile_center - after_zr_center
                    suggested_clamp = current_clamp - float(np.dot(residual, clamp_axis))
                    self._apply_positioner_pose_values(
                        x=suggested_x,
                        z=suggested_z,
                        r=suggested_r,
                        clamp=suggested_clamp,
                    )
            self._send_positioner_pose_update(identity=identity)
            self._save_spool_alignment_state(reason=f"{label} align")

            updated_T = self._chuck_link_world_T(link_name)
            updated_center = (
                self._chuck_center_world(link_name, updated_T)
                if updated_T is not None else chuck_center
            )
            updated_axis = self._chuck_axis_world(link_name, updated_T) if updated_T is not None else chuck_axis
            axis_error = float(np.rad2deg(np.arccos(np.clip(np.dot(
                self._unit_vector(profile_axis), self._unit_vector(updated_axis)), -1.0, 1.0))))
            center_error = float(np.linalg.norm(profile_center - updated_center))
            profile_for_ui = dict(profile)
            profile_for_ui["center"] = profile_center
            profile_for_ui["center_error"] = center_error
            profile_for_ui["axis_error_deg"] = axis_error
            self._send_chuck_mount_profile_update(label, profile_for_ui, identity=identity)
            self._show_column_profile_alignment(profile, updated_T if updated_T is not None else chuck_T, link_name)
            self.plotter.render()
            self.__console.info(
                f"{label} aligned to profile: "
                f"x={suggested_x:.6f}, z={suggested_z:.6f}, r={suggested_r:.3f}, clamp={suggested_clamp:.6f}, "
                f"center_error={center_error:.6f}, axis_error={axis_error:.2f}deg")
        except Exception as exc:
            self.__console.error(f"{label} profile alignment failed: {exc}")

    def _align_m_column_to_profile(self, target_point, identity=None):
        self._align_column_to_profile(
            target_point,
            identity=identity,
            link_name=self.M_CHUCK_LINK_NAME,
            label="m-column")

    def _align_f_column_positioner_to_profile(self, profile, target_center, target_axis, identity=None, label="f-column"):
        initial = np.array([
            float(getattr(self, '_positioner_x', 0.0)),
            float(getattr(self, '_positioner_z', 0.0)),
            float(getattr(self, '_positioner_r_deg', 0.0)),
            float(getattr(self, '_positioner_clamp', 0.0)),
        ], dtype=float)
        try:
            self._ensure_spool_frame_from_actor()
            target_center = np.asarray(target_center, dtype=float)
            target_axis = self._unit_vector(target_axis)
            bounds = np.array([(0.0, 4.7), (0.0, 0.85), (-180.0, 180.0), (0.0, 0.9)], dtype=float)
            current = np.array([
                np.clip(initial[0], bounds[0, 0], bounds[0, 1]),
                np.clip(initial[1], bounds[1, 0], bounds[1, 1]),
                np.clip(initial[2], bounds[2, 0], bounds[2, 1]),
                np.clip(initial[3], bounds[3, 0], bounds[3, 1]),
            ], dtype=float)
            initial_m_T = self._positioner_fk_link_world_T(current, self.M_CHUCK_LINK_NAME)
            initial_m_T_inv = np.linalg.inv(initial_m_T)
            self._log_f_column_joint_sensitivity(current)

            def moved_target_for_values(values):
                current_m_T = self._positioner_fk_link_world_T(values, self.M_CHUCK_LINK_NAME)
                delta_T = current_m_T @ initial_m_T_inv
                moved_center = delta_T[:3, :3] @ target_center + delta_T[:3, 3]
                moved_axis = self._unit_vector(delta_T[:3, :3] @ target_axis)
                return moved_center, moved_axis, delta_T

            # base_to_m_column translates the M-fixed pipe in world X; base_to_f_column_z
            # translates both pipe and F column in Z, so solve the reachable X/Z shift first.
            current_f_center, _, _ = self._positioner_chuck_center_axis_for_values(
                current, self.F_CHUCK_LINK_NAME)
            # UI x is applied to the URDF as base_to_m_column = -x, so increasing
            # x moves the M-fixed pipe in world -X while the F column stays put.
            x_delta = target_center[0] - current_f_center[0]
            z_delta = target_center[2] - current_f_center[2]
            best = current.copy()
            best[0] = np.clip(best[0] + x_delta, bounds[0, 0], bounds[0, 1])
            best[1] = np.clip(best[1] + z_delta, bounds[1, 0], bounds[1, 1])

            moved_center, moved_axis, _ = moved_target_for_values(best)
            r_zero_values = best.copy()
            r_zero_values[2] = 0.0
            r_zero_values[3] = 0.0
            f_r_zero_T = self._positioner_fk_link_world_T(r_zero_values, "f_column_r")
            target_local = np.linalg.inv(f_r_zero_T) @ np.array([
                moved_center[0], moved_center[1], moved_center[2], 1.0
            ])
            f_cfg = self._chuck_frame_config(self.F_CHUCK_LINK_NAME)
            f_offset = np.asarray(f_cfg.get("center_offset", [0.0, 0.0, 0.0]), dtype=float)
            clamp_origin = np.array([0.427, 0.9, 0.0], dtype=float)
            local_y = float(target_local[1])
            local_z = float(target_local[2])
            offset_z = float(f_offset[2])
            local_radius = float(np.hypot(local_y, local_z))
            radial_without_z = max(local_radius * local_radius - offset_z * offset_z, 0.0)
            clamp_reach = float(np.sqrt(radial_without_z))
            unclipped_clamp = float(clamp_origin[1] + f_offset[1] - clamp_reach)
            best[3] = np.clip(unclipped_clamp, bounds[3, 0], bounds[3, 1])
            solved_pre_rotation = np.array([
                clamp_origin[0] + f_offset[0],
                clamp_origin[1] - best[3] + f_offset[1],
                f_offset[2],
            ], dtype=float)
            theta_target = float(np.arctan2(local_z, local_y))
            theta_source = float(np.arctan2(solved_pre_rotation[2], solved_pre_rotation[1]))
            solved_r = float(np.rad2deg(theta_target - theta_source))
            solved_r = ((solved_r + 180.0) % 360.0) - 180.0
            best[2] = np.clip(solved_r, bounds[2, 0], bounds[2, 1])
            self.__console.info(
                "f-column analytic r/clamp solve | "
                f"target_local={np.round(target_local[:3], 5).tolist()}, "
                f"r={best[2]:.3f}deg, clamp={best[3]:.6f}, "
                f"unclipped_clamp={unclipped_clamp:.6f}")
            self._apply_positioner_pose_values(
                x=float(best[0]),
                z=float(best[1]),
                r=float(best[2]),
                clamp=float(best[3]),
                update_frames=True,
            )

            updated_center, updated_axis, updated_T = self._positioner_chuck_center_axis_for_values(
                best, self.F_CHUCK_LINK_NAME)
            moved_center, moved_axis, final_m_delta_T = moved_target_for_values(best)
            if np.dot(updated_axis, moved_axis) < 0.0:
                updated_axis = -updated_axis
            center_error = float(np.linalg.norm(moved_center - updated_center))
            axis_error = float(np.rad2deg(np.arccos(np.clip(np.dot(moved_axis, updated_axis), -1.0, 1.0))))
            max_center_error = float((self._config.get("chuck_mount_profile", {}) or {}).get(
                "f_align_max_center_error", max(float(profile.get("radius", 0.02)) * 2.0, 0.05)))
            max_axis_error = float((self._config.get("chuck_mount_profile", {}) or {}).get(
                "f_align_max_axis_error_deg", 5.0))
            if center_error > max_center_error or axis_error > max_axis_error:
                self.__console.warning(
                    "f-column alignment did not converge, applying best effort: "
                    f"center_error={center_error:.6f} (limit={max_center_error:.6f}), "
                    f"axis_error={axis_error:.2f}deg (limit={max_axis_error:.2f}deg), "
                    f"best={np.round(best, 6).tolist()}")
            if getattr(self, '_spool_world_T', None) is not None:
                self._spool_world_T = final_m_delta_T @ self._spool_world_T
                self._apply_spool_world_T()
                self._update_chuck_mount_points_after_transform(final_m_delta_T)
                self._send_spool_pose_update(identity=identity)
            moved_profile = self._transformed_pipe_profile(profile, final_m_delta_T)
            profile_for_ui = dict(moved_profile)
            profile_for_ui["center"] = moved_center
            profile_for_ui["axis"] = moved_axis
            profile_for_ui["center_error"] = center_error
            profile_for_ui["axis_error_deg"] = axis_error
            self._send_positioner_pose_update(identity=identity)
            self._send_chuck_mount_profile_update(label, profile_for_ui, identity=identity)
            self._show_column_profile_alignment(moved_profile, updated_T, self.F_CHUCK_LINK_NAME)
            self.plotter.render()
            fk_backend = "Pinocchio" if pin is not None else "RobotModel"
            self.__console.info(
                f"{label} {fk_backend} FK aligned to profile: "
                f"x={best[0]:.6f}, z={best[1]:.6f}, r={best[2]:.3f}, clamp={best[3]:.6f}, "
                f"center_error={center_error:.6f}, axis_error={axis_error:.2f}deg")
        except Exception as exc:
            self._apply_positioner_pose_values(
                x=float(initial[0]),
                z=float(initial[1]),
                r=float(initial[2]),
                clamp=float(initial[3]),
                update_frames=True,
            )
            self.__console.error(f"f-column positioner optimization failed: {exc}")

    def _log_f_column_joint_sensitivity(self, values):
        try:
            values = np.asarray(values, dtype=float)
            base_center, base_axis, _ = self._positioner_chuck_center_axis_for_values(
                values, self.F_CHUCK_LINK_NAME)
            probes = (
                ("base_to_m_column", np.array([0.01, 0.0, 0.0, 0.0])),
                ("base_to_f_column_z", np.array([0.0, 0.01, 0.0, 0.0])),
                ("f_column_z_to_f_column_r", np.array([0.0, 0.0, 1.0, 0.0])),
                ("f_column_r_to_f_column_passive_clamp", np.array([0.0, 0.0, 0.0, 0.01])),
            )
            parts = []
            for name, step in probes:
                probe = values + step
                center, axis, _ = self._positioner_chuck_center_axis_for_values(
                    probe, self.F_CHUCK_LINK_NAME)
                d_center = center - base_center
                d_axis_deg = float(np.rad2deg(np.arccos(np.clip(np.dot(base_axis, axis), -1.0, 1.0))))
                parts.append(f"{name}: dC={np.round(d_center, 5).tolist()}, dA={d_axis_deg:.3f}deg")
            self.__console.info("f-column Pinocchio joint sensitivity | " + " | ".join(parts))
        except Exception as exc:
            self.__console.warning(f"failed to log f-column joint sensitivity: {exc}")

    def _transformed_pipe_profile(self, profile, transform):
        transformed = dict(profile)
        R = np.asarray(transform[:3, :3], dtype=float)
        t = np.asarray(transform[:3, 3], dtype=float)
        for key in ("center", "end_center"):
            if key in transformed and transformed[key] is not None:
                transformed[key] = R @ np.asarray(transformed[key], dtype=float) + t
        if "axis" in transformed and transformed["axis"] is not None:
            transformed["axis"] = self._unit_vector(R @ np.asarray(transformed["axis"], dtype=float))
        if "fit_points" in transformed and transformed["fit_points"] is not None:
            pts = np.asarray(transformed["fit_points"], dtype=float)
            transformed["fit_points"] = (R @ pts.T).T + t
        return transformed

    def _update_chuck_mount_points_after_transform(self, transform):
        updated_points = []
        for point, local_point in zip(self._chuck_mount_points, self._chuck_mount_local_points):
            if local_point is not None:
                local_h = np.ones(4, dtype=float)
                local_h[:3] = np.asarray(local_point, dtype=float)
                updated_points.append((self._spool_world_T @ local_h)[:3])
            else:
                updated_points.append(transform[:3, :3] @ np.asarray(point, dtype=float) + transform[:3, 3])
        self._chuck_mount_points = updated_points
        self._refresh_chuck_mount_markers()

    def _add_profile_cylinder_actor(self, center, axis, radius, length, color="cyan", alpha=0.22):
        try:
            center = np.asarray(center, dtype=float)
            axis = self._unit_vector(axis)
            radius = float(radius)
            length = float(length)
            if radius <= 0.0 or length <= 0.0 or np.linalg.norm(axis) < 1e-12:
                return
            ref = np.array([0.0, 0.0, 1.0])
            if abs(float(np.dot(ref, axis))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            u = self._unit_vector(np.cross(axis, ref))
            v = self._unit_vector(np.cross(axis, u))
            n = 64
            half = length * 0.5
            verts = []
            for z in (-half, half):
                cap_center = center + axis * z
                for i in range(n):
                    theta = 2.0 * np.pi * i / n
                    verts.append(cap_center + radius * (np.cos(theta) * u + np.sin(theta) * v))
            faces = []
            for i in range(n):
                j = (i + 1) % n
                faces.append([i, j, n + j, n + i])
            faces.append(list(range(n - 1, -1, -1)))
            faces.append(list(range(n, 2 * n)))
            actor = vedo.Mesh([np.asarray(verts, dtype=float), faces])
            actor.c(color).alpha(alpha).wireframe()
            actor.pickable(False)
            self._chuck_profile_actors.append(actor)
            self.plotter.add(actor)
        except Exception as exc:
            self.__console.warning(f"Failed to draw profile cylinder: {exc}")

    def _add_profile_fit_points_actor(self, points, color="magenta"):
        if points is None:
            return
        try:
            points = np.asarray(points, dtype=float)
            if len(points) == 0:
                return
            actor = vedo.Points(points).c(color).ps(4)
            actor.pickable(False)
            self._chuck_profile_actors.append(actor)
            self.plotter.add(actor)
        except Exception as exc:
            self.__console.warning(f"Failed to draw profile fit points: {exc}")

    def _add_alignment_reference_actor(
        self,
        origin,
        axis,
        axis_len,
        label="ALIGN_REF",
        color="yellow",
        far_point=None,
    ):
        try:
            origin = np.asarray(origin, dtype=float)
            axis = self._unit_vector(axis)
            axis_len = float(axis_len)
            if np.linalg.norm(axis) < 1e-12 or axis_len <= 0.0:
                raise RuntimeError("__ef_pose_collision_groups_rendered__")
            marker = vedo.Sphere(pos=origin, r=max(axis_len * 0.07, 0.012), c=color)
            marker.pickable(False)
            self._chuck_profile_actors.append(marker)
            self.plotter.add(marker)

            try:
                arrow = vedo.Arrow(origin, origin + axis * axis_len, s=0.002, c=color)
            except Exception:
                arrow = vedo.Line(origin, origin + axis * axis_len, c=color, lw=8)
            arrow.pickable(False)
            self._chuck_profile_actors.append(arrow)
            self.plotter.add(arrow)

            if far_point is not None:
                far_point = np.asarray(far_point, dtype=float)
                far_marker = vedo.Sphere(pos=far_point, r=max(axis_len * 0.045, 0.009), c="gray")
                far_marker.pickable(False)
                self._chuck_profile_actors.append(far_marker)
                self.plotter.add(far_marker)
                far_line = vedo.Line(origin, far_point, c=color, lw=2)
                far_line.alpha(0.45)
                far_line.pickable(False)
                self._chuck_profile_actors.append(far_line)
                self.plotter.add(far_line)

            text = vedo.Text3D(label, pos=origin + axis * axis_len * 1.06, s=axis_len * 0.12, c=color)
            text.pickable(False)
            self._chuck_profile_actors.append(text)
            self.plotter.add(text)
        except Exception as exc:
            self.__console.warning(f"Failed to draw alignment reference: {exc}")

    def _show_two_chuck_profile_alignment(self, f_profile, m_profile, f_T, m_T, transform):
        self._clear_chuck_profile_visuals(render=False)
        profile_items = [
            (f_profile, f_T, "cyan", "lime"),
            (m_profile, m_T, "deepskyblue", "green"),
        ]
        for profile, chuck_T, profile_color, chuck_color in profile_items:
            center = np.asarray(profile["center"], dtype=float)
            axis = np.asarray(profile["axis"], dtype=float)
            radius = float(profile["radius"])
            aligned_center = transform[:3, :3] @ center + transform[:3, 3]
            aligned_axis = transform[:3, :3] @ axis
            chuck_link_name = self.F_CHUCK_LINK_NAME if chuck_T is f_T else self.M_CHUCK_LINK_NAME
            chuck_center = self._chuck_center_world(chuck_link_name, chuck_T)
            chuck_axis = self._chuck_axis_world(
                chuck_link_name,
                chuck_T,
            )
            axis_len = max(radius * 8.0, 0.15)
            self._add_profile_cylinder_actor(
                aligned_center,
                aligned_axis,
                radius,
                axis_len * 2.0,
                color=profile_color,
            )
            for start, vec, color in (
                (aligned_center, aligned_axis, profile_color),
                (chuck_center, chuck_axis, chuck_color),
            ):
                vec = self._unit_vector(vec)
                actor = vedo.Line(start, start + vec * axis_len, c=color, lw=5)
                actor.pickable(False)
                self._chuck_profile_actors.append(actor)
                self.plotter.add(actor)
            marker = vedo.Sphere(pos=aligned_center, r=max(radius * 0.10, 0.01), c=profile_color)
            marker.pickable(False)
            self._chuck_profile_actors.append(marker)
            self.plotter.add(marker)
        m_center = np.asarray(m_profile["center"], dtype=float)
        aligned_m_center = transform[:3, :3] @ m_center + transform[:3, 3]
        m_chuck_center = np.asarray(m_T[:3, 3], dtype=float)
        error_line = vedo.Line(m_chuck_center, aligned_m_center, c="red", lw=4)
        error_line.pickable(False)
        self._chuck_profile_actors.append(error_line)
        self.plotter.add(error_line)

    def _show_column_profile_alignment(self, profile, chuck_T, link_name):
        self._clear_chuck_profile_visuals(render=False)
        cylinder_center = np.asarray(profile["center"], dtype=float)
        profile_center = np.asarray(profile.get("end_center", cylinder_center), dtype=float)
        profile_axis = self._unit_vector(profile["axis"])
        radius = float(profile["radius"])
        chuck_center = self._chuck_center_world(link_name, chuck_T)
        chuck_axis = self._unit_vector(self._chuck_axis_world(link_name, chuck_T))
        axis_len = max(radius * 8.0, 0.15)
        self._add_profile_cylinder_actor(
            cylinder_center,
            profile_axis,
            radius,
            axis_len * 2.0,
            color="cyan",
        )
        self._add_profile_fit_points_actor(profile.get("fit_points"), color="magenta")
        self._add_alignment_reference_actor(
            profile_center,
            profile_axis,
            axis_len * 1.35,
            label="PIPE_ALIGN_REF",
            color="yellow",
            far_point=profile.get("far_end_center"),
        )
        self._add_alignment_reference_actor(
            chuck_center,
            chuck_axis,
            axis_len * 1.15,
            label="CHUCK_TARGET",
            color="lime",
        )
        for start, vec, color in (
            (cylinder_center, profile_axis, "cyan"),
            (chuck_center, chuck_axis, "green"),
        ):
            actor = vedo.Line(start, start + vec * axis_len, c=color, lw=5)
            actor.pickable(False)
            self._chuck_profile_actors.append(actor)
            self.plotter.add(actor)
        error_line = vedo.Line(chuck_center, profile_center, c="red", lw=4)
        error_line.pickable(False)
        self._chuck_profile_actors.append(error_line)
        self.plotter.add(error_line)
        marker = vedo.Sphere(pos=profile_center, r=max(radius * 0.10, 0.01), c="cyan")
        marker.pickable(False)
        self._chuck_profile_actors.append(marker)
        self.plotter.add(marker)

    def _show_m_column_profile_alignment(self, profile, m_T):
        self._show_column_profile_alignment(profile, m_T, self.M_CHUCK_LINK_NAME)

    def _show_chuck_profile_alignment(self, pipe_center, pipe_axis, pipe_radius, chuck_center, chuck_axis, transform, fit_points=None, pipe_origin=None):
        self._clear_chuck_profile_visuals(render=False)
        pipe_center = np.asarray(pipe_center, dtype=float)
        pipe_axis = np.asarray(pipe_axis, dtype=float)
        chuck_center = np.asarray(chuck_center, dtype=float)
        chuck_axis = np.asarray(chuck_axis, dtype=float)
        aligned_center = transform[:3, :3] @ pipe_center + transform[:3, 3]
        aligned_axis = transform[:3, :3] @ pipe_axis
        if pipe_origin is None:
            pipe_origin = pipe_center
        aligned_origin = transform[:3, :3] @ np.asarray(pipe_origin, dtype=float) + transform[:3, 3]
        axis_len = max(float(pipe_radius) * 8.0, 0.15)
        self._add_profile_cylinder_actor(
            aligned_center,
            aligned_axis,
            pipe_radius,
            axis_len * 2.0,
            color="cyan",
        )
        if fit_points is not None:
            fit_points = np.asarray(fit_points, dtype=float)
            aligned_fit_points = (transform[:3, :3] @ fit_points.T).T + transform[:3, 3]
            self._add_profile_fit_points_actor(aligned_fit_points, color="magenta")
        self._add_alignment_reference_actor(
            aligned_origin,
            aligned_axis,
            axis_len * 1.35,
            label="PIPE_ALIGN_REF",
            color="yellow",
        )
        self._add_alignment_reference_actor(
            chuck_center,
            chuck_axis,
            axis_len * 1.15,
            label="CHUCK_TARGET",
            color="lime",
        )
        for start, axis, color in (
            (aligned_center, aligned_axis, "cyan"),
            (chuck_center, chuck_axis, "lime"),
        ):
            end = np.asarray(start, dtype=float) + np.asarray(axis, dtype=float) / np.linalg.norm(axis) * axis_len
            actor = vedo.Line(start, end, c=color, lw=5)
            actor.pickable(False)
            self._chuck_profile_actors.append(actor)
            self.plotter.add(actor)
        error_line = vedo.Line(chuck_center, aligned_origin, c="red", lw=4)
        error_line.pickable(False)
        self._chuck_profile_actors.append(error_line)
        self.plotter.add(error_line)
        marker = vedo.Sphere(pos=aligned_origin, r=max(float(pipe_radius) * 0.10, 0.01), c="cyan")
        marker.pickable(False)
        self._chuck_profile_actors.append(marker)
        self.plotter.add(marker)

    def _clear_chuck_frame_visuals(self, render=True):
        for actor in getattr(self, '_chuck_frame_actors', []) or []:
            try:
                self.plotter.remove(actor)
            except Exception:
                pass
        self._chuck_frame_actors = []
        if render:
            self.plotter.render()

    def _add_frame_visual(self, center, R, label, axis_len, center_color):
        center = np.asarray(center, dtype=float)
        colors = ("red", "green", "blue")
        axis_names = ("X", "Y", "Z")
        for i in range(3):
            axis = np.asarray(R[:, i], dtype=float)
            end = center + self._unit_vector(axis) * axis_len
            line = vedo.Line(center, end, c=colors[i], lw=5)
            line.pickable(False)
            self._chuck_frame_actors.append(line)
            self.plotter.add(line)
            self._add_chuck_frame_text(f"{label}-{axis_names[i]}", end, axis_len * 0.08, colors[i])
        sphere = vedo.Sphere(pos=center, r=axis_len * 0.07, c=center_color)
        sphere.pickable(False)
        self._chuck_frame_actors.append(sphere)
        self.plotter.add(sphere)
        self._add_chuck_frame_text(
            label,
            center + np.array([0.0, 0.0, axis_len * 0.18]),
            axis_len * 0.1,
            center_color,
        )

    def _add_chuck_frame_text(self, text, pos, size, color):
        try:
            actor = vedo.Text3D(text, pos=pos, s=size, c=color)
            actor.pickable(False)
            self._chuck_frame_actors.append(actor)
            self.plotter.add(actor)
        except Exception as exc:
            self.__console.warning(f"Failed to draw chuck frame label '{text}': {exc}")

    def _show_chuck_frames(self, render=True):
        self._clear_chuck_frame_visuals(render=False)
        if getattr(self, '_spool_positioner_fixed', False):
            if render:
                self.plotter.render()
            return
        f_T = self._chuck_link_world_T(self.F_CHUCK_LINK_NAME)
        m_T = self._chuck_link_world_T(self.M_CHUCK_LINK_NAME)
        if f_T is None or m_T is None:
            if render:
                self.plotter.render()
            return
        pts = [f_T[:3, 3], m_T[:3, 3], self._chuck_center_world(self.F_CHUCK_LINK_NAME, f_T), self._chuck_center_world(self.M_CHUCK_LINK_NAME, m_T)]
        extent = float(np.linalg.norm(np.max(pts, axis=0) - np.min(pts, axis=0)))
        axis_len = max(extent * 0.12, 0.18)
        self._add_frame_visual(f_T[:3, 3], f_T[:3, :3], "F_LINK", axis_len, "orange")
        self._add_frame_visual(m_T[:3, 3], m_T[:3, :3], "M_LINK", axis_len, "purple")

        for link_name, label, color in (
            (self.F_CHUCK_LINK_NAME, "F_CHUCK", "cyan"),
            (self.M_CHUCK_LINK_NAME, "M_CHUCK", "yellow"),
        ):
            T = f_T if link_name == self.F_CHUCK_LINK_NAME else m_T
            link_origin = np.asarray(T[:3, 3], dtype=float)
            center = self._chuck_center_world(link_name, T)
            axis = self._unit_vector(self._chuck_axis_world(link_name, T))
            offset_line = vedo.Line(link_origin, center, c="white", lw=4)
            offset_line.pickable(False)
            self._chuck_frame_actors.append(offset_line)
            self.plotter.add(offset_line)
            line = vedo.Line(center, center + axis * axis_len * 1.25, c=color, lw=8)
            line.pickable(False)
            self._chuck_frame_actors.append(line)
            self.plotter.add(line)
            marker = vedo.Sphere(pos=center, r=axis_len * 0.09, c=color)
            marker.pickable(False)
            self._chuck_frame_actors.append(marker)
            self.plotter.add(marker)
            self._add_chuck_frame_text(label, center + axis * axis_len * 1.35, axis_len * 0.1, color)
            self._add_chuck_frame_text(
                f"{label}_OFFSET",
                (link_origin + center) * 0.5,
                axis_len * 0.075,
                "white",
            )
        if render:
            self.plotter.render()

    def _set_chuck_mount_config(self, chuck_mount_config):
        if not isinstance(chuck_mount_config, dict):
            return
        current = self._config.setdefault("chuck_mount", {})
        for column_name in ("f_column", "m_column"):
            values = chuck_mount_config.get(column_name)
            if not isinstance(values, dict):
                continue
            column_cfg = current.setdefault(column_name, {})
            for key in ("center_offset", "axis"):
                if key not in values:
                    continue
                try:
                    vec = np.asarray(values[key], dtype=float).reshape(3)
                    column_cfg[key] = vec.tolist()
                except Exception:
                    self.__console.warning(f"Invalid chuck_mount.{column_name}.{key}: {values[key]}")
        self._show_chuck_frames(render=True)
        self.__console.info(f"chuck mount config updated: {current}")

    def _clear_ik_failure_visuals(self, render=True):
        for actor in getattr(self, '_ik_failure_actors', []) or []:
            try:
                self.plotter.remove(actor)
            except Exception:
                pass
        self._ik_failure_actors = []
        if render:
            self.plotter.render()

    def _clear_inspection_goal_pose_visuals(self, render=True):
        for actor in getattr(self, '_inspection_goal_pose_actors', []) or []:
            try:
                self.plotter.remove(actor)
            except Exception:
                pass
        self._inspection_goal_pose_actors = []
        for actor in getattr(self, '_inspection_goal_robot_actors', []) or []:
            try:
                self.plotter.remove(actor)
            except Exception:
                pass
        self._inspection_goal_robot_actors = []
        if render:
            self.plotter.render()

    def _target_pose_to_link_T(self, robot_name, target_pose):
        """시각화용 target pose를 backend 기준 world transform으로 변환한다.

        Args:
            robot_name: target pose를 표시할 로봇 이름.
            target_pose: 4x4 transform, 6D pose, 또는 3D point.

        Returns:
            np.ndarray shape=(4, 4): target frame의 world transform.

        계산 과정:
            IK/path planning과 동일하게 RoboticsBackend.target_world_T를 우선 사용한다.
            3D point만 들어온 경우에는 현재 robot q의 target frame orientation을 유지한 채
            translation만 target point로 교체한다. backend가 준비되지 않은 초기 구간에서는
            기존 RobotModel FK 기반 계산으로 fallback한다.
        """
        model = self._find_robot(robot_name)
        backend = getattr(self, "_robotics_backend", None)
        if backend is not None:
            try:
                pin_model = backend.robot_model(robot_name)
                q_reference = (
                    self._current_robot_q(model, pin_model)
                    if model is not None else backend.neutral_q(robot_name)
                )
                return backend.target_world_T(
                    robot_name,
                    target_pose,
                    q_reference,
                    self._robot_target_link_name(robot_name),
                )
            except Exception as exc:
                self.__console.debug(
                    f"target pose backend conversion failed; using viewer fallback: "
                    f"robot={robot_name}, error={exc}")
        target_arr = np.asarray(target_pose, dtype=float)
        if target_arr.shape == (4, 4):
            return target_arr.copy()
        if target_arr.size >= 6:
            return self._pose_to_T(target_arr.reshape(-1)[:6])
        current_T = None
        if model is not None:
            link_name = self._robot_target_link_name(robot_name)
            current_T = model.get_link_world_T(link_name) if link_name is not None else None
        T = np.eye(4) if current_T is None else np.asarray(current_T, dtype=float).copy()
        T[:3, 3] = target_arr.reshape(-1)[:3]
        return T

    def _show_inspection_goal_pose(self, robot_name, target_pose, clear=False, render=True):
        if clear:
            self._clear_inspection_goal_pose_visuals(render=False)
        target_T = self._target_pose_to_link_T(robot_name, target_pose)
        color = "orange" if robot_name == "dda_rb10_1300e" else "violet"
        actors = self._target_pose_mesh_actors(robot_name, target_T, color=color, alpha=0.24)
        actors.extend(self._pose_frame_actors(target_T, scale=0.20, axes=(0, 1, 2), show_origin=False))
        try:
            label = "DDA_GOAL" if robot_name == "dda_rb10_1300e" else "RT_GOAL"
            text = vedo.Text3D(label, pos=target_T[:3, 3] + np.array([0.0, 0.0, 0.12]), s=0.04, c=color)
            text.pickable(False)
            actors.append(text)
        except Exception:
            pass
        self._inspection_goal_pose_actors.extend(actors)
        if actors:
            self.plotter.add(*actors)
        if render:
            self.plotter.render()
        return target_T

    def _show_inspection_goal_robot_pose(
        self,
        robot_name,
        q,
        pin_model=None,
        joint_names=None,
        clear=False,
        render=True,
    ):
        if clear:
            self._clear_inspection_goal_pose_visuals(render=False)
        model = self._find_robot(robot_name)
        if model is None:
            return []

        q = np.asarray(q, dtype=float)
        if joint_names is not None:
            names = [str(name) for name in joint_names]
        elif pin_model is None:
            names = list(getattr(model, "_joint_cfg", {}).keys())
        else:
            names = self._pin_joint_names(pin_model)
        original_q = {name: float(model._joint_cfg.get(name, 0.0)) for name in names}
        actors = []
        try:
            if pin_model is not None:
                self._apply_robot_q(model, pin_model, q)
            else:
                for i, joint_name in enumerate(names[:len(q)]):
                    model.set_joint(joint_name, float(q[i]))
                model.update_fk()

            color = "orange" if robot_name == "dda_rb10_1300e" else "deepskyblue"
            for actor in getattr(model, "actors", []) or []:
                try:
                    preview = actor.clone(deep=True)
                except TypeError:
                    preview = actor.clone()
                preview.c(color).alpha(0.22).pickable(False)
                actors.append(preview)

            target_link = self._robot_target_link_name(robot_name)
            target_T = model.get_link_world_T(target_link) if target_link is not None else None
            if target_T is not None:
                actors.extend(self._pose_frame_actors(target_T, scale=0.22, axes=(0, 1, 2), show_origin=True))
        except Exception as exc:
            self.__console.warning(f"failed to show IK goal robot preview: robot={robot_name}, error={exc}")
        finally:
            for joint_name, value in original_q.items():
                try:
                    model.set_joint(joint_name, value)
                except Exception:
                    pass
            try:
                model.update_fk()
            except Exception:
                pass

        self._inspection_goal_robot_actors.extend(actors)
        if actors:
            self.plotter.add(*actors)
        if render:
            self.plotter.render()
        return actors

    def _clear_inspection_visuals(self, clear_point=True):
        if getattr(self, '_inspection_path_actor', None) is not None:
            actors = self._inspection_path_actor
            if not isinstance(actors, (list, tuple)):
                actors = [actors]
            for actor in actors:
                try:
                    self.plotter.remove(actor)
                except Exception:
                    pass
            self._inspection_path_actor = None
        self._clear_ik_failure_visuals(render=False)
        self._clear_inspection_goal_pose_visuals(render=False)
        if clear_point:
            self._clear_ef_pose_visuals()
        if clear_point and getattr(self, '_inspection_marker', None) is not None:
            self.plotter.remove(self._inspection_marker)
            self._inspection_marker = None
            self._inspection_point = None
        self.plotter.render()

    def _clear_path_playback_marker(self):
        marker = getattr(self, '_path_playback_marker', None)
        markers = marker.values() if isinstance(marker, dict) else [marker]
        for item in markers:
            if item is not None:
                try:
                    self.plotter.remove(item)
                except Exception:
                    pass
        self._path_playback_marker = None

    def _clear_ef_pose_visuals(self, clear_poses=True):
        for actor in getattr(self, '_ef_pose_actors', []) or []:
            try:
                self.plotter.remove(actor)
            except Exception:
                pass
        self._ef_pose_actors = []
        if clear_poses:
            self._ef_target_poses = {}
            self._ef_pose_groups = []

    def _clear_robot_tcp_axes(self, render=True):
        for actor in getattr(self, '_robot_tcp_axis_actors', []) or []:
            try:
                self.plotter.remove(actor)
            except Exception:
                pass
        self._robot_tcp_axis_actors = []
        if render:
            self.plotter.render()

    def _show_robot_tcp_axes(self, render=True):
        self._clear_robot_tcp_axes(render=False)
        actors = []
        for robot_name in ("dda_rb10_1300e", "rb20_1900es"):
            model = self._find_robot(robot_name)
            if model is None:
                continue
            # Display the URDF TCP frame used by EF pose, IK, and path planning.
            # Mesh/source geometry is attached separately through target_to_mesh transforms.
            target_link = self._robot_target_link_name(robot_name)
            T = model.get_link_world_T(target_link) if target_link is not None else None
            if T is None:
                continue
            T = np.asarray(T, dtype=float)
            scale = 0.20 if robot_name == "dda_rb10_1300e" else 0.24
            actors.extend(self._pose_frame_actors(T, scale=scale, axes=(0, 1, 2), show_origin=True))
            try:
                label = "DDA_EF" if robot_name == "dda_rb10_1300e" else "RT_SOURCE"
                text = vedo.Text3D(label, pos=T[:3, 3] + np.array([0.0, 0.0, scale * 0.22]), s=scale * 0.13, c="black")
                text.pickable(False)
                actors.append(text)
            except Exception:
                pass
        self._robot_tcp_axis_actors = actors
        if actors:
            self.plotter.add(*actors)
        if render:
            self.plotter.render()

    def _robot_target_link_name(self, robot_name):
        if robot_name == "rb20_1900es":
            return "rt_link_end"
        if robot_name == "dda_rb10_1300e":
            return "dda_link_end"
        model = self._find_robot(robot_name)
        if model is not None and model._urdf is not None:
            for preferred in ("tcp", "link_end", "end"):
                for link in model._urdf.links:
                    lname = getattr(link, "name", "")
                    if preferred in lname.lower():
                        return lname
        return self._robot_tcp_link_name(robot_name)

    def _robot_tcp_link_name(self, robot_name):
        if robot_name == "rb20_1900es":
            return "rt_tcp"
        if robot_name == "dda_rb10_1300e":
            return "dda_link_end"
        model = self._find_robot(robot_name)
        if model is not None and model._urdf is not None:
            for link in model._urdf.links:
                lname = getattr(link, "name", "")
                if "tcp" in lname.lower():
                    return lname
        return None

    def _robot_mesh_link_name(self, robot_name):
        if robot_name == "rb20_1900es":
            return "rt_link_end"
        if robot_name == "dda_rb10_1300e":
            return "dda_link_end"
        return self._robot_target_link_name(robot_name)

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

    def _ef_pose_offset_T(self, frame_config):
        if not isinstance(frame_config, dict):
            return None
        offset = frame_config.get("pose_to_link_offset")
        if offset is None:
            return None
        if isinstance(offset, dict) and "matrix" in offset:
            T = np.asarray(offset["matrix"], dtype=float)
        elif isinstance(offset, dict):
            T = np.eye(4)
            T[:3, 3] = np.asarray(offset.get("xyz", [0.0, 0.0, 0.0]), dtype=float)
            T[:3, :3] = self._rpy_matrix(offset.get("rpy", [0.0, 0.0, 0.0]))
        else:
            T = np.asarray(offset, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"ef_pose pose_to_link_offset must be 4x4, got shape={T.shape}")
        return T

    def _T_to_pose(self, T):
        T = np.asarray(T, dtype=float)
        Rm = T[:3, :3]
        sy = float(np.sqrt(Rm[0, 0] * Rm[0, 0] + Rm[1, 0] * Rm[1, 0]))
        if sy > 1e-9:
            roll = np.arctan2(Rm[2, 1], Rm[2, 2])
            pitch = np.arctan2(-Rm[2, 0], sy)
            yaw = np.arctan2(Rm[1, 0], Rm[0, 0])
        else:
            roll = np.arctan2(-Rm[1, 2], Rm[1, 1])
            pitch = np.arctan2(-Rm[2, 0], sy)
            yaw = 0.0
        return np.asarray([T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw], dtype=float)

    def _urdf_joint_origin_T(self, joint):
        T = np.eye(4)
        origin = getattr(joint, "origin", None)
        if origin is not None:
            T[:3, :3] = self._rpy_matrix(origin.rpy)
            T[:3, 3] = origin.xyz
        return T

    def _urdf_relative_link_T(self, urdf, source_link_name, target_link_name, fallback_T=None):
        child_to_joint = {joint.child: joint for joint in getattr(urdf, "joints", [])}
        cache = {}

        def root_to_link(link_name):
            if link_name in cache:
                return cache[link_name]
            joint = child_to_joint.get(link_name)
            if joint is None:
                cache[link_name] = np.eye(4)
                return cache[link_name]
            T = root_to_link(joint.parent) @ self._urdf_joint_origin_T(joint)
            cache[link_name] = T
            return T

        try:
            return np.linalg.inv(root_to_link(source_link_name)) @ root_to_link(target_link_name)
        except Exception:
            return np.eye(4) if fallback_T is None else fallback_T

    def _pose_frame_actors(self, pose, scale=0.18, axes=(0, 1, 2), show_origin=True):
        pose_arr = np.asarray(pose, dtype=float)
        T = pose_arr if pose_arr.shape == (4, 4) else self._pose_to_T(pose_arr)
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

    def _target_to_mesh_link_T(self, robot_name):
        model = self._find_robot(robot_name)
        if model is None or getattr(model, "_urdf", None) is None:
            return np.eye(4)
        target_link = self._robot_target_link_name(robot_name)
        mesh_link = self._robot_mesh_link_name(robot_name)
        if target_link == mesh_link:
            return np.eye(4)
        return self._urdf_relative_link_T(model._urdf, target_link, mesh_link, fallback_T=np.eye(4))

    def _link_mesh_actors_at_T(self, robot_name, link_name, T, color, alpha=0.28):
        model = self._find_robot(robot_name)
        if model is None:
            return []
        mesh_list = getattr(model, "_link_mesh_data", {}).get(link_name, [])
        if not mesh_list:
            self.__console.warning(f"EF pose mesh unavailable: robot={robot_name}, link={link_name}")
            return []
        T = np.asarray(T, dtype=float)
        actors = []
        for local_verts, faces in mesh_list:
            verts = (T[:3, :3] @ np.asarray(local_verts, dtype=float).T).T + T[:3, 3]
            actor = vedo.Mesh([verts, np.asarray(faces, dtype=np.int32)])
            actor.c(color).alpha(alpha)
            actor.pickable(False)
            actors.append(actor)
        return actors

    def _target_pose_mesh_actors(self, robot_name, target_T, color, alpha=0.28):
        mesh_link = self._robot_mesh_link_name(robot_name)
        mesh_T = np.asarray(target_T, dtype=float) @ self._target_to_mesh_link_T(robot_name)
        return self._link_mesh_actors_at_T(robot_name, mesh_link, mesh_T, color=color, alpha=alpha)

    def _ef_pose_mesh_actors(self, pose_name, pose):
        robot_name = self._ef_pose_robot_name(pose_name)
        T = self._pose_to_T(pose)
        color = "gold" if pose_name == "DDA" else "deepskyblue"
        return self._target_pose_mesh_actors(robot_name, T, color=color, alpha=0.28)

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

    def _show_ef_ranked_candidate_target_groups(self, target_groups):
        self._clear_ef_pose_visuals(clear_poses=False)
        actors = []
        params = self._config.get("ef_pose", {}) or {}
        max_groups = int(params.get("visualize_candidate_limit", 12))
        groups_to_show = list(target_groups or [])[:max(1, max_groups)]
        for rank, group_info in enumerate(groups_to_show):
            alpha = 0.38 if rank == 0 else max(0.22, 0.30 - rank * 0.04)
            show_axes = True
            pair_origins = []
            for robot_name, target_info in group_info.get("targets", {}).items():
                pose_name = target_info.get("pose_name", robot_name)
                target_T = np.asarray(target_info["target_T"], dtype=float)
                pair_origins.append((pose_name, target_T[:3, 3].copy()))
                if pose_name == "DDA":
                    color = "gold" if rank == 0 else "orange"
                else:
                    color = "deepskyblue" if rank == 0 else "cyan"
                actors.extend(self._target_pose_mesh_actors(robot_name, target_T, color=color, alpha=alpha))
                if show_axes:
                    scale = 0.22 if str(pose_name).startswith("RT") else 0.18
                    actors.extend(self._pose_frame_actors(
                        target_T,
                        scale=scale,
                        axes=(0, 1, 2),
                        show_origin=(rank == 0),
                    ))
            if len(pair_origins) >= 2:
                try:
                    dda_origin = next(origin for name, origin in pair_origins if name == "DDA")
                    rt_origin = next(origin for name, origin in pair_origins if str(name).startswith("RT"))
                    pair_color = "black" if rank == 0 else "gray"
                    connector = vedo.Line(dda_origin, rt_origin, c=pair_color, lw=4 if rank == 0 else 2)
                    connector.alpha(0.75 if rank == 0 else 0.45)
                    connector.pickable(False)
                    actors.append(connector)
                except Exception:
                    pass
            try:
                first_target = next(iter(group_info.get("targets", {}).values()))
                label_T = np.asarray(first_target["target_T"], dtype=float)
                label = f"#{rank + 1} {group_info.get('rt_name', '')}"
                metrics = group_info.get("priority", {}) or {}
                label += (
                    f" y=({metrics.get('dda_y', 0.0):.2f},{metrics.get('rt_neg_y', 0.0):.2f})"
                    f" d/u=({metrics.get('rt_view_down45', 0.0):.2f},{metrics.get('rt_view_up45', 0.0):.2f})"
                )
                text = vedo.Text3D(
                    label,
                    pos=label_T[:3, 3] + np.array([0.0, 0.0, 0.08 + 0.018 * rank]),
                    s=0.025,
                    c="black" if rank == 0 else "gray",
                )
                text.alpha(0.85 if rank == 0 else 0.45)
                text.pickable(False)
                actors.append(text)
            except Exception:
                pass
        self._ef_pose_actors = actors
        if actors:
            self.plotter.add(*actors)
            self.plotter.render()

    def _extract_ef_poses_from_group(self, group):
        orientation_group = group.get("0") or group.get("90") or next(iter(group.values()), None)
        if not orientation_group:
            return {}
        return self._extract_ef_poses_from_orientation_group(orientation_group)

    def _extract_ef_poses_from_orientation_group(self, orientation_group):
        poses = {}
        if orientation_group.get("DDA") is not None:
            poses["DDA"] = np.asarray(orientation_group["DDA"], dtype=float)
        if orientation_group.get("RT1") is not None:
            poses["RT1"] = np.asarray(orientation_group["RT1"], dtype=float)
        if orientation_group.get("RT2") is not None:
            poses["RT2"] = np.asarray(orientation_group["RT2"], dtype=float)
        return poses

    def _show_ef_collision_pose_groups(self, pose_groups, max_groups=3):
        if not pose_groups:
            return 0
        actors = []
        shown = 0
        for group_idx, group in enumerate(list(pose_groups)[:max(1, int(max_groups))]):
            poses = self._extract_ef_poses_from_group(group)
            if not poses:
                continue
            shown += 1
            for name, pose in poses.items():
                robot_name = self._ef_pose_robot_name(name)
                T = self._pose_to_T(pose)
                color = "crimson" if name == "DDA" else "orangered"
                alpha = 0.20 if group_idx > 0 else 0.32
                scale = 0.22 if name.startswith("RT") else 0.18
                actors.extend(self._target_pose_mesh_actors(robot_name, T, color=color, alpha=alpha))
                actors.extend(self._pose_frame_actors(T, scale=scale, axes=(0, 1, 2), show_origin=False))
                try:
                    label = f"COLLISION_{group_idx}_{name}"
                    text = vedo.Text3D(label, pos=T[:3, 3] + np.array([0.0, 0.0, scale * 0.35]), s=scale * 0.12, c=color)
                    text.pickable(False)
                    actors.append(text)
                except Exception:
                    pass
        self._ef_pose_actors.extend(actors)
        if actors:
            self.plotter.add(*actors)
            self.plotter.render()
        return shown

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

    def _pose_determinator_point_cloud(self, normal_radius=None):
        pts = self._get_spool_points()
        if pts is None or len(pts) < 10:
            raise RuntimeError("loaded spool point cloud is not available")
        pts = np.asarray(pts, dtype=np.float64)
        bbox_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        radius = max(float(normal_radius) if normal_radius is not None else bbox_diag * 0.005, 1e-6)
        pcd = _o3d.geometry.PointCloud()
        pcd.points = _o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(
            search_param=_o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=30)
        )
        pcd.normalize_normals()
        return pcd

    def _extract_ef_poses(self, pose_groups):
        if not pose_groups:
            raise RuntimeError("poseDeterminator returned no valid pose group")
        group = pose_groups[0]
        poses = self._extract_ef_poses_from_group(group)
        if not poses:
            raise RuntimeError("pose group is empty")
        return poses

    def _determine_ef_pose(self, request_data):
        identity = request_data.get("_identity")
        result = {"status": "failed"}
        total_t0 = time.perf_counter()
        try:
            self._clear_ik_failure_visuals(render=False)
            target = getattr(self, '_inspection_point', None)
            if target is None:
                raise RuntimeError("inspection point is not selected")

            pose_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "plugins", "poseDeterminator")
            )
            if pose_dir not in sys.path:
                sys.path.insert(0, pose_dir)
            from EndEffectorPoseOptimizer import EndEffectorPoseOptimizer

            optimizer = EndEffectorPoseOptimizer(debug_mode=True)
            stage_t0 = time.perf_counter()
            optimizer._scan_data = self._pose_determinator_point_cloud()
            pcd_elapsed = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()
            params = self._config.get("ef_pose", {}) or {}
            frame_cfg = params.get("frames", {}) or {}
            dda_frame_cfg = frame_cfg.get("dda", {}) or {}
            rt_frame_cfg = frame_cfg.get("rt", {}) or {}
            dda_end_link = str(dda_frame_cfg.get("end_link", "dda_link_end"))
            dda_tcp_joint = str(dda_frame_cfg.get("tcp_joint", "dda_joint_tcp"))
            rt_end_link = str(rt_frame_cfg.get("end_link", "rt_link_end"))
            rt_tcp_joint = str(rt_frame_cfg.get("tcp_joint", "rt_joint_end"))
            dda_pipe_facing_axis = np.asarray(
                dda_frame_cfg.get("pipe_facing_axis", [1.0, 0.0, 0.0]),
                dtype=float,
            )
            dda_pipe_parallel_axis = dda_frame_cfg.get("pipe_parallel_axis")
            optimizer.set_dda_pipe_facing_axis(
                dda_pipe_facing_axis,
                None if dda_pipe_parallel_axis is None else np.asarray(dda_pipe_parallel_axis, dtype=float),
            )
            rt_pipe_facing_axis = np.asarray(
                rt_frame_cfg.get("pipe_facing_axis", [0.0, -1.0, 0.0]),
                dtype=float,
            )
            optimizer.set_rt_pipe_facing_axis(rt_pipe_facing_axis)
            dda_pose_to_link = self._ef_pose_offset_T(dda_frame_cfg)
            rt_pose_to_link = self._ef_pose_offset_T(rt_frame_cfg)
            backend = getattr(self, "_robotics_backend", None)
            if backend is None:
                raise RuntimeError("robotics backend is not initialized")
            dda_mesh, dda_tcp_to_link = backend.end_effector_collision_geometry(
                "dda_rb10_1300e",
                dda_end_link,
                dda_tcp_joint,
                pose_to_link_offset=dda_pose_to_link,
            )
            rt_mesh, rt_tcp_to_link = backend.end_effector_collision_geometry(
                "rb20_1900es",
                rt_end_link,
                rt_tcp_joint,
                pose_to_link_offset=rt_pose_to_link,
            )
            optimizer.set_DDA_geometry(dda_mesh, dda_tcp_to_link)
            optimizer.set_RT_geometry(rt_mesh, rt_tcp_to_link)
            scan_data = optimizer._scan_data
            optimizer.set_collision_checker(
                lambda link_model, tcp_pose, tcp_to_link_pose_T, margin=0.05, sample_count=5000:
                    backend.check_mesh_point_cloud_overlap(
                        link_model,
                        tcp_pose,
                        tcp_to_link_pose_T,
                        scan_data,
                        margin=margin,
                        sample_count=sample_count,
                    )
            )
            urdf_elapsed = time.perf_counter() - stage_t0
            self.__console.info(
                "EF pose optimizer frames: "
                "geometry_backend=robotics, "
                f"DDA(end_link={dda_end_link}, tcp_joint={dda_tcp_joint}, "
                f"pose_to_link_t={None if dda_pose_to_link is None else np.round(dda_pose_to_link[:3, 3], 5).tolist()}, "
                f"pipe_facing_axis={np.round(dda_pipe_facing_axis, 5).tolist()}), "
                f"RT(end_link={rt_end_link}, tcp_joint={rt_tcp_joint}, "
                f"pose_to_link_t={None if rt_pose_to_link is None else np.round(rt_pose_to_link[:3, 3], 5).tolist()}, "
                f"pipe_facing_axis={np.round(rt_pipe_facing_axis, 5).tolist()})")
            for robot_name in ("dda_rb10_1300e", "rb20_1900es"):
                rel_T = self._target_to_mesh_link_T(robot_name)
                self.__console.info(
                    "EF pose URDF frame map: "
                    f"robot={robot_name}, target={self._robot_target_link_name(robot_name)}, "
                    f"mesh={self._robot_mesh_link_name(robot_name)}, "
                    f"target_to_mesh_t={np.round(rel_T[:3, 3], 5).tolist()}, "
                    f"target_to_mesh_y={np.round(rel_T[:3, 1], 5).tolist()}")

            target = np.asarray(target, dtype=float)
            stage_t0 = time.perf_counter()
            optimizer.calculate_pipe_profile(
                target,
                sampling_size_for_calculating_normal=float(
                    params.get("sampling_size_for_calculating_normal", 0.01)),
                radius_offset_for_sampling_points_in_sphere=float(
                    params.get("radius_offset_for_sampling_points_in_sphere", 0.003)),
            )
            profile_elapsed = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()
            _, pose_groups = optimizer.calculate_DDA_RT_pose_for_taking_xray(
                target,
                num_candidates=int(params.get("num_candidates", 8)),
                distance_from_dda_to_surface=float(params.get("distance_from_dda_to_surface", 0.01)),
                distance_from_dda_to_rt=float(params.get("distance_from_dda_to_rt", 0.3)),
                angle_of_rt=float(params.get("angle_of_rt", 10.0)),
            )
            pose_elapsed = time.perf_counter() - stage_t0
            debug_info = getattr(optimizer, "debuging_info", {}) or {}
            self.__console.info(
                "EF pose candidate summary: "
                f"base={len(debug_info.get('dda_base_candidates', []))}, "
                f"valid_base={len(debug_info.get('valid_base_dda_poses', []))}, "
                f"base_collisions={debug_info.get('base_dda_collision_count', 'n/a')}, "
                f"complete_groups={debug_info.get('complete_pose_group_count', 'n/a')}, "
                f"partial_groups={debug_info.get('partial_pose_group_count', 'n/a')}, "
                f"rotated_dda_collisions={debug_info.get('rotated_dda_collision_count', 'n/a')}, "
                f"rejected_groups={debug_info.get('rejected_pose_group_count', 'n/a')}, "
                f"rt1_collisions={debug_info.get('rt1_collision_count', 'n/a')}, "
                f"rt2_collisions={debug_info.get('rt2_collision_count', 'n/a')}, "
                f"dda_front_extent={debug_info.get('dda_mesh_front_extent_along_facing_axis', 'n/a')}, "
                f"dda_candidate_radius={debug_info.get('dda_candidate_centerline_radius', 'n/a')}, "
                f"rt_front_extent={debug_info.get('rt_mesh_front_extent_along_facing_axis', 'n/a')}, "
                f"rt_adjusted_distance={debug_info.get('rt_adjusted_distance_from_dda_to_rt', 'n/a')}, "
                f"used_partial_fallback={bool(debug_info.get('used_partial_pose_group_fallback', False))}")
            dda_minus_y_dots = debug_info.get("dda_minus_y_dot_pipe_center", [])
            dda_config_dots = debug_info.get("dda_configured_facing_dot_pipe_center", [])
            if dda_minus_y_dots:
                self.__console.info(
                    "EF pose DDA axis check: "
                    f"configured_axis={debug_info.get('dda_pipe_facing_axis_local')}, "
                    f"parallel_axis={debug_info.get('dda_pipe_parallel_axis_local')}, "
                    f"minus_y_dot_pipe_center=[{min(dda_minus_y_dots):.4f}, {max(dda_minus_y_dots):.4f}], "
                    f"configured_dot_pipe_center=[{min(dda_config_dots):.4f}, {max(dda_config_dots):.4f}]")
            try:
                poses = self._extract_ef_poses(pose_groups)
            except Exception as extract_exc:
                collision_groups = debug_info.get("collision_pose_groups", [])
                if not collision_groups:
                    collision_groups = debug_info.get("rejected_pose_groups", [])
                shown_collision_groups = self._show_ef_collision_pose_groups(collision_groups)
                candidate_poses = debug_info.get("valid_base_dda_poses")
                if candidate_poses is None or len(candidate_poses) == 0:
                    candidate_poses = debug_info.get("dda_base_candidates", [])
                self._show_ef_pose_candidates(candidate_poses)
                self.__console.warning(
                    "EF pose extraction failed; showing collision groups: "
                    f"shown={shown_collision_groups}, available={len(collision_groups)}, error={extract_exc}")
                result = {
                    "status": "failed",
                    "message": str(extract_exc),
                    "candidate_count": len(candidate_poses),
                    "collision_group_count": len(collision_groups),
                    "shown_collision_group_count": shown_collision_groups,
                    "elapsed": time.perf_counter() - total_t0,
                    "timing": {
                        "point_cloud": pcd_elapsed,
                        "urdf": urdf_elapsed,
                        "pipe_profile": profile_elapsed,
                        "pose_candidates": pose_elapsed,
                    },
                }
                return
            self._ef_pose_groups = list(pose_groups or [])
            ranked_target_groups = self._ef_pose_selected_candidate_target_groups()
            if ranked_target_groups:
                poses = {
                    target_info["pose_name"]: self._T_to_pose(target_info["target_T"])
                    for target_info in ranked_target_groups[0]["targets"].values()
                }
            self._ef_target_poses = poses
            if ranked_target_groups:
                self._show_ef_ranked_candidate_target_groups(ranked_target_groups)
            else:
                self._show_ef_target_poses(poses)
            candidate_poses = debug_info.get("valid_base_dda_poses")
            if candidate_poses is None or len(candidate_poses) == 0:
                candidate_poses = debug_info.get("dda_base_candidates", [])
            self._show_ef_pose_candidates(candidate_poses)
            result = {
                "status": "success",
                "poses": {name: pose.tolist() for name, pose in poses.items()},
                "pose_group_count": len(getattr(self, "_ef_pose_groups", []) or []),
                "ranked_candidate_count": len(ranked_target_groups),
                "candidate_count": len(candidate_poses),
                "elapsed": time.perf_counter() - total_t0,
                "timing": {
                    "point_cloud": pcd_elapsed,
                    "urdf": urdf_elapsed,
                    "pipe_profile": profile_elapsed,
                    "pose_candidates": pose_elapsed,
                },
            }
            self.__console.info(
                "EF pose determined: "
                f"poses={list(poses.keys())}, candidates={len(candidate_poses)}, "
                f"elapsed={result['elapsed']:.3f}s "
                f"(pcd={pcd_elapsed:.3f}s, urdf={urdf_elapsed:.3f}s, "
                f"profile={profile_elapsed:.3f}s, pose={pose_elapsed:.3f}s)")
        except Exception as exc:
            if str(exc) != "__ef_pose_collision_groups_rendered__":
                elapsed = time.perf_counter() - total_t0
                result = {"status": "failed", "message": str(exc), "elapsed": elapsed}
                self.__console.error(f"EF pose determination failed after {elapsed:.3f}s: {exc}")
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
        raise RuntimeError(
            f"planner '{planner_name}' is not supported for robot q-space planning. "
            f"supported={sorted(q_space_planners)}")

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

    def _configure_inspection_planner(
        self,
        planner,
        obstacle_mesh,
        start,
        goal,
        step_size,
        max_iter,
        robot_name=None,
        pin_cache=None,
        timings=None,
    ):
        setup_t0 = time.perf_counter()
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
        if robot_name is not None:
            backend = getattr(self, "_robotics_backend", None)
            if backend is None:
                raise RuntimeError("robotics backend is not initialized")
            planner.robotics_backend = backend
            planner.robotics_robot_name = robot_name
        if timings is not None:
            timings["planner_bounds_config"] = time.perf_counter() - setup_t0
        collision_obstacle_mesh = obstacle_mesh
        if robot_name is not None:
            model = self._find_robot(robot_name)
            urdf_path = getattr(model, "urdf_path", None) if model is not None else None
            if urdf_path:
                try:
                    urdf_t0 = time.perf_counter()
                    planner.pin_model = backend.robot_model(robot_name)
                    planner.pin_data = planner.pin_model.createData()
                    if timings is not None:
                        timings["planner_robotics_model"] = time.perf_counter() - urdf_t0
                    self._log_pinocchio_robot_model(robot_name, urdf_path, planner.pin_model)
                except Exception as exc:
                    raise RuntimeError(f"inspection path robotics model setup failed: {exc}") from exc
            if getattr(planner, "pin_model", None) is not None and model is not None:
                base_T = np.asarray(getattr(model, "_base_T", np.eye(4)), dtype=float)
                if base_T.shape == (4, 4) and not np.allclose(base_T, np.eye(4)):
                    transform_t0 = time.perf_counter()
                    collision_obstacle_mesh = copy.deepcopy(obstacle_mesh)
                    collision_obstacle_mesh.transform(np.linalg.inv(base_T))
                    if timings is not None:
                        timings["planner_obstacle_base_transform"] = time.perf_counter() - transform_t0
                    self.__console.debug(
                        "inspection path: transformed obstacle mesh into robot base frame for collision | "
                        f"robot={robot_name}, base_t={np.round(base_T[:3, 3], 5).tolist()}")
        obstacle_t0 = time.perf_counter()
        planner.add_collision_object(collision_obstacle_mesh)
        if timings is not None:
            timings["planner_obstacle_bvh"] = time.perf_counter() - obstacle_t0
        self._log_pinocchio_collision_targets(robot_name, planner)
        return bounds

    def _log_pinocchio_collision_targets(self, robot_name, planner):
        if planner is None or not hasattr(planner, "pinocchio_collision_geometry_summary"):
            return
        logged = getattr(self, "_logged_pinocchio_collision_targets", set())
        geom_model = getattr(planner, "pin_geom_model", None)
        static_ids = tuple(getattr(planner, "_pin_static_object_ids", []) or [])
        key = (robot_name, len(getattr(geom_model, "geometryObjects", []) or []), len(getattr(geom_model, "collisionPairs", []) or []), static_ids)
        if key in logged:
            return
        logged.add(key)
        self._logged_pinocchio_collision_targets = logged
        geometries = planner.pinocchio_collision_geometry_summary()
        robot_geometries = [item for item in geometries if item.get("kind") == "robot"]
        static_geometries = [item for item in geometries if item.get("kind") == "static"]
        robot_self_pairs = planner.pinocchio_collision_pair_summary(include_robot_self=True, include_static=False)
        static_pairs = planner.pinocchio_collision_pair_summary(include_robot_self=False, include_static=True)
        positioner_checked = self._planner_has_positioner_collision(planner)
        self.__console.info(
            "Pinocchio collision targets: "
            f"robot={robot_name}, robot_geoms={len(robot_geometries)}, "
            f"static_geoms={len(static_geometries)}, "
            f"robot_self_pairs={len(robot_self_pairs)}, robot_static_pairs={len(static_pairs)}, "
            f"positioner_collision_checked={positioner_checked}")
        if not positioner_checked:
            self.__console.debug(
                "Pinocchio collision targets: positioner URDF is not part of this planner collision model; "
                "positioner collision is skipped for this path check.")

    def _planner_has_positioner_collision(self, planner):
        try:
            geometries = planner.pinocchio_collision_geometry_summary()
            pairs = planner.pinocchio_collision_pair_summary(include_robot_self=True, include_static=True)
        except Exception:
            return False

        def is_positioner_name(name):
            name = str(name).lower()
            return "positioner" in name or "f_column" in name or "m_column" in name

        if not any(
            is_positioner_name(item.get("name", "")) or is_positioner_name(item.get("parent_joint_name", ""))
            for item in geometries
        ):
            return False
        return any(
            is_positioner_name(pair.get("first", "")) or is_positioner_name(pair.get("second", ""))
            for pair in pairs
        )

    def _log_pinocchio_robot_model(self, robot_name, urdf_path, pin_model):
        logged = getattr(self, "_logged_pinocchio_models", set())
        try:
            urdf_mtime_ns = os.stat(urdf_path).st_mtime_ns
        except OSError:
            urdf_mtime_ns = None
        key = (robot_name, urdf_path, urdf_mtime_ns)
        if key in logged or pin_model is None:
            return
        logged.add(key)
        self._logged_pinocchio_models = logged
        joint_names = self._pin_joint_names(pin_model)
        track_joints = [name for name in joint_names if "linear_track" in name or "carriage" in name]
        lo = np.asarray(pin_model.lowerPositionLimit, dtype=float)
        hi = np.asarray(pin_model.upperPositionLimit, dtype=float)
        joint_limits = {
            name: [float(lo[i]), float(hi[i])]
            for i, name in enumerate(joint_names[:len(lo)])
        }
        track_joint_placements = {}
        for name in track_joints:
            try:
                joint_id = int(pin_model.getJointId(name))
                placement = pin_model.jointPlacements[joint_id]
                track_joint_placements[name] = {
                    "parent_to_joint_translation": np.asarray(placement.translation, dtype=float).tolist(),
                    "parent_to_joint_rotation": np.asarray(placement.rotation, dtype=float).round(6).tolist(),
                }
            except Exception:
                continue
        self.__console.info(
            f"Pinocchio model for {robot_name}: urdf={urdf_path}, "
            f"mtime_ns={urdf_mtime_ns}, "
            f"nq={pin_model.nq}, joints={joint_names}, track_joints={track_joints}, "
            f"limits={joint_limits}, track_joint_placements={track_joint_placements}")

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

        backend = getattr(self, "_robotics_backend", None)
        if backend is None:
            self.__console.warning(f"{reason}: robotics backend is not initialized")
            return []
        results = []
        for model in getattr(self, '_robot_models', []):
            if not getattr(model, "name", None):
                continue
            try:
                backend.configure_collision(
                    model.name,
                    static_meshes=[obstacle_mesh],
                    sample_resolution=float(self._config.get("planner_collision_sample_resolution", 0.05)),
                )
                q = self._current_robot_q(model, backend.robot_model(model.name))
                collision = backend.check_collision(model.name, q, return_pairs=True)
                result = {
                    "robot": getattr(model, "name", ""),
                    "object_geom_id": None,
                    "collision": bool(collision.collision),
                    "pairs": [list(pair) for pair in collision.pairs],
                }
                results.append(result)
                if collision.collision:
                    pair_text = ", ".join(f"{a} <-> {b}" for a, b in collision.pairs)
                    self.__console.warning(
                        f"{reason}: current robot collision detected for {model.name}: {pair_text}")
                else:
                    self.__console.info(
                        f"{reason}: spool mesh added to robotics backend for {model.name}, "
                        "no collision at current q")
            except Exception as exc:
                self.__console.warning(
                    f"{reason}: robotics spool collision probe failed for "
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

    def _robot_joint_state_payload(self, robot_names=None):
        names = set(robot_names) if robot_names is not None else None
        robots = {}
        for model in getattr(self, '_robot_models', []):
            robot_name = getattr(model, 'name', None)
            if not robot_name or robot_name == "positioner":
                continue
            if names is not None and robot_name not in names:
                continue
            urdf = getattr(model, '_urdf', None)
            if urdf is None:
                continue
            joints = {}
            for joint in getattr(urdf, 'joints', []):
                if getattr(joint, 'type', None) == "fixed":
                    continue
                joints[joint.name] = float(model._joint_cfg.get(joint.name, 0.0))
            robots[robot_name] = joints
        return {"robots": robots}

    def _send_robot_joint_state_update(self, robot_names=None, identity=None, throttle_s=0.0):
        identity = identity if identity is not None else getattr(self, '_robot_joint_state_identity', None)
        if not (hasattr(self, 'zapi') and self.zapi and identity):
            return
        now = time.monotonic()
        if throttle_s > 0.0 and now - float(getattr(self, '_last_robot_joint_state_sent', 0.0)) < throttle_s:
            return
        payload = self._robot_joint_state_payload(robot_names)
        if not payload.get("robots"):
            return
        self.zapi.update_robot_joint_state(payload, identity=identity)
        self._last_robot_joint_state_sent = now

    def _pin_target_frame_id(self, pin_model, robot_name):
        backend = getattr(self, "_robotics_backend", None)
        if backend is None:
            raise RuntimeError("robotics backend is not initialized")
        try:
            handle = backend.robot_handle(robot_name)
            if handle.model is pin_model:
                return backend.frame_id(robot_name, self._robot_target_link_name(robot_name))
        except Exception:
            pass
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
        backend = getattr(self, "_robotics_backend", None)
        if backend is None:
            raise RuntimeError("robotics backend is not initialized")
        try:
            handle = backend.robot_handle(robot_name)
            if handle.model is pin_model:
                return backend.frame_world_T(
                    robot_name,
                    q,
                    self._robot_target_link_name(robot_name),
                )
        except Exception:
            pass
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

    def _inspection_target_world_T(self, model, pin_model, robot_name, target_world, q_reference):
        """검사 목표 입력을 backend 기준 world transform으로 변환한다.

        Args:
            model, pin_model: 이전 호출부 호환용 인자. 계산은 backend가 수행한다.
            robot_name: backend에 등록된 로봇 이름.
            target_world: 4x4 transform, 6D pose, 또는 3D target point.
            q_reference: 3D target point 입력일 때 현재 orientation을 가져올 기준 q.

        Returns:
            np.ndarray shape=(4, 4): 목표 TCP world transform.

        계산 과정:
            RoboticsBackend.target_world_T에 위임한다. viewer는 target link 이름만 제공한다.
        """
        backend = getattr(self, "_robotics_backend", None)
        if backend is None:
            raise RuntimeError("robotics backend is not initialized")
        return backend.target_world_T(
            robot_name,
            target_world,
            q_reference,
            self._robot_target_link_name(robot_name),
        )

    def _inspection_ik_config(self):
        """inspection IK 설정을 viewer config와 path_planning 기본값에서 병합한다.

        Args:
            없음. self._config를 참조한다.

        Returns:
            dict: max_iter, tol, damping, dt, position_only_tol 등이 채워진 IK 설정.

        계산 과정:
            inspection_ik 섹션 값을 우선 사용하고, 없으면 path_planning의 ik_* 값을 기본값으로 채운다.
        """
        cfg = dict((self._config.get("inspection_ik", {}) or {}))
        path_cfg = self._config.get("path_planning", {}) or {}
        cfg.setdefault("max_iter", path_cfg.get("ik_max_iter", 1000))
        cfg.setdefault("tol", path_cfg.get("ik_tol", 1e-4))
        cfg.setdefault("damping", path_cfg.get("ik_damping", 1e-3))
        cfg.setdefault("dt", path_cfg.get("ik_dt", 0.35))
        cfg.setdefault("position_only_tol", path_cfg.get("ik_position_only_tol", 0.01))
        return cfg

    def _save_inspection_ik_experiment(
        self,
        robot_name,
        robot_model,
        pin_model,
        target_pose,
        goal_q,
        ik_result=None,
    ):
        trace = (getattr(self, "_last_inspection_ik_trace", {}) or {}).get(robot_name)
        if not trace:
            return None
        try:
            out_dir = Path(
                getattr(
                    self,
                    "_inspection_ik_experiment_dir",
                    Path(self._config.get("experiment_dir", "experiment")) / "inspection_ik",
                )
            )
            stamp = time.strftime("%Y%m%d_%H%M%S")
            safe_robot = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(robot_name))
            normalize_label = "normalized" if bool((ik_result or {}).get("normalize", False)) else "raw"
            if bool((ik_result or {}).get("fallback", False)):
                status_label = "fallback"
            elif bool((ik_result or {}).get("success", False)):
                status_label = "success"
            else:
                status_label = "failed"
            if bool((ik_result or {}).get("collision", False)):
                status_label = f"{status_label}_collision"
            status_dir = "collision" if "collision" in status_label else status_label
            out_dir = out_dir / status_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"inspection_ik_{stamp}_{safe_robot}_{normalize_label}_{status_label}"
            csv_path = out_dir / f"{stem}.csv"
            json_path = out_dir / f"{stem}.json"
            joint_names = self._pin_joint_names(pin_model)

            with csv_path.open("w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "iteration",
                    "err_norm",
                    "position_error",
                    "orientation_error",
                    "tcp_x",
                    "tcp_y",
                    "tcp_z",
                ] + [f"q{i}_{name}" for i, name in enumerate(joint_names)]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in trace:
                    q = np.asarray(row.get("q", []), dtype=float).reshape(-1)
                    tcp = np.asarray(row.get("tcp_world", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
                    data = {
                        "iteration": int(row.get("iteration", 0)),
                        "err_norm": float(row.get("err_norm", np.nan)),
                        "position_error": float(row.get("position_error", np.nan)),
                        "orientation_error": float(row.get("orientation_error", np.nan)),
                        "tcp_x": float(tcp[0]) if tcp.size > 0 else np.nan,
                        "tcp_y": float(tcp[1]) if tcp.size > 1 else np.nan,
                        "tcp_z": float(tcp[2]) if tcp.size > 2 else np.nan,
                    }
                    for i, name in enumerate(joint_names):
                        data[f"q{i}_{name}"] = float(q[i]) if i < q.size else np.nan
                    writer.writerow(data)

            target_T = self._inspection_target_world_T(
                robot_model,
                pin_model,
                robot_name,
                target_pose,
                np.asarray(goal_q, dtype=float),
            )
            meta = {
                "robot_name": robot_name,
                "urdf_path": os.path.abspath(getattr(robot_model, "urdf_path", "")),
                "base_pose": list(getattr(robot_model, "base_pose", [0, 0, 0, 0, 0, 0])),
                "joint_names": joint_names,
                "target_link_name": self._robot_target_link_name(robot_name),
                "csv_path": str(csv_path),
                "target_T": None if target_T is None else np.asarray(target_T, dtype=float).tolist(),
                "goal_q": np.asarray(goal_q, dtype=float).reshape(-1).tolist(),
                "ik_result": ik_result or {},
            }
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            self.__console.info(
                f"inspection IK experiment saved: robot={robot_name}, csv={csv_path}, meta={json_path}")
            return {"csv": str(csv_path), "meta": str(json_path)}
        except Exception as exc:
            self.__console.warning(f"failed to save inspection IK experiment: robot={robot_name}, error={exc}")
            return None

    def _q_path_to_target_poses(self, model, pin_model, robot_name, q_path, sample_resolution=0.03):
        """raw q path를 target frame의 world position waypoint로 변환한다.

        Args:
            model, pin_model: FK 계산에 필요한 robot model.
            robot_name: 변환 대상 로봇 이름.
            q_path: raw q waypoint list.
            sample_resolution: 인접 q 사이 보간 간격.

        Returns:
            list[np.ndarray]: 각 항목은 [x, y, z, 0, 0, 0] 형태의 표시용 pose.

        계산 과정:
            q-space edge를 sample_resolution 기준으로 보간하고, 각 q에서 target frame FK를 계산해
            world position만 추출한다. orientation은 현재 path line 표시에는 사용하지 않는다.
        """
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
        """q path를 TCP 표시 path로 변환한다.

        Args:
            model, pin_model, robot_name, q_path, sample_resolution: _q_path_to_target_poses와 동일.

        Returns:
            list[np.ndarray]: viewer path actor 생성에 쓰는 TCP waypoint.

        계산 과정:
            현재 기준 TCP와 target frame이 동일하므로 _q_path_to_target_poses에 그대로 위임한다.
        """
        return self._q_path_to_target_poses(model, pin_model, robot_name, q_path, sample_resolution)

    def _show_inspection_path(self, path, robot_name=None, clear=True):
        if clear:
            self._clear_inspection_visuals(clear_point=False)
        pts = np.asarray([np.asarray(p, dtype=float)[:3] for p in path], dtype=float)
        if len(pts) < 2:
            return
        color = "gold" if robot_name == "dda_rb10_1300e" else "limegreen"
        actor = vedo.Line(pts).c(color).lw(5)
        actor.pickable(False)
        existing = getattr(self, '_inspection_path_actor', None)
        if existing is None or clear:
            self._inspection_path_actor = [actor]
        elif isinstance(existing, list):
            existing.append(actor)
        else:
            self._inspection_path_actor = [existing, actor]
        self.plotter.add(actor)
        self.plotter.render()

    def _show_ik_failure_reached_pose(self, robot_name, final_T, target_T=None):
        if final_T is None:
            return
        final_T = np.asarray(final_T, dtype=float)
        pos = final_T[:3, 3]
        axis_len = 0.22
        actors = []
        if target_T is not None:
            target_T_arr = np.asarray(target_T, dtype=float)
            if not getattr(self, '_inspection_goal_pose_actors', []):
                goal_color = "orange" if robot_name == "dda_rb10_1300e" else "violet"
                actors.extend(self._target_pose_mesh_actors(
                    robot_name,
                    target_T_arr,
                    color=goal_color,
                    alpha=0.34,
                ))
                actors.extend(self._pose_frame_actors(
                    target_T_arr,
                    scale=axis_len * 0.9,
                    axes=(0, 1, 2),
                    show_origin=False,
                ))
        marker = vedo.Sphere(pos=pos, r=0.055, c="red")
        marker.pickable(False)
        actors.append(marker)
        for axis_idx, color in ((0, "red"), (1, "green"), (2, "blue")):
            arrow = vedo.Arrow(pos, pos + final_T[:3, axis_idx] * axis_len, s=0.0008, c=color)
            arrow.alpha(0.65)
            arrow.pickable(False)
            actors.append(arrow)
        if target_T is not None:
            target_pos = np.asarray(target_T, dtype=float)[:3, 3]
            line = vedo.Line(pos, target_pos, c="red", lw=4)
            line.pickable(False)
            actors.append(line)
            try:
                goal_text = vedo.Text3D(
                    f"{robot_name} IK goal",
                    pos=target_pos + np.array([0.0, 0.0, 0.12]),
                    s=0.04,
                    c="orange" if robot_name == "dda_rb10_1300e" else "violet",
                )
                goal_text.pickable(False)
                actors.append(goal_text)
            except Exception:
                pass
        try:
            text = vedo.Text3D(f"{robot_name} IK reached", pos=pos + np.array([0.0, 0.0, 0.12]), s=0.04, c="red")
            text.pickable(False)
            actors.append(text)
        except Exception:
            pass
        self._ik_failure_actors.extend(actors)
        self.plotter.add(*actors)
        self.plotter.render()

    def _show_inspection_ik_pose_result(self, robot_name, reached_T, target_T=None, success=True, fallback=False):
        if reached_T is None:
            return
        reached_T = np.asarray(reached_T, dtype=float)
        pos = reached_T[:3, 3]
        axis_len = 0.22
        marker_color = "cyan" if success and not fallback else "red"
        target_color = "orange" if robot_name == "dda_rb10_1300e" else "violet"
        actors = []

        if target_T is not None:
            target_T_arr = np.asarray(target_T, dtype=float)
            actors.extend(self._target_pose_mesh_actors(
                robot_name,
                target_T_arr,
                color=target_color,
                alpha=0.28,
            ))
            actors.extend(self._pose_frame_actors(
                target_T_arr,
                scale=axis_len * 0.9,
                axes=(0, 1, 2),
                show_origin=False,
            ))
            line = vedo.Line(pos, target_T_arr[:3, 3], c=marker_color, lw=3)
            line.pickable(False)
            actors.append(line)

        marker = vedo.Sphere(pos=pos, r=0.045, c=marker_color)
        marker.pickable(False)
        actors.append(marker)
        for axis_idx, color in ((0, "red"), (1, "green"), (2, "blue")):
            arrow = vedo.Arrow(pos, pos + reached_T[:3, axis_idx] * axis_len, s=0.0007, c=color)
            arrow.alpha(0.65)
            arrow.pickable(False)
            actors.append(arrow)
        try:
            text = vedo.Text3D(
                f"{robot_name} IK {'OK' if success and not fallback else 'fallback'}",
                pos=pos + np.array([0.0, 0.0, 0.12]),
                s=0.04,
                c=marker_color,
            )
            text.pickable(False)
            actors.append(text)
        except Exception:
            pass

        self._ik_failure_actors.extend(actors)
        self.plotter.add(*actors)
        self.plotter.render()

    def _show_ik_failure_markers(self, robot_names=None, failure_infos=None):
        failures = failure_infos if failure_infos is not None else (getattr(self, "_last_ik_failure", {}) or {})
        if robot_names is not None:
            failures = {name: failures.get(name) for name in robot_names}
        for robot_name, info in failures.items():
            if not info:
                continue
            final_T = info.get("final_T")
            target_T = info.get("target_T")
            try:
                self._show_ik_failure_reached_pose(robot_name, final_T, target_T)
                final_position = info.get("final_position")
                target_position = info.get("target_position")
                self.__console.info(
                    f"IK failure marker shown: robot={robot_name}, "
                    f"final={final_position}, target={target_position}")
            except Exception as exc:
                self.__console.warning(f"failed to show IK failure marker for {robot_name}: {exc}")

    def _check_inspection_ik_for_robot(self, request_data, robot_name, target_pose, obstacle_mesh=None):
        """검사 목표 pose의 IK 가능 여부를 확인한다.

        Args:
            request_data: UI/ZAPI에서 전달된 IK 옵션과 start_q override.
            robot_name: 검사할 로봇 이름.
            target_pose: 목표 TCP pose. 4x4 transform, 6D pose, 3D point를 허용한다.
            obstacle_mesh: 호환성용 인자. IK check 계산에는 직접 사용하지 않는다.

        Returns:
            dict: IK 성공/실패, start_q, goal_q, collision 여부, reached/target pose, timing.

        계산 과정:
            1. viewer에서 현재 TCP pose와 robot model만 조회한다.
            2. backend model과 start_q를 준비한다.
            3. InspectionPlanningBase.check_inspection_ik_for_robot에 계산을 위임한다.
            4. 반환된 trace/stat/failure를 viewer 상태에 저장하고 실험 로그 파일을 남긴다.
        """
        total_t0 = time.perf_counter()
        timings = {}

        stage_t0 = time.perf_counter()
        start = self._get_robot_tcp_pose(robot_name)
        if start is None:
            raise RuntimeError(f"robot TCP not found: {robot_name}")
        goal = np.zeros(6, dtype=float)
        target_arr = np.asarray(target_pose, dtype=float)
        if target_arr.shape == (4, 4):
            goal[:3] = target_arr[:3, 3]
        else:
            flat_target = target_arr.reshape(-1)
            goal[:min(6, flat_target.size)] = flat_target[:min(6, flat_target.size)]
        robot_model = self._find_robot(robot_name)
        if robot_model is None:
            raise RuntimeError(f"robot model not found: {robot_name}")
        timings["target_setup"] = time.perf_counter() - stage_t0

        stage_t0 = time.perf_counter()
        planner_name = str(request_data.get("planner", "ik_check"))
        backend = getattr(self, "_robotics_backend", None)
        if backend is None:
            raise RuntimeError("robotics backend is not initialized")
        pin_model = backend.robot_model(robot_name)
        timings["robotics_model_lookup"] = time.perf_counter() - stage_t0

        stage_t0 = time.perf_counter()
        start_q = np.zeros(pin_model.nq, dtype=float)
        start_overrides = request_data.get("_start_q_override_by_robot") or {}
        if robot_name in start_overrides:
            try:
                start_q = np.asarray(start_overrides[robot_name], dtype=float)
                if start_q.shape[0] != pin_model.nq:
                    raise ValueError(f"expected nq={pin_model.nq}, got {start_q.shape[0]}")
            except Exception as exc:
                self.__console.warning(
                    f"inspection IK check start_q override ignored: robot={robot_name}, error={exc}")
                start_q = np.zeros(pin_model.nq, dtype=float)
        timings["start_q_setup"] = time.perf_counter() - stage_t0
        self.__console.debug(
            f"inspection IK check input: robot={robot_name}, "
            f"start_q={np.round(start_q, 5).tolist()}, "
            f"target_world_pose={np.round(goal, 5).tolist()}")
        service = getattr(self, "_inspection_planning_base", None)
        if service is None:
            raise RuntimeError("inspection planning base is not initialized")
        result = service.check_inspection_ik_for_robot(
            InspectionIKRequest(
                robot_name=robot_name,
                target_pose=target_pose,
                start_tcp_pose=start,
                start_q=start_q,
                frame_name=self._robot_target_link_name(robot_name),
                joint_names=self._pin_joint_names(pin_model),
                planner_name=planner_name,
                ik_config=self._inspection_ik_config(),
                ik_solver=request_data.get("ik_solver"),
                ik_normalize=request_data.get("ik_normalize"),
            )
        )
        timings.update(result.get("timing", {}))
        timings["target_setup"] = timings.get("target_setup", 0.0)
        timings["robotics_model_lookup"] = timings.get("robotics_model_lookup", 0.0)
        timings["start_q_setup"] = timings.get("start_q_setup", 0.0)
        result["timing"] = timings
        goal_q = np.asarray(result["goal_q"], dtype=float)
        ik_result = result.get("ik_result", {})
        ik_failure = result.get("ik_failure")
        if ik_failure:
            self._last_ik_failure = getattr(self, "_last_ik_failure", {})
            self._last_ik_failure[robot_name] = ik_failure
        self._last_inspection_ik_stats = getattr(self, "_last_inspection_ik_stats", {})
        self._last_inspection_ik_stats[robot_name] = {
            "iterations": ik_result.get("iterations"),
            "elapsed": ik_result.get("elapsed"),
            "max_iter": ik_result.get("max_iter"),
            "solver": ik_result.get("solver"),
            "normalize": ik_result.get("normalize"),
            "converged": ik_result.get("success"),
            "position_only": False,
        }
        self._last_inspection_ik_trace = getattr(self, "_last_inspection_ik_trace", {})
        self._last_inspection_ik_trace[robot_name] = result.get("ik_trace", [])
        level = self.__console.info if ik_result.get("success") and not ik_result.get("collision") else self.__console.warning
        level(
            "inspection IK result: "
            f"robot={robot_name}, success={bool(ik_result.get('success'))}, "
            f"fallback={bool(result.get('ik_fallback'))}, "
            f"position_error={float(ik_result.get('position_error', float('inf'))):.5f}m, "
            f"orientation_error={float(ik_result.get('orientation_error', float('inf'))):.5f}rad, "
            f"collision={bool(ik_result.get('collision'))}, "
            f"collision_pairs={int(ik_result.get('collision_pair_count', 0))}, "
            f"iterations={ik_result.get('iterations', '-')}")
        self.__console.info(
            "inspection IK q: "
            f"robot={robot_name}, "
            f"start_q={np.round(start_q, 6).tolist()}, "
            f"target_q={np.round(goal_q, 6).tolist()}")
        ik_experiment = self._save_inspection_ik_experiment(
            robot_name,
            robot_model,
            pin_model,
            target_pose,
            goal_q,
            ik_result=ik_result,
        )
        result["ik_experiment"] = ik_experiment
        timings["total"] = time.perf_counter() - total_t0
        result["timing"] = timings
        return result

    def _plan_inspection_path_for_robot(self, request_data, robot_name, target_pose, obstacle_mesh=None):
        """검사 목표 pose까지 한 로봇의 q-space path를 계산한다.

        Args:
            request_data: planner 이름, step size, max_iter, IK 옵션, timeout을 포함한 요청 dict.
            robot_name: 경로를 계산할 로봇 이름.
            target_pose: 목표 TCP pose. 4x4 transform, 6D pose, 3D point를 허용한다.
            obstacle_mesh: collision scene에 넣을 배관 mesh. None이면 현재 로드된 배관 mesh를 사용한다.

        Returns:
            dict: q_path, TCP display path, IK 결과, collision verification, timing을 포함한 계획 결과.

        계산 과정:
            1. viewer 상태에서 obstacle mesh, 현재 TCP pose, robot model을 조회한다.
            2. 선택된 q-space planner를 생성하고 robotics backend collision scene을 구성한다.
            3. InspectionPlanningBase.plan_q_path_for_robot에 IK, q planning, path 검증을 위임한다.
            4. 반환된 q_path를 viewer 표시용 TCP waypoint로 변환하고 실험 로그를 저장한다.
        """
        total_t0 = time.perf_counter()
        timings = {}
        stage_t0 = time.perf_counter()
        # 1) ?꾩옱 諛곌????μ븷臾?mesh濡?以鍮꾪븳??
        if obstacle_mesh is None:
            obstacle_mesh = self._current_spool_collision_mesh()
        if obstacle_mesh is None:
            raise RuntimeError("loaded pipe is not available")
        timings["obstacle_mesh"] = time.perf_counter() - stage_t0

        # 2) planner ?낅젰???쒖옉 TCP pose? 紐⑺몴 pose瑜?world 醫뚰몴怨?湲곗??쇰줈 ?뺣━?쒕떎.
        stage_t0 = time.perf_counter()
        start = self._get_robot_tcp_pose(robot_name)
        if start is None:
            raise RuntimeError(f"robot TCP not found: {robot_name}")
        goal = np.zeros(6, dtype=float)
        target_arr = np.asarray(target_pose, dtype=float)
        if target_arr.shape == (4, 4):
            goal[:3] = target_arr[:3, 3]
        else:
            flat_target = target_arr.reshape(-1)
            goal[:min(6, flat_target.size)] = flat_target[:min(6, flat_target.size)]
        robot_model = self._find_robot(robot_name)
        if robot_model is None:
            raise RuntimeError(f"robot model not found: {robot_name}")
        timings["target_setup"] = time.perf_counter() - stage_t0

        # 3) ?붿껌??q-space planner瑜?留뚮뱾怨? Pinocchio 異⑸룎 紐⑤뜽/URDF源뚯? ?ㅼ젙?쒕떎.
        stage_t0 = time.perf_counter()
        planner_name = self._inspection_q_space_planner_name(request_data.get("planner", "rrt_connect"))
        planner = self._load_path_planner(planner_name)
        self._configure_inspection_planner(
            planner,
            obstacle_mesh,
            start,
            goal,
            float(request_data.get("step_size", 0.08)),
            int(request_data.get("max_iter", 3000)),
            robot_name=robot_name,
            pin_cache=(getattr(self, "_pinocchio_robot_collision_cache", {}) or {}).get(robot_name),
            timings=timings)
        if getattr(planner, "pin_model", None) is None:
            raise RuntimeError("Pinocchio collision model is not configured")
        timings["planner_setup"] = time.perf_counter() - stage_t0
        self.__console.debug(
            "inspection path planner setup timing: "
            f"robot={robot_name}, total={timings.get('planner_setup', 0.0):.3f}s, "
            f"bounds={timings.get('planner_bounds_config', 0.0):.3f}s, "
            f"urdf_collision_model={timings.get('planner_pinocchio_urdf_collision_model', 0.0):.3f}s, "
            f"obstacle_transform={timings.get('planner_obstacle_base_transform', 0.0):.3f}s, "
            f"obstacle_bvh={timings.get('planner_obstacle_bvh', 0.0):.3f}s")

        # 4) IK solve, q-space planning, collision verification은 robotics base에 위임한다.
        stage_t0 = time.perf_counter()
        start_q = self._current_robot_q(robot_model, planner.pin_model)
        start_overrides = request_data.get("_start_q_override_by_robot") or {}
        if robot_name in start_overrides:
            try:
                start_q = np.asarray(start_overrides[robot_name], dtype=float)
                if start_q.shape[0] != planner.pin_model.nq:
                    raise ValueError(f"expected nq={planner.pin_model.nq}, got {start_q.shape[0]}")
            except Exception as exc:
                self.__console.warning(
                    f"inspection path start_q override ignored: robot={robot_name}, error={exc}")
                start_q = self._current_robot_q(robot_model, planner.pin_model)
        self.__console.debug(
            f"inspection path IK input: robot={robot_name}, "
            f"start_q={np.round(start_q, 5).tolist()}, "
            f"target_world_pose={np.round(goal, 5).tolist()}")
        planning_timeout = float(request_data.get(
            "planning_timeout",
            (self._config.get("path_planning", {}) or {}).get("planning_timeout", 0.0),
        ))
        service = getattr(self, "_inspection_planning_base", None)
        if service is None:
            raise RuntimeError("inspection planning base is not initialized")
        plan = service.plan_q_path_for_robot(
            planner=planner,
            ik_request=InspectionIKRequest(
                robot_name=robot_name,
                target_pose=target_pose,
                start_tcp_pose=start,
                start_q=start_q,
                frame_name=self._robot_target_link_name(robot_name),
                joint_names=self._pin_joint_names(planner.pin_model),
                planner_name=planner_name,
                ik_config=self._inspection_ik_config(),
                ik_solver=request_data.get("ik_solver"),
                ik_normalize=request_data.get("ik_normalize"),
            ),
            q_start=start_q,
            planning_timeout=planning_timeout,
        )
        verification = plan.get("verification", {}) or {}
        positioner_checked = self._planner_has_positioner_collision(planner)
        verification.update({
            "positioner_collision_checked": bool(positioner_checked),
            "positioner_collision_note": (
                None if positioner_checked
                else "positioner URDF is not included in this planner collision model"
            ),
        })
        plan["verification"] = verification
        timings.update(plan.get("timing", {}))
        timings["planner_setup"] = timings.get("planner_setup", time.perf_counter() - stage_t0)
        goal_q = np.asarray(plan["goal_q"], dtype=float)
        q_path = [np.asarray(q, dtype=float) for q in plan.get("q_path", [])]
        if plan.get("ik_failure"):
            self._last_ik_failure = getattr(self, "_last_ik_failure", {})
            self._last_ik_failure[robot_name] = plan["ik_failure"]
        self._last_inspection_ik_stats = getattr(self, "_last_inspection_ik_stats", {})
        ik_result = plan.get("ik_result", {})
        self._last_inspection_ik_stats[robot_name] = {
            "iterations": ik_result.get("iterations"),
            "elapsed": ik_result.get("elapsed"),
            "max_iter": ik_result.get("max_iter"),
            "solver": ik_result.get("solver"),
            "normalize": ik_result.get("normalize"),
            "converged": ik_result.get("success"),
            "position_only": False,
        }
        self._last_inspection_ik_trace = getattr(self, "_last_inspection_ik_trace", {})
        self._last_inspection_ik_trace[robot_name] = plan.get("ik_trace", [])
        self.__console.info(
            "inspection path IK q: "
            f"robot={robot_name}, "
            f"start_q={np.round(start_q, 6).tolist()}, "
            f"target_q={np.round(goal_q, 6).tolist()}")
        ik_experiment = self._save_inspection_ik_experiment(
            robot_name,
            robot_model,
            planner.pin_model,
            target_pose,
            goal_q,
            ik_result=ik_result,
        )

        # 5) q path를 viewer 표시용 TCP waypoint로 변환한다.
        stage_t0 = time.perf_counter()
        display_resolution = float(request_data.get("display_step_size", request_data.get("step_size", 0.08)))
        path = self._q_path_to_tcp_poses(
            robot_model,
            planner.pin_model,
            robot_name,
            q_path,
            sample_resolution=display_resolution)
        if len(path) < 2:
            raise RuntimeError("planned q path could not be converted to TCP path")
        timings["path_conversion"] = time.perf_counter() - stage_t0
        timings["total"] = time.perf_counter() - total_t0
        self.__console.info(
            "inspection path timing: "
            f"robot={robot_name}, planner={planner_name}, "
            f"target={timings.get('target_setup', 0.0):.3f}s, "
            f"setup={timings.get('planner_setup', 0.0):.3f}s, "
            f"ik={timings.get('ik', 0.0):.3f}s, "
            f"planning={timings.get('planning', 0.0):.3f}s, "
            f"verify={timings.get('collision_verification', 0.0):.3f}s, "
            f"convert={timings.get('path_conversion', 0.0):.3f}s, "
            f"total={timings.get('total', 0.0):.3f}s, "
            f"collision_edges={plan.get('verification', {}).get('colliding_edges', 0)}, "
            f"collision_preview={plan.get('collision_preview')}, "
            f"collision_preview_reason={plan.get('collision_preview_reason')}")
        plan.update({
            "path": [np.asarray(p, dtype=float) for p in path],
            "ik_experiment": ik_experiment,
            "timing": timings,
        })
        return plan

    def _plan_inspection_path(self, request_data):
        identity = request_data.get("_identity")
        result = {"status": "failed"}
        try:
            self._clear_ik_failure_visuals(render=False)
            target = getattr(self, '_inspection_point', None)
            if target is None:
                raise RuntimeError("inspection point is not selected")
            robot_name = request_data.get("robot", "rb20_1900es")
            self._show_inspection_goal_pose(robot_name, target, clear=True, render=True)
            plan = self._plan_inspection_path_for_robot(request_data, robot_name, target)
            q_path = plan["q_path"]
            path = plan["path"]
            self._last_inspection_q_path = [np.asarray(q, dtype=float) for q in q_path]
            self._last_inspection_edge_collisions = plan.get("edge_collisions", [])
            self._last_inspection_robot = robot_name
            self._last_inspection_path = [np.asarray(p, dtype=float) for p in path]
            self._last_inspection_plans = {robot_name: plan}
            self._show_inspection_ik_pose_result(
                robot_name,
                plan.get("ik_reached_T"),
                plan.get("ik_target_T"),
                success=not plan.get("ik_fallback", False),
                fallback=plan.get("ik_fallback", False),
            )
            self._show_inspection_goal_robot_pose(
                robot_name,
                plan["q_path"][-1],
                joint_names=plan.get("pin_joint_names"),
                clear=False,
                render=False,
            )
            self._show_inspection_path(path, robot_name=robot_name)
            if plan.get("ik_failure"):
                self._show_ik_failure_markers([robot_name], failure_infos={robot_name: plan["ik_failure"]})
            result = {
                "status": plan.get("status", "success"),
                "planner": plan["planner"],
                "robot": robot_name,
                "waypoints": plan["waypoints"],
                "init_q": np.asarray(q_path[0], dtype=float).round(6).tolist(),
                "target_q": np.asarray(q_path[-1], dtype=float).round(6).tolist(),
                "elapsed": plan["elapsed"],
                "start": plan["start"],
                "goal": plan["goal"],
                "verification": plan["verification"],
                "robot_links_considered": plan["robot_links_considered"],
                "collision_preview": plan["collision_preview"],
                "collision_preview_reason": plan.get("collision_preview_reason"),
                "fallback_reason": plan.get("fallback_reason"),
                "ik_fallback": plan.get("ik_fallback", False),
                "ik_failure": plan.get("ik_failure"),
                "ik_result": plan.get("ik_result"),
                "timing": plan.get("timing", {}),
            }
            if plan["collision_preview"]:
                self.__console.warning(
                    f"inspection q path kept for collision preview: {plan['planner']}, "
                    f"{len(q_path)} waypoints, {plan['elapsed']:.2f}s")
            else:
                self.__console.info(
                    f"inspection q path OK: {plan['planner']}, {len(q_path)} waypoints, {plan['elapsed']:.2f}s")
        except InspectionIKFailure as e:
            result = {
                "status": "failed",
                "message": str(e),
                "ik_failure": e.failure_info,
            }
            robot_name = request_data.get("robot", "rb20_1900es")
            self._show_ik_failure_markers([robot_name], failure_infos={robot_name: e.failure_info})
            self.__console.error(f"inspection path failed: {e}")
        except Exception as e:
            result = {"status": "failed", "message": str(e)}
            robot_name = request_data.get("robot", "rb20_1900es")
            self._show_ik_failure_markers([robot_name])
            self.__console.error(f"inspection path failed: {e}")
        if hasattr(self, 'zapi') and self.zapi and identity:
            self.zapi.reply_inspection_path(result, identity=identity)

    def _ef_pose_plan_targets(self):
        poses = getattr(self, '_ef_target_poses', {}) or {}
        targets = {}
        for pose_name, pose in poses.items():
            robot_name = self._ef_pose_robot_name(pose_name)
            target_T = self._pose_to_T(pose)
            targets[robot_name] = {
                "pose_name": pose_name,
                "target_T": target_T,
            }
        return targets

    def _ef_pose_plan_target_groups(self):
        groups = getattr(self, "_ef_pose_groups", []) or []
        target_groups = []
        for group_idx, group in enumerate(groups):
            poses = self._extract_ef_poses_from_group(group)
            targets = {}
            for pose_name, pose in poses.items():
                robot_name = self._ef_pose_robot_name(pose_name)
                target_T = self._pose_to_T(pose)
                targets[robot_name] = {
                    "pose_name": pose_name,
                    "target_T": target_T,
                    "group_index": group_idx,
                    "inspection_pose_name": f"寃???먯꽭 {group_idx + 1}",
                }
            if targets:
                target_groups.append({
                    "index": group_idx,
                    "name": f"寃???먯꽭 {group_idx + 1}",
                    "targets": targets,
                })
        if target_groups:
            return target_groups
        targets = self._ef_pose_plan_targets()
        if targets:
            return [{"index": 0, "name": "寃???먯꽭 1", "targets": targets}]
        return []

    def _ef_pose_target_origin(self):
        target = getattr(self, "_inspection_point", None)
        if target is None:
            return np.zeros(3, dtype=float)
        return np.asarray(target, dtype=float).reshape(3)

    def _ef_pose_direction_from_target(self, pose, target_origin):
        T = self._pose_to_T(pose)
        direction = T[:3, 3] - np.asarray(target_origin, dtype=float).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return np.zeros(3, dtype=float), T
        return direction / norm, T

    def _ef_pose_rt_pipe_facing_axis(self):
        frame_cfg = ((self._config.get("ef_pose", {}) or {}).get("frames", {}) or {}).get("rt", {}) or {}
        axis = np.asarray(frame_cfg.get("pipe_facing_axis", [0.0, -1.0, 0.0]), dtype=float).reshape(3)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return np.asarray([0.0, -1.0, 0.0], dtype=float)
        return axis / norm

    def _ef_pose_slot_priority(self, slot_data, target_origin):
        poses = self._extract_ef_poses_from_orientation_group(slot_data)
        dda_pose = poses.get("DDA")
        rt_items = [(name, pose) for name, pose in poses.items() if name.startswith("RT")]
        if dda_pose is None or not rt_items:
            return None

        dda_dir, _ = self._ef_pose_direction_from_target(dda_pose, target_origin)
        world_pos_y = np.asarray([0.0, 1.0, 0.0], dtype=float)
        world_neg_y = np.asarray([0.0, -1.0, 0.0], dtype=float)
        world_z = np.asarray([0.0, 0.0, 1.0], dtype=float)
        ideal_dda = np.asarray([1.0, 1.0, 0.0], dtype=float)
        ideal_dda /= np.linalg.norm(ideal_dda)
        ideal_rt_position = np.asarray([0.0, -1.0, 1.0], dtype=float)
        ideal_rt_position /= np.linalg.norm(ideal_rt_position)
        ideal_rt_view = np.asarray([0.0, 1.0, -1.0], dtype=float)
        ideal_rt_view /= np.linalg.norm(ideal_rt_view)
        ideal_rt_view_up = np.asarray([0.0, 1.0, 1.0], dtype=float)
        ideal_rt_view_up /= np.linalg.norm(ideal_rt_view_up)
        rt_local_facing = self._ef_pose_rt_pipe_facing_axis()

        dda_y = float(np.dot(dda_dir, world_pos_y))
        dda_ideal = float(np.dot(dda_dir, ideal_dda))
        dda_vertical = abs(float(np.dot(dda_dir, world_z)))
        best = None
        all_items = []
        for rt_name, rt_pose in rt_items:
            rt_dir, rt_T = self._ef_pose_direction_from_target(rt_pose, target_origin)
            rt_view = rt_T[:3, :3] @ rt_local_facing
            rt_view_norm = float(np.linalg.norm(rt_view))
            if rt_view_norm > 1e-9:
                rt_view = rt_view / rt_view_norm
            rt_neg_y = float(np.dot(rt_dir, world_neg_y))
            rt_ideal = float(np.dot(rt_dir, ideal_rt_position))
            rt_vertical = abs(float(np.dot(rt_dir, world_z)))
            rt_view_down45 = float(np.dot(rt_view, ideal_rt_view))
            rt_view_up45 = float(np.dot(rt_view, ideal_rt_view_up))
            rt_view_down = float(np.dot(rt_view, -world_z))
            rt_view_up = float(np.dot(rt_view, world_z))
            rt_view_to_pipe = float(np.dot(rt_view, -rt_dir))
            y_score = (1.0 - dda_y) + (1.0 - rt_neg_y)
            ideal_score = (
                1.4 * (1.0 - rt_view_down45)
                + 0.8 * (1.0 - rt_view_to_pipe)
                + 0.7 * (1.0 - dda_ideal)
                + 0.4 * (1.0 - rt_ideal)
            )
            vertical_score = -(dda_vertical + rt_vertical)
            preferred_y_side = dda_y > 0.25 and rt_neg_y > 0.25 and rt_view_down > 0.25
            sort_key = (
                0 if preferred_y_side else 1,
                ideal_score if preferred_y_side else vertical_score,
                y_score,
            )
            item = {
                "rt_name": rt_name,
                "rt_pose": rt_pose,
                "poses": {
                    "DDA": dda_pose,
                    rt_name: rt_pose,
                },
                "sort_key": sort_key,
                "metrics": {
                    "dda_y": dda_y,
                    "rt_neg_y": rt_neg_y,
                    "dda_ideal": dda_ideal,
                    "rt_ideal": rt_ideal,
                    "dda_vertical": dda_vertical,
                    "rt_vertical": rt_vertical,
                    "rt_view_down45": rt_view_down45,
                    "rt_view_up45": rt_view_up45,
                    "rt_view_down": rt_view_down,
                    "rt_view_up": rt_view_up,
                    "rt_view_to_pipe": rt_view_to_pipe,
                    "rt_view_x": float(rt_view[0]),
                    "rt_view_y": float(rt_view[1]),
                    "rt_view_z": float(rt_view[2]),
                },
            }
            all_items.append(item)
            if best is None or item["sort_key"] < best["sort_key"]:
                best = item
        if best is None:
            return None
        return {
            "poses": best["poses"],
            "rt_name": best["rt_name"],
            "sort_key": best["sort_key"],
            "metrics": best["metrics"],
            "alternatives": [
                {
                    "poses": item["poses"],
                    "rt_name": item["rt_name"],
                    "sort_key": item["sort_key"],
                    "metrics": item["metrics"],
                }
                for item in sorted(all_items, key=lambda item: item["sort_key"])
            ],
        }

    def _select_ef_pose_candidate_sets(self, ranked_slots, selected_limit):
        if not ranked_slots:
            return []
        limit = max(1, int(selected_limit))
        if limit == 1 or len(ranked_slots) == 1:
            return ranked_slots[:1]

        def _metric(item, key):
            return float((item.get("priority", {}).get("metrics", {}) or {}).get(key, 0.0))

        def _rt_view(item):
            metrics = item.get("priority", {}).get("metrics", {}) or {}
            return np.asarray([
                float(metrics.get("rt_view_x", 0.0)),
                float(metrics.get("rt_view_y", 0.0)),
                float(metrics.get("rt_view_z", 0.0)),
            ], dtype=float)

        def _same_candidate(a, b):
            return (
                int(a.get("group_index", -1)) == int(b.get("group_index", -2))
                and str(a.get("slot_name")) == str(b.get("slot_name"))
                and str(a.get("priority", {}).get("rt_name")) == str(b.get("priority", {}).get("rt_name"))
            )

        down_candidates = sorted(
            ranked_slots,
            key=lambda item: (
                -_metric(item, "rt_view_down45"),
                -_metric(item, "rt_view_to_pipe"),
                item["priority"]["sort_key"],
            ),
        )
        up_candidates = sorted(
            ranked_slots,
            key=lambda item: (
                -_metric(item, "rt_view_up45"),
                -_metric(item, "rt_view_to_pipe"),
                item["priority"]["sort_key"],
            ),
        )

        down = next((item for item in down_candidates if _metric(item, "rt_view_down45") > 0.25), None)
        up = next(
            (
                item for item in up_candidates
                if _metric(item, "rt_view_up45") > 0.25
                and (down is None or not _same_candidate(item, down))
            ),
            None,
        )
        if down is not None and up is not None:
            return [down, up][:limit]

        best_pair = None
        best_score = -float("inf")
        for i, first in enumerate(ranked_slots):
            for second in ranked_slots[i + 1:]:
                if _same_candidate(first, second):
                    continue
                v1 = _rt_view(first)
                v2 = _rt_view(second)
                dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
                vertical_separation = abs(_metric(first, "rt_view_z") - _metric(second, "rt_view_z"))
                opposite_vertical = (
                    1.0 if _metric(first, "rt_view_z") * _metric(second, "rt_view_z") < 0.0 else 0.0
                )
                side_separation = abs(_metric(first, "rt_neg_y") - _metric(second, "rt_neg_y"))
                score = 2.0 * opposite_vertical + vertical_separation + 0.5 * (1.0 - dot) + 0.2 * side_separation
                if score > best_score:
                    best_score = score
                    best_pair = (first, second)
        if best_pair is not None:
            return list(best_pair)[:limit]
        return ranked_slots[:limit]

    def _ef_pose_selected_candidate_target_groups(self):
        groups = getattr(self, "_ef_pose_groups", []) or []
        selected_group = groups[0] if groups else None
        target_groups = []
        if groups:
            target_origin = self._ef_pose_target_origin()
            ranked_slots = []

            def _rank_slot_sort_key(item):
                key, value = item
                try:
                    return (0, float(value.get("_actual_deg")))
                except Exception:
                    try:
                        return (0, float(key))
                    except Exception:
                        return (1, str(key))

            for group_idx, group in enumerate(groups):
                if not isinstance(group, dict):
                    continue
                slot_items = [
                    (slot_name, slot_data)
                    for slot_name, slot_data in sorted(group.items(), key=_rank_slot_sort_key)
                    if isinstance(slot_data, dict)
                ]
                for slot_name, slot_data in slot_items:
                    priority = self._ef_pose_slot_priority(slot_data, target_origin)
                    if priority is None:
                        continue
                    for candidate_priority in priority.get("alternatives", [priority]):
                        ranked_slots.append({
                            "group_index": group_idx,
                            "slot_name": str(slot_name),
                            "actual_deg": slot_data.get("_actual_deg"),
                            "priority": candidate_priority,
                        })

            ranked_slots.sort(key=lambda item: (
                item["priority"]["sort_key"],
                int(item["group_index"]),
                str(item["slot_name"]),
            ))

            params = self._config.get("ef_pose", {}) or {}
            selected_limit = int(params.get("selected_candidate_limit", 2))
            selected_slots = self._select_ef_pose_candidate_sets(ranked_slots, selected_limit)

            for slot_idx, item in enumerate(selected_slots):
                priority = item["priority"]
                targets = {}
                for pose_name, pose in priority["poses"].items():
                    robot_name = self._ef_pose_robot_name(pose_name)
                    targets[robot_name] = {
                        "pose_name": pose_name,
                        "target_T": self._pose_to_T(pose),
                        "group_index": int(item["group_index"]),
                        "slot_name": str(item["slot_name"]),
                        "inspection_pose_name": f"野꺜???癒?쉭 {slot_idx + 1}",
                        "priority": priority["metrics"],
                    }
                if targets:
                    actual_deg = item.get("actual_deg")
                    suffix = f" ({actual_deg} deg)" if actual_deg is not None else f" ({item['slot_name']})"
                    target_groups.append({
                        "index": slot_idx,
                        "group_index": int(item["group_index"]),
                        "slot_name": str(item["slot_name"]),
                        "rt_name": priority["rt_name"],
                        "priority": priority["metrics"],
                        "name": f"野꺜???癒?쉭 {slot_idx + 1}{suffix}",
                        "targets": targets,
                    })

            if target_groups:
                preview = []
                for group in target_groups[:8]:
                    metrics = group.get("priority", {})
                    preview.append(
                        f"{group['name']}:group={group.get('group_index')},slot={group.get('slot_name')},"
                        f"rt={group.get('rt_name')},dda_y={metrics.get('dda_y', 0.0):.3f},"
                        f"rt_-y={metrics.get('rt_neg_y', 0.0):.3f},"
                        f"rt_down45={metrics.get('rt_view_down45', 0.0):.3f},"
                        f"rt_up45={metrics.get('rt_view_up45', 0.0):.3f}"
                    )
                self.__console.info("EF pose candidate priority: " + " | ".join(preview))
                return target_groups
        if selected_group:
            def _slot_sort_key(item):
                key, value = item
                try:
                    return (0, float(key))
                except Exception:
                    try:
                        return (0, float(value.get("_actual_deg")))
                    except Exception:
                        return (1, str(key))

            slot_items = [
                (slot_name, slot_data)
                for slot_name, slot_data in sorted(selected_group.items(), key=_slot_sort_key)
                if isinstance(slot_data, dict)
            ]
            for slot_idx, (slot_name, slot_data) in enumerate(slot_items):
                poses = self._extract_ef_poses_from_orientation_group(slot_data)
                targets = {}
                for pose_name, pose in poses.items():
                    robot_name = self._ef_pose_robot_name(pose_name)
                    targets[robot_name] = {
                        "pose_name": pose_name,
                        "target_T": self._pose_to_T(pose),
                        "group_index": 0,
                        "slot_name": str(slot_name),
                        "inspection_pose_name": f"寃???먯꽭 {slot_idx + 1}",
                    }
                if targets:
                    actual_deg = slot_data.get("_actual_deg")
                    suffix = f" ({actual_deg} deg)" if actual_deg is not None else f" ({slot_name})"
                    target_groups.append({
                        "index": slot_idx,
                        "slot_name": str(slot_name),
                        "name": f"寃???먯꽭 {slot_idx + 1}{suffix}",
                        "targets": targets,
                    })
        if target_groups:
            return target_groups
        targets = self._ef_pose_plan_targets()
        if targets:
            return [{"index": 0, "name": "寃???먯꽭 1", "targets": targets}]
        return []

    def _check_ef_pose_ik(self, request_data):
        identity = request_data.get("_identity")
        result = {"status": "failed"}
        total_t0 = time.perf_counter()
        failures = {}
        ik_failures = {}
        try:
            self._clear_inspection_visuals(clear_point=False)
            target_groups = self._ef_pose_selected_candidate_target_groups()
            if not target_groups:
                raise RuntimeError("EF poses are not determined")
            self._clear_inspection_goal_pose_visuals(render=False)
            for group_info in target_groups:
                for robot_name, target_info in group_info["targets"].items():
                    self._show_inspection_goal_pose(
                        robot_name,
                        target_info["target_T"],
                        clear=False,
                        render=False,
                    )
            self.plotter.render()

            stage_t0 = time.perf_counter()
            obstacle_mesh = self._current_spool_collision_mesh()
            if obstacle_mesh is None:
                raise RuntimeError("loaded pipe is not available")
            obstacle_elapsed = time.perf_counter() - stage_t0

            group_sequence = []
            for group_info in target_groups:
                group_name = group_info["name"]
                targets = group_info["targets"]
                checks = {}
                group_failures = {}
                group_ik_failures = {}
                group_request = dict(request_data)
                group_request["_start_q_override_by_robot"] = {}
                self.__console.info(
                    f"EF pose IK check: {group_name} ({len(targets)} robots), "
                    "start_q=zeros")
                for robot_name, target_info in targets.items():
                    failure_key = f"{group_name}:{robot_name}"
                    try:
                        check = self._check_inspection_ik_for_robot(
                            group_request,
                            robot_name,
                            target_info["target_T"],
                            obstacle_mesh,
                        )
                        check["pose_name"] = target_info["pose_name"]
                        check["inspection_pose_name"] = group_name
                        check["inspection_pose_index"] = int(group_info["index"])
                        checks[robot_name] = check
                        if check.get("ik_failure"):
                            group_ik_failures[robot_name] = check["ik_failure"]
                            ik_failures[failure_key] = check["ik_failure"]
                            self._last_ik_failure = getattr(self, "_last_ik_failure", {})
                            self._last_ik_failure[robot_name] = check["ik_failure"]
                    except InspectionIKFailure as exc:
                        group_failures[robot_name] = str(exc)
                        failures[failure_key] = str(exc)
                        if exc.failure_info:
                            group_ik_failures[robot_name] = exc.failure_info
                            ik_failures[failure_key] = exc.failure_info
                            self._last_ik_failure = getattr(self, "_last_ik_failure", {})
                            self._last_ik_failure[robot_name] = exc.failure_info
                        self.__console.error(f"EF pose IK check failed for {failure_key}: {exc}")
                    except Exception as exc:
                        group_failures[robot_name] = str(exc)
                        failures[failure_key] = str(exc)
                        self.__console.error(f"EF pose IK check failed for {failure_key}: {exc}")
                group_sequence.append({
                    "index": int(group_info["index"]),
                    "name": group_name,
                    "checks": checks,
                    "failures": group_failures,
                    "ik_failures": group_ik_failures,
                })

            if ik_failures:
                plain_ik_failures = {
                    key.split(":")[-1]: value
                    for key, value in ik_failures.items()
                }
                self._show_ik_failure_markers(plain_ik_failures.keys(), failure_infos=plain_ik_failures)
            elif failures:
                self._show_ik_failure_markers([key.split(":")[-1] for key in failures.keys()])

            for group in group_sequence:
                for robot_name, check in group["checks"].items():
                    self._show_inspection_ik_pose_result(
                        robot_name,
                        check.get("ik_reached_T"),
                        check.get("ik_target_T"),
                        success=not check.get("ik_fallback", False),
                        fallback=check.get("ik_fallback", False),
                    )
                    self._show_inspection_goal_robot_pose(
                        robot_name,
                        check.get("goal_q"),
                        joint_names=check.get("pin_joint_names"),
                        clear=False,
                        render=False,
                    )
            self.plotter.render()

            all_checks = {
                f"{group['name']}:{robot_name}": check
                for group in group_sequence
                for robot_name, check in group["checks"].items()
            }
            if not all_checks:
                raise RuntimeError(f"all EF pose IK checks failed: {failures}")

            wall_elapsed = time.perf_counter() - total_t0
            first_group = next(group for group in group_sequence if group["checks"])
            result = {
                "mode": "ik_check",
                "status": "success" if not failures and not ik_failures else "partial",
                "planner": request_data.get("planner", "rrt_connect"),
                "inspection_groups": [
                    {
                        "name": group["name"],
                        "index": group["index"],
                        "robots": {
                            robot_name: {
                                "pose_name": check.get("pose_name"),
                                "init_q": np.asarray(check["start_q"], dtype=float).round(6).tolist(),
                                "target_q": np.asarray(check["goal_q"], dtype=float).round(6).tolist(),
                                "ik_fallback": check.get("ik_fallback", False),
                                "ik_result": check.get("ik_result"),
                                "ik_solver": check.get("ik_solver"),
                                "ik_normalize": check.get("ik_normalize"),
                                "timing": check.get("timing", {}),
                            }
                            for robot_name, check in group["checks"].items()
                        },
                        "failures": group["failures"],
                    }
                    for group in group_sequence
                ],
                "robots": {
                    robot_name: {
                        "pose_name": check.get("pose_name"),
                        "init_q": np.asarray(check["start_q"], dtype=float).round(6).tolist(),
                        "target_q": np.asarray(check["goal_q"], dtype=float).round(6).tolist(),
                        "ik_fallback": check.get("ik_fallback", False),
                        "ik_result": check.get("ik_result"),
                        "ik_solver": check.get("ik_solver"),
                        "ik_normalize": check.get("ik_normalize"),
                        "timing": check.get("timing", {}),
                    }
                    for robot_name, check in first_group["checks"].items()
                },
                "failures": failures,
                "ik_failures": ik_failures,
                "wall_elapsed": wall_elapsed,
                "timing": {
                    "obstacle_mesh": obstacle_elapsed,
                    "ik_wall": wall_elapsed,
                    "ik_sum": float(sum(
                        (check.get("timing", {}) or {}).get("ik", 0.0)
                        for check in all_checks.values())),
                },
            }
            self.__console.info(
                "EF pose IK checked by inspection pose: "
                + " | ".join(
                    f"{group['name']}["
                    + ", ".join(
                        f"{robot}(success={check.get('ik_result', {}).get('success', False)}, "
                        f"solver={check.get('ik_solver')}, "
                        f"normalize={check.get('ik_normalize')}, "
                        f"fallback={check.get('ik_fallback', False)})"
                        for robot, check in group["checks"].items())
                    + "]"
                    for group in group_sequence if group["checks"])
                + f", wall={wall_elapsed:.3f}s, obstacle={obstacle_elapsed:.3f}s")
        except Exception as e:
            elapsed = time.perf_counter() - total_t0
            result = {
                "mode": "ik_check",
                "status": "failed",
                "message": str(e),
                "elapsed": elapsed,
                "failures": failures,
                "ik_failures": ik_failures,
            }
            self.__console.error(f"EF pose IK check failed after {elapsed:.3f}s: {e}")
        if hasattr(self, 'zapi') and self.zapi and identity:
            self.zapi.reply_inspection_path(result, identity=identity)

    def _plan_ef_pose_paths(self, request_data):
        identity = request_data.get("_identity")
        result = {"status": "failed"}
        total_t0 = time.perf_counter()
        failures = {}
        ik_failures = {}
        try:
            self._clear_inspection_visuals(clear_point=False)
            target_groups = self._ef_pose_selected_candidate_target_groups()
            if not target_groups:
                raise RuntimeError("EF poses are not determined")
            self._clear_inspection_goal_pose_visuals(render=False)
            for group_info in target_groups:
                for robot_name, target_info in group_info["targets"].items():
                    self._show_inspection_goal_pose(
                        robot_name,
                        target_info["target_T"],
                        clear=False,
                        render=False,
                    )
            self.plotter.render()
            stage_t0 = time.perf_counter()
            obstacle_mesh = self._current_spool_collision_mesh()
            if obstacle_mesh is None:
                raise RuntimeError("loaded pipe is not available")
            obstacle_elapsed = time.perf_counter() - stage_t0

            group_sequence = []
            start_q_overrides = {}
            planning_timeout = float(request_data.get(
                "planning_timeout",
                (self._config.get("path_planning", {}) or {}).get("planning_timeout", 0.0),
            ))
            future_timeout = None if planning_timeout <= 0 else planning_timeout + 2.0

            for group_info in target_groups:
                group_name = group_info["name"]
                targets = group_info["targets"]
                plans = {}
                group_failures = {}
                group_ik_failures = {}
                group_request = dict(request_data)
                group_request["_start_q_override_by_robot"] = {
                    name: np.asarray(q, dtype=float).tolist()
                    for name, q in start_q_overrides.items()
                }
                max_workers = min(len(targets), int(request_data.get("max_workers", len(targets))))
                start_source = "previous inspection pose" if start_q_overrides else "current robot pose"
                self.__console.info(
                    f"EF pose path planning: {group_name} ({len(targets)} robots), "
                    f"start={start_source}, start_overrides={list(start_q_overrides.keys())}")
                executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
                futures = {
                    executor.submit(
                        self._plan_inspection_path_for_robot,
                        group_request,
                        robot_name,
                        target_info["target_T"],
                        obstacle_mesh,
                    ): (robot_name, target_info)
                    for robot_name, target_info in targets.items()
                }
                try:
                    for future in as_completed(futures, timeout=future_timeout):
                        robot_name, target_info = futures[future]
                        failure_key = f"{group_name}:{robot_name}"
                        try:
                            plan = future.result()
                            plan["pose_name"] = target_info["pose_name"]
                            plan["inspection_pose_name"] = group_name
                            plan["inspection_pose_index"] = int(group_info["index"])
                            plans[robot_name] = plan
                            if plan.get("q_path"):
                                start_q_overrides[robot_name] = np.asarray(plan["q_path"][-1], dtype=float)
                            if plan.get("ik_failure"):
                                group_ik_failures[robot_name] = plan["ik_failure"]
                                ik_failures[failure_key] = plan["ik_failure"]
                                self._last_ik_failure = getattr(self, "_last_ik_failure", {})
                                self._last_ik_failure[robot_name] = plan["ik_failure"]
                        except InspectionIKFailure as exc:
                            group_failures[robot_name] = str(exc)
                            failures[failure_key] = str(exc)
                            if exc.failure_info:
                                group_ik_failures[robot_name] = exc.failure_info
                                ik_failures[failure_key] = exc.failure_info
                                self._last_ik_failure = getattr(self, "_last_ik_failure", {})
                                self._last_ik_failure[robot_name] = exc.failure_info
                            self.__console.error(f"EF pose path failed for {failure_key}: {exc}")
                        except Exception as exc:
                            group_failures[robot_name] = str(exc)
                            failures[failure_key] = str(exc)
                            self.__console.error(f"EF pose path failed for {failure_key}: {exc}")
                except FuturesTimeoutError:
                    for future, (robot_name, _target_info) in futures.items():
                        if future.done():
                            continue
                        future.cancel()
                        failure_key = f"{group_name}:{robot_name}"
                        group_failures[robot_name] = f"path planning timeout ({planning_timeout:.1f}s)"
                        failures[failure_key] = group_failures[robot_name]
                        self.__console.error(f"EF pose path timeout for {failure_key}: {planning_timeout:.1f}s")
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

                group_sequence.append({
                    "index": int(group_info["index"]),
                    "name": group_name,
                    "plans": plans,
                    "failures": group_failures,
                    "ik_failures": group_ik_failures,
                })

            if ik_failures:
                plain_ik_failures = {
                    key.split(":")[-1]: value
                    for key, value in ik_failures.items()
                }
                self._show_ik_failure_markers(plain_ik_failures.keys(), failure_infos=plain_ik_failures)
            elif failures:
                self._show_ik_failure_markers([key.split(":")[-1] for key in failures.keys()])

            all_plans = {
                f"{group['name']}:{robot_name}": plan
                for group in group_sequence
                for robot_name, plan in group["plans"].items()
            }
            if not all_plans:
                raise RuntimeError(f"all EF pose path plans failed: {failures}")
            plan_wall_elapsed = time.perf_counter() - total_t0

            self._last_inspection_plan_sequence = [
                {"name": group["name"], "plans": group["plans"]}
                for group in group_sequence
                if group["plans"]
            ]
            first_group = next(group for group in group_sequence if group["plans"])
            self._last_inspection_plans = first_group["plans"]
            for group in group_sequence:
                for robot_name, plan in group["plans"].items():
                    self._show_inspection_ik_pose_result(
                        robot_name,
                        plan.get("ik_reached_T"),
                        plan.get("ik_target_T"),
                        success=not plan.get("ik_fallback", False),
                        fallback=plan.get("ik_fallback", False),
                    )
                    self._show_inspection_goal_robot_pose(
                        robot_name,
                        plan["q_path"][-1],
                        joint_names=plan.get("pin_joint_names"),
                        clear=False,
                        render=False,
                    )
                    self._show_inspection_path(plan["path"], robot_name=robot_name, clear=False)
                    if plan.get("planning_error") and plan.get("reached_T") is not None:
                        self._show_ik_failure_reached_pose(
                            robot_name,
                            plan.get("reached_T"),
                            None,
                        )

            # Keep one path as the legacy playback target; start simulation uses all plans.
            first_robot, first_plan = next(iter(first_group["plans"].items()))
            self._last_inspection_q_path = first_plan["q_path"]
            self._last_inspection_edge_collisions = first_plan.get("edge_collisions", [])
            self._last_inspection_robot = first_robot
            self._last_inspection_path = first_plan["path"]

            has_partial_plan = any(plan.get("status") != "success" for plan in all_plans.values())
            result = {
                "status": "success" if not failures and not ik_failures and not has_partial_plan else "partial",
                "planner": request_data.get("planner", "rrt_connect"),
                "inspection_groups": [
                    {
                        "name": group["name"],
                        "index": group["index"],
                        "robots": {
                            robot_name: {
                                "pose_name": plan.get("pose_name"),
                                "waypoints": plan["waypoints"],
                                "init_q": np.asarray(plan["q_path"][0], dtype=float).round(6).tolist(),
                                "target_q": np.asarray(plan["q_path"][-1], dtype=float).round(6).tolist(),
                                "elapsed": plan["elapsed"],
                                "verification": plan["verification"],
                                "collision_preview": plan["collision_preview"],
                                "collision_preview_reason": plan.get("collision_preview_reason"),
                                "fallback_reason": plan.get("fallback_reason"),
                                "ik_fallback": plan.get("ik_fallback", False),
                                "ik_result": plan.get("ik_result"),
                                "ik_solver": plan.get("ik_solver"),
                                "ik_normalize": plan.get("ik_normalize"),
                                "planning_error": plan.get("planning_error"),
                                "timing": plan.get("timing", {}),
                            }
                            for robot_name, plan in group["plans"].items()
                        },
                        "failures": group["failures"],
                    }
                    for group in group_sequence
                ],
                "robots": {
                    robot_name: {
                        "pose_name": plan.get("pose_name"),
                        "waypoints": plan["waypoints"],
                        "init_q": np.asarray(plan["q_path"][0], dtype=float).round(6).tolist(),
                        "target_q": np.asarray(plan["q_path"][-1], dtype=float).round(6).tolist(),
                        "elapsed": plan["elapsed"],
                        "verification": plan["verification"],
                        "collision_preview": plan["collision_preview"],
                        "collision_preview_reason": plan.get("collision_preview_reason"),
                        "fallback_reason": plan.get("fallback_reason"),
                        "ik_fallback": plan.get("ik_fallback", False),
                        "ik_result": plan.get("ik_result"),
                        "ik_solver": plan.get("ik_solver"),
                        "ik_normalize": plan.get("ik_normalize"),
                        "planning_error": plan.get("planning_error"),
                        "timing": plan.get("timing", {}),
                    }
                    for robot_name, plan in first_group["plans"].items()
                },
                "failures": failures,
                "ik_failures": ik_failures,
                "total_elapsed": float(sum(plan["elapsed"] for plan in all_plans.values())),
                "wall_elapsed": plan_wall_elapsed,
                "timing": {
                    "obstacle_mesh": obstacle_elapsed,
                    "planning_wall": plan_wall_elapsed,
                    "planning_sum": float(sum(plan["elapsed"] for plan in all_plans.values())),
                },
            }
            self.__console.info(
                "EF pose paths planned by inspection pose: "
                + " | ".join(
                    f"{group['name']}["
                    + ", ".join(
                        f"{robot}({plan['waypoints']} wp, {plan['elapsed']:.3f}s)"
                        for robot, plan in group["plans"].items())
                    + "]"
                    for group in group_sequence if group["plans"])
                + f", wall={plan_wall_elapsed:.3f}s, obstacle={obstacle_elapsed:.3f}s")
        except Exception as e:
            elapsed = time.perf_counter() - total_t0
            result = {
                "status": "failed",
                "message": str(e),
                "elapsed": elapsed,
                "failures": failures,
                "ik_failures": ik_failures,
            }
            self.__console.error(f"EF pose path planning failed after {elapsed:.3f}s: {e}")
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

        # 3. Step manipulator joint animations (蹂닿컙 ?대룞)
        now = time.time()
        dt = 0.0 if self._last_anim_time is None else (now - self._last_anim_time)
        self._last_anim_time = now
        if self._joint_animations and dt > 0:
            self._step_joint_animations(min(dt, 0.1))   # ??dt???대옩??
        if (getattr(self, '_path_playback', None) is not None
                or getattr(self, '_robot_path_playback', None) is not None) and dt > 0:
            self._step_path_playback(min(dt, 0.1))

        return True

    def _find_robot(self, name):
        for m in getattr(self, '_robot_models', []):
            if getattr(m, 'name', None) == name:
                return m
        return None

    def _robot_urdf_path(self, name, root_path=None, default_path=None):
        root_path = os.path.abspath(str(root_path or self._config.get("root_path", os.getcwd())))
        for item in self._config.get("urdf", []) or []:
            if item.get("name") != name:
                continue
            path = item.get("path") or default_path
            if not path:
                break
            return path if os.path.isabs(path) else os.path.join(root_path, path)
        if default_path:
            return default_path if os.path.isabs(default_path) else os.path.join(root_path, default_path)
        raise RuntimeError(f"URDF path is not configured: {name}")

    def _step_joint_animations(self, dt):
        """?쒖꽦 議곗씤???좊땲硫붿씠?섏쓣 ?щ떎由ш섦 ?띾룄 ?꾨줈?뚯씪濡????ㅽ뀦 吏꾪뻾.
        媛??accel)?쇰줈 max_speed源뚯? ?щ┛ ???쒗빆, target ?꾨떖 ??媛먯냽???뺤?.
        """
        still = []
        changed = False
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

            # ?뺤? ?꾨컯: ?⑥? 嫄곕━쨌?띾룄媛 異⑸텇???묒쑝硫??ㅻ깄
            if dist <= 1e-6 and vel <= accel * dt:
                model.set_joint(jn, tgt); model.update_fk()
                changed = True
                continue

            # 媛먯냽???꾩슂??嫄곕━ = v짼 / (2a). 洹몃낫??媛源뚯슦硫?媛먯냽, ?꾨땲硫?媛???쒗빆
            stop_dist = (vel * vel) / (2.0 * accel)
            if dist <= stop_dist:
                vel = max(0.0, vel - accel * dt)      # 媛먯냽
            else:
                vel = min(vmax, vel + accel * dt)     # 媛????vmax ?쒗빆

            new_cur = cur + direction * vel * dt
            # target??吏?섏튂硫??ㅻ깄?섍퀬 醫낅즺
            if (tgt - new_cur) * direction <= 0:
                model.set_joint(jn, tgt); model.update_fk()
                changed = True
                continue

            anim["vel"] = vel
            model.set_joint(jn, new_cur); model.update_fk()
            changed = True
            still.append(anim)
        self._joint_animations = still
        if changed:
            self._show_robot_tcp_axes(render=False)
            self._send_robot_joint_state_update(throttle_s=0.0 if not still else 0.05)
            self.plotter.render()

    def _set_joint_animation(self, robot_name, joint_name, target, speed, accel=None, identity=None):
        """?대떦 濡쒕큸/議곗씤?몄쓽 湲곗〈 ?좊땲硫붿씠?섏쓣 援먯껜?섍퀬 ?щ떎由ш섦 ?꾨줈?뚯씪濡??대룞 ?쒖옉.
        accel 誘몄?????speed??2諛???0.5s 媛??濡?湲곕낯 ?ㅼ젙.
        """
        model = self._find_robot(robot_name)
        if model is None or model._urdf is None:
            self.__console.warning(f"move_manipulator: 濡쒕큸 ?놁쓬 '{robot_name}'")
            return
        if joint_name not in model._urdf._joint_map:
            self.__console.warning(f"move_manipulator: 議곗씤???놁쓬 '{joint_name}'")
            return
        spd = float(speed)
        acc = float(accel) if accel is not None else max(spd * 2.0, 1e-6)
        # 媛숈? (robot, joint)???꾩옱 ?띾룄???댁뼱諛쏆븘 遺?쒕읇寃??ы?寃뚰똿
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
        if identity is not None:
            self._robot_joint_state_identity = identity
        self.__console.info(
            f"move_manipulator: {robot_name}.{joint_name} ??{target} (vmax={spd}, accel={acc})")

    def _stop_joint_animation(self, robot_name, joint_name=None):
        """?대떦 濡쒕큸(?먮뒗 ?뱀젙 議곗씤?????좊땲硫붿씠?섏쓣 利됱떆 以묒?."""
        model = self._find_robot(robot_name)
        self._joint_animations = [
            a for a in self._joint_animations
            if not (a["model"] is model and (joint_name is None or a["joint"] == joint_name))
        ]
        self.__console.info(f"stop_manipulator: {robot_name} {joint_name or '(all)'}")

    def _reset_robot_base_pose(self, robot_name=None, identity=None):
        """Reset collaborative robot joints to their URDF zero/base configuration."""
        target_names = None
        if robot_name:
            if isinstance(robot_name, (list, tuple, set)):
                target_names = {str(name) for name in robot_name}
            else:
                target_names = {str(robot_name)}

        reset_names = []
        self._robot_path_playback = None
        self._path_playback = None
        self._clear_path_playback_marker()
        self._clear_collision_highlights()
        self._clear_ik_failure_visuals(render=False)
        for model in getattr(self, '_robot_models', []):
            name = getattr(model, 'name', None)
            if not name or name == "positioner":
                continue
            if target_names is not None and name not in target_names:
                continue
            urdf = getattr(model, '_urdf', None)
            if urdf is None:
                continue
            for joint in getattr(urdf, 'joints', []):
                if getattr(joint, 'type', None) == "fixed":
                    continue
                model.set_joint(joint.name, 0.0)
            model.update_fk()
            reset_names.append(name)

        if not reset_names:
            self.__console.warning(f"reset_robot_base_pose: no robot matched ({robot_name or 'all'})")
            return False
        self._show_robot_tcp_axes(render=False)
        self._send_robot_joint_state_update(reset_names, identity=identity, throttle_s=0.0)
        self.plotter.render()
        self.__console.info(f"reset_robot_base_pose: reset {reset_names}")
        return True

    def _inspection_plan_collision_reason(self, plan):
        if not plan:
            return None
        verification = plan.get("verification") or {}
        colliding_edges = int(verification.get("colliding_edges", 0) or 0)
        edge_collisions = plan.get("edge_collisions") or verification.get("edge_collisions") or []
        if plan.get("collision_preview"):
            if plan.get("planning_error"):
                return f"planner_error={plan.get('planning_error')}"
            if colliding_edges:
                return f"colliding_edges={colliding_edges}"
            return "collision_preview"
        if colliding_edges:
            return f"colliding_edges={colliding_edges}"
        if edge_collisions:
            return f"edge_collisions={len(edge_collisions)}"
        return None

    def _warn_collision_preview_playback(self, plans):
        risky = {}
        for robot_name, plan in (plans or {}).items():
            reason = self._inspection_plan_collision_reason(plan)
            if reason:
                risky[robot_name] = reason
        if not risky:
            return False
        self._clear_collision_highlights()
        for plan in (plans or {}).values():
            edge_collisions = plan.get("edge_collisions") or []
            if edge_collisions:
                self._highlight_collision_pairs(edge_collisions[0].get("pairs", []))
        self.__console.warning(
            "execute_inspection_path: planned path is not collision-free; playback is allowed for inspection | "
            + ", ".join(f"{robot}({reason})" for robot, reason in risky.items()))
        self.plotter.render()
        return True

    def _start_path_playback(self, speed=0.2, identity=None):
        """Replay the last planned inspection q path by moving the robot model."""
        if identity is not None:
            self._robot_joint_state_identity = identity
        sequence = getattr(self, "_last_inspection_plan_sequence", []) or []
        if sequence:
            return self._start_inspection_sequence_path_playback(sequence, speed=speed, identity=identity)
        plans = getattr(self, '_last_inspection_plans', {}) or {}
        if plans:
            self._warn_collision_preview_playback(plans)
        if len(plans) > 1:
            return self._start_multi_robot_path_playback(plans, speed=speed, identity=identity)

        q_path = getattr(self, '_last_inspection_q_path', None)
        robot_name = getattr(self, '_last_inspection_robot', None)
        model = self._find_robot(robot_name) if robot_name else None
        if q_path is None or len(q_path) < 2 or model is None:
            self.__console.warning("execute_inspection_path: planned path媛 ?놁뒿?덈떎")
            return False

        if getattr(self, '_last_inspection_edge_collisions', []):
            self.__console.warning(
                "execute_inspection_path: planned path has collision edges; playback is allowed for inspection")

        q_pts = np.asarray([np.asarray(q, dtype=float) for q in q_path], dtype=float)
        seg_lengths = np.linalg.norm(np.diff(q_pts, axis=0), axis=1)
        if not np.any(seg_lengths > 1e-9):
            self.__console.warning("execute_inspection_path: q path length is zero")
            return False

        pin_model = self._build_pin_model_for_robot(model)
        if pin_model is None:
            self.__console.warning("execute_inspection_path: Pinocchio model ?앹꽦 ?ㅽ뙣")
            return False

        self._clear_collision_highlights()
        path = getattr(self, '_last_inspection_path', None)
        pts = np.asarray([np.asarray(p, dtype=float)[:3] for p in path], dtype=float) if path else None
        self._clear_path_playback_marker()
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
        self._send_robot_joint_state_update([robot_name], identity=identity)
        self.plotter.render()
        self.__console.info(f"execute_inspection_path: robot playback started ({len(q_pts)} waypoints)")
        return True

    def _start_inspection_sequence_path_playback(self, sequence, speed=0.2, identity=None):
        valid_sequence = [group for group in sequence if group.get("plans")]
        if not valid_sequence:
            self.__console.warning("execute_inspection_path: inspection pose sequence is empty")
            return False
        self._inspection_sequence_playback = {
            "sequence": valid_sequence,
            "index": 0,
            "speed": max(float(speed), 1e-6),
            "identity": identity,
        }
        return self._start_next_inspection_sequence_group()

    def _start_next_inspection_sequence_group(self):
        seq_state = getattr(self, "_inspection_sequence_playback", None)
        if not seq_state:
            return False
        sequence = seq_state.get("sequence", [])
        idx = int(seq_state.get("index", 0))
        if idx >= len(sequence):
            self._inspection_sequence_playback = None
            self.__console.info("execute_inspection_path: inspection pose sequence playback finished")
            return False
        group = sequence[idx]
        seq_state["index"] = idx + 1
        self.__console.info(
            f"execute_inspection_path: start {group.get('name', f'inspection pose {idx + 1}')} "
            f"({idx + 1}/{len(sequence)})")
        return self._start_multi_robot_path_playback(
            group.get("plans", {}),
            speed=seq_state.get("speed", 0.2),
            identity=seq_state.get("identity"),
        )

    def _start_multi_robot_path_playback(self, plans, speed=0.2, identity=None):
        if identity is not None:
            self._robot_joint_state_identity = identity
        self._warn_collision_preview_playback(plans)
        self._clear_collision_highlights()
        self._clear_path_playback_marker()

        playback_robots = {}
        markers = {}
        for robot_name, plan in plans.items():
            q_path = plan.get("q_path")
            model = self._find_robot(robot_name)
            if q_path is None or len(q_path) < 2 or model is None:
                self.__console.warning(f"execute_inspection_path: skip {robot_name}; planned path is missing")
                continue
            q_pts = np.asarray([np.asarray(q, dtype=float) for q in q_path], dtype=float)
            seg_lengths = np.linalg.norm(np.diff(q_pts, axis=0), axis=1)
            if not np.any(seg_lengths > 1e-9):
                self.__console.warning(f"execute_inspection_path: skip {robot_name}; path length is zero")
                continue
            pin_model = self._build_pin_model_for_robot(model)
            if pin_model is None:
                self.__console.warning(f"execute_inspection_path: skip {robot_name}; Pinocchio model failed")
                continue

            path = plan.get("path")
            pts = np.asarray([np.asarray(p, dtype=float)[:3] for p in path], dtype=float) if path else None
            if pts is not None and len(pts) > 0:
                color = "gold" if robot_name == "dda_rb10_1300e" else "dodgerblue"
                marker = vedo.Sphere(pos=pts[0], r=0.055, c=color)
                marker.pickable(False)
                markers[robot_name] = marker
                self.plotter.add(marker)

            playback_robots[robot_name] = {
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
                    for item in plan.get("edge_collisions", [])
                },
                "logged_collision_edges": set(),
            }
            if playback_robots[robot_name]["edge_collisions"]:
                edges = sorted(playback_robots[robot_name]["edge_collisions"].keys())
                self.__console.warning(
                    f"execute_inspection_path: {robot_name} collision detected between waypoints {edges}")
                self._log_path_playback_collision(playback_robots[robot_name], 0)
            self._apply_robot_q(model, pin_model, q_pts[0])

        if not playback_robots:
            self.__console.warning("execute_inspection_path: planned path媛 ?놁뒿?덈떎")
            return False
        self._path_playback_marker = markers
        self._robot_path_playback = playback_robots
        self._path_playback = None
        self._send_robot_joint_state_update(playback_robots.keys(), identity=identity)
        self._show_robot_tcp_axes(render=False)
        self.plotter.render()
        self.__console.info(
            "execute_inspection_path: multi-robot playback started ("
            + ", ".join(f"{name}:{len(rb['q_points'])} wp" for name, rb in playback_robots.items())
            + ")")
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
            return self._build_pin_model_from_urdf(model.urdf_path)
        except Exception:
            return None

    def _step_robot_path_playback(self, dt):
        rb = getattr(self, '_robot_path_playback', None)
        if rb is None:
            return
        if isinstance(rb, dict) and "q_points" not in rb:
            updated_names = list(rb.keys())
            finished = []
            for robot_name, robot_pb in list(rb.items()):
                if self._step_single_robot_path_playback(robot_pb, dt, render=False):
                    finished.append(robot_name)
            for robot_name in finished:
                rb.pop(robot_name, None)
            if not rb:
                self._send_robot_joint_state_update(updated_names, throttle_s=0.0)
                self._robot_path_playback = None
                if getattr(self, "_inspection_sequence_playback", None):
                    self.__console.info("execute_inspection_path: multi-robot playback finished; moving to next inspection pose")
                    self._start_next_inspection_sequence_group()
                else:
                    self.__console.info("execute_inspection_path: multi-robot playback finished")
            else:
                self._send_robot_joint_state_update(updated_names, throttle_s=0.05)
            self._show_robot_tcp_axes(render=False)
            self.plotter.render()
            return

        finished = self._step_single_robot_path_playback(rb, dt, render=True)
        self._send_robot_joint_state_update([rb["robot_name"]], throttle_s=0.0 if finished else 0.05)

    def _step_single_robot_path_playback(self, rb, dt, render=True):
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
            if render:
                self._robot_path_playback = None
                self.__console.info("execute_inspection_path: robot playback finished")
            finished = True
        else:
            length = float(seg_lengths[idx])
            ratio = 0.0 if length <= 1e-9 else seg_s / length
            q = q_pts[idx] * (1.0 - ratio) + q_pts[idx + 1] * ratio
            self._apply_robot_q(model, pin_model, q)
            rb["seg_idx"] = idx
            rb["seg_s"] = seg_s
            finished = False

        marker = getattr(self, '_path_playback_marker', None)
        if isinstance(marker, dict):
            marker = marker.get(rb["robot_name"])
        if marker is not None:
            tcp_T = self._pin_tcp_world_T(model, pin_model, q, rb["robot_name"])
            if tcp_T is not None:
                marker.pos(tcp_T[:3, 3])
        if render:
            self._show_robot_tcp_axes(render=False)
            self.plotter.render()
        return finished

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
        # ?ㅽ? ?ъ쫰 = chuck 湲곗? ?ㅽ봽??(?ъ??붾꼫媛 ?吏곸뿬??媛믪? 洹몃?濡?
        self._sync_spool_offset_from_world_T()
        x, y, z = getattr(self, '_spool_offset_xyz', (0.0, 0.0, 0.0))
        return {
            "x": float(x), "y": float(y), "z": float(z),
            "x_rotation": float(getattr(self, '_spool_offset_xrot', 0.0)),
            "z_rotation": float(getattr(self, '_spool_offset_zrot', 0.0)),
        }

    def _send_spool_pose_update(self, identity=None):
        if hasattr(self, 'zapi') and self.zapi and identity:
            self.zapi.update_spool_pose(self._get_spool_pose_payload(), identity=identity)

    def _get_positioner_pose_payload(self):
        return {
            "x": float(getattr(self, '_positioner_x', 0.0)),
            "z": float(getattr(self, '_positioner_z', 0.0)),
            "r": float(getattr(self, '_positioner_r_deg', 0.0)),
            "clamp": float(getattr(self, '_positioner_clamp', 0.0)),
        }

    def _spool_alignment_state_path(self, spool_path=None):
        path = spool_path or getattr(self, '_spool_source_path', None)
        if not path:
            return None
        return Path(path).with_suffix(".json")

    def _spool_alignment_state_payload(self):
        fix_f = bool(getattr(self, '_spool_fix_r', False))
        fix_z = bool(getattr(self, '_spool_fix_m_column_z', False))
        source_path = getattr(self, '_spool_source_path', None)
        payload = {
            "version": 2,
            "geometry_file": Path(source_path).name if source_path else None,
            "spool_file": Path(source_path).name if source_path else None,
            "positioner": self._get_positioner_pose_payload(),
            "spool": self._get_spool_pose_payload(),
            "fix_f_column_r": fix_f,
            "fix_m_column_z": fix_z,
            "fixation": {
                "fixed": bool(getattr(self, '_spool_positioner_fixed', False)),
                "fix_f_column_r": fix_f,
                "fix_m_column_z": fix_z,
            },
            "chuck_mount_points": self._get_chuck_mount_points_payload(),
        }
        payload.update(payload["spool"])
        return payload

    def _save_spool_alignment_state(self, spool_path=None, reason=""):
        state_path = self._spool_alignment_state_path(spool_path)
        if state_path is None:
            self.__console.warning("Cannot save spool alignment state: no spool path")
            return False
        try:
            payload = self._spool_alignment_state_payload()
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            suffix = f" ({reason})" if reason else ""
            self.__console.info(f"Saved spool alignment state{suffix}: {state_path}")
            return True
        except Exception as exc:
            self.__console.error(f"Failed to save spool alignment state: {exc}")
            return False

    def _apply_robot_joint_state_payload(self, robots):
        if not isinstance(robots, dict):
            return []
        updated = []
        for model in getattr(self, '_robot_models', []):
            robot_name = getattr(model, 'name', None)
            joints = robots.get(robot_name)
            if not isinstance(joints, dict):
                continue
            joint_map = model._urdf._joint_map if model._urdf else {}
            changed = False
            for joint_name, value in joints.items():
                if joint_name not in joint_map:
                    continue
                try:
                    model.set_joint(joint_name, float(value))
                    changed = True
                except Exception:
                    continue
            if changed:
                model.update_fk()
                updated.append(robot_name)
        return updated

    def _load_spool_alignment_state(self, spool_path=None, identity=None):
        state_path = self._spool_alignment_state_path(spool_path)
        if state_path is None or not state_path.exists():
            if state_path is not None:
                self.__console.info(f"No spool alignment state found: {state_path}")
            return False
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            geometry_file = payload.get("geometry_file") or payload.get("spool_file")
            if geometry_file and spool_path and Path(geometry_file).name != Path(spool_path).name:
                self.__console.warning(
                    f"Spool alignment state geometry mismatch: state={geometry_file}, spool={Path(spool_path).name}")

            positioner = payload.get("positioner")
            if isinstance(positioner, dict):
                self._apply_positioner_pose_values(
                    x=positioner.get("x"),
                    z=positioner.get("z"),
                    r=positioner.get("r"),
                    clamp=positioner.get("clamp"),
                    update_frames=False,
                )

            spool = payload.get("spool", payload)
            has_spool_pose = any(k in spool for k in ("x", "y", "z", "x_rotation", "z_rotation"))
            if has_spool_pose:
                self._spool_offset_xyz = [
                    float(spool.get("x", 0.0)),
                    float(spool.get("y", 0.0)),
                    float(spool.get("z", 0.0)),
                ]
                self._spool_offset_xrot = float(spool.get("x_rotation", 0.0))
                self._spool_offset_zrot = float(spool.get("z_rotation", 0.0))
                self._render_spool_offset()

            fixation = payload.get("fixation", {})
            fix_f = bool(payload.get("fix_f_column_r", fixation.get("fix_f_column_r", False)))
            fix_z = bool(payload.get("fix_m_column_z", fixation.get("fix_m_column_z", False)))
            self._spool_fix_r = fix_f
            self._spool_fix_m_column_z = fix_z
            self._spool_positioner_fixed = bool(fixation.get("fixed", fix_f or fix_z))
            if self._spool_positioner_fixed:
                self._ensure_spool_frame_from_actor()
                self._clear_chuck_profile_visuals(render=False)

            mount_points = payload.get("chuck_mount_points")
            if mount_points:
                self._set_chuck_mount_points(
                    mount_points.get("points", []),
                    mount_points.get("local_points"),
                )

            Tc_now = self._chuck_world_T()
            if Tc_now is not None:
                self._chuck_prev_T = Tc_now
            self._show_chuck_frames(render=False)
            self.plotter.render()

            self._send_positioner_pose_update(identity=identity)
            self._send_spool_pose_update(identity=identity)
            self.__console.info(f"Loaded spool alignment state: {state_path}")
            return True
        except Exception as exc:
            self.__console.error(f"Failed to load spool alignment state: {exc}")
            return False

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
        """?ㅽ? actor瑜????먭뎔?쇰줈 援먯껜 (?꾪꽣 寃곌낵 諛섏쁺)."""
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
        # ?ㅽ봽??紐⑤뜽 ?쇨??? world ?먯쓣 ?꾩옱 chuck@offset 湲곗? local濡??섏궛
        Tc = self._chuck_world_T()
        if Tc is not None and getattr(self, '_spool_local_verts', None) is not None:
            Tinv = np.linalg.inv(Tc @ self._spool_offset_T())
            self._spool_local_verts = (Tinv[:3, :3] @ new_pts.T).T + Tinv[:3, 3]
        self.plotter.render()

    def _filter_loaded_spool(self, request_data):
        """?꾩옱 濡쒕뱶???ㅽ???吏곸젒 ?몄씠利??꾪꽣(SOR/CCL)瑜??곸슜."""
        pts = self._get_spool_points()
        if pts is None:
            self.__console.warning("filter_spool: 濡쒕뱶???ㅽ????놁뒿?덈떎")
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
                    self.__console.warning("filter_spool(ccl): ?곌껐?붿냼 ?놁쓬")
                    return
                uniq, cnts = np.unique(valid, return_counts=True)
                kept = pts[labels == uniq[np.argmax(cnts)]]
            else:
                self.__console.warning(f"filter_spool: ?????녿뒗 method '{method}'")
                return
            self._replace_spool_points(kept)
            self.__console.info(f"filter_spool({method}): {n0} ??{len(kept)} ??(?쒓굅 {n0 - len(kept)})")
        except Exception as e:
            self.__console.error(f"filter_spool ?ㅽ뙣: {e}")

    def _reconstruct_loaded_spool_mesh(self, request_data):
        """?꾩옱 濡쒕뱶???ㅽ? ?먭뎔?쇰줈 硫붿떆 ?ш굔(Marching Cubes) ???쒖떆."""
        pts = self._get_spool_points()
        if pts is None:
            self.__console.warning("reconstruct_mesh: 濡쒕뱶???ㅽ????놁뒿?덈떎")
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
                self.__console.warning("reconstruct_mesh: 鍮?硫붿떆")
                return
            vmesh = vedo.Mesh([verts, faces]).c("gray")

            # 湲곗〈 pcd ?ㅽ? + ?댁쟾 ?ш굔 硫붿떆 ?쒓굅
            old_pcd = getattr(self, '_loaded_spool_mesh', None)
            if old_pcd is not None:
                self.plotter.remove(old_pcd)
            old_recon = getattr(self, '_spool_recon_mesh', None)
            if old_recon is not None and old_recon is not old_pcd:
                self.plotter.remove(old_recon)

            self.plotter.add(vmesh)
            # ?ш굔 硫붿떆瑜????ㅽ?濡??쇱븘 ?ㅽ봽??紐⑤뜽???곌껐 ???ъ??붾꼫 異붿쥌(媛숈씠 ?대룞)
            self._loaded_spool_mesh = vmesh
            self._spool_recon_mesh = vmesh
            Tc = self._chuck_world_T()
            T = (getattr(self, '_spool_world_T', None)
                 if getattr(self, '_spool_world_T', None) is not None
                 else ((Tc @ self._spool_offset_T()) if Tc is not None else np.eye(4)))
            Tinv = np.linalg.inv(T)
            # verts(?붾뱶) ??local 濡??섏궛??world = T @ local ?좎? (?꾩옱 ?꾩튂 蹂댁〈 + 異붿쥌 媛??
            self._spool_local_verts = (Tinv[:3, :3] @ verts.T).T + Tinv[:3, 3]
            self._spool_world_T = T
            if Tc is not None:
                self._chuck_prev_T = Tc
            self.plotter.render()
            self._probe_current_spool_pinocchio_collision("reconstruct_mesh")
            self.__console.info(f"reconstruct_mesh: ?뺤젏 {len(verts)}, 硫?{len(faces)} (pcd ?쒓굅, 硫붿떆媛 ?ㅽ?濡??꾪솚)")
        except Exception as e:
            self.__console.error(f"reconstruct_mesh ?ㅽ뙣: {e}")

    def _save_loaded_spool(self, request_data):
        """?꾩옱 寃곌낵瑜???? ?ш굔 硫붿떆媛 ?덉쑝硫?硫붿떆瑜? ?놁쑝硫??먭뎔?????"""
        path = request_data.get("path")
        if not path:
            return
        try:
            recon = getattr(self, '_spool_recon_mesh', None)
            if recon is not None and hasattr(recon, "vertices") and hasattr(recon, "cells"):
                # ???mesh??spool local frame?쇰줈 湲곕줉?쒕떎. ??JSON??chuck 湲곗? offset??
                # ?ㅼ떆 ?곸슜?섎㈃ load ???숈씪??pose濡??뚯븘媛????덈떎.
                verts = getattr(self, '_spool_local_verts', None)
                if verts is None:
                    verts = np.asarray(recon.vertices)
                m = _o3d.geometry.TriangleMesh()
                m.vertices = _o3d.utility.Vector3dVector(np.asarray(verts, dtype=float))
                m.triangles = _o3d.utility.Vector3iVector(np.asarray(recon.cells, dtype=np.int32))
                m.compute_vertex_normals()
                _o3d.io.write_triangle_mesh(path, m)
                self.__console.info(f"save_spool: local-frame 硫붿떆 ???{path}")
            else:
                pts = self._get_spool_points()
                if pts is None:
                    self.__console.warning("save_spool: ??ν븷 ?ㅽ????놁뒿?덈떎")
                    return
                pcd = _o3d.geometry.PointCloud()
                pcd.points = _o3d.utility.Vector3dVector(pts)
                _o3d.io.write_point_cloud(path, pcd)
                self.__console.info(f"save_spool: ?먭뎔 ???{path} ({len(pts)} ??")
        except Exception as e:
            self.__console.error(f"save_spool ?ㅽ뙣: {e}")

    # --- ?ㅽ? ?꾨젅??怨좎젙(媛뺤껜 遺李? ?좏떥 ---
    F_CHUCK_LINK_NAME = "f_column_passive_clamp"
    M_CHUCK_LINK_NAME = "m_column_passive_r"
    CHUCK_LINK_NAME = M_CHUCK_LINK_NAME

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
        """center瑜?吏?섎뒗 axis ?섎젅濡?deg???뚯쟾?섎뒗 4x4 (?붾뱶)."""
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
        """column m chuck joint(m_column_passive_r) 留곹겕??4x4 ?붾뱶 蹂??"""
        for model in getattr(self, '_robot_models', []):
            if hasattr(model, 'get_link_world_T'):
                T = model.get_link_world_T(self.CHUCK_LINK_NAME)
                if T is not None:
                    return np.asarray(T, dtype=float)
        return None

    def _spool_offset_T(self):
        """UI(=chuck 湲곗?) ?ㅽ봽???ъ쫰瑜?4x4 蹂?섏쑝濡? spool_world = T_chuck @ T_offset @ local"""
        x, y, z = getattr(self, '_spool_offset_xyz', (0.0, 0.0, 0.0))
        xrot = getattr(self, '_spool_offset_xrot', 0.0)
        zrot = getattr(self, '_spool_offset_zrot', 0.0)
        return self._transl([x, y, z]) @ self._rotz(zrot) @ self._rotx(xrot)

    def _sync_spool_offset_from_world_T(self):
        """Update stored chuck-relative spool offset from the current world transform."""
        Tc = self._chuck_world_T()
        T_world = getattr(self, '_spool_world_T', None)
        if Tc is None or T_world is None:
            return False
        try:
            T_offset = np.linalg.inv(Tc) @ np.asarray(T_world, dtype=float)
            Rm = T_offset[:3, :3]
            xrot = float(np.rad2deg(np.arctan2(Rm[2, 1], Rm[2, 2])))
            zrot = float(np.rad2deg(np.arctan2(Rm[1, 0], Rm[0, 0])))
            self._spool_offset_xyz = np.asarray(T_offset[:3, 3], dtype=float).tolist()
            self._spool_offset_xrot = xrot
            self._spool_offset_zrot = zrot
            return True
        except Exception as exc:
            self.__console.warning(f"Failed to sync spool offset from world transform: {exc}")
            return False

    def _apply_spool_world_T(self):
        """?꾩옱 _spool_world_T 濡??ㅽ? actor ?뺤젏 媛깆떊 (world = T @ local)."""
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
        Mesh濡?濡쒕뱶???ㅽ?泥섎읆 local frame???녿뒗 寃쎌슦, ?꾩옱 ?붾㈃ 醫뚰몴瑜?
        ?꾩옱 chuck@offset 湲곗? local frame?쇰줈 ?섏궛???댄썑 fixation ?대룞??媛?ν븯寃??쒕떎.
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
        """?섎룞 諛곗튂: ?꾩옱 chuck 湲곗??쇰줈 ?ㅽ????덈? 諛곗튂 (spool_world = T_chuck @ T_offset)."""
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
                            identity = request_data.get("_identity")
                            self._clear_collision_highlights()
                            import pathlib as _pl
                            mesh, _geom_kind, _mesh_o3d, _pcd = self._load_spool_geometry_with_normals(path)
                            if mesh is not None:
                                # ?ㅽ? ?꾩튂??chuck 議곗씤??m_column_passive_r)瑜??먯젏?쇰줈 蹂몃떎.
                                # spool_world = T_chuck @ T_offset @ local
                                #  - local: ?먭뎔??centroid濡?以묒떖??+ 湲곕낯?뺣젹(chuck 湲곗? ?곸닔)
                                #    ???ъ??붾꼫 ?꾩튂? 臾닿??섍쾶 reload ????긽 ?꾩옱 chuck 湲곗??쇰줈 諛곗튂
                                #  - T_offset: UI(chuck 湲곗?) ?꾩튂/?뚯쟾, 泥섏쓬??0
                                _is_pcd = _pl.Path(path).suffix.lower() == ".pcd"
                                _is_point_cloud = _geom_kind == "point_cloud"
                                _default_x = -0.442  # 泥?湲몄씠留뚰겮 x濡?(chuck 湲곗?)

                                # 湲곗〈??濡쒕뱶???ㅽ?/?ш굔 硫붿떆 紐⑤몢 ?쒓굅 (???뚯씠?꾨줈 援먯껜)
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
                                self._spool_source_path = path

                                if _is_pcd:
                                    _pts = np.asarray(_pcd.points, dtype=np.float64)
                                    _visual_pts = np.asarray(mesh.vertices, dtype=np.float64)
                                    scaled = _pts
                                    visual_scaled = _visual_pts
                                    centroid = scaled.mean(axis=0)
                                    Rz = self._rotz(-90)[:3, :3]
                                    # centroid 以묒떖????-90???뺣젹 ??chuck 湲곗? x ?ㅽ봽???곸닔)
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
                                    # ??λ맂 PLY/mesh??spool local frame(m)?쇰줈 媛꾩＜?쒕떎.
                                    # ??JSON??chuck 湲곗? offset???곸슜?섎㈃ ?????pose濡?蹂듭썝?쒕떎.
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
                                self._load_spool_alignment_state(path, identity=identity)
                                self._probe_current_spool_pinocchio_collision("load_spool")
                                self.__console.info(f"Successfully loaded {path}")
                                
                                # Send reply
                                if hasattr(self, 'zapi') and self.zapi:
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

                    # ?ㅽ????꾩옱 bounding box 以묒떖??mirror origin?쇰줈 ?ъ슜
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
                    # ?ㅽ? ?꾩튂/?뚯쟾??chuck 湲곗? ?ㅽ봽?뗭쑝濡??ㅼ젙 (x,y,z,x_rot,z_rot)
                    spool = getattr(self, '_loaded_spool_mesh', None)
                    if spool is None:
                        self.__console.warning("move_spool: 濡쒕뱶???ㅽ?(PCD)???놁뒿?덈떎")
                        return True
                    new_xyz = [
                        float(request_data.get("x", 0.0)),
                        float(request_data.get("y", 0.0)),
                        float(request_data.get("z", 0.0)),
                    ]
                    new_xrot = float(request_data.get("x_rotation", 0.0))
                    new_zrot = float(request_data.get("z_rotation", 0.0))

                    # ??λ맂 mesh/ply泥섎읆 world 醫뚰몴濡?濡쒕뱶?섏뼱 local frame???녿뒗 寃쎌슦:
                    # ?붿껌??offset 湲곗??쇰줈 local????궛???꾩옱 ?붾㈃ ?꾩튂瑜?蹂댁〈??梨?
                    # ?댄썑 offset/positioner 異붿쥌 紐⑤뜽???몄엯?쒕떎.
                    if getattr(self, '_spool_local_verts', None) is None:
                        pts = self._get_spool_points()
                        Tc = self._chuck_world_T()
                        if pts is None or Tc is None:
                            self.__console.warning("move_spool: ?ㅽ? local frame 珥덇린???ㅽ뙣")
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
                    self._spool_fix_m_column_z = fix_m_column_z
                    self._spool_positioner_fixed = fix_m_column_z or fix_f_column_r
                    if fix_m_column_z or fix_f_column_r:
                        self._ensure_spool_frame_from_actor()
                        self._clear_chuck_profile_visuals(render=False)
                        self._clear_chuck_frame_visuals(render=False)
                    Tc_now = self._chuck_world_T()
                    if Tc_now is not None:
                        self._chuck_prev_T = Tc_now
                    if self._spool_positioner_fixed:
                        self.plotter.render()
                    self._save_spool_alignment_state(reason="fixation")
                    self.__console.info(
                        "Spool-positioner fixation set: "
                        f"fixed={self._spool_positioner_fixed}, "
                        f"fix_f={fix_f_column_r}, fix_z={fix_m_column_z}")
                elif command == "move_positioner":
                    import math
                    axis = request_data.get("axis")
                    position = float(request_data.get("position", 0.0))
                    velocity = float(request_data.get("velocity", 0.0))
                    fix_m_column_z = bool(request_data.get("fix_m_column_z", False))
                    fix_f_column_r = bool(request_data.get("fix_f_column_r", False))
                    self._spool_fix_r = fix_f_column_r
                    self._spool_fix_m_column_z = fix_m_column_z
                    self._spool_positioner_fixed = fix_m_column_z or fix_f_column_r
                    if fix_m_column_z or fix_f_column_r:
                        self._ensure_spool_frame_from_actor()
                    if self._spool_positioner_fixed and axis not in ("r", "z"):
                        self.__console.warning(
                            f"Positioner {axis} move rejected: mount is fixed; only r/z axes can move")
                        self._send_positioner_pose_update(identity=request_data.get("_identity"))
                        return
                    prev_positioner_r = float(getattr(self, '_positioner_r_deg', 0.0))
                    if axis == "x":
                        self._positioner_x = position
                    elif axis == "z":
                        self._positioner_z = position
                    elif axis == "r":
                        self._positioner_r_deg = position
                    elif axis == "clamp":
                        self._positioner_clamp = position

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
                            # prismatic y-axis, range -0.9~0; UI value 0~0.9 ??joint = -position
                            model.set_joint("f_column_r_to_f_column_passive_clamp", -position)
                        else:
                            continue
                        model.update_fk()

                    # ?ㅽ? 異붿쥌? "怨좎젙??異?????댁꽌留? (泥댄겕 ???섎㈃ ???곕씪媛?
                    Tc_now = self._chuck_world_T()
                    has_frame = (getattr(self, '_spool_world_T', None) is not None
                                 and getattr(self, '_spool_local_verts', None) is not None)
                    if has_frame and Tc_now is not None:
                        if axis in ("x", "z") and fix_m_column_z and getattr(self, '_chuck_prev_T', None) is not None:
                            # column m 怨좎젙: chuck 蹂묒쭊?됰쭔???ㅽ? ?됲뻾?대룞
                            dt = Tc_now[:3, 3] - self._chuck_prev_T[:3, 3]
                            T = np.eye(4); T[:3, 3] = dt
                            self._spool_world_T = T @ self._spool_world_T
                            self._apply_spool_world_T()
                            self._update_chuck_mount_points_after_transform(T)
                            self._send_spool_pose_update(identity=request_data.get("_identity"))
                        elif axis == "r" and fix_f_column_r:
                            # column r 怨좎젙: chuck joint 以묒떖쨌異?chuck x異? 湲곗??쇰줈 ?ㅽ? ?뚯쟾
                            delta_r = position - prev_positioner_r
                            m_T = self._chuck_link_world_T(self.M_CHUCK_LINK_NAME)
                            m_cfg = self._chuck_frame_config(self.M_CHUCK_LINK_NAME)
                            r_rotation_sign = float(m_cfg.get("r_rotation_sign", -1.0))
                            if m_T is not None:
                                center = self._chuck_center_world(self.M_CHUCK_LINK_NAME, m_T)
                                axis_w = self._chuck_axis_world(self.M_CHUCK_LINK_NAME, m_T)
                            else:
                                center = Tc_now[:3, 3]
                                axis_w = Tc_now[:3, :3] @ np.array([1.0, 0.0, 0.0])
                            Rm = self._rot_about_axis(axis_w, center, delta_r * r_rotation_sign)
                            self._spool_world_T = Rm @ self._spool_world_T
                            self._apply_spool_world_T()
                            self._update_chuck_mount_points_after_transform(Rm)
                            self._send_spool_pose_update(identity=request_data.get("_identity"))
                    if Tc_now is not None:
                        self._chuck_prev_T = Tc_now
                    if axis == "r":
                        self._positioner_r_deg = position

                    self._show_chuck_frames(render=False)
                    self.plotter.render()
                    if self._spool_positioner_fixed:
                        self._save_spool_alignment_state(reason=f"fixed move {axis}")
                    self.__console.info(f"Positioner {axis} moved to {position} (vel={velocity})")
                elif command == "move_manipulator":
                    self._set_joint_animation(
                        request_data.get("robot"),
                        request_data.get("joint"),
                        request_data.get("target", 0.0),
                        request_data.get("speed", 1.0),
                        request_data.get("accel"),
                        identity=request_data.get("_identity"))
                elif command == "stop_manipulator":
                    self._stop_joint_animation(
                        request_data.get("robot"),
                        request_data.get("joint"))
                elif command == "reset_robot_base_pose":
                    self._reset_robot_base_pose(
                        request_data.get("robot"),
                        identity=request_data.get("_identity"))
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
                        self._clear_ik_failure_visuals(render=False)
                    self.__console.info(
                        "inspection pick mode enabled" if self._inspection_pick_enabled
                        else "inspection pick mode disabled")
                elif command == "pick_chuck_mount_points":
                    enabled = bool(request_data.get("enabled", True))
                    self._chuck_mount_pick_enabled = enabled
                    self._chuck_mount_pick_identity = request_data.get("_identity")
                    self._chuck_mount_align_on_pick = bool(request_data.get("align_on_pick", False))
                    self._chuck_mount_align_target = str(request_data.get("align_target", "f")).lower()
                    if enabled:
                        self._inspection_pick_enabled = False
                        if bool(request_data.get("clear", True)):
                            self._clear_chuck_mount_points()
                    self.__console.info(
                        (f"chuck mount align mode enabled: click {self._chuck_mount_align_target}-column mount point"
                         if self._chuck_mount_align_on_pick
                         else "chuck mount pick mode enabled: click fixed-side point, then moving-side point")
                        if enabled else "chuck mount pick mode disabled")
                elif command == "set_chuck_mount_points":
                    self._set_chuck_mount_points(
                        request_data.get("points", []),
                        request_data.get("local_points"))
                elif command == "set_chuck_mount_config":
                    self._set_chuck_mount_config(request_data.get("chuck_mount", {}))
                elif command == "clear_chuck_mount_points":
                    self._chuck_mount_pick_enabled = False
                    self._clear_chuck_mount_points()
                elif command == "plan_inspection_path":
                    self._plan_inspection_path(request_data)
                elif command == "plan_ef_pose_paths":
                    self._plan_ef_pose_paths(request_data)
                elif command == "check_ef_pose_ik":
                    self._check_ef_pose_ik(request_data)
                elif command == "determine_ef_pose":
                    self._determine_ef_pose(request_data)
                elif command == "clear_inspection_path":
                    self._inspection_pick_enabled = False
                    self._path_playback = None
                    self._robot_path_playback = None
                    self._clear_collision_highlights()
                    self._clear_inspection_visuals(clear_point=True)
                    self._clear_path_playback_marker()
                    self._last_inspection_path = None
                    self._last_inspection_q_path = None
                    self._last_inspection_edge_collisions = []
                    self._last_inspection_robot = None
                    self._last_inspection_plans = {}
                    self._last_inspection_plan_sequence = []
                elif command == "execute_inspection_path":
                    self._start_path_playback(
                        request_data.get("speed", 0.2),
                        identity=request_data.get("_identity"))
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

