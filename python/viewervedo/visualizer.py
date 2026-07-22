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
from viewervedo import geometry_utils as geom_utils
from viewervedo import pipe_alignment_utils
from viewervedo import vedo_visual_utils
from plugins.pluginbase.plannerbase import PlannerBase
from plugins.robotics.backend import RobotDescription
from plugins.robotics.inspection_experiment_logger import InspectionExperimentLogger
from plugins.robotics.inspection_planning_base import InspectionIKRequest, InspectionPlanningBase
from plugins.robotics.pinocchio_backend import PinocchioRoboticsBackend
from plugins.poseDeterminator.EndEffectorPoseOptimizer import EndEffectorPoseOptimizer


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
        self._inspection_ik_experiment_logger = InspectionExperimentLogger(experiment_root)
        self._inspection_ik_experiment_dir = self._inspection_ik_experiment_logger.session_dir
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

        # 매니퓰레이터 조인트 애니메이션(보간 이동) 상태
        # 각 항목: {"model", "joint", "target", "speed"}  speed 단위/프레임당 = unit/s
        self._joint_animations = []
        self._last_anim_time = None
        self._inspection_pick_enabled = False
        self._inspection_pick_identity = None
        self._inspection_point = None
        self._inspection_points = []
        self._inspection_marker = None
        self._inspection_markers = []
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
        self._inspection_target_groups = []
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
        try:
            self.plotter.add_callback("RightButtonPressEvent", self._on_right_mouse_click)
        except Exception:
            try:
                self.plotter.add_callback("right mouse click", self._on_right_mouse_click)
            except Exception:
                pass
        self._show_chuck_frames(render=False)
        self._show_robot_tcp_axes(render=False)


    
    def _process_request(self, request_data):
        """Process a request from the ZApi queue."""
        try:
            if isinstance(request_data, dict):
                command = request_data.get("command")
                handler = self._request_handlers().get(command)
                if handler is None:
                    self.__console.warning(f"Unknown request command: {command}")
                    return None
                return handler(request_data)
            elif isinstance(request_data, (list, tuple)) and len(request_data) >= 2:
                 pass  # Handle raw messages if any
        except Exception as e:
            self.__console.error(f"Error processing request: {e}")

    def _request_handlers(self):
        """Return ZApi command handlers keyed by request command name."""
        return {
            
            "load_spool": self._handle_request_load_spool,                              # 배관 geometry를 로드하고 이전 align 상태를 복원한다.
            "flip_spool_x": self._handle_request_flip_spool_x,                          # 현재 배관 actor를 x축 기준으로 반전한다.
            "move_spool": self._handle_request_move_spool,                              # UI에서 입력된 배관 offset/회전을 적용한다.
            "set_spool_fixation": self._handle_request_set_spool_fixation,              # 배관-포지셔너 고정 상태를 갱신한다.

            "move_positioner": self._handle_request_move_positioner,                    # 포지셔너 조인트를 이동하고 고정 배관을 동기화한다.
            "move_manipulator": self._handle_request_move_manipulator,                  # 협동로봇 단일 조인트 이동 애니메이션을 시작한다.
            "stop_manipulator": self._handle_request_stop_manipulator,                  # 협동로봇 조인트 이동 애니메이션을 중지한다.

            "reset_robot_base_pose": self._handle_request_reset_robot_base_pose,        # 로봇을 설정된 base pose로 초기화한다.
            "filter_spool": self._handle_request_filter_spool,                          # 로드된 배관 점군을 필터링한다.
            "reconstruct_mesh": self._handle_request_reconstruct_mesh,                  # 배관 점군에서 mesh를 재구성한다.
            "save_spool": self._handle_request_save_spool,                              # 현재 배관 geometry와 align 상태를 저장한다.
            "pick_inspection_point": self._handle_request_pick_inspection_point,        # 검사 지점 선택 모드를 켜거나 끈다.
            "pick_chuck_mount_points": self._handle_request_pick_chuck_mount_points,    # chuck mount 기준점 선택/align 모드를 설정한다.
            "set_chuck_mount_points": self._handle_request_set_chuck_mount_points,      # 외부에서 전달된 chuck mount 점을 반영한다.
            "set_chuck_mount_config": self._handle_request_set_chuck_mount_config,      # chuck mount frame/offset 설정을 갱신한다.
            "clear_chuck_mount_points": self._handle_request_clear_chuck_mount_points,  # 선택된 chuck mount 점을 초기화한다.

            "determine_ef_pose": self._handle_request_determine_ef_pose,                # 선택 지점 기준으로 검사 end-effector pose 후보를 계산한다.
            "check_ef_pose_ik": self._handle_request_check_ef_pose_ik,                  # EF pose 후보들의 IK 가능 여부를 검사한다.
            "plan_inspection_path": self._handle_request_plan_inspection_path,          # 검사 pose target group을 순차적으로 경로 계획한다.
            
            "clear_inspection_path": self._handle_request_clear_inspection_path,        # 검사 경로/시각화/충돌 표시를 초기화한다.
            "execute_inspection_path": self._handle_request_execute_inspection_path,    # 계산된 검사 경로 playback을 시작한다.
            "load_test_weld_point": self._handle_request_load_test_weld_point,          # 테스트용 weld point CSV 경로를 처리한다.
        }

    def _handle_request_load_spool(self, request_data):
        """배관 파일을 로드하고 viewer/align/cache 상태를 새 geometry 기준으로 초기화한다."""
        path = request_data.get("path")
        if not path:
            return None
        self.__console.info(f"Loading Spool: {path}")
        identity = request_data.get("_identity")
        try:
            self._clear_collision_highlights()
            import pathlib as _pl
            mesh, _geom_kind, _mesh_o3d, _pcd = self._load_spool_geometry_with_normals(path)
            if mesh is None:
                self.__console.error(f"Failed to load mesh from {path}")
                if hasattr(self, 'zapi') and self.zapi:
                    self.zapi.reply_load_spool(path, False, identity=identity)
                return None

            # spool 위치는 chuck joint(m_column_passive_r)를 원점으로 본다.
            # spool_world = T_chuck @ T_offset @ local
            _is_pcd = _pl.Path(path).suffix.lower() == ".pcd"
            _is_point_cloud = _geom_kind == "point_cloud"
            _default_x = -0.442  # chuck 길이만큼 x 방향 기본 offset

            self._remove_loaded_spool_actors()
            self._reset_loaded_spool_state(path, _pcd, _mesh_o3d)

            if _is_pcd:
                _pts = np.asarray(_pcd.points, dtype=np.float64)
                _visual_pts = np.asarray(mesh.vertices, dtype=np.float64)
                centroid = _pts.mean(axis=0)
                Rz = self._rotz(-90)[:3, :3]
                # centroid 기준으로 -90도 정렬 후 chuck 기준 x offset을 더한다.
                self._spool_full_local_points = (
                    (Rz @ (_pts - centroid).T).T + np.array([_default_x, 0.0, 0.0]))
                self._spool_local_verts = (
                    (Rz @ (_visual_pts - centroid).T).T + np.array([_default_x, 0.0, 0.0]))
            elif _is_point_cloud:
                self._spool_full_local_points = np.asarray(_pcd.points, dtype=float).copy()
                self._spool_local_verts = np.asarray(mesh.vertices, dtype=float).copy()
            else:
                # 저장된 PLY/mesh는 spool local frame(m)으로 간주한다.
                if hasattr(mesh, "vertices"):
                    self._spool_local_verts = np.asarray(mesh.vertices, dtype=float).copy()
                    self._spool_full_local_points = self._spool_local_verts.copy()
                else:
                    self._spool_local_verts = None
                    self._spool_full_local_points = None
                if hasattr(mesh, "cells"):
                    self._spool_recon_mesh = mesh

            self.plotter.add(mesh)
            self._loaded_spool_mesh = mesh
            self._render_spool_offset()
            self.plotter.render()
            self._load_spool_alignment_state(path, identity=identity)
            self._probe_current_spool_pinocchio_collision("load_spool")
            self.__console.info(f"Successfully loaded {path}")
            if hasattr(self, 'zapi') and self.zapi:
                self.zapi.reply_load_spool(path, True, identity=identity)
        except Exception as e:
            self.__console.error(f"Exception loading mesh: {e}")
            if hasattr(self, 'zapi') and self.zapi:
                self.zapi.reply_load_spool(path, False, identity=identity)
        return None

    def _remove_loaded_spool_actors(self):
        """Remove existing spool actors before loading new geometry."""
        _old_sp = getattr(self, '_loaded_spool_mesh', None)
        if _old_sp is not None:
            self.plotter.remove(_old_sp)
            self._loaded_spool_mesh = None
        _old_rc = getattr(self, '_spool_recon_mesh', None)
        if _old_rc is not None:
            if _old_rc is not _old_sp:
                self.plotter.remove(_old_rc)
            self._spool_recon_mesh = None

    def _reset_loaded_spool_state(self, path, pcd, mesh_o3d):
        """Reset spool pose/cache fields for newly loaded geometry."""
        self._spool_offset_xyz = [0.0, 0.0, 0.0]
        self._spool_offset_xrot = 0.0
        self._spool_offset_zrot = 0.0
        self._spool_fix_r = False
        self._positioner_r_deg = 0.0
        self._spool_world_T = None
        self._chuck_prev_T = None
        self._loaded_spool_x_flipped = False
        self._loaded_spool_point_cloud = pcd
        self._loaded_spool_open3d_mesh = mesh_o3d
        self._spool_full_local_points = None
        self._spool_source_path = path

    def _handle_request_flip_spool_x(self, _request_data):
        """로드된 배관 actor를 현재 bounding box 중심 기준 x축 mirror로 반전한다."""
        spool = getattr(self, '_loaded_spool_mesh', None)
        if spool is None or (isinstance(spool, (list, tuple)) and len(spool) == 0):
            self.__console.warning("Cannot flip spool X direction: no spool loaded")
            return True

        actors = spool if isinstance(spool, (list, tuple)) else [spool]
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
        self.__console.info(f"Flipped spool X direction: {self._loaded_spool_x_flipped}")
        return True

    def _handle_request_move_spool(self, request_data):
        """UI에서 전달된 chuck 기준 배관 offset과 회전을 적용한다."""
        spool = getattr(self, '_loaded_spool_mesh', None)
        if spool is None:
            self.__console.warning("move_spool: loaded spool is not available")
            return True
        new_xyz = [
            float(request_data.get("x", 0.0)),
            float(request_data.get("y", 0.0)),
            float(request_data.get("z", 0.0)),
        ]
        new_xrot = float(request_data.get("x_rotation", 0.0))
        new_zrot = float(request_data.get("z_rotation", 0.0))

        # 저장된 mesh/ply처럼 world 좌표로 로드되어 local frame이 없는 경우 현재 화면 위치를 보존한다.
        if getattr(self, '_spool_local_verts', None) is None:
            pts = self._get_spool_points()
            Tc = self._chuck_world_T()
            if pts is None or Tc is None:
                self.__console.warning("move_spool: failed to initialize spool local frame")
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
        return True

    def _handle_request_set_spool_fixation(self, request_data):
        """배관과 포지셔너 mount 사이의 고정 플래그를 갱신하고 현재 chuck frame을 저장한다."""
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

    def _handle_request_move_positioner(self, request_data):
        """포지셔너 조인트를 이동하고 고정 상태이면 배관 pose도 함께 동기화한다."""
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
                # prismatic y-axis, range -0.9~0; UI value 0~0.9 maps to joint = -position
                model.set_joint("f_column_r_to_f_column_passive_clamp", -position)
            else:
                continue
            model.update_fk()

        self._sync_fixed_spool_after_positioner_move(axis, position, prev_positioner_r, request_data)
        self._show_chuck_frames(render=False)
        self.plotter.render()
        if self._spool_positioner_fixed:
            self._save_spool_alignment_state(reason=f"fixed move {axis}")
        self.__console.info(f"Positioner {axis} moved to {position} (vel={velocity})")

    def _sync_fixed_spool_after_positioner_move(self, axis, position, prev_positioner_r, request_data):
        """Move the loaded spool with fixed chuck constraints after positioner motion."""
        Tc_now = self._chuck_world_T()
        has_frame = (getattr(self, '_spool_world_T', None) is not None
                     and getattr(self, '_spool_local_verts', None) is not None)
        if has_frame and Tc_now is not None:
            if axis in ("x", "z") and self._spool_fix_m_column_z and getattr(self, '_chuck_prev_T', None) is not None:
                # m-column 고정: chuck 병진 이동만 spool에 평행 이동으로 반영한다.
                dt = Tc_now[:3, 3] - self._chuck_prev_T[:3, 3]
                T = np.eye(4)
                T[:3, 3] = dt
                self._spool_world_T = T @ self._spool_world_T
                self._apply_spool_world_T()
                self._update_chuck_mount_points_after_transform(T)
                self._send_spool_pose_update(identity=request_data.get("_identity"))
            elif axis == "r" and self._spool_fix_r:
                # r-axis 고정: m chuck 중심과 chuck x축 기준으로 spool을 회전한다.
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

    def _handle_request_move_manipulator(self, request_data):
        """협동로봇 특정 조인트의 목표 이동 애니메이션을 등록한다."""
        self._set_joint_animation(
            request_data.get("robot"),
            request_data.get("joint"),
            request_data.get("target", 0.0),
            request_data.get("speed", 1.0),
            request_data.get("accel"),
            identity=request_data.get("_identity"))

    def _handle_request_stop_manipulator(self, request_data):
        """협동로봇 조인트 이동 애니메이션을 중지한다."""
        self._stop_joint_animation(request_data.get("robot"), request_data.get("joint"))

    def _handle_request_reset_robot_base_pose(self, request_data):
        """선택 로봇 또는 전체 로봇을 설정된 base pose로 되돌린다."""
        self._reset_robot_base_pose(request_data.get("robot"), identity=request_data.get("_identity"))

    def _handle_request_pick_inspection_point(self, request_data):
        """viewer mouse click을 검사 지점 선택으로 해석하도록 pick mode를 전환한다."""
        self._inspection_pick_enabled = bool(request_data.get("enabled", True))
        self._inspection_pick_identity = request_data.get("_identity")
        if bool(request_data.get("clear", False)):
            self._clear_inspection_points(render=False)
        if self._inspection_pick_enabled:
            self._chuck_mount_pick_enabled = False
            self._clear_ik_failure_visuals(render=False)
            self._inspection_pick_multi_select = bool(request_data.get("multi_select", True))
        self.__console.info(
            "inspection pick mode enabled" if self._inspection_pick_enabled
            else "inspection pick mode disabled")

    def _handle_request_pick_chuck_mount_points(self, request_data):
        """viewer mouse click을 chuck mount 기준점 선택 또는 align 입력으로 해석한다."""
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

    def _handle_request_set_chuck_mount_points(self, request_data):
        """외부에서 전달된 chuck mount world/local point를 viewer 상태에 반영한다."""
        self._set_chuck_mount_points(request_data.get("points", []), request_data.get("local_points"))

    def _handle_request_set_chuck_mount_config(self, request_data):
        """UI/config에서 전달된 chuck mount frame offset 설정을 갱신한다."""
        self._set_chuck_mount_config(request_data.get("chuck_mount", {}))

    def _handle_request_clear_chuck_mount_points(self, _request_data):
        """선택된 chuck mount 점과 관련 pick mode를 초기화한다."""
        self._chuck_mount_pick_enabled = False
        self._clear_chuck_mount_points()

    def _handle_request_clear_inspection_path(self, _request_data):
        """검사 경로, playback 상태, 충돌 표시, 검사 지점 시각화를 초기화한다."""
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

    def _handle_request_execute_inspection_path(self, request_data):
        """최근 계산된 검사 경로 playback을 시작한다."""
        self._start_path_playback(
            request_data.get("speed", 0.2),
            identity=request_data.get("_identity"))

    def _handle_request_load_test_weld_point(self, request_data):
        """테스트용 weld point CSV 경로를 받아 로그에 기록한다."""
        path = request_data.get("path")
        if path:
            self.__console.info(f"Loading Test Weld Point from CSV: {path}")
            # CSV 포맷이 확정되면 실제 렌더링 로직을 여기에 붙인다.
            self.__console.info(f"Successfully handled test weld point CSV path: {path}")

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
            self._cache_robot_collision_model(name, full_path)

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
                f"backend={backend.name}, dof={backend.dof(robot_name)}")
            return handle
        except Exception as exc:
            raise RuntimeError(f"failed to register robotics backend model for {robot_name}: {exc}") from exc

    def _cache_robot_collision_model(self, robot_name, urdf_path):
        backend = getattr(self, "_robotics_backend", None)
        if backend is None or not hasattr(backend, "collision_model_cache"):
            return
        cache = getattr(self, "_pinocchio_robot_collision_cache", None)
        if cache is None:
            self._pinocchio_robot_collision_cache = {}
            cache = self._pinocchio_robot_collision_cache
        if robot_name in cache:
            return
        try:
            t0 = time.perf_counter()
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
            geom_count = len(getattr(geom_model, "geometryObjects", []) or [])
            pair_count = len(getattr(geom_model, "collisionPairs", []) or [])
            self.__console.info(
                "Cached robot collision model: "
                f"backend={backend.name}, robot={robot_name}, urdf={urdf_path}, "
                f"geoms={geom_count}, "
                f"pairs={pair_count}, "
                f"elapsed={time.perf_counter() - t0:.3f}s")
        except Exception as exc:
            raise RuntimeError(f"failed to cache robotics collision model for {robot_name}: {exc}") from exc

    def _cache_robot_pinocchio_collision_model(self, robot_name, urdf_path):
        """Backward-compatible alias. Prefer _cache_robot_collision_model()."""
        return self._cache_robot_collision_model(robot_name, urdf_path)

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
        if self._is_right_mouse_event(event):
            self._on_right_mouse_click(event)
            return

        if getattr(self, '_chuck_mount_pick_enabled', False):
            self._handle_chuck_mount_pick(event)
            return

        if not getattr(self, '_inspection_pick_enabled', False):
            return
        pts = self._get_spool_points()
        if pts is None or len(pts) == 0:
            self.__console.warning("inspection pick: loaded pipe point cloud is not available")
            return

        picked = getattr(event, "picked3d", None)
        if picked is None:
            self.__console.warning("inspection pick: no picked pipe surface point")
            return

        picked = np.asarray(picked, dtype=float)
        # PCD/mesh pick 모두에서 실제 pipe point로 스냅할 수 있도록 nearest point를 저장한다.
        idx = int(np.argmin(np.linalg.norm(pts - picked, axis=1)))
        point = np.asarray(pts[idx], dtype=float)
        self._set_inspection_point(point)
        if not bool(getattr(self, "_inspection_pick_multi_select", True)):
            self._inspection_pick_enabled = False

        identity = getattr(self, '_inspection_pick_identity', None)
        if hasattr(self, 'zapi') and self.zapi and identity:
            self.zapi.update_inspection_point({
                "point": point.tolist(),
                "points": [p.tolist() for p in getattr(self, "_inspection_points", [])],
            }, identity=identity)
        self.__console.info(
            f"inspection point picked: {np.round(point, 4)}, "
            f"count={len(getattr(self, '_inspection_points', []) or [])}")

    @staticmethod
    def _is_right_mouse_event(event):
        """vedo/VTK 이벤트 객체에서 우클릭 여부를 가능한 범위에서 판별한다."""
        values = [
            getattr(event, "button", None),
            getattr(event, "name", None),
            getattr(event, "event", None),
            getattr(event, "event_name", None),
            getattr(event, "eventName", None),
        ]
        for value in values:
            if value is None:
                continue
            if isinstance(value, (int, float)) and int(value) == 3:
                return True
            text = str(value).lower()
            if "right" in text or text in {"3", "rightbutton"}:
                return True
        return False

    def _on_right_mouse_click(self, _event=None):
        """우클릭 시 현재 선택 모드를 종료한다. 선택된 포인트는 유지한다."""
        ended = False
        if getattr(self, "_inspection_pick_enabled", False):
            self._inspection_pick_enabled = False
            self._inspection_pick_identity = None
            ended = True
        if getattr(self, "_chuck_mount_pick_enabled", False):
            self._chuck_mount_pick_enabled = False
            self._chuck_mount_pick_identity = None
            ended = True
        if ended:
            self.__console.info(
                "pick mode finished by right click: "
                f"inspection_points={len(getattr(self, '_inspection_points', []) or [])}")

    def _set_inspection_point(self, point):
        point = np.asarray(point, dtype=float)
        self._inspection_point = point
        self._inspection_points = list(getattr(self, "_inspection_points", []) or [])
        self._inspection_points.append(point)
        self._clear_ik_failure_visuals(render=False)
        self._clear_ef_pose_visuals()
        marker = vedo.Sphere(pos=point, r=0.045, c="tomato")
        marker.pickable(False)
        self._inspection_marker = marker
        self._inspection_markers = list(getattr(self, "_inspection_markers", []) or [])
        self._inspection_markers.append(marker)
        self.plotter.add(marker)
        self.plotter.render()

    def _clear_inspection_points(self, render=True):
        """선택된 검사 지점과 marker를 모두 초기화한다."""
        markers = list(getattr(self, "_inspection_markers", []) or [])
        single_marker = getattr(self, "_inspection_marker", None)
        if single_marker is not None and single_marker not in markers:
            markers.append(single_marker)
        for marker in markers:
            try:
                self.plotter.remove(marker)
            except Exception:
                pass
        self._inspection_marker = None
        self._inspection_markers = []
        self._inspection_point = None
        self._inspection_points = []
        if render:
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
        return geom_utils.rotation_between_vectors(source, target)

    @staticmethod
    def _unit_vector(vector):
        return geom_utils.unit_vector(vector)

    def _signed_angle_about_axis(self, source, target, axis):
        return geom_utils.signed_angle_about_axis(source, target, axis)

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

            T_align = pipe_alignment_utils.profile_to_chuck_transform(
                pipe_axis,
                pipe_origin,
                alignment_axis,
                chuck_center,
            )
            R_align = T_align[:3, :3]
            aligned_profile = pipe_alignment_utils.transformed_profile_alignment_summary(
                pipe_axis,
                pipe_origin,
                pipe_radius,
                alignment_axis,
                chuck_center,
                T_align,
            )
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
        self.__console.debug(
            "PipeEndProfileAnalyzer: "
            f"anchor_idx={anchor_idx}, distance_threshold={distance_threshold:.6f}, min_points={min_points}")
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
        return geom_utils.frame_from_primary_and_reference(primary, reference)

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
            T_align = pipe_alignment_utils.profile_to_chuck_transform(
                f_axis,
                source_f,
                f_chuck_axis,
                target_f,
            )
            R_align = T_align[:3, :3]

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
        return geom_utils.transformed_pipe_profile(profile, transform)

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
            actor = vedo_visual_utils.profile_cylinder_actor(
                center,
                axis,
                radius,
                length,
                color=color,
                alpha=alpha,
            )
            if actor is None:
                return
            self._chuck_profile_actors.append(actor)
            self.plotter.add(actor)
        except Exception as exc:
            self.__console.warning(f"Failed to draw profile cylinder: {exc}")

    def _add_profile_fit_points_actor(self, points, color="magenta"):
        try:
            actor = vedo_visual_utils.fit_points_actor(points, color=color, point_size=4)
            if actor is None:
                return
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
            actors = vedo_visual_utils.alignment_reference_actors(
                origin,
                axis,
                axis_len,
                label=label,
                color=color,
                far_point=far_point,
            )
            if not actors:
                raise RuntimeError("__ef_pose_collision_groups_rendered__")
            self._chuck_profile_actors.extend(actors)
            self.plotter.add(*actors)
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
            self._clear_inspection_points(render=False)
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
            self._inspection_target_groups = []

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
        return geom_utils.rpy_matrix(rpy)

    def _pose_to_T(self, pose):
        return geom_utils.pose_to_T(pose)

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
        return geom_utils.T_to_pose(T)

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
        return vedo_visual_utils.pose_frame_actors(
            pose,
            scale=scale,
            axes=axes,
            show_origin=show_origin,
        )

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

    def _show_ef_target_groups(self, target_groups):
        """EF pose target group들을 시각화한다.

        RT 쪽 mesh/frame/connector 색으로 positioner 회전 필요 여부를 표시한다:
        초록(limegreen) = 회전 없이 접근 가능(first), 주황(orangered) = 회전 필요(second).
        판정은 `_inspection_group_is_reachable_now`(RT back-axis world x 부호) 기준이다.
        DDA는 이 판정과 무관하므로 항상 gold로 표시한다.
        """
        self._clear_ef_pose_visuals(clear_poses=False)
        actors = []
        for group_info in list(target_groups or []):
            reachable = self._inspection_group_is_reachable_now(group_info)
            rt_color = "limegreen" if reachable else "orangered"
            show_axes = True
            pair_origins = []
            self.__console.debug(
                f"showing EF pose group: {group_info}, reachable={reachable}")
            for robot_name, pose_name, target_T in self._inspection_group_pose_items(group_info):
                pair_origins.append((pose_name, target_T[:3, 3].copy()))
                color = "gold" if pose_name == "DDA" else rt_color
                actors.extend(self._target_pose_mesh_actors(robot_name, target_T, color=color, alpha=0.3))
                if show_axes:
                    scale = 0.22 if str(pose_name).startswith("RT") else 0.18
                    actors.extend(self._pose_frame_actors(
                        target_T,
                        scale=scale,
                        axes=(0, 1, 2),
                        show_origin=(True),
                    ))
            if len(pair_origins) >= 2:
                try:
                    dda_origin  = next(origin for name, origin in pair_origins if name == "DDA")
                    rt_origin   = next(origin for name, origin in pair_origins if str(name).startswith("RT"))
                    connector   = vedo.Line(dda_origin, rt_origin, c=rt_color, lw=4)
                    connector.alpha(0.75)
                    connector.pickable(False)
                    actors.append(connector)
                except Exception:
                    pass
        self._ef_pose_actors = actors
        self.__console.info(f"showing {len(actors)} EF pose actors for {len(target_groups)} target groups")
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

    def _handle_request_determine_ef_pose(self, request_data):
        """선택된 검사 지점 여러 개를 순회해 EF pose target group 목록을 만든다."""
        identity = request_data.get("_identity")
        result = {"status": "failed"}
        total_t0 = time.perf_counter()
        try:
            self._clear_ik_failure_visuals(render=False)
            inspection_points = [
                np.asarray(point, dtype=float)
                for point in (getattr(self, "_inspection_points", []) or [])
            ]
            if not inspection_points and getattr(self, "_inspection_point", None) is not None:
                inspection_points = [np.asarray(self._inspection_point, dtype=float)]
            if not inspection_points:
                raise RuntimeError("inspection point is not selected")

            pose_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "plugins", "poseDeterminator")
            )
            if pose_dir not in sys.path:
                sys.path.insert(0, pose_dir)

            params = self._config.get("ef_pose", {}) or {}
            optimizer_logging = params.get("logging", {}) or {}
            optimizer = EndEffectorPoseOptimizer(
                debug_mode=bool(params.get("debug_mode", True)),
                log_path=optimizer_logging.get("log_path"),
                log_dir=optimizer_logging.get("log_dir"),
                log_level=optimizer_logging.get("level", "DEBUG"),
                console_level=optimizer_logging.get("console_level"),
                file_level=optimizer_logging.get("file_level"),
                logger_name=optimizer_logging.get("name", "flame_robotics"),
                force_logger_config=optimizer_logging.get("force"),
            )
            stage_t0 = time.perf_counter()
            optimizer._scan_data = self._pose_determinator_point_cloud()
            pcd_elapsed = time.perf_counter() - stage_t0

            stage_t0 = time.perf_counter()
            frame_cfg = params.get("frames", {}) or {}
            dda_frame_cfg = frame_cfg.get("dda", {}) or {}
            rt_frame_cfg = frame_cfg.get("rt", {}) or {}
            dda_end_link = str(dda_frame_cfg.get("end_link", "dda_link_end"))
            dda_tcp_joint = str(dda_frame_cfg.get("tcp_joint", "dda_joint_tcp"))
            rt_end_link = str(rt_frame_cfg.get("end_link", "rt_link_end"))
            rt_tcp_joint = str(rt_frame_cfg.get("tcp_joint", "rt_joint_end"))
            dda_pipe_facing_axis = np.asarray(
                dda_frame_cfg.get("pipe_facing_axis", [1.0, 0.0, 0.0]), dtype=float)
            dda_pipe_parallel_axis = dda_frame_cfg.get("pipe_parallel_axis")
            optimizer.set_dda_pipe_facing_axis(
                dda_pipe_facing_axis,
                None if dda_pipe_parallel_axis is None else np.asarray(dda_pipe_parallel_axis, dtype=float),
            )
            rt_pipe_facing_axis = np.asarray(
                rt_frame_cfg.get("pipe_facing_axis", [0.0, -1.0, 0.0]), dtype=float)
            optimizer.set_rt_pipe_facing_axis(rt_pipe_facing_axis)

            dda_pose_to_link = self._ef_pose_offset_T(dda_frame_cfg)
            rt_pose_to_link = self._ef_pose_offset_T(rt_frame_cfg)
            backend = getattr(self, "_robotics_backend", None)
            if backend is None:
                raise RuntimeError("robotics backend is not initialized")
            
            dda_mesh, dda_tcp_to_link = backend.end_effector_collision_geometry(
                "dda_rb10_1300e", dda_end_link, dda_tcp_joint, pose_to_link_offset=dda_pose_to_link)
            rt_mesh, rt_tcp_to_link = backend.end_effector_collision_geometry(
                "rb20_1900es", rt_end_link, rt_tcp_joint, pose_to_link_offset=rt_pose_to_link)
            
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
                f"pipe_facing_axis={np.round(dda_pipe_facing_axis, 5).tolist()}, "
                f"pipe_parallel_axis={None if dda_pipe_parallel_axis is None else np.round(np.asarray(dda_pipe_parallel_axis, dtype=float), 5).tolist()}), "
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

            all_target_groups = []
            target_failures = []
            profile_elapsed = 0.0
            pose_elapsed = 0.0

            for point_index, target in enumerate(inspection_points):
                try:
                    optimizer.debuging_info = {}
                    stage_t0 = time.perf_counter()
                    optimizer.calculate_pipe_profile(
                        target,
                        sampling_size_for_calculating_normal=float(
                            params.get("sampling_size_for_calculating_normal", 0.01)),
                        radius_offset_for_sampling_points_in_sphere=float(
                            params.get("radius_offset_for_sampling_points_in_sphere", 0.003)),
                    )
                    profile_dt = time.perf_counter() - stage_t0
                    profile_elapsed += profile_dt

                    stage_t0 = time.perf_counter()
                    target_groups = optimizer.calculate_DDA_RT_pose_for_taking_xray(
                        target,
                        num_candidates=int(params.get("num_candidates", 9)),
                        distance_from_dda_to_surface=float(params.get("distance_from_dda_to_surface", 0.01)),
                        distance_from_dda_to_rt=float(params.get("distance_from_dda_to_rt", 0.3)),
                        angle_of_rt=float(params.get("angle_of_rt", 10.0)),
                        rt_pipe_facing_axis=rt_pipe_facing_axis,
                        pose_name_to_robot_name=self._ef_pose_robot_name,
                        force_90_fallback=bool(params.get("force_90_fallback", False)),
                    )
                    target_groups = list(target_groups or [])
                    pose_dt = time.perf_counter() - stage_t0
                    pose_elapsed += pose_dt

                    debug_info = getattr(optimizer, "debuging_info", {}) or {}
                    base_candidates = debug_info.get("dda_base_candidates")
                    valid_base_candidates = debug_info.get("valid_base_dda_poses")
                    base_count = "n/a" if base_candidates is None else len(base_candidates)
                    valid_base_count = (
                        "n/a" if valid_base_candidates is None else len(valid_base_candidates)
                    )
                    self.__console.info(
                        "EF pose candidate summary: "
                        f"point={point_index + 1}/{len(inspection_points)}, "
                        f"target_point={np.round(target, 4).tolist()}, "
                        f"base={base_count}, "
                        f"valid_base_dda_poses={valid_base_count}, "
                        f"target_groups={len(target_groups or [])}, "
                        f"strategy={debug_info.get('selected_pose_pair_strategy', 'n/a')}, "
                        f"angle_of_rt={debug_info.get('rt_angle_of_rt_input_deg', 'n/a')}, "
                        f"complete_groups={debug_info.get('complete_pose_group_count', 'n/a')}, "
                        f"partial_groups={debug_info.get('partial_pose_group_count', 'n/a')}, "
                        f"rejected_groups={debug_info.get('rejected_pose_group_count', 'n/a')}, "
                        f"used_partial_fallback={bool(debug_info.get('used_partial_pose_group_fallback', False))}, "
                        f"elapsed=profile {profile_dt * 1000:.1f}ms + pose {pose_dt * 1000:.1f}ms")

                    # rt60_items = []
                    # for group in list(target_groups or []):
                    #     priority = group.get("priority", {}) or {}
                    #     rt_angle = priority.get("preferred_rt_angle_deg")
                    #     if rt_angle is None:
                    #         continue
                    #     rt_angle    = float(rt_angle)
                    #     # 기준 각도는 group마다 다르다(3쌍=±60, 2쌍=±45).
                    #     direct_ref  = priority.get("direct_rt_reference_deg")
                    #     ref_deg     = abs(float(direct_ref)) if direct_ref is not None else 60.0
                    #     plus_dev    = abs((rt_angle - ref_deg + 180.0) % 360.0 - 180.0)
                    #     minus_dev   = abs((rt_angle + ref_deg + 180.0) % 360.0 - 180.0)
                    #     nearest     = ref_deg if plus_dev <= minus_dev else -ref_deg
                    #     deviation   = min(plus_dev, minus_dev)
                    #     rt60_items.append({
                    #         "name": group.get("name"),
                    #         "slot": group.get("slot_name"),
                    #         "rt": group.get("rt_name"),
                    #         "angle": round(rt_angle, 3),
                    #         "nearest": nearest,
                    #         "deviation": round(float(deviation), 3),
                    #         "requires_positioner_rotation": bool(
                    #             priority.get("requires_positioner_rotation", False)),
                    #         "direct_reference": priority.get("direct_rt_reference_deg"),
                    #         "direct_deviation": priority.get("direct_rt_deviation_deg"),
                    #     })


                    # if not target_groups:
                    #     target_failures.append({
                    #         "point_index": point_index,
                    #         "point": target.tolist(),
                    #         "message": "poseDeterminator returned no target group",
                    #         "complete_group_count": debug_info.get("complete_pose_group_count"),
                    #         "partial_group_count": debug_info.get("partial_pose_group_count"),
                    #         "rejected_group_count": debug_info.get("rejected_pose_group_count"),
                    #     })
                    #     self.__console.warning(
                    #         "EF pose target group missing for selected point: "
                    #         f"point={point_index + 1}, "
                    #         f"complete={debug_info.get('complete_pose_group_count', 'n/a')}, "
                    #         f"partial={debug_info.get('partial_pose_group_count', 'n/a')}, "
                    #         f"rejected={debug_info.get('rejected_pose_group_count', 'n/a')}")
                    #     continue

                    all_target_groups.extend(list(target_groups or []))

                except Exception as point_exc:
                    target_failures.append({
                        "point_index": point_index,
                        "point": target.tolist(),
                        "message": str(point_exc),
                    })
                    self.__console.error(
                        f"EF pose failed for selected point {point_index + 1}: {point_exc}")

            if not all_target_groups:
                raise RuntimeError(
                    f"poseDeterminator returned no valid target group for "
                    f"{len(inspection_points)} selected point(s): {target_failures}")

            self._ef_pose_groups = []
            self._inspection_target_groups = all_target_groups
            self._show_ef_target_groups(all_target_groups)

            result = {
                "status": "success",
                "target_groups": all_target_groups,
                "inspection_point_count": len(inspection_points),
                "target_group_count": len(all_target_groups),
                "target_failures": target_failures,
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
                f"points={len(inspection_points)}, target_groups={len(all_target_groups)}, "
                f"elapsed={result['elapsed']:.3f}s "
                f"(pcd={pcd_elapsed:.3f}s, urdf={urdf_elapsed:.3f}s, "
                f"profile={profile_elapsed:.3f}s, pose={pose_elapsed:.3f}s)")
        except Exception as exc:
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
        q_space_planners = {"rrt_connect", "rrt_star", "direct_path"}
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
        # planner 하위 속성을 직접 건드리지 않고 추상 클래스의 configure()만 호출한다.
        backend = None
        if robot_name is not None:
            backend = getattr(self, "_robotics_backend", None)
            if backend is None:
                raise RuntimeError("robotics backend is not initialized")
        planner.configure(
            bounds=bounds,
            step_size=float(step_size),
            max_iter=int(max_iter),
            robotics_backend=backend,
            robotics_robot_name=robot_name,
        )
        if timings is not None:
            timings["planner_bounds_config"] = time.perf_counter() - setup_t0
        collision_obstacle_mesh = obstacle_mesh
        if robot_name is not None:
            model = self._find_robot(robot_name)
            urdf_path = getattr(model, "urdf_path", None) if model is not None else None
            if urdf_path:
                try:
                    urdf_t0 = time.perf_counter()
                    robot_backend_model = backend.robot_model(robot_name)
                    planner.configure(robot_model=robot_backend_model)
                    if timings is not None:
                        timings["planner_robotics_model"] = time.perf_counter() - urdf_t0
                    self._log_robot_backend_model(robot_name, urdf_path, robot_backend_model)
                except Exception as exc:
                    raise RuntimeError(f"inspection path robotics model setup failed: {exc}") from exc
            if getattr(planner, "_has_robot_q_space_model", lambda: False)() and model is not None:
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
        self._log_robot_collision_targets(robot_name, planner)
        return bounds

    def _log_robot_collision_targets(self, robot_name, planner):
        if planner is None or not hasattr(planner, "pinocchio_collision_geometry_summary"):
            return
        logged = getattr(self, "_logged_robot_collision_targets", set())
        geom_model = getattr(planner, "pin_geom_model", None)
        static_ids = tuple(getattr(planner, "_pin_static_object_ids", []) or [])
        backend_name = getattr(getattr(planner, "robotics_backend", None), "name", "pinocchio")
        geometries = planner.pinocchio_collision_geometry_summary()
        pairs_all = planner.pinocchio_collision_pair_summary(include_robot_self=True, include_static=True)
        key = (backend_name, robot_name, len(geometries), len(pairs_all), static_ids)
        if key in logged:
            return
        logged.add(key)
        self._logged_robot_collision_targets = logged
        robot_geometries = [item for item in geometries if item.get("kind") == "robot"]
        static_geometries = [item for item in geometries if item.get("kind") == "static"]
        robot_self_pairs = planner.pinocchio_collision_pair_summary(include_robot_self=True, include_static=False)
        static_pairs = planner.pinocchio_collision_pair_summary(include_robot_self=False, include_static=True)
        positioner_checked = self._planner_has_positioner_collision(planner)
        self.__console.info(
            "robot collision targets: "
            f"backend={backend_name}, robot={robot_name}, robot_geoms={len(robot_geometries)}, "
            f"static_geoms={len(static_geometries)}, "
            f"robot_self_pairs={len(robot_self_pairs)}, robot_static_pairs={len(static_pairs)}, "
            f"positioner_collision_checked={positioner_checked}")
        if not positioner_checked:
            self.__console.debug(
                "robot collision targets: positioner URDF is not part of this planner collision model; "
                "positioner collision is skipped for this path check.")

    def _log_pinocchio_collision_targets(self, robot_name, planner):
        """Backward-compatible alias. Prefer _log_robot_collision_targets()."""
        return self._log_robot_collision_targets(robot_name, planner)

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

    def _log_robot_backend_model(self, robot_name, urdf_path, robot_backend_model):
        logged = getattr(self, "_logged_robot_backend_models", set())
        try:
            urdf_mtime_ns = os.stat(urdf_path).st_mtime_ns
        except OSError:
            urdf_mtime_ns = None
        key = (robot_name, urdf_path, urdf_mtime_ns)
        if key in logged or robot_backend_model is None:
            return
        logged.add(key)
        self._logged_robot_backend_models = logged
        backend = getattr(self, "_robotics_backend", None)
        backend_name = getattr(backend, "name", "unknown")
        joint_names = self._robot_joint_names(robot_name, robot_backend_model)
        track_joints = [name for name in joint_names if "linear_track" in name or "carriage" in name]
        try:
            lo, hi, _ = backend.joint_limits_for_metric(robot_name, normalize=True)
        except Exception:
            lo = np.asarray(getattr(robot_backend_model, "lowerPositionLimit", []), dtype=float)
            hi = np.asarray(getattr(robot_backend_model, "upperPositionLimit", []), dtype=float)
        lo = np.asarray([] if lo is None else lo, dtype=float)
        hi = np.asarray([] if hi is None else hi, dtype=float)
        joint_limits = {
            name: [float(lo[i]), float(hi[i])]
            for i, name in enumerate(joint_names[:min(len(lo), len(hi))])
        }
        track_joint_placements = {}
        for name in track_joints:
            try:
                joint_id = int(robot_backend_model.getJointId(name))
                placement = robot_backend_model.jointPlacements[joint_id]
                track_joint_placements[name] = {
                    "parent_to_joint_translation": np.asarray(placement.translation, dtype=float).tolist(),
                    "parent_to_joint_rotation": np.asarray(placement.rotation, dtype=float).round(6).tolist(),
                }
            except Exception:
                continue
        self.__console.info(
            f"robot backend model for {robot_name}: backend={backend_name}, urdf={urdf_path}, "
            f"mtime_ns={urdf_mtime_ns}, "
            f"dof={self._robot_dof(robot_name, robot_backend_model)}, joints={joint_names}, track_joints={track_joints}, "
            f"limits={joint_limits}, track_joint_placements={track_joint_placements}")

    def _log_pinocchio_robot_model(self, robot_name, urdf_path, pin_model):
        """Backward-compatible alias. Prefer _log_robot_backend_model()."""
        return self._log_robot_backend_model(robot_name, urdf_path, pin_model)

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

    def _robot_joint_names(self, robot_name, robot_backend_model=None):
        backend = getattr(self, "_robotics_backend", None)
        if backend is not None and robot_name:
            try:
                return [str(name) for name in backend.joint_names(robot_name)]
            except Exception:
                pass
        if robot_backend_model is not None and hasattr(robot_backend_model, "names"):
            nq = int(getattr(robot_backend_model, "nq", len(list(robot_backend_model.names)) - 1))
            return [str(name) for name in list(robot_backend_model.names)[1:1 + nq]]
        return []

    def _robot_dof(self, robot_name, robot_backend_model=None):
        backend = getattr(self, "_robotics_backend", None)
        if backend is not None and robot_name:
            try:
                return int(backend.dof(robot_name))
            except Exception:
                pass
        if robot_backend_model is not None and hasattr(robot_backend_model, "nq"):
            return int(robot_backend_model.nq)
        names = self._robot_joint_names(robot_name, robot_backend_model)
        return len(names)

    def _pin_joint_names(self, pin_model):
        return self._robot_joint_names(None, pin_model)

    def _current_robot_q(self, model, pin_model=None, robot_name=None):
        robot_name = robot_name or getattr(model, "name", None)
        dof = self._robot_dof(robot_name, pin_model)
        q = np.zeros(dof, dtype=float)
        for i, joint_name in enumerate(self._robot_joint_names(robot_name, pin_model)):
            if i >= q.size:
                break
            q[i] = float(model._joint_cfg.get(joint_name, 0.0))
        return q

    def _apply_robot_q(self, model, pin_model, q, robot_name=None):
        robot_name = robot_name or getattr(model, "name", None)
        for i, joint_name in enumerate(self._robot_joint_names(robot_name, pin_model)):
            if i >= len(q):
                break
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
            return backend.frame_world_T(
                robot_name,
                q,
                self._robot_target_link_name(robot_name),
            )
        except Exception:
            pass
        if pin is None or pin_model is None:
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
            joint_names = self._robot_joint_names(robot_name, pin_model)
            target_T = self._inspection_target_world_T(
                robot_model,
                pin_model,
                robot_name,
                target_pose,
                np.asarray(goal_q, dtype=float),
            )
            logger = getattr(self, "_inspection_ik_experiment_logger", None)
            if logger is None:
                logger = InspectionExperimentLogger(
                    Path(self._config.get("experiment_dir", "experiment")) / "inspection_ik"
                )
                self._inspection_ik_experiment_logger = logger
                self._inspection_ik_experiment_dir = logger.session_dir
            saved = logger.save(
                robot_name=robot_name,
                urdf_path=getattr(robot_model, "urdf_path", ""),
                base_pose=getattr(robot_model, "base_pose", [0, 0, 0, 0, 0, 0]),
                joint_names=joint_names,
                target_link_name=self._robot_target_link_name(robot_name),
                target_T=target_T,
                goal_q=goal_q,
                trace=trace,
                ik_result=ik_result or {},
            )
            if not saved:
                return None
            self.__console.info(
                f"inspection IK experiment saved: robot={robot_name}, csv={saved['csv']}, meta={saved['meta']}")
            return saved
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
        robot_backend_model = backend.robot_model(robot_name)
        robot_dof = self._robot_dof(robot_name, robot_backend_model)
        timings["robotics_model_lookup"] = time.perf_counter() - stage_t0

        stage_t0 = time.perf_counter()
        start_q = np.zeros(robot_dof, dtype=float)
        start_overrides = request_data.get("_start_q_override_by_robot") or {}
        if robot_name in start_overrides:
            try:
                start_q = np.asarray(start_overrides[robot_name], dtype=float)
                if start_q.shape[0] != robot_dof:
                    raise ValueError(f"expected dof={robot_dof}, got {start_q.shape[0]}")
            except Exception as exc:
                self.__console.warning(
                    f"inspection IK check start_q override ignored: robot={robot_name}, error={exc}")
                start_q = np.zeros(robot_dof, dtype=float)
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
                joint_names=self._robot_joint_names(robot_name, robot_backend_model),
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
            robot_backend_model,
            target_pose,
            goal_q,
            ik_result=ik_result,
        )
        result["ik_experiment"] = ik_experiment
        timings["total"] = time.perf_counter() - total_t0
        result["timing"] = timings
        return result

    def _partition_and_sort_inspection_groups(self, target_groups):
        """target group을 first/second로 나누고 각각 RT 위치 x오름차순·z내림차순으로 정렬한다."""
        first_groups, second_groups = [], []
        for group_info in list(target_groups or []):
            reachable = self._inspection_group_is_reachable_now(group_info)
            self.__console.info(
                f"inspection group reachability: {group_info.get('name')} -> reachable={reachable}")
            if reachable:
                first_groups.append(group_info)
            else:
                second_groups.append(group_info)

        def sort_key(group_info):
            rt_pos = self._inspection_group_rt_position(group_info)
            return (float(rt_pos[0]), -float(rt_pos[2]))  # x 우선 오름차순, z 내림차순

        first_groups.sort(key=sort_key)
        second_groups.sort(key=sort_key)
        return first_groups, second_groups

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
        # 1) 현재 배관을 collision obstacle mesh로 준비한다.
        if obstacle_mesh is None:
            obstacle_mesh = self._current_spool_collision_mesh()
        if obstacle_mesh is None:
            raise RuntimeError("loaded pipe is not available")
        timings["obstacle_mesh"] = time.perf_counter() - stage_t0

        # 2) planner 입력인 시작 TCP pose와 목표 pose를 world 좌표계 기준으로 정리한다.
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

        # 3) 요청된 q-space planner를 만들고 robotics backend collision scene을 설정한다.
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
        if not getattr(planner, "_has_robot_q_space_model", lambda: False)():
            raise RuntimeError("robot q-space model is not configured")
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
        robot_backend_model = getattr(planner, "robot_model", None) or getattr(planner, "pin_model", None)
        robot_dof = self._robot_dof(robot_name, robot_backend_model)
        start_q = self._current_robot_q(robot_model, robot_backend_model, robot_name=robot_name)
        start_overrides = request_data.get("_start_q_override_by_robot") or {}
        if robot_name in start_overrides:
            try:
                start_q = np.asarray(start_overrides[robot_name], dtype=float)
                if start_q.shape[0] != robot_dof:
                    raise ValueError(f"expected dof={robot_dof}, got {start_q.shape[0]}")
            except Exception as exc:
                self.__console.warning(
                    f"inspection path start_q override ignored: robot={robot_name}, error={exc}")
                start_q = self._current_robot_q(robot_model, robot_backend_model, robot_name=robot_name)
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
                joint_names=self._robot_joint_names(robot_name, robot_backend_model),
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
            robot_backend_model,
            target_pose,
            goal_q,
            ik_result=ik_result,
        )

        # 5) q path를 viewer 표시용 TCP waypoint로 변환한다.
        stage_t0 = time.perf_counter()
        display_resolution = float(request_data.get("display_step_size", request_data.get("step_size", 0.08)))
        path = self._q_path_to_tcp_poses(
            robot_model,
            robot_backend_model,
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

    # def _inspection_target_groups_for_planning(self, request_data):
    #     """경로 계획에 사용할 target group 목록을 반환한다.

    #     입력:
    #         request_data(dict):
    #             - command: "plan_inspection_path".
    #             - target_groups: 선택 사항. 이미 구성된 target group list를 직접 넘길 때 사용한다.
    #             - use_ef_pose_targets: True이면 저장된 검사 target group을 사용한다.
    #             - robot: 수동 검사점 계획에 사용할 로봇 이름. 기본값은 "rb20_1900es".
    #             - pose_name: 수동 검사점 target 이름. 기본값은 "manual".

    #     출력:
    #         list[dict]:
    #             [
    #                 {
    #                     "index": int,
    #                     "name": str,
    #                     "targets": {
    #                         robot_name: {
    #                             "pose_name": str,
    #                             "target_T": np.ndarray,  # 3D point 또는 4x4 pose
    #                             "inspection_pose_name": str,
    #                         }
    #                     },
    #                 }
    #             ]
    #     """
    #     if isinstance(request_data.get("target_groups"), list):
    #         return request_data["target_groups"]

    #     if bool(request_data.get("use_ef_pose_targets", False)):
    #         target_groups = self._inspection_target_groups
    #         if not target_groups:
    #             raise RuntimeError("EF poses are not determined")
    #         return target_groups

    #     inspection_points = [
    #         np.asarray(point, dtype=float)
    #         for point in (getattr(self, "_inspection_points", []) or [])
    #     ]
    #     if not inspection_points and getattr(self, "_inspection_point", None) is not None:
    #         inspection_points = [np.asarray(self._inspection_point, dtype=float)]
    #     if not inspection_points:
    #         raise RuntimeError("inspection point is not selected")
    #     robot_name = request_data.get("robot", "rb20_1900es")
    #     pose_name = request_data.get("pose_name", "manual")
    #     target_groups = []
    #     for index, target in enumerate(inspection_points):
    #         group_name = f"Inspection pose {index + 1}"
    #         target_groups.append({
    #             "index": index,
    #             "name": group_name,
    #             "source_point_index": index,
    #             "source_point": target.tolist(),
    #             "targets": {
    #                 robot_name: {
    #                     "pose_name": pose_name if len(inspection_points) == 1 else f"{pose_name}_{index + 1}",
    #                     "target_T": target,
    #                     "inspection_pose_name": group_name,
    #                     "source_point_index": index,
    #                     "source_point": target.tolist(),
    #                 }
    #             },
    #         })
    #     return target_groups

    def _inspection_group_pose_items(self, group_info):
        """단순화된 target group에서 (robot_name, pose_name, target_T) 목록을 만든다.

        target group 구조: {name, index, target_point, dda_pose, rt_pose}.
        positioner 회전 필요 여부는 여기서 판단하지 않고 base planner가 rt_pose로 직접 판단한다.
        로봇 이름은 pose_name으로 매핑한다(DDA -> dda 로봇, RT -> rt 로봇).
        """
        items = []
        dda_pose = group_info.get("dda_pose")
        if dda_pose is not None:
            items.append((self._ef_pose_robot_name("DDA"), "DDA", np.asarray(dda_pose, dtype=float)))
        rt_pose = group_info.get("rt_pose")
        if rt_pose is not None:
            items.append((self._ef_pose_robot_name("RT"), "RT", np.asarray(rt_pose, dtype=float)))
        if not items:
            self.__console.warning(
                "inspection group has no dda_pose/rt_pose: "
                f"keys={list(group_info.keys())}, name={group_info.get('name')}")
        return items

    def _rt_pipe_facing_axis_config(self):
        """설정 파일에 정의된 RT의 pipe-facing local 축을 반환한다."""
        frame_cfg = (self._config.get("ef_pose", {}) or {}).get("frames", {}) or {}
        rt_frame_cfg = frame_cfg.get("rt", {}) or {}
        return np.asarray(rt_frame_cfg.get("pipe_facing_axis", [0.0, -1.0, 0.0]), dtype=float)

    def _inspection_group_is_reachable_now(self, group_info):
        """group이 positioner 회전 없이 지금 바로 접근 가능한지 여부.

        RT source가 배관을 바라보는 방향의 반대(=상위 링크와 연결되는 방향, back-axis)를
        world로 변환해 x,y 평면에 투영했을 때 x가 음수이면 회전 없이 접근 가능(first),
        아니면 positioner를 돌려야 한다(second).
        """
        rt_pose = group_info.get("rt_pose")
        if rt_pose is None:
            return False
        rt_T = np.asarray(rt_pose, dtype=float)
        back_axis_local = -self._rt_pipe_facing_axis_config()
        back_axis_world_y = float((rt_T[:3, :3] @ back_axis_local)[1])
        return back_axis_world_y < 0.0

    def _inspection_group_rt_position(self, group_info):
        """정렬 기준으로 쓸 RT endeffector target 위치(world)를 반환한다."""
        rt_pose = group_info.get("rt_pose")
        if rt_pose is not None:
            return np.asarray(rt_pose, dtype=float)[:3, 3]
        return np.zeros(3, dtype=float)



    def _positioner_r_rotation_transform(self, delta_r_deg):
        """포지셔너 r축(=m-chuck 축) 기준 delta_r_deg 회전 world transform(4x4)을 만든다.

        실제 포지셔너/spool을 움직이지 않고 second group 계획용 가상 변환으로만 쓴다.
        회전 축/중심/부호는 실제 r-axis 이동(_sync_fixed_spool_after_positioner_move)과 동일하게 맞춘다.
        """
        m_T = self._chuck_link_world_T(self.M_CHUCK_LINK_NAME)
        m_cfg = self._chuck_frame_config(self.M_CHUCK_LINK_NAME)
        r_rotation_sign = float(m_cfg.get("r_rotation_sign", -1.0))
        if m_T is not None:
            center = self._chuck_center_world(self.M_CHUCK_LINK_NAME, m_T)
            axis_w = self._chuck_axis_world(self.M_CHUCK_LINK_NAME, m_T)
        else:
            Tc = self._chuck_world_T()
            if Tc is None:
                raise RuntimeError("positioner chuck transform is not available")
            center = Tc[:3, 3]
            axis_w = Tc[:3, :3] @ np.array([1.0, 0.0, 0.0])
        return self._rot_about_axis(axis_w, center, float(delta_r_deg) * r_rotation_sign)

    @staticmethod
    def _transform_target_pose(target_T, transform):
        """target pose(4x4)에 world 변환을 적용한다. transform이 None이면 원본을 그대로 쓴다."""
        target_T = np.asarray(target_T, dtype=float)
        if transform is None:
            return target_T
        return np.asarray(transform, dtype=float) @ target_T

    def _plan_inspection_group_sequence(
        self,
        groups,
        obstacle_mesh,
        request_data,
        *,
        start_q_overrides,
        failures,
        ik_failures,
        planning_timeout,
        future_timeout,
        group_offset=0,
        pose_transform=None,
    ):
        """group 목록을 순차 계획한다. 이전 group의 마지막 q를 다음 group start q로 넘긴다.

        pose_transform이 주어지면(예: 포지셔너 가상 회전) 각 target pose에 적용한 뒤 계획한다.
        start_q_overrides/failures/ik_failures는 in-place로 갱신하며, group_sequence 항목 list를 반환한다.
        """
        group_sequence = []
        for sequence_index, group_info in enumerate(groups):
            seq_i = group_offset + sequence_index
            group_index = int(group_info.get("index", seq_i))
            group_name = str(group_info.get("name", f"Inspection pose {seq_i + 1}"))
            pose_items = self._inspection_group_pose_items(group_info)
            plans = {}
            group_failures = {}
            group_ik_failures = {}
            group_request = dict(request_data)
            group_request["_start_q_override_by_robot"] = {
                name: np.asarray(q, dtype=float).tolist()
                for name, q in start_q_overrides.items()
            }
            max_workers = min(len(pose_items), int(request_data.get("max_workers", len(pose_items))))
            start_source = "previous inspection pose" if start_q_overrides else "current robot pose"
            self.__console.info(
                f"inspection path planning: {group_name} ({len(pose_items)} robots), "
                f"start={start_source}, start_overrides={list(start_q_overrides.keys())}, "
                f"pose_transform={'yes' if pose_transform is not None else 'no'}")
            executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
            futures = {
                executor.submit(
                    self._plan_inspection_path_for_robot,
                    group_request,
                    robot_name,
                    self._transform_target_pose(target_T, pose_transform),
                    obstacle_mesh,
                ): (robot_name, pose_name)
                for robot_name, pose_name, target_T in pose_items
            }
            try:
                for future in as_completed(futures, timeout=future_timeout):
                    robot_name, pose_name = futures[future]
                    failure_key = f"{group_name}:{robot_name}"
                    try:
                        plan = future.result()
                        plan["pose_name"] = pose_name
                        plan["inspection_pose_name"] = group_name
                        plan["inspection_pose_index"] = group_index
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
                        self.__console.error(f"inspection path failed for {failure_key}: {exc}")
                    except Exception as exc:
                        group_failures[robot_name] = str(exc)
                        failures[failure_key] = str(exc)
                        self.__console.error(f"inspection path failed for {failure_key}: {exc}")
            except FuturesTimeoutError:
                for future, (robot_name, _pose_name) in futures.items():
                    if future.done():
                        continue
                    future.cancel()
                    failure_key = f"{group_name}:{robot_name}"
                    group_failures[robot_name] = f"path planning timeout ({planning_timeout:.1f}s)"
                    failures[failure_key] = group_failures[robot_name]
                    self.__console.error(f"inspection path timeout for {failure_key}: {planning_timeout:.1f}s")
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            group_sequence.append({
                "index": group_index,
                "name": group_name,
                "plans": plans,
                "failures": group_failures,
                "ik_failures": group_ik_failures,
            })
        return group_sequence

    def _handle_request_plan_inspection_path(self, request_data):
        """검사 target group 하나 이상에 대해 로봇 경로를 순차 계획한다.

        입력:
            request_data(dict):
                - command: "plan_inspection_path".
                - planner: 사용할 path planner 이름.
                - robot: 수동 검사점 계획 시 사용할 단일 로봇 이름.
                - target_groups: 선택 사항. 여러 검사 자세를 직접 지정할 때 사용한다.
                - use_ef_pose_targets: True이면 determine_ef_pose에서 저장한 여러 검사 자세를 사용한다.
                - planning_timeout: 선택 사항. group별 future timeout.
                - max_workers: 선택 사항. 같은 group 안에서 병렬 계획할 로봇 수.
                - _identity: ZApi 응답 식별자.

        출력:
            ZApi reply_inspection_path(result):
                result(dict)는 status/planner/inspection_groups/robots/failures/
                ik_failures/timing을 포함한다.

        연산:
            1. 수동 pick point 또는 EF pose 결과를 동일한 target group 구조로 변환한다.
            2. 각 group 안의 로봇들은 병렬로 계획한다.
            3. 다음 group은 이전 group에서 계산된 각 로봇의 마지막 q를 start q로 사용한다.
            4. viewer playback 상태와 path/goal pose 시각화를 갱신한다.
        """
        identity = request_data.get("_identity")
        result = {"status": "failed"}
        total_t0 = time.perf_counter()
        failures = {}
        ik_failures = {}
        try:
            self._clear_inspection_visuals(clear_point=False)
            # target group을 접근 가능(first)/불가(second)로 나누고 각각 정렬한다.
            # 현재 로봇 위치는 base 위치로 가정한다. first를 계획한 뒤 포지셔너를
            # 룰베이스로 가상 회전하고 second를 이어서 계획한다.
            first_groups, second_groups = self._partition_and_sort_inspection_groups(
                self._inspection_target_groups
            )
            
            self._inspection_second_groups = second_groups
            self.__console.info(
                "inspection groups partitioned: "
                f"first(reachable)={len(first_groups)}, second(deferred)={len(second_groups)}, "
                f"first_order={[g.get('name') for g in first_groups]}")
            if not first_groups:
                raise RuntimeError(
                    f"no reachable inspection group now (deferred={len(second_groups)})")
            self._clear_inspection_goal_pose_visuals(render=False)
            for group_info in first_groups:
                for robot_name, _pose_name, target_T in self._inspection_group_pose_items(group_info):
                    self._show_inspection_goal_pose(
                        robot_name,
                        target_T,
                        clear=False,
                        render=False,
                    )
            self.plotter.render()

            stage_t0 = time.perf_counter()
            obstacle_mesh = self._current_spool_collision_mesh()
            if obstacle_mesh is None:
                raise RuntimeError("loaded pipe is not available")
            obstacle_elapsed = time.perf_counter() - stage_t0

            start_q_overrides = {}
            planning_timeout = float(request_data.get(
                "planning_timeout",
                (self._config.get("path_planning", {}) or {}).get("planning_timeout", 0.0),
            ))
            future_timeout = None if planning_timeout <= 0 else planning_timeout + 2.0

            # 1) first group(현재 접근 가능) 경로 계획.
            group_sequence = self._plan_inspection_group_sequence(
                first_groups,
                obstacle_mesh,
                request_data,
                start_q_overrides=start_q_overrides,
                failures=failures,
                ik_failures=ik_failures,
                planning_timeout=planning_timeout,
                future_timeout=future_timeout,
            )

            # 2) first -> second 전환: 룰베이스 고정각으로 포지셔너 r축을 가상 회전한다.
            #    실제 포지셔너/spool은 움직이지 않고, second pose와 collision mesh만 회전시켜 계획한다.
            if second_groups:
                delta_r_deg = float(request_data.get(
                    "positioner_second_group_r_deg",
                    (self._config.get("path_planning", {}) or {}).get(
                        "positioner_second_group_r_deg", 180.0),
                ))
                rotation_T = self._positioner_r_rotation_transform(delta_r_deg)
                rotated_obstacle_mesh = copy.deepcopy(obstacle_mesh)
                rotated_obstacle_mesh.transform(rotation_T)
                self.__console.info(
                    "positioner rule-based virtual rotation for second group: "
                    f"delta_r={delta_r_deg:.1f}deg, groups={len(second_groups)}")
                # 3) second group 경로 계획 (회전된 pose/mesh 기준, first의 마지막 q에서 이어서).
                group_sequence += self._plan_inspection_group_sequence(
                    second_groups,
                    rotated_obstacle_mesh,
                    request_data,
                    start_q_overrides=start_q_overrides,
                    failures=failures,
                    ik_failures=ik_failures,
                    planning_timeout=planning_timeout,
                    future_timeout=future_timeout,
                    group_offset=len(first_groups),
                    pose_transform=rotation_T,
                )

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
                raise RuntimeError(f"all inspection path plans failed: {failures}")
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
                        self._show_ik_failure_reached_pose(robot_name, plan.get("reached_T"), None)

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
                            robot_name: self._inspection_plan_result_for_robot(plan)
                            for robot_name, plan in group["plans"].items()
                        },
                        "failures": group["failures"],
                    }
                    for group in group_sequence
                ],
                "robots": {
                    robot_name: self._inspection_plan_result_for_robot(plan)
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
                "inspection paths planned by target group: "
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
            self.__console.error(f"inspection path planning failed after {elapsed:.3f}s: {e}")
        if hasattr(self, 'zapi') and self.zapi and identity:
            self.zapi.reply_inspection_path(result, identity=identity)

    def _inspection_plan_result_for_robot(self, plan):
        """단일 로봇 plan dict를 ZApi 응답용 요약 dict로 변환한다.

        입력:
            plan(dict): _plan_inspection_path_for_robot() 결과.
                필수 키: q_path, waypoints, elapsed, verification, collision_preview.
                선택 키: pose_name, fallback_reason, ik_result, planning_error, timing.

        출력:
            dict:
                pose_name, waypoints, init_q, target_q, elapsed, verification,
                collision_preview, ik_result, timing 등을 포함한다.
        """
        return {
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


    def _handle_request_check_ef_pose_ik(self, request_data):
        """저장된 EF pose target group들에 대해 IK 가능 여부를 검사하고 goal pose를 표시한다."""
        identity = request_data.get("_identity")
        result = {"status": "failed"}
        total_t0 = time.perf_counter()
        failures = {}
        ik_failures = {}
        try:
            self._clear_inspection_visuals(clear_point=False)
            # target_groups = self._inspection_target_groups_for_planning
            target_groups = self._inspection_target_groups
            if not target_groups:
                raise RuntimeError("EF poses are not determined")
            self._clear_inspection_goal_pose_visuals(render=False)
            for group_info in target_groups:
                for robot_name, _pose_name, target_T in self._inspection_group_pose_items(group_info):
                    self._show_inspection_goal_pose(
                        robot_name,
                        target_T,
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
            for sequence_index, group_info in enumerate(target_groups):
                group_index = int(group_info.get("index", sequence_index))
                group_name = str(group_info.get("name", f"Inspection pose {sequence_index + 1}"))
                pose_items = self._inspection_group_pose_items(group_info)
                checks = {}
                group_failures = {}
                group_ik_failures = {}
                group_request = dict(request_data)
                group_request["_start_q_override_by_robot"] = {}
                self.__console.info(
                    f"EF pose IK check: {group_name} ({len(pose_items)} robots), "
                    "start_q=zeros")
                for robot_name, pose_name, target_T in pose_items:
                    failure_key = f"{group_name}:{robot_name}"
                    try:
                        check = self._check_inspection_ik_for_robot(
                            group_request,
                            robot_name,
                            target_T,
                            obstacle_mesh,
                        )
                        check["pose_name"] = pose_name
                        check["inspection_pose_name"] = group_name
                        check["inspection_pose_index"] = group_index
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
                    "index": group_index,
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

        # 3. Step manipulator joint animations (interpolated motion)
        now = time.time()
        dt = 0.0 if self._last_anim_time is None else (now - self._last_anim_time)
        self._last_anim_time = now
        if self._joint_animations and dt > 0:
            self._step_joint_animations(min(dt, 0.1))   # dt가 너무 커지는 것을 방지
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
        """활성 조인트 애니메이션을 사다리꼴 속도 프로파일로 한 스텝 진행한다.
        가속(accel)으로 max_speed까지 올린 뒤 등속, target 도달 시 감속/정지한다.
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

            # 정지 판정: 남은 거리와 속도가 충분히 작으면 종료
            if dist <= 1e-6 and vel <= accel * dt:
                model.set_joint(jn, tgt); model.update_fk()
                changed = True
                continue

            # 감속에 필요한 거리 = v^2 / (2a). 그보다 가까우면 감속, 아니면 가속/등속
            stop_dist = (vel * vel) / (2.0 * accel)
            if dist <= stop_dist:
                vel = max(0.0, vel - accel * dt)      # 감속
            else:
                vel = min(vmax, vel + accel * dt)     # 가속 후 vmax 제한

            new_cur = cur + direction * vel * dt
            # target을 지나치면 target으로 스냅하고 종료
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
        """해당 로봇/조인트의 기존 애니메이션을 교체하고 사다리꼴 프로파일로 이동을 시작한다.
        accel 미지정 시 speed의 2배 또는 0.5s 가속 기준으로 기본 설정한다.
        """
        model = self._find_robot(robot_name)
        if model is None or model._urdf is None:
            self.__console.warning(f"move_manipulator: robot not found '{robot_name}'")
            return
        if joint_name not in model._urdf._joint_map:
            self.__console.warning(f"move_manipulator: joint not found '{joint_name}'")
            return
        spd = float(speed)
        acc = float(accel) if accel is not None else max(spd * 2.0, 1e-6)
        # 같은 (robot, joint)의 현재 속도를 이어받아 부드럽게 재타게팅한다.
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
            f"move_manipulator: {robot_name}.{joint_name} -> {target} (vmax={spd}, accel={acc})")

    def _stop_joint_animation(self, robot_name, joint_name=None):
        """해당 로봇 또는 특정 조인트의 애니메이션을 즉시 중지한다."""
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
            self.__console.warning("execute_inspection_path: planned path is not available")
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
            self.__console.warning("execute_inspection_path: failed to create Pinocchio model")
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
            self.__console.warning("execute_inspection_path: planned path is not available")
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
        # spool pose = chuck 기준 spool offset. 사용자가 UI에서 조정한 값을 그대로 내보낸다.
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
        """spool actor를 새 점군으로 교체한다. 필터 결과 반영에 사용한다."""
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
        # spool 모델 일관성을 위해 world point를 현재 chuck@offset 기준 local로 환산한다.
        Tc = self._chuck_world_T()
        if Tc is not None and getattr(self, '_spool_local_verts', None) is not None:
            Tinv = np.linalg.inv(Tc @ self._spool_offset_T())
            self._spool_local_verts = (Tinv[:3, :3] @ new_pts.T).T + Tinv[:3, 3]
        self.plotter.render()

    def _handle_request_filter_spool(self, request_data):
        """현재 로드된 spool에 직접 노이즈 필터(SOR/CCL)를 적용한다."""
        pts = self._get_spool_points()
        if pts is None:
            self.__console.warning("filter_spool: loaded spool is not available")
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
                    self.__console.warning("filter_spool(ccl): no connected component found")
                    return
                uniq, cnts = np.unique(valid, return_counts=True)
                kept = pts[labels == uniq[np.argmax(cnts)]]
            else:
                self.__console.warning(f"filter_spool: unknown method '{method}'")
                return
            self._replace_spool_points(kept)
            self.__console.info(f"filter_spool({method}): {n0} -> {len(kept)} points (removed {n0 - len(kept)})")
        except Exception as e:
            self.__console.error(f"filter_spool failed: {e}")

    def _handle_request_reconstruct_mesh(self, request_data):
        """현재 로드된 spool 점군으로 mesh를 재구성(Marching Cubes)해 표시한다."""
        pts = self._get_spool_points()
        if pts is None:
            self.__console.warning("reconstruct_mesh: loaded spool is not available")
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
                self.__console.warning("reconstruct_mesh: empty mesh")
                return
            vmesh = vedo.Mesh([verts, faces]).c("gray")

            # 기존 pcd spool과 이전 재구성 mesh 제거
            old_pcd = getattr(self, '_loaded_spool_mesh', None)
            if old_pcd is not None:
                self.plotter.remove(old_pcd)
            old_recon = getattr(self, '_spool_recon_mesh', None)
            if old_recon is not None and old_recon is not old_pcd:
                self.plotter.remove(old_recon)

            self.plotter.add(vmesh)
            # 재구성 mesh를 spool로 사용해 positioner/chuck 추종 시 같이 움직이도록 한다.
            self._loaded_spool_mesh = vmesh
            self._spool_recon_mesh = vmesh
            Tc = self._chuck_world_T()
            T = (getattr(self, '_spool_world_T', None)
                 if getattr(self, '_spool_world_T', None) is not None
                 else ((Tc @ self._spool_offset_T()) if Tc is not None else np.eye(4)))
            Tinv = np.linalg.inv(T)
            # verts(world)를 local로 환산해 world = T @ local 관계를 유지한다.
            self._spool_local_verts = (Tinv[:3, :3] @ verts.T).T + Tinv[:3, 3]
            self._spool_world_T = T
            if Tc is not None:
                self._chuck_prev_T = Tc
            self.plotter.render()
            self._probe_current_spool_pinocchio_collision("reconstruct_mesh")
            self.__console.info(f"reconstruct_mesh: vertices={len(verts)}, faces={len(faces)} (pcd replaced by mesh)")
        except Exception as e:
            self.__console.error(f"reconstruct_mesh failed: {e}")

    def _handle_request_save_spool(self, request_data):
        """현재 spool 결과를 저장한다. 재구성 mesh가 있으면 mesh, 없으면 point cloud를 저장한다."""
        path = request_data.get("path")
        if not path:
            return
        try:
            recon = getattr(self, '_spool_recon_mesh', None)
            if recon is not None and hasattr(recon, "vertices") and hasattr(recon, "cells"):
                # 저장 mesh는 spool local frame으로 기록한다. JSON의 chuck 기준 offset을 다시
                # 적용하면 load 후 동일한 pose로 복원할 수 있다.
                verts = getattr(self, '_spool_local_verts', None)
                if verts is None:
                    verts = np.asarray(recon.vertices)
                m = _o3d.geometry.TriangleMesh()
                m.vertices = _o3d.utility.Vector3dVector(np.asarray(verts, dtype=float))
                m.triangles = _o3d.utility.Vector3iVector(np.asarray(recon.cells, dtype=np.int32))
                m.compute_vertex_normals()
                _o3d.io.write_triangle_mesh(path, m)
                self.__console.info(f"save_spool: saved local-frame mesh {path}")
            else:
                pts = self._get_spool_points()
                if pts is None:
                    self.__console.warning("save_spool: no spool data to save")
                    return
                pcd = _o3d.geometry.PointCloud()
                pcd.points = _o3d.utility.Vector3dVector(pts)
                _o3d.io.write_point_cloud(path, pcd)
                self.__console.info(f"save_spool: saved point cloud {path} ({len(pts)} points)")
        except Exception as e:
            self.__console.error(f"save_spool failed: {e}")

    # --- spool frame fixation (rigid mount assumption) ---
    F_CHUCK_LINK_NAME = "f_column_passive_clamp"
    M_CHUCK_LINK_NAME = "m_column_passive_r"
    CHUCK_LINK_NAME = M_CHUCK_LINK_NAME

    @staticmethod
    def _rotz(deg):
        return geom_utils.rotz(deg)

    @staticmethod
    def _rotx(deg):
        return geom_utils.rotx(deg)

    @staticmethod
    def _transl(v):
        return geom_utils.transl(v)

    @staticmethod
    def _rot_about_axis(axis, center, deg):
        """center를 지나는 axis 둘레로 deg만큼 회전하는 4x4 변환을 만든다."""
        return geom_utils.rot_about_axis(axis, center, deg)

    def _chuck_world_T(self):
        """m-column chuck joint(m_column_passive_r) link의 4x4 world transform을 반환한다."""
        for model in getattr(self, '_robot_models', []):
            if hasattr(model, 'get_link_world_T'):
                T = model.get_link_world_T(self.CHUCK_LINK_NAME)
                if T is not None:
                    return np.asarray(T, dtype=float)
        return None

    def _spool_offset_T(self):
        """UI의 chuck 기준 spool pose를 4x4 transform으로 변환한다."""
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
        """현재 _spool_world_T로 spool actor 정점을 갱신한다. world = T @ local."""
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
        mesh로 로드된 spool처럼 local frame이 없는 경우, 현재 화면 좌표를
        현재 chuck@offset 기준 local frame으로 환산해 이후 fixation 이동이 가능하게 한다.
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
        """수동 배치: 현재 chuck 기준으로 spool을 배치한다. spool_world = T_chuck @ T_offset."""
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

