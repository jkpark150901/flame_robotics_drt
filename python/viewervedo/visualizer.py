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
import json
import pickle
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
from robot_core.service import (
    OPERATION_PLAN_SINGLE_TARGET,
    OPERATION_POSE_DETERMINE,
    submit_robot_core_request,
)
from viewervedo.robot import RobotModel, load_robots_from_config
from viewervedo import geometry_utils as geom_utils
from viewervedo import pipe_alignment_utils
from viewervedo import vedo_visual_utils
from plugins.pluginbase.plannerbase import PlannerBase
from plugins.pathplanner import Q_SPACE_PLANNER_MODULES
try:
    from plugins.pathplanner.ompl import OMPLPlannerBase, SUPPORTED_ALGORITHMS as OMPL_SUPPORTED_ALGORITHMS
except ImportError:
    OMPLPlannerBase = None
    OMPL_SUPPORTED_ALGORITHMS = ()
from plugins.robotics.backend import RobotDescription
from plugins.robotics.inspection_experiment_logger import InspectionExperimentLogger
from plugins.robotics.inspection_planning_base import InspectionIKRequest, InspectionPlanningBase
from plugins.robotics.inspection_workflow import to_jsonable, resolve_target_groups_with_rotation
from plugins.robotics.pinocchio_backend import PinocchioRoboticsBackend
from plugins.poseDeterminator.EndEffectorPoseOptimizer import EndEffectorPoseOptimizer


class InspectionIKFailure(RuntimeError):
    def __init__(self, message, failure_info=None):
        super().__init__(message)
        self.failure_info = failure_info or {}


# add_collision_objects() is always called as [obstacle_mesh, positioner_mesh]
# (_configure_inspection_planner - the only call site), and
# PinocchioRoboticsBackend._add_static_mesh names static geometry
# "collision_object_{registration order}" - so index 0 is always the pipe and
# index 1 is always the positioner housing, stably, throughout this app.
# "collision_object_N" alone forces the reader to remember/look up that
# mapping every time; label it inline instead.
_STATIC_COLLISION_OBJECT_LABELS = {"collision_object_0": "pipe", "collision_object_1": "positioner"}


def _label_collision_pairs(pairs):
    def _label(name):
        friendly = _STATIC_COLLISION_OBJECT_LABELS.get(name)
        return f"{friendly}({name})" if friendly else name
    return [[_label(a), _label(b)] for a, b in (pairs or [])]


class Visualizer:
    def __init__(self, config:dict=None):
        if config is None:
            config = {}
        self._config = config
    
        self.__console  = ConsoleLogger.get_logger()
        experiment_root = Path(config.get("debug_dir", "debug")) / "inspection_ik"
        self._inspection_ik_experiment_logger   = InspectionExperimentLogger(experiment_root)
        self._inspection_ik_experiment_dir      = self._inspection_ik_experiment_logger.session_dir
        self.__console.info(f"inspection IK experiment session: {self._inspection_ik_experiment_dir}")
        # Thread-safe request queue (populated by Zapi)
        self._request_queue = deque(maxlen=100)
        self._queue_lock    = threading.Lock()
        self._robot_core = None

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
            "save_inspection_points": self._handle_request_save_inspection_points,      # 선택된 검사 지점들을 JSON 파일로 저장한다.
            "load_inspection_points": self._handle_request_load_inspection_points,      # JSON 파일에서 검사 지점들을 복원한다.
            "save_planning_snapshot": self._handle_request_save_planning_snapshot,      # 결정된 EF pose + collision scene을 벤치마킹용 snapshot(pickle)으로 저장한다.
            "pick_chuck_mount_points": self._handle_request_pick_chuck_mount_points,    # chuck mount 기준점 선택/align 모드를 설정한다.
            "set_chuck_mount_points": self._handle_request_set_chuck_mount_points,      # 외부에서 전달된 chuck mount 점을 반영한다.
            "set_chuck_mount_config": self._handle_request_set_chuck_mount_config,      # chuck mount frame/offset 설정을 갱신한다.
            "clear_chuck_mount_points": self._handle_request_clear_chuck_mount_points,  # 선택된 chuck mount 점을 초기화한다.

            "determine_ef_pose": self._handle_request_determine_ef_pose,                # 선택 지점 기준으로 검사 end-effector pose 후보를 계산한다.
            "check_ef_pose_ik": self._handle_request_check_ef_pose_ik,                  # EF pose 후보들의 IK 가능 여부를 검사한다.
            "plan_single_target": self._handle_request_plan_single_target,              # 로봇 하나의 source_q -> target_pose 단일 경로를 계획한다.
            "prepare_next_inspection_phase": self._handle_request_prepare_next_inspection_phase,  # 회전 필요 phase 진입 전 팔을 안전 자세로 접고 포지셔너를 돌린다.
            "robot_core_completed": self._handle_robot_core_completed,                  # Robot Core 결과만 시각화한다.
            
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
            self._invalidate_spool_collision_mesh_cache()
            self._render_spool_offset()
            self.plotter.render()
            self._load_spool_alignment_state(path, identity=identity)
            self._probe_current_spool_pinocchio_collision("load_spool")
            # 비싼 alpha-shape collision mesh를 지금(로드 시점) 딱 한 번 만들어 spool local
            # frame으로 보관한다. 이후 path planning / positioner 회전에서는 재생성 없이
            # 현재 _spool_world_T rigid 변환만 다시 적용한다.
            try:
                self._current_spool_collision_mesh()
            except Exception as prebuild_exc:
                self.__console.debug(f"spool collision mesh prebuild skipped: {prebuild_exc}")
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
            self._invalidate_spool_collision_mesh_cache()
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

        self._invalidate_positioner_collision_mesh_cache()
        spool_T_before = np.asarray(getattr(self, '_spool_world_T', None), dtype=float).copy() \
            if getattr(self, '_spool_world_T', None) is not None else None
        self._sync_fixed_spool_after_positioner_move(axis, position, prev_positioner_r, request_data)
        if axis == "r" and getattr(self, '_spool_fix_r', False) and abs(position - prev_positioner_r) > 1e-9:
            # 배관(spool)이 실제로 r축과 같이 돌았으면, 저장된 ef target pose도 배관에
            # 고정돼 있다고 가정하고 정확히 같은 rigid transform으로 같이 돌린다.
            # _sync_fixed_spool_after_positioner_move가 spool에 적용한 것과 완전히
            # 동일한 축/중심/부호 공식(_positioner_r_rotation_transform)을 재사용한다.
            try:
                rotation_T = self._positioner_r_rotation_transform(position - prev_positioner_r)
                self._rotate_inspection_target_groups(rotation_T)
                self._verify_positioner_rotation_kept_poses_attached(
                    spool_T_before, np.asarray(getattr(self, '_spool_world_T'), dtype=float), rotation_T)
                self._verify_rotated_ef_poses_against_current_pipe()
            except Exception as exc:
                self.__console.warning(
                    f"failed to rotate stored ef target poses with positioner r move: {exc}")
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
            elif axis == "r" and getattr(self, '_spool_fix_r', False):
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

    def _handle_request_save_inspection_points(self, request_data):
        """현재 선택된 검사 지점(pick point, 원본 좌표)들을 JSON 파일로 저장한다."""
        path = request_data.get("path")
        if not path:
            self.__console.warning("save_inspection_points: no path given")
            return
        points = getattr(self, "_inspection_points", []) or []
        try:
            payload = {"points": [np.asarray(p, dtype=float).tolist() for p in points]}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            self.__console.info(f"save_inspection_points: saved {len(points)} point(s) to {path}")
        except Exception as exc:
            self.__console.error(f"save_inspection_points failed: {exc}")

    def _handle_request_load_inspection_points(self, request_data):
        """JSON 파일에서 검사 지점들을 읽어, pick으로 찍은 것과 동일하게 복원한다."""
        path = request_data.get("path")
        if not path:
            self.__console.warning("load_inspection_points: no path given")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            raw_points = payload.get("points", [])
        except Exception as exc:
            self.__console.error(f"load_inspection_points failed: {exc}")
            return
        self._clear_inspection_points(render=False)
        for raw_point in raw_points:
            self._set_inspection_point(np.asarray(raw_point, dtype=float))
        identity = request_data.get("_identity")
        if hasattr(self, 'zapi') and self.zapi and identity:
            self.zapi.update_inspection_point({
                "points": [p.tolist() for p in getattr(self, "_inspection_points", [])],
            }, identity=identity)
        self.__console.info(f"load_inspection_points: loaded {len(raw_points)} point(s) from {path}")

    def _handle_request_save_planning_snapshot(self, request_data):
        """EF pose(target_groups)와 함께 Robot Core 경로계획에 필요한 전체 scene 상태를
        pickle 파일로 저장한다 (배관/positioner collision mesh, 로봇 joint 상태 등).

        planner 벤치마킹 스크립트(scripts/benchmark_path_planners.py)가 이 파일 하나만으로
        Visualizer 없이 headless RobotCoreEngine을 재구성할 수 있도록, 실제 plan_inspection_path
        요청에서 robot_core로 보내는 것과 동일한 snapshot(_inspection_robot_core_snapshot)을 그대로
        재사용한다.
        """
        path = request_data.get("path")
        identity = request_data.get("_identity")
        if not path:
            self.__console.warning("save_planning_snapshot: no path given")
            return
        try:
            snapshot = self._inspection_robot_core_snapshot(request_data)
        except Exception as exc:
            self.__console.error(f"save_planning_snapshot: snapshot build failed: {exc}")
            if hasattr(self, "zapi") and self.zapi and identity:
                self.zapi.reply_planning_snapshot(
                    {"status": "failed", "message": str(exc)}, identity=identity)
            return
        try:
            with open(path, "wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.__console.info(
                f"save_planning_snapshot: saved {len(snapshot.get('target_groups') or [])} "
                f"target group(s) to {path}")
            if hasattr(self, "zapi") and self.zapi and identity:
                self.zapi.reply_planning_snapshot(
                    {"status": "success", "path": str(path)}, identity=identity)
        except Exception as exc:
            self.__console.error(f"save_planning_snapshot failed: {exc}")
            if hasattr(self, "zapi") and self.zapi and identity:
                self.zapi.reply_planning_snapshot(
                    {"status": "failed", "message": str(exc)}, identity=identity)

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
        """Render playback from the plan sequence owned and supplied by SimTool."""
        plan_sequence = request_data.get("plan_sequence")
        if isinstance(plan_sequence, list):
            self._last_inspection_plan_sequence = copy.deepcopy(plan_sequence)
        if request_data.get("playback_initial_r_deg") is not None:
            self._inspection_playback_initial_r_deg = float(
                request_data.get("playback_initial_r_deg"))
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

        robot_core_config = (
            self._config.get("robot_core_service", {})
            or self._config.get("planner_service", {})
            or {}
        )
        shutdown_hotkey = str(robot_core_config.get("shutdown_hotkey", "F12"))
        if str(key).lower() == shutdown_hotkey.lower():
            self._shutdown_robot_core()
            return
        
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

    def _shutdown_robot_core(self):
        """Terminate the embedded child or external standalone Robot Core service."""
        robot_core = getattr(self, "_robot_core", None)
        if robot_core is None or not robot_core.is_running:
            self.__console.info("Robot Core is already stopped")
            return False
        pid = robot_core.pid
        try:
            shutdown = getattr(robot_core, "shutdown", None)
            if shutdown is None:
                shutdown = robot_core.stop
            stopped = bool(shutdown())
        except Exception as exc:
            self.__console.error(f"Robot Core shutdown failed: pid={pid}, error={exc}")
            return False
        if stopped:
            self.__console.info(f"Robot Core stopped by hotkey: pid={pid}")
        else:
            self.__console.warning(
                f"Robot Core shutdown request timed out: pid={pid}"
            )
        return stopped

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
        self._invalidate_positioner_collision_mesh_cache()
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
        from plugins.robotics.inspection_workflow import ef_pose_robot_name
        return ef_pose_robot_name(pose_name)

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

    def _inspection_pose_process_snapshot(self, request_data):
        points = self._get_spool_points()
        if points is None or len(points) < 10:
            raise RuntimeError("loaded spool point cloud is not available")
        selected = request_data.get("inspection_points")
        if not isinstance(selected, list):
            selected = getattr(self, "_inspection_points", []) or []
        return {
            "spool_points": np.asarray(points, dtype=float),
            "target_groups": [],
            "robot_joint_states": {
                str(getattr(model, "name", "")): {
                    str(name): float(value)
                    for name, value in (getattr(model, "_joint_cfg", {}) or {}).items()
                }
                for model in getattr(self, "_robot_models", []) or []
            },
            "inspection_points": copy.deepcopy(selected),
        }

    def _handle_request_determine_ef_pose(self, request_data):
        """선택된 검사 지점 여러 개를 순회해 EF pose target group 목록을 만든다."""
        if getattr(self, "_robot_core_worker_mode", False):
            raise RuntimeError(
                "Viewer pose determination is disabled; use robot_core.pose_service")
        robot_core = getattr(self, "_robot_core", None)
        identity = request_data.get("_identity")
        try:
            if robot_core is None or not robot_core.is_running:
                raise RuntimeError("robot core process is not running")
            core_request = copy.deepcopy(request_data)
            core_request["operation"] = OPERATION_POSE_DETERMINE
            snapshot = self._inspection_pose_process_snapshot(core_request)
            core_request["inspection_points"] = snapshot["inspection_points"]
            request_id = robot_core.submit(core_request, snapshot)
            self._active_pose_request_id = request_id
            self.__console.info(
                f"EF pose request submitted to robot core: request_id={request_id}")
            return request_id
        except Exception as exc:
            result = {"status": "failed", "message": str(exc), "elapsed": 0.0}
            self.__console.error(f"EF pose process submission failed: {exc}")
            if hasattr(self, "zapi") and self.zapi and identity:
                self.zapi.reply_ef_pose(result, identity=identity)
            return None
        identity = request_data.get("_identity")
        result = {"status": "failed"}
        total_t0 = time.perf_counter()
        try:
            self._clear_ik_failure_visuals(render=False)
            requested_points = request_data.get("inspection_points")
            inspection_points = [
                np.asarray(point, dtype=float)
                for point in (
                    requested_points
                    if isinstance(requested_points, list)
                    else (getattr(self, "_inspection_points", []) or [])
                )
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
                    # optimizer는 포인트마다 독립적으로 "Inspection pose 1"부터 이름을 다시 매긴다.
                    # 여러 포인트를 합치면 이름이 중복되므로, 어느 검사 포인트의 몇 번째 포즈인지
                    # 알 수 있도록 이름 앞에 포인트 번호를 붙이고 index도 전체 기준으로 다시 매긴다.
                    for local_index, group_info in enumerate(target_groups):
                        group_info["point_index"] = point_index
                        group_info["name"] = (
                            f"Point {point_index + 1} - {group_info.get('name', f'pose {local_index + 1}')}"
                        )
                        group_info["index"] = len(all_target_groups) + local_index
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
        if module_name in OMPL_SUPPORTED_ALGORITHMS:
            if OMPLPlannerBase is None:
                raise RuntimeError(
                    f"OMPL planner '{module_name}' requested but the native OMPL "
                    "python bindings are not installed in this environment")
            planner = OMPLPlannerBase()
            planner.configure_ompl({"algorithm": module_name})
            return planner
        if module_name not in Q_SPACE_PLANNER_MODULES:
            raise RuntimeError(
                f"unsupported planner: {module_name!r}. "
                f"supported={sorted(Q_SPACE_PLANNER_MODULES) + list(OMPL_SUPPORTED_ALGORITHMS)}")
        from plugins.pluginbase.plannerbase import PlannerBase
        module = importlib.import_module(f"plugins.pathplanner.{module_name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, PlannerBase)
                and obj is not PlannerBase
                and obj.__module__ == module.__name__
            ):
                return obj()
        raise RuntimeError(f"Planner plugin class not found: {module_name}")

    def _load_path_optimizer(self, module_name):
        if not module_name:
            return None
        from plugins.pluginbase.optimizerbase import OptimizerBase
        module = importlib.import_module(f"plugins.optimizer.{module_name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, OptimizerBase) and obj is not OptimizerBase:
                return obj()
        raise RuntimeError(f"Optimizer plugin class not found: {module_name}")

    def _apply_path_optimizer(self, optimizer_name, q_path, planner):
        if not optimizer_name:
            return list(q_path or []), None
        optimizer = self._load_path_optimizer(optimizer_name)
        optimized_path = optimizer.optimize(list(q_path or []), planner)
        status = getattr(optimizer, "last_optimization_status", None)
        return [np.asarray(q, dtype=float) for q in (optimized_path or [])], status

    @staticmethod
    def _path_optimization_requested(request_data):
        optimizer_name = request_data.get("optimizer")
        return bool(request_data.get("optimize_path", bool(optimizer_name)))

    def _planner_timeout(self, request_data, planner_name=None):
        """Return the timeout explicitly supplied by the ZAPI request."""
        explicit_timeout = request_data.get("planning_timeout")
        return 0.0 if explicit_timeout is None else float(explicit_timeout)

    def _inspection_q_space_planner_name(self, planner_name):
        planner_name = str(planner_name or "rrt_connect")
        if planner_name in Q_SPACE_PLANNER_MODULES or planner_name in OMPL_SUPPORTED_ALGORITHMS:
            return planner_name
        raise RuntimeError(
            f"planner '{planner_name}' is not supported for robot q-space planning. "
            f"supported={sorted(Q_SPACE_PLANNER_MODULES) + list(OMPL_SUPPORTED_ALGORITHMS)}")

    def _invalidate_spool_collision_mesh_cache(self):
        """배관 geometry 자체가 바뀌면(새 배관 로드/제거/재구성) 호출한다 - 전체 재생성.

        단순 world pose 변경(포지셔너 r 회전, 척 이동)에는 쓰지 않는다. 그런 이동은
        _current_spool_collision_mesh가 local-frame 보관본에 _spool_world_T를 rigid 변환으로
        다시 적용해 처리하므로 alpha-shape 재생성이 필요 없다.
        """
        self._spool_collision_mesh_cache = None
        self._spool_collision_mesh_cache_T = None
        self._spool_collision_mesh_local = None

    def _current_spool_collision_mesh(self):
        """현재 spool의 collision mesh를 반환한다.

        alpha-shape 재구성은 비싸므로 배관을 로드할 때 딱 한 번만 만들어 spool local
        frame으로 보관하고, 이후 배관이 움직이면(_spool_world_T 변경) 그 보관본에 현재
        _spool_world_T를 rigid 변환으로 다시 적용하기만 한다(재생성 없음). 이 변환은 spool
        actor 정점을 갱신하는 _apply_spool_world_T의 world = T @ local과 완전히 동일한
        _spool_world_T를 쓰므로, collision mesh는 항상 시각 배관과 같은 위치에 있다.
        """
        # local-frame 보관본을 만들려면 _spool_world_T(빌드 기준 frame)가 있어야 한다.
        # 없으면 지금 actor 위치로 fixation frame을 확정해 둔다 - 그래야 이후 배관이
        # 움직여도 재생성 없이 rigid 변환만으로 따라간다.
        if getattr(self, '_spool_world_T', None) is None:
            self._ensure_spool_frame_from_actor()
        T_now = getattr(self, '_spool_world_T', None)
        local_mesh = getattr(self, '_spool_collision_mesh_local', None)

        if local_mesh is not None and T_now is not None:
            cached = getattr(self, '_spool_collision_mesh_cache', None)
            cached_T = getattr(self, '_spool_collision_mesh_cache_T', None)
            if cached is not None and cached_T is not None and np.allclose(cached_T, T_now):
                return cached
            world_mesh = copy.deepcopy(local_mesh)
            world_mesh.transform(np.asarray(T_now, dtype=float))
            self._spool_collision_mesh_cache = world_mesh
            self._spool_collision_mesh_cache_T = np.asarray(T_now, dtype=float).copy()
            return world_mesh

        cached = getattr(self, '_spool_collision_mesh_cache', None)
        if cached is not None:
            return cached

        mesh = self._build_spool_collision_mesh()
        self._spool_collision_mesh_cache = mesh
        if mesh is not None and T_now is not None:
            # 빌드 시점의 world mesh를 spool local frame으로 환산해 보관한다. 이후 이동은
            # 이 보관본에 _spool_world_T rigid 변환만 다시 적용한다.
            local = copy.deepcopy(mesh)
            local.transform(np.linalg.inv(np.asarray(T_now, dtype=float)))
            self._spool_collision_mesh_local = local
            self._spool_collision_mesh_cache_T = np.asarray(T_now, dtype=float).copy()
        else:
            self._spool_collision_mesh_local = None
            self._spool_collision_mesh_cache_T = None
        return mesh

    def _rotate_inspection_target_groups(self, rotation_T):
        """저장된 ef target pose(dda_pose/rt_pose/target_point)를 배관과 같이 회전시킨다.

        지금까지는 UI에서 positioner r축을 실제로 돌려도(_handle_request_move_positioner)
        배관(spool)만 움직이고 self._inspection_target_groups에 저장된 ef pose는 그대로
        남아 있었다. 그래서 실제로 배관을 돌린 뒤 이 pose로 계획/검증하면 "배관은 이미
        돌았는데 ef pose는 원래 자리"인 채로 비교하게 되어 원래는 안 겹쳐야 할 것이
        겹치는 것처럼 보였다. ef pose는 배관에 고정돼 있다고 가정하고, 배관에 적용한 것과
        똑같은 rigid transform을 여기에도 그대로 적용한다.
        """
        rotation_T = np.asarray(rotation_T, dtype=float)
        for group_info in getattr(self, "_inspection_target_groups", []) or []:
            # Also rotate *_resolved if present (set by resolve_target_
            # groups_with_rotation() - e.g. "Save Planning Snapshot" mutates
            # these onto these SAME group dicts, in place). inspection_group_
            # pose_items() prefers dda_pose_resolved/rt_pose_resolved over
            # the raw pose whenever it's present - rotating only the raw
            # field left the resolved one frozen at whatever angle it was
            # last baked at, so neither the display nor planning ever saw
            # this rotation take effect once a resolved field existed.
            for key in ("dda_pose", "dda_pose_resolved", "rt_pose", "rt_pose_resolved"):
                pose = group_info.get(key)
                if pose is not None:
                    group_info[key] = rotation_T @ np.asarray(pose, dtype=float)
            point = group_info.get("target_point")
            if point is not None:
                point = np.asarray(point, dtype=float)
                group_info["target_point"] = rotation_T[:3, :3] @ point + rotation_T[:3, 3]

    def _verify_positioner_rotation_kept_poses_attached(self, spool_T_before, spool_T_after, rotation_T):
        """ef pose가 배관에 대해 상대적으로 정말 고정된 채 유지됐는지 수치로 검증한다.

        ef pose와 배관에 똑같은 rigid transform을 적용했다면, "배관 local frame 기준 ef pose
        좌표"는 회전 전/후로 절대 안 변해야 한다. 이 값이 실제로 달라지면 두 변환이 어딘가
        어긋났다는 확실한 증거다.
        """
        if spool_T_before is None or spool_T_after is None:
            return
        try:
            inv_before = np.linalg.inv(spool_T_before)
            inv_after = np.linalg.inv(spool_T_after)
        except np.linalg.LinAlgError:
            return
        max_delta = 0.0
        checked = 0
        for group_info in getattr(self, "_inspection_target_groups", []) or []:
            for key in ("dda_pose", "rt_pose"):
                pose_after = group_info.get(key)
                if pose_after is None:
                    continue
                pose_after = np.asarray(pose_after, dtype=float)
                pose_before = np.linalg.inv(rotation_T) @ pose_after
                local_before = inv_before @ pose_before
                local_after = inv_after @ pose_after
                delta = float(np.max(np.abs(local_before[:3, 3] - local_after[:3, 3])))
                max_delta = max(max_delta, delta)
                checked += 1
        self.__console.info(
            "positioner r move: ef-pose-to-pipe attachment check | "
            f"checked_poses={checked}, max_local_offset_delta={max_delta:.6f}"
            + (" (OK, rigidly attached)" if max_delta < 1e-6 else " (WARNING: drifted)"))

    def _verify_rotated_ef_poses_against_current_pipe(self):
        """positioner r 이동으로 ef pose를 회전시킨 직후, 지금(회전 반영된) 배관 mesh를
        기준으로 순수 기하학적 충돌만 확인하고 화면에 표시한다. IK는 안 푼다 - ef pose
        지점 자체가 배관 mesh 안에 들어가 있는지(signed distance < 0)만 본다.

        rotation 로직이 수학적으로 맞더라도, 배관이 완전한 회전대칭이 아니면 회전 후
        실제로 충돌하는 pose가 생길 수 있다 - 그건 버그가 아니라 실제 형상 문제이므로,
        여기서 바로 잡아내서 로그 + 화면 마커로 남긴다. 정상 pose는 기존 goal-pose
        마커(주황/보라 반투명)로, 충돌 pose는 IK 실패와 동일한 빨간 마커로 표시한다.
        """
        obstacle_mesh = self._current_spool_collision_mesh()
        if obstacle_mesh is None:
            return
        try:
            t_mesh = _o3d.t.geometry.TriangleMesh.from_legacy(obstacle_mesh)
            scene = _o3d.t.geometry.RaycastingScene()
            scene.add_triangles(t_mesh)
        except Exception as exc:
            self.__console.debug(f"post-rotation collision check skipped (scene build failed): {exc}")
            return
        # 회전 전 pose로 그려졌던 마커(determine ef pose 때 _show_ef_target_poses로 그린
        # 것, 이전 회전 검증에서 그린 goal-pose 마커 둘 다)는 지금 더 이상 유효하지 않으니
        # 지운다. clear_poses=False로 self._inspection_target_groups 데이터는 건드리지 않는다.
        self._clear_ef_pose_visuals(clear_poses=False)
        self._clear_inspection_goal_pose_visuals(render=False)
        checked = 0
        failed = []
        for group_info in getattr(self, "_inspection_target_groups", []) or []:
            for robot_name, pose_name, target_T in self._inspection_group_pose_items(group_info):
                target_T = np.asarray(target_T, dtype=float)
                if target_T.shape != (4, 4):
                    target_T = self._pose_to_T(target_T.reshape(-1)[:6])
                pos = target_T[:3, 3].astype(np.float32)
                query = _o3d.core.Tensor([pos], dtype=_o3d.core.Dtype.Float32)
                signed_distance = float(scene.compute_signed_distance(query).numpy()[0])
                checked += 1
                is_colliding = signed_distance < 0.0
                if is_colliding:
                    failed.append(
                        f"{group_info.get('name')}:{robot_name}(signed_distance={signed_distance:.4f})")
                    self._show_ik_failure_reached_pose(robot_name, target_T, target_T)
                else:
                    self._show_inspection_goal_pose(robot_name, target_T, clear=False, render=False)
        self.plotter.render()
        log_fn = self.__console.warning if failed else self.__console.info
        log_fn(
            "post-rotation ef pose collision check (current pipe pose, geometry only): "
            f"checked={checked}, failed={len(failed)}"
            + (f", failures={failed}" if failed else ""))

    def _invalidate_positioner_collision_mesh_cache(self):
        """포지셔너 joint(x/z/r/clamp)가 움직인 뒤 collision mesh 캐시를 무효화한다."""
        self._positioner_collision_mesh_cache = None

    def _ef_pose_workspace_reject_reason(self, world_xyz):
        """ef pose(목표 지점)의 world 위치가 설정된 고정 workspace 박스 밖이면 사유를 반환한다.

        path_planning.enable_ef_pose_workspace_limit가 false면(기본) 항상 None(제한 없음).
        박스 안이면 None, 밖이면 어느 축이 벗어났는지 담은 문자열을 반환한다.
        """
        cfg = self._config.get("path_planning", {}) or {}
        if not bool(cfg.get("enable_ef_pose_workspace_limit", False)):
            return None
        bounds = cfg.get("ef_pose_workspace_bounds", {}) or {}
        pos = np.asarray(world_xyz, dtype=float).reshape(-1)[:3]
        violations = []
        for i, axis in enumerate(("x", "y", "z")):
            lo = float(bounds.get(f"{axis}_min", -np.inf))
            hi = float(bounds.get(f"{axis}_max", np.inf))
            if pos[i] < lo or pos[i] > hi:
                violations.append(f"{axis}={pos[i]:.4f} not in [{lo:.4f}, {hi:.4f}]")
        if not violations:
            return None
        return "ef_pose_out_of_workspace: " + ", ".join(violations)

    def _base_frame_collision_mesh(self, robot_name, kind, source_mesh, base_T):
        """world-frame collision mesh를 로봇 base frame으로 옮긴 결과를 캐시해 재사용한다.

        같은 source mesh(id 동일)와 같은 robot이면 이전에 만든 변환 결과를 그대로 반환한다.
        target마다 새 객체를 만들면 backend의 BVH 캐시(mesh id 기반)가 매번 miss해서 100k
        규모 mesh의 BVH를 매번 다시 쌓느라 setup이 크게 느려진다. (robot, kind)별로 가장
        최근 것 하나만 보관하고, source mesh가 바뀌면(배관 회전 등) 그때만 다시 변환한다.

        Args:
            robot_name: 캐시 구분용 로봇 이름.
            kind: "obstacle" / "positioner" 등 mesh 종류 구분자.
            source_mesh: world 좌표계 원본 mesh.
            base_T: 로봇 base의 world transform. inv(base_T)로 base frame에 옮긴다.
        """
        if source_mesh is None:
            return None
        cache = getattr(self, '_base_frame_collision_cache', None)
        if cache is None:
            cache = {}
            self._base_frame_collision_cache = cache
        key = (robot_name, kind)
        entry = cache.get(key)
        if entry is not None and entry[0] is source_mesh:
            return entry[1]
        transformed = copy.deepcopy(source_mesh)
        transformed.transform(np.linalg.inv(np.asarray(base_T, dtype=float)))
        # source_mesh를 함께 보관해 id 재사용(GC 후 다른 객체가 같은 id)을 막고,
        # 다음 호출에서 동일 객체 여부를 정확히 판별한다.
        cache[key] = (source_mesh, transformed)
        return transformed

    def _build_positioner_collision_mesh(self):
        """포지셔너(chuck/column 등) 하드웨어의 현재 world pose 기준 collision mesh를 만든다.

        지금까지 path planning의 collision scene에는 배관(obstacle_mesh)과 로봇만 들어가고
        포지셔너 URDF는 빠져 있었다(`_planner_has_positioner_collision`가 항상 False). 그래서
        회전된 second-group target(포지셔너 쪽으로 더 붙는 자세)에 로봇이 도달했을 때 실제로는
        EF가 포지셔너 하드웨어와 겹쳐도 이 체크로는 못 잡았다. 포지셔너 자체(그 중에서도
        r-joint 이후 파이프만 도는 부분 제외)는 실제로 움직이지 않으므로, second-group 가상
        회전과 무관하게 항상 "현재" pose 그대로 하나의 static mesh로 추가한다.

        한 배치 안에서 target마다(=이 함수 호출마다) 새 mesh 객체를 만들면 backend의
        static mesh identity 캐시(`_configured_mesh_key`)가 매번 깨져 collision scene을
        불필요하게 다시 빌드하게 된다. 포지셔너는 planning 도중에는 움직이지 않으므로
        캐시해서 같은 객체를 재사용하고, 실제로 포지셔너가 움직였을 때만 무효화한다.
        """
        cached = getattr(self, "_positioner_collision_mesh_cache", None)
        if cached is not None:
            return cached
        mesh = self._build_positioner_collision_mesh_uncached()
        self._positioner_collision_mesh_cache = mesh
        return mesh

    def _build_positioner_collision_mesh_uncached(self):
        model = self._positioner_robot_model()
        if model is None:
            return None
        actors = [
            actor for actor in (getattr(model, "actors", None) or [])
            if actor is not None
            and hasattr(actor, "vertices")
            and hasattr(actor, "cells")
            and len(getattr(actor, "cells", [])) > 0
        ]
        if not actors:
            return None
        all_verts = []
        all_faces = []
        offset = 0
        for actor in actors:
            verts = np.asarray(actor.vertices, dtype=float)
            faces = np.asarray(actor.cells, dtype=np.int32)
            if verts.size == 0 or faces.size == 0:
                continue
            all_verts.append(verts)
            all_faces.append(faces + offset)
            offset += len(verts)
        if not all_verts:
            return None
        mesh = _o3d.geometry.TriangleMesh()
        mesh.vertices = _o3d.utility.Vector3dVector(np.vstack(all_verts))
        mesh.triangles = _o3d.utility.Vector3iVector(np.vstack(all_faces))
        mesh.compute_vertex_normals()
        return mesh

    def _build_spool_collision_mesh(self):
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

    def _path_planning_fixed_joint_options(
        self,
        request_data,
        *,
        robot_name,
        robot_backend_model,
        frame_name,
        start_q,
        target_world_pose,
        nearest_point=None,
    ):
        cfg = (self._config.get("path_planning", {}) or {})
        raw_joints = request_data.get("fixed_joints", cfg.get("fixed_joints"))
        raw_indices = request_data.get("fixed_joint_indices", cfg.get("fixed_joint_indices"))
        if raw_joints is None and raw_indices is None:
            return {}

        fixed_q, track_indices, track_values = self._inspection_track_fixed_q(
            robot_name,
            robot_backend_model,
            frame_name,
            start_q,
            target_world_pose,
            nearest_point,
        )
        if not track_indices:
            return {}

        linear_idx = track_indices[0]
        carriage_idx = track_indices[1] if len(track_indices) > 1 else None
        selected = set()

        def add_token(token):
            text = str(token).strip().lower()
            if text in {"linear", "linear_track", "track_1", "joint_1"}:
                selected.add(linear_idx)
            elif text in {"carriage", "carriage_track", "track_2", "joint_2"} and carriage_idx is not None:
                selected.add(carriage_idx)

        if isinstance(raw_joints, dict):
            for key in raw_joints.keys():
                add_token(key)
        elif raw_joints is True:
            selected.update(idx for idx in (linear_idx, carriage_idx) if idx is not None)
        elif raw_joints is not None:
            items = [raw_joints] if isinstance(raw_joints, (str, bytes)) else list(raw_joints)
            for item in items:
                add_token(item)

        if raw_indices is not None:
            raw_index_items = [raw_indices] if np.isscalar(raw_indices) else list(raw_indices)
            indices = [int(v) for v in raw_index_items]
            one_based = bool(indices) and all(v in (1, 2) for v in indices) and 0 not in indices
            for raw_idx in indices:
                if one_based:
                    if raw_idx == 1:
                        selected.add(linear_idx)
                    elif raw_idx == 2 and carriage_idx is not None:
                        selected.add(carriage_idx)
                    continue
                if raw_idx == linear_idx or raw_idx == 0:
                    selected.add(linear_idx)
                elif carriage_idx is not None and (raw_idx == carriage_idx or raw_idx == 1):
                    selected.add(carriage_idx)

        selected_indices = [idx for idx in track_indices if idx in selected]
        selected_values = [float(fixed_q[idx]) for idx in selected_indices]
        return {
            "fixed_joint_indices": selected_indices,
            "fixed_joint_values": selected_values,
        } if selected_indices else {}

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
        fixed_joints=None,
        fixed_joint_indices=None,
        fixed_joint_values=None,
    ):
        setup_t0 = time.perf_counter()
        mn = np.minimum(start[:3], goal[:3])
        mx = np.maximum(start[:3], goal[:3])
        offset_cfg = (self._config.get("path_planning", {}) or {}).get(
            "workspace_check_offset", {}
        ) or {}
        pad = np.array(
            [
                float(offset_cfg.get("x", 0.5)),
                float(offset_cfg.get("y", 0.5)),
                float(offset_cfg.get("z", 0.5)),
            ],
            dtype=float,
        )
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
        # q-space planner는 원래 world 좌표계 제한이 없다(joint limit만 봄). 옵트인
        # 설정(path_planning.enable_workspace_check)이 켜져 있으면, IK가 풀리는 대상
        # frame(로봇의 target link)의 FK world position을 위 bounds로 제한한다.
        enable_workspace_check = bool(
            (self._config.get("path_planning", {}) or {}).get("enable_workspace_check", False)
        )
        workspace_check_frame_name = (
            self._robot_target_link_name(robot_name)
            if enable_workspace_check and robot_name is not None
            else None
        )
        planner.configure(
            bounds=bounds,
            step_size=float(step_size),
            max_iter=int(max_iter),
            robotics_backend=backend,
            robotics_robot_name=robot_name,
            workspace_check_frame_name=workspace_check_frame_name,
            fixed_joints=fixed_joints,
            fixed_joint_indices=fixed_joint_indices,
            fixed_joint_values=fixed_joint_values,
            joint_names=self._robot_joint_names(robot_name) if robot_name is not None else None,
        )
        if OMPLPlannerBase is not None and isinstance(planner, OMPLPlannerBase):
            planner.configure_ompl({
                **planner.ompl_config,
                "algorithm": planner.algorithm,
                "step_size": float(step_size),
            })
        if timings is not None:
            timings["planner_bounds_config"] = time.perf_counter() - setup_t0
        collision_obstacle_mesh = obstacle_mesh
        collision_positioner_mesh = self._build_positioner_collision_mesh()
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
                    # base_T!=identity(레일 마운트 등)면 obstacle을 로봇 base frame으로 옮겨야
                    # 한다. 이 deepcopy+transform 결과와 그로부터 만드는 BVH는 target마다 다시
                    # 하면(매번 새 객체라 backend BVH 캐시가 계속 miss) 100k mesh 기준 setup이
                    # 수 초씩 걸린다. source mesh(id)와 robot이 같으면 변환 결과를 재사용해
                    # 같은 객체를 넘겨야 backend BVH 캐시가 hit한다.
                    transform_t0 = time.perf_counter()
                    collision_obstacle_mesh = self._base_frame_collision_mesh(
                        robot_name, "obstacle", obstacle_mesh, base_T)
                    collision_positioner_mesh = self._base_frame_collision_mesh(
                        robot_name, "positioner", collision_positioner_mesh, base_T)
                    if timings is not None:
                        timings["planner_obstacle_base_transform"] = time.perf_counter() - transform_t0
                    self.__console.debug(
                        "inspection path: transformed obstacle mesh into robot base frame for collision | "
                        f"robot={robot_name}, base_t={np.round(base_T[:3, 3], 5).tolist()}")
        obstacle_t0 = time.perf_counter()
        # 배관 + 포지셔너 mesh를 한 번에 등록한다(개별 add는 configure_collision을 두 번
        # 호출해 BVH 캐시 key를 계속 어긋나게 만든다).
        planner.add_collision_objects([collision_obstacle_mesh, collision_positioner_mesh])
        if timings is not None:
            timings["planner_obstacle_bvh"] = time.perf_counter() - obstacle_t0
        self._log_robot_collision_targets(robot_name, planner)
        # Unconditional (not deduplicated like _log_robot_collision_targets)
        # fingerprint of the static meshes actually registered for THIS
        # call, so two calls that report different collision outcomes for
        # the same q on the same robot can be diff'd directly instead of
        # assumed identical - centroid/vertex-count alone would catch a
        # stale-cache or wrong-frame bug (see _base_frame_collision_mesh).
        def _mesh_fingerprint(mesh):
            if mesh is None or not mesh.has_triangles():
                return None
            verts = np.asarray(mesh.vertices, dtype=float)
            return {
                "n_vertices": int(verts.shape[0]),
                "centroid": np.round(verts.mean(axis=0), 5).tolist(),
                "bbox_min": np.round(verts.min(axis=0), 5).tolist(),
                "bbox_max": np.round(verts.max(axis=0), 5).tolist(),
            }
        self.__console.debug(
            "inspection path: collision scene fingerprint | "
            f"robot={robot_name}, obstacle={_mesh_fingerprint(collision_obstacle_mesh)}, "
            f"positioner={_mesh_fingerprint(collision_positioner_mesh)}")
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
        self.__console.debug(
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

    def _seed_track_joint_q_for_world_axis(
        self,
        robot_name,
        robot_backend_model,
        frame_name,
        q,
        *,
        joint_keyword,
        fallback_index,
        world_axis,
        target_world_value,
        label,
    ):
        """track joint 값을 world axis 목표 위치에 가깝게 FK 민감도로 옮긴다."""
        backend = getattr(self, "_robotics_backend", None)
        if backend is None:
            return q
        joint_names = self._robot_joint_names(robot_name, robot_backend_model)
        track_idx = next((i for i, name in enumerate(joint_names) if joint_keyword in str(name)), None)
        if track_idx is None and fallback_index is not None and fallback_index < len(joint_names):
            track_idx = int(fallback_index)
        if track_idx is None or track_idx >= len(q):
            return q
        q = np.asarray(q, dtype=float).copy()
        axis = int(world_axis)
        try:
            T0 = backend.frame_world_T(robot_name, q, frame_name)
            probe_q = q.copy()
            delta = 0.05
            probe_q[track_idx] += delta
            T1 = backend.frame_world_T(robot_name, probe_q, frame_name)
        except Exception as exc:
            self.__console.debug(f"{label} track seeding skipped: robot={robot_name}, error={exc}")
            return q
        if T0 is None or T1 is None:
            return q
        sensitivity = (float(T1[axis, 3]) - float(T0[axis, 3])) / delta
        if abs(sensitivity) < 1e-6:
            return q
        track_delta = (float(target_world_value) - float(T0[axis, 3])) / sensitivity
        new_value = float(q[track_idx]) + track_delta
        try:
            lo, hi, _ = backend.joint_limits_for_metric(robot_name, normalize=False)
            new_value = float(np.clip(new_value, lo[track_idx], hi[track_idx]))
        except Exception:
            pass
        self.__console.debug(
            f"{label} track seeded: robot={robot_name}, "
            f"track_idx={track_idx}, old={q[track_idx]:.4f}, new={new_value:.4f}, "
            f"target_world_axis={axis}, target_world_value={target_world_value:.4f}")
        q[track_idx] = new_value
        return q

    def _seed_linear_track_q_for_world_x(self, robot_name, robot_backend_model, frame_name, q, target_world_x):
        """linear track joint의 IK 초기값을 target world x 위치에 가깝게 미리 옮긴다.

        Args:
            robot_name: backend에 등록된 로봇 이름.
            robot_backend_model: pinocchio 모델(joint 이름 조회용).
            frame_name: IK를 풀 대상 frame 이름(FK 기준점).
            q: 원래 IK 초기값(start_q).
            target_world_x: 검사 목표 pose의 world x 좌표.

        Returns:
            linear track 성분만 조정된 새 q. track joint가 없거나 FK로 축 민감도를
            구할 수 없으면 원래 q를 그대로 반환한다.

        계산 과정:
            현재 q에서의 frame world x와, track joint를 살짝 움직였을 때의 world x
            변화량(민감도)을 FK로 직접 구해서, 그 비율로 필요한 track 이동량을 역산한다.
            축 방향/부호를 가정하지 않고 실제 FK로 구하므로 URDF의 joint 축 방향이
            바뀌어도 안전하다. IK를 대신하는 게 아니라 초기 추정값만 더 가깝게 준다 -
            이후 IK는 전체 joint를 그대로 계속 풀어서 조정한다.
        """
        return self._seed_track_joint_q_for_world_axis(
            robot_name,
            robot_backend_model,
            frame_name,
            q,
            joint_keyword="linear_track",
            fallback_index=0,
            world_axis=0,
            target_world_value=target_world_x,
            label="linear",
        )

    def _seed_carriage_track_q_for_world_y(self, robot_name, robot_backend_model, frame_name, q, target_world_y):
        return self._seed_track_joint_q_for_world_axis(
            robot_name,
            robot_backend_model,
            frame_name,
            q,
            joint_keyword="carriage",
            fallback_index=1,
            world_axis=1,
            target_world_value=target_world_y,
            label="carriage",
        )

    def _inspection_track_fixed_q(self, robot_name, robot_backend_model, frame_name, start_q, target_world_pose, nearest_point):
        q = np.asarray(start_q, dtype=float).copy()
        target_world_pose = np.asarray(target_world_pose, dtype=float).reshape(-1)
        nearest = None if nearest_point is None else np.asarray(nearest_point, dtype=float).reshape(-1)
        q = self._seed_linear_track_q_for_world_x(
            robot_name, robot_backend_model, frame_name, q, float(target_world_pose[0]))
        carriage_y = float(nearest[1]) if nearest is not None and nearest.size >= 2 else float(target_world_pose[1])
        q = self._seed_carriage_track_q_for_world_y(
            robot_name, robot_backend_model, frame_name, q, carriage_y)
        joint_names = self._robot_joint_names(robot_name, robot_backend_model)
        linear_idx = next((i for i, name in enumerate(joint_names) if "linear_track" in str(name)), 0)
        carriage_idx = next((i for i, name in enumerate(joint_names) if "carriage" in str(name)), 1)
        indices = [idx for idx in (linear_idx, carriage_idx) if idx is not None and idx < q.size]
        values = [float(q[idx]) for idx in indices]
        return q, indices, values

    def _zero_q_keep_linear_track(self, robot_name, q):
        """positioner 회전 직전 안전 자세: linear_track만 유지하고 나머지 joint는 0으로.

        Args:
            robot_name: backend에 등록된 로봇 이름.
            q: 유지할 linear_track 값을 담고 있는 원본 q(first group 마지막 자세).

        Returns:
            나머지 joint는 0, linear_track 성분만 원래 값을 유지한 새 q.
            backend/joint 이름을 못 찾으면 원래 q를 그대로 반환한다.
        """
        backend = getattr(self, "_robotics_backend", None)
        if backend is None:
            return q
        robot_backend_model = backend.robot_model(robot_name) if robot_name is not None else None
        joint_names = self._robot_joint_names(robot_name, robot_backend_model)
        q = np.asarray(q, dtype=float)
        zeroed = np.zeros_like(q)
        for i, name in enumerate(joint_names[:len(q)]):
            if "linear_track" in name:
                zeroed[i] = q[i]
        return zeroed

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
        cfg.setdefault("max_iter", path_cfg.get("ik_max_iter", 3000))
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
                    Path(self._config.get("debug_dir", "debug")) / "inspection_ik"
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
            log_fn = self.__console.warning if saved.get("plot") else self.__console.debug
            log_fn(
                f"inspection IK experiment saved: robot={robot_name}, csv={saved['csv']}, "
                f"meta={saved['meta']}, plot={saved.get('plot')}")
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
        # ef pose가 고정 workspace 박스 밖이면 IK 확인도 하지 않고 실패로 남긴다
        # (path planning과 같은 기준으로 걸러 결과가 어긋나지 않게 한다).
        workspace_reject = self._ef_pose_workspace_reject_reason(goal[:3])
        if workspace_reject is not None:
            raise RuntimeError(workspace_reject)
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

        # obstacle_mesh를 실제 collision scene에 반영한다. 이전에는 이 인자를 그냥
        # 버렸는데, 그러면 collision 판정이 이 robot handle에 마지막으로 설정된
        # (다른 요청 때의, 혹은 회전 안 된) 배관 기준으로 이뤄져 IK check 결과가
        # 실제 요청 mesh와 어긋날 수 있었다.
        if obstacle_mesh is not None:
            stage_t0 = time.perf_counter()
            sample_resolution = float(self._config.get("planner_collision_sample_resolution", 0.05))
            # obstacle_mesh(월드 좌표계)를 그대로 등록하면 안 된다. pinocchio 모델은
            # base_T만큼 world에서 옮겨 붙은 로봇의 "base 기준" 좌표계로 kinematics를
            # 풀기 때문에(_configure_inspection_planner의 path-planning 쪽과 동일 로직),
            # base_T가 identity가 아닌 로봇(예: rail에 offset으로 마운트된 로봇)은 여기서
            # world-frame mesh를 그대로 쓰면 실제와 다른 위치에서 collision을 검사하게 되어
            # "IK check는 collision 없음" / "path planning은 start_collision" 처럼 두 체크가
            # 어긋나는 원인이 됐다.
            collision_obstacle_mesh = obstacle_mesh
            # path planning(_configure_inspection_planner)과 마찬가지로 포지셔너 하드웨어도
            # static obstacle로 같이 등록한다 - 안 그러면 이 IK-only 체크는 EF가 포지셔너와
            # 겹쳐도 잡아내지 못해 실제 path planning의 positioner collision과 어긋난다.
            collision_positioner_mesh = self._build_positioner_collision_mesh()
            model = self._find_robot(robot_name)
            base_T = np.asarray(getattr(model, "_base_T", np.eye(4)), dtype=float) if model is not None else np.eye(4)
            if base_T.shape == (4, 4) and not np.allclose(base_T, np.eye(4)):
                # base frame 변환 결과를 캐시해 같은 객체를 재사용한다(backend BVH 캐시 hit).
                collision_obstacle_mesh = self._base_frame_collision_mesh(
                    robot_name, "obstacle", obstacle_mesh, base_T)
                collision_positioner_mesh = self._base_frame_collision_mesh(
                    robot_name, "positioner", collision_positioner_mesh, base_T)
            static_meshes = [collision_obstacle_mesh]
            if collision_positioner_mesh is not None:
                static_meshes.append(collision_positioner_mesh)
            backend.configure_collision(
                robot_name, static_meshes=static_meshes, sample_resolution=sample_resolution)
            timings["collision_scene_setup"] = time.perf_counter() - stage_t0

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
        collision_pairs = ik_result.get("collision_pairs") or []
        collision_pairs_text = ", ".join(f"{a}<->{b}" for a, b in collision_pairs) or "-"
        level(
            "inspection IK result: "
            f"robot={robot_name}, success={bool(ik_result.get('success'))}, "
            f"fallback={bool(result.get('ik_fallback'))}, "
            f"position_error={float(ik_result.get('position_error', float('inf'))):.5f}m, "
            f"orientation_error={float(ik_result.get('orientation_error', float('inf'))):.5f}rad, "
            f"collision={bool(ik_result.get('collision'))}, "
            f"collision_pairs=[{collision_pairs_text}], "
            f"iterations={ik_result.get('iterations', '-')}")
        # start_q/target_q는 위 "inspection IK check input" debug 로그에 이미 있고,
        # 여기서 또 INFO로 찍으면 매 로봇/포즈마다 같은 값이 중복으로 콘솔을 채운다.
        self.__console.debug(
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

    def _plan_inspection_path_for_robot(
        self, request_data, robot_name, target_pose, obstacle_mesh=None, context_label=None
    ):
        """검사 목표 pose까지 한 로봇의 q-space path를 계산한다.

        Args:
            request_data: planner 이름, step size, max_iter, IK 옵션, timeout을 포함한 요청 dict.
            robot_name: 경로를 계산할 로봇 이름.
            target_pose: 목표 TCP pose. 4x4 transform, 6D pose, 3D point를 허용한다.
            obstacle_mesh: collision scene에 넣을 배관 mesh. None이면 현재 로드된 배관 mesh를 사용한다.
            context_label: 로그 구분용 "몇 번째 포인트/자세" 식별 문자열(예: "Point 2 - Inspection pose 2:DDA").
                None이면 robot_name만 쓴다. planner의 탐색/타이밍 로그 맨 앞에 붙는다.

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
        label = context_label or robot_name
        stage_t0 = time.perf_counter()
        # 1) 현재 배관을 collision obstacle mesh로 준비한다.
        if obstacle_mesh is None:
            obstacle_mesh = self._current_spool_collision_mesh()
        if obstacle_mesh is None:
            raise RuntimeError("loaded pipe is not available")
        timings["obstacle_mesh"] = time.perf_counter() - stage_t0

        # 2) planner 입력인 시작 TCP pose와 목표 pose를 world 좌표계 기준으로 정리한다.
        #    workspace bound는 실제 planning이 출발하는 start_q(override 포함, 예: 여러
        #    group을 이어 계획할 때 이전 group의 마지막 q)의 FK world position을 기준으로
        #    잡아야 한다. _get_robot_tcp_pose()는 "현재 뷰어 상 로봇 위치"라서, override로
        #    start_q가 바뀌는 경우 실제 start_q와 어긋나 start_out_of_workspace가 잘못
        #    발생하는 버그가 있었다 - 그래서 start_q를 먼저 구하고 그 FK로 start pose를 만든다.
        stage_t0 = time.perf_counter()
        robot_model = self._find_robot(robot_name)
        if robot_model is None:
            raise RuntimeError(f"robot model not found: {robot_name}")
        backend = getattr(self, "_robotics_backend", None)
        robot_backend_model = (
            backend.robot_model(robot_name) if (backend is not None and robot_name is not None) else None
        )
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

        start = self._T_to_pose(self._pin_target_world_T(robot_model, robot_backend_model, start_q, robot_name))
        if start is None:
            raise RuntimeError(f"robot TCP not found: {robot_name}")
        goal = np.zeros(6, dtype=float)
        target_arr = np.asarray(target_pose, dtype=float)
        if target_arr.shape == (4, 4):
            goal[:3] = target_arr[:3, 3]
        else:
            flat_target = target_arr.reshape(-1)
            goal[:min(6, flat_target.size)] = flat_target[:min(6, flat_target.size)]
        # ef pose(목표 지점)가 설정된 고정 workspace 박스 밖이면 계획하지 않고 실패 처리한다.
        workspace_reject = self._ef_pose_workspace_reject_reason(goal[:3])
        if workspace_reject is not None:
            self.__console.warning(
                f"inspection path skipped: [{label}] robot={robot_name}, {workspace_reject}")
            raise RuntimeError(f"planning failed for target: {workspace_reject}")
        timings["target_setup"] = time.perf_counter() - stage_t0

        # 3) 요청된 q-space planner를 만들고 robotics backend collision scene을 설정한다.
        stage_t0 = time.perf_counter()
        planner_name = self._inspection_q_space_planner_name(request_data.get("planner", "rrt_connect"))
        planner = self._load_path_planner(planner_name)
        planner.debug_context = label
        frame_name = self._robot_target_link_name(robot_name)
        nearest_point = request_data.get("_inspection_target_point")
        fixed_joint_options = self._path_planning_fixed_joint_options(
            request_data,
            robot_name=robot_name,
            robot_backend_model=robot_backend_model,
            frame_name=frame_name,
            start_q=start_q,
            target_world_pose=goal,
            nearest_point=nearest_point,
        )
        self._configure_inspection_planner(
            planner,
            obstacle_mesh,
            start,
            goal,
            float(request_data.get("step_size", 0.08)),
            int(request_data.get("max_iter", 3000)),
            robot_name=robot_name,
            pin_cache=(getattr(self, "_pinocchio_robot_collision_cache", {}) or {}).get(robot_name),
            timings=timings,
            **fixed_joint_options)
        if not getattr(planner, "_has_robot_q_space_model", lambda: False)():
            raise RuntimeError("robot q-space model is not configured")
        timings["planner_setup"] = time.perf_counter() - stage_t0
        self.__console.debug(
            "inspection path planner setup timing: "
            f"robot={robot_name}, total={timings.get('planner_setup', 0.0):.3f}s, "
            f"bounds={timings.get('planner_bounds_config', 0.0):.3f}s, "
            f"robotics_model={timings.get('planner_robotics_model', 0.0):.3f}s, "
            f"obstacle_transform={timings.get('planner_obstacle_base_transform', 0.0):.3f}s, "
            f"obstacle_bvh={timings.get('planner_obstacle_bvh', 0.0):.3f}s")

        # 4) IK solve, q-space planning, collision verification은 robotics base에 위임한다.
        #    robot_backend_model/start_q(override 포함)는 위 2)에서 이미 구했으므로 재사용한다.
        stage_t0 = time.perf_counter()
        # path planning에서만: IK solver가 goal_q를 풀 때 쓰는 초기 추정값(q_init)만
        # linear track에 한해 target의 world x 위치 근처로 미리 옮겨준다. 로봇의 실제
        # 현재 위치(q_start, path planning이 실제로 출발하는 지점)는 절대 안 건드린다 -
        # start_q는 그대로 두고 IK 요청에만 별도 변수(ik_start_q)로 넘긴다.
        # check_ef_pose_ik는 그대로 둔다.
        ik_start_q = self._seed_linear_track_q_for_world_x(
            robot_name, robot_backend_model, frame_name,
            start_q, float(goal[0]))
        if fixed_joint_options.get("fixed_joint_indices"):
            for idx, value in zip(
                fixed_joint_options.get("fixed_joint_indices", []),
                fixed_joint_options.get("fixed_joint_values", []),
            ):
                if 0 <= int(idx) < ik_start_q.shape[0]:
                    ik_start_q[int(idx)] = float(value)
        self.__console.debug(
            f"inspection path IK input: [{label}] robot={robot_name}\n"
            f"  start_q    = {np.round(start_q, 5).tolist()}\n"
            f"  ik_start_q = {np.round(ik_start_q, 5).tolist()}\n"
            f"  target_world_pose = {np.round(goal, 5).tolist()}")
        planning_timeout = self._planner_timeout(request_data, planner_name)
        service = getattr(self, "_inspection_planning_base", None)
        if service is None:
            raise RuntimeError("inspection planning base is not initialized")
        plan = service.plan_q_path_for_robot(
            planner=planner,
            ik_request=InspectionIKRequest(
                robot_name      =robot_name,
                target_pose     =target_pose,
                start_tcp_pose  =start,
                start_q         =ik_start_q,
                frame_name      =frame_name,
                joint_names     =self._robot_joint_names(robot_name, robot_backend_model),
                planner_name    =planner_name,
                ik_config       =self._inspection_ik_config(),
                ik_solver       =request_data.get("ik_solver"),
                ik_normalize    =request_data.get("ik_normalize"),
            ),
            q_start=start_q,
            planning_timeout=planning_timeout,
            lock_linear_track=bool(request_data.get("lock_linear_track", False)),
            console=self.__console,
        )
        if fixed_joint_options.get("fixed_joint_indices"):
            plan["fixed_joint_indices"] = list(fixed_joint_options.get("fixed_joint_indices", []))
            plan["fixed_joint_values"] = list(fixed_joint_options.get("fixed_joint_values", []))
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
        # service(plan_q_path_for_robot)가 반환하는 timing에도 "target_setup"/"total" 키가
        # 있는데, 여기서 재는 target_setup(1~2단계: obstacle/goal 준비)과 total(전체)과는
        # 다른 걸 재는 값이라 그대로 update()하면 이름이 겹쳐 덮어써진다. 겹치는 두 키는
        # 무시하고(ik_ 접두어로만 갖고 온다) 나머지 IK/planning/verify 단계 키만 그대로 합친다.
        service_timing = plan.get("timing", {}) or {}
        timings["ik_target_setup"] = service_timing.get("target_setup", 0.0)
        for key in ("ik", "ik_result_check", "planning", "collision_verification"):
            if key in service_timing:
                timings[key] = service_timing[key]
        plan["convergence_csv"] = getattr(planner, "last_convergence_csv", None)
        plan["convergence_plot"] = getattr(planner, "last_convergence_plot", None)
        plan["exploration_csv"] = getattr(planner, "last_exploration_csv", None)
        plan["exploration_plot"] = getattr(planner, "last_exploration_plot", None)
        goal_q = np.asarray(plan["goal_q"], dtype=float)
        q_path = [np.asarray(q, dtype=float) for q in plan.get("q_path", [])]
        optimizer_name = request_data.get("optimizer")
        optimize_path = self._path_optimization_requested(request_data)
        plan["optimizer"] = optimizer_name
        plan["optimization_enabled"] = optimize_path
        # Scene context (positioner attitude this target/path was actually
        # collision-checked against) an optimizer's own debug/playback output
        # (e.g. stomp.py's _save_playback_trajectory) needs to record
        # alongside its saved q_path - otherwise a saved run has no way to
        # know the pipe/positioner should be shown rotated during playback.
        # Plain attributes on planner (not a new optimize() parameter) so
        # this doesn't touch OptimizerBase's signature for every optimizer.
        planner.debug_positioner_r_deg = float(request_data.get("positioner_r_deg", 0.0) or 0.0)
        planner.debug_obstacle_rotated = request_data.get("obstacle_rotation_T") is not None
        if optimize_path and optimizer_name and q_path:
            stage_t0 = time.perf_counter()
            try:
                optimized_q_path, optimization_status = self._apply_path_optimizer(
                    optimizer_name,
                    q_path,
                    planner,
                )
                if optimized_q_path:
                    q_path = optimized_q_path
                    plan["q_path"] = q_path
                    plan["waypoints"] = len(q_path)
                    plan["optimization_status"] = optimization_status
                    verification = planner.verify_path(q_path)
                    if (
                        verification.get("colliding_edges", 0) != 0
                        or verification.get("colliding_waypoints", 0) != 0
                    ):
                        plan["collision_preview"] = True
                        plan["collision_preview_reason"] = (
                            plan.get("collision_preview_reason")
                            or "optimized_path_collision"
                        )
                    plan["verification"] = verification
                else:
                    plan["optimization_status"] = "optimizer_empty_path"
                    plan["collision_preview"] = True
                    plan["collision_preview_reason"] = (
                        plan.get("collision_preview_reason")
                        or "optimizer_empty_path"
                    )
            except Exception as opt_exc:
                plan["optimization_status"] = "optimizer_failed"
                plan["optimization_error"] = str(opt_exc)
                plan["collision_preview"] = True
                plan["collision_preview_reason"] = (
                    plan.get("collision_preview_reason")
                    or f"optimizer_failed: {opt_exc}"
                )
            timings["optimization"] = time.perf_counter() - stage_t0
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
        # self.__console.info(
        #     "inspection path IK q: "
        #     f"robot={robot_name}, "
        #     f"start_q={np.round(start_q, 6).tolist()}, "
        #     f"target_q={np.round(goal_q, 6).tolist()}")
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
            # q_path가 1점(planner가 경로를 못 찾아 start_q로만 fallback한 경우 등)이면
            # 여기서 걸린다. "TCP path 변환 실패"는 증상일 뿐 원인이 아니므로, planner가
            # 이미 파악한 실패 사유(planning_error/collision_preview_reason/ik_failure)를
            # 그대로 실어 로그에서 바로 원인을 알 수 있게 한다.
            reason = (
                plan.get("planning_error")
                or plan.get("collision_preview_reason")
                or (plan.get("ik_failure") or {}).get("type")
                or f"q_path has only {len(q_path)} point(s)"
            )
            if "final_verification_failed" in str(reason):
                # Distinguish "the target pose itself collides" (a waypoint
                # collision - no planner/resolution choice will fix this,
                # the pose needs more clearance from the pipe) from "the
                # path grazes the pipe between two otherwise-valid poses"
                # (an edge collision - OMPL's own internal motion-validator
                # resolution missed it; lowering normalized_resolution so
                # OMPL checks edges more finely during search itself may
                # help). Both get caught by the same post-solve verify_path()
                # re-check, so the raw status string alone can't tell them
                # apart. Prefer planner.last_verification - the verify_path()
                # result from *inside* OMPLPlannerBase, on the actual path
                # that collided - over plan["verification"], which by this
                # point is a re-check of the single-point [q_start] fallback
                # path (the real q_path got discarded) and would trivially
                # report 0/0 with nothing meaningful to show.
                verification = getattr(planner, "last_verification", None) or plan.get("verification") or {}
                n_waypoints = int(verification.get("colliding_waypoints", 0))
                n_edges = int(verification.get("colliding_edges", 0))
                waypoint_indices = [w["waypoint"] for w in verification.get("waypoint_collisions", [])]
                edge_indices = [e["edge"] for e in verification.get("edge_collisions", [])]
                reason = (
                    f"{reason}(colliding_waypoints={n_waypoints}{waypoint_indices or ''}, "
                    f"colliding_edges={n_edges}{edge_indices or ''})"
                )
            if "start_collision" in str(reason):
                # start_q(현재/이어받은 로봇 위치) 자체가 충돌이라는 뜻이므로, 화면에서
                # 바로 확인할 수 있게 그 pose에 IK 실패 마커와 같은 표시를 남긴다.
                try:
                    collision_T = self._pin_target_world_T(robot_model, robot_backend_model, start_q, robot_name)
                    self._show_ik_failure_reached_pose(robot_name, collision_T, None)
                except Exception as marker_exc:
                    self.__console.debug(
                        f"start_collision marker skipped: robot={robot_name}, error={marker_exc}")
                # 어느 geometry pair가 충돌했는지(예: 이전 group에서 이어받은 q가 second
                # group의 가상 회전된 배관 mesh와 겹치는지) 원인 문자열에 같이 실어 둔다.
                collision_pairs = getattr(planner, "last_collision_pairs", None)
                if collision_pairs:
                    reason = f"{reason}(pairs={_label_collision_pairs(collision_pairs)})"
            elif "goal_collision" in str(reason):
                # goal_q(IK가 이 target_pose를 풀어서 도달한 자세) 자체가 충돌이라는 뜻이므로,
                # 화면에서 바로 확인할 수 있게 그 pose에 IK 실패 마커와 같은 표시를 남긴다.
                try:
                    collision_T = self._pin_target_world_T(robot_model, robot_backend_model, goal_q, robot_name)
                    self._show_ik_failure_reached_pose(robot_name, collision_T, target_pose)
                except Exception as marker_exc:
                    self.__console.debug(
                        f"goal_collision marker skipped: robot={robot_name}, error={marker_exc}")
                collision_pairs = getattr(planner, "last_collision_pairs", None)
                if collision_pairs:
                    reason = f"{reason}(pairs={_label_collision_pairs(collision_pairs)})"
            else:
                # Any other planner status (e.g. final_verification_failed) that
                # also populated last_collision_pairs (see rrt_star.py/
                # OMPLPlannerBase) - surface it the same way instead of only
                # special-casing the start/goal pre-check reasons.
                collision_pairs = getattr(planner, "last_collision_pairs", None)
                if collision_pairs:
                    reason = f"{reason}(pairs={_label_collision_pairs(collision_pairs)})"
            failure_exc = RuntimeError(f"planning failed for target: {reason}")
            # This raises out of _plan_inspection_path_for_robot entirely, so
            # plan_single_target's except-block (path_planning_service.py)
            # never sees the local `plan`/`planner` here and would otherwise
            # return a bare {"status": "failed", "message": ...} with no
            # iterations/solve_time/etc - attach them to the exception itself
            # so that data survives (see plan_single_target's except-block,
            # which reads exc.planner_stats/exc.q_path).
            failure_exc.planner_stats = dict(getattr(planner, "last_ompl_stats", {}) or {})
            # The actual (colliding) q_path, if there is one - prefer the
            # local `q_path` (whatever this function last computed - the
            # post-optimizer result if an optimizer ran, works for *any*
            # planner including direct_path+stomp/trajopt/...) and only fall
            # back to planner.last_failed_q_path (OMPL-only - see its
            # docstring) for the case where q_path itself is empty (e.g. the
            # planner's own generate() returned nothing at all). Without
            # this, a failed plan_single_target result always had
            # "q_path": [] with no way to see *what* was attempted, only the
            # collision_pairs summary string - can't play it back or tell
            # which waypoint/edge index was the actual failure.
            failed_q_path = q_path if len(q_path) else (getattr(planner, "last_failed_q_path", None) or [])
            failure_exc.q_path = [np.asarray(q, dtype=float).tolist() for q in failed_q_path]
            # Which waypoint/edge index (into exc.q_path above) actually
            # collided - see verify_path()'s waypoint_collisions/
            # edge_collisions (plannerbase.py) for the shape. Only OMPLPlanner
            # Base's own internal final-verification failure populates this
            # (last_verification) with real indices; start_collision/
            # goal_collision are single-point failures with no index to give.
            failure_exc.verification = dict(getattr(planner, "last_verification", None) or {})
            raise failure_exc
        timings["path_conversion"] = time.perf_counter() - stage_t0
        timings["total"] = time.perf_counter() - total_t0
        # collision_preview가 켜졌는데 이유가 없어 보이는 경우가 잦았던 건 ik_fallback(IK
        # 미수렴) 때문에 켜진 걸 여기 로그에는 안 보여줘서였다. ik_fallback을 같이 남기고,
        # preview/실패 상태면 info 대신 warning으로 눈에 띄게 한다.
        is_preview_or_failed = bool(plan.get("collision_preview")) or bool(plan.get("ik_fallback"))
        log_fn = self.__console.warning if is_preview_or_failed else self.__console.info
        planner_iteration_count = getattr(planner, "last_iteration_count", None)
        log_fn(
            f"inspection path timing: [{label}] robot={robot_name}, planner={planner_name}, "
            f"optimizer={optimizer_name or 'none'}\n"
            f"  target={timings.get('target_setup', 0.0):.3f}s"
            f"  setup={timings.get('planner_setup', 0.0):.3f}s"
            f"  ik_target_setup={timings.get('ik_target_setup', 0.0):.3f}s\n"
            f"  ik={timings.get('ik', 0.0):.3f}s"
            f"  ik_result_check={timings.get('ik_result_check', 0.0):.3f}s"
            f"  planning={timings.get('planning', 0.0):.3f}s"
            f"  optimization={timings.get('optimization', 0.0):.3f}s"
            f"  iteration={planner_iteration_count}\n"
            f"  verify={timings.get('collision_verification', 0.0):.3f}s"
            f"  convert={timings.get('path_conversion', 0.0):.3f}s"
            f"  total={timings.get('total', 0.0):.3f}s\n"
            f"  ik_fallback={bool(plan.get('ik_fallback'))}"
            f"  collision_edges={plan.get('verification', {}).get('colliding_edges', 0)}"
            f"  collision_preview={plan.get('collision_preview')}\n"
            f"  collision_preview_reason={plan.get('collision_preview_reason')}")
        plan.update({
            "path": [np.asarray(p, dtype=float) for p in path],
            "ik_experiment": ik_experiment,
            "timing": timings,
        })
        return plan

    def _plan_retreat_path_for_robot(
        self, request_data, robot_name, start_q, safe_q, obstacle_mesh, context_label=None
    ):
        """first group 종료 자세(start_q)에서 안전 자세(safe_q)까지 q-space 경로를 계획한다.

        배관을 회전시키기 전에 로봇을 배관에서 물러난 안전 자세로 되돌리는 전이 동작이다.
        goal_q(safe_q)가 이미 주어져 있으므로 IK 없이 planner.generate(start_q, safe_q)만 쓴다.
        obstacle_mesh는 회전 전(현재) 배관이어야 한다 - 이 복귀 동작은 배관이 아직 안 돈
        상태에서 일어나기 때문이다.

        Returns:
            dict: _plan_inspection_path_for_robot과 동일한 키 구조의 plan(재생/요약 코드가
                그대로 소비할 수 있도록 waypoints/elapsed/verification 등을 모두 채운다).
        """
        total_t0 = time.perf_counter()
        label = context_label or f"{robot_name}:retreat"
        if obstacle_mesh is None:
            obstacle_mesh = self._current_spool_collision_mesh()
        if obstacle_mesh is None:
            raise RuntimeError("loaded pipe is not available")
        robot_model = self._find_robot(robot_name)
        if robot_model is None:
            raise RuntimeError(f"robot model not found: {robot_name}")
        backend = getattr(self, "_robotics_backend", None)
        robot_backend_model = (
            backend.robot_model(robot_name) if (backend is not None and robot_name is not None) else None
        )
        start_q = np.asarray(start_q, dtype=float)
        safe_q = np.asarray(safe_q, dtype=float)
        start = self._T_to_pose(self._pin_target_world_T(robot_model, robot_backend_model, start_q, robot_name))
        goal_pose = self._T_to_pose(self._pin_target_world_T(robot_model, robot_backend_model, safe_q, robot_name))
        goal = np.zeros(6, dtype=float)
        goal[:3] = np.asarray(goal_pose, dtype=float)[:3]

        planner_name = self._inspection_q_space_planner_name(request_data.get("planner", "rrt_connect"))
        planner = self._load_path_planner(planner_name)
        planner.debug_context = label
        timings = {}
        frame_name = self._robot_target_link_name(robot_name)
        fixed_joint_options = self._path_planning_fixed_joint_options(
            request_data,
            robot_name=robot_name,
            robot_backend_model=robot_backend_model,
            frame_name=frame_name,
            start_q=start_q,
            target_world_pose=goal,
            nearest_point=goal,
        )
        self._configure_inspection_planner(
            planner, obstacle_mesh, start, goal,
            float(request_data.get("step_size", 0.08)),
            int(request_data.get("max_iter", 3000)),
            robot_name=robot_name,
            pin_cache=(getattr(self, "_pinocchio_robot_collision_cache", {}) or {}).get(robot_name),
            timings=timings,
            **fixed_joint_options)
        if not getattr(planner, "_has_robot_q_space_model", lambda: False)():
            raise RuntimeError("robot q-space model is not configured")
        planning_timeout = self._planner_timeout(request_data, planner_name)
        if planning_timeout > 0 and hasattr(planner, "planning_deadline"):
            planner.planning_deadline = time.monotonic() + planning_timeout
        stage_t0 = time.perf_counter()
        try:
            q_path = planner.generate(start_q, safe_q)
        finally:
            if hasattr(planner, "planning_deadline"):
                planner.planning_deadline = None
        planning_elapsed = time.perf_counter() - stage_t0
        collision_preview_reason = None
        if not bool(getattr(planner, "last_returned_path_reaches_goal", True)):
            collision_preview_reason = str(getattr(planner, "last_planning_status", None) or "planner_latest_branch")
        if not q_path:
            q_path = [start_q]
            collision_preview_reason = collision_preview_reason or "planner_empty_start_only"
        optimizer_name = request_data.get("optimizer")
        optimize_path = self._path_optimization_requested(request_data)
        optimization_status = None
        optimization_error = None
        optimization_elapsed = 0.0
        if optimize_path and optimizer_name and q_path:
            stage_t0 = time.perf_counter()
            try:
                optimized_q_path, optimization_status = self._apply_path_optimizer(
                    optimizer_name,
                    q_path,
                    planner,
                )
                if optimized_q_path:
                    q_path = optimized_q_path
                else:
                    optimization_status = "optimizer_empty_path"
                    collision_preview_reason = collision_preview_reason or "optimizer_empty_path"
            except Exception as opt_exc:
                optimization_status = "optimizer_failed"
                optimization_error = str(opt_exc)
                collision_preview_reason = collision_preview_reason or f"optimizer_failed: {opt_exc}"
            optimization_elapsed = time.perf_counter() - stage_t0
        verification = planner.verify_path(q_path)
        if verification.get("colliding_edges", 0) != 0 or verification.get("colliding_waypoints", 0) != 0:
            collision_preview_reason = collision_preview_reason or "returned_path_collision"
        q_path = [np.asarray(q, dtype=float) for q in q_path]
        display_resolution = float(request_data.get("display_step_size", request_data.get("step_size", 0.08)))
        path = self._q_path_to_tcp_poses(
            robot_model, robot_backend_model, robot_name, q_path, sample_resolution=display_resolution)
        if len(path) < 2:
            reason = collision_preview_reason or f"q_path has only {len(q_path)} point(s)"
            raise RuntimeError(f"retreat planning failed for {robot_name}: {reason}")
        collision_preview = collision_preview_reason is not None
        self.__console.info(
            f"inspection retreat plan: [{label}] robot={robot_name}, "
            f"optimizer={optimizer_name or 'none'}, waypoints={len(q_path)}, "
            f"planning={planning_elapsed:.3f}s, optimization={optimization_elapsed:.3f}s, "
            f"collision_preview={collision_preview}, reason={collision_preview_reason}")
        return {
            "q_path": q_path,
            "path": [np.asarray(p, dtype=float) for p in path],
            "goal_q": safe_q,
            "waypoints": len(q_path),
            "elapsed": planning_elapsed,
            "optimizer": optimizer_name,
            "optimization_enabled": optimize_path,
            "optimization_status": optimization_status,
            "optimization_error": optimization_error,
            "convergence_csv": getattr(planner, "last_convergence_csv", None),
            "convergence_plot": getattr(planner, "last_convergence_plot", None),
            "fixed_joint_indices": list(fixed_joint_options.get("fixed_joint_indices", [])),
            "fixed_joint_values": list(fixed_joint_options.get("fixed_joint_values", [])),
            "verification": verification,
            "edge_collisions": verification.get("edge_collisions", []),
            "collision_preview": collision_preview,
            "collision_preview_reason": collision_preview_reason,
            "status": "success" if not collision_preview else "partial",
            "ik_fallback": False,
            "ik_failure": None,
            "ik_result": {},
            "ik_reached_T": None,
            "ik_target_T": None,
            "reached_T": None,
            "planning_error": None,
            "pin_joint_names": self._robot_joint_names(robot_name, robot_backend_model),
            "pose_name": "retreat",
            "timing": {
                "planning": planning_elapsed,
                "optimization": optimization_elapsed,
                "total": time.perf_counter() - total_t0,
            },
        }

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
        from plugins.robotics.inspection_workflow import inspection_group_pose_items
        items = inspection_group_pose_items(group_info)
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
        from plugins.robotics.inspection_workflow import group_is_reachable
        return group_is_reachable(group_info, rt_pipe_facing_axis=self._rt_pipe_facing_axis_config())

    def _inspection_group_rt_position(self, group_info):
        """정렬 기준으로 쓸 RT endeffector target 위치(world)를 반환한다."""
        rt_pose = group_info.get("rt_pose")
        if rt_pose is not None:
            return np.asarray(rt_pose, dtype=float)[:3, 3]
        return np.zeros(3, dtype=float)



    def _move_positioner_r_to(self, r_deg, identity=None, visualize_verification=True):
        """포지셔너 r축을 절대각 r_deg로 실제 이동시킨다(playback에서 second group 진입 시 사용).

        _handle_request_move_positioner(axis="r")와 동일한 이동/spool 동기화 로직을 쓴다.
        visualize_verification=False면 ef pose 충돌 검증 시각화(_verify_rotated_ef_poses_...)를
        생략한다 - path planning 도중 임시로 회전시킬 때는 화면 마커를 건드리면 계획 결과
        시각화가 덮어써지므로 끈다(ef pose 회전 자체는 그대로 수행한다).
        """
        import math
        prev_positioner_r = float(getattr(self, '_positioner_r_deg', 0.0))
        self._positioner_r_deg = float(r_deg)
        for model in getattr(self, '_robot_models', []):
            joint_map = model._urdf._joint_map if model._urdf else {}
            if "f_column_z_to_f_column_r" in joint_map:
                model.set_joint("f_column_z_to_f_column_r", math.radians(r_deg))
                model.update_fk()
        self._invalidate_positioner_collision_mesh_cache()
        spool_T_before = np.asarray(getattr(self, '_spool_world_T', None), dtype=float).copy() \
            if getattr(self, '_spool_world_T', None) is not None else None
        self._sync_fixed_spool_after_positioner_move("r", r_deg, prev_positioner_r, {})
        if getattr(self, '_spool_fix_r', False) and abs(r_deg - prev_positioner_r) > 1e-9:
            try:
                rotation_T = self._positioner_r_rotation_transform(r_deg - prev_positioner_r)
                self._rotate_inspection_target_groups(rotation_T)
                if visualize_verification:
                    self._verify_positioner_rotation_kept_poses_attached(
                        spool_T_before, np.asarray(getattr(self, '_spool_world_T'), dtype=float), rotation_T)
                    self._verify_rotated_ef_poses_against_current_pipe()
                    # _rotate_inspection_target_groups only updates the
                    # stored pose DATA (dda_pose/rt_pose/target_point) -
                    # the drawn EF pose marker actors (_ef_pose_actors) are
                    # never touched by it, so without this the markers stay
                    # at their pre-rotation positions even though the pipe
                    # (and the data used for subsequent planning) actually
                    # rotated. Re-show from the now-rotated data.
                    self._show_ef_target_groups(getattr(self, "_inspection_target_groups", []) or [])
            except Exception as exc:
                self.__console.warning(
                    f"failed to rotate stored ef target poses with positioner r move: {exc}")
        self._show_chuck_frames(render=False)
        if self._spool_positioner_fixed:
            self._save_spool_alignment_state(reason="inspection sequence playback r move")
        self._send_positioner_pose_update(identity=identity)
        self.__console.info(f"execute_inspection_path: positioner rotated to r={r_deg:.1f}deg for playback")

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

    def _inspection_robot_core_snapshot(self, request_data):
        """Create the serializable scene state required by Robot Core."""
        obstacle_mesh = self._current_spool_collision_mesh()
        if obstacle_mesh is None or not obstacle_mesh.has_triangles():
            raise RuntimeError("loaded pipe collision mesh is not available")
        target_groups = request_data.get("target_groups")
        if not isinstance(target_groups, list):
            target_groups = getattr(self, "_inspection_target_groups", []) or []
        if not target_groups:
            raise RuntimeError("EF poses are not determined")
        positioner_mesh = self._build_positioner_collision_mesh()
        delta_r_deg = float(request_data.get(
            "positioner_second_group_r_deg",
            (self._config.get("path_planning", {}) or {}).get(
                "positioner_second_group_r_deg", 180.0),
        ))
        second_group_rotation_T = None
        if bool(getattr(self, "_spool_fix_r", False)):
            second_group_rotation_T = self._positioner_r_rotation_transform(delta_r_deg)
        # Bake the positioner rotation into every group's pose up front
        # (dda_pose_resolved/rt_pose_resolved) so the snapshot holds the
        # complete, actually-reachable pose set on its own - a consumer
        # doesn't need to separately know which groups needed a rotation and
        # reapply second_group_rotation_T itself to get the real target.
        resolved_target_groups = resolve_target_groups_with_rotation(
            target_groups,
            rotation_T=second_group_rotation_T,
            rt_pipe_facing_axis=self._rt_pipe_facing_axis_config(),
        )
        return {
            "spool_vertices": np.asarray(obstacle_mesh.vertices, dtype=float),
            "spool_triangles": np.asarray(obstacle_mesh.triangles, dtype=np.int32),
            "spool_fix_r": bool(getattr(self, "_spool_fix_r", False)),
            "positioner_r_deg": float(getattr(self, "_positioner_r_deg", 0.0)),
            "positioner_vertices": (
                np.asarray(positioner_mesh.vertices, dtype=float)
                if positioner_mesh is not None else np.empty((0, 3), dtype=float)
            ),
            "positioner_triangles": (
                np.asarray(positioner_mesh.triangles, dtype=np.int32)
                if positioner_mesh is not None else np.empty((0, 3), dtype=np.int32)
            ),
            "second_group_rotation_T": second_group_rotation_T,
            # Consumers that recompute phase/reachability from target_groups
            # (partition_and_sort_target_groups - the benchmark script,
            # test_ompl_planning.py) must use this exact axis, or their
            # rotation-needed classification can disagree with what was
            # actually used to produce dda_pose_resolved/rt_pose_resolved
            # above, silently double-rotating or under-rotating a pose.
            "rt_pipe_facing_axis": self._rt_pipe_facing_axis_config().tolist(),
            "target_groups": resolved_target_groups,
            "robot_joint_states": {
                str(getattr(model, "name", "")): {
                    str(name): float(value)
                    for name, value in (getattr(model, "_joint_cfg", {}) or {}).items()
                }
                for model in getattr(self, "_robot_models", []) or []
            },
        }

    def _robot_core_scene_snapshot(self):
        """Lean scene snapshot for a single plan_single_target Robot Core call.

        Unlike _inspection_robot_core_snapshot() (kept only for the "Save
        Planning Snapshot" export), this carries no target_groups/positioner
        rotation data - Robot Core no longer reasons about groups or rotation
        at all. The live obstacle mesh already reflects whatever positioner
        state SimTool has put the scene in (it commands rotation via the
        existing move_positioner request before calling plan_single_target
        for a rotated-phase target), so nothing else is needed here.
        """
        obstacle_mesh = self._current_spool_collision_mesh()
        if obstacle_mesh is None or not obstacle_mesh.has_triangles():
            raise RuntimeError("loaded pipe collision mesh is not available")
        positioner_mesh = self._build_positioner_collision_mesh()
        return {
            "spool_vertices": np.asarray(obstacle_mesh.vertices, dtype=float),
            "spool_triangles": np.asarray(obstacle_mesh.triangles, dtype=np.int32),
            "positioner_vertices": (
                np.asarray(positioner_mesh.vertices, dtype=float)
                if positioner_mesh is not None else np.empty((0, 3), dtype=float)
            ),
            "positioner_triangles": (
                np.asarray(positioner_mesh.triangles, dtype=np.int32)
                if positioner_mesh is not None else np.empty((0, 3), dtype=np.int32)
            ),
            "robot_joint_states": {
                str(getattr(model, "name", "")): {
                    str(name): float(value)
                    for name, value in (getattr(model, "_joint_cfg", {}) or {}).items()
                }
                for model in getattr(self, "_robot_models", []) or []
            },
        }

    def _handle_request_plan_single_target(self, request_data):
        """Plan one robot's path to one target pose via Robot Core.

        SimTool owns target-group splitting/sorting, positioner rotation
        decisions, and start_q chaining across a multi-target sequence now
        (see ROBOT_CORE_DECOUPLING_PLAN.md) - this handler only resolves the
        live scene snapshot and current robot pose (if SimTool didn't supply
        an explicit start_q) and forwards a single source_q -> target_pose
        request to Robot Core.
        """
        identity = request_data.get("_identity")
        robot_core = getattr(self, "_robot_core", None)
        robot_name = request_data.get("robot")
        client_request_id = request_data.get("request_id")

        def _fail(message):
            self.__console.error(f"plan_single_target failed: {message}")
            result = {"status": "failed", "message": str(message), "elapsed": 0.0}
            if hasattr(self, "zapi") and self.zapi and identity:
                self.zapi.reply_plan_single_target(
                    result, identity=identity, client_request_id=client_request_id)
            return None

        if not robot_name:
            return _fail("robot is required")
        target_pose = request_data.get("target_pose")
        if target_pose is None:
            return _fail("target_pose is required")

        try:
            snapshot = self._robot_core_scene_snapshot()
        except Exception as exc:
            return _fail(exc)

        start_q = request_data.get("start_q")
        if start_q is None:
            model = self._find_robot(robot_name)
            if model is None:
                return _fail(f"robot model not found: {robot_name}")
            backend = getattr(self, "_robotics_backend", None)
            robot_backend_model = (
                backend.robot_model(robot_name) if backend is not None else None)
            start_q = self._current_robot_q(
                model, robot_backend_model, robot_name=robot_name).tolist()

        core_request = {
            "operation": OPERATION_PLAN_SINGLE_TARGET,
            "robot_name": robot_name,
            "start_q": start_q,
            "target_pose": target_pose,
            "planner": request_data.get("planner", "rrt_connect"),
            "step_size": request_data.get("step_size", 0.08),
            "max_iter": request_data.get("max_iter", 3000),
            "fixed_joints": request_data.get("fixed_joints"),
            "fixed_joint_indices": request_data.get("fixed_joint_indices"),
            "fixed_joint_values": request_data.get("fixed_joint_values"),
            "planning_timeout": request_data.get("planning_timeout"),
            "context_label": request_data.get("context_label"),
            "ik_solver": request_data.get("ik_solver", "pybullet"),
            "ik_normalize": request_data.get("ik_normalize", False),
            "optimizer": request_data.get("optimizer"),
            "optimize_path": request_data.get("optimize_path", bool(request_data.get("optimizer"))),
            # If this target's pose was resolved against a positioner rotation
            # (see inspection_workflow.resolve_target_groups_with_rotation),
            # the collision pipe mesh must be rotated to match, or start/goal
            # collision is checked against the wrong pipe position. The
            # caller (whoever built target_pose) supplies the same transform.
            "obstacle_rotation_T": request_data.get("obstacle_rotation_T"),
            "lock_linear_track": bool(request_data.get("lock_linear_track", False)),
            "_identity": identity,
            "_client_request_id": client_request_id,
        }
        # keep the robot's joint-state subscriber up to date with whoever is
        # driving planning right now
        self._robot_joint_state_identity = identity

        request_id, failure = submit_robot_core_request(
            robot_core, core_request, snapshot,
            console=self.__console, not_running_message="Robot Core is not running")
        if failure is not None:
            return _fail(failure.get("message", "submission failed"))

        return request_id

    def _handle_robot_core_completed(self, completion):
        """Apply a Robot Core result to Viewer state, render it, and reply through ZAPI."""
        if completion.get("operation") == OPERATION_POSE_DETERMINE:
            return self._handle_pose_process_completed(completion)
        return self._handle_plan_single_target_completed(completion)

    def _handle_request_prepare_next_inspection_phase(self, request_data):
        """Retreat the given robots to a safe posture, then rotate the
        positioner by r_deg_delta - InspectionSequencer (SimTool-side) calls
        this once the "reachable" phase finishes and rotation-needed groups
        remain, before it starts dispatching plan_single_target for them.

        InspectionSequencer itself has no robot-model/joint-name access (it
        only speaks ZAPI - see ROBOT_CORE_DECOUPLING_PLAN.md), so the actual
        zero_non_linear_track_joints computation has to happen here, where
        _robot_joint_names/_current_robot_q are available. This mirrors
        test_ompl_planning.py's retreat-before-rotation choreography
        (zero_non_linear_track_joints's docstring) instead of leaving the arm
        wherever the last pre-rotation target left it - an arbitrary pose
        with no guaranteed clearance once the positioner actually rotates.

        Returns {robot_name: retreated_q, ...} so the sequencer can chain the
        next phase's plan_single_target start_q from this instead of the
        live (pre-retreat) pose.
        """
        from plugins.robotics.inspection_workflow import zero_non_linear_track_joints

        identity = request_data.get("_identity")
        client_request_id = request_data.get("request_id")
        robot_names = request_data.get("robots") or []
        r_deg_delta = float(request_data.get("r_deg_delta", 180.0))

        start_q_by_robot = {}
        try:
            backend = getattr(self, "_robotics_backend", None)
            for robot_name in robot_names:
                model = self._find_robot(robot_name)
                if model is None:
                    continue
                robot_backend_model = backend.robot_model(robot_name) if backend is not None else None
                joint_names = self._robot_joint_names(robot_name, robot_backend_model)
                current_q = self._current_robot_q(
                    model, robot_backend_model, robot_name=robot_name).tolist()
                retreated_q = zero_non_linear_track_joints(current_q, joint_names)
                self._apply_robot_q(
                    model, robot_backend_model, np.asarray(retreated_q, dtype=float), robot_name=robot_name)
                start_q_by_robot[robot_name] = retreated_q
            if robot_names:
                self._send_robot_joint_state_update(robot_names, identity=identity)
                self.plotter.render()

            target_r_deg = float(getattr(self, "_positioner_r_deg", 0.0)) + r_deg_delta
            self.__console.info(
                f"prepare_next_inspection_phase: retreated {list(start_q_by_robot)}, "
                f"rotating positioner {getattr(self, '_positioner_r_deg', 0.0):.1f} -> {target_r_deg:.1f}deg")
            self._move_positioner_r_to(target_r_deg, identity=identity)

            result = {
                "status": "success",
                "start_q_by_robot": to_jsonable(start_q_by_robot),
                "positioner_r_deg": float(getattr(self, "_positioner_r_deg", 0.0)),
                # The rotated target groups (_rotate_inspection_target_groups
                # already ran, inside _move_positioner_r_to above) - SimTool's
                # own copy of target_groups (InspectionSequencer._deferred_
                # groups) is a separate, JSON-round-tripped snapshot from EF
                # pose determination time and was never rotated, so without
                # this the rotation-needed phase's plan_single_target
                # requests would carry pre-rotation target poses against a
                # collision scene whose pipe/positioner HAS actually
                # rotated - a real (not just visual) mismatch, not merely a
                # display bug. InspectionSequencer.on_phase_prepared()
                # substitutes these in before building phase-2 jobs.
                "target_groups": to_jsonable(getattr(self, "_inspection_target_groups", []) or []),
            }
        except Exception as exc:
            self.__console.error(f"prepare_next_inspection_phase failed: {exc}")
            result = {"status": "failed", "message": str(exc)}

        if hasattr(self, "zapi") and self.zapi and identity:
            self.zapi.reply_prepare_next_inspection_phase(
                result, identity=identity, client_request_id=client_request_id)
        return result

    def _handle_plan_single_target_completed(self, completion):
        """Apply a plan_single_target result: move the robot to its final q for
        immediate visual feedback, push a joint-state update (so SimTool's
        start_q for this robot's next target is fresh without polling), and
        reply to whichever SimTool sequencer step is waiting on this target.
        """
        identity = completion.get("_identity")
        client_request_id = completion.get("_client_request_id")
        if completion.get("status") != "completed":
            # Robot core already logs the failure itself (execute_request's
            # except-branch, or _on_process_died/_watch_stale_requests for a
            # crashed/unresponsive service) - viewer just forwards the result.
            message = completion.get("error", "Robot Core failed")
            result = {"status": "failed", "message": message, "elapsed": 0.0}
        else:
            output = completion.get("output") or {}
            result = dict(output.get("result") or {
                "status": "failed",
                "message": "Robot Core returned no result",
            })
            q_path = output.get("q_path") or []
            result["q_path"] = to_jsonable(q_path)
            if q_path and result.get("status") == "success":
                robot_name = completion.get("robot_name")
                model = self._find_robot(robot_name) if robot_name else None
                if model is not None:
                    pin_model = getattr(getattr(self, "_robotics_backend", None), "robot_model", None)
                    pin_model = pin_model(robot_name) if callable(pin_model) else None
                    self._apply_robot_q(model, pin_model, np.asarray(q_path[-1], dtype=float))
                    self._send_robot_joint_state_update([robot_name], identity=identity)
                    # Draw this target's TCP trajectory - dropped during the
                    # ROBOT_CORE_DECOUPLING_PLAN.md refactor (the old
                    # single-shot planner path called _show_inspection_path
                    # via _apply_inspection_planner_output, which the new
                    # per-target InspectionSequencer flow never replaced).
                    # clear=False: each target appends its own segment
                    # instead of erasing the previous target's - a fresh
                    # "Plan Inspection Path" click clears via the existing
                    # "Clear Inspection Path" request instead (see
                    # window.py's __on_btn_plan_inspection_path_clicked).
                    try:
                        tcp_poses = self._q_path_to_target_poses(model, pin_model, robot_name, q_path)
                        if tcp_poses:
                            self._show_inspection_path(tcp_poses, robot_name=robot_name, clear=False)
                    except Exception as exc:
                        self.__console.warning(
                            f"plan_single_target: TCP path visualization failed for {robot_name}: {exc}")
                    self.plotter.render()
        if hasattr(self, "zapi") and self.zapi and identity:
            self.zapi.reply_plan_single_target(
                result, identity=identity, client_request_id=client_request_id)
        return result

    def _handle_pose_process_completed(self, completion):
        """Render a robot-core pose result and forward the serializable result to SimTool."""
        identity = completion.get("_identity")
        self.__console.info(
            f"EF pose callback received: request_id={completion.get('request_id')} "
            f"status={completion.get('status')} robot={completion.get('robot_name')}")
        if completion.get("status") != "completed":
            result = {
                "status": "failed",
                "message": completion.get("error", "robot core pose request failed"),
                "elapsed": 0.0,
            }
        else:
            result = (completion.get("output") or {}).get("result") or {
                "status": "failed",
                "message": "robot core returned no pose result",
            }
        result = to_jsonable(result)
        is_latest = completion.get("request_id") == getattr(
            self, "_active_pose_request_id", completion.get("request_id"))
        groups = result.get("target_groups") or []
        self.__console.info(
            f"EF pose callback result: status={result.get('status')}, "
            f"target_groups={len(groups)}, is_latest={is_latest}"
            + (f", message={result.get('message')}" if result.get("status") != "success" else ""))
        if result.get("status") == "success" and groups and is_latest:
            self._inspection_target_groups = copy.deepcopy(groups)
            self._show_ef_target_groups(groups)
        if hasattr(self, "zapi") and self.zapi and identity:
            self.zapi.reply_ef_pose(result, identity=identity)
        return result

    def _apply_inspection_planner_output(self, sequence, initial_r_deg, result):
        """Store and visualize a completed process result on the render thread."""
        self._last_inspection_plan_sequence = sequence
        self._inspection_playback_initial_r_deg = float(initial_r_deg)
        earliest_group = next((group for group in sequence if group.get("plans")), None)
        if earliest_group is None:
            return
        self._last_inspection_plans = earliest_group["plans"]

        failures = result.get("failures", {}) or {}
        ik_failures = result.get("ik_failures", {}) or {}
        if ik_failures:
            plain_failures = {key.split(":")[-1]: value for key, value in ik_failures.items()}
            self._show_ik_failure_markers(plain_failures.keys(), failure_infos=plain_failures)
        elif failures:
            self._show_ik_failure_markers([key.split(":")[-1] for key in failures])

        for group in sequence:
            for robot_name, plan in (group.get("plans") or {}).items():
                self._show_inspection_ik_pose_result(
                    robot_name,
                    plan.get("ik_reached_T"),
                    plan.get("ik_target_T"),
                    success=not plan.get("ik_fallback", False),
                    fallback=plan.get("ik_fallback", False),
                )
                q_path = plan.get("q_path") or []
                if q_path:
                    self._show_inspection_goal_robot_pose(
                        robot_name,
                        q_path[-1],
                        joint_names=plan.get("pin_joint_names"),
                        clear=False,
                        render=False,
                    )
                self._show_inspection_path(
                    plan.get("path") or [], robot_name=robot_name, clear=False)
                if plan.get("planning_error") and plan.get("reached_T") is not None:
                    self._show_ik_failure_reached_pose(robot_name, plan.get("reached_T"), None)

        preview_robot, preview_plan = next(iter(earliest_group["plans"].items()))
        self._last_inspection_q_path = preview_plan.get("q_path") or []
        self._last_inspection_edge_collisions = preview_plan.get("edge_collisions", [])
        self._last_inspection_robot = preview_robot
        self._last_inspection_path = preview_plan.get("path") or []
        self.plotter.render()

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
            "optimizer": plan.get("optimizer"),
            "optimization_enabled": plan.get("optimization_enabled", False),
            "optimization_status": plan.get("optimization_status"),
            "optimization_error": plan.get("optimization_error"),
            "fixed_joint_indices": plan.get("fixed_joint_indices"),
            "fixed_joint_values": plan.get("fixed_joint_values"),
            "convergence_csv": plan.get("convergence_csv"),
            "convergence_plot": plan.get("convergence_plot"),
            "exploration_csv": plan.get("exploration_csv"),
            "exploration_plot": plan.get("exploration_plot"),
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
        # 포즈 개수가 많으면 goal pose가 다 겹쳐 보여서 실패한 것만 보이게 걸러낼 수 있게 한다.
        show_only_failed = bool(
            (self._config.get("display_options", {}) or {}).get("show_only_failed_ik_poses", False)
        )
        try:
            self._clear_inspection_visuals(clear_point=False)
            # target_groups = self._inspection_target_groups_for_planning
            target_groups = self._inspection_target_groups
            if not target_groups:
                raise RuntimeError("EF poses are not determined")
            self._clear_inspection_goal_pose_visuals(render=False)
            if not show_only_failed:
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

            # group마다 positioner 회전 필요 여부(_inspection_group_is_reachable_now)를 보고,
            # 회전이 필요한 group은 SimTool의 InspectionSequencer가 plan_single_target을
            # 통해 계획할 때와 같은 방식으로 pose/obstacle mesh를 가상 회전시킨 뒤 IK를
            # 확인한다. 회전을 감안하지 않으면 IK check 결과가 실제 계획 결과와 어긋난다.
            delta_r_deg = float(request_data.get(
                "positioner_second_group_r_deg",
                (self._config.get("path_planning", {}) or {}).get(
                    "positioner_second_group_r_deg", 180.0),
            ))
            rotation_T = None
            rotated_obstacle_mesh = None

            # 실제 path planning처럼 "현재 로봇 pose"를 IK 초기 추정값으로 쓴다. IK는 국소
            # solver라 초기값에 따라 수렴 여부/결과가 달라질 수 있어서, q=0에서만 확인하면
            # 실제 계획 때(현재 q에서 시작) 결과와 어긋날 수 있다. 로봇별로 한 번만 계산해
            # 이 핸들러의 모든 group/check에 동일하게 재사용한다.
            current_q_by_robot = {}
            for group_info in target_groups:
                for robot_name, _pose_name, _target_T in self._inspection_group_pose_items(group_info):
                    if robot_name in current_q_by_robot:
                        continue
                    robot_model = self._find_robot(robot_name)
                    backend = getattr(self, "_robotics_backend", None)
                    if robot_model is None or backend is None:
                        continue
                    robot_backend_model = backend.robot_model(robot_name)
                    current_q_by_robot[robot_name] = self._current_robot_q(
                        robot_model, robot_backend_model, robot_name=robot_name
                    ).tolist()

            group_sequence = []
            for sequence_index, group_info in enumerate(target_groups):
                group_index = int(group_info.get("index", sequence_index))
                group_name = str(group_info.get("name", f"Inspection pose {sequence_index + 1}"))
                pose_items = self._inspection_group_pose_items(group_info)
                checks = {}
                group_failures = {}
                group_ik_failures = {}
                group_request = dict(request_data)
                group_request["_start_q_override_by_robot"] = current_q_by_robot

                needs_rotation = not self._inspection_group_is_reachable_now(group_info)
                if needs_rotation and not getattr(self, '_spool_fix_r', False):
                    # 배관이 실제로 chuck에 고정돼 있지 않으면(_spool_fix_r=False) positioner
                    # r축을 돌려도 배관은 안 따라 돈다 - 이 group을 가상 회전으로 "도달 가능"
                    # 취급하면 실제/미리보기 상태와 어긋나므로 명확히 실패 처리한다.
                    self.__console.warning(
                        f"EF pose IK check: {group_name} skipped - spool is not fixed to chuck "
                        "(_spool_fix_r=False), pipe will not actually follow r-axis rotation")
                    for robot_name, pose_name, _target_T in pose_items:
                        failure_key = f"{group_name}:{robot_name}"
                        msg = (
                            "positioner_not_fixed_to_spool: pipe does not actually follow r-axis "
                            "rotation while spool-to-chuck fixation is off")
                        group_failures[robot_name] = msg
                        failures[failure_key] = msg
                    group_sequence.append({
                        "index": group_index,
                        "name": group_name,
                        "checks": checks,
                        "failures": group_failures,
                        "ik_failures": group_ik_failures,
                    })
                    continue
                if needs_rotation and rotation_T is None:
                    rotation_T = self._positioner_r_rotation_transform(delta_r_deg)
                    rotated_obstacle_mesh = copy.deepcopy(obstacle_mesh)
                    rotated_obstacle_mesh.transform(rotation_T)
                group_pose_transform = rotation_T if needs_rotation else None
                group_obstacle_mesh = rotated_obstacle_mesh if needs_rotation else obstacle_mesh

                self.__console.info(
                    f"EF pose IK check: {group_name} ({len(pose_items)} robots), "
                    f"start_q=current_pose, needs_positioner_rotation={needs_rotation}")
                for robot_name, pose_name, target_T in pose_items:
                    failure_key = f"{group_name}:{robot_name}"
                    try:
                        check = self._check_inspection_ik_for_robot(
                            group_request,
                            robot_name,
                            self._transform_target_pose(target_T, group_pose_transform),
                            group_obstacle_mesh,
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
                    is_failed = (
                        bool(check.get("ik_fallback", False))
                        or bool(check.get("collision", False))
                        or robot_name in group.get("ik_failures", {})
                        or robot_name in group.get("failures", {})
                    )
                    if show_only_failed and not is_failed:
                        continue
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
            # 합쳐진 전체 checks가 아니라, group_sequence 중 check가 하나라도 있는 첫 번째
            # group 하나만 고른다(단일 group의 로봇별 check dict).
            earliest_checked_group = next(group for group in group_sequence if group["checks"])
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
                    for robot_name, check in earliest_checked_group["checks"].items()
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
            # 한 줄에 9개 포즈를 다 욱여넣으면 안 보이니, pose(group)별로 한 줄씩 나누고
            # 실패한 pose는 warning으로 남긴다. "실패"는 IK 미수렴(fallback)/충돌/체크 자체
            # 예외(failures/ik_failures) 중 하나라도 있으면이다.
            failed_group_names = []
            for group in group_sequence:
                if not group["checks"]:
                    continue
                robot_texts = []
                group_failed = bool(group["failures"]) or bool(group["ik_failures"])
                for robot, check in group["checks"].items():
                    robot_ok = (
                        bool(check.get("ik_result", {}).get("success", False))
                        and not check.get("ik_fallback", False)
                        and not check.get("ik_result", {}).get("collision", False)
                    )
                    if not robot_ok:
                        group_failed = True
                    robot_texts.append(f"{robot}={'ok' if robot_ok else 'FAIL'}")
                log_fn = self.__console.warning if group_failed else self.__console.info
                if group_failed:
                    failed_group_names.append(group["name"])
                log_fn(f"EF pose IK check: {group['name']}: {', '.join(robot_texts)}")

            n_total = sum(1 for group in group_sequence if group["checks"])
            n_failed = len(failed_group_names)
            summary_fn = self.__console.warning if n_failed else self.__console.info
            summary_fn(
                f"EF pose IK check summary: {n_total - n_failed}/{n_total} poses ok"
                + (f", failed=[{', '.join(failed_group_names)}]" if n_failed else "")
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
        # 포지셔너를 실제로 돌려야 하는 group이 하나라도 있으면(positioner_r_deg != 0),
        # _move_positioner_r_to가 파이프 상태(_spool_fix_r 등)에 의존하는 로직을 타므로
        # 파이프가 먼저 로드되어 있어야 한다 - 없으면 회전해도 파이프가 안 따라 도는 등
        # 애매하게 동작하느니, 명확한 안내와 함께 여기서 막는다.
        needs_positioner_move = any(
            abs(float(group.get("positioner_r_deg", 0.0) or 0.0)) > 1e-9 for group in valid_sequence)
        if needs_positioner_move and self._current_spool_collision_mesh() is None:
            self.__console.warning(
                "execute_inspection_path: this sequence needs the positioner to rotate "
                "(positioner_r_deg != 0 for at least one group) but no pipe is loaded - "
                "load a pipe first, then retry playback.")
            return False
        # playback은 반드시 "배관 초기 자세"에서 시작해야 한다. 직전 playback이 second group
        # 회전 상태로 끝났거나 배관이 다른 이유로 돌아가 있을 수 있으므로, 계획 시점에 저장해
        # 둔 초기 r 각도로 명시적으로 되돌린 뒤 시작한다.
        initial_r_deg = getattr(self, '_inspection_playback_initial_r_deg', None)
        if initial_r_deg is None:
            initial_r_deg = float(valid_sequence[0].get("positioner_r_deg", 0.0))
        initial_r_deg = float(initial_r_deg)
        if abs(float(getattr(self, '_positioner_r_deg', 0.0)) - initial_r_deg) > 1e-9:
            self.__console.info(
                f"execute_inspection_path: resetting positioner to initial r={initial_r_deg:.1f}deg "
                "before playback")
            self._move_positioner_r_to(initial_r_deg, identity=identity)
            self.plotter.render()
        self._inspection_sequence_playback = {
            "sequence": valid_sequence,
            "index": 0,
            "speed": max(float(speed), 1e-6),
            "identity": identity,
            "initial_r_deg": initial_r_deg,
        }
        return self._start_next_inspection_sequence_group()

    def _build_retreat_animation_plans(self, plans, identity=None):
        """Per-robot {robot: {"q_path": [current_q, retreated_q]}} for the
        robots in `plans` whose live pose differs from that group's plan's
        own first waypoint (which is exactly the "retreated to safe pose"
        q the plan was computed from - see _handle_request_prepare_next_
        inspection_phase, which returns THIS q as start_q_by_robot for
        planning). Two waypoints is enough - _start_multi_robot_path_
        playback/​_step_robot_path_playback already interpolate linearly in
        joint space between waypoints, same as any other path segment.
        Robots already at (or very near) that pose are skipped, so a
        no-op retreat doesn't start a zero-length animation."""
        retreat_plans = {}
        for robot_name, plan in plans.items():
            q_path = plan.get("q_path")
            if not q_path:
                continue
            target_q = np.asarray(q_path[0], dtype=float)
            model = self._find_robot(robot_name)
            if model is None:
                continue
            backend = getattr(self, "_robotics_backend", None)
            robot_backend_model = backend.robot_model(robot_name) if backend is not None else None
            try:
                current_q = self._current_robot_q(model, robot_backend_model, robot_name=robot_name)
            except Exception:
                continue
            if current_q.shape != target_q.shape or np.allclose(current_q, target_q, atol=1e-6):
                continue
            retreat_plans[robot_name] = {"q_path": [current_q.tolist(), target_q.tolist()]}
        return retreat_plans

    def _start_next_inspection_sequence_group(self):
        seq_state = getattr(self, "_inspection_sequence_playback", None)
        if not seq_state:
            return False

        # Resume point after a retreat-to-safe-pose sub-animation (below)
        # finishes: rotate the positioner, then actually start this group's
        # real path - NOT the normal "advance to next group" branch below,
        # which would otherwise skip straight past this group.
        pending = seq_state.get("pending_group")
        if pending is not None:
            seq_state["pending_group"] = None
            group, idx = pending["group"], pending["idx"]
            sequence = seq_state.get("sequence", [])
            target_r_deg = group.get("positioner_r_deg")
            if target_r_deg is not None and float(target_r_deg) != float(getattr(self, '_positioner_r_deg', 0.0)):
                self.__console.info(
                    f"execute_inspection_path: rotating positioner to {float(target_r_deg):.1f}deg "
                    "after retreat")
                self._move_positioner_r_to(target_r_deg, identity=seq_state.get("identity"))
                self.plotter.render()
            self.__console.info(
                f"execute_inspection_path: start {group.get('name', f'inspection pose {idx + 1}')} "
                f"({idx + 1}/{len(sequence)})")
            return self._start_multi_robot_path_playback(
                group.get("plans", {}), speed=seq_state.get("speed", 0.2), identity=seq_state.get("identity"))

        sequence = seq_state.get("sequence", [])
        idx = int(seq_state.get("index", 0))
        if idx >= len(sequence):
            # playback이 끝나면 배관을 초기 자세로 되돌린다. 안 그러면 배관이 second group
            # 회전 상태(initial+180)로 남고, 다음 계획이 그 각도를 새 초기값으로 잡아
            # 회전각이 180 -> 360 -> 540 ...으로 누적된다(r=720deg 버그).
            initial_r_deg = seq_state.get("initial_r_deg")
            self._inspection_sequence_playback = None
            if initial_r_deg is not None and abs(
                    float(getattr(self, '_positioner_r_deg', 0.0)) - float(initial_r_deg)) > 1e-9:
                self.__console.info(
                    f"execute_inspection_path: restoring positioner to initial r={float(initial_r_deg):.1f}deg "
                    "after playback")
                self._move_positioner_r_to(float(initial_r_deg), identity=seq_state.get("identity"))
                self.plotter.render()
            self.__console.info("execute_inspection_path: inspection pose sequence playback finished")
            return False
        group = sequence[idx]
        seq_state["index"] = idx + 1
        target_r_deg = group.get("positioner_r_deg")
        needs_rotation = target_r_deg is not None and float(target_r_deg) != float(getattr(self, '_positioner_r_deg', 0.0))
        if needs_rotation:
            # Requirement: retreat to a safe posture (visually animated, not
            # an instant snap) BEFORE the positioner rotates - see
            # _handle_request_prepare_next_inspection_phase's identical
            # choreography for the live-planning side; this is the playback-
            # side equivalent, now actually visible instead of a teleport.
            retreat_plans = self._build_retreat_animation_plans(group.get("plans", {}), identity=seq_state.get("identity"))
            if retreat_plans:
                seq_state["pending_group"] = {"group": group, "idx": idx}
                self.__console.info(
                    "execute_inspection_path: retreating to safe pose before positioner rotation "
                    f"({sorted(retreat_plans)})")
                if self._start_multi_robot_path_playback(
                        retreat_plans, speed=seq_state.get("speed", 0.2), identity=seq_state.get("identity")):
                    return True
                # No robot actually needed to move (or playback couldn't
                # start) - fall through and rotate+play immediately instead
                # of leaving pending_group set with nothing to resume it.
                seq_state["pending_group"] = None
            self._move_positioner_r_to(target_r_deg, identity=seq_state.get("identity"))
            self.plotter.render()
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
        # self.__console.warning(
        #     "execute_inspection_path: collision between "
        #     f"waypoint {edge_idx} -> {edge_idx + 1} ({pair_text})")
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
        self._invalidate_spool_collision_mesh_cache()
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
            self._invalidate_spool_collision_mesh_cache()
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
            # 배관 geometry는 그대로고 world pose만 바뀐 이동이다. local-frame collision
            # mesh 보관본이 이미 있으면 재생성하지 않는다 - _current_spool_collision_mesh가
            # 바뀐 _spool_world_T를 감지해 rigid 변환만 다시 적용한다. 아직 보관본이 없으면
            # (T=None 상태에서 처음 만들어진 경우 등) 한 번만 전체 재생성하도록 무효화한다.
            if getattr(self, '_spool_collision_mesh_local', None) is None:
                self._invalidate_spool_collision_mesh_cache()
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

    def set_robot_core(self, robot_core):
        """Attach the Robot Core client used for pose determination and planning."""
        self._robot_core = robot_core

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

