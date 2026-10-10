from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import open3d as o3d

from plugins.robotics.inspection_experiment_logger import InspectionExperimentLogger
from robot_core.path_planning_service import plan_single_target
from robot_core.pose_service import PoseDeterminationService
from robot_core.service import (
    DEFAULT_OPERATION,
    OPERATION_PLAN_SINGLE_TARGET,
    OPERATION_POSE_DETERMINE,
)
from util.logger.console import ConsoleLogger
from viewervedo.visualizer import Visualizer


class _NullPlotter:
    def add(self, *args, **kwargs):
        return self

    def remove(self, *args, **kwargs):
        return self

    def render(self, *args, **kwargs):
        return self


class RobotCoreEngine(Visualizer):
    """Run existing numerical routines without constructing a Viewer window."""

    def __init__(self, config, snapshot):
        self._config = copy.deepcopy(config or {})
        self._Visualizer__console = ConsoleLogger.get_logger()
        self.plotter = _NullPlotter()
        self._robot_core_worker_mode = True
        self._robot_core = None
        self._request_queue = []
        self._robot_models = []
        self._inspection_points = []
        self._inspection_point = None
        self._inspection_target_groups = copy.deepcopy(snapshot.get("target_groups") or [])
        self._inspection_second_groups = []
        self._inspection_goal_pose_actors = []
        self._inspection_goal_robot_actors = []
        self._ef_pose_actors = []
        self._ef_pose_groups = []
        self._ef_target_poses = {}
        self._ik_failure_actors = []
        self._inspection_path_actor = None
        self._last_inspection_plan_sequence = []
        self._last_ik_failure = {}
        self._base_frame_collision_cache = {}
        self._positioner_collision_mesh_cache = None
        self._spool_collision_mesh_cache = None
        self._spool_collision_mesh_cache_T = None
        self._spool_collision_mesh_local = None
        self._spool_world_T = np.eye(4, dtype=float)
        self._spool_fix_r = bool(snapshot.get("spool_fix_r", False))
        self._positioner_r_deg = float(snapshot.get("positioner_r_deg", 0.0))
        self._planner_second_group_rotation_T = snapshot.get("second_group_rotation_T")
        self._worker_spool_points = np.asarray(snapshot.get("spool_points", []), dtype=float)
        self._worker_spool_mesh = self._mesh_from_snapshot(
            snapshot.get("spool_vertices"), snapshot.get("spool_triangles"))
        self._worker_positioner_mesh = self._mesh_from_snapshot(
            snapshot.get("positioner_vertices"), snapshot.get("positioner_triangles"))

        experiment_root = Path(self._config.get("debug_dir", "debug")) / "inspection_ik"
        self._inspection_ik_experiment_logger = InspectionExperimentLogger(
            experiment_root / f"worker_{os.getpid()}")
        self._inspection_ik_experiment_dir = self._inspection_ik_experiment_logger.session_dir
        self._setup_robots(self._config)
        self._apply_robot_joint_states(snapshot.get("robot_joint_states") or {})

    @staticmethod
    def _mesh_from_snapshot(vertices, triangles):
        vertices = np.asarray(vertices if vertices is not None else [], dtype=float)
        triangles = np.asarray(triangles if triangles is not None else [], dtype=np.int32)
        if vertices.size == 0 or triangles.size == 0:
            return None
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.reshape(-1, 3))
        mesh.triangles = o3d.utility.Vector3iVector(triangles.reshape(-1, 3))
        mesh.compute_vertex_normals()
        return mesh

    def _apply_robot_joint_states(self, states):
        for model in self._robot_models:
            state = states.get(str(getattr(model, "name", "")), {})
            for joint_name, value in state.items():
                try:
                    model.set_joint(str(joint_name), float(value))
                except Exception:
                    continue
            model.update_fk()

    def _get_spool_points(self):
        return self._worker_spool_points

    def _current_spool_collision_mesh(self):
        return self._worker_spool_mesh

    def _build_positioner_collision_mesh(self):
        return self._worker_positioner_mesh

    def _show_ef_target_groups(self, _groups):
        return None

    def _show_inspection_goal_pose(self, *args, **kwargs):
        return []

    def _show_inspection_goal_robot_pose(self, *args, **kwargs):
        return []

    def _show_inspection_ik_pose_result(self, *args, **kwargs):
        return []

    def _show_inspection_path(self, *args, **kwargs):
        return None

    def _show_ik_failure_markers(self, *args, **kwargs):
        return None

    def _show_ik_failure_reached_pose(self, *args, **kwargs):
        return None


def execute_robot_core_request(config, request, snapshot):
    """Execute one serializable pose or planning request in the robot-core process."""
    worker = RobotCoreEngine(config, snapshot)
    operation = str(request.get("operation", DEFAULT_OPERATION))

    if operation == OPERATION_POSE_DETERMINE:
        inspection_points = request.get("inspection_points")
        if inspection_points is None:
            inspection_points = snapshot.get("inspection_points")
        spool_points = snapshot.get("spool_points")
        if spool_points is None:
            spool_points = []
        # service.py's execute_request/zapi.py's zapi_execute_request already
        # log generic "received request_id=... operation=pose_determine"
        # lines, but neither knows the actual EF pose request content (how
        # many inspection points, whether a spool point cloud is even
        # attached) - that's only available here, right before the viewer's
        # "Determine EF Pose" request actually starts running.
        ConsoleLogger.get_logger().info(
            f"Robot Core received EF pose request: "
            f"{len(inspection_points or [])} inspection point(s), "
            f"{len(spool_points)} spool point(s)")
        # Same obstacle mesh(es) plan_single_target collision-checks against
        # (see PoseDeterminationService's obstacle_meshes docstring) - not a
        # separately reconstructed one, so pose determination and path
        # planning agree on what "collision" means for this pipe.
        obstacle_meshes = [
            worker._current_spool_collision_mesh(),
            worker._build_positioner_collision_mesh(),
        ]
        service = PoseDeterminationService(config, worker._robotics_backend, obstacle_meshes=obstacle_meshes)
        return {"result": service.determine(
            inspection_points or [],
            spool_points,
        )}

    if operation == OPERATION_PLAN_SINGLE_TARGET:
        return plan_single_target(worker, dict(request))

    raise ValueError(
        f"Unknown Robot Core operation: {operation!r} "
        f"(expected {OPERATION_POSE_DETERMINE!r} or {OPERATION_PLAN_SINGLE_TARGET!r})")
