from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from plugins.poseDeterminator.EndEffectorPoseOptimizer import EndEffectorPoseOptimizer
from plugins.robotics.inspection_workflow import to_jsonable
from util.logger.console import ConsoleLogger

# Points sampled per obstacle mesh for _mesh_sample_points() - a fast,
# non-mesh-collision-engine way to keep pose determination's point-cloud-
# proximity check consistent with what path planning's real mesh collision
# will see: dense enough that "no scan point within 1mm" isn't left blind to
# small solid-mesh protrusions a sparser raw scan happened to skip.
_MESH_SAMPLE_POINTS_PER_OBSTACLE = 200000


class PoseDeterminationService:
    """Robot Core-owned pipe profile fitting and DDA/RT pose generation."""

    DDA_ROBOT = "dda_rb10_1300e"
    RT_ROBOT = "rb20_1900es"

    def __init__(self, config: Mapping[str, Any], robotics_backend, obstacle_meshes=None):
        self.config = dict(config or {})
        self.backend = robotics_backend
        self.console = ConsoleLogger.get_logger()
        # The SAME obstacle mesh(es) path planning collision-checks against
        # (RobotCoreEngine._current_spool_collision_mesh()/
        # _build_positioner_collision_mesh(), passed in by worker.py). NOT
        # fed into a mesh collision engine (PyBullet mesh-vs-mesh was too
        # slow even without reconstruction - see git history) - instead
        # sampled down to points once in _optimizer() and merged into the
        # point cloud EndEffectorPoseOptimizer's existing fast point-to-point
        # proximity check already uses, so pose determination stays point-
        # based (fast) while checking against the same geometry planning
        # uses (consistent) instead of a separately-noisy raw scan.
        self.obstacle_meshes = [m for m in (obstacle_meshes or []) if m is not None]

    @staticmethod
    def _robot_name(pose_name: str) -> str:
        return PoseDeterminationService.DDA_ROBOT if str(pose_name).upper().startswith("DDA") else PoseDeterminationService.RT_ROBOT

    @staticmethod
    def _pose_offset(frame_config):
        if not isinstance(frame_config, dict):
            return None
        offset = frame_config.get("pose_to_link_offset")
        if offset is None:
            return None
        if isinstance(offset, dict) and "matrix" in offset:
            transform = np.asarray(offset["matrix"], dtype=float)
        elif isinstance(offset, dict):
            transform = np.eye(4, dtype=float)
            transform[:3, 3] = np.asarray(offset.get("xyz", [0.0, 0.0, 0.0]), dtype=float)
            transform[:3, :3] = Rotation.from_euler(
                "xyz", offset.get("rpy", [0.0, 0.0, 0.0])).as_matrix()
        else:
            transform = np.asarray(offset, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError(
                f"ef_pose pose_to_link_offset must be 4x4, got shape={transform.shape}")
        return transform

    @staticmethod
    def _mesh_sample_points(meshes, n_per_mesh: int = _MESH_SAMPLE_POINTS_PER_OBSTACLE):
        """Merge dense point samples off each obstacle mesh's surface into one
        PointCloud - just sampling, no collision engine, so this stays cheap
        (O(n_per_mesh) per mesh) unlike a mesh-vs-mesh collision check."""
        clouds = []
        for mesh in meshes:
            if mesh is None or not mesh.has_triangles():
                continue
            clouds.append(mesh.sample_points_uniformly(number_of_points=int(n_per_mesh)))
        if not clouds:
            return None
        merged = clouds[0]
        for cloud in clouds[1:]:
            merged += cloud
        return merged

    @staticmethod
    def _point_cloud_collision_check(link_model, tcp_pose, tcp_to_link_pose_T, reference_pcd,
                                      margin: float = 0.05, sample_count: int = 5000,
                                      threshold: float = 0.001) -> bool:
        """Same point-to-point proximity check as EndEffectorPoseOptimizer's
        built-in __check_collision fallback (EF mesh sample points vs nearest
        reference point, no mesh reconstruction/collision engine), just
        parameterized on reference_pcd so it can check against points sampled
        from planning's own obstacle mesh instead of the raw scan - see
        _mesh_sample_points()."""
        import copy as _copy

        tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(-1)
        tcp_to_link_pose_T = np.asarray(tcp_to_link_pose_T, dtype=float)
        tcp_pose_T = np.eye(4)
        tcp_pose_T[:3, :3] = Rotation.from_euler("xyz", tcp_pose[3:6]).as_matrix()
        tcp_pose_T[:3, 3] = tcp_pose[:3]
        link_pose_T = tcp_pose_T @ tcp_to_link_pose_T

        mesh_copy = _copy.deepcopy(link_model)
        mesh_copy.transform(link_pose_T)

        aabb = mesh_copy.get_axis_aligned_bounding_box()
        margin_vec = np.array([margin, margin, margin], dtype=float)
        crop_box = o3d.geometry.AxisAlignedBoundingBox(
            aabb.min_bound - margin_vec, aabb.max_bound + margin_vec)
        idx = crop_box.get_point_indices_within_bounding_box(reference_pcd.points)
        if not idx:
            return False

        sub_pcd = reference_pcd.select_by_index(idx)
        mesh_pcd = mesh_copy.sample_points_uniformly(number_of_points=int(sample_count))
        distances = sub_pcd.compute_point_cloud_distance(mesh_pcd)
        return any(float(d) <= float(threshold) for d in distances)

    @staticmethod
    def _point_cloud(points: Sequence[Sequence[float]]):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) < 10:
            raise RuntimeError("loaded spool point cloud is not available")
        bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        radius = max(bbox_diag * 0.005, 1e-6)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        cloud.normalize_normals()
        return cloud

    def _optimizer(self, scan_data):
        params = self.config.get("ef_pose", {}) or {}
        logging_config = params.get("logging", {}) or {}
        optimizer = EndEffectorPoseOptimizer(
            debug_mode=bool(params.get("debug_mode", True)),
            log_path=logging_config.get("log_path"),
            log_dir=logging_config.get("log_dir"),
            log_level=logging_config.get("level", "DEBUG"),
            console_level=logging_config.get("console_level"),
            file_level=logging_config.get("file_level"),
            logger_name=logging_config.get("name", "flame_robotics"),
            force_logger_config=logging_config.get("force"),
        )
        optimizer._scan_data = scan_data

        frames = params.get("frames", {}) or {}
        dda_frame = frames.get("dda", {}) or {}
        rt_frame = frames.get("rt", {}) or {}
        dda_facing_axis = np.asarray(
            dda_frame.get("pipe_facing_axis", [1.0, 0.0, 0.0]), dtype=float)
        dda_parallel_axis = dda_frame.get("pipe_parallel_axis")
        rt_facing_axis = np.asarray(
            rt_frame.get("pipe_facing_axis", [0.0, -1.0, 0.0]), dtype=float)
        optimizer.set_dda_pipe_facing_axis(
            dda_facing_axis,
            None if dda_parallel_axis is None else np.asarray(dda_parallel_axis, dtype=float),
        )
        optimizer.set_rt_pipe_facing_axis(rt_facing_axis)

        dda_mesh, dda_tcp_to_link = self.backend.end_effector_collision_geometry(
            self.DDA_ROBOT,
            str(dda_frame.get("end_link", "dda_link_end")),
            str(dda_frame.get("tcp_joint", "dda_joint_tcp")),
            pose_to_link_offset=self._pose_offset(dda_frame),
        )
        rt_mesh, rt_tcp_to_link = self.backend.end_effector_collision_geometry(
            self.RT_ROBOT,
            str(rt_frame.get("end_link", "rt_link_end")),
            str(rt_frame.get("tcp_joint", "rt_joint_end")),
            pose_to_link_offset=self._pose_offset(rt_frame),
        )
        optimizer.set_DDA_geometry(dda_mesh, dda_tcp_to_link)
        optimizer.set_RT_geometry(rt_mesh, rt_tcp_to_link)
        mesh_pcd = self._mesh_sample_points(self.obstacle_meshes) if self.obstacle_meshes else None
        if mesh_pcd is not None:
            # Still the fast point-to-point proximity check (no mesh
            # collision engine - PyBullet mesh-vs-mesh was too slow even
            # without reconstruction), just checked against points sampled
            # from planning's own obstacle mesh instead of the raw scan, so
            # pose determination and path planning agree on the pipe/
            # positioner's actual shape instead of two different
            # approximations of it.
            optimizer.set_collision_checker(
                lambda link_model, tcp_pose, tcp_to_link_pose_T, margin=0.05, sample_count=5000:
                    self._point_cloud_collision_check(
                        link_model, tcp_pose, tcp_to_link_pose_T, mesh_pcd,
                        margin=margin, sample_count=sample_count))
        # else: leave optimizer's collision checker unset - falls back to
        # EndEffectorPoseOptimizer.__check_collision's built-in point-cloud-
        # proximity check against the raw scan (no obstacle mesh was
        # available to sample from).
        return optimizer, params, rt_facing_axis

    def determine(self, inspection_points, spool_points):
        """Determine all DDA/RT target groups and return a complete response payload."""
        started = time.perf_counter()
        try:
            targets = [
                np.asarray(point, dtype=float).reshape(-1)[:3]
                for point in (inspection_points or [])
            ]
            if not targets:
                raise RuntimeError("inspection point is not selected")

            stage_started = time.perf_counter()
            scan_data = self._point_cloud(spool_points)
            point_cloud_elapsed = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            optimizer, params, rt_facing_axis = self._optimizer(scan_data)
            setup_elapsed = time.perf_counter() - stage_started

            target_groups = []
            failures = []
            profile_elapsed = 0.0
            pose_elapsed = 0.0
            for point_index, target in enumerate(targets):
                try:
                    optimizer.debuging_info = {}
                    stage_started = time.perf_counter()
                    optimizer.calculate_pipe_profile(
                        target,
                        sampling_size_for_calculating_normal=float(
                            params.get("sampling_size_for_calculating_normal", 0.01)),
                        radius_offset_for_sampling_points_in_sphere=float(
                            params.get("radius_offset_for_sampling_points_in_sphere", 0.003)),
                    )
                    profile_elapsed += time.perf_counter() - stage_started

                    stage_started = time.perf_counter()
                    point_groups = list(
                        optimizer.calculate_DDA_RT_pose_for_taking_xray(
                            target,
                            num_candidates=int(params.get("num_candidates", 9)),
                            distance_from_dda_to_surface=float(
                                params.get("distance_from_dda_to_surface", 0.01)),
                            distance_from_dda_to_rt=float(
                                params.get("distance_from_dda_to_rt", 0.3)),
                            angle_of_rt=float(params.get("angle_of_rt", 10.0)),
                            rt_pipe_facing_axis=rt_facing_axis,
                            pose_name_to_robot_name=self._robot_name,
                            force_90_fallback=bool(params.get("force_90_fallback", False)),
                        )
                        or []
                    )
                    pose_elapsed += time.perf_counter() - stage_started
                    start_index = len(target_groups)
                    for local_index, group in enumerate(point_groups):
                        group["point_index"] = point_index
                        group["name"] = (
                            f"Point {point_index + 1} - "
                            f"{group.get('name', f'pose {local_index + 1}')}"
                        )
                        group["index"] = start_index + local_index
                    target_groups.extend(point_groups)
                except Exception as exc:
                    failures.append({
                        "point_index": point_index,
                        "point": target.tolist(),
                        "message": str(exc),
                    })
                    self.console.error(
                        f"Robot Core pose determination failed for point {point_index + 1}: {exc}")

            if not target_groups:
                raise RuntimeError(
                    f"poseDeterminator returned no valid target group for "
                    f"{len(targets)} selected point(s): {failures}")
            elapsed = time.perf_counter() - started
            result = {
                "status": "success",
                "source": "robot_core",
                "target_groups": target_groups,
                "inspection_point_count": len(targets),
                "target_group_count": len(target_groups),
                "target_failures": failures,
                "elapsed": elapsed,
                "timing": {
                    "point_cloud": point_cloud_elapsed,
                    "robot_core_setup": setup_elapsed,
                    "pipe_profile": profile_elapsed,
                    "pose_candidates": pose_elapsed,
                },
            }
            self.console.info(
                f"Robot Core pose result: points={len(targets)}, "
                f"groups={len(target_groups)}, elapsed={elapsed:.3f}s")
            return to_jsonable(result)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self.console.error(f"Robot Core pose determination failed after {elapsed:.3f}s: {exc}")
            return {
                "status": "failed",
                "source": "robot_core",
                "message": str(exc),
                "elapsed": elapsed,
            }
