from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from plugins.poseDeterminator.EndEffectorPoseOptimizer import EndEffectorPoseOptimizer
from plugins.poseDeterminator.PipeProfileFitting import PipeProfileParams
from plugins.robotics.inspection_workflow import to_jsonable
from util.logger.console import ConsoleLogger

# Points sampled per obstacle mesh for _mesh_sample_points() - a fast,
# non-mesh-collision-engine way to keep pose determination's point-cloud-
# proximity check consistent with what path planning's real mesh collision
# will see: dense enough that "no scan point within 1mm" isn't left blind to
# small solid-mesh protrusions a sparser raw scan happened to skip.
_MESH_SAMPLE_POINTS_PER_OBSTACLE = 200000


class _ObstacleCollisionChecker:
    """EF mesh vs. 장애물 샘플 점 근접 검사 (PoseDeterminationService._point_cloud_collision_check와 같은 판정).

    판정은 동일하되, 충돌 시 어느 장애물(pipe/positioner)의 어느 점과 몇 m 떨어져 걸렸는지를
    last_hit에 남긴다. 후보가 "눈에 안 보이는데 충돌로 탈락"하는 경우 원인을 추적하기 위한 것이다.
    """

    def __init__(self, reference_pcd, labels, label_names, threshold=0.001):
        self.reference_pcd = reference_pcd
        self.reference_points = np.asarray(reference_pcd.points, dtype=float)
        self.labels = np.asarray(labels, dtype=np.int32)
        self.label_names = list(label_names)
        self.threshold = float(threshold)
        self.last_hit = None

    def __call__(self, link_model, tcp_pose, tcp_to_link_pose_T, margin=0.05, sample_count=5000):
        import copy as _copy

        self.last_hit = None
        tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(-1)
        tcp_pose_T = np.eye(4)
        tcp_pose_T[:3, :3] = Rotation.from_euler("xyz", tcp_pose[3:6]).as_matrix()
        tcp_pose_T[:3, 3] = tcp_pose[:3]
        link_pose_T = tcp_pose_T @ np.asarray(tcp_to_link_pose_T, dtype=float)

        mesh_copy = _copy.deepcopy(link_model)
        mesh_copy.transform(link_pose_T)
        aabb = mesh_copy.get_axis_aligned_bounding_box()
        margin_vec = np.array([margin, margin, margin], dtype=float)
        crop_box = o3d.geometry.AxisAlignedBoundingBox(
            aabb.min_bound - margin_vec, aabb.max_bound + margin_vec)
        idx = np.asarray(crop_box.get_point_indices_within_bounding_box(self.reference_pcd.points), dtype=int)
        if idx.size == 0:
            return False

        sub_pcd = self.reference_pcd.select_by_index(idx.tolist())
        mesh_pcd = mesh_copy.sample_points_uniformly(number_of_points=int(sample_count))
        distances = np.asarray(sub_pcd.compute_point_cloud_distance(mesh_pcd), dtype=float)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) > self.threshold:
            return False
        ref_index = int(idx[nearest])
        label = int(self.labels[ref_index])
        self.last_hit = {
            "obstacle": self.label_names[label] if 0 <= label < len(self.label_names) else str(label),
            "distance": float(distances[nearest]),
            "point": self.reference_points[ref_index].tolist(),
        }
        return True


class PoseDeterminationService:
    """Robot Core-owned pipe profile fitting and DDA/RT pose generation."""

    DDA_ROBOT = "dda_rb10_1300e"
    RT_ROBOT = "rb20_1900es"

    def __init__(self, config: Mapping[str, Any], robotics_backend, obstacle_meshes=None):
        self.config = dict(config or {})
        self.backend = robotics_backend
        self.console = ConsoleLogger.get_logger()
        # World-frame obstacle meshes shared with planning; no point-sampling fallback.
        default_names = ("pipe", "positioner")
        named = [
            (default_names[i] if i < len(default_names) else f"obstacle{i}", m)
            for i, m in enumerate(obstacle_meshes or [])
            if m is not None
        ]
        self.obstacle_names = [name for name, _ in named]
        self.obstacle_meshes = [m for _, m in named]

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
    def _labeled_mesh_sample_points(meshes, n_per_mesh: int = _MESH_SAMPLE_POINTS_PER_OBSTACLE):
        """_mesh_sample_points와 같은 샘플링 + 각 점이 어느 mesh에서 나왔는지 label 배열."""
        clouds, labels = [], []
        for i, mesh in enumerate(meshes):
            if mesh is None or not mesh.has_triangles():
                continue
            cloud = mesh.sample_points_uniformly(number_of_points=int(n_per_mesh))
            clouds.append(cloud)
            labels.append(np.full(len(cloud.points), i, dtype=np.int32))
        if not clouds:
            return None, None
        merged = clouds[0]
        for cloud in clouds[1:]:
            merged += cloud
        return merged, np.concatenate(labels)

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
        optimizer.set_rt_roll_about_facing_deg(float(rt_frame.get("roll_about_facing_deg", 0.0)))

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
        dda_mesh = optimizer.set_DDA_geometry(dda_mesh, dda_tcp_to_link)
        rt_mesh = optimizer.set_RT_geometry(rt_mesh, rt_tcp_to_link)
        # Fail closed: no point-sampling fallback when the URDF/PyBullet
        # checker cannot be initialized. End-link geometry and threshold match planning.
        optimizer.set_collision_checker(self.backend.create_pose_collision_checker(
            self.obstacle_meshes, self.obstacle_names,
            [(self.DDA_ROBOT, str(dda_frame.get("end_link", "dda_link_end")), dda_mesh),
             (self.RT_ROBOT, str(rt_frame.get("end_link", "rt_link_end")), rt_mesh)]))
        return optimizer, params, rt_facing_axis

    @staticmethod
    def _profile_debug(debug_info, point_index, input_point, offset, max_cluster_points=300, max_fit_points=2000):
        """파이프 프로파일 피팅 중간 결과(ROI, 클러스터, 추정 구, 최종 실린더)를 viewer 표시용으로 정리한다."""

        def _arr(value):
            return None if value is None else np.asarray(value, dtype=float).reshape(-1).tolist()

        def _subsample(points, limit):
            pts = np.asarray(points, dtype=float).reshape(-1, 3)
            if len(pts) > limit:
                pts = pts[np.linspace(0, len(pts) - 1, limit).astype(int)]
            return pts.tolist()

        roi = debug_info.get("pipe_profile_sampling_cylinder") or {}
        clusters = debug_info.get("pipe_profile_clusters") or []
        estimated_radius = debug_info.get("estimated_radius")
        fit = debug_info.get("pipe_profile_fit")
        # cluster 방식은 최종 피팅 입력(points_in_sphere), RANSAC 방식은 인라이어를 피팅 점으로 본다.
        fit_points = debug_info.get("points_in_sphere")
        if fit_points is None:
            fit_points = debug_info.get("pipe_profile_inliers")
        extent = None
        if fit is not None and fit_points is not None and len(fit_points):
            axis = np.asarray(fit["direction"], dtype=float)
            axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
            proj = (np.asarray(fit_points, dtype=float).reshape(-1, 3) - np.asarray(fit["center"], dtype=float)) @ axis
            extent = [float(proj.min()), float(proj.max())]
        validation = debug_info.get("pipe_profile_validation") or {}
        return {
            "point_index": int(point_index),
            "input_point": _arr(input_point),
            "surface_point": _arr(debug_info.get("pipe_profile_target_point")),
            "normal_axis": _arr(debug_info.get("pipe_profile_normal_axis")),
            "roi": {
                "start": _arr(roi.get("start")),
                "axis": _arr(roi.get("axis")),
                "radius": roi.get("radius"),
                "height_range": list(roi.get("height_range") or []),
            } if roi else None,
            "clusters": [
                {"count": len(c), "points": _subsample(c, max_cluster_points)}
                for c in clusters[:8] if len(c)
            ],
            "cluster_count": len(clusters),
            "estimated_center": _arr(debug_info.get("estimated_center")),
            "estimated_radius": None if estimated_radius is None else float(estimated_radius),
            "sphere_radius": None if estimated_radius is None else float(estimated_radius) + float(offset),
            "fit_points": [] if fit_points is None else _subsample(fit_points, max_fit_points),
            "fit": None if fit is None else {**fit, "extent": extent},
            "method": debug_info.get("pipe_profile_method"),
            "neighborhood": _subsample(debug_info["pipe_profile_neighborhood"], max_fit_points)
            if debug_info.get("pipe_profile_neighborhood") is not None else [],
            "validation": {
                "ok": bool(validation.get("ok", True)),
                "reasons": list(validation.get("reasons") or []),
                "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                            for k, v in (validation.get("metrics") or {}).items()},
            } if validation else None,
        }

    def _rejected_candidates(self, debug_info, point_index):
        """optimizer의 candidate_log에서 충돌로 탈락한 후보를 viewer 표시용(4x4 pose)으로 변환한다."""
        rejected = []
        for entry in (debug_info.get("candidate_log") or []):
            if not entry.get("collided"):
                continue
            pose = np.asarray(entry["pose"], dtype=float).reshape(-1)
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler("xyz", pose[3:6]).as_matrix()
            T[:3, 3] = pose[:3]
            part = str(entry.get("part", "DDA"))
            rejected.append({
                "point_index": int(point_index),
                "candidate_index": entry.get("candidate_index"),
                "angle_deg": entry.get("angle_deg"),
                "part": part,
                "robot": self._robot_name(part),
                "pose": T.tolist(),
                "hit": entry.get("hit"),
            })
        if rejected:
            counts = {}
            for item in rejected:
                obstacle = (item["hit"] or {}).get("obstacle", "unknown")
                key = f"{item['part']}:{obstacle}"
                counts[key] = counts.get(key, 0) + 1
            self.console.info(
                f"Robot Core pose rejected candidates: point={point_index + 1}, "
                f"total={len(rejected)}, by part:obstacle={counts}")
        return rejected

    def determine(self, inspection_points, spool_points):
        """Determine all DDA/RT target groups and return a complete response payload."""
        started = time.perf_counter()
        pipe_profiles = []
        rejected_candidates = []  # 실패 응답에도 실어 보내 viewer가 탈락 원인을 표시할 수 있게 한다.
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
            profile_params = PipeProfileParams.from_config(params)
            unknown_profile_keys = PipeProfileParams.unknown_keys(params)
            if unknown_profile_keys:
                self.console.warning(f"ef_pose.pipe_profile: unknown key(s) ignored: {unknown_profile_keys}")
            self.console.info(f"Robot Core pipe profile params: {profile_params}")
            setup_elapsed = time.perf_counter() - stage_started

            target_groups = []
            failures = []
            profile_elapsed = 0.0
            pose_elapsed = 0.0
            for point_index, target in enumerate(targets):
                try:
                    optimizer.debuging_info = {}
                    stage_started = time.perf_counter()
                    optimizer.calculate_pipe_profile(target, profile_params)
                    profile_elapsed += time.perf_counter() - stage_started
                    pipe_profiles.append(self._profile_debug(
                        optimizer.debuging_info, point_index, target, profile_params.sphere_radius_offset))

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
                    debug_info = getattr(optimizer, "debuging_info", {}) or {}
                    base_candidates = debug_info.get("dda_base_candidates")
                    valid_base_candidates = debug_info.get("valid_base_dda_poses")
                    summary = {
                        "base_candidates": None if base_candidates is None else len(base_candidates),
                        "valid_base_dda_poses": None if valid_base_candidates is None else len(valid_base_candidates),
                        "complete_groups": debug_info.get("complete_pose_group_count"),
                        "partial_groups": debug_info.get("partial_pose_group_count"),
                        "rejected_groups": debug_info.get("rejected_pose_group_count"),
                        "estimated_radius": debug_info.get("estimated_radius"),
                    }
                    self.console.info(
                        f"Robot Core pose candidates: point={point_index + 1}/{len(targets)}, "
                        f"target_point={np.round(target, 4).tolist()}, groups={len(point_groups)}, {summary}")
                    rejected_candidates.extend(self._rejected_candidates(debug_info, point_index))
                    if not point_groups:
                        failures.append({
                            "point_index": point_index,
                            "point": target.tolist(),
                            "message": "poseDeterminator returned no target group",
                            **summary,
                        })
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
                    if not any(p["point_index"] == point_index for p in pipe_profiles):
                        # 프로파일 피팅 자체가 실패한 경우에도 중간 결과(클러스터 등)를 남긴다.
                        pipe_profiles.append(self._profile_debug(
                            getattr(optimizer, "debuging_info", {}) or {}, point_index, target,
                            profile_params.sphere_radius_offset))
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
                "rejected_candidates": rejected_candidates,
                "pipe_profiles": pipe_profiles,
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
            return to_jsonable({
                "status": "failed",
                "source": "robot_core",
                "message": str(exc),
                "rejected_candidates": rejected_candidates,
                "pipe_profiles": pipe_profiles,
                "elapsed": elapsed,
            })
