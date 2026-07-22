from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from urdf_parser_py.urdf import URDF

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover - depends on runtime environment
    pin = None

try:
    import hppfcl
except ImportError:  # pragma: no cover - depends on runtime environment
    try:
        import coal as hppfcl
    except ImportError:
        hppfcl = None

from plugins.robotics.backend import (
    CollisionResult,
    IKOptions,
    IKResult,
    IKTracePoint,
    RobotDescription,
    RoboticsBackend,
)
from plugins.robotics.ik_solver import (
    damped_least_squares_step,
    normalized_damped_least_squares_step,
    solve_qp_ik_step,
)


@dataclass
class PinocchioRobotHandle:
    description: RobotDescription
    model: Any
    data: Any
    geom_model: Any = None
    geom_data: Any = None
    robot_geom_ids: List[int] = field(default_factory=list)
    static_object_ids: List[int] = field(default_factory=list)
    sample_resolution: float = 0.05


class PinocchioRoboticsBackend(RoboticsBackend):
    """RoboticsBackend implementation using Pinocchio and hpp-fcl/coal."""

    name = "pinocchio"

    def __init__(self):
        if pin is None:
            raise RuntimeError("pinocchio is not available")
        self._robots: Dict[str, PinocchioRobotHandle] = {}

    def register_robot(self, description: RobotDescription) -> PinocchioRobotHandle:
        urdf_path = os.path.abspath(description.urdf_path)
        model = self._build_model_from_urdf(urdf_path)
        handle = PinocchioRobotHandle(
            description=RobotDescription(
                name=description.name,
                urdf_path=urdf_path,
                base_T=np.asarray(description.base_T, dtype=float).copy(),
                package_dirs=list(description.package_dirs or [os.path.dirname(urdf_path)]),
                target_frame=description.target_frame,
            ),
            model=model,
            data=model.createData(),
        )
        self._robots[description.name] = handle
        return handle

    def robot_model(self, robot_name: str) -> Any:
        return self._handle(robot_name).model

    def robot_handle(self, robot_name: str) -> PinocchioRobotHandle:
        return self._handle(robot_name)

    def dof(self, robot_name: str) -> int:
        return int(self._handle(robot_name).model.nq)

    def joint_names(self, robot_name: str) -> List[str]:
        model = self._handle(robot_name).model
        return [str(model.names[i]) for i in range(1, model.njoints)]

    def neutral_q(self, robot_name: str) -> np.ndarray:
        model = self._handle(robot_name).model
        try:
            return np.asarray(pin.neutral(model), dtype=float)
        except Exception:
            return np.zeros(model.nq, dtype=float)

    def joint_limits_for_metric(self, robot_name: str, normalize: bool = True):
        if not bool(normalize):
            return None, None, None
        model = self._handle(robot_name).model
        lo = np.asarray(model.lowerPositionLimit, dtype=float).copy()
        hi = np.asarray(model.upperPositionLimit, dtype=float).copy()
        invalid = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
        lo[invalid] = -np.pi
        hi[invalid] = np.pi
        span = hi - lo
        span[span < 1e-9] = 1.0
        return lo, hi, span

    def normalize_q(self, robot_name: str, q: Sequence[float], normalize: bool = True) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        lo, _, span = self.joint_limits_for_metric(robot_name, normalize=normalize)
        if span is None:
            return q.copy()
        return (q - lo) / span

    def denormalize_q(self, robot_name: str, q_metric: Sequence[float], normalize: bool = True) -> np.ndarray:
        q_metric = np.asarray(q_metric, dtype=float)
        lo, hi, span = self.joint_limits_for_metric(robot_name, normalize=normalize)
        if span is None:
            return q_metric.copy()
        return np.minimum(np.maximum(lo + q_metric * span, lo), hi)

    def joint_distance(
        self,
        robot_name: str,
        q_a: Sequence[float],
        q_b: Sequence[float],
        normalize: bool = True,
    ) -> float:
        a = self.normalize_q(robot_name, q_a, normalize=normalize)
        b = self.normalize_q(robot_name, q_b, normalize=normalize)
        return float(np.linalg.norm(b - a))

    def joint_distances(
        self,
        robot_name: str,
        q_points: Sequence[Sequence[float]],
        q_ref: Sequence[float],
        normalize: bool = True,
    ) -> np.ndarray:
        pts = np.asarray(q_points, dtype=float)
        ref = np.asarray(q_ref, dtype=float)
        if pts.ndim == 1:
            return np.asarray([self.joint_distance(robot_name, pts, ref, normalize=normalize)], dtype=float)
        lo, _, span = self.joint_limits_for_metric(robot_name, normalize=normalize)
        if span is None:
            return np.linalg.norm(pts - ref, axis=1)
        return np.linalg.norm(((pts - lo) / span) - ((ref - lo) / span), axis=1)

    def steer_joint_state(
        self,
        robot_name: str,
        from_state: Sequence[float],
        to_state: Sequence[float],
        step_size: float,
        normalize: bool = True,
    ) -> np.ndarray:
        if not bool(normalize):
            direction = np.asarray(to_state, dtype=float) - np.asarray(from_state, dtype=float)
            length = float(np.linalg.norm(direction))
            if length < 1e-12:
                return np.asarray(from_state, dtype=float).copy()
            return np.asarray(from_state, dtype=float) + direction / length * min(float(step_size), length)

        from_norm = self.normalize_q(robot_name, from_state, normalize=True)
        to_norm = self.normalize_q(robot_name, to_state, normalize=True)
        direction = to_norm - from_norm
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            return np.asarray(from_state, dtype=float).copy()
        new_norm = from_norm + direction / length * min(float(step_size), length)
        return self.denormalize_q(robot_name, new_norm, normalize=True)

    def sample_configuration(self, robot_name: str) -> np.ndarray:
        lo, hi, _span = self.joint_limits_for_metric(robot_name, normalize=True)
        return np.random.uniform(lo, hi)

    def end_effector_collision_geometry(
        self,
        robot_name: str,
        end_link_name: str,
        tcp_joint_name,
        pose_to_link_offset=None,
    ):
        handle = self._handle(robot_name)
        urdf = URDF.from_xml_file(handle.description.urdf_path)
        link = urdf.link_map[end_link_name]
        collision = getattr(link, "collision", None)
        if collision is None or collision.geometry is None:
            raise RuntimeError(f"collision geometry is not available: {robot_name}:{end_link_name}")

        mesh_path = Path(str(collision.geometry.filename).replace("file://", ""))
        if not mesh_path.is_absolute():
            mesh_path = Path(handle.description.urdf_path).resolve().parent / mesh_path
        mesh_path = mesh_path.resolve()
        link_mesh = o3d.io.read_triangle_mesh(str(mesh_path))

        scale = getattr(collision.geometry, "scale", 1.0)
        if isinstance(scale, list):
            if len(scale) == 3 and not np.allclose(scale, scale[0]):
                vertices = np.asarray(link_mesh.vertices, dtype=float)
                link_mesh.vertices = o3d.utility.Vector3dVector(vertices * np.asarray(scale, dtype=float))
            else:
                link_mesh.scale(float(scale[0]), np.zeros(3, dtype=np.float64))
        elif isinstance(scale, (int, float)):
            link_mesh.scale(float(scale), np.zeros(3, dtype=np.float64))

        origin = getattr(collision, "origin", None)
        T_collision = np.eye(4)
        if origin is not None:
            T_collision[:3, :3] = R.from_euler("xyz", origin.rpy).as_matrix()
            T_collision[:3, 3] = origin.xyz
        link_mesh.transform(T_collision)

        if pose_to_link_offset is not None:
            return link_mesh, self._offset_to_transform(pose_to_link_offset)

        tcp_joint_names = (tcp_joint_name,) if isinstance(tcp_joint_name, str) else tuple(tcp_joint_name)
        tcp_joint = None
        for candidate_name in tcp_joint_names:
            tcp_joint = urdf.joint_map.get(candidate_name)
            if tcp_joint is not None:
                break
        if tcp_joint is None:
            raise KeyError(f"TCP joint not found. candidates={tcp_joint_names}")

        fallback_T = self._joint_origin_T(tcp_joint)
        end_to_tcp_T = self._relative_link_transform(
            urdf,
            end_link_name,
            tcp_joint.child,
            fallback_T=fallback_T,
        )
        return link_mesh, np.linalg.inv(end_to_tcp_T)

    def frame_id(self, robot_name: str, frame_name: Optional[str] = None) -> int:
        handle = self._handle(robot_name)
        target = frame_name or handle.description.target_frame
        if target:
            try:
                fid = handle.model.getFrameId(str(target))
                if fid < handle.model.nframes:
                    return fid
            except Exception:
                pass
        return handle.model.nframes - 1

    def frame_world_T(self, robot_name: str, q: Sequence[float], frame_name: Optional[str] = None) -> np.ndarray:
        handle = self._handle(robot_name)
        q = np.asarray(q, dtype=float)
        pin.forwardKinematics(handle.model, handle.data, q)
        pin.updateFramePlacements(handle.model, handle.data)
        local_T = np.asarray(handle.data.oMf[self.frame_id(robot_name, frame_name)].homogeneous, dtype=float)
        return np.asarray(handle.description.base_T, dtype=float) @ local_T

    def target_world_T(
        self,
        robot_name: str,
        target_world: Any,
        q_reference: Sequence[float],
        frame_name: Optional[str] = None,
    ) -> np.ndarray:
        target_arr = np.asarray(target_world, dtype=float)
        if target_arr.shape == (4, 4):
            return target_arr.copy()
        if target_arr.size >= 6:
            pose = target_arr.reshape(-1)[:6]
            T = np.eye(4)
            T[:3, :3] = R.from_euler("xyz", pose[3:6]).as_matrix()
            T[:3, 3] = pose[:3]
            return T
        current_world_T = self.frame_world_T(robot_name, q_reference, frame_name)
        target_T = current_world_T.copy()
        target_T[:3, 3] = target_arr.reshape(-1)[:3]
        return target_T

    def solve_ik(
        self,
        robot_name: str,
        target_world_T: np.ndarray,
        q_init: Sequence[float],
        options: Optional[IKOptions] = None,
        frame_name: Optional[str] = None,
    ) -> IKResult:
        options = options or IKOptions()
        handle = self._handle(robot_name)
        model = handle.model
        data = model.createData()
        fid = self.frame_id(robot_name, frame_name)
        frame_name = frame_name or handle.description.target_frame or str(model.frames[fid].name)

        target_world_T = np.asarray(target_world_T, dtype=float)
        target_local_T = np.linalg.inv(handle.description.base_T) @ target_world_T
        target_se3 = pin.SE3(target_local_T[:3, :3], target_local_T[:3, 3])

        solver_name = str(options.solver or "normalized_dls").lower()
        use_qp = solver_name in ("qp", "qp_ik")
        normalize = bool(options.normalize) if options.normalize is not None else solver_name not in (
            "dls",
            "classic_dls",
            "legacy_dls",
        )
        solver_label = "qp" if use_qp else ("dls_normalized" if normalize else "dls")

        t0 = time.perf_counter()
        q = np.asarray(q_init, dtype=float).copy()
        trace: List[IKTracePoint] = []

        def record(iteration: int):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            err_vec = pin.log6(data.oMf[fid].inverse() * target_se3).vector
            current_T = np.asarray(data.oMf[fid].homogeneous, dtype=float)
            position_error = float(np.linalg.norm(current_T[:3, 3] - target_local_T[:3, 3]))
            rot_delta = current_T[:3, :3].T @ target_local_T[:3, :3]
            cos_angle = (float(np.trace(rot_delta)) - 1.0) * 0.5
            orientation_error = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            if options.record_trace:
                trace.append(
                    IKTracePoint(
                        iteration=int(iteration),
                        q=q.copy(),
                        err_norm=float(np.linalg.norm(err_vec)),
                        position_error=position_error,
                        orientation_error=orientation_error,
                        tcp_world=(handle.description.base_T @ current_T)[:3, 3].copy(),
                    )
                )
            return err_vec, position_error, orientation_error

        err, position_error, orientation_error = record(0)
        for iteration in range(int(options.max_iter)):
            if np.linalg.norm(err) < float(options.tol):
                return self._ik_result(
                    True,
                    q,
                    solver_label,
                    normalize,
                    iteration,
                    time.perf_counter() - t0,
                    position_error,
                    orientation_error,
                    target_world_T,
                    trace,
                    robot_name,
                    frame_name,
                )

            if use_qp:
                q = solve_qp_ik_step(
                    model,
                    data,
                    q,
                    frame_name,
                    target_se3,
                    dt=options.dt,
                    solver=options.backend_solver,
                )
            elif normalize:
                q = normalized_damped_least_squares_step(
                    pin,
                    model,
                    data,
                    q,
                    fid,
                    err,
                    damping=options.damping,
                    dt=options.dt,
                )
            else:
                q = damped_least_squares_step(
                    pin,
                    model,
                    data,
                    q,
                    fid,
                    err,
                    damping=options.damping,
                    dt=options.dt,
                )
            err, position_error, orientation_error = record(iteration + 1)

        position_only = position_error < float(options.position_only_tol)
        final_q = q.copy()
        final_T = self.frame_world_T(robot_name, final_q, frame_name)
        failure_info = self.classify_ik_failure(
            robot_name,
            final_q,
            target_world_T,
            final_T,
            orientation_error=orientation_error,
            max_iter=options.max_iter,
        )
        return self._ik_result(
            position_only,
            final_q,
            solver_label,
            normalize,
            int(options.max_iter),
            time.perf_counter() - t0,
            position_error,
            orientation_error,
            target_world_T,
            trace,
            robot_name,
            frame_name,
            position_only=position_only,
            failure_info={} if position_only else failure_info,
            message="" if position_only else "IK failed to converge",
        )

    def classify_ik_failure(
        self,
        robot_name: str,
        q: Sequence[float],
        target_world_T: np.ndarray,
        final_T: Optional[np.ndarray],
        orientation_error: float = float("inf"),
        max_iter: int = 0,
    ) -> Dict[str, Any]:
        handle = self._handle(robot_name)
        model = handle.model
        q = np.asarray(q, dtype=float)
        lo = np.asarray(model.lowerPositionLimit, dtype=float)
        hi = np.asarray(model.upperPositionLimit, dtype=float)
        names = self.joint_names(robot_name)
        saturated = []
        for i, joint_name in enumerate(names[:len(q)]):
            lower = lo[i] if i < len(lo) else -np.inf
            upper = hi[i] if i < len(hi) else np.inf
            if not np.isfinite(lower) or not np.isfinite(upper):
                continue
            span = max(abs(float(upper - lower)), 1.0)
            eps = max(1e-5, span * 1e-3)
            if q[i] <= lower + eps:
                saturated.append({
                    "joint": str(joint_name),
                    "side": "lower",
                    "value": float(q[i]),
                    "limit": float(lower),
                })
            elif q[i] >= upper - eps:
                saturated.append({
                    "joint": str(joint_name),
                    "side": "upper",
                    "value": float(q[i]),
                    "limit": float(upper),
                })

        target_world_T = np.asarray(target_world_T, dtype=float)
        position_error = float("inf")
        if final_T is not None:
            position_error = float(np.linalg.norm(np.asarray(final_T)[:3, 3] - target_world_T[:3, 3]))
        if saturated and position_error > 0.05:
            failure_type = "likely_reach_or_joint_limit"
        elif saturated:
            failure_type = "joint_limit_saturation"
        elif position_error > 0.25:
            failure_type = "likely_unreachable"
        else:
            failure_type = "ik_non_convergence"
        return {
            "type": failure_type,
            "robot": robot_name,
            "position_error": position_error,
            "orientation_error": float(orientation_error),
            "max_iter": int(max_iter),
            "saturated_joints": saturated,
            "final_q": q.tolist(),
            "target_position": target_world_T[:3, 3].tolist(),
            "final_position": None if final_T is None else np.asarray(final_T)[:3, 3].tolist(),
            "final_T": None if final_T is None else np.asarray(final_T).tolist(),
            "target_T": target_world_T.tolist(),
        }

    def configure_collision(
        self,
        robot_name: str,
        static_meshes: Optional[Iterable[Any]] = None,
        sample_resolution: float = 0.05,
    ) -> None:
        if hppfcl is None:
            raise RuntimeError("hppfcl/coal is not available")
        handle = self._handle(robot_name)
        package_dirs = handle.description.package_dirs or [os.path.dirname(handle.description.urdf_path)]
        handle.geom_model = pin.buildGeomFromUrdf(
            handle.model,
            handle.description.urdf_path,
            pin.GeometryType.COLLISION,
            None,
            [os.path.abspath(p) for p in package_dirs],
        )
        handle.geom_model.addAllCollisionPairs()
        self._remove_adjacent_pairs(handle)
        handle.robot_geom_ids = list(range(len(handle.geom_model.geometryObjects)))
        handle.static_object_ids = []
        handle.sample_resolution = float(sample_resolution)
        for mesh in static_meshes or []:
            self._add_static_mesh(handle, mesh, recreate_data=False)
        handle.geom_data = pin.GeometryData(handle.geom_model)

    def check_collision(self, robot_name: str, q: Sequence[float], return_pairs: bool = False) -> CollisionResult:
        handle = self._handle(robot_name)
        self._require_collision(handle)
        q = np.asarray(q, dtype=float)
        has_collision = bool(
            pin.computeCollisions(
                handle.model,
                handle.data,
                handle.geom_model,
                handle.geom_data,
                q,
                False,
            )
        )
        pairs = self._active_collision_pairs(handle) if return_pairs else []
        return CollisionResult(has_collision, pairs=pairs, q=q.copy(), backend=self.name)

    def check_edge_collision(
        self,
        robot_name: str,
        q_from: Sequence[float],
        q_to: Sequence[float],
        return_pairs: bool = False,
    ) -> CollisionResult:
        handle = self._handle(robot_name)
        self._require_collision(handle)
        q_from = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)
        dist = float(np.linalg.norm(q_to - q_from))
        steps = max(1, int(np.ceil(dist / max(float(handle.sample_resolution), 1e-9))))
        for i in range(steps + 1):
            alpha = i / steps
            q = (1.0 - alpha) * q_from + alpha * q_to
            result = self.check_collision(robot_name, q, return_pairs=return_pairs)
            if result.collision:
                result.alpha = float(alpha)
                return result
        return CollisionResult(False, backend=self.name)

    def check_mesh_point_cloud_overlap(
        self,
        link_model: Any,
        tcp_pose: Sequence[float],
        tcp_to_link_pose_T: np.ndarray,
        scan_data: Any,
        margin: float = 0.05,
        sample_count: int = 5000,
        threshold: float = 0.001,
    ) -> bool:
        tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(-1)
        if tcp_pose.size < 6:
            raise ValueError(f"tcp_pose must have at least 6 values, got {tcp_pose.size}")
        tcp_to_link_pose_T = np.asarray(tcp_to_link_pose_T, dtype=float)
        if tcp_to_link_pose_T.shape != (4, 4):
            raise ValueError(f"tcp_to_link_pose_T must be 4x4, got {tcp_to_link_pose_T.shape}")

        tcp_pose_T = np.eye(4)
        tcp_pose_T[:3, :3] = R.from_euler("xyz", tcp_pose[3:6]).as_matrix()
        tcp_pose_T[:3, 3] = tcp_pose[:3]
        link_pose_T = tcp_pose_T @ tcp_to_link_pose_T

        mesh_copy = copy.deepcopy(link_model)
        mesh_copy.transform(link_pose_T)

        aabb = mesh_copy.get_axis_aligned_bounding_box()
        margin_vec = np.array([margin, margin, margin], dtype=float)
        crop_box = o3d.geometry.AxisAlignedBoundingBox(
            aabb.min_bound - margin_vec,
            aabb.max_bound + margin_vec,
        )
        idx = crop_box.get_point_indices_within_bounding_box(scan_data.points)
        if not idx:
            return False

        sub_pcd = scan_data.select_by_index(idx)
        mesh_pcd = mesh_copy.sample_points_uniformly(number_of_points=int(sample_count))
        distances = sub_pcd.compute_point_cloud_distance(mesh_pcd)
        return any(float(distance) <= float(threshold) for distance in distances)

    def collision_model_cache(self, robot_name: str) -> Dict[str, Any]:
        handle = self._handle(robot_name)
        return {
            "pin_model": handle.model,
            "pin_geom_model": copy.deepcopy(handle.geom_model) if handle.geom_model is not None else None,
            "robot_geom_ids": list(handle.robot_geom_ids),
        }

    def collision_geometry_summary(self, robot_name: str) -> List[Dict[str, Any]]:
        handle = self._handle(robot_name)
        if handle.geom_model is None:
            return []
        static_ids = set(handle.static_object_ids)
        names = list(handle.model.names)
        summary = []
        for geom_id, geom in enumerate(handle.geom_model.geometryObjects):
            parent_joint = int(geom.parentJoint)
            joint_name = names[parent_joint] if 0 <= parent_joint < len(names) else str(parent_joint)
            summary.append({
                "id": int(geom_id),
                "name": str(geom.name),
                "parent_joint": parent_joint,
                "parent_joint_name": str(joint_name),
                "kind": "static" if geom_id in static_ids else "robot",
            })
        return summary

    def collision_pair_summary(
        self,
        robot_name: str,
        include_robot_self: bool = True,
        include_static: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        handle = self._handle(robot_name)
        if handle.geom_model is None:
            return []
        static_ids = set(handle.static_object_ids)
        pairs = []
        for pair_id, pair in enumerate(handle.geom_model.collisionPairs):
            first = handle.geom_model.geometryObjects[pair.first]
            second = handle.geom_model.geometryObjects[pair.second]
            first_static = int(pair.first) in static_ids
            second_static = int(pair.second) in static_ids
            is_static_pair = first_static or second_static
            if is_static_pair and not include_static:
                continue
            if not is_static_pair and not include_robot_self:
                continue
            pairs.append({
                "id": int(pair_id),
                "first": str(first.name),
                "second": str(second.name),
                "kind": "robot_static" if is_static_pair else "robot_self",
            })
            if limit is not None and len(pairs) >= int(limit):
                break
        return pairs

    def _handle(self, robot_name: str) -> PinocchioRobotHandle:
        if robot_name not in self._robots:
            raise KeyError(f"robot is not registered: {robot_name}")
        return self._robots[robot_name]

    def _build_model_from_urdf(self, urdf_path: str):
        if hasattr(pin, "buildModelFromUrdf"):
            return pin.buildModelFromUrdf(urdf_path)
        if hasattr(pin, "buildModelFromURDF"):
            return pin.buildModelFromURDF(urdf_path)
        if hasattr(pin, "buildModelsFromUrdf"):
            models = pin.buildModelsFromUrdf(urdf_path)
            if isinstance(models, tuple) and models:
                return models[0]
        from pinocchio.robot_wrapper import RobotWrapper

        return RobotWrapper.BuildFromURDF(urdf_path).model

    @staticmethod
    def _offset_to_transform(offset) -> np.ndarray:
        if isinstance(offset, dict):
            if "matrix" in offset:
                T = np.asarray(offset["matrix"], dtype=float)
            else:
                xyz = np.asarray(offset.get("xyz", [0.0, 0.0, 0.0]), dtype=float)
                rpy = np.asarray(offset.get("rpy", [0.0, 0.0, 0.0]), dtype=float)
                T = np.eye(4)
                T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
                T[:3, 3] = xyz
        else:
            T = np.asarray(offset, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"pose_to_link_offset must be 4x4, got shape={T.shape}")
        return T.astype(float, copy=True)

    @staticmethod
    def _joint_origin_T(joint) -> np.ndarray:
        T = np.eye(4)
        origin = getattr(joint, "origin", None)
        if origin is not None:
            T[:3, :3] = R.from_euler("xyz", origin.rpy).as_matrix()
            T[:3, 3] = origin.xyz
        return T

    @classmethod
    def _relative_link_transform(cls, urdf: URDF, source_link_name: str, target_link_name: str, fallback_T):
        child_to_joint = {joint.child: joint for joint in urdf.joints}
        cache: Dict[str, np.ndarray] = {}

        def root_to_link(link_name: str) -> np.ndarray:
            if link_name in cache:
                return cache[link_name]
            joint = child_to_joint.get(link_name)
            if joint is None:
                cache[link_name] = np.eye(4)
                return cache[link_name]
            T = root_to_link(joint.parent) @ cls._joint_origin_T(joint)
            cache[link_name] = T
            return T

        try:
            return np.linalg.inv(root_to_link(source_link_name)) @ root_to_link(target_link_name)
        except Exception:
            return np.asarray(fallback_T, dtype=float)

    def _ik_result(
        self,
        success,
        q,
        solver,
        normalize,
        iterations,
        elapsed,
        position_error,
        orientation_error,
        target_T,
        trace,
        robot_name,
        frame_name,
        position_only=False,
        failure_info=None,
        message="",
    ) -> IKResult:
        final_T = self.frame_world_T(robot_name, q, frame_name) if q is not None else None
        return IKResult(
            success=bool(success),
            q=None if q is None else np.asarray(q, dtype=float).copy(),
            solver=solver,
            normalize=bool(normalize),
            iterations=int(iterations),
            elapsed=float(elapsed),
            position_only=bool(position_only),
            position_error=float(position_error),
            orientation_error=float(orientation_error),
            final_T=final_T,
            target_T=np.asarray(target_T, dtype=float).copy(),
            trace=trace,
            failure_info=dict(failure_info or {}),
            message=message,
        )

    def _require_collision(self, handle: PinocchioRobotHandle) -> None:
        if handle.geom_model is None or handle.geom_data is None:
            raise RuntimeError(f"collision is not configured: {handle.description.name}")

    def _add_static_mesh(self, handle: PinocchioRobotHandle, mesh, recreate_data=True):
        vertices = np.asarray(mesh.vertices, dtype=float)
        triangles = np.asarray(mesh.triangles if hasattr(mesh, "triangles") else mesh.faces, dtype=np.int32)
        if triangles.shape[1] > 3:
            triangles = triangles[:, :3]
        vec_vertices = hppfcl.StdVec_Vec3s()
        vec_triangles = hppfcl.StdVec_Triangle()
        for vertex in vertices:
            vec_vertices.append(vertex)
        for tri in triangles:
            vec_triangles.append(hppfcl.Triangle(int(tri[0]), int(tri[1]), int(tri[2])))
        bvh = hppfcl.BVHModelOBBRSS()
        bvh.beginModel(len(vec_vertices), len(vec_triangles))
        bvh.addSubModel(vec_vertices, vec_triangles)
        bvh.endModel()
        bvh.computeLocalAABB()
        geom_obj = pin.GeometryObject(
            f"collision_object_{len(handle.static_object_ids)}",
            0,
            pin.SE3.Identity(),
            bvh,
        )
        geom_id = handle.geom_model.addGeometryObject(geom_obj)
        for robot_geom_id in handle.robot_geom_ids:
            pair = pin.CollisionPair(robot_geom_id, geom_id)
            if not handle.geom_model.existCollisionPair(pair):
                handle.geom_model.addCollisionPair(pair)
        handle.static_object_ids.append(geom_id)
        if recreate_data:
            handle.geom_data = pin.GeometryData(handle.geom_model)
        return geom_id

    def _remove_adjacent_pairs(self, handle: PinocchioRobotHandle) -> None:
        kept_pairs = []
        for pair in list(handle.geom_model.collisionPairs):
            first = handle.geom_model.geometryObjects[pair.first]
            second = handle.geom_model.geometryObjects[pair.second]
            if self._is_adjacent_pair(handle.model, first.parentJoint, second.parentJoint):
                continue
            kept_pairs.append(pin.CollisionPair(pair.first, pair.second))
        handle.geom_model.removeAllCollisionPairs()
        for pair in kept_pairs:
            handle.geom_model.addCollisionPair(pair)

    def _is_adjacent_pair(self, model, joint_a, joint_b) -> bool:
        if joint_a == joint_b:
            return True
        try:
            return model.parents[joint_a] == joint_b or model.parents[joint_b] == joint_a
        except Exception:
            return False

    def _active_collision_pairs(self, handle: PinocchioRobotHandle) -> List[tuple]:
        pairs = []
        for idx, result in enumerate(handle.geom_data.collisionResults):
            if not result.isCollision():
                continue
            pair = handle.geom_model.collisionPairs[idx]
            first = handle.geom_model.geometryObjects[pair.first].name
            second = handle.geom_model.geometryObjects[pair.second].name
            pairs.append((str(first), str(second)))
        return pairs
