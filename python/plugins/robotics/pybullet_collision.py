"""PyBullet-based real geometric collision check for EF pose determination.

Replaces PinocchioRoboticsBackend.check_mesh_point_cloud_overlap()'s point-
cloud-proximity heuristic (EF mesh points vs nearest raw pipe scan point -
see pinocchio_backend.py:738-776, and EndEffectorPoseOptimizer's private
__check_collision, which duplicated the same heuristic independently) with
an actual solid mesh-vs-mesh collision query via PyBullet's
getClosestPoints() - the same kind of real geometric check path planning
already does (Pinocchio/FCL, pin.computeCollisions), instead of "is any
EF-mesh sample point within 1mm of a raw scan point".

Critically, this takes the SAME obstacle mesh objects planning already
collision-checks against - RobotCoreEngine._current_spool_collision_mesh()/
_build_positioner_collision_mesh() (robot_core/worker.py), which return the
exact triangle mesh baked into the planning snapshot - instead of
reconstructing a new (and possibly differently-shaped) mesh from raw scan
points here. A pose that clears pose-determination but then fails path
planning's goal_collision check (EF mesh vs the SAME pipe mesh, just via
Pinocchio instead of PyBullet) was exactly the bug this closes: two
different collision systems checking two different approximations of the
same pipe used to be able to disagree; now both check the identical mesh,
just through different geometry engines.
"""

from __future__ import annotations

import atexit
from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

try:
    import pybullet as pb
except ImportError:
    pb = None

_client_id: Optional[int] = None


def _ensure_client() -> int:
    global _client_id
    if pb is None:
        raise RuntimeError("pybullet is not installed - pip install pybullet")
    if _client_id is None:
        _client_id = pb.connect(pb.DIRECT)
        atexit.register(_disconnect_client)
    return _client_id


def _disconnect_client():
    global _client_id
    if pb is not None and _client_id is not None:
        try:
            pb.disconnect(_client_id)
        except Exception:
            pass
        _client_id = None


def build_pipe_mesh_from_scan(scan_data, alpha: float = 0.06,
                               max_points: int = 1000000) -> o3d.geometry.TriangleMesh:
    """Reconstruct a solid triangle mesh from a raw point cloud via Open3D
    alpha-shape. Fallback only - prefer passing planning's own already-built
    mesh (RobotCoreEngine._current_spool_collision_mesh()) to
    PyBulletCollisionChecker instead of reconstructing a separate one here;
    this exists for callers that only have a raw point cloud (no live
    RobotCoreEngine/mesh available).

    Downsamples to max_points first - alpha-shape reconstruction time blows
    up with point count, and a raw scan can be ~1M points.
    """
    points = np.asarray(scan_data.points)
    if max_points > 0 and len(points) > max_points:
        step = int(np.ceil(len(points) / max_points))
        scan_data = scan_data.select_by_index(np.arange(0, len(points), step))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(scan_data, alpha)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    if not mesh.has_triangles():
        raise RuntimeError(
            f"pipe mesh reconstruction (alpha-shape, alpha={alpha}) produced no triangles - "
            "try a larger alpha or check the scan point cloud")
    return mesh


class PyBulletCollisionChecker:
    """One PyBullet static body per given obstacle mesh (pipe, positioner, ...),
    plus one lazily-created body per distinct EF link_model - reused across
    every candidate pose check in a single PoseDeterminationService.determine()
    call, since the mesh geometry never changes between candidates, only
    the EF's world pose does (moved via resetBasePositionAndOrientation
    instead of recreating the collision shape every call - createCollisionShape
    is comparatively expensive and this runs once per candidate pose,
    typically hundreds of times per determine() call)."""

    def __init__(self, obstacle_meshes: List[o3d.geometry.TriangleMesh]):
        _ensure_client()
        self._obstacle_bodies = [
            self._make_static_body(mesh) for mesh in obstacle_meshes
            if mesh is not None and mesh.has_triangles()
        ]
        if not self._obstacle_bodies:
            raise RuntimeError("PyBulletCollisionChecker: no obstacle mesh with triangles given")
        self._ef_bodies: Dict[int, int] = {}  # id(link_model) -> pybullet body id
        self._ef_mesh_refs: Dict[int, Any] = {}  # keeps link_model alive so id() can't be reused

    @staticmethod
    def _make_static_body(o3d_mesh: o3d.geometry.TriangleMesh) -> int:
        vertices = np.asarray(o3d_mesh.vertices, dtype=float)
        triangles = np.asarray(o3d_mesh.triangles, dtype=np.int32)
        if vertices.size == 0 or triangles.size == 0:
            raise RuntimeError("mesh has no geometry to build a PyBullet collision shape from")
        shape_id = pb.createCollisionShape(
            pb.GEOM_MESH, vertices=vertices.tolist(), indices=triangles.flatten().tolist())
        return pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape_id)

    def _ef_body_for(self, link_model) -> int:
        key = id(link_model)
        body = self._ef_bodies.get(key)
        if body is None:
            body = self._make_static_body(link_model)
            self._ef_bodies[key] = body
            self._ef_mesh_refs[key] = link_model
        return body

    def check(self, link_model, tcp_pose, tcp_to_link_pose_T,
              margin: float = 0.05, sample_count: int = 5000, threshold: float = 0.001) -> bool:
        """Same (link_model, tcp_pose, tcp_to_link_pose_T, margin=, sample_count=)
        call signature as the old point-cloud-proximity checker (drop-in for
        EndEffectorPoseOptimizer.set_collision_checker / pose_service.py's
        collision-checker wiring) - margin/sample_count are accepted but
        unused (PyBullet's getClosestPoints needs neither an AABB crop nor
        surface sampling, it queries the exact meshes). True if the EF mesh
        at this candidate pose is within `threshold` of any obstacle mesh,
        or actually penetrating (PyBullet's reported distance can go
        negative)."""
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

        ef_body = self._ef_body_for(link_model)
        pos = link_pose_T[:3, 3]
        quat = R.from_matrix(link_pose_T[:3, :3]).as_quat()  # xyzw - matches pybullet's convention
        pb.resetBasePositionAndOrientation(ef_body, pos.tolist(), quat.tolist())

        for obstacle_body in self._obstacle_bodies:
            if pb.getClosestPoints(obstacle_body, ef_body, distance=float(threshold)):
                return True
        return False
