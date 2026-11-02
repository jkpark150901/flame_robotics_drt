"""PyBullet end-effector pose checks against planning obstacle meshes.

PyBulletURDFLinkCollisionChecker uses the same URDF importer, collision
shapes and zero-distance threshold as PyBulletRobotScene. The older standalone
mesh checker remains available for callers without a URDF, but production pose
determination does not fall back to it or to point proximity checks.
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
    """The ONE PyBullet DIRECT client of the process, shared by the collision scene, the EF checker and the IK solver.

    Body ids only mean something inside the client that created them. The IK module used to open its own client; whichever
    connected first became the default client (id 0) of calls made without physicsClientId, so obstacle bodies were created
    in one client and queried in the other (wrong or missing contacts, and a native crash: Robot Core exitcode -11)."""
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


# PyBullet limits of one GEOM_MESH shape (measured: 131072 vertices / 524288 indices; more -> "createCollisionShape
# failed"), with some headroom
_MAX_SHAPE_VERTICES = 120000
_MAX_SHAPE_TRIANGLES = 160000


def _mesh_chunks(vertices: np.ndarray, triangles: np.ndarray):
    """[(vertices, triangles)] pieces of a triangle mesh, each within the per-shape limits (one piece if it fits).
    Triangles are taken in spatial order (sorted along the longest bounding-box axis) so the pieces are compact."""
    if len(vertices) <= _MAX_SHAPE_VERTICES and len(triangles) <= _MAX_SHAPE_TRIANGLES:
        return [(vertices, triangles)]
    centres = vertices[triangles].mean(axis=1)
    axis = int(np.argmax(np.ptp(vertices, axis=0)))
    order = np.argsort(centres[:, axis], kind="stable")
    n_tri = min(_MAX_SHAPE_TRIANGLES, _MAX_SHAPE_VERTICES // 3)   # 3 vertices per triangle at most
    chunks = []
    for start in range(0, len(order), n_tri):
        tri = triangles[order[start:start + n_tri]]
        used, inverse = np.unique(tri.ravel(), return_inverse=True)
        chunks.append((vertices[used], inverse.reshape(-1, 3).astype(np.int32)))
    return chunks


_CHUNK_BODIES: Dict[int, List[int]] = {}   # first body id -> all chunk body ids of one big static mesh


def body_chunks(body: int) -> List[int]:
    return _CHUNK_BODIES.get(int(body), [int(body)])


def closest_points(body_a: int, body_b: int, distance: float, **kwargs):
    """pb.getClosestPoints over every chunk of body_a / body_b (see make_static_mesh_body). link-index kwargs apply to
    the side that is not chunked (a chunk is a single-base body)."""
    if int(body_a) not in _CHUNK_BODIES and int(body_b) not in _CHUNK_BODIES:
        return pb.getClosestPoints(body_a, body_b, distance, **kwargs)
    out = []
    for a in body_chunks(body_a):
        for b in body_chunks(body_b):
            out.extend(pb.getClosestPoints(a, b, distance, **kwargs) or [])
    return out


# Search radii for min_distance(). One getClosestPoints call against the dense scanned pipe with a 0.3 m radius returns
# ~95k points and their distances are garbage (-3437 m, -3e283; p1 Point 1 pose 1 DDA seed waypoint 3: true 1.5 mm, 0.2 m
# radius still fine with ~34k points) - the radius grows only until something is found, so the big query never runs near
# the pipe.
_RADII = (0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)
_MIN_PLAUSIBLE = -0.5   # deeper penetration than this is a broken result, not geometry


def min_distance(body_a: int, body_b: int, max_distance: float, **kwargs) -> float:
    """Smallest signed distance between body_a and body_b (closest_points), max_distance when nothing is that close.

    Searches growing radii up to max_distance: the first radius r_k with any point holds the true minimum, and since
    nothing was found within r_(k-1) that minimum lies in (r_(k-1), r_k]. Large-radius queries return garbage (0 m,
    -0.5 m, -3e283 with nothing within 0.3 m - p2 Point 1 pose 1 RT link_slide), so values outside that interval are
    dropped; if none is left the interval is searched again with finer radii, and r_(k-1) (a lower bound) is returned
    when that still finds nothing plausible."""
    max_distance = float(max_distance)
    radii = [r for r in _RADII if r < max_distance] + [max_distance]

    def plausible(points, lower):
        ds = [p[8] for p in points if np.isfinite(p[8]) and p[8] >= _MIN_PLAUSIBLE and p[8] > lower]
        return min(ds) if ds else None

    prev = None
    for r in radii:
        points = closest_points(body_a, body_b, r, **kwargs)
        if points:
            lower = -np.inf if prev is None else prev - 1e-6   # penetration is only possible below the first radius
            d = plausible(points, lower)
            if d is None and prev is not None:
                for rr in np.linspace(prev, r, 6)[1:-1]:
                    finer = closest_points(body_a, body_b, float(rr), **kwargs)
                    d = plausible(finer, lower) if finer else None
                    if d is not None:
                        break
            if d is None:
                return float(prev) if prev is not None else 0.0
            return float(min(d, max_distance))
        prev = r
    return max_distance


def move_static_body(body: int, position, orientation, client: int) -> None:
    for b in body_chunks(body):
        pb.resetBasePositionAndOrientation(b, position, orientation, physicsClientId=client)


def remove_static_body(body: int, client: int) -> None:
    for b in body_chunks(body):
        try:
            pb.removeBody(b, physicsClientId=client)
        except Exception:
            pass
    _CHUNK_BODIES.pop(int(body), None)


def make_static_mesh_body(o3d_mesh: o3d.geometry.TriangleMesh, concave: bool = True, mass: float = 0.0) -> int:
    """Static (mass 0) PyBullet body from a triangle mesh. concave=True keeps the exact triangle mesh (BVH) - needed for
    obstacles such as the C-shaped positioner columns, whose convex hull would swallow the space the arms work in;
    concave=False gives the convex hull, which is what PyBullet builds for every robot-link mesh loaded from a URDF
    (so an EF mesh checked as a hull matches the same link checked inside the robot scene)."""
    client = _ensure_client()
    vertices = np.asarray(o3d_mesh.vertices, dtype=float)
    triangles = np.asarray(o3d_mesh.triangles, dtype=np.int32)
    if vertices.size == 0 or triangles.size == 0:
        raise RuntimeError("mesh has no geometry to build a PyBullet collision shape from")
    if concave:
        chunks = _mesh_chunks(vertices, triangles)
        bodies = []
        for v, t in chunks:
            shape = pb.createCollisionShape(
                pb.GEOM_MESH, vertices=v.tolist(), indices=t.flatten().tolist(),
                flags=pb.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=client)
            bodies.append(pb.createMultiBody(baseMass=float(mass), baseCollisionShapeIndex=shape, physicsClientId=client))
        # a mesh larger than one PyBullet shape allows (a scanned pipe loaded without downsampling) becomes several static
        # bodies, one per chunk (triangles unchanged, no decimation). Callers keep the first id as "the obstacle body";
        # closest_points()/move_static_body()/remove_static_body() act on all of its chunks. (Chunks as fixed links of one
        # multibody do not work: PyBullet reports distance 0 for triangle-mesh links.)
        if len(bodies) > 1:
            _CHUNK_BODIES[bodies[0]] = bodies
        return bodies[0]
    else:
        # vertices only -> PyBullet builds the convex hull. Passing indices would make a triangle-mesh shape, and a
        # triangle mesh against another triangle mesh has no collision algorithm (always "no contact").
        shape_id = pb.createCollisionShape(pb.GEOM_MESH, vertices=vertices.tolist(), physicsClientId=client)
    return pb.createMultiBody(baseMass=float(mass), baseCollisionShapeIndex=shape_id, physicsClientId=client)


class PyBulletCollisionChecker:
    """One PyBullet static body per given obstacle mesh (pipe, positioner, ...),
    plus one lazily-created body per distinct EF link_model - reused across
    every candidate pose check in a single PoseDeterminationService.determine()
    call, since the mesh geometry never changes between candidates, only
    the EF's world pose does (moved via resetBasePositionAndOrientation
    instead of recreating the collision shape every call - createCollisionShape
    is comparatively expensive and this runs once per candidate pose,
    typically hundreds of times per determine() call)."""

    def __init__(self, obstacle_meshes: List[o3d.geometry.TriangleMesh], obstacle_names: Optional[List[str]] = None,
                 threshold: float = 0.001):
        _ensure_client()
        meshes = [m for m in obstacle_meshes if m is not None and m.has_triangles()]
        names = list(obstacle_names or [])
        self._obstacle_names = [names[i] if i < len(names) else f"obstacle{i}" for i in range(len(meshes))]
        self.threshold = float(threshold)
        self.last_hit: Optional[Dict[str, Any]] = None
        self._obstacle_bodies = [self._make_static_body(mesh, concave=True) for mesh in meshes]
        if not self._obstacle_bodies:
            raise RuntimeError("PyBulletCollisionChecker: no obstacle mesh with triangles given")
        self._ef_bodies: Dict[int, int] = {}  # id(link_model) -> pybullet body id
        self._ef_mesh_refs: Dict[int, Any] = {}  # keeps link_model alive so id() can't be reused

    @staticmethod
    def _make_static_body(o3d_mesh: o3d.geometry.TriangleMesh, concave: bool = True) -> int:
        # The moving EF body must have mass > 0: PyBullet's broadphase never reports static-vs-static pairs, so two
        # mass-0 bodies always came back "no contact" (nothing is stepped, the mass only enables the pair).
        return make_static_mesh_body(o3d_mesh, concave=concave, mass=0.0 if concave else 1.0)

    def _ef_body_for(self, link_model) -> int:
        key = id(link_model)
        body = self._ef_bodies.get(key)
        if body is None:
            body = self._make_static_body(link_model, concave=False)
            self._ef_bodies[key] = body
            self._ef_mesh_refs[key] = link_model
        return body

    def __call__(self, link_model, tcp_pose, tcp_to_link_pose_T, margin: float = 0.05, sample_count: int = 5000) -> bool:
        return self.check(link_model, tcp_pose, tcp_to_link_pose_T, margin, sample_count, self.threshold)

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
        pb.resetBasePositionAndOrientation(ef_body, pos.tolist(), quat.tolist(), physicsClientId=_ensure_client())

        self.last_hit = None
        for name, obstacle_body in zip(self._obstacle_names, self._obstacle_bodies):
            points = closest_points(obstacle_body, ef_body, float(threshold), physicsClientId=_ensure_client())
            if points:
                nearest = min(points, key=lambda c: c[8])
                self.last_hit = {"obstacle": name, "distance": float(nearest[8]), "point": list(nearest[5])}
                return True
        return False


class PyBulletURDFLinkCollisionChecker(PyBulletCollisionChecker):
    """Pose-only checks using the exact URDF importer used by PyBulletRobotScene.

    Each probe is an independent URDF body. Only the requested end link is
    queried, so no IK is needed and the planning robot's joint state is untouched.
    """

    def __init__(self, obstacle_meshes, obstacle_names=None):
        super().__init__(obstacle_meshes, obstacle_names, threshold=0.0)
        self._probes = {}

    def register_link(self, link_model, urdf_path, link_name):
        from plugins.robotics.pybullet_scene import _quiet_stdio
        client = _ensure_client()
        with _quiet_stdio():
            body = pb.loadURDF(str(urdf_path), useFixedBase=True, physicsClientId=client)
        try:
            base_name = pb.getBodyInfo(body, physicsClientId=client)[0].decode()
            indices = {base_name: -1}
            for i in range(pb.getNumJoints(body, physicsClientId=client)):
                indices[pb.getJointInfo(body, i, physicsClientId=client)[12].decode()] = i
            index = indices[link_name]
            if not pb.getCollisionShapeData(body, index, physicsClientId=client):
                raise ValueError(f"URDF link has no collision geometry: {link_name}")
            base = self._transform(*pb.getBasePositionAndOrientation(body, physicsClientId=client))
            if index < 0:
                info = pb.getDynamicsInfo(body, -1, physicsClientId=client)
                link = base @ np.linalg.inv(self._transform(info[3], info[4]))
            else:
                state = pb.getLinkState(body, index, computeForwardKinematics=True, physicsClientId=client)
                link = self._transform(state[4], state[5])
            self._probes[id(link_model)] = (body, index, np.linalg.inv(link) @ base)
            self._ef_mesh_refs[id(link_model)] = link_model
        except Exception:
            pb.removeBody(body, physicsClientId=client)
            raise

    @staticmethod
    def _transform(position, quaternion):
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(quaternion).as_matrix()
        transform[:3, 3] = position
        return transform

    def check(self, link_model, tcp_pose, tcp_to_link_pose_T,
              margin=0.05, sample_count=5000, threshold=0.0):
        # Planning's end-link/static collision threshold is zero. Sampling and
        # legacy pose thresholds must not change the acceptance criterion.
        pose = np.asarray(tcp_pose, dtype=float).reshape(-1)
        offset = np.asarray(tcp_to_link_pose_T, dtype=float)
        if pose.size != 6 or offset.shape != (4, 4):
            raise ValueError("expected a 6D candidate pose and 4x4 link offset")
        transform = np.eye(4)
        transform[:3, :3] = R.from_euler('xyz', pose[3:]).as_matrix()
        transform[:3, 3] = pose[:3]
        body, index, link_to_base = self._probes[id(link_model)]
        base = transform @ offset @ link_to_base
        client = _ensure_client()
        pb.resetBasePositionAndOrientation(body, base[:3, 3].tolist(),
            R.from_matrix(base[:3, :3]).as_quat().tolist(), physicsClientId=client)
        self.last_hit = None
        for name, obstacle in zip(self._obstacle_names, self._obstacle_bodies):
            contacts = closest_points(body, obstacle, 0.0, linkIndexA=index, physicsClientId=client)
            if contacts and min(c[8] for c in contacts) <= 0.0:
                hit = min(contacts, key=lambda c: c[8])
                self.last_hit = {'obstacle': name, 'distance': float(hit[8]), 'point': list(hit[6])}
                return True
        return False

    def close(self):
        client = _ensure_client()
        for body, _, _ in getattr(self, '_probes', {}).values():
            pb.removeBody(body, physicsClientId=client)
        self._probes = {}
        self._ef_mesh_refs = {}
        for body in getattr(self, '_obstacle_bodies', []):
            remove_static_body(body, client)
        self._obstacle_bodies = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
