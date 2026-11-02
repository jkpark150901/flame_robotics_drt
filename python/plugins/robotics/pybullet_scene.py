"""PyBullet mesh-based collision scene of one robot: the single collision engine for planning, verification and metrics.

Pose determination (pybullet_collision.PyBulletCollisionChecker), the planners' validity checks, path verification, the
approach-clearance constraint and the clearance/APF/evaluation metrics all go through the same engine and the same
meshes now. Before, pose determination used a point-cloud proximity test while planning used Pinocchio/FCL on a
differently-shaped geometry, so a pose accepted by one could be rejected by the other (typically a start posture that
"collides" before the planner starts).

Geometry rules (identical for every consumer):
- robot links: loaded from the URDF by PyBullet (box/cylinder/sphere primitives and STL/DAE meshes). Mesh links are
  convex hulls - PyBullet's behaviour for URDF meshes; an EF mesh checked in pose determination is built the same way.
- obstacles (pipe, positioner): exact triangle meshes (concave BVH), named collision_object_0, collision_object_1, ...
- pairs: robot link vs every obstacle, plus robot self pairs between non-adjacent links.
Distances are PyBullet closest-point distances (negative = penetration), reported per (link, obstacle).
"""

from __future__ import annotations

import contextlib
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pybullet as pb
except ImportError:  # pragma: no cover
    pb = None

from plugins.robotics.pybullet_collision import (_ensure_client, closest_points, make_static_mesh_body, min_distance,
                                                 remove_static_body)

_SUFFIX = re.compile(r"_\d+$")


def link_of_geometry(geom_name: str) -> str:
    """Pinocchio geometry name ("rt_link2_0") -> link name ("rt_link2")."""
    return _SUFFIX.sub("", str(geom_name))


@contextlib.contextmanager
def _quiet_stdio():
    """Silence C-level stdout/stderr (PyBullet's importer warnings) for the duration of the block."""
    import os
    import sys
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    saved = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


class PyBulletRobotScene:
    def __init__(self, urdf_path: str, joint_names: Sequence[str], robot_pairs: Iterable[Tuple[str, str]]):
        self.client = _ensure_client()
        self.urdf_path = str(urdf_path)
        with _quiet_stdio():  # PyBullet prints a "No inertial data" warning per link on the C stdout
            self.body = pb.loadURDF(self.urdf_path, useFixedBase=True, physicsClientId=self.client)
        self.joint_names = list(joint_names)

        self._link_index: Dict[str, int] = {pb.getBodyInfo(self.body, physicsClientId=self.client)[0].decode(): -1}
        joint_index: Dict[str, int] = {}
        for i in range(pb.getNumJoints(self.body, physicsClientId=self.client)):
            info = pb.getJointInfo(self.body, i, physicsClientId=self.client)
            self._link_index[info[12].decode()] = i
            if info[2] != pb.JOINT_FIXED:
                joint_index[info[1].decode()] = i
        missing = [n for n in self.joint_names if n not in joint_index]
        if missing:
            raise RuntimeError(f"PyBullet scene: joints not found in {urdf_path}: {missing}")
        self._q_joint_indices = [joint_index[n] for n in self.joint_names]

        # links that actually have a collision shape
        self.collision_links: Dict[str, int] = {
            name: idx for name, idx in self._link_index.items()
            if pb.getCollisionShapeData(self.body, idx, physicsClientId=self.client)}
        pairs = set()
        for a, b in robot_pairs:
            a, b = link_of_geometry(a), link_of_geometry(b)
            if a != b and a in self.collision_links and b in self.collision_links:
                pairs.add(tuple(sorted((a, b))))
        self.self_pairs: List[Tuple[str, str]] = sorted(pairs)

        self.peer_robots = {}
        self.static_bodies: List[int] = []
        self.static_names: List[str] = []
        self._clearance_links: Tuple[str, ...] = ()
        self._clearance_margin = 0.0

    # -- scene ---------------------------------------------------------------------------------------------------
    def set_static_meshes(self, meshes: Sequence[Any]) -> None:
        for body in self.static_bodies:
            remove_static_body(body, self.client)
        self.static_bodies, self.static_names = [], []
        for mesh in meshes or []:
            if mesh is None or not mesh.has_triangles():
                continue
            self.static_names.append(f"collision_object_{len(self.static_bodies)}")
            self.static_bodies.append(make_static_mesh_body(mesh, concave=True))

    def set_peer_robot(self, name, urdf_path, joint_names, q, relative_base_T):
        """Independent URDF copy: peer joints cannot be changed by another planner."""
        from scipy.spatial.transform import Rotation
        entry = self.peer_robots.get(name)
        if entry is None or entry[0].urdf_path != str(urdf_path):
            if entry is not None:
                pb.removeBody(entry[0].body, physicsClientId=self.client)
            peer = PyBulletRobotScene(urdf_path, joint_names, [])
            pos, quat = pb.getBasePositionAndOrientation(peer.body, physicsClientId=self.client)
            inertial = np.eye(4)
            inertial[:3, :3] = Rotation.from_quat(quat).as_matrix()
            inertial[:3, 3] = pos
            entry = self.peer_robots[name] = (peer, inertial)
        peer, inertial = entry
        world = np.asarray(relative_base_T) @ inertial
        pb.resetBasePositionAndOrientation(peer.body, world[:3, 3],
            Rotation.from_matrix(world[:3, :3]).as_quat(), physicsClientId=self.client)
        peer.set_q(q)

    def peer_collisions(self):
        hits = []
        own_links = {idx: name for name, idx in self.collision_links.items()}
        for name, (peer, _) in self.peer_robots.items():
            other_links = {idx: link for link, idx in peer.collision_links.items()}
            for point in pb.getClosestPoints(self.body, peer.body, 0.0, physicsClientId=self.client):
                if point[8] <= 0:
                    hits.append((f"{own_links.get(point[3], point[3])}_0",
                                 f"{other_links.get(point[4], point[4])}_0"))
        return list(dict.fromkeys(hits))

    def set_clearance(self, link_names: Optional[Sequence[str]], margin: float) -> int:
        """Treat the listed links as colliding with an obstacle closer than `margin` (approach-constraint)."""
        names = tuple(str(n) for n in (link_names or []))
        self._clearance_margin = max(float(margin or 0.0), 0.0)
        self._clearance_links = tuple(
            l for l in self.collision_links if any(l == n or l.startswith(n + "_") or n.startswith(l + "_") for n in names)
        ) if self._clearance_margin > 0.0 else ()
        return len(self._clearance_links)

    def set_q(self, q: Sequence[float]) -> None:
        for idx, value in zip(self._q_joint_indices, np.asarray(q, dtype=float).tolist()):
            pb.resetJointState(self.body, idx, value, physicsClientId=self.client)

    # -- queries -------------------------------------------------------------------------------------------------
    def check(self, q: Sequence[float]) -> List[Tuple[str, str]]:
        """Colliding pairs ([] = collision free) at q. Pair names use the "<link>_0" geometry naming."""
        self.set_q(q)
        hits: List[Tuple[str, str]] = self.peer_collisions()
        for a, b in self.self_pairs:
            if pb.getClosestPoints(self.body, self.body, 0.0, linkIndexA=self.collision_links[a],
                                   linkIndexB=self.collision_links[b], physicsClientId=self.client):
                hits.append((f"{a}_0", f"{b}_0"))
        for link, idx in self.collision_links.items():
            margin = self._clearance_margin if link in self._clearance_links else 0.0
            for name, body in zip(self.static_names, self.static_bodies):
                points = closest_points(self.body, body, margin, linkIndexA=idx, physicsClientId=self.client)
                if points and min(p[8] for p in points) <= margin:
                    hits.append((f"{link}_0", name))
        return hits

    def collides(self, q: Sequence[float]) -> bool:
        self.set_q(q)
        if self.peer_collisions():
            return True
        for a, b in self.self_pairs:
            if pb.getClosestPoints(self.body, self.body, 0.0, linkIndexA=self.collision_links[a],
                                   linkIndexB=self.collision_links[b], physicsClientId=self.client):
                return True
        for link, idx in self.collision_links.items():
            margin = self._clearance_margin if link in self._clearance_links else 0.0
            for body in self.static_bodies:
                points = closest_points(self.body, body, margin, linkIndexA=idx, physicsClientId=self.client)
                if points and min(p[8] for p in points) <= margin:
                    return True
        return False

    def distances(self, q: Sequence[float], max_distance: float = 0.3) -> List[Dict[str, Any]]:
        """[{link, obstacle, distance}] for every (link, obstacle); distance clipped to max_distance when farther.

        The cost of getClosestPoints grows with the search radius (a 1 m radius made one path metric take ~19 s), and
        nothing here (APF d0 <= 0.3 m, clearance margins) looks beyond ~0.3 m, so 0.3 m is the default cap. Growing radii
        (pybullet_collision.min_distance): one 0.3 m query against the scanned pipe returned garbage distances."""
        self.set_q(q)
        out = []
        for link, idx in self.collision_links.items():
            for name, body in zip(self.static_names, self.static_bodies):
                d = min_distance(self.body, body, max_distance, linkIndexA=idx, physicsClientId=self.client)
                out.append({"link": f"{link}_0", "obstacle": name, "distance": float(d)})
        return out

    def self_distances(self, q: Sequence[float], max_distance: float = 0.1) -> List[Dict[str, Any]]:
        """[{a, b, distance}] for every non-adjacent link pair of the robot (negative = penetrating)."""
        self.set_q(q)
        out = []
        for a, b in self.self_pairs:
            points = pb.getClosestPoints(self.body, self.body, max_distance, linkIndexA=self.collision_links[a],
                                         linkIndexB=self.collision_links[b], physicsClientId=self.client)
            d = min((p[8] for p in points), default=max_distance)
            out.append({"a": f"{a}_0", "b": f"{b}_0", "distance": float(d)})
        return out
