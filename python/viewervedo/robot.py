"""
Robot URDF Loader for Visualizer
@note
- Uses urdf_parser.urdf.URDF to load and parse URDF files
- Leverages built-in forward kinematics from the URDF parser
- Converts trimesh meshes to vedo actors for rendering
"""

import os
import numpy as np
import vedo
import trimesh
from typing import List
from urdf_parser.urdf import URDF
from util.logger.console import ConsoleLogger


def _make_base_transform(base_pose: List[float]) -> np.ndarray:
    """Create a 4x4 homogeneous transformation matrix from [x, y, z, roll, pitch, yaw]."""
    x, y, z = base_pose[0], base_pose[1], base_pose[2]
    r, p, yaw = base_pose[3], base_pose[4], base_pose[5]

    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [x, y, z]
    return T


def _trimesh_to_vedo(tm: trimesh.Trimesh, name: str = "") -> vedo.Mesh:
    """Convert a trimesh.Trimesh object to a vedo.Mesh."""
    mesh = vedo.Mesh([tm.vertices, tm.faces])
    mesh.name = name
    return mesh


class RobotModel:
    """Loads a URDF via urdf_parser and produces vedo actors for each link.
    Supports live FK updates via set_joint / update_fk.
    """

    def __init__(self, name: str, urdf_path: str, base_pose: List[float] = None):
        self.name = name
        self.urdf_path = urdf_path
        self.base_pose = base_pose or [0, 0, 0, 0, 0, 0]
        self.actors: List[vedo.Mesh] = []
        self._console = ConsoleLogger.get_logger()

        self._urdf = None
        self._base_T = None
        self._joint_cfg = {}  # joint_name -> float
        self._last_fk = None  # cached FK result from last update_fk call

        # link_name -> list of (vertices_in_link_frame, faces)
        self._link_mesh_data = {}
        # link_name -> list of vedo.Mesh actors
        self._link_actors = {}

    def load(self) -> List[vedo.Mesh]:
        """Load the URDF and create vedo mesh actors with FK transforms applied."""
        self._urdf = URDF.load(self.urdf_path)
        self._base_T = _make_base_transform(self.base_pose)

        self._console.debug(f"[Robot:{self.name}] Loaded URDF '{self._urdf.name}' "
                            f"({len(self._urdf.links)} links, {len(self._urdf.joints)} joints)")

        fk = self._urdf.link_fk(cfg=None)

        for link, link_T in fk.items():
            world_T = self._base_T @ link_T
            link_mesh_list = []
            link_actor_list = []

            for visual in link.visuals:
                geom = visual.geometry
                for tm in geom.meshes:
                    tm_copy = tm.copy()

                    if geom.mesh is not None and geom.mesh.scale is not None:
                        S = np.eye(4)
                        S[:3, :3] = np.diag(geom.mesh.scale)
                        tm_copy.apply_transform(S)

                    if visual.origin is not None:
                        tm_copy.apply_transform(visual.origin)

                    # Store link-local vertices (scale + visual origin applied) for FK updates
                    local_verts = tm_copy.vertices.copy()
                    faces = tm_copy.faces.copy()
                    link_mesh_list.append((local_verts, faces))

                    # Apply world transform for initial render
                    tm_world = tm_copy.copy()
                    tm_world.apply_transform(world_T)

                    actor = _trimesh_to_vedo(tm_world, f"{self.name}_{link.name}")
                    if visual.material is not None and visual.material.color is not None:
                        rgba = visual.material.color
                        actor.c((rgba[0], rgba[1], rgba[2])).alpha(rgba[3])
                    else:
                        actor.c('steelblue').alpha(0.9)

                    link_actor_list.append(actor)
                    self.actors.append(actor)

            if link_mesh_list:
                self._link_mesh_data[link.name] = link_mesh_list
                self._link_actors[link.name] = link_actor_list

        self._console.info(f"[Robot:{self.name}] Created {len(self.actors)} mesh actors")
        return self.actors

    def set_joint(self, joint_name: str, value: float):
        """Set a joint value for the next FK update."""
        self._joint_cfg[joint_name] = value

    def get_link_world_pos(self, link_name: str):
        """Return world-space xyz of a link's origin based on current joint config.
        Returns None if the link is not found.
        """
        T = self.get_link_world_T(link_name)
        return T[:3, 3].copy() if T is not None else None

    def get_link_world_T(self, link_name: str):
        """Return the 4x4 world-space transform of a link based on current joint config.
        Returns None if the link is not found.
        """
        fk = self._last_fk
        if fk is None:
            if self._urdf is None:
                return None
            fk = self._urdf.link_fk(cfg=self._joint_cfg)
            self._last_fk = fk
        for link_obj, link_T in fk.items():
            if link_obj.name == link_name:
                return (self._base_T @ link_T).copy()
        return None

    def update_fk(self):
        """Recompute FK with current joint values and update actor vertices in-place."""
        if self._urdf is None:
            return

        fk = self._urdf.link_fk(cfg=self._joint_cfg)
        self._last_fk = fk

        for link_obj, link_T in fk.items():
            link_name = link_obj.name
            mesh_list = self._link_mesh_data.get(link_name, [])
            actor_list = self._link_actors.get(link_name, [])

            world_T = self._base_T @ link_T
            R = world_T[:3, :3]
            t = world_T[:3, 3]

            for i, (local_verts, _) in enumerate(mesh_list):
                if i >= len(actor_list):
                    break
                new_verts = (R @ local_verts.T).T + t
                actor_list[i].vertices = new_verts


def load_robots_from_config(config: dict) -> List[vedo.Mesh]:
    """Load all robot models defined in config and return their vedo actors.

    Config format:
        "urdf": [
            {"name": "robot1", "path": "urdf/robot.urdf", "base": [x, y, z, r, p, yaw]},
            ...
        ]
    """
    console = ConsoleLogger.get_logger()
    all_actors = []

    urdf_entries = config.get("urdf", [])
    if not urdf_entries:
        return all_actors

    root_path = config.get("root_path", "")

    for entry in urdf_entries:
        name = entry.get("name", "unknown")
        path = entry.get("path", "")
        base = entry.get("base", [0, 0, 0, 0, 0, 0])

        full_path = os.path.join(str(root_path), path) if root_path else path

        if not os.path.exists(full_path):
            console.error(f"[Robot] URDF file not found: {full_path}")
            continue

        console.info(f"[Robot] Loading {name} from {full_path}")
        robot = RobotModel(name=name, urdf_path=full_path, base_pose=base)
        all_actors.extend(robot.load())

    return all_actors
