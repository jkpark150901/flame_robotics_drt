from abc import ABC, abstractmethod
import numpy as np
import os
from typing import List, Union, Optional
try:
    import pinocchio as pin
except ImportError:
    pin = None
else:
    if not hasattr(pin, "buildModelFromUrdf"):
        pin = None
try:
    import hppfcl
except ImportError:
    try:
        import coal as hppfcl
    except ImportError:
        hppfcl = None

class PlannerBase(ABC):
    """
    Abstract base class for path planning algorithms.
    """

    def __init__(self):
        self.collision_objects = []
        self.static_objects = self.collision_objects
        self.tool_mesh = None
        self.pin_model = None
        self.pin_data = None
        self.pin_geom_model = None
        self.pin_geom_data = None
        self._pin_robot_geom_ids = []
        self._pin_static_object_ids = []
        self.pin_collision_sample_resolution = 1.0
        self._pin_collision_dim_warning_shown = False

    def configure_collision(self, config: dict, default_sample_resolution: float = 1.0):
        self.pin_collision_sample_resolution = float(
            config.get("pinocchio_collision_sample_resolution", default_sample_resolution)
        )
        if config.get("pinocchio_collision", False):
            self.setup_pinocchio_collision(
                config.get("robot_urdf"),
                config.get("package_dirs"),
            )

    @abstractmethod
    def generate(self, current_pose: Union[List[float], np.ndarray], target_pose: Union[List[float], np.ndarray], step_callback: Optional[callable] = None) -> List[np.ndarray]:
        """
        Generate a path from current_pose to target_pose.
        
        Args:
            current_pose: Current end-effector pose [x, y, z, roll, pitch, yaw]
            target_pose: Target end-effector pose [x, y, z, roll, pitch, yaw]
                         Orientation components can be NaN (don't care).
                         
        Returns:
            List of waypoints (numpy arrays).
        """
        pass

    def add_collision_object(self, object_model):
        """
        Add a static obstacle mesh to the shared collision backend.
        
        Args:
            object_model: Mesh-like object with vertices and triangles/faces/cells.
        """
        self.collision_objects.append(object_model)
        if self.pin_geom_model is not None:
            try:
                return self._add_pinocchio_collision_mesh(object_model)
            except Exception as e:
                print(f"Error adding object to Pinocchio collision scene: {e}")
        return None

    def add_static_object(self, object_model):
        """Backward-compatible alias for older planner callers."""
        return self.add_collision_object(object_model)

    def set_tool_geometry(self, tool_mesh):
        """
        Set the tool geometry for collision checking.
        
        Args:
            tool_mesh: Open3D TriangleMesh of the tool.
                       The mesh should be defined relative to the end-effector frame (origin at mount point).
        """
        self.tool_mesh = tool_mesh

    def setup_pinocchio_collision(self, urdf_path, package_dirs=None, ignore_adjacent_pairs=True):
        """Enable Pinocchio/hpp-fcl collision checking for q-space planners."""
        if pin is None:
            raise RuntimeError("robotics Pinocchio is not installed")
        if hppfcl is None:
            raise RuntimeError("hppfcl/coal is not installed")
        if not urdf_path:
            raise ValueError("robot_urdf is required for Pinocchio collision")

        urdf_path = os.path.abspath(urdf_path)
        if package_dirs is None:
            package_dirs = [os.path.dirname(urdf_path)]
        elif isinstance(package_dirs, str):
            package_dirs = [package_dirs]
        package_dirs = [os.path.abspath(p) for p in package_dirs]

        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()
        self.pin_geom_model = pin.buildGeomFromUrdf(
            self.pin_model, urdf_path, pin.GeometryType.COLLISION, None, package_dirs
        )
        self.pin_geom_model.addAllCollisionPairs()
        if ignore_adjacent_pairs:
            self._remove_adjacent_pinocchio_collision_pairs()
        self._pin_robot_geom_ids = list(range(len(self.pin_geom_model.geometryObjects)))
        self._pin_static_object_ids = []
        for object_model in self.collision_objects:
            self._add_pinocchio_collision_mesh(object_model, recreate_data=False)
        self.pin_geom_data = pin.GeometryData(self.pin_geom_model)
        return self.pin_geom_model

    def _add_pinocchio_collision_mesh(self, object_model, recreate_data=True):
        if self.pin_geom_model is None:
            return None

        bvh = self._mesh_to_hppfcl_bvh(object_model)
        name = f"collision_object_{len(self._pin_static_object_ids)}"
        geom_obj = pin.GeometryObject(name, 0, pin.SE3.Identity(), bvh)
        geom_id = self.pin_geom_model.addGeometryObject(geom_obj)
        for robot_geom_id in self._pin_robot_geom_ids:
            pair = pin.CollisionPair(robot_geom_id, geom_id)
            if not self.pin_geom_model.existCollisionPair(pair):
                self.pin_geom_model.addCollisionPair(pair)
        self._pin_static_object_ids.append(geom_id)
        if recreate_data:
            self.pin_geom_data = pin.GeometryData(self.pin_geom_model)
        return geom_id

    def _mesh_to_hppfcl_bvh(self, mesh):
        vertices, triangles = self._extract_mesh_arrays(mesh)
        if len(vertices) == 0 or len(triangles) == 0:
            raise ValueError("collision object mesh must have vertices and triangles")

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
        return bvh

    def _extract_mesh_arrays(self, mesh):
        vertices = np.asarray(mesh.vertices, dtype=float)
        if hasattr(mesh, "triangles"):
            triangles = np.asarray(mesh.triangles, dtype=np.int32)
        elif hasattr(mesh, "faces"):
            triangles = np.asarray(mesh.faces, dtype=np.int32)
        elif hasattr(mesh, "cells"):
            triangles = np.asarray(mesh.cells, dtype=np.int32)
        else:
            raise ValueError("mesh must expose triangles, faces, or cells")

        if triangles.ndim != 2:
            raise ValueError("mesh triangle array must be 2-dimensional")
        if triangles.shape[1] > 3:
            triangles = triangles[:, :3]
        if triangles.shape[1] != 3:
            raise ValueError("mesh triangle array must have 3 indices per face")
        return vertices, triangles

    def _remove_adjacent_pinocchio_collision_pairs(self):
        if self.pin_model is None or self.pin_geom_model is None:
            return

        kept_pairs = []
        for pair in list(self.pin_geom_model.collisionPairs):
            first = self.pin_geom_model.geometryObjects[pair.first]
            second = self.pin_geom_model.geometryObjects[pair.second]
            if self._is_adjacent_pinocchio_pair(first.parentJoint, second.parentJoint):
                continue
            kept_pairs.append(pin.CollisionPair(pair.first, pair.second))

        self.pin_geom_model.removeAllCollisionPairs()
        for pair in kept_pairs:
            self.pin_geom_model.addCollisionPair(pair)

    def _is_adjacent_pinocchio_pair(self, joint_a, joint_b):
        if joint_a == joint_b:
            return True
        parents = self.pin_model.parents
        if joint_a < len(parents) and parents[joint_a] == joint_b:
            return True
        if joint_b < len(parents) and parents[joint_b] == joint_a:
            return True
        return False

    def check_pinocchio_collision(self, q, return_pairs=False):
        if self.pin_model is None or self.pin_geom_model is None or self.pin_geom_data is None:
            raise RuntimeError("Pinocchio collision is not configured")

        q = np.asarray(q, dtype=float)
        if q.shape[0] != self.pin_model.nq:
            raise ValueError(f"q dimension mismatch: got {q.shape[0]}, expected {self.pin_model.nq}")

        has_collision = pin.computeCollisions(
            self.pin_model, self.pin_data, self.pin_geom_model, self.pin_geom_data, q, False
        )
        if not return_pairs:
            return bool(has_collision)

        pairs = []
        for idx, result in enumerate(self.pin_geom_data.collisionResults):
            if not result.isCollision():
                continue
            pair = self.pin_geom_model.collisionPairs[idx]
            pairs.append((
                self.pin_geom_model.geometryObjects[pair.first].name,
                self.pin_geom_model.geometryObjects[pair.second].name,
            ))
        return bool(has_collision), pairs

    def _check_pinocchio_collision(self, p1, p2):
        if self.pin_model is None:
            return None

        q1 = np.asarray(p1, dtype=float)
        q2 = np.asarray(p2, dtype=float)
        if q1.shape[0] != self.pin_model.nq or q2.shape[0] != self.pin_model.nq:
            raise ValueError(
                "Pinocchio collision requires q-space states: "
                f"got {q1.shape[0]}->{q2.shape[0]}, expected nq={self.pin_model.nq}"
            )

        length = float(np.linalg.norm(q2 - q1))
        resolution = max(float(self.pin_collision_sample_resolution), 1e-9)
        steps = max(1, int(np.ceil(length / resolution)))

        for i in range(steps + 1):
            alpha = i / steps
            q = (1.0 - alpha) * q1 + alpha * q2
            if pin.computeCollisions(
                self.pin_model, self.pin_data, self.pin_geom_model, self.pin_geom_data, q, True
            ):
                return True
        return False

    def collision_pairs_along_edge(self, p1, p2):
        if self.pin_model is None:
            return []

        q1 = np.asarray(p1, dtype=float)
        q2 = np.asarray(p2, dtype=float)
        if q1.shape[0] != self.pin_model.nq or q2.shape[0] != self.pin_model.nq:
            return []

        length = float(np.linalg.norm(q2 - q1))
        resolution = max(float(self.pin_collision_sample_resolution), 1e-9)
        steps = max(1, int(np.ceil(length / resolution)))
        pairs = []
        seen = set()

        for i in range(steps + 1):
            alpha = i / steps
            q = (1.0 - alpha) * q1 + alpha * q2
            hit, hit_pairs = self.check_pinocchio_collision(q, return_pairs=True)
            if not hit:
                continue
            for pair in hit_pairs:
                key = tuple(pair)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(pair)
        return pairs

    def _check_collision(self, p1, p2):
        pin_collision = self._check_pinocchio_collision(p1, p2)
        if pin_collision is not None:
            return pin_collision
        return False

    def _sample_pinocchio_configuration(self):
        if self.pin_model is None:
            raise RuntimeError("Pinocchio collision is not configured")
        lo = np.asarray(self.pin_model.lowerPositionLimit, dtype=float).copy()
        hi = np.asarray(self.pin_model.upperPositionLimit, dtype=float).copy()

        invalid = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
        lo[invalid] = -np.pi
        hi[invalid] = np.pi
        return np.random.uniform(lo, hi)

    def _steer_state(self, from_state, to_state, step_size):
        direction = np.asarray(to_state, dtype=float) - np.asarray(from_state, dtype=float)
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            return np.asarray(from_state, dtype=float).copy()
        return np.asarray(from_state, dtype=float) + direction / length * min(float(step_size), length)
