from abc import ABC
import numpy as np
import os
import time
import csv
import json
from pathlib import Path
from typing import List, Union, Optional
try:
    import pinocchio as pin
except ImportError:
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
        self.planning_deadline = None
        self.normalize_joint_space = True
        self.use_joint_space_planning = False
        self.robotics_backend = None
        self.robotics_robot_name = None
        self.debug_exploration = False
        self.debug_output_dir = os.path.join(os.getcwd(), "debug", "planner")
        self.last_exploration_csv = None
        self.last_exploration_plot = None

    def _check_planning_deadline(self):
        deadline = getattr(self, "planning_deadline", None)
        if deadline is not None and time.monotonic() > float(deadline):
            raise TimeoutError("path planning timeout")

    def configure_collision(self, config: dict, default_sample_resolution: float = 1.0):
        self.pin_collision_sample_resolution = float(
            config.get("pinocchio_collision_sample_resolution", default_sample_resolution)
        )
        if config.get("pinocchio_collision", False):
            self.setup_pinocchio_collision(
                config.get("robot_urdf"),
                config.get("package_dirs"),
            )

    def generate(self, current_pose: Union[List[float], np.ndarray], target_pose: Union[List[float], np.ndarray], step_callback: Optional[callable] = None) -> List[np.ndarray]:
        """경로 생성을 위한 공통 진입점.

        Args:
            current_pose: 시작 상태. workspace planner에서는 pose/state, joint-space planner에서는 raw q.
            target_pose: 목표 상태. workspace planner에서는 pose/state, joint-space planner에서는 raw q.
            step_callback: 탐색 중 tree 상태를 외부로 전달하기 위한 선택 콜백.

        Returns:
            waypoint list. 실패하면 빈 list.

        계산 과정:
            1. 입력을 numpy 배열로 변환한다.
            2. ``use_joint_space_planning`` 속성이 True이고 Pinocchio model이 설정되어 있으면
               raw q 입력으로 간주하고 ``_generate_joint_space`` 로 분기한다.
            3. joint-space 사용 시 입력 차원이 pin_model.nq와 다르면 예외를 발생시킨다.
            4. 그 외에는 ``_generate_workspace`` 로 분기한다.

        주의:
            joint-space 여부는 입력 shape로 추론하지 않고 class/instance 속성값으로 결정한다.
            단, Pinocchio model이 아직 없는 단독 알고리즘 테스트 상황에서는 workspace 구현으로 내려간다.
        """
        current_pose = np.asarray(current_pose, dtype=float)
        target_pose = np.asarray(target_pose, dtype=float)

        if getattr(self, "use_joint_space_planning", False) and self.pin_model is not None:
            if current_pose.shape[0] != self.pin_model.nq or target_pose.shape[0] != self.pin_model.nq:
                raise ValueError(
                    f"{self.__class__.__name__} is configured for joint-space planning, "
                    f"so generate() must receive q-space states with nq={self.pin_model.nq}; "
                    f"got {current_pose.shape[0]}->{target_pose.shape[0]}"
                )
            return self._generate_joint_space(current_pose, target_pose, step_callback=step_callback)

        return self._generate_workspace(current_pose, target_pose, step_callback=step_callback)

    def _generate_workspace(self, current_pose, target_pose, step_callback=None):
        """workspace 경로 생성 구현부.

        Args:
            current_pose: 시작 workspace 상태.
            target_pose: 목표 workspace 상태.
            step_callback: 탐색 중 tree 상태 콜백.

        Returns:
            subclass가 구현한 waypoint list.

        계산 과정:
            PlannerBase는 분기만 담당하므로 기본 구현은 예외를 발생시킨다.
            workspace planner subclass가 이 함수를 구현해야 한다.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement _generate_workspace()")

    def _generate_joint_space(self, start_q, goal_q, step_callback=None):
        """joint-space 경로 생성 구현부.

        Args:
            start_q: 시작 raw q.
            goal_q: 목표 raw q.
            step_callback: 탐색 중 tree 상태 콜백.

        Returns:
            subclass가 구현한 raw q waypoint list.

        계산 과정:
            PlannerBase는 분기만 담당하므로 기본 구현은 예외를 발생시킨다.
            joint-space planner subclass가 이 함수를 구현해야 한다.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement _generate_joint_space()")

    def _robotics_collision_backend(self):
        """현재 planner에 연결된 robotics backend와 robot 이름을 반환한다.

        Args:
            없음. ``robotics_backend``와 ``robotics_robot_name`` 속성을 읽는다.

        Returns:
            (backend, robot_name). 둘 중 하나라도 없으면 (None, None).

        계산 과정:
            Viewer가 planner를 설정할 때 backend와 robot 이름을 주입한다.
            이 값이 있으면 충돌 scene 구성, 단일 q 충돌 검사, edge 충돌 검사는
            PlannerBase 내부 구현 대신 backend 구현을 사용한다.
        """
        backend = getattr(self, "robotics_backend", None)
        robot_name = getattr(self, "robotics_robot_name", None)
        if backend is None or not robot_name:
            return None, None
        return backend, robot_name

    def add_collision_object(self, object_model):
        """
        Add a static obstacle mesh to the shared collision backend.
        
        Args:
            object_model: Mesh-like object with vertices and triangles/faces/cells.
        """
        self.collision_objects.append(object_model)
        backend, robot_name = self._robotics_collision_backend()
        if backend is not None:
            backend.configure_collision(
                robot_name,
                static_meshes=self.collision_objects,
                sample_resolution=self.pin_collision_sample_resolution,
            )
            handle = backend.robot_handle(robot_name) if hasattr(backend, "robot_handle") else None
            if handle is not None:
                self.pin_model = handle.model
                self.pin_data = handle.data
                self.pin_geom_model = handle.geom_model
                self.pin_geom_data = handle.geom_data
                self._pin_robot_geom_ids = list(getattr(handle, "robot_geom_ids", []) or [])
                self._pin_static_object_ids = list(getattr(handle, "static_object_ids", []) or [])
            return self._pin_static_object_ids[-1] if self._pin_static_object_ids else None
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
            raise RuntimeError("pinocchio is not installed")
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

    def pinocchio_collision_geometry_summary(self):
        """Return the collision geometries currently registered in Pinocchio."""
        if self.pin_model is None or self.pin_geom_model is None:
            return []
        static_ids = set(getattr(self, "_pin_static_object_ids", []))
        names = list(self.pin_model.names)
        summary = []
        for geom_id, geom in enumerate(self.pin_geom_model.geometryObjects):
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

    def pinocchio_collision_pair_summary(self, include_robot_self=True, include_static=True, limit=None):
        """Return the collision pairs checked by Pinocchio."""
        if self.pin_model is None or self.pin_geom_model is None:
            return []
        static_ids = set(getattr(self, "_pin_static_object_ids", []))
        pairs = []
        for pair_id, pair in enumerate(self.pin_geom_model.collisionPairs):
            first = self.pin_geom_model.geometryObjects[pair.first]
            second = self.pin_geom_model.geometryObjects[pair.second]
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

    def check_pinocchio_collision(self, q, return_pairs=False):
        self._check_planning_deadline()
        backend, robot_name = self._robotics_collision_backend()
        if backend is not None:
            result = backend.check_collision(robot_name, q, return_pairs=return_pairs)
            return (result.collision, result.pairs) if return_pairs else result.collision
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
        backend, robot_name = self._robotics_collision_backend()
        if backend is not None:
            return bool(backend.check_edge_collision(robot_name, p1, p2, return_pairs=False).collision)
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
            self._check_planning_deadline()
            alpha = i / steps
            q = (1.0 - alpha) * q1 + alpha * q2
            if pin.computeCollisions(
                self.pin_model, self.pin_data, self.pin_geom_model, self.pin_geom_data, q, True
            ):
                return True
        return False

    def collision_pairs_along_edge(self, p1, p2):
        self._check_planning_deadline()
        backend, robot_name = self._robotics_collision_backend()
        if backend is not None:
            return list(backend.check_edge_collision(robot_name, p1, p2, return_pairs=True).pairs)
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
            self._check_planning_deadline()
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
        self._check_planning_deadline()
        pin_collision = self._check_pinocchio_collision(p1, p2)
        if pin_collision is not None:
            return pin_collision
        return False

    @staticmethod
    def _json_vector(value, digits=6):
        """벡터 값을 CSV 셀에 넣기 좋은 JSON 문자열로 변환한다.

        Args:
            value: None 또는 array-like 벡터.
            digits: 반올림 소수 자리 수.

        Returns:
            None이면 빈 문자열, 값이 있으면 JSON 배열 문자열.

        계산 과정:
            입력을 1차원 float 배열로 펼친 뒤 지정 자리수로 반올림하고 json.dumps로 직렬화한다.
        """
        if value is None:
            return ""
        arr = np.asarray(value, dtype=float).reshape(-1)
        return json.dumps(np.round(arr, digits).tolist())

    @staticmethod
    def _json_pairs(pairs):
        """충돌 pair 목록을 CSV 셀에 넣기 좋은 JSON 문자열로 변환한다.

        Args:
            pairs: (first, second) 형태의 충돌 geometry pair 목록.

        Returns:
            pair가 없으면 빈 문자열, 있으면 JSON 배열 문자열.

        계산 과정:
            tuple pair를 list pair로 바꾼 뒤 ensure_ascii=False 옵션으로 직렬화한다.
        """
        if not pairs:
            return ""
        return json.dumps([list(pair) for pair in pairs], ensure_ascii=False)

    def _edge_collision_info(self, q_from, q_to):
        """두 q 상태를 잇는 edge의 충돌 정보를 계산한다.

        Args:
            q_from: edge 시작 raw q.
            q_to: edge 끝 raw q.

        Returns:
            (hit, pairs, collision_q, collision_alpha).
            hit은 충돌 여부, pairs는 충돌 geometry pair 목록,
            collision_q는 edge에서 처음 충돌한 raw q,
            collision_alpha는 q_from=0, q_to=1 기준의 보간 위치다.

        계산 과정:
            먼저 edge 전체 충돌 여부를 빠르게 확인하고, 충돌이 있으면
            _first_collision_along_edge로 최초 충돌 지점과 pair를 다시 샘플링한다.
        """
        hit = bool(self._check_collision(q_from, q_to))
        pairs = []
        collision_q = None
        collision_alpha = None
        if hit:
            try:
                collision_q, collision_alpha, pairs = self._first_collision_along_edge(q_from, q_to)
            except Exception:
                if hasattr(self, "collision_pairs_along_edge"):
                    try:
                        pairs = self.collision_pairs_along_edge(q_from, q_to)
                    except Exception:
                        pairs = []
        return hit, pairs, collision_q, collision_alpha

    def _first_collision_along_edge(self, q_from, q_to):
        """edge를 일정 간격으로 샘플링해 최초 충돌 q를 찾는다.

        Args:
            q_from: edge 시작 raw q.
            q_to: edge 끝 raw q.

        Returns:
            (collision_q, collision_alpha, pairs).
            충돌이 없거나 Pinocchio model이 없으면 (None, None, []).

        계산 과정:
            q_from과 q_to 사이를 pin_collision_sample_resolution 기준으로 나누고,
            각 보간 q에서 Pinocchio collision pair를 검사한다.
            가장 먼저 충돌한 q와 alpha를 반환한다.
        """
        backend, robot_name = self._robotics_collision_backend()
        if backend is not None:
            result = backend.check_edge_collision(robot_name, q_from, q_to, return_pairs=True)
            if result.collision:
                return result.q, result.alpha, result.pairs
            return None, None, []
        if self.pin_model is None:
            return None, None, []

        q1 = np.asarray(q_from, dtype=float)
        q2 = np.asarray(q_to, dtype=float)
        if q1.shape[0] != self.pin_model.nq or q2.shape[0] != self.pin_model.nq:
            return None, None, []

        length = float(np.linalg.norm(q2 - q1))
        resolution = max(float(self.pin_collision_sample_resolution), 1e-9)
        steps = max(1, int(np.ceil(length / resolution)))
        for i in range(steps + 1):
            self._check_planning_deadline()
            alpha = i / steps
            q = (1.0 - alpha) * q1 + alpha * q2
            hit, pairs = self.check_pinocchio_collision(q, return_pairs=True)
            if hit:
                return q.copy(), float(alpha), pairs
        return None, None, []

    def verify_path(self, path):
        """q-space path의 waypoint/edge 충돌 여부를 검증한다.

        Args:
            path: raw q waypoint list.

        Returns:
            dict: colliding_edges, colliding_waypoints, collision_pairs,
            edge_collisions, waypoint_collisions, end_link_colliding, backend.

        계산 과정:
            1. 모든 waypoint에서 단일 q collision을 확인한다.
            2. 연속 waypoint 사이 edge를 샘플링해 충돌 pair를 확인한다.
            3. 중복 pair는 제거해서 요약 목록으로 반환한다.
            robotics backend가 연결되어 있으면 PlannerBase 내부 Pinocchio 구현 대신
            backend의 check_collision/check_edge_collision을 사용한다.
        """
        colliding_edges = 0
        colliding_waypoints = 0
        collision_pairs = []
        edge_collisions = []
        waypoint_collisions = []
        seen_pairs = set()
        poses = [np.asarray(p, dtype=float) for p in path]

        for waypoint_idx, q in enumerate(poses):
            try:
                hit, pairs = self.check_pinocchio_collision(q, return_pairs=True)
            except Exception:
                hit, pairs = False, []
            if hit:
                colliding_waypoints += 1
                waypoint_collisions.append({
                    "waypoint": int(waypoint_idx),
                    "pairs": [list(pair) for pair in pairs],
                })
            for pair in pairs:
                key = tuple(pair)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                collision_pairs.append(list(pair))

        for edge_idx, (a_pose, b_pose) in enumerate(zip(poses[:-1], poses[1:])):
            pairs = self.collision_pairs_along_edge(a_pose, b_pose)
            if pairs or self._check_collision(a_pose, b_pose):
                colliding_edges += 1
                edge_collisions.append({
                    "edge": edge_idx,
                    "from_waypoint": edge_idx,
                    "to_waypoint": edge_idx + 1,
                    "pairs": [list(pair) for pair in pairs],
                })
            for pair in pairs:
                key = tuple(pair)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                collision_pairs.append(list(pair))

        return {
            "colliding_edges": colliding_edges,
            "colliding_waypoints": colliding_waypoints,
            "collision_pairs": collision_pairs,
            "edge_collisions": edge_collisions,
            "waypoint_collisions": waypoint_collisions,
            "end_link_colliding": any(
                "link_end" in str(a).lower() or "link_end" in str(b).lower()
                for a, b in collision_pairs
            ),
            "backend": getattr(getattr(self, "robotics_backend", None), "name", "pinocchio"),
        }

    def _joint_limits_for_metric(self):
        """joint-space 거리 계산에 사용할 lower/upper/span을 만든다.

        Args:
            없음. self.pin_model과 self.normalize_joint_space를 사용한다.

        Returns:
            (lo, hi, span). 정규화를 사용하지 않거나 모델이 없으면 (None, None, None).

        계산 과정:
            Pinocchio joint limit을 읽고, 무한대/비정상 limit은 [-pi, pi]로 대체한다.
            span이 너무 작으면 1.0으로 보정해서 0 나눗셈을 막는다.
        """
        backend = getattr(self, "robotics_backend", None)
        robot_name = getattr(self, "robotics_robot_name", None)
        if backend is not None and robot_name:
            try:
                return backend.joint_limits_for_metric(
                    robot_name,
                    normalize=bool(getattr(self, "normalize_joint_space", True)),
                )
            except Exception:
                pass
        if not getattr(self, "normalize_joint_space", True) or self.pin_model is None:
            return None, None, None
        lo = np.asarray(self.pin_model.lowerPositionLimit, dtype=float).copy()
        hi = np.asarray(self.pin_model.upperPositionLimit, dtype=float).copy()
        invalid = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
        lo[invalid] = -np.pi
        hi[invalid] = np.pi
        span = hi - lo
        span[span < 1e-9] = 1.0
        return lo, hi, span

    def _normalize_joint_q(self, q):
        """raw q를 joint limit 기준 normalized q로 변환한다.

        Args:
            q: raw joint vector.

        Returns:
            normalized joint vector. 정규화가 꺼져 있으면 raw q copy.

        계산 과정:
            (q - lower) / (upper - lower)를 계산한다.
            prismatic/revolute joint 스케일 차이를 거리 계산에서 줄이기 위한 변환이다.
        """
        backend = getattr(self, "robotics_backend", None)
        robot_name = getattr(self, "robotics_robot_name", None)
        q = np.asarray(q, dtype=float)
        if backend is not None and robot_name:
            try:
                return backend.normalize_q(
                    robot_name,
                    q,
                    normalize=bool(getattr(self, "normalize_joint_space", True)),
                )
            except Exception:
                pass
        lo, _, span = self._joint_limits_for_metric()
        if span is None:
            return q.copy()
        return (q - lo) / span

    def _denormalize_joint_q(self, q_norm):
        """normalized q를 raw q로 되돌린다.

        Args:
            q_norm: normalized joint vector.

        Returns:
            raw joint vector. 정규화가 꺼져 있으면 입력 copy.

        계산 과정:
            lower + q_norm * span을 계산한 뒤 joint limit 안으로 clamp한다.
        """
        backend = getattr(self, "robotics_backend", None)
        robot_name = getattr(self, "robotics_robot_name", None)
        q_norm = np.asarray(q_norm, dtype=float)
        if backend is not None and robot_name:
            try:
                return backend.denormalize_q(
                    robot_name,
                    q_norm,
                    normalize=bool(getattr(self, "normalize_joint_space", True)),
                )
            except Exception:
                pass
        lo, hi, span = self._joint_limits_for_metric()
        if span is None:
            return q_norm.copy()
        return np.minimum(np.maximum(lo + q_norm * span, lo), hi)

    def _joint_distance(self, q_a, q_b):
        """두 raw q 사이의 normalized joint 거리 하나를 계산한다.

        Args:
            q_a: 첫 번째 raw q.
            q_b: 두 번째 raw q.

        Returns:
            float 거리.

        계산 과정:
            두 q를 normalized q로 바꾼 뒤 유클리드 norm을 계산한다.
        """
        backend = getattr(self, "robotics_backend", None)
        robot_name = getattr(self, "robotics_robot_name", None)
        if backend is not None and robot_name:
            try:
                return backend.joint_distance(
                    robot_name,
                    q_a,
                    q_b,
                    normalize=bool(getattr(self, "normalize_joint_space", True)),
                )
            except Exception:
                pass
        return float(np.linalg.norm(self._normalize_joint_q(q_b) - self._normalize_joint_q(q_a)))

    def _joint_distances(self, q_points, q_ref):
        """여러 q와 기준 q 사이의 normalized joint 거리를 한 번에 계산한다.

        Args:
            q_points: raw q 배열 또는 raw q list.
            q_ref: 기준 raw q.

        Returns:
            각 q_points 원소와 q_ref 사이의 거리 배열.

        계산 과정:
            정규화가 켜져 있으면 lower/span으로 모든 q를 normalized space로 변환한 뒤
            axis=1 norm을 계산한다. 꺼져 있으면 raw q norm을 사용한다.
        """
        pts = np.asarray(q_points, dtype=float)
        ref = np.asarray(q_ref, dtype=float)
        backend = getattr(self, "robotics_backend", None)
        robot_name = getattr(self, "robotics_robot_name", None)
        if backend is not None and robot_name:
            try:
                return backend.joint_distances(
                    robot_name,
                    pts,
                    ref,
                    normalize=bool(getattr(self, "normalize_joint_space", True)),
                )
            except Exception:
                pass
        if pts.ndim == 1:
            return np.asarray([self._joint_distance(pts, ref)], dtype=float)
        lo, _, span = self._joint_limits_for_metric()
        if span is None:
            return np.linalg.norm(pts - ref, axis=1)
        return np.linalg.norm(((pts - lo) / span) - ((ref - lo) / span), axis=1)

    def _steer_joint_state(self, from_state, to_state, step_size):
        """from_state에서 to_state 방향으로 step_size만큼 전진한 raw q를 만든다.

        Args:
            from_state: 시작 raw q.
            to_state: 목표 방향 raw q.
            step_size: normalized joint space에서의 최대 이동 거리.

        Returns:
            새 raw q.

        계산 과정:
            두 q를 normalized space로 바꾼 뒤 방향 벡터를 만들고,
            min(step_size, 거리)만큼 이동한 normalized q를 raw q로 되돌린다.
        """
        backend = getattr(self, "robotics_backend", None)
        robot_name = getattr(self, "robotics_robot_name", None)
        if backend is not None and robot_name:
            try:
                return backend.steer_joint_state(
                    robot_name,
                    from_state,
                    to_state,
                    step_size,
                    normalize=bool(getattr(self, "normalize_joint_space", True)),
                )
            except Exception:
                pass
        if not getattr(self, "normalize_joint_space", True) or self.pin_model is None:
            return self._steer_state(from_state, to_state, step_size)
        from_norm = self._normalize_joint_q(from_state)
        to_norm = self._normalize_joint_q(to_state)
        direction = to_norm - from_norm
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            return np.asarray(from_state, dtype=float).copy()
        new_norm = from_norm + direction / length * min(float(step_size), length)
        return self._denormalize_joint_q(new_norm)

    def _new_exploration_rows(self):
        """탐색 로그 row 컨테이너를 만든다.

        Args:
            없음. self.debug_exploration을 사용한다.

        Returns:
            debug_exploration이 True면 빈 list, 아니면 None.

        계산 과정:
            planner 구현부가 None 여부만 확인해서 logging 비용을 피할 수 있게 한다.
        """
        return [] if getattr(self, "debug_exploration", False) else None

    def _record_exploration(
        self,
        rows,
        iteration,
        phase,
        sample_type="",
        nearest_idx=None,
        from_q=None,
        to_q=None,
        sample_q=None,
        collision=False,
        collision_pairs=None,
        collision_q=None,
        collision_alpha=None,
        accepted=False,
        reason="",
        node_count=None,
        cost=None,
        elapsed_s=None,
        phase_elapsed_s=None,
        collision_check_elapsed_s=None,
        new_node_collision_count=None,
        random_new_node_collision_count=None,
        rewire_collision_count=None,
    ):
        """탐색 중 발생한 하나의 이벤트를 CSV row 형태로 기록한다.

        Args:
            rows: _new_exploration_rows가 만든 list 또는 None.
            iteration: 반복 번호.
            phase: 이벤트 단계 이름. 예: extend, choose_parent, add_node, rewire, connect_goal.
            sample_type: goal_bias/random 등 샘플 종류.
            nearest_idx: 관련 node index.
            from_q: edge 시작 raw q.
            to_q: edge 끝 raw q 또는 후보 raw q.
            sample_q: 이번 반복에서 샘플링한 raw q.
            collision: 충돌 여부.
            collision_pairs: 충돌 geometry pair 목록.
            collision_q: edge에서 처음 충돌한 raw q.
            collision_alpha: edge 보간 기준 최초 충돌 위치.
            accepted: tree에 반영되었는지 여부.
            reason: 이벤트 상세 이유.
            node_count: 이벤트 시점의 tree node 수.
            cost: 이벤트와 관련된 누적 cost.

        Returns:
            None. rows를 in-place로 갱신한다.

        계산 과정:
            numpy 배열은 JSON 문자열로 직렬화하고, 숫자/boolean 값은 CSV에 쓰기 쉬운 scalar로 정리한다.
        """
        if rows is None:
            return
        rows.append({
            "iteration": int(iteration) if iteration is not None else -1,
            "phase": phase,
            "sample_type": sample_type,
            "nearest_idx": "" if nearest_idx is None else int(nearest_idx),
            "node_count": "" if node_count is None else int(node_count),
            "elapsed_s": "" if elapsed_s is None else float(elapsed_s),
            "phase_elapsed_s": "" if phase_elapsed_s is None else float(phase_elapsed_s),
            "collision_check_elapsed_s": "" if collision_check_elapsed_s is None else float(collision_check_elapsed_s),
            "new_node_collision_count": "" if new_node_collision_count is None else int(new_node_collision_count),
            "random_new_node_collision_count": (
                "" if random_new_node_collision_count is None else int(random_new_node_collision_count)
            ),
            "rewire_collision_count": "" if rewire_collision_count is None else int(rewire_collision_count),
            "accepted": bool(accepted),
            "collision": bool(collision),
            "collision_pairs": self._json_pairs(collision_pairs),
            "collision_q": self._json_vector(collision_q),
            "collision_alpha": "" if collision_alpha is None else float(collision_alpha),
            "reason": reason,
            "cost": "" if cost is None else float(cost),
            "from_q": self._json_vector(from_q),
            "to_q": self._json_vector(to_q),
            "sample_q": self._json_vector(sample_q),
        })

    def _save_exploration_debug(self, rows, mode, status, path_waypoints=None):
        """탐색 로그를 CSV와 PNG 그래프로 저장한다.

        Args:
            rows: _record_exploration으로 누적한 row list.
            mode: 로그 종류. 예: joint.
            status: success, goal_collision, no_goal_connection 등 종료 상태.
            path_waypoints: 성공 시 최종 path waypoint 수.

        Returns:
            (csv_path, plot_path). 저장할 row가 없으면 (None, None).

        계산 과정:
            debug_output_dir 아래에 planner명/mode/robot명/time/status 기반 파일명을 만들고,
            CSV를 쓴 뒤 _save_exploration_plot으로 PNG 요약 그래프를 생성한다.
        """
        if not rows:
            return None, None
        out_dir = Path(getattr(self, "debug_output_dir", os.path.join(os.getcwd(), "debug", "planner")))
        out_dir.mkdir(parents=True, exist_ok=True)
        robot_name = "robot"
        try:
            robot_name = str(getattr(self.pin_model, "name", "") or "robot")
        except Exception:
            pass
        planner_name = self.__class__.__name__.lower()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = out_dir / f"{planner_name}_{mode}_{robot_name}_{stamp}_{status}"
        csv_path = base.with_suffix(".csv")
        fieldnames = [
            "iteration",
            "phase",
            "sample_type",
            "nearest_idx",
            "node_count",
            "elapsed_s",
            "phase_elapsed_s",
            "collision_check_elapsed_s",
            "new_node_collision_count",
            "random_new_node_collision_count",
            "rewire_collision_count",
            "accepted",
            "collision",
            "collision_pairs",
            "collision_q",
            "collision_alpha",
            "reason",
            "cost",
            "from_q",
            "to_q",
            "sample_q",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        plot_path = None
        try:
            plot_path = self._save_exploration_plot(rows, base.with_suffix(".png"), path_waypoints)
        except Exception as exc:
            print(f"{self.__class__.__name__} exploration plot failed: {exc}")
        self.last_exploration_csv = str(csv_path)
        self.last_exploration_plot = None if plot_path is None else str(plot_path)
        print(
            f"{self.__class__.__name__} exploration debug saved: "
            f"csv={self.last_exploration_csv}, plot={self.last_exploration_plot}")
        return self.last_exploration_csv, self.last_exploration_plot

    def _save_exploration_plot(self, rows, plot_path, path_waypoints=None):
        """탐색 로그 row를 간단한 PNG 그래프로 저장한다.

        Args:
            rows: 탐색 이벤트 row list.
            plot_path: 저장할 PNG 경로.
            path_waypoints: 성공 시 최종 path waypoint 수.

        Returns:
            성공 시 plot_path, matplotlib을 사용할 수 없으면 None.

        계산 과정:
            위 그래프에는 iteration별 node count와 collision/accepted marker를 표시하고,
            아래 그래프에는 phase별 이벤트 발생 위치를 scatter로 표시한다.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"{self.__class__.__name__} exploration plot skipped: matplotlib unavailable ({exc})")
            return None

        iterations = np.asarray([int(row["iteration"]) for row in rows], dtype=int)
        elapsed_values = []
        for row in rows:
            value = row.get("elapsed_s", "")
            try:
                elapsed_values.append(float(value))
            except Exception:
                elapsed_values.append(float(row.get("iteration", 0)))
        elapsed = np.asarray(elapsed_values, dtype=float)
        node_counts = np.asarray([
            np.nan if row["node_count"] == "" else int(row["node_count"])
            for row in rows
        ], dtype=float)
        collisions = np.asarray([bool(row["collision"]) for row in rows], dtype=bool)
        accepted = np.asarray([bool(row["accepted"]) for row in rows], dtype=bool)
        phases = [str(row["phase"]) for row in rows]
        collision_time = np.asarray([
            0.0 if row.get("collision_check_elapsed_s", "") == "" else float(row.get("collision_check_elapsed_s", 0.0))
            for row in rows
        ], dtype=float)
        cumulative_collision_time = np.cumsum(collision_time)

        def _carry_forward_count(column):
            values = []
            current = 0
            for row in rows:
                value = row.get(column, "")
                if value != "":
                    try:
                        current = int(value)
                    except Exception:
                        pass
                values.append(current)
            return np.asarray(values, dtype=float)

        new_node_collision_count = _carry_forward_count("new_node_collision_count")
        random_new_node_collision_count = _carry_forward_count("random_new_node_collision_count")
        rewire_collision_count = _carry_forward_count("rewire_collision_count")

        fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
        axes[0].plot(elapsed, node_counts, color="tab:blue", linewidth=1.5, label="nodes")
        if np.any(collisions):
            axes[0].scatter(
                elapsed[collisions],
                node_counts[collisions],
                color="tab:red",
                s=18,
                label="collision reject",
                zorder=3,
            )
        if np.any(accepted):
            axes[0].scatter(
                elapsed[accepted],
                node_counts[accepted],
                color="tab:green",
                s=10,
                alpha=0.45,
                label="accepted",
                zorder=2,
            )
        axes[0].set_ylabel("Node Count")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best")

        phase_order = [
            "extend",
            "choose_parent",
            "add_node",
            "rewire",
            "connect_goal",
            "start_collision",
            "goal_collision",
        ]
        phase_names = [name for name in phase_order if name in set(phases)]
        phase_names.extend(sorted(set(phases) - set(phase_names)))
        phase_to_y = {name: i for i, name in enumerate(phase_names)}
        y = np.asarray([phase_to_y[name] for name in phases], dtype=float)
        phase_colors = {
            "extend": "tab:red",
            "choose_parent": "tab:orange",
            "rewire": "tab:purple",
            "connect_goal": "tab:brown",
            "add_node": "tab:green",
        }
        colors = [
            phase_colors.get(phase, "tab:gray") if collision else ("tab:green" if is_accepted else "tab:gray")
            for phase, collision, is_accepted in zip(phases, collisions, accepted)
        ]
        axes[1].scatter(elapsed, y, c=colors, s=15, alpha=0.8)
        axes[1].set_yticks(list(phase_to_y.values()))
        axes[1].set_yticklabels(list(phase_to_y.keys()))
        axes[1].set_ylabel("Event / Collision Source")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(
            elapsed,
            cumulative_collision_time,
            color="tab:red",
            linewidth=1.5,
            label="cumulative collision-check time",
        )
        if np.any(collision_time > 0.0):
            axes[2].scatter(
                elapsed[collision_time > 0.0],
                cumulative_collision_time[collision_time > 0.0],
                c=["tab:red" if c else "tab:blue" for c in collisions[collision_time > 0.0]],
                s=12,
                alpha=0.65,
                label="collision checks",
            )
        axes[2].set_xlabel("Elapsed Time (s)")
        axes[2].set_ylabel("Collision Check Time (s)")
        axes[2].grid(True, alpha=0.25)
        axes[2].legend(loc="best")

        axes[3].plot(
            elapsed,
            new_node_collision_count,
            color="tab:red",
            linewidth=1.4,
            label="new-node collisions",
        )
        axes[3].plot(
            elapsed,
            random_new_node_collision_count,
            color="tab:pink",
            linewidth=1.2,
            linestyle="--",
            label="random new-node collisions",
        )
        axes[3].plot(
            elapsed,
            rewire_collision_count,
            color="tab:purple",
            linewidth=1.4,
            label="rewire collisions",
        )
        axes[3].set_xlabel("Elapsed Time (s)")
        axes[3].set_ylabel("Collision Count")
        axes[3].grid(True, alpha=0.25)
        axes[3].legend(loc="best")

        top = axes[0].twiny()
        top.set_xlim(axes[0].get_xlim())
        if len(elapsed) > 1 and float(elapsed[-1] - elapsed[0]) > 1e-9:
            ticks = axes[0].get_xticks()
            iter_ticks = np.interp(ticks, elapsed, iterations)
            top.set_xticks(ticks)
            top.set_xticklabels([str(int(round(v))) for v in iter_ticks])
        top.set_xlabel("Iteration (approx.)")

        title = f"{self.__class__.__name__} Exploration"
        if path_waypoints is not None:
            title += f" | path waypoints={path_waypoints}"
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        return plot_path

    def _sample_pinocchio_configuration(self):
        self._check_planning_deadline()
        backend = getattr(self, "robotics_backend", None)
        robot_name = getattr(self, "robotics_robot_name", None)
        if backend is not None and robot_name:
            try:
                return backend.sample_configuration(robot_name)
            except Exception:
                pass
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


