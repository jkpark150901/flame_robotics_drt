from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union
import numpy as np
import logging
import os
import threading
import time
import csv
import json
from pathlib import Path
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

from plugins.robotics.hppfcl_compat import build_bvh_model

@dataclass
class PlanningTarget:
    """계획할 단일 target pose. 하나의 로봇 job 안에서 순서대로 계획된다."""
    name: str
    target_pose: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InspectionGroup:
    """여러 로봇의 target을 함께 묶은 검사 그룹. Visualizer가 만들어 넘긴다."""
    name: str
    targets_by_robot: Dict[str, List[PlanningTarget]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotPlanningJob:
    """한 로봇이 순서대로 계획해야 할 target 목록."""
    robot_name: str
    start_q: Optional[np.ndarray]
    targets: List[PlanningTarget]
    obstacle_mesh: Any = None
    planner_name: str = "rrt_connect"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetPlanningResult:
    """target 하나의 계획 결과."""
    target_name: str
    success: bool
    q_path: List[np.ndarray] = field(default_factory=list)
    tcp_path: List[np.ndarray] = field(default_factory=list)
    goal_q: Optional[np.ndarray] = None
    error: Optional[str] = None
    ik_failure: Optional[Dict[str, Any]] = None
    verification: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    # 표준 필드 외에 소비자(Visualizer)가 렌더링/응답용으로 쓰는 원본 데이터를
    # 담아두는 확장 슬롯. plan 문서에는 없지만, IK/렌더링 세부정보를 잃지 않기 위해 추가했다.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotPlanningResult:
    """한 로봇 job 전체(순차 target 목록)의 계획 결과."""
    robot_name: str
    success: bool
    target_results: List[TargetPlanningResult]
    final_q: Optional[np.ndarray]
    error: Optional[str] = None
    timing: Dict[str, float] = field(default_factory=dict)


@dataclass
class BatchPlanningResult:
    """여러 로봇 job을 병렬로 계획한 전체 결과."""
    success: bool
    robot_results: Dict[str, RobotPlanningResult]
    failures: Dict[str, str]
    ik_failures: Dict[str, Dict[str, Any]]
    wall_elapsed: float
    cancelled: bool = False
    timing: Dict[str, float] = field(default_factory=dict)


@dataclass
class GroupPartitionResult:
    """검사 그룹을 접근 가능/유예로 나눈 결과."""
    reachable: List[Any]
    deferred: List[Any]
    evaluation_errors: Dict[str, str] = field(default_factory=dict)


@dataclass
class _JointSpaceTestCollisionResult:
    collision: bool
    pairs: List[tuple] = field(default_factory=list)
    q: Optional[np.ndarray] = None
    alpha: Optional[float] = None


class _JointSpaceTestBackend:
    """Small q-space backend for planner unit tests without Pinocchio."""

    name = "joint_space_test"

    def __init__(
        self,
        *,
        dof: int,
        lower_limits,
        upper_limits,
        collision_fn=None,
        edge_collision_fn=None,
        sample_fn=None,
        sample_resolution: float = 0.1,
    ):
        self._dof = int(dof)
        self._lo = np.asarray(lower_limits, dtype=float).reshape(self._dof)
        self._hi = np.asarray(upper_limits, dtype=float).reshape(self._dof)
        invalid = ~np.isfinite(self._lo) | ~np.isfinite(self._hi) | (self._hi <= self._lo)
        self._lo[invalid] = -np.pi
        self._hi[invalid] = np.pi
        self._span = self._hi - self._lo
        self._span[self._span < 1e-9] = 1.0
        self._collision_fn = collision_fn
        self._edge_collision_fn = edge_collision_fn
        self._sample_fn = sample_fn
        self._sample_resolution = max(float(sample_resolution), 1e-9)

    def dof(self, robot_name):
        return self._dof

    def configure_collision(self, robot_name, static_meshes=None, sample_resolution=None):
        if sample_resolution is not None:
            self._sample_resolution = max(float(sample_resolution), 1e-9)

    def _coerce_collision_result(self, value, *, q=None, alpha=None):
        if isinstance(value, _JointSpaceTestCollisionResult):
            return value
        pairs = []
        collision = bool(value)
        hit_q = q
        hit_alpha = alpha
        if isinstance(value, dict):
            collision = bool(value.get("collision", value.get("hit", False)))
            pairs = [tuple(pair) for pair in value.get("pairs", [])]
            hit_q = value.get("q", q)
            hit_alpha = value.get("alpha", alpha)
        elif isinstance(value, tuple):
            collision = bool(value[0]) if len(value) >= 1 else False
            if len(value) >= 2 and value[1] is not None:
                pairs = [tuple(pair) for pair in value[1]]
            if len(value) >= 3:
                hit_q = value[2]
            if len(value) >= 4:
                hit_alpha = value[3]
        return _JointSpaceTestCollisionResult(
            collision=collision,
            pairs=pairs,
            q=None if hit_q is None else np.asarray(hit_q, dtype=float),
            alpha=None if hit_alpha is None else float(hit_alpha),
        )

    def check_collision(self, robot_name, q, return_pairs=False):
        q = np.asarray(q, dtype=float).reshape(self._dof)
        value = False if self._collision_fn is None else self._collision_fn(q)
        result = self._coerce_collision_result(value, q=q, alpha=0.0)
        return result

    def check_edge_collision(self, robot_name, q_from, q_to, return_pairs=False):
        q_from = np.asarray(q_from, dtype=float).reshape(self._dof)
        q_to = np.asarray(q_to, dtype=float).reshape(self._dof)
        if self._edge_collision_fn is not None:
            value = self._edge_collision_fn(q_from, q_to)
            return self._coerce_collision_result(value)

        distance = float(np.linalg.norm(q_to - q_from))
        steps = max(1, int(np.ceil(distance / self._sample_resolution)))
        for i in range(steps + 1):
            alpha = i / steps
            q = (1.0 - alpha) * q_from + alpha * q_to
            result = self.check_collision(robot_name, q, return_pairs=True)
            if result.collision:
                result.q = q
                result.alpha = float(alpha)
                return result
        return _JointSpaceTestCollisionResult(False, [], None, None)

    def sample_configuration(self, robot_name):
        if self._sample_fn is not None:
            return np.asarray(self._sample_fn(), dtype=float).reshape(self._dof)
        return np.random.uniform(self._lo, self._hi)

    def joint_limits_for_metric(self, robot_name, normalize=True):
        if not normalize:
            return None, None, None
        return self._lo.copy(), self._hi.copy(), self._span.copy()

    def normalize_q(self, robot_name, q, normalize=True):
        q = np.asarray(q, dtype=float).reshape(self._dof)
        if not normalize:
            return q.copy()
        return (q - self._lo) / self._span

    def denormalize_q(self, robot_name, q_norm, normalize=True):
        q_norm = np.asarray(q_norm, dtype=float).reshape(self._dof)
        if not normalize:
            return q_norm.copy()
        return np.minimum(np.maximum(self._lo + q_norm * self._span, self._lo), self._hi)

    def joint_distance(self, robot_name, q_a, q_b, normalize=True):
        return float(np.linalg.norm(
            self.normalize_q(robot_name, q_b, normalize=normalize)
            - self.normalize_q(robot_name, q_a, normalize=normalize)
        ))

    def joint_distances(self, robot_name, q_points, q_ref, normalize=True):
        pts = np.asarray(q_points, dtype=float)
        ref = np.asarray(q_ref, dtype=float)
        if pts.ndim == 1:
            return np.asarray([self.joint_distance(robot_name, pts, ref, normalize=normalize)])
        return np.linalg.norm(
            np.asarray([self.normalize_q(robot_name, q, normalize=normalize) for q in pts])
            - self.normalize_q(robot_name, ref, normalize=normalize),
            axis=1,
        )

    def steer_joint_state(self, robot_name, from_state, to_state, step_size, normalize=True):
        from_norm = self.normalize_q(robot_name, from_state, normalize=normalize)
        to_norm = self.normalize_q(robot_name, to_state, normalize=normalize)
        direction = to_norm - from_norm
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            return np.asarray(from_state, dtype=float).copy()
        new_norm = from_norm + direction / length * min(float(step_size), length)
        return self.denormalize_q(robot_name, new_norm, normalize=normalize)

    def collision_geometry_summary(self, robot_name):
        return []

    def collision_pair_summary(
        self,
        robot_name,
        include_robot_self=True,
        include_static=True,
        limit=None,
    ):
        return []


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
        self.debug_convergence = False
        self.debug_output_dir = os.path.join(os.getcwd(), "debug", "planner")
        self.last_exploration_csv = None
        self.last_exploration_plot = None
        self.last_convergence_csv = None
        self.last_convergence_plot = None
        self._convergence_rows = None
        self._convergence_context = None
        # 어떤 검사 지점/자세/로봇에 대한 계획인지 로그에서 바로 구분할 수 있게 호출부
        # (viewer)가 채워주는 식별 라벨. 예: "Point 2 - Inspection pose 2 / dda_rb10_1300e".
        self.debug_context = None
        # q-space(joint-space) 계획에서 world-space workspace 제한을 적용하고 싶을 때
        # 쓰는 FK 기준 frame 이름. None이면(기본) workspace 체크를 하지 않는다 - bounds는
        # 있지만 이 이름이 없으면 기존 동작 그대로다.
        self.workspace_check_frame_name = None
        self.fixed_joint_indices = []
        self.fixed_joint_values = None
        self.fixed_joint_names = []
        self._fixed_joint_reference_q = None

    def _debug_prefix(self):
        context = getattr(self, "debug_context", None)
        return f"[{context}] " if context else ""

    def _debug_output_path(self, fallback_name="planner"):
        """Return a debug output directory rooted under a folder named debug."""
        configured = getattr(self, "debug_output_dir", None)
        path = Path(configured) if configured else Path(fallback_name)
        parts_lower = [part.lower() for part in path.parts]
        if path.is_absolute():
            if "debug" in parts_lower:
                return path
            return Path(os.getcwd()) / "debug" / path.name
        if parts_lower and parts_lower[0] == "debug":
            return Path(os.getcwd()) / path
        return Path(os.getcwd()) / "debug" / path

    def _debug_file_logger(self):
        """탐색/타이밍 로그 전용 파일 logger. 콘솔/메인 로그와 섞이지 않는 별도 파일에 DEBUG로 남긴다.

        이전에는 print()로 바로 stdout에 찍어서 항상 보였는데, 양이 많고 매 target마다
        반복돼서 메인 콘솔/로그를 채웠다. 여기서는 별도 파일 handler를 가진 전용 logger를
        만들어 DEBUG 레벨로만 남기고 root/console logger로는 전파(propagate)하지 않는다.
        """
        cached = getattr(self, "_debug_file_logger_instance", None)
        if cached is not None:
            return cached

        out_dir = self._debug_output_path("planner")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"{self.__class__.__name__.lower()}_debug.log"

        logger_name = f"flame_robotics.{self.__class__.__name__.lower()}_debug"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # 콘솔/메인 로그 파일로 새어나가지 않게 한다.
        if not logger.handlers:
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
            logger.addHandler(handler)

        self._debug_file_logger_instance = logger
        return logger

    def _log_block(self, title, lines):
        """탐색/타이밍 로그를 '어느 지점/자세/로봇' 표시 + 줄바꿈된 블록으로 별도 debug 파일에 남긴다.

        Args:
            title: 블록 제목(예: "joint-space summary").
            lines: 본문에 한 줄씩 들어갈 문자열 목록.
        """
        header = f"{self._debug_prefix()}{self.__class__.__name__} {title}"
        body = "\n".join(f"  {line}" for line in lines)
        self._debug_file_logger().debug(f"{header}\n{body}" if body else header)

    def _check_planning_deadline(self):
        deadline = getattr(self, "planning_deadline", None)
        if deadline is not None and time.monotonic() > float(deadline):
            raise TimeoutError("path planning timeout")

    def configure(
        self,
        *,
        bounds=None,
        step_size=None,
        max_iter=None,
        collision_sample_resolution=None,
        robotics_backend=None,
        robotics_robot_name=None,
        robot_model=None,
        workspace_check_frame_name=None,
        fixed_joint_indices=None,
        fixed_joint_values=None,
        fixed_joints=None,
        joint_names=None,
    ):
        """Planner 공통 설정 진입점.

        외부(viewer 등)가 planner의 하위 클래스 속성을 직접 건드리지 않고
        이 추상 클래스 메서드만 호출해 설정하도록 한다. 하위 클래스가 특정
        속성(bounds/step_size/max_iter 등)을 노출하지 않으면 조용히 건너뛴다.

        Args:
            bounds: workspace planner의 sampling bounds dict. q-space planner가
                workspace_check_frame_name과 함께 쓰면 FK 기반 world 제한으로도 쓰인다.
            step_size: 확장/샘플링 step 크기. collision 샘플 해상도 기본값으로도 쓰인다.
            max_iter: 최대 반복 수. step 이름이 다른 planner를 위해 두 속성 모두 설정한다.
            collision_sample_resolution: edge collision 샘플 해상도. 미지정 시 step_size를 쓴다.
            robotics_backend: 충돌/모델 조회에 쓰는 robotics backend.
            robotics_robot_name: backend에서 사용할 로봇 이름.
            robot_model: q-space 모델. nq/createData가 있으면 pin_model/pin_data로도 연결한다.
            workspace_check_frame_name: 지정하면 q-space planner가 이 frame의 FK world
                position을 bounds로 제한한다(q-space 자체엔 world 제한이 없어서). None(기본)이면
                기존처럼 workspace 체크를 하지 않는다.
            fixed_joint_indices/fixed_joint_values/fixed_joints: q-space planning 중
                특정 joint를 고정하기 위한 설정. fixed_joints는 {index_or_name: value}
                dict 또는 index/name list를 받을 수 있다. value가 None이면 시작 q 값으로 고정한다.
        """
        if bounds is not None and hasattr(self, "bounds"):
            self.bounds = bounds
        if step_size is not None:
            if hasattr(self, "step_size"):
                self.step_size = float(step_size)
            if collision_sample_resolution is None:
                self.pin_collision_sample_resolution = float(step_size)
        if max_iter is not None:
            if hasattr(self, "max_iter"):
                self.max_iter = int(max_iter)
            if hasattr(self, "max_iterations"):
                self.max_iterations = int(max_iter)
        if collision_sample_resolution is not None:
            self.pin_collision_sample_resolution = float(collision_sample_resolution)
        if robotics_backend is not None:
            self.robotics_backend = robotics_backend
        if robotics_robot_name is not None:
            self.robotics_robot_name = robotics_robot_name
        if robot_model is not None:
            self.robot_model = robot_model
            if hasattr(robot_model, "nq"):
                self.pin_model = robot_model
            if hasattr(robot_model, "createData"):
                self.pin_data = robot_model.createData()
        if workspace_check_frame_name is not None:
            self.workspace_check_frame_name = workspace_check_frame_name
        if fixed_joints is not None or fixed_joint_indices is not None or fixed_joint_values is not None:
            self.configure_fixed_joints(
                fixed_joints=fixed_joints,
                fixed_joint_indices=fixed_joint_indices,
                fixed_joint_values=fixed_joint_values,
                joint_names=joint_names,
            )

    def _resolve_joint_index(self, joint, joint_names=None):
        if isinstance(joint, (int, np.integer)):
            return int(joint)
        text = str(joint)
        if text.strip().isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        names = list(joint_names or getattr(self, "fixed_joint_names", []) or [])
        if text in names:
            return int(names.index(text))
        raise ValueError(f"fixed joint not found: {joint}")

    def configure_fixed_joints(
        self,
        *,
        fixed_joints=None,
        fixed_joint_indices=None,
        fixed_joint_values=None,
        joint_names=None,
    ):
        """Configure q-space joints that should remain fixed during planning.

        fixed_joints can be a dict of {joint_index_or_name: value_or_None} or a
        list of indices/names. None values are resolved from start_q per planning run.
        """
        if joint_names is not None:
            self.fixed_joint_names = [str(name) for name in joint_names]

        entries = []
        if isinstance(fixed_joints, dict):
            entries.extend(fixed_joints.items())
        elif fixed_joints is not None:
            entries.extend((item, None) for item in fixed_joints)
        if fixed_joint_indices is not None:
            indices_input = list(fixed_joint_indices)
            values = fixed_joint_values
            if values is None or isinstance(values, (str, bytes)) or np.isscalar(values):
                values = [values] * len(indices_input)
            else:
                values = list(values)
            for idx, value in zip(indices_input, values):
                entries.append((idx, value))

        indices = []
        values = []
        for joint, value in entries:
            idx = self._resolve_joint_index(joint, joint_names=joint_names)
            if idx in indices:
                values[indices.index(idx)] = value
                continue
            indices.append(idx)
            values.append(value)

        self.fixed_joint_indices = indices
        self.fixed_joint_values = values if indices else None
        self._fixed_joint_reference_q = None

    def _has_fixed_joint_constraints(self):
        return bool(getattr(self, "fixed_joint_indices", []) or [])

    def _fixed_joint_value_for(self, local_idx, joint_idx, reference_q=None):
        values = getattr(self, "fixed_joint_values", None)
        if values is not None and local_idx < len(values) and values[local_idx] is not None:
            if isinstance(values[local_idx], str) and values[local_idx].strip().lower() in {"", "start", "current"}:
                pass
            else:
                return float(values[local_idx])
        ref = reference_q
        if ref is None:
            ref = getattr(self, "_fixed_joint_reference_q", None)
        if ref is None:
            return None
        ref = np.asarray(ref, dtype=float).reshape(-1)
        if 0 <= int(joint_idx) < ref.size:
            return float(ref[int(joint_idx)])
        return None

    def _apply_fixed_joints(self, q, reference_q=None):
        if not self._has_fixed_joint_constraints():
            return np.asarray(q, dtype=float).copy()
        arr = np.asarray(q, dtype=float).copy()
        for local_idx, joint_idx in enumerate(getattr(self, "fixed_joint_indices", []) or []):
            joint_idx = int(joint_idx)
            if joint_idx < 0 or joint_idx >= arr.size:
                continue
            value = self._fixed_joint_value_for(local_idx, joint_idx, reference_q=reference_q)
            if value is not None:
                arr[joint_idx] = value
        return arr

    def _prepare_fixed_joint_constraints(self, start_q, goal_q):
        start_q = np.asarray(start_q, dtype=float).copy()
        goal_q = np.asarray(goal_q, dtype=float).copy()
        if not self._has_fixed_joint_constraints():
            return start_q, goal_q
        self._fixed_joint_reference_q = start_q.copy()
        return (
            self._apply_fixed_joints(start_q, reference_q=start_q),
            self._apply_fixed_joints(goal_q, reference_q=start_q),
        )

    def _workspace_position_ok(self, q):
        """q-space q의 FK world position이 self.bounds 안에 있는지 확인한다.

        Args:
            q: raw q(joint vector).

        Returns:
            bool. workspace_check_frame_name이나 backend/bounds가 없으면(옵트인 안 됨)
            항상 True(체크 안 함) - 기존 동작을 그대로 유지하기 위한 fail-open이다.

        계산 과정:
            backend.frame_world_T로 지정한 frame의 world position을 FK로 구하고,
            bounds의 x/y/z min/max 안에 있는지만 본다.
        """
        frame_name = getattr(self, "workspace_check_frame_name", None)
        bounds = getattr(self, "bounds", None)
        if not frame_name or not bounds:
            return True
        backend, robot_name = self._robotics_collision_backend()
        if backend is None:
            return True
        try:
            T = backend.frame_world_T(robot_name, q, frame_name)
        except Exception:
            return True
        pos = np.asarray(T, dtype=float)[:3, 3]
        return (
            bounds.get("x_min", -np.inf) <= pos[0] <= bounds.get("x_max", np.inf)
            and bounds.get("y_min", -np.inf) <= pos[1] <= bounds.get("y_max", np.inf)
            and bounds.get("z_min", -np.inf) <= pos[2] <= bounds.get("z_max", np.inf)
        )

    def configure_collision(self, config: dict, default_sample_resolution: float = 1.0):
        self.pin_collision_sample_resolution = float(
            config.get("pinocchio_collision_sample_resolution", default_sample_resolution)
        )
        if config.get("pinocchio_collision", False):
            self.setup_pinocchio_collision(
                config.get("robot_urdf"),
                config.get("package_dirs"),
            )

    def configure_joint_space_test_environment(
        self,
        *,
        dof: int | None = None,
        lower_limits=None,
        upper_limits=None,
        collision_fn=None,
        edge_collision_fn=None,
        sample_fn=None,
        robot_name: str = "test_robot",
        sample_resolution: float | None = None,
        use_joint_space_planning: bool = True,
    ):
        """Configure a lightweight q-space backend for planner tests.

        This keeps unit tests independent from Pinocchio/robotics scene setup while
        exercising the same PlannerBase q-space helpers used by production planners.
        """
        if dof is None:
            if lower_limits is not None:
                dof = int(np.asarray(lower_limits, dtype=float).reshape(-1).shape[0])
            elif upper_limits is not None:
                dof = int(np.asarray(upper_limits, dtype=float).reshape(-1).shape[0])
            else:
                raise ValueError("dof is required when joint limits are not provided")
        dof = int(dof)
        if lower_limits is None:
            lower_limits = np.full(dof, -np.pi, dtype=float)
        if upper_limits is None:
            upper_limits = np.full(dof, np.pi, dtype=float)

        resolution = (
            self.pin_collision_sample_resolution
            if sample_resolution is None
            else float(sample_resolution)
        )
        self.robotics_backend = _JointSpaceTestBackend(
            dof=dof,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            collision_fn=collision_fn,
            edge_collision_fn=edge_collision_fn,
            sample_fn=sample_fn,
            sample_resolution=resolution,
        )
        self.robotics_robot_name = str(robot_name)
        self.pin_collision_sample_resolution = max(float(resolution), 1e-9)
        self.use_joint_space_planning = bool(use_joint_space_planning)
        return self.robotics_backend

    def partition_and_sort_groups(
        self,
        groups: Sequence[Any],
        *,
        is_reachable: Callable[[Any], bool],
        reachable_sort_key: Optional[Callable[[Any], Any]] = None,
        deferred_sort_key: Optional[Callable[[Any], Any]] = None,
        reachability_error_policy: Literal["defer", "raise"] = "defer",
    ) -> GroupPartitionResult:
        """검사 그룹을 접근 가능(reachable)/유예(deferred)로 나누고 각각 정렬한다.

        Args:
            groups: InspectionGroup 또는 이와 동등한 group 객체 목록.
            is_reachable: group 하나를 받아 "지금 바로 접근 가능한지" bool을 반환하는 콜백.
                판단 자체는 application(Visualizer)이 하고, 여기서는 그 결과로 분류만 한다.
            reachable_sort_key: reachable 목록 정렬 키. None이면 입력 순서를 유지한다.
            deferred_sort_key: deferred 목록 정렬 키. None이면 입력 순서를 유지한다.
            reachability_error_policy: is_reachable이 예외를 던졌을 때 처리 방식.
                "defer"면 해당 group을 deferred로 넘기고 에러를 기록한다. "raise"면 즉시 전파한다.

        Returns:
            GroupPartitionResult(reachable, deferred, evaluation_errors).

        계산 과정:
            입력 순서대로 순회하며 is_reachable을 평가해 reachable/deferred로 나눈 뒤,
            각각 안정 정렬(동률은 입력 순서 유지)을 적용한다.
        """
        reachable: List[Any] = []
        deferred: List[Any] = []
        evaluation_errors: Dict[str, str] = {}

        for group in groups:
            group_name = getattr(group, "name", None)
            if group_name is None and isinstance(group, dict):
                group_name = group.get("name")
            try:
                ok = bool(is_reachable(group))
            except Exception as exc:
                evaluation_errors[str(group_name)] = str(exc)
                if reachability_error_policy == "raise":
                    raise
                deferred.append(group)
                continue
            (reachable if ok else deferred).append(group)

        if reachable_sort_key is not None:
            reachable.sort(key=reachable_sort_key)
        if deferred_sort_key is not None:
            deferred.sort(key=deferred_sort_key)

        return GroupPartitionResult(
            reachable=reachable, deferred=deferred, evaluation_errors=evaluation_errors
        )

    def plan_target_sequence(
        self,
        job: RobotPlanningJob,
        plan_target_fn: Callable[[RobotPlanningJob, PlanningTarget, Optional[np.ndarray]], TargetPlanningResult],
        *,
        fail_policy: Literal["stop_robot", "skip_target", "raise"] = "stop_robot",
        timeout_sec: Optional[float] = None,
        cancellation_event: Optional[threading.Event] = None,
    ) -> RobotPlanningResult:
        """한 로봇의 target들을 순서대로 계획하며 마지막 q를 다음 target의 시작 q로 넘긴다.

        Args:
            job: 계획할 target 목록과 시작 q(job.start_q)를 담은 RobotPlanningJob.
            plan_target_fn: (job, target, start_q) -> TargetPlanningResult. 실제 IK/경로 생성은
                이 콜백에 위임한다(로봇 backend/IK는 application 쪽 지식이라 PlannerBase가 직접 모른다).
            fail_policy: target 실패 시 처리 방식.
                "stop_robot": 남은 target을 건너뛰고 중단.
                "skip_target": 실패를 기록하고 마지막 성공 q에서 다음 target을 계속 시도.
                "raise": 첫 실패를 즉시 예외로 전파.
            timeout_sec: 이 로봇 job 전체에 적용할 예산(초). 초과하면 남은 target은 실패 처리한다.
            cancellation_event: 설정되면(다른 로봇의 stop_all 등) 이후 target을 즉시 중단한다.

        Returns:
            RobotPlanningResult(robot_name, success, target_results, final_q, error, timing).

        계산 과정:
            1. current_q = job.start_q로 시작한다.
            2. target을 선언 순서대로 plan_target_fn(job, target, current_q)로 계획한다.
            3. 성공하면 q_path[-1]을 다음 current_q로 쓴다. 실패한 target의 q는 절대 쓰지 않는다.
            4. timeout/cancellation은 다음 target 진입 전에만 확인한다(cooperative, 강제 중단 아님).
        """
        wall_t0 = time.perf_counter()
        deadline = None if timeout_sec is None else time.monotonic() + float(timeout_sec)
        current_q = job.start_q
        target_results: List[TargetPlanningResult] = []
        robot_success = True
        robot_error: Optional[str] = None

        for target in job.targets:
            if cancellation_event is not None and cancellation_event.is_set():
                target_results.append(TargetPlanningResult(
                    target_name=target.name, success=False, error="cancelled"))
                robot_success = False
                continue
            if deadline is not None and time.monotonic() > deadline:
                target_results.append(TargetPlanningResult(
                    target_name=target.name, success=False, error="robot planning timeout"))
                robot_success = False
                if fail_policy == "raise":
                    robot_error = "robot planning timeout"
                    break
                if fail_policy == "stop_robot":
                    break
                continue  # skip_target: 마지막 성공 q 그대로 다음 target 진행

            try:
                target_result = plan_target_fn(job, target, current_q)
            except Exception as exc:
                target_result = TargetPlanningResult(
                    target_name=target.name, success=False, error=str(exc))

            target_results.append(target_result)
            if target_result.success:
                if target_result.q_path:
                    current_q = target_result.q_path[-1]
                continue

            robot_success = False
            if fail_policy == "raise":
                robot_error = target_result.error or f"target '{target.name}' failed"
                break
            if fail_policy == "stop_robot":
                break
            # skip_target: current_q(마지막 성공 상태)를 그대로 유지하고 다음 target 진행

        result = RobotPlanningResult(
            robot_name=job.robot_name,
            success=robot_success,
            target_results=target_results,
            final_q=current_q,
            error=robot_error,
            timing={"wall": time.perf_counter() - wall_t0},
        )
        if fail_policy == "raise" and robot_error is not None:
            raise RuntimeError(f"robot '{job.robot_name}' planning failed: {robot_error}")
        return result

    def plan_batch(
        self,
        jobs: Sequence[RobotPlanningJob],
        plan_target_fn: Callable[[RobotPlanningJob, PlanningTarget, Optional[np.ndarray]], TargetPlanningResult],
        *,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        fail_policy: Literal["stop_robot", "skip_target", "stop_all", "raise"] = "stop_robot",
        timeout_sec: Optional[float] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> BatchPlanningResult:
        """서로 다른 로봇 job들을 병렬로, 각 로봇의 target들은 순차로 계획한다.

        Args:
            jobs: robot별 RobotPlanningJob 목록. 서로 다른 로봇은 독립적으로 병렬 실행된다.
            plan_target_fn: plan_target_sequence에 그대로 전달되는 단일 target 계획 콜백.
            parallel: False면 job을 순서대로 하나씩(디버깅용) 처리한다.
            max_workers: 동시 작업자 수. None이면 job 개수만큼(그 이상은 의미 없음).
            fail_policy: "stop_robot"/"skip_target"은 plan_target_sequence로 그대로 전달된다.
                "stop_all"은 실패 발생 시 cancellation_event를 세팅해 다른 job도 협조적으로 멈추게 한다.
                "raise"는 첫 실패를 재전파하되, 먼저 cancellation을 걸어 나머지 작업을 정리한다.
            timeout_sec: 각 로봇 job에 적용되는 예산(초). robot-level 마감이다.
            executor: 외부에서 준 executor. 주면 이 함수는 shutdown하지 않는다(소유권은 호출자).
                None이면 이 함수가 만들고 끝나면 닫는다.

        Returns:
            BatchPlanningResult(success, robot_results, failures, ik_failures, wall_elapsed, cancelled).

        계산 과정:
            1. job마다 plan_target_sequence를 서로 다른 스레드에서 실행한다(로봇 간 병렬).
            2. 실패한 target들을 "robot:target" 키로 failures/ik_failures에 모은다.
            3. 모든 로봇이 성공해야 전체 success다.
        """
        wall_t0 = time.perf_counter()
        cancellation_event = threading.Event()
        owns_executor = executor is None
        jobs = list(jobs)
        worker_count = max(1, min(int(max_workers), len(jobs))) if max_workers else max(1, len(jobs))

        robot_results: Dict[str, RobotPlanningResult] = {}

        def _run_job(job: RobotPlanningJob) -> RobotPlanningResult:
            return self.plan_target_sequence(
                job,
                plan_target_fn,
                fail_policy="raise" if fail_policy == "raise" else (
                    "stop_robot" if fail_policy == "stop_all" else fail_policy
                ),
                timeout_sec=timeout_sec,
                cancellation_event=cancellation_event,
            )

        if parallel and len(jobs) > 1:
            pool = executor if executor is not None else ThreadPoolExecutor(max_workers=worker_count)
            try:
                future_map = {pool.submit(_run_job, job): job for job in jobs}
                for future in as_completed(future_map):
                    job = future_map[future]
                    try:
                        job_result = future.result()
                    except Exception as exc:
                        job_result = RobotPlanningResult(
                            robot_name=job.robot_name,
                            success=False,
                            target_results=[],
                            final_q=job.start_q,
                            error=str(exc),
                        )
                    robot_results[job.robot_name] = job_result
                    if not job_result.success and fail_policy == "stop_all":
                        cancellation_event.set()
            finally:
                if owns_executor:
                    pool.shutdown(wait=True)
        else:
            for job in jobs:
                if cancellation_event.is_set():
                    robot_results[job.robot_name] = RobotPlanningResult(
                        robot_name=job.robot_name, success=False, target_results=[],
                        final_q=job.start_q, error="cancelled",
                    )
                    continue
                try:
                    job_result = _run_job(job)
                except Exception as exc:
                    job_result = RobotPlanningResult(
                        robot_name=job.robot_name, success=False, target_results=[],
                        final_q=job.start_q, error=str(exc),
                    )
                robot_results[job.robot_name] = job_result
                if not job_result.success and fail_policy == "stop_all":
                    cancellation_event.set()

        failures: Dict[str, str] = {}
        ik_failures: Dict[str, Dict[str, Any]] = {}
        for robot_name, job_result in robot_results.items():
            for target_result in job_result.target_results:
                if target_result.success:
                    continue
                key = f"{robot_name}:{target_result.target_name}"
                failures[key] = target_result.error or "planning failed"
                if target_result.ik_failure:
                    ik_failures[key] = target_result.ik_failure
            if job_result.error and not job_result.target_results:
                failures[f"{robot_name}:job"] = job_result.error

        overall_success = bool(robot_results) and all(r.success for r in robot_results.values())
        batch_result = BatchPlanningResult(
            success=overall_success,
            robot_results=robot_results,
            failures=failures,
            ik_failures=ik_failures,
            wall_elapsed=time.perf_counter() - wall_t0,
            cancelled=cancellation_event.is_set(),
        )
        if fail_policy == "raise" and not overall_success:
            first_error = next(
                (r.error for r in robot_results.values() if r.error),
                next(iter(failures.values()), "batch planning failed"),
            )
            raise RuntimeError(str(first_error))
        return batch_result

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
        target_pose  = np.asarray(target_pose, dtype=float)

        if getattr(self, "use_joint_space_planning", False) and self._has_robot_q_space_model():
            dof = self._robot_dof()
            if current_pose.shape[0] != dof or target_pose.shape[0] != dof:
                raise ValueError(
                    f"{self.__class__.__name__} is configured for joint-space planning, "
                    f"so generate() must receive q-space states with dof={dof}; "
                    f"got {current_pose.shape[0]}->{target_pose.shape[0]}"
                )
            current_pose, target_pose = self._prepare_fixed_joint_constraints(current_pose, target_pose)
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

    def _robot_dof(self):
        """Return q dimension from robotics backend first, then legacy Pinocchio model."""
        backend, robot_name = self._robotics_collision_backend()
        if backend is not None:
            try:
                return int(backend.dof(robot_name))
            except Exception:
                pass
        if self.pin_model is not None:
            return int(self.pin_model.nq)
        return None

    def _has_robot_q_space_model(self):
        """Whether this planner has enough robot information for q-space planning."""
        return self._robot_dof() is not None

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

    def add_collision_objects(self, object_models):
        """여러 static obstacle mesh를 한 번의 configure_collision 호출로 등록한다.

        add_collision_object를 mesh마다 호출하면 configure_collision이 매번 실행되고,
        그때마다 static_meshes 목록이 달라져(1개 -> 2개) backend의 BVH 캐시 key가 매번
        달라진다. 그러면 target마다 BVH를 다시 쌓느라(100k mesh 기준 수 초) setup이 느려진다.
        전체 목록을 한 번에 넘기면 key가 안정적이라 같은 mesh들에 대해 캐시가 hit한다.
        """
        added = [m for m in (object_models or []) if m is not None]
        if not added:
            return None
        self.collision_objects.extend(added)
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
            last = None
            for object_model in added:
                try:
                    last = self._add_pinocchio_collision_mesh(object_model)
                except Exception as e:
                    print(f"Error adding object to Pinocchio collision scene: {e}")
            return last
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

        return build_bvh_model(hppfcl, vertices, triangles)

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
        backend, robot_name = self._robotics_collision_backend()
        if backend is not None:
            return list(backend.collision_geometry_summary(robot_name))
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

    def static_geometry_aabb_summary(self):
        """등록된 static obstacle(배관, positioner 등)의 world AABB를 이름과 함께 반환한다.

        배관이 실제로 회전된 위치에 등록됐는지("가상 회전"이 정말 collision scene에
        반영됐는지)를 숫자로 바로 확인하기 위한 진단용이다.
        """
        try:
            geometries = self.pinocchio_collision_geometry_summary()
        except Exception:
            return []
        return [
            {
                "name": item.get("name"),
                "aabb_min": item.get("aabb_min"),
                "aabb_max": item.get("aabb_max"),
            }
            for item in geometries
            if item.get("kind") == "static" and item.get("aabb_min") is not None
        ]

    def pinocchio_collision_pair_summary(self, include_robot_self=True, include_static=True, limit=None):
        """Return the collision pairs checked by Pinocchio."""
        backend, robot_name = self._robotics_collision_backend()
        if backend is not None:
            return list(backend.collision_pair_summary(
                robot_name,
                include_robot_self=include_robot_self,
                include_static=include_static,
                limit=limit,
            ))
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

    def check_robot_collision(self, q, return_pairs=False):
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

    def check_pinocchio_collision(self, q, return_pairs=False):
        """Backward-compatible alias. Prefer check_robot_collision()."""
        return self.check_robot_collision(q, return_pairs=return_pairs)

    def _check_robot_edge_collision(self, p1, p2):
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

    def _check_pinocchio_collision(self, p1, p2):
        """Backward-compatible alias. Prefer _check_robot_edge_collision()."""
        return self._check_robot_edge_collision(p1, p2)

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
            hit, hit_pairs = self.check_robot_collision(q, return_pairs=True)
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
        robot_collision = self._check_robot_edge_collision(p1, p2)
        if robot_collision is not None:
            return robot_collision
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
            hit, pairs = self.check_robot_collision(q, return_pairs=True)
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
                hit, pairs = self.check_robot_collision(q, return_pairs=True)
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
                    self._apply_fixed_joints(q_a),
                    self._apply_fixed_joints(q_b),
                    normalize=bool(getattr(self, "normalize_joint_space", True)),
                )
            except Exception:
                pass
        return float(np.linalg.norm(
            self._normalize_joint_q(self._apply_fixed_joints(q_b))
            - self._normalize_joint_q(self._apply_fixed_joints(q_a))
        ))

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
        ref = self._apply_fixed_joints(q_ref)
        if pts.ndim == 1:
            pts = self._apply_fixed_joints(pts)
        elif self._has_fixed_joint_constraints():
            pts = np.asarray([self._apply_fixed_joints(q) for q in pts], dtype=float)
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
        from_state = self._apply_fixed_joints(from_state)
        to_state = self._apply_fixed_joints(to_state)
        if backend is not None and robot_name:
            try:
                return self._apply_fixed_joints(backend.steer_joint_state(
                    robot_name,
                    from_state,
                    to_state,
                    step_size,
                    normalize=bool(getattr(self, "normalize_joint_space", True)),
                ))
            except Exception:
                pass
        if not getattr(self, "normalize_joint_space", True) or self.pin_model is None:
            return self._apply_fixed_joints(self._steer_state(from_state, to_state, step_size))
        from_norm = self._normalize_joint_q(from_state)
        to_norm = self._normalize_joint_q(to_state)
        direction = to_norm - from_norm
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            return np.asarray(from_state, dtype=float).copy()
        new_norm = from_norm + direction / length * min(float(step_size), length)
        return self._apply_fixed_joints(self._denormalize_joint_q(new_norm))

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

    def _new_convergence_rows(self):
        """Planner 수렴 로그 row 컨테이너를 만든다."""
        return [] if (
            getattr(self, "debug_convergence", False)
            or getattr(self, "debug_exploration", False)
        ) else None

    def _planner_state_distance(self, space, state_a, state_b):
        """q-space/task-space 상태 사이의 목표 수렴 거리 metric."""
        if state_a is None or state_b is None:
            return None
        a = np.asarray(state_a, dtype=float).reshape(-1)
        b = np.asarray(state_b, dtype=float).reshape(-1)
        if a.shape != b.shape:
            n = min(a.size, b.size)
            if n <= 0:
                return None
            a = a[:n]
            b = b[:n]
        if str(space).lower() in {"q", "joint", "joint_space", "q_space"}:
            try:
                return float(self._joint_distance(a, b))
            except Exception:
                return float(np.linalg.norm(a - b))
        if str(space).lower() in {"task", "task_space", "workspace"} and a.size >= 3 and b.size >= 3:
            weights = getattr(self, "weights", {}) or {}
            w_pos = float(weights.get("pos", 1.0))
            w_ori = float(weights.get("orient", 0.5))
            pos = w_pos * float(np.sum((a[:3] - b[:3]) ** 2))
            ori = 0.0
            if a.size > 3 and b.size > 3:
                ori_n = min(a.size, b.size) - 3
                ori = w_ori * float(np.sum((a[3:3 + ori_n] - b[3:3 + ori_n]) ** 2))
            return float(np.sqrt(pos + ori))
        return float(np.linalg.norm(a - b))

    def _begin_convergence_debug(self, space, start_state, goal_state):
        """현재 planning run의 수렴 로그 컨텍스트를 시작한다."""
        rows = self._new_convergence_rows()
        self._convergence_rows = rows
        self._convergence_context = {
            "space": str(space),
            "start": None if start_state is None else np.asarray(start_state, dtype=float).copy(),
            "goal": None if goal_state is None else np.asarray(goal_state, dtype=float).copy(),
            "best_distance_to_goal": float("inf"),
            "start_time": time.perf_counter(),
        }
        if rows is not None:
            self._record_convergence(
                rows,
                iteration=0,
                phase="start",
                state=start_state,
                node_count=1,
                accepted=True,
                reason="start_state",
            )
        return rows

    def _record_convergence(
        self,
        rows,
        iteration,
        phase,
        *,
        state=None,
        sample_state=None,
        node_count=None,
        cost=None,
        accepted=False,
        collision=False,
        reason="",
        elapsed_s=None,
    ):
        """q/task space planning의 수렴 상태를 CSV row로 기록한다."""
        if rows is None:
            return
        context = getattr(self, "_convergence_context", None) or {}
        space = str(context.get("space", "unknown"))
        goal = context.get("goal")
        state_distance = self._planner_state_distance(space, state, goal)
        sample_distance = self._planner_state_distance(space, sample_state, goal)

        best_distance = context.get("best_distance_to_goal", float("inf"))
        if state_distance is not None and np.isfinite(state_distance):
            best_distance = min(float(best_distance), float(state_distance))
        context["best_distance_to_goal"] = best_distance

        self._convergence_context = context

        if elapsed_s is None:
            start_time = context.get("start_time")
            elapsed_s = "" if start_time is None else time.perf_counter() - float(start_time)

        normalized_iteration = max(0, int(iteration)) if iteration is not None else 0
        rows.append({
            "iteration": normalized_iteration,
            "space": space,
            "phase": phase,
            "elapsed_s": elapsed_s,
            "distance_to_goal": "" if state_distance is None else float(state_distance),
            "best_distance_to_goal": (
                "" if not np.isfinite(best_distance) else float(best_distance)
            ),
            "sample_distance_to_goal": "" if sample_distance is None else float(sample_distance),
            "accepted": bool(accepted),
            "collision": bool(collision),
            "reason": reason,
            "state": self._json_vector(state),
            "sample_state": self._json_vector(sample_state),
        })

    def _record_convergence_from_path(self, space, path, status="path"):
        """완성된 path waypoint들을 수렴 로그로 기록한다."""
        rows = self._begin_convergence_debug(space, path[0] if path else None, path[-1] if path else None)
        if rows is None:
            return rows
        for idx, state in enumerate(path or []):
            self._record_convergence(
                rows,
                iteration=idx,
                phase=status,
                state=state,
                accepted=True,
            )
        return rows

    def _save_convergence_debug(self, rows=None, space=None, status="finished", path_waypoints=None):
        """Planner 수렴 로그를 CSV와 PNG 그래프로 저장한다."""
        if rows is None:
            rows = getattr(self, "_convergence_rows", None)
        if not rows:
            return None, None
        context = getattr(self, "_convergence_context", None) or {}
        if space is None:
            space = context.get("space", "unknown")

        out_dir = self._debug_output_path("planner")
        out_dir.mkdir(parents=True, exist_ok=True)
        robot_name = "robot"
        try:
            robot_name = str(getattr(self, "robotics_robot_name", "") or getattr(self.pin_model, "name", "") or "robot")
        except Exception:
            pass
        planner_name = self.__class__.__name__.lower()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = out_dir / f"{planner_name}_{space}_convergence_{robot_name}_{stamp}_{status}"
        csv_path = base.with_suffix(".csv")
        fieldnames = [
            "iteration",
            "space",
            "phase",
            "elapsed_s",
            "distance_to_goal",
            "best_distance_to_goal",
            "sample_distance_to_goal",
            "accepted",
            "collision",
            "reason",
            "state",
            "sample_state",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        plot_path = None
        try:
            plot_path = self._save_convergence_plot(rows, base.with_suffix(".png"), path_waypoints)
        except Exception as exc:
            print(f"{self.__class__.__name__} convergence plot failed: {exc}")

        self.last_convergence_csv = str(csv_path)
        self.last_convergence_plot = None if plot_path is None else str(plot_path)
        self._log_block("convergence debug saved", [
            f"csv={self.last_convergence_csv}",
            f"plot={self.last_convergence_plot}",
        ])
        return self.last_convergence_csv, self.last_convergence_plot

    def _save_convergence_plot(self, rows, plot_path, path_waypoints=None):
        """수렴 CSV row를 PNG 그래프로 저장한다."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
        except Exception as exc:
            print(f"{self.__class__.__name__} convergence plot skipped: matplotlib unavailable ({exc})")
            return None

        elapsed = []
        iterations = []
        distance = []
        best_distance = []
        collisions = []
        accepted = []
        states = []
        goals = []
        context = getattr(self, "_convergence_context", None) or {}
        goal_state = context.get("goal")
        for row in rows:
            iterations.append(int(row.get("iteration", -1)))
            try:
                elapsed.append(float(row.get("elapsed_s", "")))
            except Exception:
                elapsed.append(float(iterations[-1]))

            def _float_or_nan(key):
                try:
                    value = row.get(key, "")
                    return np.nan if value == "" else float(value)
                except Exception:
                    return np.nan

            distance.append(_float_or_nan("distance_to_goal"))
            best_distance.append(_float_or_nan("best_distance_to_goal"))
            collisions.append(bool(row.get("collision", False)))
            accepted.append(bool(row.get("accepted", False)))
            try:
                states.append(np.asarray(json.loads(row.get("state") or "[]"), dtype=float))
            except Exception:
                states.append(np.asarray([], dtype=float))
            goals.append(None if goal_state is None else np.asarray(goal_state, dtype=float).reshape(-1))

        elapsed = np.asarray(elapsed, dtype=float)
        iterations = np.asarray(iterations, dtype=int)
        distance = np.asarray(distance, dtype=float)
        best_distance = np.asarray(best_distance, dtype=float)
        collisions = np.asarray(collisions, dtype=bool)
        accepted = np.asarray(accepted, dtype=bool)

        space = str(rows[-1].get("space", "unknown")) if rows else "unknown"
        valid_states = [state for state in states if state.size > 0]
        state_dim = max((state.size for state in valid_states), default=0)
        state_matrix = np.full((len(states), state_dim), np.nan, dtype=float)
        for idx, state in enumerate(states):
            n = min(state.size, state_dim)
            if n > 0:
                state_matrix[idx, :n] = state[:n]
        goal = goals[-1] if goals and goals[-1] is not None else None

        if space in {"task_space", "task", "workspace"} and state_dim > 3:
            fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        else:
            fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        axes[0].plot(iterations, distance, color="tab:blue", alpha=0.45, linewidth=1.0, label="distance to goal")
        axes[0].plot(iterations, best_distance, color="tab:green", linewidth=1.8, label="best distance")
        if np.any(collisions):
            axes[0].scatter(iterations[collisions], distance[collisions], s=16, color="tab:red", label="collision")
        axes[0].set_ylabel("Goal Distance")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best")

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
        if space in {"q_space", "q", "joint", "joint_space"}:
            for dim in range(state_dim):
                axes[1].plot(
                    iterations,
                    state_matrix[:, dim],
                    linewidth=1.2,
                    color=colors[dim % len(colors)],
                    label=f"q{dim + 1}",
                )
                if goal is not None and dim < goal.size:
                    axes[1].axhline(goal[dim], color=colors[dim % len(colors)], linestyle="--", alpha=0.25)
            axes[1].set_ylabel("Joint Angle")
            axes[1].set_xlabel("Iteration")
            axes[1].legend(loc="best", ncol=min(4, max(1, state_dim)))
        else:
            pos_labels = ["x", "y", "z"]
            for dim in range(min(3, state_dim)):
                axes[1].plot(
                    iterations,
                    state_matrix[:, dim],
                    linewidth=1.4,
                    color=colors[dim % len(colors)],
                    label=pos_labels[dim],
                )
                if goal is not None and dim < goal.size:
                    axes[1].axhline(goal[dim], color=colors[dim % len(colors)], linestyle="--", alpha=0.25)
            axes[1].set_ylabel("EF Position")
            axes[1].legend(loc="best")
            if len(axes) > 2:
                ori_labels = ["roll", "pitch", "yaw"]
                for dim in range(3, min(6, state_dim)):
                    axes[2].plot(
                        iterations,
                        state_matrix[:, dim],
                        linewidth=1.2,
                        color=colors[(dim - 3) % len(colors)],
                        label=ori_labels[dim - 3],
                    )
                    if goal is not None and dim < goal.size:
                        axes[2].axhline(goal[dim], color=colors[(dim - 3) % len(colors)], linestyle="--", alpha=0.25)
                axes[2].set_ylabel("EF Orientation")
                axes[2].set_xlabel("Iteration")
                axes[2].legend(loc="best")
            else:
                axes[1].set_xlabel("Iteration")
        axes[1].grid(True, alpha=0.25)
        if len(axes) > 2:
            axes[2].grid(True, alpha=0.25)

        for axis in axes:
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))

        space_title = "Joint Angles" if space in {"q_space", "q", "joint", "joint_space"} else "EF Pose"
        title = f"{self.__class__.__name__} Convergence | {space_title}"
        if path_waypoints is not None:
            title += f" | path waypoints={path_waypoints}"
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        return plot_path

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
        normalized_iteration = max(0, int(iteration)) if iteration is not None else 0
        rows.append({
            "iteration": normalized_iteration,
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

        convergence_rows = getattr(self, "_convergence_rows", None)
        if convergence_rows is not None:
            state = to_q
            if state is None:
                state = collision_q if collision_q is not None else from_q
            self._record_convergence(
                convergence_rows,
                iteration=iteration,
                phase=phase,
                state=state,
                sample_state=sample_q,
                node_count=node_count,
                cost=cost,
                accepted=accepted,
                collision=collision,
                reason=reason,
                elapsed_s=elapsed_s,
            )

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
            self._save_convergence_debug(status=status, path_waypoints=path_waypoints)
            return None, None
        out_dir = self._debug_output_path("planner")
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
        self._save_convergence_debug(status=status, path_waypoints=path_waypoints)
        self.last_exploration_csv = str(csv_path)
        self.last_exploration_plot = None if plot_path is None else str(plot_path)
        self._log_block("exploration debug saved", [
            f"csv={self.last_exploration_csv}",
            f"plot={self.last_exploration_plot}",
        ])
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
            from matplotlib.ticker import MaxNLocator
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
        axes[0].plot(iterations, node_counts, color="tab:blue", linewidth=1.5, label="nodes")
        if np.any(collisions):
            axes[0].scatter(
                iterations[collisions],
                node_counts[collisions],
                color="tab:red",
                s=18,
                label="collision reject",
                zorder=3,
            )
        if np.any(accepted):
            axes[0].scatter(
                iterations[accepted],
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
        axes[1].scatter(iterations, y, c=colors, s=15, alpha=0.8)
        axes[1].set_yticks(list(phase_to_y.values()))
        axes[1].set_yticklabels(list(phase_to_y.keys()))
        axes[1].set_ylabel("Event / Collision Source")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(
            iterations,
            cumulative_collision_time,
            color="tab:red",
            linewidth=1.5,
            label="cumulative collision-check time",
        )
        if np.any(collision_time > 0.0):
            axes[2].scatter(
                iterations[collision_time > 0.0],
                cumulative_collision_time[collision_time > 0.0],
                c=["tab:red" if c else "tab:blue" for c in collisions[collision_time > 0.0]],
                s=12,
                alpha=0.65,
                label="collision checks",
            )
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Collision Check Time (s)")
        axes[2].grid(True, alpha=0.25)
        axes[2].legend(loc="best")

        axes[3].plot(
            iterations,
            new_node_collision_count,
            color="tab:red",
            linewidth=1.4,
            label="new-node collisions",
        )
        axes[3].plot(
            iterations,
            random_new_node_collision_count,
            color="tab:pink",
            linewidth=1.2,
            linestyle="--",
            label="random new-node collisions",
        )
        axes[3].plot(
            iterations,
            rewire_collision_count,
            color="tab:purple",
            linewidth=1.4,
            label="rewire collisions",
        )
        axes[3].set_xlabel("Iteration")
        axes[3].set_ylabel("Collision Count")
        axes[3].grid(True, alpha=0.25)
        axes[3].legend(loc="best")

        for axis in axes:
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))

        title = f"{self.__class__.__name__} Exploration"
        if path_waypoints is not None:
            title += f" | path waypoints={path_waypoints}"
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        return plot_path

    def _sample_robot_configuration(self):
        self._check_planning_deadline()
        backend = getattr(self, "robotics_backend", None)
        robot_name = getattr(self, "robotics_robot_name", None)
        if backend is not None and robot_name:
            try:
                return self._apply_fixed_joints(backend.sample_configuration(robot_name))
            except Exception:
                pass
        if self.pin_model is None:
            raise RuntimeError("Pinocchio collision is not configured")
        lo = np.asarray(self.pin_model.lowerPositionLimit, dtype=float).copy()
        hi = np.asarray(self.pin_model.upperPositionLimit, dtype=float).copy()

        invalid = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
        lo[invalid] = -np.pi
        hi[invalid] = np.pi
        return self._apply_fixed_joints(np.random.uniform(lo, hi))

    def _sample_pinocchio_configuration(self):
        """Backward-compatible alias. Prefer _sample_robot_configuration()."""
        return self._sample_robot_configuration()

    def _steer_state(self, from_state, to_state, step_size):
        direction = np.asarray(to_state, dtype=float) - np.asarray(from_state, dtype=float)
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            return np.asarray(from_state, dtype=float).copy()
        return np.asarray(from_state, dtype=float) + direction / length * min(float(step_size), length)


