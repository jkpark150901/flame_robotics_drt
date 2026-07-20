from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from plugins.robotics.backend import IKOptions, IKResult, RoboticsBackend


@dataclass
class InspectionIKRequest:
    """한 로봇의 검사 자세 IK 입력값.

    Args:
        robot_name: backend에 등록된 로봇 이름.
        target_pose: 목표 TCP pose. 4x4 matrix, 6D pose, 또는 3D 위치를 허용한다.
        start_tcp_pose: viewer가 표시 중인 현재 TCP pose. 결과 로그의 start 필드에 사용한다.
        start_q: IK initial guess로 사용할 raw joint vector.
        frame_name: IK를 풀 대상 frame/link 이름.
        joint_names: raw q 순서와 대응되는 joint 이름 목록.
        planner_name: UI에서 선택된 planner 이름. IK check에서는 식별용 문자열이다.
        ik_config: damping, dt, tol, max_iter 등 IK 설정 dict.
        ik_solver: 요청에서 override된 solver 이름.
        ik_normalize: 요청에서 override된 joint 정규화 여부.
    """

    robot_name: str
    target_pose: Any
    start_tcp_pose: Sequence[float]
    start_q: Sequence[float]
    frame_name: str
    joint_names: Sequence[str]
    planner_name: str = "ik_check"
    ik_config: Dict[str, Any] = field(default_factory=dict)
    ik_solver: Optional[str] = None
    ik_normalize: Optional[bool] = None


class InspectionPlanningBase:
    """검사 IK/path planning 계산 코어.

    Viewer는 UI 상태, 시각화, ZAPI 응답을 담당하고 이 클래스는 로봇 backend를
    이용한 목표 pose 변환, IK solve, 충돌 요약, q-space path 검증용 데이터 구성을 담당한다.
    """

    def __init__(self, backend: RoboticsBackend):
        """InspectionPlanningBase를 초기화한다.

        Args:
            backend: URDF/FK/IK/collision을 제공하는 robotics backend.

        Returns:
            없음.

        계산 과정:
            전달받은 backend 참조를 저장한다. 실제 solver 라이브러리 종류는 backend 내부에 숨긴다.
        """
        self.backend = backend

    @staticmethod
    def target_goal_vector(target_pose: Any) -> np.ndarray:
        """로그/응답용 6D goal vector를 만든다.

        Args:
            target_pose: 4x4 transform, 6D pose, 또는 3D position.

        Returns:
            np.ndarray shape=(6,). 4x4 입력이면 translation만 채우고 rpy는 0으로 둔다.

        계산 과정:
            target_pose를 numpy 배열로 바꾼 뒤 shape에 따라 앞 3개 또는 6개 값을 복사한다.
        """
        goal = np.zeros(6, dtype=float)
        target_arr = np.asarray(target_pose, dtype=float)
        if target_arr.shape == (4, 4):
            goal[:3] = target_arr[:3, 3]
        else:
            flat_target = target_arr.reshape(-1)
            goal[:min(6, flat_target.size)] = flat_target[:min(6, flat_target.size)]
        return goal

    @staticmethod
    def ik_options(ik_config: Dict[str, Any], ik_solver=None, ik_normalize=None) -> IKOptions:
        """UI/config 값을 backend IK option으로 변환한다.

        Args:
            ik_config: damping, dt, tol, max_iter, qp_solver 등을 담은 dict.
            ik_solver: 요청 단위 solver override.
            ik_normalize: 요청 단위 normalize override.

        Returns:
            IKOptions 인스턴스.

        계산 과정:
            solver 이름을 먼저 결정하고, normalize가 명시되지 않았으면 legacy DLS 계열만
            raw joint-space로 간주한다. 나머지 수치 파라미터는 config 기본값으로 채운다.
        """
        solver_name = str(ik_solver or ik_config.get("solver", "normalized_dls") or "normalized_dls").lower()
        if ik_normalize is None:
            normalize_value = solver_name not in ("dls", "classic_dls", "legacy_dls")
        else:
            normalize_value = bool(ik_normalize)
        return IKOptions(
            solver=solver_name,
            normalize=normalize_value,
            damping=float(ik_config.get("damping", 1e-3)),
            dt=float(ik_config.get("dt", 0.35)),
            tol=float(ik_config.get("tol", 1e-4)),
            max_iter=int(ik_config.get("max_iter", 1000)),
            position_only_tol=float(ik_config.get("position_only_tol", 0.01)),
            backend_solver=str(ik_config.get("qp_solver", "quadprog")),
            record_trace=True,
        )

    @staticmethod
    def trace_to_rows(result: IKResult):
        """IKResult trace를 viewer/experiment 저장용 dict list로 변환한다.

        Args:
            result: backend.solve_ik 결과.

        Returns:
            list[dict]. 각 항목은 iteration, error, q, tcp_world를 포함한다.

        계산 과정:
            dataclass trace point의 ndarray를 copy해서 외부 변경에 영향을 받지 않게 한다.
        """
        return [
            {
                "iteration": item.iteration,
                "err_norm": item.err_norm,
                "position_error": item.position_error,
                "orientation_error": item.orientation_error,
                "q": item.q.copy(),
                "tcp_world": item.tcp_world.copy(),
            }
            for item in result.trace
        ]

    def ik_result_summary(self, request: InspectionIKRequest, q, target_world_T, ik_result: IKResult, fallback=False):
        """IK 결과를 UI 응답에 넣기 좋은 dict로 요약한다.

        Args:
            request: 원본 IK 요청.
            q: 성공 q 또는 fallback q.
            target_world_T: backend 기준 목표 transform.
            ik_result: backend.solve_ik 결과.
            fallback: 실패 후 마지막 q를 사용하는지 여부.

        Returns:
            dict: success/fallback/error/collision/iteration/solver/normalize 정보.

        계산 과정:
            backend FK로 reached pose를 다시 계산하고, 목표 transform과의 position/orientation
            error를 계산한다. collision model이 구성되어 있으면 현재 q collision도 함께 검사한다.
        """
        q = np.asarray(q, dtype=float)
        reached_T = self.backend.frame_world_T(request.robot_name, q, request.frame_name)
        target_world_T = np.asarray(target_world_T, dtype=float)
        position_error = float(np.linalg.norm(reached_T[:3, 3] - target_world_T[:3, 3]))
        rot_delta = reached_T[:3, :3].T @ target_world_T[:3, :3]
        cos_angle = (float(np.trace(rot_delta)) - 1.0) * 0.5
        orientation_error = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        collision = False
        collision_pair_count = 0
        try:
            collision_result = self.backend.check_collision(request.robot_name, q, return_pairs=True)
            collision = bool(collision_result.collision)
            collision_pair_count = len(collision_result.pairs)
        except Exception:
            pass
        return {
            "success": bool(ik_result.success),
            "fallback": bool(fallback),
            "position_error": position_error,
            "orientation_error": orientation_error,
            "collision": collision,
            "collision_pair_count": int(collision_pair_count),
            "iterations": ik_result.iterations,
            "elapsed": ik_result.elapsed,
            "max_iter": request.ik_config.get("max_iter"),
            "solver": ik_result.solver,
            "normalize": ik_result.normalize,
        }

    def check_inspection_ik_for_robot(self, request: InspectionIKRequest) -> Dict[str, Any]:
        """한 로봇의 검사 목표 pose에 대해 IK만 확인한다.

        Args:
            request: robot name, target pose, start q, frame name, IK 설정을 담은 요청 객체.

        Returns:
            dict: status, start_q, goal_q, IK result, failure info, reached/target transform, timing.

        계산 과정:
            1. target_pose를 backend 기준 world transform으로 해석한다.
            2. IKOptions를 구성하고 backend.solve_ik를 호출한다.
            3. 실패했더라도 backend가 반환한 마지막 q가 있으면 fallback q로 유지한다.
            4. reached pose, collision 여부, error를 요약한다.
            5. viewer가 바로 저장/시각화할 수 있도록 trace와 transform을 포함해 반환한다.
        """
        total_t0 = time.perf_counter()
        timings: Dict[str, float] = {}

        stage_t0 = time.perf_counter()
        start_q = np.asarray(request.start_q, dtype=float)
        target_world_T = self.backend.target_world_T(
            request.robot_name,
            request.target_pose,
            start_q,
            request.frame_name,
        )
        goal = self.target_goal_vector(request.target_pose)
        timings["target_setup"] = time.perf_counter() - stage_t0

        stage_t0 = time.perf_counter()
        options = self.ik_options(request.ik_config, request.ik_solver, request.ik_normalize)
        result = self.backend.solve_ik(
            request.robot_name,
            target_world_T,
            start_q,
            options=options,
            frame_name=request.frame_name,
        )
        timings["ik"] = time.perf_counter() - stage_t0

        ik_success = bool(result.success)
        ik_fallback = False
        ik_failure = None
        goal_q = result.q
        if goal_q is None:
            goal_q = start_q.copy()
            ik_fallback = True
            ik_failure = result.failure_info or self.backend.classify_ik_failure(
                request.robot_name,
                goal_q,
                target_world_T,
                result.final_T,
                orientation_error=result.orientation_error,
                max_iter=options.max_iter,
            )
        elif not ik_success:
            ik_fallback = True
            ik_failure = result.failure_info or self.backend.classify_ik_failure(
                request.robot_name,
                goal_q,
                target_world_T,
                result.final_T,
                orientation_error=result.orientation_error,
                max_iter=options.max_iter,
            )

        stage_t0 = time.perf_counter()
        ik_summary = self.ik_result_summary(
            request,
            goal_q,
            target_world_T,
            result,
            fallback=ik_fallback,
        )
        reached_T = self.backend.frame_world_T(request.robot_name, goal_q, request.frame_name)
        timings["ik_result_check"] = time.perf_counter() - stage_t0
        timings["total"] = time.perf_counter() - total_t0

        ik_collision = bool(ik_summary.get("collision", False))
        return {
            "status": "partial" if (ik_fallback or ik_collision) else "success",
            "planner": request.planner_name,
            "robot": request.robot_name,
            "pin_joint_names": list(request.joint_names),
            "start_q": start_q,
            "goal_q": np.asarray(goal_q, dtype=float),
            "start": np.asarray(request.start_tcp_pose, dtype=float).reshape(-1).tolist(),
            "goal": goal.tolist(),
            "ik_fallback": ik_fallback,
            "ik_failure": ik_failure,
            "ik_result": ik_summary,
            "ik_solver": result.solver,
            "ik_normalize": result.normalize,
            "collision_free": not ik_collision,
            "ik_reached_T": np.asarray(reached_T, dtype=float),
            "ik_target_T": np.asarray(target_world_T, dtype=float),
            "ik_trace": self.trace_to_rows(result),
            "timing": timings,
        }

    def plan_q_path_for_robot(
        self,
        *,
        planner,
        ik_request: InspectionIKRequest,
        q_start: Sequence[float],
        planning_timeout: float = 0.0,
    ) -> Dict[str, Any]:
        """IK 목표 q까지 q-space path planning을 수행한다.

        Args:
            planner: PlannerBase 호환 planner. generate와 verify_path를 제공해야 한다.
            ik_request: 목표 pose와 IK 설정.
            q_start: path planning 시작 raw q.
            planning_timeout: planner deadline. 0 이하면 비활성화.

        Returns:
            dict: IK check 결과에 q_path, verification, planning timing을 추가한 결과.

        계산 과정:
            1. check_inspection_ik_for_robot으로 목표 q를 구한다.
            2. planner.generate(q_start, goal_q)를 호출한다.
            3. timeout/empty path/fallback 상태를 collision preview 사유로 기록한다.
            4. planner.verify_path로 반환 path의 충돌 여부를 검증한다.
        """
        result = self.check_inspection_ik_for_robot(ik_request)
        q_start = np.asarray(q_start, dtype=float)
        goal_q = np.asarray(result["goal_q"], dtype=float)
        planning_error = None
        forced_collision_preview = bool(result.get("ik_fallback", False))
        fallback_reason = None

        if planning_timeout > 0 and hasattr(planner, "planning_deadline"):
            planner.planning_deadline = time.monotonic() + float(planning_timeout)
        stage_t0 = time.perf_counter()
        wall_t0 = time.time()
        try:
            q_path = planner.generate(q_start, goal_q)
        except Exception as exc:
            if "timeout" not in str(exc).lower():
                raise
            planning_error = str(exc)
            q_path = []
        finally:
            if hasattr(planner, "planning_deadline"):
                planner.planning_deadline = None
        result["elapsed"] = time.time() - wall_t0
        result["timing"]["planning"] = time.perf_counter() - stage_t0

        returned_reaches_goal = bool(getattr(planner, "last_returned_path_reaches_goal", True))
        planner_status = getattr(planner, "last_planning_status", None)
        if planning_error is not None:
            forced_collision_preview = True
            fallback_reason = "planner_timeout_no_tree_path"
        elif not returned_reaches_goal:
            forced_collision_preview = True
            fallback_reason = str(planner_status or "planner_latest_branch")
        if not q_path:
            q_path = [q_start]
            forced_collision_preview = True
            fallback_reason = fallback_reason or "planner_empty_start_only"

        stage_t0 = time.perf_counter()
        verification = planner.verify_path(q_path)
        result["timing"]["collision_verification"] = time.perf_counter() - stage_t0
        collision_preview_reason = fallback_reason
        if verification.get("colliding_edges", 0) != 0 or verification.get("colliding_waypoints", 0) != 0:
            forced_collision_preview = True
            collision_preview_reason = collision_preview_reason or "returned_path_collision"

        result.update({
            "status": "partial" if (result.get("ik_fallback") or forced_collision_preview) else "success",
            "q_path": [np.asarray(q, dtype=float) for q in q_path],
            "edge_collisions": verification.get("edge_collisions", []),
            "waypoints": len(q_path),
            "verification": verification,
            "robot_links_considered": True,
            "collision_preview": forced_collision_preview,
            "planning_error": planning_error,
            "fallback_reason": fallback_reason,
            "collision_preview_reason": collision_preview_reason,
            "reached_T": result.get("ik_reached_T"),
        })
        return result
