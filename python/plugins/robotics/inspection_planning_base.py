from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from plugins.robotics.backend import IKOptions, IKResult, RoboticsBackend


@dataclass
class ApproachSpec:
    """DDA처럼 표면에 수직으로만 진입해야 하는 end effector의 접근 제약.

    경로는 [retreat] + [free] + [approach] 세 구간으로 만들어진다.
        approach: 사전접근점(pre) -> 목표. target frame의 facing 축 방향 직선(카르테시안)이라
                  접근 방향이 구조적으로 표면 수직이다. 여기서만 표면 근처(clearance 미만) 허용.
        free:     (retreat 끝점) -> pre. 일반 q-space planner/optimizer 구간. 이 구간에서는
                  clearance_links가 표면에서 min_clearance 안으로 못 들어간다(충돌로 판정).
        retreat:  시작 자세가 이미 표면 근처(이전 target의 진입 자세를 이어받은 경우)면 그 자세의
                  facing 축 반대로 물러나는 직선 구간. 아니면 비어 있다.

    Args:
        facing_axis_local: target frame 기준, end effector가 표면을 바라보는 축(단위 벡터).
        standoff: 사전접근점을 목표에서 facing 반대로 얼마나 띄울지(m).
        clearance_links: 표면 근처 통과를 막을 링크(geometry) 이름. 예: ["dda_link_end"].
        min_clearance: free 구간에서 clearance_links가 지켜야 할 최소 표면 거리(m).
            standoff보다 작아야 pre/retreat 끝점이 이 조건을 만족한다.
        step: 직선 구간 IK 보간 간격(m).
        max_joint_step: 인접 보간점 사이 허용 q 변화 크기. 넘으면 IK 해 가지가 바뀐 것(elbow flip 등)으로 본다.
    """

    facing_axis_local: Sequence[float]
    standoff: float = 0.15
    clearance_links: Sequence[str] = ()
    min_clearance: float = 0.10
    step: float = 0.02
    max_joint_step: float = 0.3

    def __post_init__(self):
        axis = np.asarray(self.facing_axis_local, dtype=float).reshape(3)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            raise ValueError("ApproachSpec.facing_axis_local must be non-zero")
        self.facing_axis_local = (axis / norm).tolist()
        if not (0.0 < float(self.min_clearance) < float(self.standoff)):
            raise ValueError(
                f"ApproachSpec requires 0 < min_clearance < standoff, got "
                f"min_clearance={self.min_clearance}, standoff={self.standoff}")
        if not self.clearance_links:
            raise ValueError("ApproachSpec.clearance_links is required")


class ApproachPlanningError(RuntimeError):
    """진입/퇴출 직선 구간을 만들 수 없을 때(IK 실패, 해 불연속 등). reason은 collision_preview_reason으로 노출된다."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


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


#: Resolution (rad, q-space) used for the FINAL accept/reject verify_path()
#: call - deliberately finer than planning's usual step_size (0.08 default).
#: Search-time edge checking (OMPL's state validity checker during RRT*/RRT-
#: Connect exploration, STOMP's own cost evaluation) stays at the coarser
#: step_size for speed; only the one verification pass that actually decides
#: collision_preview/status uses this. Found in practice: an RRTConnect+STOMP
#: result verify_path() called clean at 0.08 resolution (0 colliding edges)
#: turned out to have an actual, if brief (2 samples out of 255), penetration
#: once independently re-checked at 0.02 - a collision "notch" narrower than
#: one 0.08 sample interval, invisible to that check by construction. This is
#: the fix: verify at a resolution fine enough that such a notch can't hide
#: between two sampled points.
FINAL_VERIFICATION_RESOLUTION = 0.02


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
            solver 이름을 먼저 결정한다. 나머지 수치 파라미터는 config 기본값으로 채운다.
        """
        solver_name = str(ik_solver or ik_config.get("solver", "pybullet") or "pybullet").lower()
        normalize_value = bool(ik_normalize) if ik_normalize is not None else False
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
        reached_T       = self.backend.frame_world_T(request.robot_name, q, request.frame_name)
        target_world_T  = np.asarray(target_world_T, dtype=float)
        position_error  = float(np.linalg.norm(reached_T[:3, 3] - target_world_T[:3, 3]))
        rot_delta = reached_T[:3, :3].T @ target_world_T[:3, :3]
        cos_angle = (float(np.trace(rot_delta)) - 1.0) * 0.5
        orientation_error = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        collision = False
        collision_pairs: list = []
        collision_check_error = None
        try:
            collision_result        = self.backend.check_collision(request.robot_name, q, return_pairs=True)
            collision               = bool(collision_result.collision)
            collision_pairs         = [list(pair) for pair in collision_result.pairs]
            if collision:
                pair_text = ", ".join(f"{a} <-> {b}" for a, b in collision_result.pairs)
                print(f"ik_result_summary: collision for {request.robot_name}: {pair_text}")
        except Exception as exc:
            # 여기서 조용히 넘기면 collision check가 실제로 실패했는데도 "collision=False"로
            # 보고돼, planner가 나중에 같은 q를 충돌로 판정해도 IK 단계 결과와 왜 다른지
            # 알 수 없게 된다. 실패 사유를 그대로 남긴다.
            collision_check_error = str(exc)
            print(
                f"ik_result_summary: collision check failed for {request.robot_name}, "
                f"treating as unknown (not collision-free): {exc}"
            )
        return {
            "success":              bool(ik_result.success),
            "fallback":             bool(fallback),
            "position_error":       position_error,
            "orientation_error":    orientation_error,
            "collision":            collision,
            "collision_check_error": collision_check_error,
            "collision_pairs":      collision_pairs,
            "collision_pair_count": len(collision_pairs),
            "iterations":           int(ik_result.iterations),
            "elapsed":              float(ik_result.elapsed),
            "max_iter":             int(request.ik_config.get("max_iter")),
            "solver":               str(ik_result.solver),
            "normalize":            bool(ik_result.normalize),
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
        start_q  = np.asarray(request.start_q, dtype=float)
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

    @staticmethod
    def _linear_track_indices(joint_names) -> list:
        """joint 이름 목록에서 linear track(prismatic 레일) joint의 인덱스를 찾는다."""
        from plugins.robotics.inspection_workflow import linear_track_indices
        return linear_track_indices(joint_names)

    def _pin_joint_values(self, robot_name: str, fixed_values: Dict[int, float]):
        """지정한 joint 인덱스를 고정하도록 로봇 모델의 position limit을 임시로 좁힌다.

        [v, v+eps]로 좁히면 backend.sample_configuration이 그 joint를 항상 v 근처로만
        뽑고(레일이 planning 중 안 움직임), joint_limits_for_metric은 span이 1e-9 미만이라
        1.0으로 보정돼(거리 metric에서 이 joint 기여가 ~0) 정규화도 안전하다. lo==hi로
        두면 `hi <= lo` invalid 처리로 [-pi, pi]로 리셋돼 고정이 풀리므로 아주 작은
        eps를 둔다. 로봇마다 별도 model이라 다른 로봇 병렬 planning에 영향을 주지 않는다.

        Returns:
            원래 limit으로 되돌리는 restore 콜러블. 모델 접근 실패 시 None.
        """
        try:
            handle = self.backend.robot_handle(robot_name)
            model = handle.model
        except Exception:
            return None
        try:
            orig_lo = np.asarray(model.lowerPositionLimit, dtype=float).copy()
            orig_hi = np.asarray(model.upperPositionLimit, dtype=float).copy()
        except Exception:
            return None
        lo = orig_lo.copy()
        hi = orig_hi.copy()
        eps = 1e-11
        for i, v in fixed_values.items():
            if 0 <= i < lo.shape[0]:
                lo[i] = float(v)
                hi[i] = float(v) + eps
        model.lowerPositionLimit = lo
        model.upperPositionLimit = hi

        def restore():
            model.lowerPositionLimit = orig_lo
            model.upperPositionLimit = orig_hi

        return restore

    # ------------------------------------------------------------------
    # 표면 수직 진입(ApproachSpec) 지원
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def link_clearance(self, robot_name: str, link_names: Optional[Sequence[str]], margin: float):
        """with 블록 안에서 link_names가 장애물 margin(m) 안으로 들어가면 충돌로 판정한다.

        블록이 끝나면(예외 포함) 반드시 0으로 되돌린다 - backend의 collision data는 호출 사이에
        재사용되므로 남겨 두면 다음 target의 계획이 조용히 오염된다. margin이 적용된 pair가
        하나도 없으면(backend 미지원 등) 제약이 실제로는 안 걸린 것이므로 조용히 넘기지 않고 예외를 낸다.
        """
        enabled = bool(link_names) and float(margin or 0.0) > 0.0
        try:
            if enabled:
                applied = int(self.backend.set_link_clearance(robot_name, link_names, margin) or 0)
                if applied <= 0:
                    raise ApproachPlanningError("clearance_not_enforced")
            yield
        finally:
            if enabled:
                with contextlib.suppress(Exception):
                    self.backend.set_link_clearance(robot_name, [], 0.0)

    def _clear_link_clearance(self, robot_name: str) -> None:
        with contextlib.suppress(Exception):
            self.backend.set_link_clearance(robot_name, [], 0.0)

    @contextlib.contextmanager
    def tight_verification_resolution(self, planner, resolution: float = FINAL_VERIFICATION_RESOLUTION):
        """Temporarily tighten planner's collision-sampling resolution for one
        final verify_path()/verify_approach_path() call - see
        FINAL_VERIFICATION_RESOLUTION for why. Restores the planner's own
        (usually step_size-derived) resolution afterward regardless of
        exceptions, since it's shared/reused across the next target's plan."""
        original = getattr(planner, "pin_collision_sample_resolution", None)
        if original is None:
            yield
            return
        planner.pin_collision_sample_resolution = float(resolution)
        try:
            yield
        finally:
            planner.pin_collision_sample_resolution = original

    def link_min_distance(self, robot_name: str, q, link_names: Sequence[str]) -> Optional[float]:
        """link_names에 해당하는 링크의 장애물까지 최소 거리. 거리 데이터가 없으면 None."""
        names = [str(n) for n in link_names]
        distances = [
            float(e["distance"])
            for e in self.backend.link_obstacle_distances(robot_name, q)
            if any(e["link"] == n or str(e["link"]).startswith(n + "_") for n in names)
        ]
        return min(distances) if distances else None

    def _linear_ik_segment(
        self,
        request: InspectionIKRequest,
        q_seed: np.ndarray,
        p_end: np.ndarray,
        spec: ApproachSpec,
    ) -> List[np.ndarray]:
        """q_seed의 TCP 자세를 유지한 채 위치만 p_end까지 직선으로 옮기는 q 목록(q_seed 제외, 끝점 포함).

        step 간격마다 직전 해를 seed로 IK를 다시 풀어 카르테시안 직선을 만든다(관절 보간이 아님).
        """
        options = self.ik_options(request.ik_config, request.ik_solver, request.ik_normalize)
        T_seed = self.backend.frame_world_T(request.robot_name, q_seed, request.frame_name)
        p_start = T_seed[:3, 3].copy()
        p_end = np.asarray(p_end, dtype=float).reshape(3)
        n_steps = max(1, int(math.ceil(float(np.linalg.norm(p_end - p_start)) / max(spec.step, 1e-6))))
        q_prev = np.asarray(q_seed, dtype=float)
        segment: List[np.ndarray] = []
        for k in range(1, n_steps + 1):
            T_k = T_seed.copy()
            T_k[:3, 3] = p_start + (p_end - p_start) * (k / n_steps)
            result = self.backend.solve_ik(
                request.robot_name, T_k, q_prev, options=options, frame_name=request.frame_name)
            if result.q is None or not bool(result.success):
                raise ApproachPlanningError("approach_ik_failed")
            q_k = np.asarray(result.q, dtype=float)
            if float(np.linalg.norm(q_k - q_prev)) > spec.max_joint_step:
                raise ApproachPlanningError("approach_ik_discontinuity")
            segment.append(q_k)
            q_prev = q_k
        return segment

    def _build_approach_segments(
        self,
        request: InspectionIKRequest,
        q_start: np.ndarray,
        goal_q: np.ndarray,
        spec: ApproachSpec,
    ) -> Dict[str, Any]:
        """retreat(필요 시) / 사전접근점 / approach 구간의 q 목록을 만든다."""
        robot = request.robot_name
        facing_local = np.asarray(spec.facing_axis_local, dtype=float)

        # approach: goal에서 facing 반대 방향으로 standoff만큼 IK를 밟아 나간 뒤 뒤집는다.
        # goal_q를 seed로 시작하므로 approach 끝점이 IK가 정한 goal_q와 정확히 같다.
        T_goal = self.backend.frame_world_T(robot, goal_q, request.frame_name)
        facing_world = T_goal[:3, :3] @ facing_local
        pre_p = T_goal[:3, 3] - float(spec.standoff) * facing_world
        backward = self._linear_ik_segment(request, goal_q, pre_p, spec)
        pre_q = backward[-1]
        approach_qs = list(reversed(backward)) + [np.asarray(goal_q, dtype=float)]

        # retreat: 시작 자세가 이미 표면 근처면 그 자세의 facing 반대로 먼저 물러난다.
        start_clearance = self.link_min_distance(robot, q_start, spec.clearance_links)
        if start_clearance is None:
            raise ApproachPlanningError("clearance_data_unavailable")
        retreat_qs: List[np.ndarray] = [np.asarray(q_start, dtype=float)]
        if start_clearance < float(spec.min_clearance):
            T_start = self.backend.frame_world_T(robot, q_start, request.frame_name)
            retreat_p = T_start[:3, 3] - float(spec.standoff) * (T_start[:3, :3] @ facing_local)
            retreat_qs += self._linear_ik_segment(request, q_start, retreat_p, spec)

        return {
            "retreat_qs": retreat_qs,
            "pre_q": pre_q,
            "approach_qs": approach_qs,
            "start_clearance": float(start_clearance),
            "facing_world": facing_world.tolist(),
        }

    def _approach_metrics(
        self,
        request: InspectionIKRequest,
        approach_qs: Sequence[np.ndarray],
        spec: ApproachSpec,
    ) -> Dict[str, float]:
        """approach 구간이 실제로 표면 수직 직선인지 FK로 확인한 수치(각도 편차, 직선 이탈)."""
        points = np.array([
            self.backend.frame_world_T(request.robot_name, q, request.frame_name)[:3, 3] for q in approach_qs])
        T_goal = self.backend.frame_world_T(request.robot_name, approach_qs[-1], request.frame_name)
        T_pre = self.backend.frame_world_T(request.robot_name, approach_qs[0], request.frame_name)
        travel = points[-1] - points[0]
        length = float(np.linalg.norm(travel))
        if length < 1e-9:
            return {"length": 0.0, "angle_deg": 0.0, "lateral_dev_max": 0.0, "orientation_dev_deg": 0.0}
        direction = travel / length
        # 실제 이동 방향과 target frame facing 축 사이 각도.
        facing = T_goal[:3, :3] @ np.asarray(spec.facing_axis_local, dtype=float)
        angle = math.degrees(math.acos(float(np.clip(np.dot(direction, facing), -1.0, 1.0))))
        offsets = points - points[0]
        lateral = offsets - np.outer(offsets @ direction, direction)
        # 자세(회전)가 approach 내내 유지되는지.
        rot_delta = T_pre[:3, :3].T @ T_goal[:3, :3]
        orient = math.degrees(math.acos(float(np.clip((np.trace(rot_delta) - 1.0) * 0.5, -1.0, 1.0))))
        return {
            "length": length,
            "angle_deg": angle,
            "lateral_dev_max": float(np.max(np.linalg.norm(lateral, axis=1))),
            "orientation_dev_deg": orient,
        }

    def verify_approach_path(self, planner, q_path: Sequence[Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """[retreat | free | approach] 경로를 구간별 규칙으로 검증한다.

        free 구간만 clearance margin을 켜고(표면 근처 통과 금지), retreat/approach는 실제 접촉만
        충돌로 본다(표면 가까이 가는 게 그 구간의 목적이므로). 반환 dict는 PlannerBase.verify_path와
        같은 키에 segments/margin_only_violation/free_min_clearance를 더한 것이다.
        """
        robot = info["robot_name"]
        links = info["clearance_links"]
        margin = float(info["min_clearance"])
        poses = [np.asarray(q, dtype=float) for q in q_path]
        free_a, free_b = int(info["free_start_idx"]), int(info["free_end_idx"])
        segments = (
            ("retreat", 0, free_a, False),
            ("free", free_a, free_b, True),
            ("approach", free_b, len(poses) - 1, False),
        )
        merged: Dict[str, Any] = {
            "colliding_edges": 0, "colliding_waypoints": 0, "collision_pairs": [],
            "edge_collisions": [], "waypoint_collisions": [], "end_link_colliding": False,
            "backend": None, "segments": {}, "margin_only_violation": False,
        }
        self._clear_link_clearance(robot)
        for name, a, b, with_margin in segments:
            if b < a or not poses:
                continue
            seg = poses[a:b + 1]
            if with_margin:
                with self.link_clearance(robot, links, margin):
                    v = planner.verify_path(seg)
            else:
                v = planner.verify_path(seg)
            hit = bool(v.get("colliding_edges", 0) or v.get("colliding_waypoints", 0))
            if with_margin and hit:
                # 접촉 없이 margin만 어긴 경우인지 구분한다(원인 표기용).
                plain = planner.verify_path(seg)
                merged["margin_only_violation"] = not bool(
                    plain.get("colliding_edges", 0) or plain.get("colliding_waypoints", 0))
            merged["segments"][name] = {"start": a, "end": b, "colliding": hit}
            merged["colliding_edges"] += int(v.get("colliding_edges", 0))
            merged["colliding_waypoints"] += int(v.get("colliding_waypoints", 0))
            merged["end_link_colliding"] = merged["end_link_colliding"] or bool(v.get("end_link_colliding"))
            merged["backend"] = v.get("backend")
            for pair in v.get("collision_pairs", []):
                if list(pair) not in merged["collision_pairs"]:
                    merged["collision_pairs"].append(list(pair))
            for item in v.get("edge_collisions", []):
                shifted = dict(item)
                shifted["edge"] = int(item["edge"]) + a
                shifted["from_waypoint"] = int(item["from_waypoint"]) + a
                shifted["to_waypoint"] = int(item["to_waypoint"]) + a
                merged["edge_collisions"].append(shifted)
            for item in v.get("waypoint_collisions", []):
                shifted = dict(item)
                shifted["waypoint"] = int(item["waypoint"]) + a
                merged["waypoint_collisions"].append(shifted)

        free_distances = [
            d for d in (self.link_min_distance(robot, q, links) for q in poses[free_a:free_b + 1]) if d is not None]
        merged["free_min_clearance"] = min(free_distances) if free_distances else None
        return merged

    def plan_q_path_for_robot(
        self,
        *,
        planner,
        ik_request: InspectionIKRequest,
        q_start: Sequence[float],
        planning_timeout: float = 0.0,
        lock_linear_track: bool = False,
        console=None,
        approach: Optional[ApproachSpec] = None,
    ) -> Dict[str, Any]:
        """IK 목표 q까지 q-space path planning을 수행한다.

        Args:
            planner: PlannerBase 호환 planner. generate와 verify_path를 제공해야 한다.
            ik_request: 목표 pose와 IK 설정.
            q_start: path planning 시작 raw q.
            planning_timeout: planner deadline. 0 이하면 비활성화.
            approach: 주어지면 목표 대신 사전접근점까지 planner로 계획하고, 목표까지는 facing 축
                방향 직선으로 잇는다(ApproachSpec 참고). 결과의 "approach"에 구간 인덱스/수직도 수치가 담긴다.
            lock_linear_track: 기본 False(권장). True면 룰베이스로 linear track을 먼저
                목표값으로 이동시킨 뒤, 그 값에 고정한 상태로 나머지 joint만 path planning한다
                (탐색 공간 축소로 속도↑). 하지만 그 "먼저 이동" 구간은 팔은 이전 자세 그대로
                둔 채 레일만 검사 없이 옮기는 가상의 중간 자세(plan_start_q)로, 팔이 큰 자세
                차이(예: positioner 회전 그룹 안에서 target이 바뀔 때)에서 옮겨가는 도중이면
                이 중간 자세가 positioner 등과 실제로 충돌할 수 있다(start_collision으로 걸림 -
                레일과 팔을 같이 보간하는 일반 경로였다면 안 걸렸을 충돌). False면 track도 다른
                joint와 똑같이 planner가 함께 보간하므로 이 문제가 없다.

        Returns:
            dict: IK check 결과에 q_path, verification, planning timing을 추가한 결과.

        계산 과정:
            1. check_inspection_ik_for_robot으로 목표 q를 구한다.
            2. (룰베이스) linear track을 목표값으로 옮긴 prep q에서 시작하고 track을 고정한다.
            3. planner.generate(plan_start_q, goal_q)를 호출한다.
            4. timeout/empty path/fallback 상태를 collision preview 사유로 기록한다.
            5. track 이동 구간을 앞에 붙이고 planner.verify_path로 전체 path 충돌을 검증한다.
        """
        result = self.check_inspection_ik_for_robot(ik_request)
        q_start = np.asarray(q_start, dtype=float)
        goal_q = np.asarray(result["goal_q"], dtype=float)
        planning_error = None
        forced_collision_preview = bool(result.get("ik_fallback", False))
        # "collision_preview=True인데 reason=None"으로 보이면 원인 파악이 안 된다.
        # ik_fallback 때문에 강제된 경우 그 사실을 바로 사유에 남긴다(아래 planner 단계
        # 분기들이 있으면 그쪽이 더 구체적이라 그대로 덮어쓴다).
        fallback_reason = "ik_fallback" if forced_collision_preview else None

        # 표면 수직 진입: 목표 대신 사전접근점까지만 planner에 맡기고, 목표까지는 직선 approach로 잇는다.
        # ik_fallback이면 goal_q 자체가 못 미더우므로 approach 없이 기존 경로(+fallback 표시)로 둔다.
        approach_segments: Optional[Dict[str, Any]] = None
        approach_info: Optional[Dict[str, Any]] = None
        planner_goal_q = goal_q
        if approach is not None and not forced_collision_preview:
            try:
                approach_segments = self._build_approach_segments(ik_request, q_start, goal_q, approach)
            except ApproachPlanningError as exc:
                return self._approach_failure_result(result, q_start, exc.reason)
            q_start = approach_segments["retreat_qs"][-1]
            planner_goal_q = approach_segments["pre_q"]

        # 룰베이스 linear track 고정: track을 먼저 목표값으로 옮긴 자세(plan_start_q)에서
        # planning을 시작하고, 그 로봇 모델의 track limit을 좁혀 track이 planning 중 안
        # 움직이게 한다. q_start -> plan_start_q(레일 이동) 구간은 나중에 path 앞에 붙인다.
        plan_start_q = q_start
        track_prepended = False
        track_restore = None
        planner_fixed_indices = list(getattr(planner, "fixed_joint_indices", []) or [])
        planner_fixed_values = list(getattr(planner, "fixed_joint_values", []) or [])
        if planner_fixed_indices:
            plan_start_q = q_start.copy()
            fixed_values = {}
            for local_idx, joint_idx in enumerate(planner_fixed_indices):
                joint_idx = int(joint_idx)
                if not (0 <= joint_idx < plan_start_q.shape[0]):
                    continue
                value = None
                if local_idx < len(planner_fixed_values) and planner_fixed_values[local_idx] is not None:
                    try:
                        value = float(planner_fixed_values[local_idx])
                    except Exception:
                        value = None
                if value is None:
                    value = float(q_start[joint_idx])
                plan_start_q[joint_idx] = value
                fixed_values[joint_idx] = value
            track_prepended = not np.allclose(plan_start_q, q_start)
            track_restore = self._pin_joint_values(ik_request.robot_name, fixed_values)
        elif lock_linear_track:
            track_indices = self._linear_track_indices(ik_request.joint_names)
            if track_indices and goal_q.shape[0] == q_start.shape[0]:
                plan_start_q = q_start.copy()
                for i in track_indices:
                    plan_start_q[i] = goal_q[i]
                track_prepended = not np.allclose(plan_start_q, q_start)
                track_restore = self._pin_joint_values(
                    ik_request.robot_name, {i: float(goal_q[i]) for i in track_indices})

        if console is not None:
            ik_result = result.get("ik_result", {}) or {}
            console.debug(
                f"inspection path IK result: robot={ik_request.robot_name}\n"
                f"  ik_success       = {ik_result.get('success')}\n"
                f"  ik_fallback      = {result.get('ik_fallback')}\n"
                f"  position_error   = {ik_result.get('position_error')}\n"
                f"  orientation_error= {ik_result.get('orientation_error')}\n"
                f"  goal_q           = {np.round(goal_q, 5).tolist()}\n"
                f"  plan_start_q     = {np.round(plan_start_q, 5).tolist()} "
                f"(track_prepended={track_prepended})\n"
                f"  plan_start_q==goal_q -> {np.allclose(plan_start_q, goal_q)}")

        if planning_timeout > 0 and hasattr(planner, "planning_deadline"):
            planner.planning_deadline = time.monotonic() + float(planning_timeout)
        stage_t0 = time.perf_counter()
        wall_t0 = time.time()
        # free 구간 계획은 clearance margin을 켠 채로 한다(표면 근처 통과 금지). 블록을 벗어나면 자동 해제.
        clearance_ctx = (
            self.link_clearance(ik_request.robot_name, approach.clearance_links, approach.min_clearance)
            if approach_segments is not None else contextlib.nullcontext())
        approach_error: Optional[str] = None
        try:
            with clearance_ctx:
                q_path = planner.generate(plan_start_q, planner_goal_q)
        except ApproachPlanningError as exc:
            approach_error = exc.reason
            q_path = []
        except Exception as exc:
            if "timeout" not in str(exc).lower():
                raise
            planning_error = str(exc)
            q_path = []
        finally:
            if hasattr(planner, "planning_deadline"):
                planner.planning_deadline = None
            if track_restore is not None:
                track_restore()
        if approach_error is not None:
            return self._approach_failure_result(result, q_start, approach_error)
        # 레일 이동 구간(q_start -> plan_start_q)을 경로 맨 앞에 붙인다. 이후 verify_path가
        # 이 구간도 함께 충돌 검사한다(레일 이동 중 충돌도 잡힘).
        if track_prepended and q_path:
            first = np.asarray(q_path[0], dtype=float)
            if not np.allclose(first, q_start):
                q_path = [q_start] + list(q_path)
        result["elapsed"] = time.time() - wall_t0
        result["timing"]["planning"] = time.perf_counter() - stage_t0
        result["plan_start_q"] = plan_start_q
        result["track_prepended"] = track_prepended

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
        if approach_segments is not None and not forced_collision_preview:
            retreat_qs = approach_segments["retreat_qs"]
            free_qs = [np.asarray(q, dtype=float) for q in q_path]
            free_start_idx = len(retreat_qs) - 1
            q_path = retreat_qs[:-1] + free_qs + approach_segments["approach_qs"][1:]
            approach_info = {
                "enabled": True,
                "assembled": True,
                "robot_name": ik_request.robot_name,
                "clearance_links": list(approach.clearance_links),
                "min_clearance": float(approach.min_clearance),
                "standoff": float(approach.standoff),
                "free_start_idx": free_start_idx,
                "free_end_idx": free_start_idx + len(free_qs) - 1,
                "retreat_edges": free_start_idx,
                "start_clearance": approach_segments["start_clearance"],
                "facing_world": approach_segments["facing_world"],
                **{f"approach_{k}": v for k, v in
                   self._approach_metrics(ik_request, approach_segments["approach_qs"], approach).items()},
            }
            with self.tight_verification_resolution(planner):
                verification = self.verify_approach_path(planner, q_path, approach_info)
            approach_info["free_min_clearance"] = verification.get("free_min_clearance")
            approach_info["margin_only_violation"] = bool(verification.get("margin_only_violation"))
        else:
            if approach is not None:
                approach_info = {"enabled": True, "assembled": False}
            with self.tight_verification_resolution(planner):
                verification = planner.verify_path(q_path)
        result["timing"]["collision_verification"] = time.perf_counter() - stage_t0
        collision_preview_reason = fallback_reason
        if verification.get("colliding_edges", 0) != 0 or verification.get("colliding_waypoints", 0) != 0:
            forced_collision_preview = True
            collision_preview_reason = collision_preview_reason or (
                "dda_min_clearance_violated" if verification.get("margin_only_violation")
                else "returned_path_collision")

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
            # OMPL-backed planners populate this with iterations/solve_time/
            # max_iter/timeout_sec/state_validity_calls/collision_rejects
            # (see OMPLPlannerBase._generate_joint_space); legacy planners
            # leave it empty.
            "planner_stats": dict(getattr(planner, "last_ompl_stats", {}) or {}),
            # approach를 쓰지 않았으면 None. 쓴 경우 구간 인덱스(free_start_idx/free_end_idx)와
            # 수직도/clearance 수치 - visualizer가 옵티마이저를 free 구간에만 적용할 때 쓴다.
            "approach": approach_info,
        })
        return result

    @staticmethod
    def _approach_failure_result(result: Dict[str, Any], q_start: np.ndarray, reason: str) -> Dict[str, Any]:
        """진입/퇴출 직선을 만들지 못했을 때의 결과. q_path는 start 한 점뿐이라 호출부가 실패로 처리한다."""
        result["timing"]["planning"] = 0.0
        result["timing"]["collision_verification"] = 0.0
        result.update({
            "status": "partial",
            "q_path": [np.asarray(q_start, dtype=float)],
            "edge_collisions": [],
            "waypoints": 1,
            "verification": {"colliding_edges": 0, "colliding_waypoints": 0, "collision_pairs": [],
                             "edge_collisions": [], "waypoint_collisions": []},
            "robot_links_considered": True,
            "collision_preview": True,
            "planning_error": None,
            "fallback_reason": reason,
            "collision_preview_reason": reason,
            "reached_T": result.get("ik_reached_T"),
            "elapsed": 0.0,
            "planner_stats": {},
            "approach": {"enabled": True, "assembled": False, "error": reason},
        })
        return result
