"""검사 포인트 일괄 자세 계산 워커.

메인 스레드 GUI 와 ``EndEffectorPoseOptimizer`` 사이의 단방향 격리 레이어.
포인트별 ``calculate_pipe_profile`` + 자세 계산 함수 호출 결과를
``InspectionResult`` 로 deepcopy 한 뒤 시그널로 emit 한다.
``EndEffectorPoseOptimizer`` 의 두 자세 계산 함수 모두 매 호출 시
``self.debuging_info = {}`` 로 초기화하므로 emit 직전 deepcopy 가 필수다.

``mode`` 가 ``OptimizationMode.TWO_PAIR_90`` (기본) 이면 기존 90°·2쌍
``calculate_DDA_RT_pose_for_taking_xray`` 를, ``THREE_PAIR_120`` 이면 신규
``calculate_DDA_RT_pose_for_taking_xray_3pair_120`` 을 호출한다. 두 함수의
반환 시그니처는 ``(json_str, pose_groups)`` 로 동일하며, viewer 의
``scene_builder._KEY_TO_COLOR`` 가 양쪽 키 집합 모두를 매핑한다.

호출 인자 정책 (notebook ``demo(...).ipynb`` 셀 ``d15784d4`` / ``08f981f0`` 와
동일):
  - ``calculate_pipe_profile`` ← ``position_with_offset`` (= detection_point;
    검사 대상 좌표를 외벽 표면으로 옮긴 위치)
  - 자세 계산 함수 ← ``position`` (= inspection_point; 원본 검사 좌표).
    두 함수에 같은 좌표를 넘기면 자세 후보가 모두 충돌 필터링되어 실패한다.
"""

from __future__ import annotations

import copy
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from ..models import InspectionPoint, InspectionResult, OptimizationMode


class BatchWorker(QThread):
    point_done = pyqtSignal(object)          # InspectionResult
    progress = pyqtSignal(int, int)          # (completed, total)
    batch_finished = pyqtSignal()
    batch_aborted = pyqtSignal(str)          # reason

    DEFAULT_NUM_CANDIDATES = 8
    DEFAULT_DISTANCE_FROM_DDA_TO_SURFACE = 0.01
    DEFAULT_DISTANCE_FROM_DDA_TO_RT = 0.3
    DEFAULT_ANGLE_OF_RT = 10.0
    DEFAULT_CANDIDATE_STEP_DEG = 3.0
    DEFAULT_GAP_TOLERANCE_DEG = 10.0
    DEFAULT_MODE = OptimizationMode.TWO_PAIR_90

    def __init__(
        self,
        optimizer: Any,
        points: list[InspectionPoint],
        parent=None,
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        distance_from_dda_to_surface: float = DEFAULT_DISTANCE_FROM_DDA_TO_SURFACE,
        distance_from_dda_to_rt: float = DEFAULT_DISTANCE_FROM_DDA_TO_RT,
        angle_of_rt: float = DEFAULT_ANGLE_OF_RT,
        mode: OptimizationMode = DEFAULT_MODE,
        candidate_step_deg: float = DEFAULT_CANDIDATE_STEP_DEG,
        gap_tolerance_deg: float = DEFAULT_GAP_TOLERANCE_DEG,
        allow_2pair_fallback: bool = True,
    ) -> None:
        super().__init__(parent)
        self._optimizer = optimizer
        self._points = list(points)
        self._num_candidates = num_candidates
        self._distance_from_dda_to_surface = distance_from_dda_to_surface
        self._distance_from_dda_to_rt = distance_from_dda_to_rt
        self._angle_of_rt = angle_of_rt
        self._mode = mode
        self._candidate_step_deg = candidate_step_deg
        self._gap_tolerance_deg = gap_tolerance_deg
        self._allow_2pair_fallback = allow_2pair_fallback

    def _invoke_optimizer(self, target_point):
        """현재 모드에 해당하는 optimizer 메서드를 호출하고 pose_groups 만 반환.

        ``TWO_PAIR_90``: 기존 ``calculate_DDA_RT_pose_for_taking_xray``
        ``THREE_PAIR_120``: 신규 ``calculate_DDA_RT_pose_for_taking_xray_3pair_120``
        두 함수의 반환 시그니처는 ``(json_str, pose_groups)`` 로 동일하다.
        """

        if self._mode is OptimizationMode.THREE_PAIR_120:
            _, pose_groups = self._optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(
                target_point=target_point,
                distance_from_dda_to_surface=self._distance_from_dda_to_surface,
                distance_from_dda_to_rt=self._distance_from_dda_to_rt,
                angle_of_rt=self._angle_of_rt,
                candidate_step_deg=self._candidate_step_deg,
                gap_tolerance_deg=self._gap_tolerance_deg,
                allow_2pair_fallback=self._allow_2pair_fallback,
            )
            return pose_groups

        # 기본/명시: 2-쌍 90°
        _, pose_groups = self._optimizer.calculate_DDA_RT_pose_for_taking_xray(
            target_point=target_point,
            num_candidates=self._num_candidates,
            distance_from_dda_to_surface=self._distance_from_dda_to_surface,
            distance_from_dda_to_rt=self._distance_from_dda_to_rt,
            angle_of_rt=self._angle_of_rt,
        )
        return pose_groups

    def run(self) -> None:
        total = len(self._points)
        for idx, point in enumerate(self._points):
            # 협조적 종료 — 매 포인트 시작 직전 1회 검사 (계산 중에는 atomic)
            if self.isInterruptionRequested():
                self.batch_aborted.emit("user requested cancel")
                return

            try:
                self._optimizer.calculate_pipe_profile(point.position_with_offset)
                pose_groups = self._invoke_optimizer(point.position)
                pose_groups_copy = copy.deepcopy(pose_groups)
                debug_snapshot = copy.deepcopy(getattr(self._optimizer, "debuging_info", None))
                result = InspectionResult(
                    point_index=point.index,
                    success=len(pose_groups_copy) >= 1,
                    pose_groups=pose_groups_copy,
                    debuging_info=debug_snapshot,
                    error_message=None,
                    mode=self._mode,
                )
            except RuntimeError as e:
                # 도메인 예외 (예: target_point 주변에 점군 없음 — EEPOpt:527)
                result = InspectionResult(
                    point_index=point.index,
                    success=False,
                    pose_groups=[],
                    debuging_info=None,
                    error_message=str(e),
                    mode=self._mode,
                )
            except Exception as e:  # noqa: BLE001
                # 예상 외 — 일괄 중단
                self.batch_aborted.emit(f"unexpected error at point {point.index}: {e}")
                return

            self.point_done.emit(result)
            self.progress.emit(idx + 1, total)

        self.batch_finished.emit()
