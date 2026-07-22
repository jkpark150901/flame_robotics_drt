"""tools/viewer.workers.batch_worker 단위 테스트.

알고리즘 자체는 호출하지 않고 ``EndEffectorPoseOptimizer`` 를 mock 한 뒤
``BatchWorker`` 의 시그널 발화·예외 처리·협조적 종료를 검증한다.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from python.plugins.poseDeterminator.tools.viewer.models import InspectionPoint, InspectionResult, OptimizationMode
from python.plugins.poseDeterminator.tools.viewer.workers.batch_worker import BatchWorker


@pytest.fixture()
def points() -> list[InspectionPoint]:
    return [
        InspectionPoint(index=1, position=(0.0, 0.0, 0.0), offset=(0.0, 0.0, 0.0)),
        InspectionPoint(index=2, position=(0.1, 0.0, 0.0), offset=(0.0, 0.0, 0.0)),
        InspectionPoint(index=3, position=(0.2, 0.0, 0.0), offset=(0.0, 0.0, 0.0)),
    ]


class _FakeOptimizer:
    """필요 메서드만 흉내. 호출 시퀀스 기록.

    ``per_call_delay`` 를 주면 매 자세 계산 호출 시작 시 ``time.sleep`` 을
    수행해 실제 알고리즘의 수십초 지연을 축소 모방한다 — 메인 스레드 슬롯이
    emit 사이에 처리될 시간을 확보.

    두 자세 계산 함수 (``calculate_DDA_RT_pose_for_taking_xray`` 와
    ``calculate_DDA_RT_pose_for_taking_xray_3pair_120``) 모두를 흉내내며,
    어느 쪽이 호출됐는지를 ``calls`` 의 첫 원소로 표시 ("xray" vs "xray_3p120").
    """

    def __init__(self, scenarios: list[Any], per_call_delay: float = 0.0) -> None:
        # 각 호출마다 반환할 (json_str, pose_groups) 또는 던질 예외
        self._scenarios = scenarios
        self._call_index = 0
        self._per_call_delay = per_call_delay
        self.calls: list[tuple] = []
        self.debuging_info: dict = {}

    def calculate_pipe_profile(self, target_point, *args, **kwargs) -> None:
        self.calls.append(("profile", tuple(target_point)))

    def _next_scenario(self):
        scenario = self._scenarios[self._call_index]
        self._call_index += 1
        # 매 호출마다 debuging_info 가 초기화되는 실제 동작 모방 (deepcopy 검증용)
        self.debuging_info = {"scenario_index": self._call_index}
        if isinstance(scenario, Exception):
            raise scenario
        return scenario  # (json_str, pose_groups)

    def calculate_DDA_RT_pose_for_taking_xray(self, **kwargs):  # noqa: D401
        if self._per_call_delay > 0:
            time.sleep(self._per_call_delay)
        self.calls.append(("xray", tuple(kwargs["target_point"])))
        scenario = self._next_scenario()
        if isinstance(scenario, tuple) and len(scenario) == 2:
            return scenario[1]
        return scenario

    def calculate_DDA_RT_pose_for_taking_xray_3pair_120(self, **kwargs):  # noqa: D401
        if self._per_call_delay > 0:
            time.sleep(self._per_call_delay)
        self.calls.append(("xray_3p120", tuple(kwargs["target_point"])))
        return self._next_scenario()


def test_emits_point_done_per_point_in_order(qtbot, points) -> None:
    pose_groups_a = [{"0": {"DDA": [0, 0, 0, 0, 0, 0]}, "90": {"DDA": [0, 0, 0, 0, 0, 0]}}]
    pose_groups_b: list[dict] = []
    pose_groups_c = [
        {"0": {"DDA": [1, 0, 0, 0, 0, 0]}, "90": {"DDA": [1, 0, 0, 0, 0, 0]}},
        {"0": {"DDA": [2, 0, 0, 0, 0, 0]}, "90": {"DDA": [2, 0, 0, 0, 0, 0]}},
    ]
    fake = _FakeOptimizer(
        scenarios=[
            ("[]", pose_groups_a),
            ("[]", pose_groups_b),
            ("[]", pose_groups_c),
        ]
    )
    worker = BatchWorker(fake, points)
    received: list[InspectionResult] = []
    worker.point_done.connect(received.append)

    with qtbot.waitSignal(worker.batch_finished, timeout=5000):
        worker.start()

    assert [r.point_index for r in received] == [1, 2, 3]
    assert received[0].success is True
    assert received[1].success is False  # pose_groups 비어있음
    assert received[2].success is True
    # deepcopy 검증: 결과의 debuging_info 가 다음 호출 mutation 영향 없음
    assert received[0].debuging_info == {"scenario_index": 1}
    assert received[1].debuging_info == {"scenario_index": 2}
    assert received[2].debuging_info == {"scenario_index": 3}
    # pose_groups 도 deepcopy 되어 호출자가 mutate 해도 fake 의 원본 영향 없도록
    received[2].pose_groups.append({"sentinel": True})
    assert pose_groups_c[-1] != {"sentinel": True}


def test_runtime_error_emits_failure_result_and_continues(qtbot, points) -> None:
    fake = _FakeOptimizer(
        scenarios=[
            RuntimeError("target_point 주변에 점군이 없습니다 (synthetic)"),
            ("[]", [{"0": {"DDA": [0, 0, 0, 0, 0, 0]}, "90": {"DDA": [0, 0, 0, 0, 0, 0]}}]),
            ("[]", []),
        ]
    )
    worker = BatchWorker(fake, points)
    received: list[InspectionResult] = []
    worker.point_done.connect(received.append)

    with qtbot.waitSignal(worker.batch_finished, timeout=5000):
        worker.start()

    assert len(received) == 3
    assert received[0].success is False
    assert "synthetic" in (received[0].error_message or "")
    assert received[1].success is True
    assert received[2].success is False
    assert received[2].error_message is None  # 알고리즘 결과 0개 — 도메인 예외 아님


def test_unexpected_exception_emits_batch_aborted(qtbot, points) -> None:
    fake = _FakeOptimizer(
        scenarios=[
            ("[]", []),
            ValueError("synthetic non-domain"),  # 두 번째 포인트에서 RuntimeError 가 아닌 예외
            ("[]", []),
        ]
    )
    worker = BatchWorker(fake, points)
    aborted_reasons: list[str] = []
    worker.batch_aborted.connect(aborted_reasons.append)

    with qtbot.waitSignal(worker.batch_aborted, timeout=5000):
        worker.start()
    worker.wait()  # 명시적 join

    assert any("synthetic non-domain" in r for r in aborted_reasons)


# ============================================================================
# 모드 분기 (TWO_PAIR_90 vs THREE_PAIR_120)
# ============================================================================


def test_default_mode_dispatches_to_2pair_90(qtbot, points) -> None:
    """mode 인자 생략 시 기존 90° 함수를 호출 (회귀 호환)."""

    fake = _FakeOptimizer(
        scenarios=[("[]", []), ("[]", []), ("[]", [])],
    )
    worker = BatchWorker(fake, points)
    with qtbot.waitSignal(worker.batch_finished, timeout=5000):
        worker.start()
    xray_calls = [c for c in fake.calls if c[0] == "xray"]
    xray_3p120_calls = [c for c in fake.calls if c[0] == "xray_3p120"]
    assert len(xray_calls) == len(points)
    assert len(xray_3p120_calls) == 0


def test_three_pair_120_mode_dispatches_to_new_function(qtbot, points) -> None:
    """mode=THREE_PAIR_120 → 신규 ``..._3pair_120`` 함수만 호출."""

    fake = _FakeOptimizer(
        scenarios=[("[]", []), ("[]", []), ("[]", [])],
    )
    worker = BatchWorker(fake, points, mode=OptimizationMode.THREE_PAIR_120)
    with qtbot.waitSignal(worker.batch_finished, timeout=5000):
        worker.start()
    xray_calls = [c for c in fake.calls if c[0] == "xray"]
    xray_3p120_calls = [c for c in fake.calls if c[0] == "xray_3p120"]
    assert len(xray_calls) == 0
    assert len(xray_3p120_calls) == len(points)


def test_result_carries_mode_through_to_emit(qtbot, points) -> None:
    """emit 된 InspectionResult.mode 가 worker 의 mode 와 일치."""

    fake = _FakeOptimizer(
        scenarios=[("[]", [{"0": {"DDA": [0, 0, 0, 0, 0, 0]}}])] * len(points),
    )
    worker = BatchWorker(fake, points, mode=OptimizationMode.THREE_PAIR_120)
    received: list[InspectionResult] = []
    worker.point_done.connect(received.append)
    with qtbot.waitSignal(worker.batch_finished, timeout=5000):
        worker.start()
    assert all(r.mode is OptimizationMode.THREE_PAIR_120 for r in received)


def test_three_pair_120_runtime_error_propagates_as_failure_result(qtbot, points) -> None:
    """3쌍 모드에서 발생하는 RuntimeError 도 도메인 예외로 처리 (회귀 호환 경로)."""

    fake = _FakeOptimizer(
        scenarios=[
            RuntimeError("target_point 주변에 점군이 없습니다 (synthetic 3p120)"),
            ("[]", [{"0": {"DDA": [0, 0, 0, 0, 0, 0]}}]),
            ("[]", []),
        ]
    )
    worker = BatchWorker(fake, points, mode=OptimizationMode.THREE_PAIR_120)
    received: list[InspectionResult] = []
    worker.point_done.connect(received.append)
    with qtbot.waitSignal(worker.batch_finished, timeout=5000):
        worker.start()

    assert len(received) == 3
    assert received[0].success is False
    assert "synthetic 3p120" in (received[0].error_message or "")
    assert received[0].mode is OptimizationMode.THREE_PAIR_120


def test_request_interruption_aborts_before_next_point(qtbot, points) -> None:
    """첫 포인트 emit 직후 interrupt → 두 번째 포인트 시작 전 batch_aborted."""

    fake = _FakeOptimizer(
        scenarios=[
            ("[]", [{"0": {"DDA": [0, 0, 0, 0, 0, 0]}, "90": {"DDA": [0, 0, 0, 0, 0, 0]}}]),
            ("[]", [{"0": {"DDA": [1, 0, 0, 0, 0, 0]}, "90": {"DDA": [1, 0, 0, 0, 0, 0]}}]),
            ("[]", [{"0": {"DDA": [2, 0, 0, 0, 0, 0]}, "90": {"DDA": [2, 0, 0, 0, 0, 0]}}]),
        ],
        per_call_delay=0.15,  # 메인 스레드 슬롯이 첫 emit 처리 + interrupt 요청할 시간 확보
    )
    worker = BatchWorker(fake, points)

    received: list[InspectionResult] = []

    def _on_point_done(result: InspectionResult) -> None:
        received.append(result)
        if len(received) == 1:
            worker.requestInterruption()

    worker.point_done.connect(_on_point_done)

    with qtbot.waitSignal(worker.batch_aborted, timeout=5000):
        worker.start()
    worker.wait()

    # 첫 포인트는 반드시 emit, 그 후 interrupt 가 적용되어 batch_aborted.
    # 워커는 포인트 시작 *전* 검사하므로 interrupt 요청 타이밍에 따라 1~2개까지 emit 될 수 있음.
    assert len(received) >= 1
    assert len(received) < len(points)
    assert received[0].point_index == 1
