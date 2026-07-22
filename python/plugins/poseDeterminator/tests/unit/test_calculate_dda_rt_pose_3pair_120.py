"""신규 calculate_DDA_RT_pose_for_taking_xray_3pair_120 단위 테스트.

배경:
    deep-interview + ralplan consensus를 통해 추가된 신규 메서드의 알고리즘
    동작 (3-쌍 120° 박스 제약, 2-쌍 폴백, 키 명명 정책, 슬롯 메타 필드,
    결정성)을 데이터 의존 없이 검증한다.

전략:
    - 실 mesh/scan 데이터에 의존하지 않도록 `__check_collision`과
      `__process_dda_rt_combination`을 monkeypatch로 대체.
    - `__check_collision` mock은 호출 순서 카운터 기반 — 신규 메서드의 main
      loop가 정확히 num_candidates번 순차 호출하므로 안정적.
    - `__process_dda_rt_combination` mock은 dda_pose를 그대로 반영한 슬롯
      반환 (현실적인 형태). collision은 위에서 이미 필터되었으므로 항상 valid.

데이터 의존:
    PIPE NO.3 픽스처를 사용하는 성능 테스트는 데이터 부재 시 자동 skip.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

# conftest.py가 sys.path에 plugin_pdm 루트를 추가함.
from python.plugins.poseDeterminator.EndEffectorPoseOptimizer import EndEffectorPoseOptimizer  # noqa: WPS433


# 인스턴스 멤버 mangled names (테스트에서 private 접근용)
_PIPE_DIRECTION = "_EndEffectorPoseOptimizer__pipe_direction"
_PIPE_CENTER = "_EndEffectorPoseOptimizer__pipe_center"
_PIPE_RADIUS = "_EndEffectorPoseOptimizer__pipe_radius"
_DDA_MESH = "_EndEffectorPoseOptimizer__dda_mesh"
_DDA_INV = "_EndEffectorPoseOptimizer__dda_invers_transform_mat"
_RT_MESH = "_EndEffectorPoseOptimizer__rt_mesh"
_RT_INV = "_EndEffectorPoseOptimizer__rt_invers_transform_mat"
_CHECK_COLLISION = "_EndEffectorPoseOptimizer__check_collision"
_PROCESS_COMBO = "_EndEffectorPoseOptimizer__process_dda_rt_combination"


# ============================================================================
# 헬퍼 및 픽스처
# ============================================================================


def _make_mock_optimizer():
    """real mesh/scan 없이 새 알고리즘만 검증할 수 있는 최소 옵티마이저.

    pipe profile만 직접 설정하면 `__calculate_dda_pose_candidate`가 동작한다.
    mesh/inverse는 mock된 함수들로 전달만 되므로 None이어도 무방.
    """
    opt = EndEffectorPoseOptimizer(debug_mode=False)
    # 단순 직배관 프로파일: Y축 방향, 원점 중심, 반지름 0.05m
    setattr(opt, _PIPE_DIRECTION, np.array([0.0, 1.0, 0.0]))
    setattr(opt, _PIPE_CENTER, np.array([0.0, 0.0, 0.0]))
    setattr(opt, _PIPE_RADIUS, 0.05)
    # mesh/inverse는 mock된 함수에만 전달되므로 placeholder.
    setattr(opt, _DDA_MESH, object())
    setattr(opt, _DDA_INV, np.eye(4))
    setattr(opt, _RT_MESH, object())
    setattr(opt, _RT_INV, np.eye(4))
    return opt


def _make_collision_check(collision_indices: set[int]):
    """호출 순서 카운터 기반 collision check fake.

    신규 메서드의 main loop는 후보 순서대로 정확히 한 번씩 호출한다.
    `__process_dda_rt_combination`은 별도 mock되므로 내부 collision 호출은
    이 fake에 도달하지 않는다.
    """
    counter = {"i": 0}

    def fake_check(mesh, pose, T):
        idx = counter["i"]
        counter["i"] += 1
        return idx in collision_indices

    return fake_check


def _fake_process_combo(dda_pose, angle_of_rt, distance_from_dda_to_rt):
    """RT1/RT2 모두 valid한 슬롯을 반환하는 mock."""
    return {
        "DDA": dda_pose.tolist(),
        "RT1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "RT2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


_KWARGS = dict(
    target_point=(0.05, 0.0, 0.0),
    distance_from_dda_to_surface=0.01,
    distance_from_dda_to_rt=0.3,
    angle_of_rt=10.0,
)


@pytest.fixture
def mock_optimizer(monkeypatch):
    """모든 시나리오의 공통 베이스: process_dda_rt_combination을 valid mock으로 고정."""
    opt = _make_mock_optimizer()
    monkeypatch.setattr(opt, _PROCESS_COMBO, _fake_process_combo)
    return opt


# ============================================================================
# Scenario 1: 정상 3-쌍 (모든 후보 valid)
# ============================================================================


def test_3pair_normal(monkeypatch, mock_optimizer):
    """충돌 없음 → 가장 균등한 3-쌍 (0°, 120°, 240°) 반환."""
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(set()))

    json_str, pose_groups = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(
        **_KWARGS,
    )

    assert isinstance(json_str, str)
    assert len(pose_groups) == 1
    assert set(pose_groups[0].keys()) == {"0", "120", "240"}
    # 모든 후보 valid이므로 가장 균등한 (0, 40, 80) 인덱스 선택 → 실측 (0°, 120°, 240°)
    assert pose_groups[0]["0"]["_actual_deg"] == 0
    assert pose_groups[0]["120"]["_actual_deg"] == 120
    assert pose_groups[0]["240"]["_actual_deg"] == 240


# ============================================================================
# Scenario 2: 일부 충돌 후 3-쌍
# ============================================================================


def test_3pair_with_partial_collisions(monkeypatch, mock_optimizer):
    """일부 후보가 충돌해도 다른 시작점으로 3-쌍 형성 가능.

    plan Step 7 시나리오 (b)의 명시 패턴: collision = {5, 17, 88} (3개 흩어진 충돌).
    충돌 인덱스가 결과에 절대 포함되지 않음을 적극적으로 assertion.
    """
    collision_indices = {5, 17, 88}
    # 충돌 인덱스의 실측 각도 (step_deg=3.0)
    collision_actual_degs = {idx * 3 for idx in collision_indices}  # {15, 51, 264}
    monkeypatch.setattr(
        mock_optimizer, _CHECK_COLLISION, _make_collision_check(collision_indices)
    )

    json_str, pose_groups = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(
        **_KWARGS,
    )

    assert len(pose_groups) == 1
    assert set(pose_groups[0].keys()) == {"0", "120", "240"}
    # 적극적 검증: 충돌 인덱스(15°, 51°, 264°)가 결과 슬롯에 포함되지 않음
    selected_actual_degs = {slot["_actual_deg"] for slot in pose_groups[0].values()}
    assert collision_actual_degs.isdisjoint(selected_actual_degs), (
        f"collision indices leaked into result: "
        f"{collision_actual_degs & selected_actual_degs}"
    )


def test_lexicographic_tiebreak_single_collision(monkeypatch, mock_optimizer):
    """인덱스 0 단일 충돌 → 사전순 최소 (1, 41, 81) 삼중쌍 선택 (편차 0).

    Code-reviewer M1 권고 lock-in: 모든 (i, i+40, i+80) for i=1..39가 편차 0
    동률이므로 사전순 최소 (1, 41, 81)가 결정적으로 선택되어야 함.
    실측 각도: 1*3=3, 41*3=123, 81*3=243.
    """
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check({0}))
    _, pose_groups = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(**_KWARGS)
    assert pose_groups[0]["0"]["_actual_deg"] == 3
    assert pose_groups[0]["120"]["_actual_deg"] == 123
    assert pose_groups[0]["240"]["_actual_deg"] == 243


def test_fallback_long_arc_selected(monkeypatch, mock_optimizer):
    """폴백에서 valid가 (0, 80)일 때 긴 호(240°→ 박스 밖)가 아닌 짧은 호(=120°) 채택 검증.

    Code-reviewer M2 권고: (0, 80) 갭은 80*3=240°. abs(240-120)=120>10이므로 첫 분기
    실패, other_deg=120 → 두 번째 분기 진입 → _arc_deg=120 (긴 호의 반대편 짧은 호).
    """
    n = 120
    collision = set(range(n)) - {0, 80}
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(collision))
    _, pose_groups = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(**_KWARGS)

    assert set(pose_groups[0].keys()) == {"0", "120"}
    assert pose_groups[0]["0"]["_arc_deg"] == 120
    assert pose_groups[0]["120"]["_arc_deg"] == 120
    # 실측 각도는 인덱스 그대로 (0, 80 → 0°, 240°)
    assert pose_groups[0]["0"]["_actual_deg"] == 0
    assert pose_groups[0]["120"]["_actual_deg"] == 240


# ============================================================================
# Scenario 3: 3-쌍 불가 → 2-쌍 폴백
# ============================================================================


def test_fallback_to_2pair(monkeypatch, mock_optimizer):
    """3-쌍이 불가능할 때 2-쌍 폴백 (간격 ≈ 120°)."""
    # 인덱스 0과 40만 valid (나머지 모두 충돌) → 3-쌍 불가, 2-쌍은 가능
    n = 120
    collision = set(range(n)) - {0, 40}
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(collision))

    json_str, pose_groups = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(
        **_KWARGS,
    )

    assert len(pose_groups) == 1
    assert set(pose_groups[0].keys()) == {"0", "120"}
    # Step 5의 if/elif는 짧은 호를 우선 선택 → arc_deg = 120 (절대 240 아님)
    assert pose_groups[0]["0"]["_arc_deg"] == 120
    assert pose_groups[0]["120"]["_arc_deg"] == 120
    assert pose_groups[0]["0"]["_actual_deg"] == 0
    assert pose_groups[0]["120"]["_actual_deg"] == 120


# ============================================================================
# Scenario 4: 폴백 비활성 → 빈 결과
# ============================================================================


def test_no_fallback_returns_empty(monkeypatch, mock_optimizer):
    """allow_2pair_fallback=False이고 3-쌍 0개 → 빈 결과."""
    n = 120
    collision = set(range(n)) - {0, 40}
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(collision))

    json_str, pose_groups = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(
        **_KWARGS,
        allow_2pair_fallback=False,
    )

    assert json_str == "[]"
    assert pose_groups == []


# ============================================================================
# Scenario 5: 모두 불가 → 빈 결과
# ============================================================================


def test_both_impossible_returns_empty(monkeypatch, mock_optimizer):
    """모든 후보 충돌 → 3-쌍·2-쌍 모두 불가, 빈 결과."""
    n = 120
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(set(range(n))))

    json_str, pose_groups = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(
        **_KWARGS,
    )

    assert json_str == "[]"
    assert pose_groups == []


# ============================================================================
# Scenario 6: 결정성 (같은 입력 2회 → 같은 출력)
# ============================================================================


def test_deterministic_selection(monkeypatch, mock_optimizer):
    """동일 collision map 2회 호출 → 결과 동일."""
    collision = {5, 17, 88}

    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(collision))
    json_1, groups_1 = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(**_KWARGS)

    # 두 번째 호출 위해 카운터 리셋 (새 fake 인스턴스)
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(collision))
    json_2, groups_2 = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(**_KWARGS)

    assert json_1 == json_2
    assert groups_1 == groups_2


# ============================================================================
# Scenario 7: 키 명명 정책 lock-in
# ============================================================================


def test_key_naming_lockin_3pair(monkeypatch, mock_optimizer):
    """3-쌍 결과 키가 정확히 {"0","120","240"}."""
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(set()))
    _, pose_groups = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(**_KWARGS)
    assert set(pose_groups[0].keys()) == {"0", "120", "240"}
    # 실측 양자화 각도는 그룹 키가 아닌 슬롯의 _actual_deg에만 노출
    for key in ("0", "120", "240"):
        assert "_actual_deg" in pose_groups[0][key]


def test_key_naming_lockin_fallback(monkeypatch, mock_optimizer):
    """폴백 결과 키가 정확히 {"0","120"}."""
    n = 120
    collision = set(range(n)) - {0, 40}
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(collision))
    _, pose_groups = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(**_KWARGS)
    assert set(pose_groups[0].keys()) == {"0", "120"}


# ============================================================================
# Scenario 8: 슬롯 메타 필드 존재
# ============================================================================


def test_slot_metadata_fields_present_3pair(monkeypatch, mock_optimizer):
    """3-쌍: 모든 슬롯에 _actual_deg int. 그룹 레벨에 underscore 키 없음."""
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(set()))
    _, pose_groups = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(**_KWARGS)

    group = pose_groups[0]
    # 그룹 레벨 키에 underscore-시작 없음
    for key in group.keys():
        assert not key.startswith("_"), f"group key {key!r} should not start with _"
    # 각 슬롯에 _actual_deg int, [0, 360) 범위
    for slot in group.values():
        assert "_actual_deg" in slot
        assert isinstance(slot["_actual_deg"], int)
        assert 0 <= slot["_actual_deg"] < 360


def test_slot_metadata_fields_present_fallback(monkeypatch, mock_optimizer):
    """폴백: 모든 슬롯에 _actual_deg와 _arc_deg int 동시 존재."""
    n = 120
    collision = set(range(n)) - {0, 40}
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(collision))
    _, pose_groups = mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(**_KWARGS)

    for slot in pose_groups[0].values():
        assert "_actual_deg" in slot
        assert "_arc_deg" in slot
        assert isinstance(slot["_actual_deg"], int)
        assert isinstance(slot["_arc_deg"], int)
        assert 0 <= slot["_actual_deg"] < 360
        assert 0 <= slot["_arc_deg"] < 360


# ============================================================================
# 입력 검증 (Security review L-2: NaN/inf/zero/negative/out-of-range 거부)
# ============================================================================


@pytest.mark.parametrize(
    "kwargs_override",
    [
        {"distance_from_dda_to_surface": float("nan")},
        {"distance_from_dda_to_surface": float("inf")},
        {"distance_from_dda_to_rt": float("-inf")},
        {"angle_of_rt": float("nan")},
        {"candidate_step_deg": float("nan")},
        {"candidate_step_deg": 0.0},
        {"candidate_step_deg": -3.0},
        {"gap_tolerance_deg": float("inf")},
        {"gap_tolerance_deg": -1.0},
        {"gap_tolerance_deg": 60.0},   # 박스 비배타성 한계 (배제)
        {"gap_tolerance_deg": 120.0},  # spec 범위 밖
    ],
)
def test_input_validation_rejects_degenerate_floats(monkeypatch, mock_optimizer, kwargs_override):
    """NaN/inf/0/음수/범위 초과 매개변수는 ValueError 발생."""
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(set()))
    kwargs = {**_KWARGS, **kwargs_override}
    with pytest.raises(ValueError):
        mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(**kwargs)


def test_input_validation_rejects_bad_target_point(monkeypatch, mock_optimizer):
    """target_point가 (3,) shape이 아니거나 NaN 포함 시 ValueError."""
    monkeypatch.setattr(mock_optimizer, _CHECK_COLLISION, _make_collision_check(set()))

    # 잘못된 shape
    kwargs = {**_KWARGS, "target_point": (0.0, 0.0)}  # 2D
    with pytest.raises(ValueError):
        mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(**kwargs)

    # NaN 포함
    kwargs = {**_KWARGS, "target_point": (0.0, float("nan"), 0.0)}
    with pytest.raises(ValueError):
        mock_optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(**kwargs)


# ============================================================================
# Scenario 9: 성능 게이트 (PR gate, 데이터 의존)
# ============================================================================


PLUGIN_PDM_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PLUGIN_PDM_ROOT / "data"
ROBOT_MODELS_DIR = DATA_DIR / "robot_models v1.5" / "robots"
SCAN_FILE = DATA_DIR / "PIPE NO.3_fill.ply"
TESTPOINT_FILE = DATA_DIR / "PIPE NO.3_fill_testpoint.csv"
DDA_URDF = ROBOT_MODELS_DIR / "rb10_1300e_DDA.urdf"
RT_URDF = ROBOT_MODELS_DIR / "rb10_1300e_RT.urdf"


def test_performance_pipe_no3():
    """PR gate: PIPE NO.3 + 1 검사 포인트로 단일 호출 wall-time ≤ 60s.

    soft warning if elapsed > 30s. 기존 90° 함수도 함께 측정해 배율 로깅.
    데이터/pandas 부재 시 skip.
    """
    missing = [str(p) for p in (SCAN_FILE, TESTPOINT_FILE, DDA_URDF, RT_URDF) if not p.exists()]
    if missing:
        pytest.skip(f"필요 데이터 파일 부재: {missing}")
    pd = pytest.importorskip("pandas", reason="pandas 미설치")

    import time

    optimizer = EndEffectorPoseOptimizer(debug_mode=False)
    optimizer.load_scan_data(str(SCAN_FILE))
    df = pd.read_csv(TESTPOINT_FILE).fillna(0)
    detection_points = [
        (float(row["x"] + row["dx"]), float(row["y"] + row["dy"]), float(row["z"] + row["dz"]))
        for _, row in df.iterrows()
    ]
    inspection_points = [
        (float(row["x"]), float(row["y"]), float(row["z"])) for _, row in df.iterrows()
    ]
    optimizer.load_DDA_from_urdf(str(DDA_URDF))
    optimizer.load_RT_from_urdf(str(RT_URDF))
    optimizer.calculate_pipe_profile(detection_points[0])

    # 신규 함수 측정
    start = time.perf_counter()
    _, new_groups = optimizer.calculate_DDA_RT_pose_for_taking_xray_3pair_120(
        target_point=inspection_points[0],
        distance_from_dda_to_surface=0.01,
        distance_from_dda_to_rt=0.3,
        angle_of_rt=10.0,
    )
    new_elapsed = time.perf_counter() - start

    # 기존 90° 함수 측정 (게이트 대상 아님, 로그용)
    start = time.perf_counter()
    old_groups = optimizer.calculate_DDA_RT_pose_for_taking_xray(
        target_point=inspection_points[0],
        num_candidates=8,
        distance_from_dda_to_surface=0.01,
        distance_from_dda_to_rt=0.3,
        angle_of_rt=10.0,
    )
    old_elapsed = time.perf_counter() - start

    ratio = new_elapsed / old_elapsed if old_elapsed > 0 else float("inf")
    print(
        f"\n[perf] new_3pair: {new_elapsed:.2f}s | old_90deg: {old_elapsed:.2f}s | "
        f"ratio: {ratio:.1f}x | new_groups: {len(new_groups)} | old_groups: {len(old_groups)}",
    )

    if new_elapsed > 30.0:
        warnings.warn(
            f"performance approaching threshold: {new_elapsed:.1f}s (soft limit 30s)",
            UserWarning,
        )

    # Hard PR gate
    assert new_elapsed <= 60.0, (
        f"3-pair function exceeded 60s gate: {new_elapsed:.1f}s. "
        f"Apply caching or reduce candidate_step_deg."
    )
