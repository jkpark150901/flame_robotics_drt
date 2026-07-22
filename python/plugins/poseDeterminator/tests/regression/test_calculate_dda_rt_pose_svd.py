"""한화오션 PIPE NO.3 데이터에서 SVD 비수렴 에러 회귀 테스트.

배경:
    검사 인덱스 1 에서 ``calculate_DDA_RT_pose_for_taking_xray`` 호출 시
    ``numpy.linalg.LinAlgError: SVD did not converge`` 가 발생했었다.

원인:
    ``__calculate_rt_pose_for_angle`` 에서 ``rt_x_axis``/``rt_y_axis`` 가 거의
    평행해질 때 외적이 영벡터 → 정규화 시 NaN → 회전 행렬 SVD 비수렴.

해결:
    Z축 우선·X×Z 외적 후 부호 보정 로직으로 변경. 본 회귀 테스트는 그 변경의
    효과가 유지되는지 PIPE NO.3 데이터에서 검증한다.

데이터 의존:
    필요한 파일이 없으면 자동 skip — ``data/`` 디렉토리는 .gitignore 되어
    개발자 단말에서 별도 관리한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PLUGIN_PDM_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PLUGIN_PDM_ROOT / "data"
ROBOT_MODELS_DIR = DATA_DIR / "robot_models v1.5" / "robots"

PIPE_NAME = "PIPE NO.3_fill"
SCAN_FILE = DATA_DIR / f"{PIPE_NAME}.ply"
TESTPOINT_FILE = DATA_DIR / f"{PIPE_NAME}_testpoint.csv"
DDA_URDF = ROBOT_MODELS_DIR / "rb10_1300e_DDA.urdf"
RT_URDF = ROBOT_MODELS_DIR / "rb10_1300e_RT.urdf"


@pytest.fixture(scope="module")
def pipe_no_3_optimizer():
    """PIPE NO.3 스캔/URDF 로 초기화한 EndEffectorPoseOptimizer 와 검사 좌표.

    데이터 또는 의존 패키지(pandas)가 없으면 skip.
    """
    missing = [str(p) for p in (SCAN_FILE, TESTPOINT_FILE, DDA_URDF, RT_URDF) if not p.exists()]
    if missing:
        pytest.skip(f"필요 데이터 파일 부재: {missing}")

    pd = pytest.importorskip("pandas", reason="pandas 미설치")
    from EndEffectorPoseOptimizer import EndEffectorPoseOptimizer  # noqa: WPS433 — sys.path 는 conftest 가 추가

    optimizer = EndEffectorPoseOptimizer(debug_mode=False)
    optimizer.load_scan_data(str(SCAN_FILE))

    df = pd.read_csv(TESTPOINT_FILE).fillna(0)
    detection_points = [
        (float(row["x"] + row["dx"]), float(row["y"] + row["dy"]), float(row["z"] + row["dz"]))
        for _, row in df.iterrows()
    ]
    inspection_points = [
        (float(row["x"]), float(row["y"]), float(row["z"]))
        for _, row in df.iterrows()
    ]

    optimizer.load_DDA_from_urdf(str(DDA_URDF))
    optimizer.load_RT_from_urdf(str(RT_URDF))

    return optimizer, detection_points, inspection_points


def test_inspection_point_index_1_does_not_raise(pipe_no_3_optimizer) -> None:
    """노트북 1-based 인덱스 1(=리스트 0번) 검사 포인트에서 raise 없이 결과 반환."""
    optimizer, detection_points, inspection_points = pipe_no_3_optimizer
    inspection_index = 0  # 노트북 1-based → 0-based

    optimizer.calculate_pipe_profile(detection_points[inspection_index])

    target_groups = optimizer.calculate_DDA_RT_pose_for_taking_xray(
        inspection_points[inspection_index],
        num_candidates=8,
        distance_from_dda_to_surface=0.01,
        distance_from_dda_to_rt=0.3,
        angle_of_rt=10,
    )

    assert isinstance(target_groups, list)
    # 결과가 0개여도 raise 만 없으면 회귀 방지 목표는 달성. 추가 정합성 검증은
    # 단위 테스트 영역. 만약 0개라면 collision_pose_groups 까지 검토 필요(디버그 모드).
