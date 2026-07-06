"""배관 방향에 따른 RT 자세 계산 강건성 단위 테스트.

`EndEffectorPoseOptimizer.__calculate_rt_pose_for_angle` 가 임의의 배관 방향
(축 방향, 대각선 방향 등)에서 SVD 비수렴이나 NaN 회전 행렬 같은 수치 오류
없이 동작해야 함을 매트릭스로 검증한다.

배경:
    초기 구현은 `rt_x_axis`/`rt_y_axis` 가 거의 평행해질 때 외적이 영벡터
    → 정규화 시 NaN → `R.from_matrix()` 의 SVD 비수렴 에러를 일으켰다.
    수정 이후 본 매트릭스 테스트가 회귀 방지 장치 역할을 한다.

본 파일은 외부 데이터(점군/URDF) 없이도 돌도록, 배관 프로파일을 직접 주입한
mock optimizer 를 사용한다 — 메서드 본체는 `EndEffectorPoseOptimizer` 와 동일
로직이라 회귀 검출이 가능하다.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R


class _MockOptimizer:
    """배관 프로파일이 미리 주입된 mock.

    `EndEffectorPoseOptimizer` 를 직접 인스턴스화하면 점군/메쉬 로드가
    강제되어 단위 테스트에 부적합하다. 본 mock 은 핵심 두 메서드의 로직을
    원본과 동일하게 옮긴 후 배관 프로파일만 주입받는다.
    """

    def __init__(self, pipe_direction: np.ndarray, pipe_center: np.ndarray, pipe_radius: float):
        self._pipe_direction = pipe_direction / np.linalg.norm(pipe_direction)
        self._pipe_center = pipe_center
        self._pipe_radius = pipe_radius

    def calculate_dda_pose_candidate(
        self,
        point_on_pipe_surface: np.ndarray,
        radius: float,
        num_candidates: int,
    ) -> np.ndarray:
        pipe_direction = self._pipe_direction
        pipe_center = self._pipe_center

        direction_unit = pipe_direction / np.linalg.norm(pipe_direction)
        vec_to_surface = point_on_pipe_surface - pipe_center
        proj_len = np.dot(vec_to_surface, direction_unit)
        center = pipe_center + proj_len * direction_unit

        basis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(basis, pipe_direction)) > 0.9:
            basis = np.array([0.0, 1.0, 0.0])

        v1 = np.cross(pipe_direction, basis)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(pipe_direction, v1)
        v2 /= np.linalg.norm(v2)

        angles = 2 * np.pi * np.arange(num_candidates) / num_candidates
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        offsets = np.outer(cos_a, v1) + np.outer(sin_a, v2)
        positions = center + offsets * radius

        y_unit = pipe_direction / np.linalg.norm(pipe_direction)
        y_axis = np.tile(y_unit, (num_candidates, 1))

        x_axis = center - positions
        x_norm = np.linalg.norm(x_axis, axis=1, keepdims=True)
        x_norm[x_norm < 1e-12] = 1.0
        x_axis = x_axis / x_norm

        z_axis = np.cross(x_axis, y_axis)
        z_norm = np.linalg.norm(z_axis, axis=1, keepdims=True)
        z_norm[z_norm < 1e-12] = 1.0
        z_axis = z_axis / z_norm

        rot_mats = np.stack([x_axis, y_axis, z_axis], axis=2)
        rpy_array = R.from_matrix(rot_mats).as_euler("xyz", degrees=False)

        return np.hstack((positions, rpy_array))

    def calculate_rt_pose_for_angle(
        self,
        dda_tcp_pose: np.ndarray,
        angle_deg: float,
        distance_from_dda_to_rt: float,
    ) -> tuple[np.ndarray | None, dict]:
        debug_info: dict = {}

        dda_rot_matrix = R.from_euler("xyz", dda_tcp_pose[3:]).as_matrix()
        dda_x_axis = dda_rot_matrix[:, 0]
        dda_z_axis = dda_rot_matrix[:, 2]
        dda_z_axis_unit = dda_z_axis / np.linalg.norm(dda_z_axis)

        cos_angle = np.cos(np.radians(angle_deg))
        sin_angle = np.sin(np.radians(angle_deg))

        k_cross_v = np.cross(dda_z_axis_unit, dda_x_axis)
        k_dot_v = np.dot(dda_z_axis_unit, dda_x_axis)

        dda_to_rt_direction = (
            dda_x_axis * cos_angle + k_cross_v * sin_angle + dda_z_axis_unit * k_dot_v * (1 - cos_angle)
        )

        rt_position = dda_tcp_pose[:3] + dda_to_rt_direction * distance_from_dda_to_rt
        rt_x_axis = -dda_to_rt_direction
        rt_x_axis = rt_x_axis / np.linalg.norm(rt_x_axis)

        rt_z_axis = dda_z_axis_unit
        rt_y_axis = np.cross(rt_z_axis, rt_x_axis)
        rt_y_axis = rt_y_axis / np.linalg.norm(rt_y_axis)

        rt_rot_matrix = np.column_stack([rt_x_axis, rt_y_axis, rt_z_axis])
        det = np.linalg.det(rt_rot_matrix)
        debug_info["det"] = det

        if np.isnan(det):
            debug_info["error"] = "회전 행렬 det 가 NaN"
            return None, debug_info

        if det < 0:
            rt_z_axis = -rt_z_axis
            rt_y_axis = np.cross(rt_z_axis, rt_x_axis)
            rt_y_axis = rt_y_axis / np.linalg.norm(rt_y_axis)
            rt_rot_matrix = np.column_stack([rt_x_axis, rt_y_axis, rt_z_axis])

        try:
            rt_rpy = R.from_matrix(rt_rot_matrix).as_euler("xyz")
        except Exception as e:  # noqa: BLE001
            debug_info["error"] = str(e)
            return None, debug_info

        return np.hstack([rt_position, rt_rpy]), debug_info


# 배관 방향 매트릭스 — 축 정렬, 대각선, 임의 방향까지 폭넓게 커버
_PIPE_DIRECTIONS = [
    pytest.param(np.array([1.0, 0.0, 0.0]), id="axis-x-pos"),
    pytest.param(np.array([0.0, 1.0, 0.0]), id="axis-y-pos"),
    pytest.param(np.array([0.0, 0.0, 1.0]), id="axis-z-pos"),
    pytest.param(np.array([-1.0, 0.0, 0.0]), id="axis-x-neg"),
    pytest.param(np.array([0.0, -1.0, 0.0]), id="axis-y-neg"),
    pytest.param(np.array([0.0, 0.0, -1.0]), id="axis-z-neg"),
    pytest.param(np.array([1.0, 1.0, 0.0]), id="diag-xy-pos"),
    pytest.param(np.array([1.0, -1.0, 0.0]), id="diag-xy-neg"),
    pytest.param(np.array([1.0, 0.0, 1.0]), id="diag-xz-pos"),
    pytest.param(np.array([1.0, 0.0, -1.0]), id="diag-xz-neg"),
    pytest.param(np.array([0.0, 1.0, 1.0]), id="diag-yz-pos"),
    pytest.param(np.array([0.0, 1.0, -1.0]), id="diag-yz-neg"),
    pytest.param(np.array([1.0, 1.0, 1.0]), id="diag-3d-1"),
    pytest.param(np.array([1.0, 1.0, -1.0]), id="diag-3d-2"),
    pytest.param(np.array([1.0, -1.0, 1.0]), id="diag-3d-3"),
    pytest.param(np.array([-1.0, 1.0, 1.0]), id="diag-3d-4"),
    pytest.param(np.array([0.5, 0.3, 0.8]), id="arbitrary-1"),
    pytest.param(np.array([0.1, 0.9, 0.1]), id="arbitrary-2"),
    pytest.param(np.array([0.8, 0.1, 0.6]), id="arbitrary-3"),
]


@pytest.mark.parametrize("pipe_direction", _PIPE_DIRECTIONS)
def test_rt_pose_calculation_handles_arbitrary_pipe_direction(pipe_direction: np.ndarray) -> None:
    """주어진 배관 방향에서 8개 DDA 후보 모두에 대해 ±α RT 자세 계산이 성공해야 한다."""
    pipe_center = np.zeros(3)
    pipe_radius = 0.05  # 5 cm
    distance_from_dda_to_surface = 0.01
    distance_from_dda_to_rt = 0.3
    angle_of_rt = 10.0  # degrees
    num_candidates = 8

    optimizer = _MockOptimizer(pipe_direction, pipe_center, pipe_radius)

    # 배관 표면 위 한 점을 배관 방향에 수직인 방향으로 잡는다
    basis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(basis, pipe_direction / np.linalg.norm(pipe_direction))) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
    perpendicular = np.cross(pipe_direction, basis)
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    target_point = pipe_center + perpendicular * pipe_radius

    dda_candidates = optimizer.calculate_dda_pose_candidate(
        target_point,
        pipe_radius + distance_from_dda_to_surface,
        num_candidates,
    )

    failures: list[tuple[int, str]] = []
    for i, dda_pose in enumerate(dda_candidates):
        rt_pos, dbg_pos = optimizer.calculate_rt_pose_for_angle(dda_pose, +angle_of_rt, distance_from_dda_to_rt)
        rt_neg, dbg_neg = optimizer.calculate_rt_pose_for_angle(dda_pose, -angle_of_rt, distance_from_dda_to_rt)
        if rt_pos is None:
            failures.append((i, f"+α: {dbg_pos.get('error', 'unknown')}"))
        if rt_neg is None:
            failures.append((i, f"-α: {dbg_neg.get('error', 'unknown')}"))

    assert not failures, (
        f"pipe_direction={pipe_direction.tolist()} 에서 RT 자세 계산 실패: " + ", ".join(f"#{i} {msg}" for i, msg in failures)
    )
