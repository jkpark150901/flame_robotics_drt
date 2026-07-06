"""tools/viewer.viz.visualizer_adapter 단위 테스트.

URDF 가 필요 없는 순수 변환 함수 (``pose_to_homogeneous``,
``link_transform_for_tcp_pose``) 의 동작을 검증한다. ``get_*`` 헬퍼는
``EndEffectorPoseOptimizer`` 가 URDF 를 로드한 뒤에 의미가 있으므로
data 디렉토리의 URDF 가 존재할 때만 smoke-check 한다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from python.plugins.poseDeterminator.tools.viewer import app_paths
from python.plugins.poseDeterminator.tools.viewer.viz.visualizer_adapter import (
    get_dda_inv_transform,
    get_dda_mesh,
    link_transform_for_tcp_pose,
    pose_to_homogeneous,
)


def test_pose_to_homogeneous_identity_for_zero_pose() -> None:
    T = pose_to_homogeneous([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)


def test_pose_to_homogeneous_translation_only() -> None:
    T = pose_to_homogeneous([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0], atol=1e-12)


def test_pose_to_homogeneous_yaw_90_rotates_x_to_y() -> None:
    T = pose_to_homogeneous([0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2])
    # x 축 [1,0,0] → y 축 [0,1,0]
    rotated = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


def test_link_transform_for_tcp_pose_composes_correctly() -> None:
    tcp_pose = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # 단순 X 평행이동
    inv_T = np.eye(4)
    inv_T[0, 3] = -0.1  # link 가 tcp 보다 -0.1 만큼 X 음방향
    T = link_transform_for_tcp_pose(tcp_pose, inv_T)
    # 결과: link 의 절대 X 위치 = 0.5 - 0.1 = 0.4
    assert T[0, 3] == pytest.approx(0.4)


def _try_load_dda_optimizer():
    """URDF 로드 시도. 데이터 부재 시 skip.

    한글 경로 환경의 Open3D 빈 mesh 회귀는 ``conftest.py`` 가
    ``_open3d_compat.apply()`` 를 호출해 trimesh fallback 으로 처리한다.
    """

    if not app_paths.dda_urdf_path().is_file():
        pytest.skip("DDA URDF not available")

    from python.plugins.poseDeterminator.EndEffectorPoseOptimizer import EndEffectorPoseOptimizer  # type: ignore

    opt = EndEffectorPoseOptimizer(debug_mode=False)
    opt.load_DDA_from_urdf(str(app_paths.dda_urdf_path()))
    return opt


def test_get_dda_inv_transform_after_load_urdf_smoke() -> None:
    opt = _try_load_dda_optimizer()
    T = get_dda_inv_transform(opt)
    assert isinstance(T, np.ndarray)
    assert T.shape == (4, 4)


def test_get_dda_mesh_returns_open3d_mesh_smoke() -> None:
    import open3d as o3d  # type: ignore

    opt = _try_load_dda_optimizer()
    mesh = get_dda_mesh(opt)
    assert isinstance(mesh, o3d.geometry.TriangleMesh)
    assert len(mesh.vertices) > 0
