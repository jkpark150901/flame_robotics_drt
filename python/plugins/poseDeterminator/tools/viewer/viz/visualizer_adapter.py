"""``EndEffectorPoseOptimizer`` 비공개 멤버에 접근하는 박막 + JupyterVisualizer
호출 헬퍼.

알고리즘 코드를 수정하지 않고 viewer 가 필요한 메시·변환 행렬을 읽기 전용으로
참조하기 위한 단방향 어댑터다. 비공개(name-mangled) 속성을 직접 접근하므로
클래스 내부 구조 변경에 민감 — 알고리즘 측 PR 시 본 모듈을 함께 점검할 것.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


# name-mangled 속성 이름. EndEffectorPoseOptimizer 의 멤버.
_DDA_MESH_ATTR = "_EndEffectorPoseOptimizer__dda_mesh"
_RT_MESH_ATTR = "_EndEffectorPoseOptimizer__rt_mesh"
_DDA_INV_T_ATTR = "_EndEffectorPoseOptimizer__dda_invers_transform_mat"
_RT_INV_T_ATTR = "_EndEffectorPoseOptimizer__rt_invers_transform_mat"


def get_dda_mesh(optimizer: Any):
    return getattr(optimizer, _DDA_MESH_ATTR)


def get_rt_mesh(optimizer: Any):
    return getattr(optimizer, _RT_MESH_ATTR)


def get_dda_inv_transform(optimizer: Any) -> np.ndarray:
    return getattr(optimizer, _DDA_INV_T_ATTR)


def get_rt_inv_transform(optimizer: Any) -> np.ndarray:
    return getattr(optimizer, _RT_INV_T_ATTR)


def pose_to_homogeneous(pose: list[float] | tuple | np.ndarray) -> np.ndarray:
    """``[x, y, z, roll, pitch, yaw]`` → 4×4 동차변환 행렬."""

    arr = np.asarray(pose, dtype=float)
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", arr[3:6]).as_matrix()
    T[:3, 3] = arr[:3]
    return T


def link_transform_for_tcp_pose(
    tcp_pose: list[float] | tuple | np.ndarray,
    tcp_to_link_inv_T: np.ndarray,
) -> np.ndarray:
    """TCP 자세 + (tcp→link 의 역변환 행렬) → link 메시에 적용할 4×4 변환."""

    tcp_T = pose_to_homogeneous(tcp_pose)
    return tcp_T @ tcp_to_link_inv_T
