"""PyBullet 기반 빠른 IK solver.

pink(QP)/직접 구현 DLS와 달리, PyBullet의 `calculateInverseKinematics`는 C++로 구현된
Levenberg-Marquardt 계열 solver라서 같은 반복 횟수 기준으로 훨씬 빠르다. 여기서는 그
결과 q 하나만 계산해서 돌려주고, 실제 수렴 판정/오차 계산/궤적 기록은 기존처럼
pinocchio_backend.py의 Pinocchio FK(`record()`)로 한다 - 두 백엔드의 오차 계산 방식이
달라서 생기는 불일치를 막기 위함이다.

URDF는 로봇당 한 번만 로드해서 캐시한다(매 IK 호출마다 새로 로드하면 느려짐 - 이전에
Pinocchio 쪽에서도 같은 이유로 캐싱을 추가했던 것과 동일한 이유).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pybullet as pb
except ImportError:  # pragma: no cover - depends on runtime environment
    pb = None


_CLIENT_ID: Optional[int] = None
_BODIES: Dict[Tuple[str, ...], Dict[str, Any]] = {}


def _ensure_client() -> int:
    global _CLIENT_ID
    if pb is None:
        raise RuntimeError(
            "PyBullet IK backend is not available. Install it with: pip install pybullet"
        )
    if _CLIENT_ID is None:
        # DIRECT: GUI 없는 headless 물리 클라이언트. 렌더링/충돌 계산에는 안 쓰고
        # IK(자코비안 기반 뉴턴 스텝)만 여기서 돌린다.
        _CLIENT_ID = pb.connect(pb.DIRECT)
    return _CLIENT_ID


def _quat_from_R(R: np.ndarray) -> List[float]:
    """3x3 회전행렬을 pybullet이 쓰는 [x, y, z, w] 쿼터니언으로 변환한다."""
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def _get_body(urdf_path: str, base_T: np.ndarray, joint_names: Sequence[str]) -> Dict[str, Any]:
    key = (str(urdf_path), tuple(np.asarray(base_T, dtype=float).round(6).flatten().tolist()))
    cached = _BODIES.get(key)
    if cached is not None:
        return cached

    client = _ensure_client()
    base_T = np.asarray(base_T, dtype=float)
    body_id = pb.loadURDF(
        str(urdf_path),
        basePosition=base_T[:3, 3].tolist(),
        baseOrientation=_quat_from_R(base_T[:3, :3]),
        useFixedBase=True,
        physicsClientId=client,
    )

    movable_indices: List[int] = []
    joint_index_by_name: Dict[str, int] = {}
    link_index_by_name: Dict[str, int] = {}
    for i in range(pb.getNumJoints(body_id, physicsClientId=client)):
        info = pb.getJointInfo(body_id, i, physicsClientId=client)
        joint_type = info[2]
        joint_name = info[1].decode("utf-8")
        link_name = info[12].decode("utf-8")
        link_index_by_name[link_name] = i
        if joint_type != pb.JOINT_FIXED:
            joint_index_by_name[joint_name] = i
            movable_indices.append(i)

    # 우리 joint_names(raw q 순서)를 pybullet joint index로 매핑한다. 이름이 URDF와
    # 다르면(연결이 안 되면) 매핑에서 빠지고 그 관절은 IK 대상에서 제외된다.
    order = [joint_index_by_name[name] for name in joint_names if name in joint_index_by_name]

    cached = {
        "client": client,
        "body_id": body_id,
        "movable_indices": movable_indices,
        "joint_index_by_name": joint_index_by_name,
        "link_index_by_name": link_index_by_name,
        "order": order,
    }
    _BODIES[key] = cached
    return cached


def solve(
    urdf_path: str,
    base_T: np.ndarray,
    joint_names: Sequence[str],
    q_init: Sequence[float],
    target_world_T: np.ndarray,
    frame_name: str,
    max_iter: int = 100,
    tol: float = 1e-4,
    damping: float = 1e-3,
) -> np.ndarray:
    """target_world_T까지의 IK를 PyBullet으로 한 번에 풀어 최종 q(우리 joint_names 순서)를 반환한다."""
    body = _get_body(urdf_path, base_T, joint_names)
    client = body["client"]
    body_id = body["body_id"]
    order = body["order"]
    link_index_by_name = body["link_index_by_name"]

    if frame_name not in link_index_by_name:
        raise RuntimeError(f"pybullet IK: link '{frame_name}' not found in URDF '{urdf_path}'")
    end_effector_link = link_index_by_name[frame_name]

    q_init = np.asarray(q_init, dtype=float)
    for joint_idx, value in zip(order, q_init):
        pb.resetJointState(body_id, joint_idx, float(value), physicsClientId=client)

    target_world_T = np.asarray(target_world_T, dtype=float)
    target_pos = target_world_T[:3, 3].tolist()
    target_orn = _quat_from_R(target_world_T[:3, :3])

    joint_damping = [float(damping)] * len(order)
    result = pb.calculateInverseKinematics(
        body_id,
        end_effector_link,
        target_pos,
        target_orn,
        jointDamping=joint_damping,
        maxNumIterations=int(max_iter),
        residualThreshold=float(tol),
        physicsClientId=client,
    )
    # calculateInverseKinematics는 body의 "movable joint 전체" 순서로 반환한다.
    result_by_index = dict(zip(body["movable_indices"], result))
    q_out = q_init.copy()
    for i, joint_idx in enumerate(order):
        if joint_idx in result_by_index:
            q_out[i] = float(result_by_index[joint_idx])
    return q_out
