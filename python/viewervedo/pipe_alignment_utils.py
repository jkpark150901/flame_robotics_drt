"""Pipe alignment helpers used by the Vedo viewer.

이 모듈은 배관 프로파일 축/기준점과 chuck 축/원점을 맞추는 순수 계산을 담당한다.
viewer는 클릭점 선택, actor 갱신, ZAPI 전송 같은 상태 변경만 담당한다.
"""

from __future__ import annotations

import numpy as np

from viewervedo.geometry_utils import rotation_between_vectors, unit_vector


def profile_to_chuck_transform(pipe_axis, pipe_origin, chuck_axis, chuck_center):
    """배관 프로파일 기준축/기준점을 chuck 축/원점에 맞추는 4x4 변환을 만든다.

    입력:
        pipe_axis: 배관 프로파일 축 벡터.
        pipe_origin: 배관에서 chuck 원점에 붙일 기준점.
        chuck_axis: 정렬 대상 chuck 축 벡터.
        chuck_center: 정렬 대상 chuck 원점.

    출력:
        4x4 homogeneous transform. `R @ pipe_axis`가 `chuck_axis`를 향하고,
        `R @ pipe_origin + t`가 `chuck_center`와 일치한다.
    """
    pipe_axis = unit_vector(pipe_axis)
    chuck_axis = unit_vector(chuck_axis)
    pipe_origin = np.asarray(pipe_origin, dtype=float)
    chuck_center = np.asarray(chuck_center, dtype=float)

    R_align = rotation_between_vectors(pipe_axis, chuck_axis)
    T_align = np.eye(4)
    T_align[:3, :3] = R_align
    T_align[:3, 3] = chuck_center - R_align @ pipe_origin
    return T_align


def transformed_profile_alignment_summary(pipe_axis, pipe_origin, radius, target_axis, target_center, transform):
    """정렬 transform 적용 후 profile 중심/축 오차를 요약한다.

    입력:
        pipe_axis: transform 적용 전 배관 프로파일 축.
        pipe_origin: transform 적용 전 배관 기준점.
        radius: 피팅된 배관 반지름.
        target_axis: 정렬 대상 축.
        target_center: 정렬 대상 원점.
        transform: `profile_to_chuck_transform` 등으로 만든 4x4 변환.

    출력:
        center, axis, radius, center_error, axis_error_deg를 가진 dict.
    """
    R = np.asarray(transform[:3, :3], dtype=float)
    t = np.asarray(transform[:3, 3], dtype=float)
    target_axis = unit_vector(target_axis)
    target_center = np.asarray(target_center, dtype=float)
    aligned_origin = R @ np.asarray(pipe_origin, dtype=float) + t
    aligned_axis = unit_vector(R @ np.asarray(pipe_axis, dtype=float))
    dot = float(np.clip(np.dot(aligned_axis, target_axis), -1.0, 1.0))
    return {
        "axis": aligned_axis,
        "center": aligned_origin,
        "radius": float(radius),
        "center_error": float(np.linalg.norm(aligned_origin - target_center)),
        "axis_error_deg": float(np.rad2deg(np.arccos(dot))),
    }
