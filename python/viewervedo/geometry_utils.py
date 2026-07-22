"""Geometry helpers shared by the Vedo viewer.

이 모듈은 viewer 상태나 plotter에 의존하지 않는 순수 기하 계산만 담당한다.
배관 정렬, pose 변환, 회전 행렬 계산을 여기로 모아 visualizer의 책임을 줄인다.
"""

from __future__ import annotations

import numpy as np


def unit_vector(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


def rotation_between_vectors(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < 1e-12 or target_norm < 1e-12:
        return np.eye(3)
    a = source / source_norm
    b = target / target_norm
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        basis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(a, basis))) > 0.9:
            basis = np.array([0.0, 1.0, 0.0])
        axis = unit_vector(np.cross(a, basis))
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))


def signed_angle_about_axis(source, target, axis):
    axis = unit_vector(axis)
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source = source - np.dot(source, axis) * axis
    target = target - np.dot(target, axis) * axis
    if np.linalg.norm(source) < 1e-12 or np.linalg.norm(target) < 1e-12:
        return 0.0
    source = unit_vector(source)
    target = unit_vector(target)
    sin_v = float(np.dot(axis, np.cross(source, target)))
    cos_v = float(np.clip(np.dot(source, target), -1.0, 1.0))
    return float(np.arctan2(sin_v, cos_v))


def frame_from_primary_and_reference(primary, reference):
    x_axis = np.asarray(primary, dtype=float)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-12:
        raise RuntimeError("cannot build frame from zero-length primary vector")
    x_axis = x_axis / x_norm
    ref = np.asarray(reference, dtype=float)
    ref = ref - np.dot(ref, x_axis) * x_axis
    if np.linalg.norm(ref) < 1e-12:
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(ref, x_axis))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        ref = ref - np.dot(ref, x_axis) * x_axis
    y_axis = ref / np.linalg.norm(ref)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def rpy_matrix(rpy):
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def pose_to_T(pose):
    pose = np.asarray(pose, dtype=float)
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    if pose.shape[0] >= 6:
        T[:3, :3] = rpy_matrix(pose[3:6])
    return T


def T_to_pose(T):
    T = np.asarray(T, dtype=float)
    Rm = T[:3, :3]
    sy = float(np.sqrt(Rm[0, 0] * Rm[0, 0] + Rm[1, 0] * Rm[1, 0]))
    if sy > 1e-9:
        roll = np.arctan2(Rm[2, 1], Rm[2, 2])
        pitch = np.arctan2(-Rm[2, 0], sy)
        yaw = np.arctan2(Rm[1, 0], Rm[0, 0])
    else:
        roll = np.arctan2(-Rm[1, 2], Rm[1, 1])
        pitch = np.arctan2(-Rm[2, 0], sy)
        yaw = 0.0
    return np.asarray([T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw], dtype=float)


def rotz(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    T = np.eye(4)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


def rotx(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    T = np.eye(4)
    T[1, 1] = c
    T[1, 2] = -s
    T[2, 1] = s
    T[2, 2] = c
    return T


def transl(v):
    T = np.eye(4)
    T[:3, 3] = np.asarray(v, dtype=float)
    return T


def rot_about_axis(axis, center, deg):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.eye(4)
    x, y, z = axis / n
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    C = 1 - c
    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    center = np.asarray(center, dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center - R @ center
    return T


def transformed_pipe_profile(profile, transform):
    transformed = dict(profile)
    R = np.asarray(transform[:3, :3], dtype=float)
    t = np.asarray(transform[:3, 3], dtype=float)
    for key in ("center", "end_center"):
        if key in transformed and transformed[key] is not None:
            transformed[key] = R @ np.asarray(transformed[key], dtype=float) + t
    if "axis" in transformed and transformed["axis"] is not None:
        transformed["axis"] = unit_vector(R @ np.asarray(transformed["axis"], dtype=float))
    if "fit_points" in transformed and transformed["fit_points"] is not None:
        pts = np.asarray(transformed["fit_points"], dtype=float)
        transformed["fit_points"] = (R @ pts.T).T + t
    return transformed
