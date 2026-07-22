"""Vedo actor construction helpers.

이 모듈은 actor 생성만 담당한다. plotter add/remove와 actor 생명주기 관리는
viewer가 담당한다.
"""

from __future__ import annotations

import numpy as np
import vedo

from viewervedo.geometry_utils import pose_to_T, unit_vector


def pose_frame_actors(pose, scale=0.18, axes=(0, 1, 2), show_origin=True):
    pose_arr = np.asarray(pose, dtype=float)
    T = pose_arr if pose_arr.shape == (4, 4) else pose_to_T(pose_arr)
    origin = T[:3, 3]
    actors = []
    for axis, color in ((0, "red"), (1, "green"), (2, "blue")):
        if axis not in axes:
            continue
        actor = vedo.Arrow(origin, origin + T[:3, axis] * scale, s=0.0008, c=color)
        actor.alpha(0.35)
        actor.pickable(False)
        actors.append(actor)
    if show_origin:
        marker = vedo.Sphere(pos=origin, r=scale * 0.055, c="yellow")
        marker.alpha(0.35)
        marker.pickable(False)
        actors.append(marker)
    return actors


def profile_cylinder_actor(center, axis, radius, length, color="cyan", alpha=0.22):
    center = np.asarray(center, dtype=float)
    axis = unit_vector(axis)
    radius = float(radius)
    length = float(length)
    if radius <= 0.0 or length <= 0.0 or np.linalg.norm(axis) < 1e-12:
        return None

    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, axis))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = unit_vector(np.cross(axis, ref))
    v = unit_vector(np.cross(axis, u))
    n = 64
    half = length * 0.5
    verts = []
    for z in (-half, half):
        cap_center = center + axis * z
        for i in range(n):
            theta = 2.0 * np.pi * i / n
            verts.append(cap_center + radius * (np.cos(theta) * u + np.sin(theta) * v))
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j, n + i])
    faces.append(list(range(n - 1, -1, -1)))
    faces.append(list(range(n, 2 * n)))
    actor = vedo.Mesh([np.asarray(verts, dtype=float), faces])
    actor.c(color).alpha(alpha).wireframe()
    actor.pickable(False)
    return actor


def fit_points_actor(points, color="magenta", point_size=4):
    if points is None:
        return None
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return None
    actor = vedo.Points(points).c(color).ps(point_size)
    actor.pickable(False)
    return actor


def alignment_reference_actors(
    origin,
    axis,
    axis_len,
    label="ALIGN_REF",
    color="yellow",
    far_point=None,
):
    origin = np.asarray(origin, dtype=float)
    axis = unit_vector(axis)
    axis_len = float(axis_len)
    if np.linalg.norm(axis) < 1e-12 or axis_len <= 0.0:
        return []

    actors = []
    marker = vedo.Sphere(pos=origin, r=max(axis_len * 0.07, 0.012), c=color)
    marker.pickable(False)
    actors.append(marker)

    try:
        arrow = vedo.Arrow(origin, origin + axis * axis_len, s=0.002, c=color)
    except Exception:
        arrow = vedo.Line(origin, origin + axis * axis_len, c=color, lw=8)
    arrow.pickable(False)
    actors.append(arrow)

    if far_point is not None:
        far_point = np.asarray(far_point, dtype=float)
        far_marker = vedo.Sphere(pos=far_point, r=max(axis_len * 0.045, 0.009), c="gray")
        far_marker.pickable(False)
        actors.append(far_marker)
        far_line = vedo.Line(origin, far_point, c=color, lw=2)
        far_line.alpha(0.45)
        far_line.pickable(False)
        actors.append(far_line)

    text = vedo.Text3D(label, pos=origin + axis * axis_len * 1.06, s=axis_len * 0.12, c=color)
    text.pickable(False)
    actors.append(text)
    return actors

