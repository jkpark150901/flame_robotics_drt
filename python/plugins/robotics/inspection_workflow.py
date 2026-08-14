from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Return a detached JSON-compatible representation of a workflow value."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return copy.deepcopy(value)


def _rt_pose(group: Mapping[str, Any]) -> Optional[np.ndarray]:
    pose = group.get("rt_pose")
    if pose is None:
        return None
    matrix = np.asarray(pose, dtype=float)
    return matrix if matrix.shape == (4, 4) else None


def group_is_reachable(
    group: Mapping[str, Any],
    rt_pipe_facing_axis: Iterable[float] = (0.0, -1.0, 0.0),
) -> bool:
    """Match the existing RT back-axis reachability rule without viewer state."""
    pose = _rt_pose(group)
    if pose is None:
        return False
    facing_axis = np.asarray(tuple(rt_pipe_facing_axis), dtype=float).reshape(3)
    back_axis_world = pose[:3, :3] @ -facing_axis
    return float(back_axis_world[1]) < 0.0


def linear_track_indices(joint_names: Iterable[Any]) -> List[int]:
    """Indices of linear-track (rail) joints in a joint-name list.

    Shared with InspectionPlanningBase._linear_track_indices (same
    "linear_track" substring convention) so the retreat-before-rotation reset
    below and the planner's own track-lock heuristic never drift apart.
    """
    return [i for i, name in enumerate(joint_names or []) if "linear_track" in str(name)]


def zero_non_linear_track_joints(q: Sequence[float], joint_names: Iterable[Any]) -> List[float]:
    """Reset every joint except the linear track to 0, keeping the track's
    current value.

    Used as the start_q for the first target planned after a positioner
    rotation: the robot must retreat to a safe, fully-known posture (arm
    folded to its zero pose) before the rotation happens, rather than
    starting the next plan from wherever the last (pre-rotation) target left
    the arm - an arbitrary pose that has no guaranteed clearance once the
    pipe/positioner has rotated under it.
    """
    q = [float(v) for v in q]
    track_indices = set(linear_track_indices(joint_names))
    return [v if i in track_indices else 0.0 for i, v in enumerate(q)]


def ef_pose_robot_name(pose_name: str) -> str:
    """Map a target-group pose_name ("DDA"/"RT") to its robot name.

    Shared by SimTool (sequencing) and Visualizer (rendering) so the mapping
    only lives in one place. See TARGET_GROUP_FORMAT.md.
    """
    return "dda_rb10_1300e" if pose_name == "DDA" else "rb20_1900es"


def inspection_group_pose_items(group_info: Mapping[str, Any]) -> List[tuple]:
    """(robot_name, pose_name, target_T) list for a target group. Pure - no viewer state.

    Prefers the resolved ("dda_pose_resolved"/"rt_pose_resolved") pose added by
    resolve_target_groups_with_rotation() over the raw ("dda_pose"/"rt_pose")
    one when present - the resolved pose already has any positioner rotation
    baked in, so callers get the pose that's actually reachable without having
    to know whether this group needed a rotation or reapply the transform
    themselves. group_is_reachable()/partition_and_sort_target_groups() still
    use the raw pose (see their docstrings) since rotation classification has
    to happen before the rotation is applied.
    """
    items: List[tuple] = []
    dda_pose = group_info.get("dda_pose_resolved", group_info.get("dda_pose"))
    if dda_pose is not None:
        items.append((ef_pose_robot_name("DDA"), "DDA", np.asarray(dda_pose, dtype=float)))
    rt_pose = group_info.get("rt_pose_resolved", group_info.get("rt_pose"))
    if rt_pose is not None:
        items.append((ef_pose_robot_name("RT"), "RT", np.asarray(rt_pose, dtype=float)))
    return items


def _group_sort_key(group: Mapping[str, Any], x_ascending: bool) -> tuple:
    pose = _rt_pose(group)
    position = np.zeros(3, dtype=float) if pose is None else pose[:3, 3]
    x_sign = 1.0 if x_ascending else -1.0
    return x_sign * float(position[0]), -float(position[2])


def partition_and_sort_target_groups(
    target_groups: Iterable[Mapping[str, Any]],
    *,
    rt_pipe_facing_axis: Iterable[float] = (0.0, -1.0, 0.0),
    first_x_ascending: bool = True,
    second_x_ascending: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Split pose groups into the two positioner phases used by inspection planning."""
    if second_x_ascending is None:
        second_x_ascending = not first_x_ascending
    groups = [to_jsonable(group) for group in target_groups or []]
    first = [
        group for group in groups
        if group_is_reachable(group, rt_pipe_facing_axis)
    ]
    second = [
        group for group in groups
        if not group_is_reachable(group, rt_pipe_facing_axis)
    ]
    first.sort(key=lambda group: _group_sort_key(group, first_x_ascending))
    second.sort(key=lambda group: _group_sort_key(group, bool(second_x_ascending)))
    return [
        {"name": "reachable", "requires_positioner_rotation": False, "groups": first},
        {"name": "deferred", "requires_positioner_rotation": True, "groups": second},
    ]


def resolve_target_groups_with_rotation(
    target_groups: Iterable[Mapping[str, Any]],
    *,
    rotation_T: Optional[Any] = None,
    rt_pipe_facing_axis: Iterable[float] = (0.0, -1.0, 0.0),
) -> List[Dict[str, Any]]:
    """Bake positioner rotation into every group's pose so the *complete* set
    of poses is available up front, instead of only the raw (pre-rotation)
    poses the pose optimizer produces plus a separate rotation delta that
    callers have to remember to apply conditionally.

    Adds "dda_pose_resolved"/"rt_pose_resolved" to each group:
    - groups that don't need a rotation (group_is_reachable() is True): equal
      to the raw pose.
    - groups that do need a rotation: rotation_T @ raw_pose, if rotation_T is
      given. If rotation_T is None (rotation not actually applied to the
      collision mesh - e.g. spool_fix_r is False), the raw pose is kept as
      the resolved one and "positioner_rotation_unresolved": True is set so
      consumers can tell this group's resolved pose does NOT yet reflect the
      rotation that would actually be needed to reach it.

    The raw "dda_pose"/"rt_pose" fields are left untouched - group_is_reachable()
    classifies rotation-need from the *raw* (pre-rotation) pose, so anything
    that re-derives the reachable/deferred split later (partition_and_sort_
    target_groups) still needs it.
    """
    rotation = None if rotation_T is None else np.asarray(rotation_T, dtype=float)
    resolved = []
    for group in target_groups or []:
        group = to_jsonable(group)
        needs_rotation = not group_is_reachable(group, rt_pipe_facing_axis)
        for pose_key, resolved_key in (("dda_pose", "dda_pose_resolved"), ("rt_pose", "rt_pose_resolved")):
            raw_pose = group.get(pose_key)
            if raw_pose is None:
                continue
            raw_pose = np.asarray(raw_pose, dtype=float)
            if needs_rotation and rotation is not None:
                group[resolved_key] = (rotation @ raw_pose).tolist()
            else:
                group[resolved_key] = raw_pose.tolist()
        if needs_rotation and rotation is None:
            group["positioner_rotation_unresolved"] = True
        resolved.append(group)
    return resolved


@dataclass
class InspectionWorkflowState:
    """SimTool-owned state for the inspection pose/planning/playback workflow."""

    rt_pipe_facing_axis: List[float] = field(default_factory=lambda: [0.0, -1.0, 0.0])
    selected_points: List[List[float]] = field(default_factory=list)
    pose_result: Dict[str, Any] = field(default_factory=dict)
    target_groups: List[Dict[str, Any]] = field(default_factory=list)
    target_group_phases: List[Dict[str, Any]] = field(default_factory=list)
    planning_result: Dict[str, Any] = field(default_factory=dict)
    plan_sequence: List[Dict[str, Any]] = field(default_factory=list)

    def set_selected_points(self, payload: Any) -> None:
        data = payload if isinstance(payload, Mapping) else {}
        points = data.get("points") if data else None
        if not points and data.get("point") is not None:
            points = [data.get("point")]
        self.selected_points = [
            np.asarray(point, dtype=float).reshape(-1)[:3].tolist()
            for point in (points or [])
        ]
        self.clear_pose_and_path()

    def set_pose_result(self, result: Mapping[str, Any]) -> None:
        self.pose_result = to_jsonable(dict(result or {}))
        self.target_groups = list(self.pose_result.get("target_groups") or [])
        self.target_group_phases = partition_and_sort_target_groups(
            self.target_groups,
            rt_pipe_facing_axis=self.rt_pipe_facing_axis,
        )
        self.planning_result = {}
        self.plan_sequence = []

    def planner_target_groups(self) -> List[Dict[str, Any]]:
        return [
            group
            for phase in self.target_group_phases
            for group in phase.get("groups", [])
        ]

    def set_planning_result(self, result: Mapping[str, Any]) -> None:
        self.planning_result = to_jsonable(dict(result or {}))
        self.plan_sequence = list(self.planning_result.get("plan_sequence") or [])

    def clear_pose_and_path(self) -> None:
        self.pose_result = {}
        self.target_groups = []
        self.target_group_phases = []
        self.planning_result = {}
        self.plan_sequence = []

    def clear(self) -> None:
        self.selected_points = []
        self.clear_pose_and_path()
