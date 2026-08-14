import numpy as np

from plugins.robotics.inspection_workflow import (
    InspectionWorkflowState,
    partition_and_sort_target_groups,
)


def _group(name, x, reachable):
    pose = np.eye(4)
    pose[0, 3] = x
    if reachable:
        pose[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    return {"name": name, "rt_pose": pose, "dda_pose": np.eye(4)}


def test_partition_preserves_existing_reachability_and_sort_order():
    phases = partition_and_sort_target_groups([
        _group("r3", 3.0, True),
        _group("d2", 2.0, False),
        _group("r1", 1.0, True),
        _group("d4", 4.0, False),
    ])

    assert [item["name"] for item in phases[0]["groups"]] == ["r1", "r3"]
    assert [item["name"] for item in phases[1]["groups"]] == ["d4", "d2"]
    assert isinstance(phases[0]["groups"][0]["rt_pose"], list)


def test_simtool_workflow_owns_pose_plan_and_playback_state():
    state = InspectionWorkflowState()
    state.set_selected_points({"point": [1, 2, 3], "points": [[1, 2, 3], [4, 5, 6]]})
    state.set_pose_result({"status": "success", "target_groups": [_group("r", 1.0, True)]})
    state.set_planning_result({
        "status": "success",
        "plan_sequence": [{"name": "r", "plans": {"robot": {"q_path": [[0], [1]]}}}],
    })

    assert state.selected_points == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert len(state.planner_target_groups()) == 1
    assert state.plan_sequence[0]["name"] == "r"

    state.set_selected_points({"point": [7, 8, 9]})
    assert state.target_groups == []
    assert state.plan_sequence == []
