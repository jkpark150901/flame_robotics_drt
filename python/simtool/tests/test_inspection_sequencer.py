import logging

import numpy as np

from simtool.inspection_sequencer import InspectionSequencer


class _FakeZAPI:
    def __init__(self):
        self.calls = []

    def _ZAPI_request_plan_single_target(self, **kwargs):
        self.calls.append(kwargs)


def _rt_pose(reachable_now: bool):
    """RT pose whose back-axis world-y sign selects the reachable/deferred phase.

    Matches plugins.robotics.inspection_workflow.group_is_reachable's rule:
    reachable when (pose[:3,:3] @ -[0,-1,0])[1] < 0.
    """
    pose = np.eye(4)
    pose[1, 1] = -1.0 if reachable_now else 1.0
    pose[:3, 3] = [0.1, 0.2, 0.3]
    return pose.tolist()


def _target_group(name, index, *, reachable_now):
    return {
        "name": name,
        "index": index,
        "target_point": [0.1, 0.2, 0.3],
        "dda_pose": None,
        "rt_pose": _rt_pose(reachable_now),
    }


def test_sequencer_plans_reachable_groups_and_defers_the_rest():
    zapi = _FakeZAPI()
    sequencer = InspectionSequencer(logging.getLogger("test"))
    target_groups = [
        _target_group("A", 0, reachable_now=True),
        _target_group("B", 1, reachable_now=False),
    ]

    finished = {}
    sequencer.start(
        zapi, target_groups, planner="rrt_connect",
        on_finished=lambda summary: finished.update(summary))

    assert sequencer.is_running
    assert len(zapi.calls) == 1
    first_call = zapi.calls[0]
    assert first_call["robot"] == "rb20_1900es"  # RT-only group -> RT robot
    assert first_call["start_q"] is None  # first target for this robot: let Viewer resolve "current"

    request_id = first_call["request_id"]
    consumed = sequencer.on_reply({
        "request_id": request_id, "status": "success",
        "q_path": [[0.0, 0.0], [0.1, 0.2]],
    })

    assert consumed
    assert not sequencer.is_running
    assert finished["status"] == "success"
    assert finished["n_planned"] == 1
    assert finished["n_total"] == 1
    assert finished["deferred_groups"] == ["B"]


def test_sequencer_chains_start_q_across_targets_for_the_same_robot():
    zapi = _FakeZAPI()
    sequencer = InspectionSequencer(logging.getLogger("test"))
    target_groups = [
        _target_group("A", 0, reachable_now=True),
        _target_group("B", 1, reachable_now=True),
    ]

    sequencer.start(zapi, target_groups, planner="rrt_connect")
    first_request_id = zapi.calls[0]["request_id"]
    sequencer.on_reply({
        "request_id": first_request_id, "status": "success",
        "q_path": [[0.0, 0.0], [1.0, 2.0]],
    })

    assert len(zapi.calls) == 2
    assert zapi.calls[1]["start_q"] == [1.0, 2.0]


def test_to_plan_sequence_groups_successful_targets_for_playback():
    zapi = _FakeZAPI()
    sequencer = InspectionSequencer(logging.getLogger("test"))
    target_groups = [_target_group("A", 0, reachable_now=True)]

    sequencer.start(zapi, target_groups, planner="rrt_connect")
    request_id = zapi.calls[0]["request_id"]
    sequencer.on_reply({
        "request_id": request_id, "status": "success",
        "q_path": [[0.0, 0.0], [1.0, 2.0]],
    })

    plan_sequence = sequencer.to_plan_sequence()
    assert plan_sequence == [{
        "name": "A",
        "positioner_r_deg": 0.0,
        "plans": {"rb20_1900es": {"q_path": [[0.0, 0.0], [1.0, 2.0]]}},
    }]


def test_to_plan_sequence_excludes_failed_targets():
    zapi = _FakeZAPI()
    sequencer = InspectionSequencer(logging.getLogger("test"))
    target_groups = [_target_group("A", 0, reachable_now=True)]

    sequencer.start(zapi, target_groups, planner="rrt_connect")
    request_id = zapi.calls[0]["request_id"]
    sequencer.on_reply({"request_id": request_id, "status": "failed", "message": "no path"})

    assert sequencer.to_plan_sequence() == []


def test_sequencer_stops_the_sequence_on_first_failure():
    zapi = _FakeZAPI()
    sequencer = InspectionSequencer(logging.getLogger("test"))
    target_groups = [
        _target_group("A", 0, reachable_now=True),
        _target_group("B", 1, reachable_now=True),
    ]

    finished = {}
    sequencer.start(
        zapi, target_groups, planner="rrt_connect",
        on_finished=lambda summary: finished.update(summary))
    request_id = zapi.calls[0]["request_id"]
    sequencer.on_reply({"request_id": request_id, "status": "failed", "message": "no path"})

    assert len(zapi.calls) == 1  # second target never dispatched
    assert not sequencer.is_running
    assert finished["status"] == "failed"
    assert finished["n_planned"] == 0
