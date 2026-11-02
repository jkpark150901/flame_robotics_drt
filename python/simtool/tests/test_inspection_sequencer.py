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


def test_rotation_disabled_plans_deferred_target_at_current_angle():
    zapi = _FakeZAPI()  # deliberately no rotation request method
    seq = InspectionSequencer(logging.getLogger('test'))
    seq.start(zapi, [_target_group('B', 0, reachable_now=False)],
              planner='rrt_connect', initial_positioner_r_deg=-92.3,
              auto_rotate_positioner=False)
    assert len(zapi.calls) == 1
    assert seq._deferred_groups == []
    assert seq._current_phase_r_deg == -92.3
    seq.on_reply({'request_id': zapi.calls[0]['request_id'],
                  'status': 'success', 'q_path': [[0, 0], [0.1, 0.2]]})
    assert not seq.is_running


def test_rotation_enabled_dispatches_phase_preparation_for_deferred_only():
    class API(_FakeZAPI):
        def _ZAPI_request_prepare_next_inspection_phase(self, **kwargs):
            self.rotation = kwargs
    zapi=API()
    seq=InspectionSequencer(logging.getLogger('test'))
    seq.start(zapi, [_target_group('B',0,reachable_now=False)],planner='rrt_connect',
              initial_positioner_r_deg=90,auto_rotate_positioner=True)
    assert zapi.calls==[]
    assert zapi.rotation['r_deg_delta']==180
    assert seq._preparing_phase


def test_phase_retreat_is_saved_before_rotated_targets():
    seq = InspectionSequencer(logging.getLogger('test'))
    seq._active = True
    seq._preparing_phase = True
    seq._pending_phase_request_id = 'prep'
    seq._current_phase_r_deg = -90.0
    seq._phase_no = 1
    seq._grand_total = 0
    seq._finish = lambda: None
    path = [[0, 1], [0, .4], [0, 0]]
    assert seq.on_phase_prepared({
        'request_id': 'prep', 'status': 'success', 'positioner_r_deg': 90,
        'start_q_by_robot': {'robot': path[-1]},
        'retreat_plans': {'robot': {'q_path': path, 'elapsed': 1.2,
                                   'verification': {'colliding_edges': 0}}}})
    sequence = seq.to_plan_sequence()
    assert sequence[0]['positioner_r_deg'] == -90
    assert sequence[0]['plans']['robot']['q_path'] == path
    assert seq._results[0]['kind'] == 'retreat'
    assert seq._current_phase_r_deg == 90
