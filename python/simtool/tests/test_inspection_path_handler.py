from pathlib import Path

from plugins.robotics.inspection_workflow import InspectionWorkflowState
from simtool.inspection_path_handler import InspectionPathHandler


class _ZAPI:
    def __init__(self):
        self.playback_kwargs = None

    def _ZAPI_request_execute_inspection_path(self, **kwargs):
        self.playback_kwargs = kwargs


def test_handler_accepts_result_and_requests_owned_playback_sequence():
    workflow = InspectionWorkflowState()
    handler = InspectionPathHandler({}, workflow)
    zapi = _ZAPI()
    result = {
        "status": "success",
        "plan_sequence": [{"name": "pose-1", "plans": {}}],
        "playback_initial_r_deg": 15.0,
    }

    handler.accept_result(result)
    handler.request_playback(zapi, speed=0.5)

    assert zapi.playback_kwargs == {
        "speed": 0.5,
        "plan_sequence": result["plan_sequence"],
        "playback_initial_r_deg": 15.0,
    }


def test_viewer_does_not_execute_planning_batches():
    python_root = Path(__file__).resolve().parents[2]
    source = (python_root / "viewervedo" / "visualizer.py").read_text(
        encoding="utf-8"
    )

    assert ".plan_batch(" not in source
    assert "def _plan_target_for_job" not in source
    assert "def _compute_inspection_path" not in source
