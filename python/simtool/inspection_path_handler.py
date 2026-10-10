from __future__ import annotations

from typing import Any, Dict

from plugins.robotics.inspection_workflow import InspectionWorkflowState


def resolve_planner_timeout(config, planner_name):
    timeout_config = (config or {}).get("planner_timeouts", {}) or {}
    planner_name = str(planner_name or "rrt_connect")
    return float(timeout_config.get(planner_name, timeout_config.get("default", 0.0)))


class InspectionPathHandler:
    """Own SimTool's IK-check result and playback request lifecycle.

    Multi-target planning itself now goes through InspectionSequencer
    (simtool/inspection_sequencer.py), which drives Robot Core's
    plan_single_target calls directly - this class no longer builds/sends a
    planning request. accept_result() is still used for IK-check results
    (reply_inspection_path is shared with that feature).
    """

    def __init__(self, config: Dict[str, Any], workflow: InspectionWorkflowState):
        self.config = config or {}
        self.workflow = workflow

    def accept_result(self, result):
        self.workflow.set_planning_result(result)
        return self.workflow.planning_result

    def request_playback(self, zapi, *, speed=1.0):
        if zapi is None:
            raise RuntimeError("ZAPI is not available")
        if not self.workflow.plan_sequence:
            raise RuntimeError("planned path is not available")
        zapi._ZAPI_request_execute_inspection_path(
            speed=float(speed),
            plan_sequence=self.workflow.plan_sequence,
            playback_initial_r_deg=self.workflow.planning_result.get(
                "playback_initial_r_deg"
            ),
        )
