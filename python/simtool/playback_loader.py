"""Build a Viewer-playable "plan_sequence" (see plugins.robotics.inspection_
workflow / InspectionSequencer.to_plan_sequence's format - a list of
{"name", "positioner_r_deg", "plans": {robot: {"q_path": [...]}}}) from a
debug run folder saved by test_ompl_planning.py's --target all mode:

    debug/<method>_<timestamp>/
        summary.csv
        00_<robot>_<pose>/joint_states.csv
        01_<robot>_<pose>/joint_states.csv
        ...

This is the same shape InspectionSequencer.to_plan_sequence() produces from a
live SimTool planning run, so SimTool's "Start Simulation" playback
(InspectionPathHandler.accept_result -> workflow.plan_sequence ->
Visualizer._start_path_playback) plays a loaded result file identically to a
just-planned one - no separate playback code path.

Only test_ompl_planning.py saves q_path per target (benchmark_path_planners.py
doesn't - it only computes summary metrics, see its target_metrics.csv) so
this loader is specific to that script's output shape.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


def _read_joint_states_csv(path: Path) -> List[List[float]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header: waypoint, <joint_name>, ...
        return [[float(v) for v in row[1:]] for row in reader]


def load_playback_plan_sequence(run_dir) -> Dict[str, Any]:
    """Returns {"plan_sequence": [...], "n_targets": int, "n_loaded": int,
    "skipped": [{"index", "robot_name", "pose_name", "status", "reason"}]}.

    Groups multi-robot targets (e.g. DDA + RT of the same inspection pose)
    under one plan_sequence entry by group_name, in summary.csv's index
    order, so multi-robot groups still play back together like a live
    InspectionSequencer run's groups do.

    Loads a target's joint_states.csv regardless of its recorded status
    (success or failed) - failed targets now have their actual (colliding)
    q_path saved too (see path_planning_service.py's exc.q_path plumbing),
    so a failed target can still be played back/inspected here; callers that
    only want successful ones should filter on the returned skip info's
    absence, or check each plan's "status" key.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        # Not a --target all run (e.g. a single-target run has joint_states.csv
        # directly in run_dir, with no robot/pose/group info recoverable from
        # the folder name alone) - nothing this loader can group meaningfully.
        raise FileNotFoundError(
            f"{summary_path} not found - this loader only supports test_ompl_planning.py's "
            "--target all output (a folder with summary.csv + per-target subfolders).")

    order: List[str] = []
    groups: Dict[str, Dict[str, Any]] = {}
    skipped = []
    n_targets = 0

    with open(summary_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_targets += 1
            index = row.get("index", "")
            robot_name = row.get("robot_name", "")
            pose_name = row.get("pose_name", "")
            group_name = row.get("group_name") or f"target_{index}"
            status = row.get("status", "")
            try:
                index_label = f"{int(index):02d}"
            except ValueError:
                index_label = str(index)
            subdir = run_dir / f"{index_label}_{robot_name}_{pose_name}"
            csv_path = subdir / "joint_states.csv"
            if not csv_path.exists():
                skipped.append({
                    "index": index, "robot_name": robot_name, "pose_name": pose_name,
                    "status": status, "reason": f"no joint_states.csv at {csv_path}",
                })
                continue
            q_path = _read_joint_states_csv(csv_path)
            if not q_path:
                skipped.append({
                    "index": index, "robot_name": robot_name, "pose_name": pose_name,
                    "status": status, "reason": "joint_states.csv is empty",
                })
                continue
            if group_name not in groups:
                # Positioner angle this target was actually collision-checked
                # against (see test_ompl_planning.py's _resolve_positioner_r_deg
                # / visualizer.py's planner.debug_positioner_r_deg) - NOT
                # always 0. A rotated-group target loaded with the wrong
                # angle here shows the pipe/positioner in the wrong pose
                # during playback even though the robot's q_path is correct.
                try:
                    r_deg = float(row.get("positioner_r_deg") or 0.0)
                except ValueError:
                    r_deg = 0.0
                groups[group_name] = {"name": group_name, "positioner_r_deg": r_deg, "plans": {}}
                order.append(group_name)
            groups[group_name]["plans"][robot_name] = {"q_path": q_path, "status": status}

    plan_sequence = [groups[name] for name in order]
    return {
        "plan_sequence": plan_sequence,
        "n_targets": n_targets,
        "n_loaded": sum(len(g["plans"]) for g in plan_sequence),
        "skipped": skipped,
    }


def load_playback_single_target(run_dir, robot_name: str, pose_name: str = "target",
                                 positioner_r_deg: float = 0.0) -> Dict[str, Any]:
    """Same output shape as load_playback_plan_sequence(), but for a single
    test_ompl_planning.py --target N run (joint_states.csv sits directly in
    run_dir, no summary.csv - that script never writes one for a single
    target, since a lone folder has no group/robot/pose info recoverable
    from its name alone). Separate from load_playback_plan_sequence() rather
    than folded into it because that function's grouping-by-summary.csv logic
    doesn't apply here at all - the caller already knows which one robot/pose
    this run was for and just supplies it directly.

    robot_name is required (not recoverable from the folder) - pass the same
    --robot-name/target you ran test_ompl_planning.py with.

    positioner_r_deg defaults to 0.0 - a single-target run_dir has nowhere
    the angle it was actually planned against is recorded (see
    test_ompl_planning.py's console log: "target[N] needed a positioner
    rotation..."), so pass it explicitly if that target needed rotation,
    or the pipe/positioner will render at the wrong angle during playback
    even though the robot's q_path itself is still correct.
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / "joint_states.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found - expected a single --target N run_dir "
            "(test_ompl_planning.py writes joint_states.csv directly under run_dir for one target).")

    q_path = _read_joint_states_csv(csv_path)
    if not q_path:
        return {
            "plan_sequence": [], "n_targets": 1, "n_loaded": 0,
            "skipped": [{"index": 0, "robot_name": robot_name, "pose_name": pose_name,
                         "status": "", "reason": "joint_states.csv is empty"}],
        }

    plan_sequence = [{
        "name": pose_name, "positioner_r_deg": float(positioner_r_deg),
        "plans": {robot_name: {"q_path": q_path, "status": "success"}},
    }]
    return {"plan_sequence": plan_sequence, "n_targets": 1, "n_loaded": 1, "skipped": []}
