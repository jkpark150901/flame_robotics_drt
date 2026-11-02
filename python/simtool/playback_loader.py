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

benchmark_path_planners.py's --save-paths writes this same shape too (see its
_write_path_summary_csv/_save_joint_states_csv), so this loader works
unchanged on either script's output - no format-specific branching needed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from plugins.robotics.plan_run_io import load_run_info


def _read_joint_states_csv(path: Path) -> List[List[float]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header: waypoint, <joint_name>, ...
        return [[float(v) for v in row[1:]] for row in reader]


def resolve_playback_run_dir(run_dir) -> Path:
    """Resolve a benchmark output parent to its one playable saved-path run."""
    run_dir = Path(run_dir)
    if (run_dir / "summary.csv").is_file() or (run_dir / "joint_states.csv").is_file():
        return run_dir
    candidates = sorted((run_dir / "result_paths").glob("*/summary.csv"))
    candidates += sorted(run_dir.glob("*/summary.csv"))
    if len(candidates) == 1:
        return candidates[0].parent
    return run_dir


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
    # failures.json/run.json exist for runs saved by plugins.robotics.plan_run_io (SimTool auto-save
    # and benchmark --save-paths); older runs simply have none and get failures derived from summary.csv.
    run_info = load_run_info(run_dir)
    if not summary_path.exists():
        # Not a --target all run (e.g. a single-target run has joint_states.csv
        # directly in run_dir, with no robot/pose/group info recoverable from
        # the folder name alone) - nothing this loader can group meaningfully.
        raise FileNotFoundError(
            f"{summary_path} not found - this loader only supports test_ompl_planning.py's "
            "--target all output (a folder with summary.csv + per-target subfolders).")

    # failures.json / <idx>_*/result.json (plan_run_io) carry the failed target's IK goal
    goal_by_index = {str(f.get("index")): (f.get("goal_q") or f.get("ik_goal_q")) for f in (run_info.get("failures") or [])
                     if (f.get("goal_q") or f.get("ik_goal_q")) is not None}
    order: List[str] = []
    groups: Dict[str, Dict[str, Any]] = {}
    skipped = []
    derived_failures: List[Dict[str, Any]] = []
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
            row_message = row.get("message") or ""
            failure_category = row.get("failure_category") or ""
            if status != "success":
                derived_failures.append({
                    "index": index, "group_name": group_name, "robot_name": robot_name,
                    "pose_name": pose_name, "status": status, "category": failure_category,
                    "message": row_message,
                })
            if not csv_path.exists():
                skipped.append({
                    "index": index, "robot_name": robot_name, "pose_name": pose_name,
                    "status": status,
                    "reason": f"no joint_states.csv at {csv_path}"
                              + (f" (planning {status}: {row_message})" if status != "success" and row_message else ""),
                })
                continue
            q_path = _read_joint_states_csv(csv_path)
            if not q_path:
                skipped.append({
                    "index": index, "robot_name": robot_name, "pose_name": pose_name,
                    "status": status, "reason": "joint_states.csv is empty",
                })
                continue
            # benchmark_path_planners.py's --save-paths writes this sidecar
            # (see _save_collision_sidecar) next to joint_states.csv whenever
            # the target actually collided - carries plannerbase.verify_path's
            # edge_collisions/verification shape, the same thing a LIVE plan
            # has. Without it, visualizer.py's _inspection_plan_collision_
            # reason()/_warn_collision_preview_playback() have nothing to see
            # (a loaded plan otherwise only has q_path/status) and silently
            # play back a colliding path as if it were clean - see the
            # "status" fallback added there for runs saved before this sidecar
            # existed, which still warns but can't highlight the exact links.
            plan_extra = {}
            collision_path = subdir / "collisions.json"
            if collision_path.exists():
                try:
                    with open(collision_path, "r", encoding="utf-8") as cf:
                        collision_data = json.load(cf)
                    plan_extra = {
                        "collision_preview": bool(collision_data.get("collision_preview")),
                        "edge_collisions": collision_data.get("edge_collisions") or [],
                        "verification": collision_data.get("verification") or {},
                    }
                except Exception:
                    plan_extra = {}
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
            status_extra = {}
            if status != "success":
                # Same keys InspectionSequencer.to_plan_sequence() gives a live failed plan, so the Viewer's
                # collision-preview/warning playback treats a loaded failure like a fresh one.
                status_extra = {
                    "status_kind": row.get("status_kind") or "failed",
                    "planning_error": row_message or None,
                    "collision_preview": True,
                }
            goal_q = goal_by_index.get(str(index))
            if goal_q is not None:
                status_extra["goal_q"] = goal_q   # failed target: playback shows the straight move start -> IK goal
            groups[group_name]["plans"][robot_name] = {
                "q_path": q_path, "status": status, **status_extra, **plan_extra}

    plan_sequence = [groups[name] for name in order]
    _fix_reachable_phase_angle(plan_sequence, run_info.get("run") or {})
    return {
        "plan_sequence": plan_sequence,
        "n_targets": n_targets,
        "n_loaded": sum(len(g["plans"]) for g in plan_sequence),
        "skipped": skipped,
        # Why each failed target failed: failures.json when present (has category/planner stats/colliding
        # edges), otherwise just what summary.csv recorded.
        "failures": run_info["failures"] or derived_failures,
        "unreached_groups": run_info["unreached_groups"],
        "run": run_info["run"],
        # scene the run was planned in (saved by SimTool at sequence start), None for older runs
        "scene_snapshot": (str(run_dir / "scene_snapshot.pkl") if (run_dir / "scene_snapshot.pkl").exists() else None),
    }


def _fix_reachable_phase_angle(plan_sequence: List[Dict[str, Any]], run_meta: Dict[str, Any]) -> None:
    """Runs saved before 2026-10-29 recorded the reachable (non-rotated) phase as positioner_r_deg=0.0 while the
    rotated phase has the absolute angle (e.g. 453.4 = 273.4 + 180), so playback turned the positioner to 0 deg for the
    phase-1 groups. Put the real starting angle there: run.json's initial_positioner_r_deg (newer runs record it), else
    rotated angle - positioner_second_group_r_deg."""
    angles = [float(g.get("positioner_r_deg") or 0.0) for g in plan_sequence]
    rotated = [a for a in angles if abs(a) > 1e-9]
    if not rotated or len(rotated) == len(angles):
        return
    initial = run_meta.get("initial_positioner_r_deg")
    if initial is None:
        initial = rotated[0] - float(run_meta.get("positioner_second_group_r_deg", 180.0) or 180.0)
    initial = float(initial)
    if abs(initial) < 1e-9:
        return
    for group in plan_sequence:
        if abs(float(group.get("positioner_r_deg") or 0.0)) <= 1e-9:
            group["positioner_r_deg"] = initial


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
