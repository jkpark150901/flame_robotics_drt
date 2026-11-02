"""Save a multi-target path-planning run to disk, including why each failed target failed.

A "plan run" folder has the same shape benchmark_path_planners.py --save-paths and
test_ompl_planning.py --target all already write (so SimTool's "Load Playback Result..." /
simtool.playback_loader read it unchanged), plus failure information:

    <run_dir>/
        summary.csv                    one row per target (success AND failed, even with no path)
        failures.json                  every non-success target with category + detail
        run.json                       run-level metadata (planner, options, counts, spool, ...)
        NN_<robot>_<pose>/
            joint_states.csv           q_path (written whenever the planner returned one)
            collisions.json            colliding edges/pairs, if any
            result.json                full per-target result (failures only)

Targets with no q_path (IK failure, timeout, planner never found anything) used to leave no
trace on disk at all; here they still get a summary.csv row and a failures.json entry.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SUMMARY_FIELDS = [
    "index", "group_name", "robot_name", "pose_name", "status", "message", "n_waypoints",
    "iterations", "max_iter", "solve_time", "iteration_ptc_error", "positioner_r_deg",
    "status_kind", "failure_category", "elapsed",
    "planning_time_s", "request_total", "clearance_min_m", "clearance_margin_m", "physical_min_distance_m",
    "smoothness", "n_clearance_samples", "ee_path_length_m", "joint_path_length_normalized",
    "estimated_motion_s", "estimated_ee_mean_speed_m_s", "joint_velocity_limit_deg_s",
    "linear_velocity_limit_m_s", "path_metrics_error",
]

# (substring in the failure message, category) - first match wins. Messages come from
# path_planning_service.plan_single_target / the OMPL planner bases.
_CATEGORY_RULES = (
    ("stopped by user", "cancelled"),                  # SimTool "Stop Planning" (InspectionSequencer.cancel)
    ("cancelled by user", "cancelled"),
    ("start_collision", "start_collision"),
    ("goal_collision", "goal_collision"),
    ("goal_not_connected", "goal_not_connected"),
    ("final_verification_failed", "final_verification_failed"),
    ("iteration_cap_reached", "iteration_cap_reached"),
    ("timeout_approximate", "timeout_approximate"),   # budget/iteration cap hit with only an approximate path
    ("ik_fallback", "ik_failure"),                     # no usable IK goal (did not converge / collides)
    ("timeout", "timeout"),
    ("timed out", "timeout"),
    ("ik_failure", "ik_failure"),
    ("ik failed", "ik_failure"),
    ("inverse kinematics", "ik_failure"),
    ("planner_failed", "planner_failed"),
    ("planning failed", "planning_failed"),
    ("not installed", "environment_error"),
    ("collision scene is not available", "environment_error"),
)


def classify_failure(result: Dict[str, Any]) -> str:
    """Short machine-friendly category for a non-success target ("" for success)."""
    if result.get("status") == "success":
        return ""
    if result.get("status") == "cancelled":
        return "cancelled"
    if result.get("ik_failure"):
        return "ik_failure"
    message = str(result.get("message") or "").lower()
    for needle, category in _CATEGORY_RULES:
        if needle in message:
            return category
    if result.get("collision_preview") or result.get("edge_collisions"):
        return "path_collision"
    return "unknown"


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _write_joint_states_csv(path: Path, joint_names: Optional[List[str]], q_path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dof = len(q_path[0]) if q_path else 0
    names = list(joint_names) if joint_names and len(joint_names) == dof else [f"joint_{i}" for i in range(dof)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["waypoint", *names])
        for i, q in enumerate(q_path):
            writer.writerow([i, *[float(v) for v in q]])


def default_run_dir(root: Any, planner: str, label: str = "") -> Path:
    """<root>/<YYYYmmdd_HHMMSS>_<planner>[_label]"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [stamp, str(planner or "planner")] + ([label] if label else [])
    return Path(root) / "_".join(parts)


def save_plan_run(
    run_dir: Any,
    results: Iterable[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
    joint_names_by_robot: Optional[Dict[str, List[str]]] = None,
    unreached_groups: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Write one plan run. Each result dict needs group_name/robot (or robot_name)/pose_name/status;
    optional: message, status_kind, q_path, positioner_r_deg, elapsed, planner_stats, verification,
    edge_collisions, collision_preview, ik_failure.

    Returns {"run_dir", "n_total", "n_success", "n_failed", "failures": [...]}.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    joint_names_by_robot = joint_names_by_robot or {}

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for index, result in enumerate(results):
        robot = result.get("robot_name") or result.get("robot") or ""
        pose_name = result.get("pose_name") or ""
        status = result.get("status")
        q_path = result.get("q_path") or []
        planner_stats = result.get("planner_stats") or {}
        metrics = result.get("path_metrics") or {}
        category = classify_failure(result)
        subdir = run_dir / f"{index:02d}_{robot}_{pose_name}"

        if q_path:
            _write_joint_states_csv(subdir / "joint_states.csv", joint_names_by_robot.get(robot), q_path)
        if metrics:
            subdir.mkdir(parents=True, exist_ok=True)
            with open(subdir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(_jsonable(metrics), f, indent=2, ensure_ascii=False)
        verification = result.get("verification") or {}
        edge_collisions = result.get("edge_collisions") or verification.get("edge_collisions") or []
        if edge_collisions or result.get("collision_preview"):
            subdir.mkdir(parents=True, exist_ok=True)
            with open(subdir / "collisions.json", "w", encoding="utf-8") as f:
                json.dump({
                    "collision_preview": bool(result.get("collision_preview")),
                    "edge_collisions": _jsonable(edge_collisions),
                    "verification": _jsonable(verification),
                }, f, indent=2)

        rows.append({
            "index": index, "group_name": result.get("group_name"), "robot_name": robot,
            "pose_name": pose_name, "status": status, "message": result.get("message"),
            "n_waypoints": len(q_path), "iterations": planner_stats.get("iterations"),
            "max_iter": planner_stats.get("max_iter"), "solve_time": planner_stats.get("solve_time"),
            "iteration_ptc_error": planner_stats.get("iteration_ptc_error"),
            "positioner_r_deg": float(result.get("positioner_r_deg") or 0.0),
            "status_kind": result.get("status_kind") or ("success" if status == "success" else "failed"),
            "failure_category": category, "elapsed": result.get("elapsed"),
            "planning_time_s": result.get("elapsed"),
            "request_total": result.get("request_total"),
            "clearance_min_m": metrics.get("clearance_min_m"),
            "clearance_margin_m": metrics.get("clearance_margin_m"),
            "physical_min_distance_m": metrics.get("physical_min_distance_m"),
            "smoothness": metrics.get("smoothness"),
            "n_clearance_samples": metrics.get("clearance_samples"),
            "ee_path_length_m": metrics.get("ee_path_length_m"),
            "joint_path_length_normalized": metrics.get("joint_path_length_normalized"),
            "estimated_motion_s": metrics.get("estimated_motion_s"),
            "estimated_ee_mean_speed_m_s": metrics.get("estimated_ee_mean_speed_m_s"),
            "joint_velocity_limit_deg_s": metrics.get("joint_velocity_limit_deg_s"),
            "linear_velocity_limit_m_s": metrics.get("linear_velocity_limit_m_s"),
            "path_metrics_error": result.get("path_metrics_error"),
        })

        if status != "success":
            entry = {
                "index": index, "group_name": result.get("group_name"), "robot_name": robot,
                "pose_name": pose_name, "status": status,
                "status_kind": rows[-1]["status_kind"], "category": category,
                "message": result.get("message"), "n_waypoints": len(q_path),
                "has_path": bool(q_path), "elapsed": result.get("elapsed"),
                "positioner_r_deg": rows[-1]["positioner_r_deg"],
                "planner_stats": _jsonable(planner_stats),
                "ik_failure": _jsonable(result.get("ik_failure")),
                "n_colliding_edges": len(edge_collisions),
                "colliding_edges": _jsonable(edge_collisions[:20]),
                # the IK goal the planner was asked to reach (start = joint_states.csv row 0) and the IK diagnostics
                # (seeds, collision-free count, clearance, timing) - needed to tell an IK problem from a planner one
                "goal_q": _jsonable(result.get("goal_q")),
                "ik_goal_q": _jsonable(result.get("ik_goal_q")),
                "diag": _jsonable(result.get("diag")),
            }
            failures.append(entry)
            subdir.mkdir(parents=True, exist_ok=True)
            with open(subdir / "result.json", "w", encoding="utf-8") as f:
                json.dump({**entry, "verification": _jsonable(verification)}, f, indent=2, default=str)

    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_success = sum(1 for r in rows if r["status"] == "success")
    with open(run_dir / "failures.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_total": len(rows), "n_success": n_success, "n_failed": len(failures),
            "by_category": _count_by(failures, "category"),
            "unreached_groups": list(unreached_groups or []),
            "failures": failures,
        }, f, indent=2, ensure_ascii=False, default=str)

    run_meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_total": len(rows), "n_success": n_success, "n_failed": len(failures),
        "unreached_groups": list(unreached_groups or []),
        **_jsonable(meta or {}),
    }
    evaluated = [r for r in rows if r["status"] == "success" and r.get("estimated_motion_s") is not None]
    if evaluated:
        run_meta["evaluation_summary"] = {
            "n_evaluated": len(evaluated),
            "minimum_clearance_margin_m": min(
                (r["clearance_margin_m"] for r in evaluated if r["clearance_margin_m"] is not None),
                default=None),
            "mean_smoothness": sum(r["smoothness"] for r in evaluated) / len(evaluated),
            "total_waypoints": sum(r["n_waypoints"] for r in evaluated),
            "estimated_motion_s": sum(r["estimated_motion_s"] for r in evaluated),
            "planning_time_s": sum(float(r["planning_time_s"] or 0.0) for r in rows),
        }
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False, default=str)

    return {
        "run_dir": str(run_dir), "n_total": len(rows), "n_success": n_success,
        "n_failed": len(failures), "failures": failures,
    }


def _count_by(items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        counts[str(item.get(key))] = counts.get(str(item.get(key)), 0) + 1
    return counts


def load_run_info(run_dir: Any) -> Dict[str, Any]:
    """Read failures.json / run.json (either may be missing for runs saved by older scripts)."""
    run_dir = Path(run_dir)
    info: Dict[str, Any] = {"failures": [], "unreached_groups": [], "run": {}}
    for name, key in (("failures.json", None), ("run.json", "run")):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if key == "run":
            info["run"] = data
        else:
            info["failures"] = data.get("failures") or []
            info["unreached_groups"] = data.get("unreached_groups") or []
            info["by_category"] = data.get("by_category") or {}
    return info
