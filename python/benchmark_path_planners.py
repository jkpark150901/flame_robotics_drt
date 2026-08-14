"""Benchmark inspection path-planning algorithms against a saved planning snapshot.

Usage:
    python python/benchmark_path_planners.py \
        --config python/viewervedo.cfg \
        --snapshot sample/my_snapshot.pkl \
        --planners rrt,rrt_connect,rrt_star,informed_rrt_star,bit_star,prm,direct_path \
        --repeats 5 --timeout 30

    # "<planner>+<optimizer>" runs a two-stage method: generate a path with
    # <planner>, then re-optimize it with <optimizer> (same optimizer plugins
    # SimTool's optimizer combobox uses - see plugins/optimizer/). Most useful
    # with direct_path (pure straight-line interpolation, no collision
    # avoidance of its own) as the base, so the optimizer does all the actual
    # obstacle avoidance and its result is directly comparable to a "real"
    # planner's on the smoothness/clearance metrics:
    python python/benchmark_path_planners.py \
        --config python/viewervedo.cfg --snapshot sample/my_snapshot.pkl \
        --planners direct_path+stomp,direct_path+trajopt,direct_path+gpmp2,rrt_connect \
        --repeats 5 --timeout 30

The snapshot file is produced from the Viewer/SimTool UI once EF poses are
determined: click "Save Planning Snapshot" in SimTool (after "Determine EF
Pose"). That handler bundles exactly what Robot Core needs to plan headlessly
- the same payload plan_inspection_path submits to Robot Core in production:
target_groups (determined EF poses), the spool/positioner collision meshes,
and current robot joint states. See
viewervedo.visualizer.Visualizer._handle_request_save_planning_snapshot.

This script reuses the exact production planning primitives
(robot_core.worker.RobotCoreEngine + robot_core.path_planning_service.
plan_single_target, robot_core's entire planning interface as of
ROBOT_CORE_DECOUPLING_PLAN.md) so every planner is compared against the same
collision scene, URDFs (loaded from --config), and target poses - only the
"planner" name changes between runs. Target-group splitting/sorting and
positioner-rotation handling - now SimTool's job in production - are
replicated here with the same shared helpers SimTool uses
(plugins.robotics.inspection_workflow), since a benchmark script has no GUI/
ZAPI round trips to worry about and can just do it all in one process.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import pathlib
import pickle
import statistics
import sys
import time

import numpy as np

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from common.config_loader import load_config
from util.logger.console import ConsoleLogger
from plugins.pathplanner import Q_SPACE_PLANNER_MODULES
from plugins.pathplanner.ompl import SUPPORTED_ALGORITHMS as OMPL_SUPPORTED_ALGORITHMS
from plugins.optimizer import OPTIMIZER_MODULES
from plugins.optimizer import apf

# Legacy hand-written planners (always available) + native OMPL algorithms
# (only actually usable when the OMPL python bindings are installed - if not,
# those individual runs fail with a clear error row instead of crashing the
# whole benchmark; see _run_once()'s try/except).
DEFAULT_PLANNERS = tuple(sorted(Q_SPACE_PLANNER_MODULES)) + tuple(OMPL_SUPPORTED_ALGORITHMS)

# "<planner>+<optimizer>" (e.g. "direct_path+stomp") is a two-stage method:
# generate a path with <planner> first, then hand it to <optimizer>.optimize()
# for smoothing/shortcutting/collision-aware re-optimization - most useful
# with direct_path (a pure straight-line interpolator with no collision
# avoidance of its own, see plugins/pathplanner/direct_path.py) as the first
# stage, since then the optimizer is doing all the actual obstacle-avoidance
# work and its output is directly comparable to a "real" planner's path on
# the same smoothness/clearance metrics. See _parse_method() for the parsing
# and OPTIMIZER_MODULES for what's available (stomp/trajopt/gpmp2/
# path_pruning/bspline/topp_ra - no "chomp" module exists; gpmp2 and trajopt
# are the closest available to what CHOMP does).
DIRECT_PATH_OPTIMIZED_METHODS = tuple(f"direct_path+{name}" for name in sorted(OPTIMIZER_MODULES))


def _parse_method(method_name):
    """Split a "<planner>+<optimizer>" method name into (planner, optimizer).
    Plain planner names (no "+") return (method_name, None)."""
    planner_name, sep, optimizer_name = method_name.partition("+")
    return planner_name, (optimizer_name or None) if sep else None

# Failure message path_planning_service.py uses when a group needs a positioner
# rotation but the snapshot's spool_fix_r was False: the pipe was never actually
# rotated in the collision mesh, so every target in the group is short-circuited
# to "failed" *without running any planner at all*. If this fires, the group's
# result says nothing about planner quality - see _group_breakdown() below.
POLICY_BLOCKED_MARKER = "positioner_not_fixed_to_spool"


def _load_snapshot(path, console):
    with open(path, "rb") as f:
        snapshot = pickle.load(f)
    if not snapshot.get("target_groups"):
        raise ValueError(
            f"snapshot has no target_groups: {path} "
            "(save it again after EF pose determination succeeds)")
    spool_fix_r = bool(snapshot.get("spool_fix_r", False))
    has_rotation_transform = snapshot.get("second_group_rotation_T") is not None
    console.info(f"Snapshot spool_fix_r={spool_fix_r}, has_rotation_transform={has_rotation_transform}")
    if "rt_pipe_facing_axis" not in snapshot:
        console.warning(
            "Snapshot predates rt_pipe_facing_axis being saved - falling back to the default "
            "(0,-1,0). If EF pose config uses a non-default pipe_facing_axis, this snapshot's "
            "rotation-needed classification here may disagree with what was used when its "
            "dda_pose_resolved/rt_pose_resolved were computed. Re-save it to fix.")
    if not spool_fix_r:
        console.warning(
            "Snapshot was saved with spool_fix_r=False: if this case has any "
            "target that needs a positioner rotation, robot_core will short-circuit "
            f"it to failed ({POLICY_BLOCKED_MARKER}) without running any planner - "
            "check the SimTool spool 'Fix F Column R' checkbox and re-save if you "
            "want to benchmark the rotation scenario.")
    return snapshot


def _path_length(q_path):
    if not q_path or len(q_path) < 2:
        return 0.0
    arr = np.asarray(q_path, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def _path_smoothness(q_path):
    """Discrete-curvature roughness of a q-space waypoint path: mean over
    interior waypoints of ||q[i-1] - 2*q[i] + q[i+1]|| (the discrete second
    derivative, i.e. "how much the direction changes at each waypoint").
    Assumes uniform spacing between waypoints (true for a raw planner
    output before any time-parameterization) - this is the path-only
    equivalent of a jerk/roughness metric. Lower = smoother; 0.0 = perfectly
    straight line (or path too short to have an interior point)."""
    if not q_path or len(q_path) < 3:
        return 0.0
    arr = np.asarray(q_path, dtype=float)
    second_diff = arr[:-2] - 2.0 * arr[1:-1] + arr[2:]
    return float(np.mean(np.linalg.norm(second_diff, axis=1)))


def _path_min_clearance(engine, robot_name, q_path):
    """Minimum link-to-obstacle distance over every waypoint of q_path, using
    the collision scene plan_single_target just configured for this robot
    (PinocchioRoboticsBackend.link_obstacle_distances - see its docstring).
    Only checks waypoints, not sampled edge midpoints, so this can miss a
    closer approach mid-edge - path_length/smoothness have the same
    waypoint-only limitation, this is consistent with those rather than a
    new gap.

    Returns (min_distance, link, obstacle) for the closest approach across
    the whole path, or (None, None, None) if the backend doesn't support
    distance queries or q_path is empty.
    """
    backend = getattr(engine, "_robotics_backend", None)
    if backend is None or not q_path:
        return None, None, None
    best = None
    for q in q_path:
        for entry in backend.link_obstacle_distances(robot_name, q):
            if best is None or entry["distance"] < best["distance"]:
                best = entry
    if best is None:
        return None, None, None
    return best["distance"], best["link"], best["obstacle"]


def _path_max_apf_cost(engine, robot_name, q_path, d0, eta):
    """Worst-case (maximum) APF repulsive cost over every waypoint of
    q_path, using the same potential function gpmp2.py/trajopt.py actually
    optimize against and apf_heatmap.py visualizes (plugins/optimizer/apf.py)
    - not just a raw distance number like _path_min_clearance, but the same
    "how much would the optimizer be penalized here" cost. Reuses the
    collision scene plan_single_target just configured for this robot, same
    as _path_min_clearance (already base_T-corrected by real planning - see
    visualizer.py:3780-3797's _base_frame_collision_mesh - so this is safe to
    call directly without redoing that correction here).

    Returns the worst waypoint's (cost, min_distance_at_that_waypoint) or
    (None, None) if the backend doesn't support distance queries or q_path is
    empty."""
    backend = getattr(engine, "_robotics_backend", None)
    if backend is None or not q_path:
        return None, None
    worst_cost, worst_min_dist = None, None
    for q in q_path:
        entries = backend.link_obstacle_distances(robot_name, q)
        cost = apf.repulsive_cost(entries, d0, eta)
        if worst_cost is None or cost > worst_cost:
            worst_cost = cost
            worst_min_dist = apf.min_distance(entries)
    return worst_cost, worst_min_dist


def _joint_names_for(engine, robot_name, dof):
    backend = getattr(engine, "_robotics_backend", None)
    robot_backend_model = backend.robot_model(robot_name) if backend is not None else None
    names = engine._robot_joint_names(robot_name, robot_backend_model)
    return [str(n) for n in names] if names else [f"joint_{i}" for i in range(dof)]


def _save_joint_states_csv(path, joint_names, q_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["waypoint", *joint_names])
        for i, q in enumerate(q_path):
            writer.writerow([i, *[float(v) for v in q]])


def _write_path_summary_csv(save_paths_dir, path_summary_rows):
    """Write save_paths_dir/summary.csv in exactly test_ompl_planning.py's
    --target all schema, so SimTool's "Load Playback Result" (playback_loader.py)
    can load a benchmark run's saved paths identically to a test script run -
    no format-specific branching needed on the consumer side."""
    save_paths_dir.mkdir(parents=True, exist_ok=True)
    with open(save_paths_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "group_name", "robot_name", "pose_name", "status", "message", "n_waypoints",
            "iterations", "max_iter", "solve_time", "iteration_ptc_error"])
        writer.writeheader()
        writer.writerows(path_summary_rows)


def _run_once(config, snapshot, planner_name, *, timeout, seed, step_size=0.08, save_paths_dir=None,
              apf_d0=apf.DEFAULT_D0, apf_eta=apf.DEFAULT_ETA):
    """Plan every target in the snapshot (both phases) with a fresh engine,
    one plan_single_target call per target - mirroring exactly what
    SimTool's InspectionSequencer does, plus the positioner-rotation phase it
    currently defers (safe to do here since this script has no GUI/ZAPI
    round-trip choreography to manage).

    planner_name may be a two-stage "<planner>+<optimizer>" method (see
    _parse_method/DIRECT_PATH_OPTIMIZED_METHODS) - the base planner generates
    the path and the optimizer re-optimizes it, exactly like SimTool's
    optimizer combobox does for a live plan_single_target request
    (visualizer.py._apply_path_optimizer); this script just has to pass the
    same "optimizer"/"optimize_path" request fields through.
    """
    from robot_core.worker import RobotCoreEngine
    from robot_core.path_planning_service import plan_single_target
    from plugins.robotics.inspection_workflow import (
        inspection_group_pose_items, partition_and_sort_target_groups,
        zero_non_linear_track_joints)

    base_planner_name, optimizer_name = _parse_method(planner_name)
    np.random.seed(seed)
    wall_t0 = time.perf_counter()
    engine = RobotCoreEngine(config, snapshot)
    target_groups = snapshot.get("target_groups") or []
    # Must match the axis actually used to produce dda_pose_resolved/
    # rt_pose_resolved at snapshot-save time (see visualizer.py's
    # _inspection_robot_core_snapshot) - recomputing reachability with a
    # different axis than what was used to resolve the poses would silently
    # mis-classify which groups are "already rotated" vs "still need it".
    rt_pipe_facing_axis = snapshot.get("rt_pipe_facing_axis", (0.0, -1.0, 0.0))
    phases = partition_and_sort_target_groups(target_groups, rt_pipe_facing_axis=rt_pipe_facing_axis)
    spool_fix_r = bool(snapshot.get("spool_fix_r", False))
    rotation_T = snapshot.get("second_group_rotation_T")
    initial_r_deg = float(snapshot.get("positioner_r_deg", 0.0))

    start_q_by_robot = {}
    groups = []
    target_rows = []
    path_summary_rows = []  # only populated/written if save_paths_dir is given
    target_index = 0
    total_elapsed = 0.0
    # Matches how the snapshot's second_group_rotation_T was actually computed
    # at save time (Visualizer._inspection_robot_core_snapshot always used the
    # config default, since "Save Planning Snapshot" doesn't take a custom
    # delta). Used only as a display/grouping label here.
    second_r_deg = initial_r_deg + float(
        (config.get("path_planning", {}) or {}).get("positioner_second_group_r_deg", 180.0))

    def _start_q(robot_name):
        if robot_name in start_q_by_robot:
            return start_q_by_robot[robot_name]
        model = engine._find_robot(robot_name)
        backend = getattr(engine, "_robotics_backend", None)
        robot_backend_model = backend.robot_model(robot_name) if backend is not None else None
        return engine._current_robot_q(model, robot_backend_model, robot_name=robot_name).tolist()

    def _retreat_before_rotation(robot_name):
        """Before the positioner actually rotates, the robot must be at a
        known-safe posture (arm folded to its zero pose) rather than wherever
        the last pre-rotation target left it - an arbitrary pose with no
        guaranteed clearance once the pipe/positioner rotates under it. Only
        the linear track (which doesn't move with the rotation) is kept."""
        backend = getattr(engine, "_robotics_backend", None)
        robot_backend_model = backend.robot_model(robot_name) if backend is not None else None
        joint_names = engine._robot_joint_names(robot_name, robot_backend_model)
        start_q_by_robot[robot_name] = zero_non_linear_track_joints(_start_q(robot_name), joint_names)

    def _plan_group(group, *, rotate, r_deg_label):
        nonlocal total_elapsed, target_index
        n_robots_planned = 0
        n_robot_failures = 0
        n_ik_failures = 0
        path_length = 0.0
        # inspection_group_pose_items() returns the *resolved* pose (rotation
        # already baked in by resolve_target_groups_with_rotation() at
        # snapshot-save time) - no need to reapply rotation_T to the pose here.
        # The collision obstacle (pipe) still has to be rotated to match, or
        # start/goal collision is checked against the wrong pipe position -
        # plan_single_target does that itself given obstacle_rotation_T.
        for robot_name, pose_name, target_pose in inspection_group_pose_items(group):
            output = plan_single_target(engine, {
                "robot_name": robot_name,
                "start_q": _start_q(robot_name),
                "target_pose": target_pose,
                "planner": base_planner_name,
                "step_size": step_size,
                "optimizer": optimizer_name,
                "optimize_path": bool(optimizer_name),
                "planning_timeout": timeout,
                "context_label": f"{group.get('name')}:{pose_name}",
                "obstacle_rotation_T": rotation_T if rotate else None,
            })
            result = output.get("result", {})
            total_elapsed += float(result.get("elapsed", 0.0))
            q_path = output.get("q_path") or []
            target_row = {
                "group_name": group.get("name"), "robot": robot_name, "pose_name": pose_name,
                "positioner_r_deg": r_deg_label, "status": result.get("status"),
                "smoothness": None, "min_clearance": None,
                "min_clearance_link": None, "min_clearance_obstacle": None,
                "apf_max_cost": None, "apf_max_cost_min_distance": None,
            }
            if save_paths_dir is not None and q_path:
                # Save regardless of status (success or failed) - a failed
                # optimizer/planner result (e.g. STOMP's "could not find any
                # collision-free path") still has an actual attempted q_path
                # worth playing back/inspecting (see path_planning_service.py -
                # it always returns whatever q_path it has, only the status
                # flag changes), in the exact same shape test_ompl_planning.py
                # saves so SimTool's "Load Playback Result" can load a
                # benchmark run identically to a test script run.
                subdir = save_paths_dir / f"{target_index:02d}_{robot_name}_{pose_name}"
                joint_names = _joint_names_for(engine, robot_name, len(q_path[0]))
                _save_joint_states_csv(subdir / "joint_states.csv", joint_names, q_path)
                planner_stats = result.get("planner_stats") or {}
                path_summary_rows.append({
                    "index": target_index, "group_name": group.get("name"), "robot_name": robot_name,
                    "pose_name": pose_name, "status": result.get("status"), "message": result.get("message"),
                    "n_waypoints": len(q_path), "iterations": planner_stats.get("iterations"),
                    "max_iter": planner_stats.get("max_iter"), "solve_time": planner_stats.get("solve_time"),
                    "iteration_ptc_error": planner_stats.get("iteration_ptc_error"),
                })
            target_index += 1
            if result.get("status") == "success" and q_path:
                start_q_by_robot[robot_name] = q_path[-1]
                n_robots_planned += 1
                path_length += _path_length(q_path)
                # Must run right here, before the *next* plan_single_target
                # call reconfigures this robot's collision scene for a
                # different target/obstacle rotation - link_obstacle_
                # distances() reads whatever scene is currently configured.
                target_row["smoothness"] = _path_smoothness(q_path)
                clearance, link, obstacle = _path_min_clearance(engine, robot_name, q_path)
                target_row["min_clearance"] = clearance
                target_row["min_clearance_link"] = link
                target_row["min_clearance_obstacle"] = obstacle
                apf_cost, apf_min_dist = _path_max_apf_cost(engine, robot_name, q_path, apf_d0, apf_eta)
                target_row["apf_max_cost"] = apf_cost
                target_row["apf_max_cost_min_distance"] = apf_min_dist
            else:
                n_robot_failures += 1
                if result.get("ik_failure"):
                    n_ik_failures += 1
            target_rows.append(target_row)
        n_targets = n_robots_planned + n_robot_failures
        status = (
            "success" if n_robot_failures == 0 and n_targets > 0
            else "partial" if n_robots_planned > 0
            else "failed"
        )
        groups.append({
            "group_name": group.get("name"),
            "group_index": group.get("index"),
            "positioner_r_deg": r_deg_label,
            "status": status,
            "policy_blocked": False,
            "n_robots_planned": n_robots_planned,
            "n_robot_failures": n_robot_failures,
            "n_ik_failures": n_ik_failures,
            "path_length": path_length,
        })

    try:
        for group in phases[0]["groups"]:
            _plan_group(group, rotate=False, r_deg_label=initial_r_deg)

        rotation_groups = phases[1]["groups"]
        if rotation_groups:
            if not spool_fix_r:
                # Mirrors the old path_planning_service.py policy check: without
                # spool_fix_r the pipe never actually follows the positioner, so
                # these targets are not planned at all (see POLICY_BLOCKED_MARKER).
                for group in rotation_groups:
                    items = inspection_group_pose_items(group)
                    groups.append({
                        "group_name": group.get("name"),
                        "group_index": group.get("index"),
                        "positioner_r_deg": second_r_deg,
                        "status": "blocked_policy",
                        "policy_blocked": True,
                        "n_robots_planned": 0,
                        "n_robot_failures": len(items),
                        "n_ik_failures": 0,
                        "path_length": 0.0,
                    })
            elif rotation_T is None:
                raise RuntimeError(
                    "snapshot has rotation-needed groups but no "
                    "second_group_rotation_T (re-save the snapshot)")
            else:
                # Retreat every robot that will plan against the rotated pipe
                # to a safe posture once, before the first rotation-phase
                # target - not per-group/per-target, since the positioner
                # only rotates once at this phase transition.
                retreated = set()
                for group in rotation_groups:
                    for robot_name, _pose_name, _target_T in inspection_group_pose_items(group):
                        if robot_name not in retreated:
                            _retreat_before_rotation(robot_name)
                            retreated.add(robot_name)
                for group in rotation_groups:
                    _plan_group(group, rotate=True, r_deg_label=second_r_deg)
    except Exception as exc:
        if save_paths_dir is not None:
            _write_path_summary_csv(save_paths_dir, path_summary_rows)
        _n_plans = sum(g["n_robots_planned"] for g in groups)
        _n_failures = sum(g["n_robot_failures"] for g in groups)
        row = {
            "planner": planner_name,
            "status": "error",
            "error": str(exc),
            "wall_elapsed": time.perf_counter() - wall_t0,
            "total_elapsed": total_elapsed,
            "path_length": sum(g["path_length"] for g in groups),
            "n_plans": _n_plans,
            "n_failures": _n_failures,
            "n_targets": _n_plans + _n_failures,
            "n_ik_failures": sum(g["n_ik_failures"] for g in groups),
            "n_groups_with_rotation": sum(
                1 for g in groups if abs((g["positioner_r_deg"] or 0.0) - initial_r_deg) > 1e-6),
            "mean_smoothness": None,
            "min_clearance": None,
            "max_apf_cost": None,
        }
        return row, groups, target_rows

    n_failures = sum(g["n_robot_failures"] for g in groups)
    n_ik_failures = sum(g["n_ik_failures"] for g in groups)
    status = (
        "success" if n_failures == 0 and groups
        else "partial" if any(g["n_robots_planned"] for g in groups)
        else "failed"
    )
    smoothness_values = [t["smoothness"] for t in target_rows if t["smoothness"] is not None]
    clearance_values = [t["min_clearance"] for t in target_rows if t["min_clearance"] is not None]
    apf_cost_values = [t["apf_max_cost"] for t in target_rows if t["apf_max_cost"] is not None]
    n_plans = sum(g["n_robots_planned"] for g in groups)
    row = {
        "planner": planner_name,
        "status": status,
        "error": None,
        "wall_elapsed": time.perf_counter() - wall_t0,
        "total_elapsed": total_elapsed,
        "path_length": sum(g["path_length"] for g in groups),
        "n_plans": n_plans,
        "n_failures": n_failures,
        # n_plans + n_failures = every target this run actually attempted
        # (including policy_blocked rotation targets, which are folded into
        # n_failures - see the policy_blocked branch above), i.e. the "/N" in
        # a "planned M/N targets" count.
        "n_targets": n_plans + n_failures,
        "n_ik_failures": n_ik_failures,
        "n_groups_with_rotation": sum(
            1 for g in groups if abs((g["positioner_r_deg"] or 0.0) - initial_r_deg) > 1e-6),
        "mean_smoothness": statistics.fmean(smoothness_values) if smoothness_values else None,
        # Worst case (minimum) across all planned targets in this run - the
        # number that matters for "did this run ever get uncomfortably close
        # to an obstacle", not an average that a couple of close calls could
        # hide.
        "min_clearance": min(clearance_values) if clearance_values else None,
        # Worst case (maximum) APF repulsive cost across all planned targets -
        # see _path_max_apf_cost. Complements min_clearance with the same
        # "how costly would the optimizer find this" number gpmp2.py/
        # trajopt.py actually minimize against (plugins/optimizer/apf.py).
        "max_apf_cost": max(apf_cost_values) if apf_cost_values else None,
    }
    if save_paths_dir is not None:
        _write_path_summary_csv(save_paths_dir, path_summary_rows)
    return row, groups, target_rows


def _run_once_worker(config, snapshot, planner_name, timeout, seed, step_size, save_paths_dir, apf_d0, apf_eta,
                      result_queue):
    """Subprocess entry point for _run_once_isolated - see its docstring."""
    try:
        result_queue.put(_run_once(
            config, snapshot, planner_name, timeout=timeout, seed=seed, step_size=step_size,
            save_paths_dir=save_paths_dir, apf_d0=apf_d0, apf_eta=apf_eta))
    except BaseException as exc:  # noqa: BLE001 - report it, don't let the process die silently
        result_queue.put(("__error__", str(exc)))


def _run_once_isolated(config, snapshot, planner_name, *, timeout, seed, step_size, save_paths_dir, console,
                        apf_d0=apf.DEFAULT_D0, apf_eta=apf.DEFAULT_ETA):
    """Run _run_once in a child process so a native crash in the OMPL/Pinocchio
    bindings (observed as exitcode -11/SIGSEGV in the live GUI run this script
    is meant to reproduce - see the "Robot Core process died" log line) kills
    only that one run instead of the whole benchmark and everything gathered
    so far. Mirrors robot_core.service.EmbeddedRobotCoreClient's own crash
    handling for the same reason."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_run_once_worker,
        args=(config, snapshot, planner_name, timeout, seed, step_size, save_paths_dir, apf_d0, apf_eta,
              result_queue),
        name=f"benchmark-{planner_name}",
    )
    process.start()
    # Generous grace period beyond the planning timeout itself for process
    # startup/teardown and non-planning work (collision mesh setup, etc).
    join_timeout = max(60.0, timeout * 4.0) if timeout and timeout > 0 else 300.0
    try:
        result = result_queue.get(timeout=join_timeout)
    except Exception:
        result = None
    process.join(timeout=10.0)
    if process.is_alive():
        process.terminate()
        process.join(5.0)

    crash_reason = None
    if isinstance(result, tuple) and len(result) == 2 and result[0] == "__error__":
        crash_reason = result[1]
        result = None

    if result is not None:
        return result

    exitcode = process.exitcode
    reason = crash_reason or f"subprocess crashed or timed out (exitcode={exitcode})"
    console.error(f"[{planner_name}] run isolated in subprocess failed: {reason}")
    row = {
        "planner": planner_name,
        "status": "error",
        "error": reason,
        "wall_elapsed": float(join_timeout),
        "total_elapsed": 0.0,
        "path_length": 0.0,
        "n_plans": 0,
        "n_failures": 0,
        "n_targets": 0,
        "n_ik_failures": 0,
        "n_groups_with_rotation": 0,
        "mean_smoothness": None,
        "min_clearance": None,
        "max_apf_cost": None,
    }
    return row, [], []


def run_benchmark(config, snapshot, planners, *, repeats, timeout, base_seed, step_size=0.08, save_paths_root=None,
                   apf_d0=apf.DEFAULT_D0, apf_eta=apf.DEFAULT_ETA):
    """save_paths_root: if given, every run's planned (and failed-but-attempted)
    q_paths are saved under save_paths_root/<planner>_r<repeat>/ in exactly
    test_ompl_planning.py's --target all shape (summary.csv + per-target
    <idx>_<robot>_<pose>/joint_states.csv subfolders) - loadable by
    SimTool's "Load Playback Result" with no format differences from a
    test-script run. None (default) skips saving entirely - a full
    multi-planner x multi-repeat sweep can be a lot of files."""
    console = ConsoleLogger.get_logger()
    rows = []
    group_rows = []
    target_metric_rows = []
    for planner_name in planners:
        for repeat in range(repeats):
            console.info(f"[{planner_name}] run {repeat + 1}/{repeats}...")
            save_paths_dir = (
                pathlib.Path(save_paths_root) / f"{planner_name}_r{repeat}"
                if save_paths_root else None)
            row, groups, target_rows = _run_once_isolated(
                config, snapshot, planner_name,
                timeout=timeout, seed=base_seed + repeat, step_size=step_size,
                save_paths_dir=save_paths_dir, console=console, apf_d0=apf_d0, apf_eta=apf_eta)
            row["repeat"] = repeat
            rows.append(row)
            for group in groups:
                group_rows.append({"planner": planner_name, "repeat": repeat, **group})
            for target_row in target_rows:
                target_metric_rows.append({"planner": planner_name, "repeat": repeat, **target_row})
            console.info(
                f"[{planner_name}] run {repeat + 1}/{repeats}: "
                f"status={row['status']} targets={row.get('n_plans', 0)}/{row.get('n_targets', 0)} "
                f"wall={row['wall_elapsed']:.3f}s "
                f"path_length={row['path_length']:.3f} "
                f"smoothness={row.get('mean_smoothness')} min_clearance={row.get('min_clearance')} "
                f"max_apf_cost={row.get('max_apf_cost')} "
                f"rotated_groups={row.get('n_groups_with_rotation', 0)}"
                + (f" error={row['error']}" if row.get("error") else "")
                + (f" paths_saved_to={save_paths_dir}" if save_paths_dir else ""))
            for group in groups:
                console.info(
                    f"    [{group['group_name']}] r_deg={group['positioner_r_deg']} "
                    f"status={group['status']} robots={group['n_robots_planned']} "
                    f"failures={group['n_robot_failures']} path_length={group['path_length']:.3f}")
    return rows, group_rows, target_metric_rows


def summarize(rows):
    by_planner = {}
    for row in rows:
        by_planner.setdefault(row["planner"], []).append(row)

    summary = []
    for planner_name, planner_rows in by_planner.items():
        n_runs = len(planner_rows)
        ok_rows = [r for r in planner_rows if r["status"] == "success"]
        partial_rows = [r for r in planner_rows if r["status"] == "partial"]
        wall_times = [r["wall_elapsed"] for r in planner_rows]
        path_lengths = [r["path_length"] for r in ok_rows if r["path_length"] > 0]
        smoothness_values = [r["mean_smoothness"] for r in planner_rows if r.get("mean_smoothness") is not None]
        clearance_values = [r["min_clearance"] for r in planner_rows if r.get("min_clearance") is not None]
        apf_cost_values = [r["max_apf_cost"] for r in planner_rows if r.get("max_apf_cost") is not None]
        # Target-level counts, summed across every repeat of this planner -
        # "how many individual (robot, pose) targets were actually planned
        # successfully out of how many were attempted", as opposed to
        # n_success/n_partial/n_failed above which count whole *runs* (a run
        # is "partial" if even one of its targets failed).
        n_targets_planned = sum(r.get("n_plans", 0) for r in planner_rows)
        n_targets_total = sum(r.get("n_targets", 0) for r in planner_rows)
        summary.append({
            "planner": planner_name,
            "n_runs": n_runs,
            "n_success": len(ok_rows),
            "n_partial": len(partial_rows),
            "n_failed": n_runs - len(ok_rows) - len(partial_rows),
            "success_rate": len(ok_rows) / n_runs if n_runs else 0.0,
            "n_targets_planned": n_targets_planned,
            "n_targets_total": n_targets_total,
            "target_success_rate": n_targets_planned / n_targets_total if n_targets_total else 0.0,
            "mean_wall_elapsed": statistics.fmean(wall_times) if wall_times else 0.0,
            "stdev_wall_elapsed": statistics.pstdev(wall_times) if len(wall_times) > 1 else 0.0,
            "mean_path_length": statistics.fmean(path_lengths) if path_lengths else 0.0,
            # Lower = smoother (discrete-curvature roughness, see _path_smoothness).
            "mean_smoothness": statistics.fmean(smoothness_values) if smoothness_values else None,
            # Worst-case (minimum) clearance across all runs - see _path_min_clearance.
            "min_clearance": min(clearance_values) if clearance_values else None,
            # Worst-case (maximum) APF repulsive cost across all runs - see
            # _path_max_apf_cost / plugins/optimizer/apf.py.
            "max_apf_cost": max(apf_cost_values) if apf_cost_values else None,
        })
    summary.sort(key=lambda s: (-s["success_rate"], s["mean_wall_elapsed"]))
    return summary


def summarize_by_rotation(group_rows):
    """Per-planner success rate split by whether the group needed a positioner
    rotation (positioner_r_deg != 0) - answers "does this planner get worse
    after the pipe/positioner has been rotated?"."""
    by_planner = {}
    for row in group_rows:
        by_planner.setdefault(row["planner"], []).append(row)

    summary = []
    for planner_name, planner_rows in by_planner.items():
        no_rot = [r for r in planner_rows if abs(r.get("positioner_r_deg") or 0.0) <= 1e-6]
        rot_all = [r for r in planner_rows if abs(r.get("positioner_r_deg") or 0.0) > 1e-6]
        # policy_blocked groups never actually ran a planner against a rotated
        # collision mesh (see POLICY_BLOCKED_MARKER) - excluded from the rate so
        # a "0% after rotation" doesn't get misread as a planner failure.
        rot_blocked = [r for r in rot_all if r.get("policy_blocked")]
        rot_evaluated = [r for r in rot_all if not r.get("policy_blocked")]

        def _rate(rows_subset):
            if not rows_subset:
                return None
            return sum(1 for r in rows_subset if r["status"] == "success") / len(rows_subset)

        summary.append({
            "planner": planner_name,
            "n_groups_no_rotation": len(no_rot),
            "success_rate_no_rotation": _rate(no_rot),
            "n_groups_rotation": len(rot_all),
            "n_groups_rotation_blocked": len(rot_blocked),
            "success_rate_rotation": _rate(rot_evaluated),
        })
    summary.sort(key=lambda s: s["planner"])
    return summary


def _print_rotation_summary(summary):
    if not any(s["n_groups_rotation"] for s in summary):
        print("(no positioner-rotation groups observed in this snapshot)")
        return
    n_blocked_total = sum(s["n_groups_rotation_blocked"] for s in summary)
    if n_blocked_total:
        print(
            f"!! {n_blocked_total} rotation-group run(s) were policy-blocked "
            f"({POLICY_BLOCKED_MARKER}) - snapshot's spool_fix_r was False, so the "
            "pipe was never actually rotated in the collision mesh and no planner "
            "ran against it. These are excluded from rot_rate below. Re-save the "
            "snapshot with the spool's 'Fix F Column R' checkbox enabled to "
            "actually benchmark rotation.")
    header = (
        f"{'planner':<18}{'no_rot_n':>10}{'no_rot_rate':>13}"
        f"{'rot_n':>8}{'rot_blocked':>13}{'rot_rate':>10}")
    print(header)
    print("-" * len(header))
    for row in summary:
        def _fmt(rate):
            return "-" if rate is None else f"{rate:.0%}"
        print(
            f"{row['planner']:<18}{row['n_groups_no_rotation']:>10}"
            f"{_fmt(row['success_rate_no_rotation']):>13}"
            f"{row['n_groups_rotation']:>8}{row['n_groups_rotation_blocked']:>13}"
            f"{_fmt(row['success_rate_rotation']):>10}")


def _print_summary(summary):
    header = (
        f"{'planner':<18}{'success':>9}{'partial':>9}{'failed':>8}"
        f"{'rate':>8}{'targets':>10}{'tgt_rate':>10}"
        f"{'mean_s':>10}{'stdev_s':>9}{'path_len':>10}"
        f"{'smooth':>10}{'min_clr':>10}{'max_apf':>10}")
    print(header)
    print("-" * len(header))
    for row in summary:
        def _fmt(value, spec):
            return "-" if value is None else format(value, spec)
        targets_str = f"{row['n_targets_planned']}/{row['n_targets_total']}"
        print(
            f"{row['planner']:<18}{row['n_success']:>9}{row['n_partial']:>9}"
            f"{row['n_failed']:>8}{row['success_rate']:>8.0%}"
            f"{targets_str:>10}{row['target_success_rate']:>10.0%}"
            f"{row['mean_wall_elapsed']:>10.3f}{row['stdev_wall_elapsed']:>9.3f}"
            f"{row['mean_path_length']:>10.3f}"
            f"{_fmt(row.get('mean_smoothness'), '10.4f')}"
            f"{_fmt(row.get('min_clearance'), '10.4f')}"
            f"{_fmt(row.get('max_apf_cost'), '10.4f')}")


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(pathlib.Path(__file__).with_name("viewervedo.cfg")))
    parser.add_argument("--snapshot", required=True, help="Planning snapshot .pkl saved from SimTool")
    parser.add_argument(
        "--planners", default=",".join(DEFAULT_PLANNERS),
        help=f"Comma-separated planner names. Default: {','.join(DEFAULT_PLANNERS)}")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-target planning timeout (s), 0=no limit")
    parser.add_argument(
        "--step-size", type=float, default=0.08,
        help="Planner step/range size (normalized [0,1]^N space for OMPL algorithms; also "
             "controls direct_path's waypoint spacing - see direct_path.py). Not exposed by "
             "this script before - requests silently used _plan_inspection_path_for_robot's "
             "own 0.08 fallback (visualizer.py) since no 'step_size' key was ever sent.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose-level", default="INFO")
    parser.add_argument(
        "--output", default=None,
        help="Output CSV prefix (writes <prefix>_runs.csv and <prefix>_summary.csv). "
             "Default: benchmark_<snapshot-stem>_<timestamp>")
    parser.add_argument(
        "--apf-d0", type=float, default=apf.DEFAULT_D0,
        help="APF influence distance (m) for the max_apf_cost metric - see plugins/optimizer/apf.py. "
             f"Default: {apf.DEFAULT_D0}")
    parser.add_argument(
        "--apf-eta", type=float, default=apf.DEFAULT_ETA,
        help=f"APF repulsive gain for the max_apf_cost metric. Default: {apf.DEFAULT_ETA}")
    parser.add_argument(
        "--save-paths", action="store_true",
        help="Also save each run's q_path (successful or failed - failures keep whatever "
             "path was actually attempted, e.g. STOMP's best-effort-but-still-colliding "
             "result) under <output-prefix>_paths/<planner>_r<repeat>/, in exactly "
             "test_ompl_planning.py's --target all shape (summary.csv + per-target "
             "joint_states.csv) - loadable by SimTool's 'Load Playback Result' with no "
             "format differences. Off by default: a full multi-planner x multi-repeat "
             "sweep can be a lot of files.")
    args = parser.parse_args()

    config = load_config(args.config)
    extra_config_path = pathlib.Path(args.config).resolve().parent / "path_planning.cfg"
    if extra_config_path.exists():
        config.update(load_config(extra_config_path))
    config["root_path"] = ROOT_PATH
    config["verbose_level"] = args.verbose_level.upper()
    ConsoleLogger.configure(config.get("logging", {}) or {}, force=True)
    console = ConsoleLogger.get_logger()

    snapshot = _load_snapshot(args.snapshot, console)
    planners = [p.strip() for p in args.planners.split(",") if p.strip()]
    console.info(
        f"Benchmarking {len(planners)} planner(s) x {args.repeats} repeat(s): {planners}")
    console.info(
        f"Snapshot: {args.snapshot} "
        f"({len(snapshot.get('target_groups') or [])} target group(s))")

    output_prefix = args.output or (
        f"benchmark_{pathlib.Path(args.snapshot).stem}_{time.strftime('%Y%m%d_%H%M%S')}")
    save_paths_root = f"{output_prefix}_paths" if args.save_paths else None

    rows, group_rows, target_metric_rows = run_benchmark(
        config, snapshot, planners,
        repeats=args.repeats, timeout=args.timeout, base_seed=args.seed, step_size=args.step_size,
        save_paths_root=save_paths_root, apf_d0=args.apf_d0, apf_eta=args.apf_eta)
    summary = summarize(rows)
    rotation_summary = summarize_by_rotation(group_rows)

    print()
    _print_summary(summary)
    print()
    print("-- by positioner rotation --")
    _print_rotation_summary(rotation_summary)

    runs_path = pathlib.Path(f"{output_prefix}_runs.csv")
    groups_path = pathlib.Path(f"{output_prefix}_groups.csv")
    summary_path = pathlib.Path(f"{output_prefix}_summary.csv")
    rotation_summary_path = pathlib.Path(f"{output_prefix}_rotation_summary.csv")
    target_metrics_path = pathlib.Path(f"{output_prefix}_target_metrics.csv")
    _write_csv(runs_path, rows, fieldnames=[
        "planner", "repeat", "status", "wall_elapsed", "total_elapsed",
        "path_length", "n_plans", "n_failures", "n_targets", "n_ik_failures",
        "n_groups_with_rotation", "mean_smoothness", "min_clearance", "max_apf_cost", "error",
    ])
    _write_csv(groups_path, group_rows, fieldnames=[
        "planner", "repeat", "group_index", "group_name", "positioner_r_deg",
        "status", "policy_blocked", "n_robots_planned", "n_robot_failures",
        "n_ik_failures", "path_length",
    ])
    _write_csv(summary_path, summary, fieldnames=[
        "planner", "n_runs", "n_success", "n_partial", "n_failed",
        "success_rate", "n_targets_planned", "n_targets_total", "target_success_rate",
        "mean_wall_elapsed", "stdev_wall_elapsed", "mean_path_length",
        "mean_smoothness", "min_clearance", "max_apf_cost",
    ])
    _write_csv(rotation_summary_path, rotation_summary, fieldnames=[
        "planner", "n_groups_no_rotation", "success_rate_no_rotation",
        "n_groups_rotation", "n_groups_rotation_blocked", "success_rate_rotation",
    ])
    # Per-(planner, repeat, target) smoothness/clearance breakdown - the
    # summary CSV only has the mean/worst-case per run; this is what lets you
    # find *which* target/robot was the closest call or the roughest path.
    _write_csv(target_metrics_path, target_metric_rows, fieldnames=[
        "planner", "repeat", "group_name", "robot", "pose_name", "positioner_r_deg",
        "status", "smoothness", "min_clearance", "min_clearance_link", "min_clearance_obstacle",
        "apf_max_cost", "apf_max_cost_min_distance",
    ])
    console.info(
        f"Wrote {runs_path}, {groups_path}, {summary_path}, {rotation_summary_path}, "
        f"{target_metrics_path}")


if __name__ == "__main__":
    main()
