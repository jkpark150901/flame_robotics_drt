"""Solve the goal IK of every inspection target of a planning snapshot once and store it in the snapshot.

Each target group gets "ik_goal_q" = {robot_name: [q...]} (and "ik_goal_info" = {robot_name: {...}}), plus
"ik_goal_q_pre_apf" = the same goal before the APF null-space gradient descent (ik_goal_info[...]["apf_goal_refine"]
has the cost / clearance before and after) for comparison. Planning
(benchmark_path_planners._run_once / evaluate_path_plans.py, SimTool's InspectionSequencer) then passes it as
"preset_goal_q" and uses it as the goal instead of solving the IK again - the goal posture is fixed by this step,
the path planner/optimizer only shape the path to it.

The IK runs exactly like it does inside planning (Visualizer._plan_inspection_path_for_robot with "ik_only": same
collision scene incl. the rotated pipe for phase-2 groups, same IK seed from the start posture, collision-free IK with
the APF goal refinement, path_planning.cfg inspection_ik), in planning order: phase 1 then phase 2 (positioner rotation:
pipe AND positioner rotated - the rotated positioner mesh is stored in the snapshot for planning as well),
each robot starting from the snapshot posture and then from its previous target's goal - what the planning sequence
does when every target succeeds.

    python python/precompute_ik_goals.py --snapshot sample/planning_snapshot_p1_2.pkl [--snapshot ...]

The snapshot is rewritten in place; the original is kept once as <name>.noik.pkl.
"""
import argparse
import copy
import pathlib
import pickle
import shutil
import sys
import time

import numpy as np

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_PATH / "python"))

import vedo  # noqa: E402

_Plotter = vedo.Plotter


class _OffscreenPlotter(_Plotter):   # the rotated positioner mesh is built with a (headless) Visualizer
    def __init__(self, *args, **kwargs):
        kwargs["offscreen"] = True
        super().__init__(*args, **kwargs)


vedo.Plotter = _OffscreenPlotter

import benchmark_path_planners as bench  # noqa: E402
from common.config_loader import load_config  # noqa: E402
from util.logger.console import ConsoleLogger  # noqa: E402


def add_phase2_positioner_mesh(config, snapshot, snapshot_path, console):
    """Store the positioner collision mesh at the phase-2 angle (initial r + positioner_second_group_r_deg) in the
    snapshot (positioner_vertices_phase2 / _triangles_phase2) - built by the Visualizer exactly like SimTool's
    pre-phase-2 rotation does; checked against the snapshot's own mesh at the initial angle first."""
    from viewervedo.visualizer import Visualizer
    vcfg = dict(config)
    vcfg["app_path"] = ROOT_PATH / "python" / "viewervedo"
    viewer = Visualizer(config=vcfg)
    viewer._handle_request_load_planning_snapshot({"path": str(snapshot_path)})
    m0 = viewer._build_positioner_collision_mesh()
    saved = np.asarray(snapshot.get("positioner_vertices"), dtype=float)
    if saved.shape != np.asarray(m0.vertices).shape or np.max(np.abs(saved - np.asarray(m0.vertices))) > 1e-6:
        raise RuntimeError("Visualizer positioner mesh does not match the snapshot's - cannot build the phase-2 mesh")
    r0 = float(snapshot.get("positioner_r_deg", 0.0))
    delta = float((config.get("path_planning", {}) or {}).get("positioner_second_group_r_deg", 180.0))
    viewer._move_positioner_r_to(r0 + delta, visualize_verification=False)
    m1 = viewer._build_positioner_collision_mesh()
    snapshot["positioner_vertices_phase2"] = np.asarray(m1.vertices, dtype=float)
    snapshot["positioner_triangles_phase2"] = np.asarray(m1.triangles, dtype=np.int64)
    snapshot["positioner_r_deg_phase2"] = r0 + delta
    console.info(f"phase-2 positioner mesh at r={r0 + delta:.1f} deg stored (max vertex move "
                 f"{np.max(np.abs(np.asarray(m1.vertices) - saved)):.3f} m)")


def solve_snapshot(config, snapshot, console):
    from robot_core.worker import RobotCoreEngine
    from plugins.robotics.inspection_workflow import inspection_group_pose_items, partition_and_sort_target_groups

    engine = RobotCoreEngine(config, snapshot)
    groups = snapshot.get("target_groups") or []
    phases = partition_and_sort_target_groups(
        groups, rt_pipe_facing_axis=snapshot.get("rt_pipe_facing_axis", (0.0, -1.0, 0.0)))
    rotation_T = snapshot.get("second_group_rotation_T")
    by_name = {g.get("name"): g for g in groups}
    start_q_by_robot = {}
    rows = []

    def start_q(robot_name):
        if robot_name not in start_q_by_robot:
            model = engine._find_robot(robot_name)
            bm = engine._robotics_backend.robot_model(robot_name)
            start_q_by_robot[robot_name] = engine._current_robot_q(model, bm, robot_name=robot_name).tolist()
        return start_q_by_robot[robot_name]

    for phase_no, phase in enumerate(phases[:2]):
        rotate = phase_no == 1
        if rotate and phase.get("groups") and not bool(snapshot.get("spool_fix_r", False)):
            console.warning("phase-2 groups skipped: spool not fixed to the chuck (planning skips them too)")
            continue
        if rotate and bool((config.get("path_planning", {}) or {}).get("reset_pose_before_rotation", False)):
            # planning retreats every robot before the positioner rotates (benchmark_path_planners._retreat_before_
            # rotation: track kept, other joints 0) - the first phase-2 target of each robot starts from there, so its
            # goal ("nearest to the start") is chosen from there too
            from plugins.robotics.inspection_workflow import zero_non_linear_track_joints
            for robot_name in list(start_q_by_robot):
                start_q_by_robot[robot_name] = list(zero_non_linear_track_joints(
                    start_q_by_robot[robot_name], engine._robot_joint_names(robot_name, None)))
        obstacle = engine._current_spool_collision_mesh()
        if rotate and rotation_T is not None:
            obstacle = copy.deepcopy(obstacle)
            obstacle.transform(np.asarray(rotation_T, dtype=float))
            bench.apply_phase2_positioner(engine, snapshot, console)   # positioner rotated too, not only the pipe
        for group in phase.get("groups", []):
            target_group = by_name.get(group.get("name"), group)
            for robot_name, pose_name, target_pose in inspection_group_pose_items(group):
                label = f"{group.get('name')}:{pose_name}"
                request = {
                    "robot_name": robot_name, "planner": "rrt_connect", "step_size": 0.1, "max_iter": 3000,
                    "ik_solver": (config.get("inspection_ik", {}) or {}).get("solver", "collision_free"),
                    "context_label": label, "ik_only": True,
                    "_start_q_override_by_robot": {robot_name: start_q(robot_name)},
                }
                t0 = time.perf_counter()
                try:
                    res = engine._plan_inspection_path_for_robot(
                        request, robot_name, np.asarray(target_pose, dtype=float), obstacle, context_label=label)
                except Exception as exc:
                    console.error(f"[IK] {label} ({robot_name}) failed: {exc}")
                    rows.append((label, robot_name, False, None, time.perf_counter() - t0))
                    continue
                elapsed = time.perf_counter() - t0
                ok = not res.get("ik_fallback") and bool(res.get("collision_free"))
                goal_q = [float(v) for v in np.asarray(res["goal_q"], dtype=float)]
                summary = res.get("ik_result") or {}
                info = {
                    "success": ok, "collision_free": bool(res.get("collision_free")),
                    "ik_fallback": bool(res.get("ik_fallback")), "solver": res.get("ik_solver"),
                    "position_error": summary.get("position_error"), "orientation_error": summary.get("orientation_error"),
                    "collision_pairs": [list(p) for p in summary.get("collision_pairs") or []],
                    "start_q": [float(v) for v in start_q(robot_name)], "phase": phase_no + 1, "elapsed_s": elapsed,
                }
                target_group.setdefault("ik_goal_q", {})
                target_group.setdefault("ik_goal_info", {})
                apf = ((res.get("ik_extra") or {}).get("apf_goal_refine")) or {}
                if apf:
                    # APF null-space gradient descent of the goal (collision_free_ik.apf_goal_refine): before / after
                    info["apf_goal_refine"] = {k: apf.get(k) for k in (
                        "cost_before", "cost_after", "clearance_before", "clearance_after", "accepted", "iterations",
                        "joint_change_max", "time_s", "locked_links")}
                if ok:
                    target_group["ik_goal_q"][robot_name] = goal_q
                    # goal before the APF gradient descent (= goal_q when no step was accepted) - for comparison
                    target_group.setdefault("ik_goal_q_pre_apf", {})[robot_name] = (
                        [float(v) for v in apf["q_before"]] if apf.get("q_before") is not None else goal_q)
                    start_q_by_robot[robot_name] = goal_q   # next target of this robot starts here
                else:
                    target_group["ik_goal_q"].pop(robot_name, None)   # planning solves it itself
                    info["fallback_q"] = goal_q   # least-bad solution, for display only (Viewer: saved IK "failed only")
                target_group["ik_goal_info"][robot_name] = info
                rows.append((label, robot_name, ok, info, elapsed))
                console.info(f"[IK] {label} ({robot_name}): {'OK' if ok else 'FAILED'} in {elapsed:.1f}s")
    return rows


def fill_fallback(config, snapshot, console):
    """Failed targets of an already precomputed snapshot get "fallback_q" (re-solved from their stored start_q, same
    scene/phase); successful goals are not touched."""
    from robot_core.worker import RobotCoreEngine
    from plugins.robotics.inspection_workflow import inspection_group_pose_items, partition_and_sort_target_groups

    engine = RobotCoreEngine(config, snapshot)
    groups = snapshot.get("target_groups") or []
    phases = partition_and_sort_target_groups(
        groups, rt_pipe_facing_axis=snapshot.get("rt_pipe_facing_axis", (0.0, -1.0, 0.0)))
    rotation_T = snapshot.get("second_group_rotation_T")
    by_name = {g.get("name"): g for g in groups}
    rows = []
    for phase_no, phase in enumerate(phases[:2]):
        obstacle = engine._current_spool_collision_mesh()
        if phase_no == 1 and rotation_T is not None:
            obstacle = copy.deepcopy(obstacle)
            obstacle.transform(np.asarray(rotation_T, dtype=float))
        for group in phase.get("groups", []):
            target_group = by_name.get(group.get("name"), group)
            for robot_name, pose_name, target_pose in inspection_group_pose_items(group):
                info = (target_group.get("ik_goal_info") or {}).get(robot_name)
                if not info or info.get("success") or info.get("fallback_q") is not None:
                    continue
                label = f"{group.get('name')}:{pose_name}"
                request = {
                    "robot_name": robot_name, "planner": "rrt_connect", "step_size": 0.1, "max_iter": 3000,
                    "ik_solver": (config.get("inspection_ik", {}) or {}).get("solver", "collision_free"),
                    "context_label": label, "ik_only": True,
                    "_start_q_override_by_robot": {robot_name: list(info["start_q"])},
                }
                res = engine._plan_inspection_path_for_robot(
                    request, robot_name, np.asarray(target_pose, dtype=float), obstacle, context_label=label)
                info["fallback_q"] = [float(v) for v in np.asarray(res["goal_q"], dtype=float)]
                info["fallback_collision_pairs"] = [
                    list(p) for p in (res.get("ik_result") or {}).get("collision_pairs") or []]
                rows.append((label, robot_name, info["fallback_collision_pairs"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", action="append", required=True)
    ap.add_argument("--config", default=str(ROOT_PATH / "python" / "viewervedo.cfg"))
    ap.add_argument("--verbose-level", default="WARNING")
    ap.add_argument("--fill-fallback", action="store_true",
                    help="only add fallback_q (display) to the failed targets of an already precomputed snapshot")
    args = ap.parse_args()

    config = load_config(args.config)
    extra = pathlib.Path(args.config).resolve().parent / "path_planning.cfg"
    if extra.exists():
        config.update(load_config(extra))
    config["root_path"] = ROOT_PATH
    config["verbose_level"] = args.verbose_level.upper()
    ConsoleLogger.configure(config.get("logging", {}) or {}, force=True)
    console = ConsoleLogger.get_logger()

    for snap_path in args.snapshot:
        path = pathlib.Path(snap_path).resolve()
        if args.fill_fallback:
            snapshot = bench._load_snapshot(str(path), console)
            rows = fill_fallback(config, snapshot, console)
            with open(path, "wb") as f:
                pickle.dump(snapshot, f)
            print(f"== {path.name}: fallback_q added for {len(rows)} failed target(s)")
            for label, robot, pairs in rows:
                print(f"   {label:40s} {robot:16s} collision={pairs}")
            continue
        backup = path.with_name(path.stem + ".noik.pkl")
        if not backup.exists():
            shutil.copyfile(path, backup)
        snapshot = bench._load_snapshot(str(backup), console)
        snapshot = copy.deepcopy(snapshot)
        t0 = time.perf_counter()
        add_phase2_positioner_mesh(config, snapshot, backup, console)
        rows = solve_snapshot(config, snapshot, console)
        ik_cfg = config.get("inspection_ik", {}) or {}
        snapshot["ik_goals_meta"] = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "tool": "precompute_ik_goals.py",
            "start": "snapshot robot_joint_states, then the previous goal of the same robot",
            "inspection_ik": {k: v for k, v in ik_cfg.items() if not isinstance(v, dict) or k == "ik_goal_joint_weights"},
            "n_targets": len(rows), "n_ok": sum(1 for r in rows if r[2]), "elapsed_s": time.perf_counter() - t0,
        }
        with open(path, "wb") as f:
            pickle.dump(snapshot, f)
        print(f"== {path.name}: {snapshot['ik_goals_meta']['n_ok']}/{len(rows)} goals stored "
              f"({snapshot['ik_goals_meta']['elapsed_s']:.0f}s), original kept as {backup.name}")
        for label, robot, ok, info, elapsed in rows:
            extra_txt = "" if ok or info is None else f" collision={info.get('collision_pairs')} fallback={info.get('ik_fallback')}"
            print(f"   {'OK ' if ok else 'NG '} {label:40s} {robot:16s} {elapsed:6.1f}s{extra_txt}")


if __name__ == "__main__":
    main()
