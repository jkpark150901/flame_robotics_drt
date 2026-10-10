"""Single- or full-sequence OMPL planning test against a saved planning snapshot.

Loads a snapshot (.pkl) saved from SimTool's "Save Planning Snapshot" button
and runs plan_single_target - the same primitive robot_core actually executes
- either for one (robot, pose) target or, by default, for every target in the
snapshot in order (chaining each robot's start_q from its own previous
target, exactly like SimTool's InspectionSequencer/the benchmark script).
Unlike benchmark_path_planners.py (many planners x many repeats, subprocess-
isolated so one crash doesn't kill the whole run), this script runs
everything in-process for a single planner and prints full diagnostics, so
it's meant for interactively poking at one planner across the whole pose set
- e.g. reproducing the "AORRTC: Zero-length path"/segfault issue with maximum
log verbosity, optionally under a debugger.

Usage:
    python python/test_ompl_planning.py \
        --config python/viewervedo.cfg \
        --snapshot sample/my_snapshot.pkl \
        --planner RRTConnect \
        --list                      # just list available targets and exit
    python python/test_ompl_planning.py \
        --config python/viewervedo.cfg \
        --snapshot sample/my_snapshot.pkl \
        --planner AORRTC --step-size 0.1 --max-iter 5000 --timeout 30
        # plans every target in the snapshot, in order (default)
    python python/test_ompl_planning.py \
        --config python/viewervedo.cfg \
        --snapshot sample/my_snapshot.pkl \
        --planner AORRTC --target 0
        # plans only target index 0 (from --list)
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import pathlib
import pickle
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless (WSL has no display attached to this script)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3d projection
from scipy.spatial.transform import Rotation

# Segfaults in the OMPL/Pinocchio native bindings aren't catchable Python
# exceptions - faulthandler at least prints the Python-level call stack that
# was active at the moment of the crash (which C-extension call was in
# flight), which is the only diagnostic signal we get without gdb.
faulthandler.enable()

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from common.config_loader import load_config
from util.logger.console import ConsoleLogger


def _load_snapshot(path, console):
    with open(path, "rb") as f:
        snapshot = pickle.load(f)
    if not snapshot.get("target_groups"):
        raise ValueError(
            f"snapshot has no target_groups: {path} "
            "(save it again after EF pose determination succeeds)")
    if "rt_pipe_facing_axis" not in snapshot:
        console.warning(
            "Snapshot predates rt_pipe_facing_axis being saved - falling back to the default "
            "(0,-1,0). If EF pose config uses a non-default pipe_facing_axis, this snapshot's "
            "rotation-needed classification here may disagree with what was used when its "
            "dda_pose_resolved/rt_pose_resolved were computed. Re-save it to fix.")
    return snapshot


def _all_targets(snapshot):
    """Every (robot, pose) target across all phases/groups, in the same order
    SimTool's InspectionSequencer/the benchmark script would plan them in.
    Yields dicts: index, group_name, robot_name, pose_name, target_pose,
    needs_rotation, unresolved. target_pose is already the *resolved* pose
    (rotation baked in by resolve_target_groups_with_rotation() at
    snapshot-save time) - callers don't need to reapply rotation_T.
    """
    from plugins.robotics.inspection_workflow import (
        inspection_group_pose_items, partition_and_sort_target_groups)

    phases = partition_and_sort_target_groups(
        snapshot["target_groups"],
        rt_pipe_facing_axis=snapshot.get("rt_pipe_facing_axis", (0.0, -1.0, 0.0)))
    i = 0
    for phase in phases:
        rotate = phase.get("requires_positioner_rotation", False)
        for group in phase["groups"]:
            for robot_name, pose_name, target_T in inspection_group_pose_items(group):
                yield {
                    "index": i,
                    "group_name": group.get("name"),
                    "robot_name": robot_name,
                    "pose_name": pose_name,
                    "target_pose": target_T,
                    "needs_rotation": rotate,
                    "unresolved": bool(group.get("positioner_rotation_unresolved")),
                }
                i += 1


def _list_targets(snapshot, console):
    n = 0
    for t in _all_targets(snapshot):
        console.info(
            f"[{t['index']}] group={t['group_name']!r} robot={t['robot_name']} pose={t['pose_name']} "
            f"needs_rotation={t['needs_rotation']} unresolved={t['unresolved']} "
            f"target_xyz={np.round(t['target_pose'][:3, 3], 4).tolist()}")
        n += 1
    if n == 0:
        console.warning("no targets found in this snapshot")


def _nth_target(snapshot, index):
    for t in _all_targets(snapshot):
        if t["index"] == index:
            return t
    raise IndexError(f"target index {index} out of range - use --list")


def _resolve_positioner_r_deg(snapshot, config, target):
    """The positioner angle (deg) this target's path was actually collision-
    checked against - matches how the snapshot's second_group_rotation_T was
    computed at save time (Visualizer._inspection_robot_core_snapshot always
    used the config default; see benchmark_path_planners.py's identical
    second_r_deg computation). Needed so a saved run's playback can rotate
    the positioner to match what the robot was actually planned around,
    instead of always showing it at its snapshot-save angle - see
    playback_loader.py."""
    initial_r_deg = float(snapshot.get("positioner_r_deg", 0.0))
    second_r_deg = initial_r_deg + float(
        (config.get("path_planning", {}) or {}).get("positioner_second_group_r_deg", 180.0))
    return second_r_deg if target["needs_rotation"] else initial_r_deg


def _seed_track_to_target(engine, robot_name, start_q, target_pose):
    """Overwrite just the linear_track component of start_q to sit near
    target_pose's world x - reuses visualizer.py's FK-sensitivity seeding
    (_seed_linear_track_q_for_world_x), the same helper real planning uses to
    seed the IK solver's initial guess, but applied here to the actual
    planning start_q for --seed-track-to-target (fast local testing, not a
    claim about the robot's real position). carriage (world y) is left alone
    - its target-y mapping is ambiguous (may depend on the nearest scan point
    rather than the raw target pose, see _inspection_track_fixed_q's
    nearest_point handling) so seeding it here could produce a worse guess
    than just leaving the live value."""
    backend = getattr(engine, "_robotics_backend", None)
    robot_backend_model = backend.robot_model(robot_name) if backend is not None else None
    frame_name = engine._robot_target_link_name(robot_name)
    target_arr = np.asarray(target_pose, dtype=float)
    target_xyz = target_arr[:3, 3] if target_arr.shape == (4, 4) else target_arr.reshape(-1)[:3]
    q = np.asarray(start_q, dtype=float).copy()
    q = engine._seed_linear_track_q_for_world_x(
        robot_name, robot_backend_model, frame_name, q, float(target_xyz[0]))
    return q.tolist()


def _joint_names_for(engine, robot_name):
    backend = getattr(engine, "_robotics_backend", None)
    robot_backend_model = backend.robot_model(robot_name) if backend is not None else None
    names = engine._robot_joint_names(robot_name, robot_backend_model)
    return [str(n) for n in names] if names else None


def _ee_poses(engine, robot_name, q_path):
    """(N,3) positions and (N,3) roll/pitch/yaw (deg) for every waypoint's FK."""
    backend = engine._robotics_backend
    positions = []
    eulers_deg = []
    for q in q_path:
        world_T = np.asarray(backend.frame_world_T(robot_name, q), dtype=float)
        positions.append(world_T[:3, 3])
        eulers_deg.append(Rotation.from_matrix(world_T[:3, :3]).as_euler("xyz", degrees=True))
    return np.asarray(positions), np.asarray(eulers_deg)


def _stats_caption(planner_stats):
    """One-line iteration/solve-time caption for plot annotations, or ""
    if there's nothing to show (legacy planners don't populate this)."""
    if not planner_stats:
        return ""
    caption = (
        f"iterations={planner_stats.get('iterations')} "
        f"(max_iter={planner_stats.get('max_iter')}, timeout_sec={planner_stats.get('timeout_sec')})  "
        f"solve_time={planner_stats.get('solve_time')}  "
        f"state_validity_calls={planner_stats.get('state_validity_calls')}  "
        f"collision_rejects={planner_stats.get('collision_rejects')}")
    if planner_stats.get("iteration_ptc_error"):
        caption += f"\niteration_ptc_error={planner_stats['iteration_ptc_error']}"
    return caption


def _save_results(engine, robot_name, planner_name, q_path, *, out_dir, console, planner_stats=None,
                   failure_info=None):
    """Save one target's planning result into out_dir (caller creates it):
    - joint_states.csv / joint_states.png (joint value per waypoint)
    - ee_pose.csv / ee_pose.png (task-space EF pose per waypoint, from FK)
    - planner_stats.json (iterations/solve_time/max_iter/timeout_sec/... -
      see OMPLPlannerBase._generate_joint_space's last_ompl_stats; empty for
      legacy, non-OMPL planners) - also annotated onto both plots so the
      iteration count is visible without opening a separate file.
    - failure_info.json + a red marker on joint_states.png at the colliding
      waypoint/edge, if this q_path is a *failed* result being saved for
      diagnosis (see plan_single_target's "verification"/exc.q_path - the
      collision that made this target fail, not a normal successful path).
    """
    if not q_path:
        console.warning("no q_path to save (planning did not produce a path)")
        return None

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if planner_stats:
        with open(out_dir / "planner_stats.json", "w", encoding="utf-8") as f:
            json.dump(planner_stats, f, indent=2)
    stats_caption = _stats_caption(planner_stats)

    bad_waypoints = []
    if failure_info:
        with open(out_dir / "failure_info.json", "w", encoding="utf-8") as f:
            json.dump(failure_info, f, indent=2, default=str)
        verification = failure_info.get("verification") or {}
        bad_waypoints = sorted({w["waypoint"] for w in verification.get("waypoint_collisions", [])}
                                | {e["edge"] for e in verification.get("edge_collisions", [])}
                                | {e["edge"] + 1 for e in verification.get("edge_collisions", [])})
        console.warning(
            f"FAILED result saved for diagnosis: {failure_info.get('message')} "
            f"(colliding waypoint index/indices in joint_states.csv: {bad_waypoints or 'unknown'})")

    q_arr = np.asarray(q_path, dtype=float)
    n_waypoints, dof = q_arr.shape
    joint_names = _joint_names_for(engine, robot_name) or [f"joint_{i}" for i in range(dof)]

    # 1) joint state per waypoint -> CSV
    joint_csv_path = out_dir / "joint_states.csv"
    with open(joint_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["waypoint", *joint_names])
        for i, q in enumerate(q_arr):
            writer.writerow([i, *[float(v) for v in q]])

    # 2) joint state per waypoint -> plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for j in range(dof):
        ax.plot(range(n_waypoints), q_arr[:, j], marker="o", markersize=3, label=joint_names[j])
    for i, wp in enumerate(bad_waypoints):
        if 0 <= wp < n_waypoints:
            ax.axvline(wp, color="red", linestyle="--", linewidth=1.5,
                       label="collision" if i == 0 else None)
    ax.set_xlabel("waypoint")
    ax.set_ylabel("joint value (rad)")
    title = f"{planner_name} - joint states per waypoint (robot={robot_name})"
    if bad_waypoints:
        title += f"  [FAILED - collision at waypoint {bad_waypoints}]"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if stats_caption:
        fig.text(0.01, 0.01, stats_caption, fontsize=7, color="dimgray")
    joint_png_path = out_dir / "joint_states.png"
    fig.savefig(joint_png_path, dpi=150)
    plt.close(fig)

    # 3) end-effector task-space pose per waypoint (FK) -> CSV + plot
    positions, eulers_deg = _ee_poses(engine, robot_name, q_path)
    ee_csv_path = out_dir / "ee_pose.csv"
    with open(ee_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["waypoint", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"])
        for i in range(n_waypoints):
            writer.writerow([i, *positions[i].tolist(), *eulers_deg[i].tolist()])

    fig = plt.figure(figsize=(11, 8))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker="o", markersize=3)
    ax3d.scatter(*positions[0], color="green", s=60, label="start")
    ax3d.scatter(*positions[-1], color="red", s=60, label="goal")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("EF trajectory (task space)")
    ax3d.legend()

    ax_pos = fig.add_subplot(2, 2, 2)
    for i, label in enumerate(("x", "y", "z")):
        ax_pos.plot(range(n_waypoints), positions[:, i], marker="o", markersize=3, label=label)
    ax_pos.set_xlabel("waypoint")
    ax_pos.set_ylabel("position (m)")
    ax_pos.set_title("EF position per waypoint")
    ax_pos.legend()
    ax_pos.grid(True, alpha=0.3)

    ax_rot = fig.add_subplot(2, 2, 3)
    for i, label in enumerate(("roll", "pitch", "yaw")):
        ax_rot.plot(range(n_waypoints), eulers_deg[:, i], marker="o", markersize=3, label=label)
    ax_rot.set_xlabel("waypoint")
    ax_rot.set_ylabel("angle (deg)")
    ax_rot.set_title("EF orientation per waypoint")
    ax_rot.legend()
    ax_rot.grid(True, alpha=0.3)

    ax_xy = fig.add_subplot(2, 2, 4)
    ax_xy.plot(positions[:, 0], positions[:, 1], marker="o", markersize=3)
    ax_xy.scatter(*positions[0, :2], color="green", s=60, label="start")
    ax_xy.scatter(*positions[-1, :2], color="red", s=60, label="goal")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title("EF trajectory (top-down XY)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)

    fig.suptitle(f"{planner_name} - end-effector pose (robot={robot_name})")
    fig.tight_layout()
    if stats_caption:
        fig.text(0.01, 0.01, stats_caption, fontsize=7, color="dimgray")
    ee_png_path = out_dir / "ee_pose.png"
    fig.savefig(ee_png_path, dpi=150)
    plt.close(fig)

    console.info(f"Saved planning result visualization to {out_dir}")
    return out_dir


def _live_start_q(engine, robot_name):
    model = engine._find_robot(robot_name)
    if model is None:
        raise RuntimeError(f"robot not found in snapshot: {robot_name}")
    backend = getattr(engine, "_robotics_backend", None)
    robot_backend_model = backend.robot_model(robot_name) if backend is not None else None
    return engine._current_robot_q(model, robot_backend_model, robot_name=robot_name).tolist()


def _plan_one(engine, plan_single_target, target, args, snapshot, config, *, start_q, out_dir, console):
    """Plan one target, print a result summary, and save its visualization
    under out_dir. Returns (result, q_path)."""
    method_label = f"{args.planner}+{args.optimizer}" if args.optimizer else args.planner
    robot_name = target["robot_name"]
    pose_name = target["pose_name"]
    obstacle_rotation_T = None
    if target["needs_rotation"]:
        if target["unresolved"]:
            raise RuntimeError(
                f"target[{target['index']}] ({target['group_name']}:{pose_name}) needs a "
                "positioner rotation but the snapshot's pose is unresolved (spool_fix_r was "
                "False when it was saved - re-save it with the spool's 'Fix F Column R' "
                "checkbox enabled)")
        obstacle_rotation_T = snapshot.get("second_group_rotation_T")
        console.info(
            f"target[{target['index']}] needed a positioner rotation - snapshot's resolved "
            "pose already has it applied, and the collision pipe mesh will be rotated to match")
    positioner_r_deg = _resolve_positioner_r_deg(snapshot, config, target)

    request = {
        "robot_name": robot_name,
        "start_q": start_q,
        "target_pose": target["target_pose"],
        "planner": args.planner,
        "step_size": args.step_size,
        "max_iter": args.max_iter,
        "planning_timeout": args.timeout,
        "ik_solver": args.ik_solver,
        "ik_normalize": bool(args.ik_normalize),
        "context_label": f"test:{target['group_name']}:{pose_name}",
        "obstacle_rotation_T": obstacle_rotation_T,
        "optimizer": args.optimizer,
        "optimize_path": bool(args.optimizer),
        # Read by visualizer.py's _plan_inspection_path_for_robot and stashed
        # onto planner.debug_positioner_r_deg for optimizers (e.g. stomp.py)
        # to record in their own playback output - see _resolve_positioner_r_deg.
        "positioner_r_deg": positioner_r_deg,
    }
    console.info(f"plan_single_target request: {request}")

    output = plan_single_target(engine, request)
    result = output.get("result", {})
    q_path = output.get("q_path") or []
    planner_stats = result.get("planner_stats") or {}

    target_arr = np.asarray(target["target_pose"], dtype=float)
    target_xyz = target_arr[:3, 3] if target_arr.shape == (4, 4) else target_arr.reshape(-1)[:3]

    print()
    print(f"[{target['index']}] {target['group_name']}:{pose_name} ({robot_name}) "
          f"target_xyz={np.round(target_xyz, 4).tolist()}")
    print(f"  status   = {result.get('status')}")
    print(f"  message  = {result.get('message')}")
    if result.get("status") != "success" and "goal_collision" in str(result.get("message") or ""):
        # goal_q(IK가 이 target_pose로 도달한 자세) 자체가 충돌이라는 뜻 - 여러 target을
        # 한 번에(--target all) 돌릴 때 콘솔에 이름만 스쳐지나가면 어느 지점인지 헷갈리기
        # 쉬워서, world 좌표를 별도로 강조해 찍는다 (SimTool GUI의 goal_collision 마커와
        # 같은 정보를 헤드리스 실행에서도 볼 수 있게 - visualizer.py의
        # _show_ik_failure_reached_pose는 RobotCoreEngine에서 no-op이라 여기서 나온다).
        print(f"  *** GOAL COLLISION at target_xyz={np.round(target_xyz, 4).tolist()} "
              f"(target[{target['index']}] {target['group_name']}:{pose_name}, robot={robot_name}) ***")
    print(f"  ik_failure = {result.get('ik_failure')}")
    print(f"  elapsed  = {result.get('elapsed')}")
    print(f"  n_waypoints = {len(q_path)}")
    if planner_stats:
        print(
            f"  iterations = {planner_stats.get('iterations')} "
            f"(max_iter={planner_stats.get('max_iter')}, timeout_sec={planner_stats.get('timeout_sec')}), "
            f"solve_time = {planner_stats.get('solve_time')}, "
            f"state_validity_calls = {planner_stats.get('state_validity_calls')}, "
            f"collision_rejects = {planner_stats.get('collision_rejects')}")
        if planner_stats.get("iteration_ptc_error"):
            print(f"  iteration_ptc_error = {planner_stats['iteration_ptc_error']}")
    if q_path:
        print(f"  q_path[0]  = {np.round(q_path[0], 5).tolist()}")
        print(f"  q_path[-1] = {np.round(q_path[-1], 5).tolist()}")
        failure_info = None
        if result.get("status") != "success":
            # q_path here is the *colliding* path plan_single_target normally
            # discards (see OMPLPlannerBase.last_failed_q_path / visualizer.py's
            # "planning failed for target" exception) - save it anyway so a
            # failure can still be played back and the exact collision point
            # identified, instead of only a status string.
            failure_info = {
                "status": result.get("status"), "message": result.get("message"),
                "verification": result.get("verification"),
            }
        saved_dir = _save_results(
            engine, robot_name, method_label, q_path, out_dir=out_dir, console=console,
            planner_stats=planner_stats, failure_info=failure_info)
        if saved_dir is not None:
            print(f"  results saved to {saved_dir}")
    return result, q_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(pathlib.Path(__file__).with_name("viewervedo.cfg")))
    parser.add_argument("--snapshot", required=True, help="Planning snapshot .pkl saved from SimTool")
    parser.add_argument("--list", action="store_true", help="List available (robot, pose) targets and exit")
    parser.add_argument(
        "--target", default="all",
        help="Target index from --list to plan, or 'all' (default) to plan every target in "
             "order, chaining each robot's start_q from its own previous target")
    parser.add_argument("--planner", default="RRTConnect", help="Planner name (OMPL algorithm or legacy planner)")
    parser.add_argument(
        "--optimizer", default=None,
        help="Optional post-optimization stage (plugins/optimizer/*.py module stem, e.g. "
             "stomp/trajopt/gpmp2/path_pruning/bspline/topp_ra). Generates the path with "
             "--planner first, then re-optimizes it - most useful with --planner direct_path "
             "(pure straight-line interpolation, no collision avoidance of its own) so the "
             "optimizer does all the actual obstacle avoidance. No 'chomp' module exists; "
             "gpmp2/trajopt are the closest available.")
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=30.0, help="Planning timeout (s), 0=no limit")
    parser.add_argument("--ik-solver", default="pybullet")
    parser.add_argument("--ik-normalize", action="store_true")
    parser.add_argument("--start-q", default=None,
                         help="Comma-separated override start_q for the first target planned. "
                              "Default: robot's live joint state in the snapshot. Ignored for "
                              "every target after the first for a given robot (chained instead).")
    parser.add_argument(
        "--seed-track-to-target", action="store_true",
        help="Overwrite just the linear_track component of the first target's start_q to sit "
             "near the target pose's world x, using the same FK-sensitivity seeding "
             "visualizer.py uses for the IK initial guess (_seed_linear_track_q_for_world_x) - "
             "but applied to the actual planning start_q here, not just IK's internal guess. "
             "carriage is intentionally left alone (its target-y mapping is ambiguous - see "
             "_seed_track_to_target's docstring). For quickly testing optimizer tuning without "
             "waiting out a full-length real track traverse each run - NOT a claim about where "
             "the robot actually is; combine with --start-q to also set the other joints, or "
             "leave it to seed on top of the live joint state.")
    parser.add_argument("--verbose-level", default="DEBUG")
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
    console.info(
        f"Snapshot: {args.snapshot} "
        f"({len(snapshot.get('target_groups') or [])} target group(s)), "
        f"spool_fix_r={snapshot.get('spool_fix_r')}")

    if args.list:
        _list_targets(snapshot, console)
        return

    from robot_core.worker import RobotCoreEngine
    from robot_core.path_planning_service import plan_single_target

    engine = RobotCoreEngine(config, snapshot)
    debug_dir = pathlib.Path(config.get("debug_dir", "debug"))
    method_label = f"{args.planner}+{args.optimizer}" if args.optimizer else args.planner
    run_dir = debug_dir / f"{method_label}_{time.strftime('%Y%m%d_%H%M%S')}"

    if args.target != "all":
        target = _nth_target(snapshot, int(args.target))
        start_q = (
            [float(v) for v in args.start_q.split(",")] if args.start_q
            else _live_start_q(engine, target["robot_name"]))
        if args.seed_track_to_target:
            start_q = _seed_track_to_target(engine, target["robot_name"], start_q, target["target_pose"])
        console.info(f"start_q = {np.round(start_q, 5).tolist()}")
        _plan_one(engine, plan_single_target, target, args, snapshot, config,
                  start_q=start_q, out_dir=run_dir, console=console)
        return

    targets = list(_all_targets(snapshot))
    if not targets:
        console.warning("no targets found in this snapshot")
        return
    console.info(f"Planning all {len(targets)} target(s) in order, method={method_label}")

    start_q_by_robot = {}
    if args.start_q:
        start_q_by_robot[targets[0]["robot_name"]] = [float(v) for v in args.start_q.split(",")]
    if args.seed_track_to_target:
        first = targets[0]
        base_start_q = start_q_by_robot.get(first["robot_name"]) or _live_start_q(engine, first["robot_name"])
        start_q_by_robot[first["robot_name"]] = _seed_track_to_target(
            engine, first["robot_name"], base_start_q, first["target_pose"])
        console.info(f"seeded track for {first['robot_name']}: start_q = "
                     f"{np.round(start_q_by_robot[first['robot_name']], 5).tolist()}")
    retreated_robots = set()

    summary = []
    for target in targets:
        robot_name = target["robot_name"]
        start_q = start_q_by_robot.get(robot_name) or _live_start_q(engine, robot_name)
        if target["needs_rotation"] and robot_name not in retreated_robots:
            # Before the positioner actually rotates, retreat to a safe,
            # known posture (arm folded to zero, track kept) rather than
            # starting from wherever the last pre-rotation target left the
            # arm - see zero_non_linear_track_joints()'s docstring.
            from plugins.robotics.inspection_workflow import zero_non_linear_track_joints
            backend = getattr(engine, "_robotics_backend", None)
            robot_backend_model = backend.robot_model(robot_name) if backend is not None else None
            joint_names = engine._robot_joint_names(robot_name, robot_backend_model)
            start_q = zero_non_linear_track_joints(start_q, joint_names)
            start_q_by_robot[robot_name] = start_q
            retreated_robots.add(robot_name)
            console.info(f"retreated {robot_name} before positioner rotation: start_q={np.round(start_q, 5).tolist()}")
        out_dir = run_dir / f"{target['index']:02d}_{robot_name}_{target['pose_name']}"
        try:
            result, q_path = _plan_one(
                engine, plan_single_target, target, args, snapshot, config,
                start_q=start_q, out_dir=out_dir, console=console)
        except Exception as exc:
            console.error(f"target[{target['index']}] failed: {exc}")
            target_arr = np.asarray(target["target_pose"], dtype=float)
            target_xyz = target_arr[:3, 3] if target_arr.shape == (4, 4) else target_arr.reshape(-1)[:3]
            summary.append({
                "index": target["index"], "group_name": target["group_name"],
                "robot_name": robot_name, "pose_name": target["pose_name"],
                "status": "error", "message": str(exc), "n_waypoints": 0,
                "needs_rotation": target["needs_rotation"],
                "positioner_r_deg": _resolve_positioner_r_deg(snapshot, config, target),
                "target_x": float(target_xyz[0]), "target_y": float(target_xyz[1]), "target_z": float(target_xyz[2]),
            })
            continue
        status = result.get("status")
        if status == "success" and q_path:
            start_q_by_robot[robot_name] = q_path[-1]
        planner_stats = result.get("planner_stats") or {}
        target_arr = np.asarray(target["target_pose"], dtype=float)
        target_xyz = target_arr[:3, 3] if target_arr.shape == (4, 4) else target_arr.reshape(-1)[:3]
        summary.append({
            "index": target["index"], "group_name": target["group_name"],
            "robot_name": robot_name, "pose_name": target["pose_name"],
            "status": status, "message": result.get("message"), "n_waypoints": len(q_path),
            "iterations": planner_stats.get("iterations"),
            "max_iter": planner_stats.get("max_iter"),
            "solve_time": planner_stats.get("solve_time"),
            "iteration_ptc_error": planner_stats.get("iteration_ptc_error"),
            # Positioner attitude this target was actually collision-checked
            # against - playback_loader.py reads this instead of assuming 0
            # deg, so a rotated-group target renders with the pipe/positioner
            # in the pose it was actually planned around (see
            # _resolve_positioner_r_deg / visualizer.py's
            # planner.debug_positioner_r_deg).
            "needs_rotation": target["needs_rotation"],
            "positioner_r_deg": _resolve_positioner_r_deg(snapshot, config, target),
            # World xyz of the target pose itself - so a goal_collision (or
            # any other) failure row is identifiable by location directly
            # from summary.csv, without cross-referencing group/pose name
            # against --list output.
            "target_x": float(target_xyz[0]), "target_y": float(target_xyz[1]), "target_z": float(target_xyz[2]),
        })

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "group_name", "robot_name", "pose_name", "status", "message", "n_waypoints",
            "iterations", "max_iter", "solve_time", "iteration_ptc_error",
            "needs_rotation", "positioner_r_deg", "target_x", "target_y", "target_z"])
        writer.writeheader()
        writer.writerows(summary)

    scene_meta_path = run_dir / "scene_meta.json"
    with open(scene_meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "snapshot_path": str(args.snapshot),
            "config_path": str(args.config),
            "spool_fix_r": snapshot.get("spool_fix_r"),
            "positioner_r_deg_initial": float(snapshot.get("positioner_r_deg", 0.0)),
            "positioner_second_group_r_deg_delta": float(
                (config.get("path_planning", {}) or {}).get("positioner_second_group_r_deg", 180.0)),
            "has_second_group_rotation_T": snapshot.get("second_group_rotation_T") is not None,
        }, f, indent=2)
    console.info(f"scene metadata written to {scene_meta_path}")

    n_ok = sum(1 for row in summary if row["status"] == "success")
    print()
    print(f"{n_ok}/{len(summary)} target(s) succeeded, method={method_label}")
    print(f"summary written to {summary_path}")
    for row in summary:
        print(
            f"  [{row['index']}] {row['group_name']}:{row['pose_name']} ({row['robot_name']}) "
            f"status={row['status']} iterations={row.get('iterations')}"
            f"(max_iter={row.get('max_iter')}, solve_time={row.get('solve_time')})"
            + (f" message={row['message']}" if row.get("message") else ""))


if __name__ == "__main__":
    main()
