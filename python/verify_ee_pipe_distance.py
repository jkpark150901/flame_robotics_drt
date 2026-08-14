"""Verify the end-effector-to-pipe distance at a saved path's waypoint
(default: the last one).

For an inspection RRTConnect path, the final waypoint's EE link should sit
right at the pipe surface - typically well under 10cm - that's the
definition of "reached the inspection target". This script computes that
distance two independent ways (hppfcl mesh-to-mesh, and a from-scratch
nearest-raw-vertex check) under both positioner-rotation choices, and prints
a pass/fail verdict against a threshold.

Both distance computations correct for this robot's base_T (its world mount
pose, from viewervedo.cfg's "base" entry) - pin.forwardKinematics(model,
data, q) gives placements in the robot's LOCAL model-root frame, not
multiplied by base_T, while _current_spool_collision_mesh()'s vertices are
in true WORLD frame. Real planning (visualizer.py:3780-3797,
_base_frame_collision_mesh) already corrects for this before collision
checks; skipping it reproduces exactly this robot's mount-offset distance as
a phantom "far from the pipe" error (confirmed live: a robot mounted 3.75m
away read as ~3.5m from a pipe it was actually touching).

Usage:
    python python/verify_ee_pipe_distance.py \
        --config python/viewervedo.cfg \
        --snapshot sample/planning3.pkl \
        --joint-states-csv debug/RRTConnect_.../12_dda_rb10_1300e_DDA/joint_states.csv \
        --robot-name dda_rb10_1300e \
        --link dda_link_end_0
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import pickle
import sys

import numpy as np

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from common.config_loader import load_config
from util.logger.console import ConsoleLogger
from apf_heatmap import _load_joint_states_csv, _independent_ee_pipe_check


def _hppfcl_pipe_distance(engine, backend, robot_name, q, link_name, rotate, snapshot):
    """Reconfigures the collision scene (pipe rotated or not, base_T-corrected
    like real planning) and returns the hppfcl mesh-to-mesh distance from
    link_name to the pipe (collision_object_0)."""
    obstacle_mesh = engine._current_spool_collision_mesh()
    obstacle_mesh = copy.deepcopy(obstacle_mesh)
    if rotate and snapshot.get("second_group_rotation_T") is not None:
        obstacle_mesh.transform(np.asarray(snapshot["second_group_rotation_T"], dtype=float))

    handle = backend._handle(robot_name)
    base_T = np.asarray(handle.description.base_T, dtype=float)
    positioner_mesh = engine._build_positioner_collision_mesh()
    if not np.allclose(base_T, np.eye(4)):
        base_T_inv = np.linalg.inv(base_T)
        obstacle_mesh.transform(base_T_inv)
        if positioner_mesh is not None:
            positioner_mesh = copy.deepcopy(positioner_mesh)
            positioner_mesh.transform(base_T_inv)

    backend.configure_collision(
        robot_name,
        static_meshes=[m for m in (obstacle_mesh, positioner_mesh) if m is not None],
        sample_resolution=0.02,
    )
    entries = backend.link_obstacle_distances(robot_name, q)
    pipe_entries = [e for e in entries if e["link"] == link_name and e["obstacle"] == "collision_object_0"]
    return pipe_entries[0]["distance"] if pipe_entries else None


def _resolve_link_name(backend, robot_name, requested):
    handle = backend._handle(robot_name)
    names = [str(g.name) for g in handle.geom_model.geometryObjects]
    if requested:
        if requested not in names:
            raise ValueError(f"link {requested!r} not found - available: {names}")
        return requested
    candidates = [n for n in names if "end" in n.lower()]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"--link not given and couldn't guess an unambiguous end-effector link - "
        f"candidates: {candidates or names}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(pathlib.Path(__file__).with_name("viewervedo.cfg")))
    parser.add_argument("--snapshot", required=True, help="Planning snapshot .pkl saved from SimTool")
    parser.add_argument("--joint-states-csv", required=True, help="joint_states.csv from test_ompl_planning.py")
    parser.add_argument("--robot-name", required=True)
    parser.add_argument("--link", default=None, help="EE link name (default: guess the one with 'end' in its name)")
    parser.add_argument("--waypoint", type=int, default=-1, help="Waypoint index to check (default: -1, the last one)")
    parser.add_argument(
        "--expected-max-distance", type=float, default=0.10,
        help="Pass threshold in meters for the final waypoint's EE-to-pipe distance (default: 0.10)")
    args = parser.parse_args()

    config = load_config(args.config)
    extra_config_path = pathlib.Path(args.config).resolve().parent / "path_planning.cfg"
    if extra_config_path.exists():
        config.update(load_config(extra_config_path))
    config["root_path"] = ROOT_PATH
    ConsoleLogger.configure(config.get("logging", {}) or {}, force=True)
    console = ConsoleLogger.get_logger()

    with open(args.snapshot, "rb") as f:
        snapshot = pickle.load(f)

    joint_names, q_path = _load_joint_states_csv(args.joint_states_csv)
    if not q_path:
        raise ValueError(f"no waypoints in {args.joint_states_csv}")
    waypoint_index = args.waypoint if args.waypoint >= 0 else len(q_path) + args.waypoint
    if not (0 <= waypoint_index < len(q_path)):
        raise ValueError(f"--waypoint {args.waypoint} out of range for {len(q_path)} waypoints")
    q = np.array(q_path[waypoint_index], dtype=float)
    console.info(f"Checking waypoint {waypoint_index}/{len(q_path) - 1} (of {len(q_path)} total)")

    from robot_core.worker import RobotCoreEngine
    engine = RobotCoreEngine(config, snapshot)
    model = engine._find_robot(args.robot_name)
    if model is None:
        available = [str(getattr(m, "name", "")) for m in getattr(engine, "_robot_models", [])]
        raise RuntimeError(f"robot not found in snapshot: {args.robot_name!r} - available: {available}")
    backend = engine._robotics_backend

    # Configure once so _resolve_link_name/_handle work.
    _hppfcl_pipe_distance(engine, backend, args.robot_name, q, "", False, snapshot)
    link_name = _resolve_link_name(backend, args.robot_name, args.link)
    console.info(f"Checking link: {link_name!r}")

    handle = backend._handle(args.robot_name)
    base_T = np.asarray(handle.description.base_T, dtype=float)
    console.info(f"robot base_T (local model root -> world, from viewervedo.cfg 'base'):\n{base_T}")

    if snapshot.get("second_group_rotation_T") is None:
        console.warning(
            "snapshot has no second_group_rotation_T at all - this target's pipe was never rotated "
            "for any positioner-rotation logic; the --rotate distinction below is moot.")

    rows = []
    for rotate, label in [(False, "rotate=OFF"), (True, "rotate=ON ")]:
        hppfcl_dist = _hppfcl_pipe_distance(engine, backend, args.robot_name, q, link_name, rotate, snapshot)
        _, unrotated_check, rotated_check = _independent_ee_pipe_check(
            backend, args.robot_name, q, link_name, snapshot)
        raw_dist = rotated_check if rotate else unrotated_check
        verdict = "PASS" if (hppfcl_dist is not None and hppfcl_dist <= args.expected_max_distance) else "fail"
        rows.append((label, hppfcl_dist, raw_dist, verdict))

    console.info("")
    console.info(f"{'combo':<14}{'hppfcl dist':<16}{'raw-vertex dist':<18}{'verdict'}")
    for label, hppfcl_dist, raw_dist, verdict in rows:
        console.info(f"{label:<14}{hppfcl_dist!s:<16}{raw_dist!s:<18}{verdict}")

    passing = [label for label, _, _, verdict in rows if verdict == "PASS"]
    console.info("")
    if len(passing) == 1:
        console.info(
            f"VERDICT: {passing[0].strip()} matches (<= {args.expected_max_distance}m). This target needs "
            f"--rotate {'set' if 'ON' in passing[0] else 'unset'} in other scripts (apf_heatmap.py, etc).")
    elif len(passing) > 1:
        console.info(f"VERDICT: both rotation choices pass - can't disambiguate from distance alone: {passing}")
    else:
        console.info(
            f"VERDICT: NEITHER rotation choice gets the EE within {args.expected_max_distance}m of the pipe "
            "(base_T is already corrected for in both). This genuinely isn't the inspection-contact waypoint, "
            "or there's a frame issue beyond base_T/rotation.")


if __name__ == "__main__":
    main()
