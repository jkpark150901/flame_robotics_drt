#!/usr/bin/env python3
"""
Run robot1_trajectory_rbpodo_ready.csv with rbpodo MoveJB2.

The CSV format is:
  Order,J1,J2,J3,J4,J5,J6,Speed,Acceleration

Example:
  python run_robot1_move_jb2.py --robot-ip 10.0.2.7
  python run_robot1_move_jb2.py --robot-ip 10.0.2.7 --real
  python run_robot1_move_jb2.py --dry-run
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import rbpodo as rb


DEFAULT_CSV = Path("robot1_trajectory_rbpodo_ready.csv")
DEFAULT_ROBOT_IP = "10.0.2.7"


def _load_waypoints(csv_path: Path):
    waypoints = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            joints = np.array(
                [float(row[f"J{i}"]) for i in range(1, 7)],
                dtype=float,
            )
            waypoints.append({
                "order": int(row["Order"]),
                "joints": joints,
                "speed": float(row["Speed"]),
                "accel": float(row["Acceleration"]),
            })

    if not waypoints:
        raise ValueError(f"No waypoints found in {csv_path}")
    return waypoints


def _check(rc: rb.ResponseCollector):
    rc.error().throw_if_not_empty()
    if hasattr(rc, "clear"):
        rc.clear()


def _wait_for_motion(robot: rb.Cobot, rc: rb.ResponseCollector,
                     start_timeout: float, finish_timeout: float):
    started = robot.wait_for_move_started(rc, start_timeout)
    if started.type() == rb.ReturnType.Success:
        robot.wait_for_move_finished(rc, finish_timeout)
    _check(rc)


def _main():
    parser = argparse.ArgumentParser(
        description="Flush robot1 CSV trajectory to rbpodo move_jb2_add/run."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use the real robot operation mode. Default is Simulation.",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=0.0,
        help="MoveJB2 blending value for middle waypoints. First/last use 0.0.",
    )
    parser.add_argument("--start-timeout", type=float, default=5.0)
    parser.add_argument("--finish-timeout", type=float, default=120.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands; do not connect to the robot.",
    )
    args = parser.parse_args()

    waypoints = _load_waypoints(args.csv)
    print(f"Loaded {len(waypoints)} waypoints from {args.csv}")

    if args.dry_run:
        for wp in waypoints:
            print(
                f"move_jb2_add(order={wp['order']}, "
                f"joints={wp['joints'].tolist()}, "
                f"speed={wp['speed']}, accel={wp['accel']})"
            )
        return

    robot = rb.Cobot(args.robot_ip)
    rc = rb.ResponseCollector()

    mode = rb.OperationMode.Real if args.real else rb.OperationMode.Simulation
    robot.set_operation_mode(rc, mode)
    robot.flush(rc)
    _check(rc)

    first = waypoints[0]
    print(f"Moving to start pose: order={first['order']}")
    robot.move_j(rc, first["joints"], first["speed"], first["accel"])
    robot.flush(rc)
    _wait_for_motion(robot, rc, args.start_timeout, args.finish_timeout)

    print("Clearing MoveJB2 queue")
    robot.move_jb2_clear(rc)
    robot.flush(rc)
    _check(rc)

    print("Adding waypoints with move_jb2_add")
    for idx, wp in enumerate(waypoints):
        blend = 0.0 if idx == 0 or idx == len(waypoints) - 1 else args.blend
        robot.move_jb2_add(rc, wp["joints"], 200, 400, blend)
    robot.flush(rc)
    _check(rc)

    print("Running MoveJB2 queue")
    robot.move_jb2_run(rc)
    robot.flush(rc)
    _wait_for_motion(robot, rc, args.start_timeout, args.finish_timeout)
    print("Done")


if __name__ == "__main__":
    _main()
