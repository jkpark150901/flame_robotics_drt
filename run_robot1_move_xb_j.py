#!/usr/bin/env python3
"""
Run robot1_trajectory_rbpodo_ready.csv with rbpodo MoveXB J-type motion.

The CSV format is:
  Order,J1,J2,J3,J4,J5,J6,Speed,Acceleration

Examples:
  python run_robot1_move_xb_j.py --robot-ip 10.0.2.7
  python run_robot1_move_xb_j.py --robot-ip 10.0.2.7 --real
  python run_robot1_move_xb_j.py --dry-run
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


def _blending_option(name: str):
    if name == "ratio":
        return rb.BlendingOption.Ratio
    if name == "distance":
        return rb.BlendingOption.Distance
    raise ValueError(f"Unknown blending option: {name}")


def _move_xb_option(name: str):
    if name == "speed":
        return rb.MoveXBOption.Speed
    if name == "position":
        return rb.MoveXBOption.Position
    raise ValueError(f"Unknown MoveXB option: {name}")


def _main():
    parser = argparse.ArgumentParser(
        description="Run robot1 CSV trajectory with move_xb_j_add/move_xb_run."
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
        help="Blending value for middle waypoints. First/last use 0.0.",
    )
    parser.add_argument(
        "--blend-option",
        choices=("ratio", "distance"),
        default="ratio",
        help="move_xb_j_add blending type.",
    )
    parser.add_argument(
        "--run-option",
        choices=("speed", "position"),
        default="speed",
        help="move_xb_run option.",
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
    blend_option = _blending_option(args.blend_option)
    run_option = _move_xb_option(args.run_option)
    print(f"Loaded {len(waypoints)} waypoints from {args.csv}")

    if args.dry_run:
        print("move_xb_clear()")
        for idx, wp in enumerate(waypoints):
            blend = 0.0 if idx == 0 or idx == len(waypoints) - 1 else args.blend
            print(
                f"move_xb_j_add(order={wp['order']}, "
                f"joints={wp['joints'].tolist()}, "
                f"speed={wp['speed']}, accel={wp['accel']}, "
                f"option={args.blend_option}, blend={blend})"
            )
        print(f"move_xb_run(option={args.run_option})")
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

    print("Clearing MoveXB queue")
    robot.move_xb_clear(rc)
    robot.flush(rc)
    _check(rc)

    print("Adding waypoints with move_xb_j_add")
    for idx, wp in enumerate(waypoints):
        blend = 0.0 if idx == 0 or idx == len(waypoints) - 1 else args.blend
        robot.move_xb_j_add(
            rc,
            wp["joints"],
            wp["speed"],
            wp["accel"],
            blend_option,
            blend,
        )
    robot.flush(rc)
    _check(rc)

    print("Running MoveXB queue")
    robot.move_xb_run(rc, run_option)
    robot.flush(rc)
    _wait_for_motion(robot, rc, args.start_timeout, args.finish_timeout)
    print("Done")


if __name__ == "__main__":
    _main()
