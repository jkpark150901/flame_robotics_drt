#!/usr/bin/env python3
"""
Sweep servo_j parameters in robot Simulation mode and save reference profiles.

For every gain, t2, alpha combination this script sends a 0->90 deg servo_j
ramp on one joint, records jnt_ref, and saves:
  - per-trial joint angle change time series
  - target crossing interval/time
  - final target settle time
  - last command send time

This still connects to the robot controller. Default mode is Simulation, but
verify the workspace is clear before running.
"""

import argparse
import csv
import itertools
import math
import threading
import time
from pathlib import Path

import numpy as np
import rbpodo as rb


DEFAULT_ROBOT_IP = "10.0.2.7"


def _parse_float_list(text: str):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _check_response(rc):
    rc.error().throw_if_not_empty()
    if hasattr(rc, "clear"):
        rc.clear()


def _wait_for_motion(robot, rc, start_timeout=2.0, finish_timeout=60.0):
    started = robot.wait_for_move_started(rc, start_timeout)
    if started.type() == rb.ReturnType.Success:
        robot.wait_for_move_finished(rc, finish_timeout)
    _check_response(rc)


def _sleep_until(target_time):
    dt = target_time - time.perf_counter()
    if dt > 0:
        time.sleep(dt)


def _read_jnt_ref(robot_ip):
    data_channel = rb.CobotData(robot_ip)
    data = data_channel.request_data(2.0)
    if data is None:
        raise RuntimeError("No robot data received while checking initial position")
    return np.array(data.sdata.jnt_ref, dtype=float)


def _reset_to_start_pose(robot, rc, args, start_pose):
    robot.enable_waiting_ack(rc)
    robot.move_j(rc, start_pose, args.move_j_speed, args.move_j_accel)
    _wait_for_motion(robot, rc, finish_timeout=args.finish_timeout)

    deadline = time.perf_counter() + args.reset_timeout
    while time.perf_counter() < deadline:
        current_ref = _read_jnt_ref(args.robot_ip)
        err = np.max(np.abs(current_ref - start_pose))
        if err <= args.reset_tolerance_deg:
            time.sleep(args.reset_settle_s)
            return current_ref, err
        time.sleep(0.05)

    current_ref = _read_jnt_ref(args.robot_ip)
    err = np.max(np.abs(current_ref - start_pose))
    raise RuntimeError(
        f"Failed to reset start pose within {args.reset_tolerance_deg:g} deg. "
        f"max_err={err:.4f} deg, current_ref={np.round(current_ref, 4).tolist()}"
    )


def _record_loop(robot_ip, records, stop_event, sample_hz, timing_info):
    data_channel = rb.CobotData(robot_ip)
    period = 1.0 / sample_hz
    t0 = time.perf_counter()
    timing_info["record_t0"] = t0

    while not stop_event.is_set():
        loop_start = time.perf_counter()
        try:
            data = data_channel.request_data(0.5)
            if data is not None:
                records.append({
                    "t": time.perf_counter() - t0,
                    "jnt_ref": np.array(data.sdata.jnt_ref, dtype=float),
                })
        except Exception:
            pass

        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0.0, period - elapsed))


def _find_target_crossing(t, q, target, start, end):
    direction = 1.0 if end >= start else -1.0
    signed = direction * (q - target)

    for i in range(1, len(q)):
        prev_v = signed[i - 1]
        curr_v = signed[i]
        if prev_v < 0.0 <= curr_v:
            t0, t1 = t[i - 1], t[i]
            q0, q1 = q[i - 1], q[i]
            if math.isclose(q1, q0):
                crossing_time = t1
            else:
                crossing_time = t0 + (target - q0) * (t1 - t0) / (q1 - q0)
            return {
                "crossing_time_s": crossing_time,
                "crossing_interval_start_s": t0,
                "crossing_interval_end_s": t1,
                "crossing_interval_start_deg": q0,
                "crossing_interval_end_deg": q1,
            }

    return {
        "crossing_time_s": None,
        "crossing_interval_start_s": None,
        "crossing_interval_end_s": None,
        "crossing_interval_start_deg": None,
        "crossing_interval_end_deg": None,
    }


def _find_final_settle_time(t, q, target, after_time, tolerance_deg, hold_s):
    if len(t) == 0:
        return None

    start_idx = int(np.searchsorted(t, after_time, side="left"))
    for i in range(start_idx, len(t)):
        if abs(q[i] - target) > tolerance_deg:
            continue

        if hold_s <= 0:
            return t[i]

        end_time = t[i] + hold_s
        end_idx = int(np.searchsorted(t, end_time, side="left"))
        if end_idx >= len(t):
            return None
        if np.all(np.abs(q[i:end_idx + 1] - target) <= tolerance_deg):
            return t[i]

    return None


def _nearest_command_angle(command_records, sample_t):
    if not command_records:
        return None
    idx = np.searchsorted([r["t"] for r in command_records], sample_t, side="right") - 1
    idx = int(np.clip(idx, 0, len(command_records) - 1))
    return command_records[idx]["cmd_deg"]


def _safe_float(value):
    return "" if value is None else value


def _save_trial_csv(path, t, q_ref, command_records, joint_index):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t_s",
                "joint",
                "jnt_ref_deg",
                "angle_change_deg",
                "last_sent_cmd_deg",
            ],
        )
        writer.writeheader()
        q0 = q_ref[0]
        for sample_t, sample_q in zip(t, q_ref):
            writer.writerow({
                "t_s": sample_t,
                "joint": joint_index + 1,
                "jnt_ref_deg": sample_q,
                "angle_change_deg": sample_q - q0,
                "last_sent_cmd_deg": _nearest_command_angle(command_records, sample_t),
            })


def _save_command_csv(path, command_records, joint_index):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["t_s", "joint", "cmd_deg"],
        )
        writer.writeheader()
        for row in command_records:
            writer.writerow({
                "t_s": row["t"],
                "joint": joint_index + 1,
                "cmd_deg": row["cmd_deg"],
            })


def _run_trial(robot, rc, args, trial_index, gain, t2, alpha, out_dir):
    records = []
    command_records = []
    timing_info = {}
    stop_event = threading.Event()

    record_thread = threading.Thread(
        target=_record_loop,
        args=(args.robot_ip, records, stop_event, args.sample_hz, timing_info),
        daemon=True,
    )

    joint_index = args.joint - 1
    start_pose = np.zeros(6, dtype=float)
    start_pose[joint_index] = args.start_deg

    print(
        f"[{trial_index:03d}] gain={gain:g}, t2={t2:g}, alpha={alpha:g} "
        f"-> J{args.joint} {args.start_deg:g} to {args.end_deg:g} deg"
    )

    reset_ref, reset_err = _reset_to_start_pose(robot, rc, args, start_pose)
    print(f"    reset ok: max_ref_err={reset_err:.4f} deg, ref={np.round(reset_ref, 4).tolist()}")

    record_thread.start()
    while "record_t0" not in timing_info:
        time.sleep(0.001)
    time.sleep(args.pre_record)

    last_command_perf = None
    robot.disable_waiting_ack(rc)
    stream_start_perf = time.perf_counter()
    try:
        for i in range(args.steps + 1):
            ratio = i / args.steps
            cmd_deg = args.start_deg + ratio * (args.end_deg - args.start_deg)
            q = np.zeros(6, dtype=float)
            q[joint_index] = cmd_deg
            robot.move_servo_j(rc, q, args.t1, t2, gain, alpha)
            last_command_perf = time.perf_counter()
            command_records.append({
                "t": last_command_perf - timing_info["record_t0"],
                "cmd_deg": cmd_deg,
            })
            _sleep_until(stream_start_perf + (i + 1) * args.servo_dt)
    finally:
        robot.enable_waiting_ack(rc)

    robot.wait_for_move_finished(rc, args.finish_timeout)
    _check_response(rc)
    time.sleep(args.post_record)

    stop_event.set()
    record_thread.join(timeout=2.0)

    if len(records) < 3:
        raise RuntimeError("Not enough records captured")

    t = np.array([r["t"] for r in records], dtype=float)
    q_ref_all = np.array([r["jnt_ref"] for r in records], dtype=float)
    q_ref = q_ref_all[:, joint_index]

    last_command_time = last_command_perf - timing_info["record_t0"]
    crossing = _find_target_crossing(t, q_ref, args.end_deg, args.start_deg, args.end_deg)
    final_settle_time = _find_final_settle_time(
        t,
        q_ref,
        args.end_deg,
        last_command_time,
        args.final_tolerance_deg,
        args.final_hold_s,
    )

    stem = (
        f"trial_{trial_index:03d}_"
        f"gain_{gain:g}_t2_{t2:g}_alpha_{alpha:g}".replace(".", "p")
    )
    profile_csv = out_dir / f"{stem}_profile.csv"
    command_csv = out_dir / f"{stem}_commands.csv"
    _save_trial_csv(profile_csv, t, q_ref, command_records, joint_index)
    _save_command_csv(command_csv, command_records, joint_index)

    return {
        "trial": trial_index,
        "joint": args.joint,
        "gain": gain,
        "t1_s": args.t1,
        "t2_s": t2,
        "alpha": alpha,
        "servo_dt_s": args.servo_dt,
        "start_deg": args.start_deg,
        "end_deg": args.end_deg,
        "reset_max_ref_err_deg": reset_err,
        "samples": len(t),
        "commands": len(command_records),
        "profile_csv": profile_csv.name,
        "command_csv": command_csv.name,
        "last_command_time_s": last_command_time,
        "final_settle_time_s": final_settle_time,
        "final_settle_dt_from_last_cmd_s": (
            None if final_settle_time is None else final_settle_time - last_command_time
        ),
        "final_ref_deg": q_ref[-1],
        "max_ref_deg": np.nanmax(q_ref),
        "min_ref_deg": np.nanmin(q_ref),
        **crossing,
        "crossing_dt_from_last_cmd_s": (
            None if crossing["crossing_time_s"] is None else crossing["crossing_time_s"] - last_command_time
        ),
    }


def _write_summary(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _safe_float(value) for key, value in row.items()})


def _main():
    parser = argparse.ArgumentParser(description="Sweep servo_j gain/t2/alpha in Simulation mode.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--joint", type=int, default=1, choices=range(1, 7), help="Joint number, 1-6.")
    parser.add_argument("--start-deg", type=float, default=0.0)
    parser.add_argument("--end-deg", type=float, default=90.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--servo-dt", type=float, default=0.005, help="Actual command send period.")
    parser.add_argument("--t1", type=float, default=0.01)
    parser.add_argument("--t2-values", default="0.1,0.2")
    parser.add_argument("--gain-values", default="0.2,0.5,1.0")
    parser.add_argument("--alpha-values", default="0.2,0.5,1.0")
    parser.add_argument("--sample-hz", type=float, default=100.0)
    parser.add_argument("--pre-record", type=float, default=0.3)
    parser.add_argument("--post-record", type=float, default=1.0)
    parser.add_argument("--final-tolerance-deg", type=float, default=0.05)
    parser.add_argument("--final-hold-s", type=float, default=0.1)
    parser.add_argument("--move-j-speed", type=float, default=50.0)
    parser.add_argument("--move-j-accel", type=float, default=100.0)
    parser.add_argument("--reset-tolerance-deg", type=float, default=0.05)
    parser.add_argument("--reset-settle-s", type=float, default=0.2)
    parser.add_argument("--reset-timeout", type=float, default=5.0)
    parser.add_argument("--finish-timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("servo_j_sweep_results"))
    parser.add_argument("--real", action="store_true", help="Use Real mode instead of Simulation.")
    parser.add_argument("--dry-run", action="store_true", help="Print trial list without connecting.")
    args = parser.parse_args()

    gains = _parse_float_list(args.gain_values)
    t2_values = _parse_float_list(args.t2_values)
    alphas = _parse_float_list(args.alpha_values)
    combinations = list(itertools.product(gains, t2_values, alphas))

    print(f"trials: {len(combinations)}")
    for idx, (gain, t2, alpha) in enumerate(combinations, 1):
        print(f"[{idx:03d}] gain={gain:g}, t2={t2:g}, alpha={alpha:g}")

    if args.dry_run:
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = rb.Cobot(args.robot_ip)
    rc = rb.ResponseCollector()
    mode = rb.OperationMode.Real if args.real else rb.OperationMode.Simulation
    robot.set_operation_mode(rc, mode)
    robot.set_speed_bar(rc, 1.0)
    robot.flush(rc)
    _check_response(rc)

    summary_rows = []
    for idx, (gain, t2, alpha) in enumerate(combinations, 1):
        row = _run_trial(robot, rc, args, idx, gain, t2, alpha, out_dir)
        summary_rows.append(row)
        _write_summary(out_dir / "summary.csv", summary_rows)

    print(f"saved: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    _main()
