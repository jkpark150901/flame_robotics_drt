#!/usr/bin/env python3
"""
Measure a single-joint MoveJB2 speed profile.

This script sends one MoveJB2 target that changes only one joint, records robot
feedback through CobotData, differentiates joint position, and saves CSV/PNG
outputs for checking whether MoveJB2 speed behaves like deg/s.

Examples:
  python test_jb2_joint_speed_profile.py --dry-run
  python test_jb2_joint_speed_profile.py --robot-ip 10.0.2.7 --joint 1 --delta 20 --speed 80 --accel 100
  python test_jb2_joint_speed_profile.py --robot-ip 10.0.2.7 --real --joint 1 --delta 20 --speed 80 --accel 100
"""

import argparse
import csv
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rbpodo as rb


DEFAULT_ROBOT_IP = "10.0.2.7"


def _check(rc: rb.ResponseCollector):
    rc.error().throw_if_not_empty()
    if hasattr(rc, "clear"):
        rc.clear()


def _read_state(data_ch: rb.CobotData):
    data = data_ch.request_data(2.0)
    if data is None:
        raise RuntimeError("No robot data received from CobotData")
    return np.array(data.sdata.jnt_ang, dtype=float), np.array(data.sdata.jnt_ref, dtype=float)


def _record_loop(robot_ip: str, stop_event: threading.Event, records: list, sample_hz: float):
    data_ch = rb.CobotData(robot_ip)
    period = 1.0 / sample_hz
    t0 = time.perf_counter()
    while not stop_event.is_set():
        loop_start = time.perf_counter()
        try:
            data = data_ch.request_data(0.5)
            if data is not None:
                records.append({
                    "t": time.perf_counter() - t0,
                    "jnt_ang": np.array(data.sdata.jnt_ang, dtype=float),
                    "jnt_ref": np.array(data.sdata.jnt_ref, dtype=float),
                })
        except Exception:
            pass
        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0.0, period - elapsed))


def _velocity_deg_s(t: np.ndarray, q: np.ndarray):
    if len(t) < 2:
        return np.zeros_like(q)
    return np.gradient(q, t, edge_order=1)


def _save_csv(path: Path, records: list, joint_idx: int):
    t = np.array([r["t"] for r in records], dtype=float)
    q_ang = np.array([r["jnt_ang"][joint_idx] for r in records], dtype=float)
    q_ref = np.array([r["jnt_ref"][joint_idx] for r in records], dtype=float)
    v_ang = _velocity_deg_s(t, q_ang)
    v_ref = _velocity_deg_s(t, q_ref)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t_s",
                "jnt_ang_deg",
                "jnt_ref_deg",
                "jnt_ang_vel_deg_s",
                "jnt_ref_vel_deg_s",
            ],
        )
        writer.writeheader()
        for i in range(len(t)):
            writer.writerow({
                "t_s": t[i],
                "jnt_ang_deg": q_ang[i],
                "jnt_ref_deg": q_ref[i],
                "jnt_ang_vel_deg_s": v_ang[i],
                "jnt_ref_vel_deg_s": v_ref[i],
            })

    return t, q_ang, q_ref, v_ang, v_ref


def _save_plot(path: Path, t, q_ang, q_ref, v_ang, v_ref, title: str, command_window):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, q_ang, label="jnt_ang measured", linewidth=1.5)
    axes[0].plot(t, q_ref, label="jnt_ref reference", linewidth=1.2, linestyle="--")
    axes[0].set_ylabel("position (deg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, v_ang, label="d(jnt_ang)/dt", linewidth=1.5)
    axes[1].plot(t, v_ref, label="d(jnt_ref)/dt", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("velocity (deg/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if command_window is not None:
        for ax in axes:
            ax.axvline(command_window[0], color="tab:green", alpha=0.5)
            ax.axvline(command_window[1], color="tab:red", alpha=0.5)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _main():
    parser = argparse.ArgumentParser(description="Measure MoveJB2 single-joint velocity profile.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--joint", type=int, default=1, choices=range(1, 7), help="Joint number, 1-6.")
    parser.add_argument("--delta", type=float, default=20.0, help="Joint target delta in deg.")
    parser.add_argument("--speed", type=float, default=80.0, help="MoveJB2 speed argument.")
    parser.add_argument("--accel", type=float, default=100.0, help="MoveJB2 acceleration argument.")
    parser.add_argument("--blend", type=float, default=0.0, help="MoveJB2 blending value.")
    parser.add_argument("--sample-hz", type=float, default=100.0)
    parser.add_argument("--pre-record", type=float, default=0.5)
    parser.add_argument("--post-record", type=float, default=1.0)
    parser.add_argument("--start-timeout", type=float, default=5.0)
    parser.add_argument("--finish-timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("speed_profile_results"))
    parser.add_argument("--real", action="store_true", help="Use real robot mode. Default is Simulation.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned target and exit.")
    args = parser.parse_args()

    joint_idx = args.joint - 1
    mode_name = "Real" if args.real else "Simulation"

    data_ch = rb.CobotData(args.robot_ip)
    start_ang, start_ref = _read_state(data_ch)
    start = start_ang if args.real else start_ref
    target = start.copy()
    target[joint_idx] += args.delta

    print(f"mode={mode_name}")
    print(f"start joints: {np.round(start, 4).tolist()}")
    print(f"target joints: {np.round(target, 4).tolist()}")
    print(f"test joint=J{args.joint}, delta={args.delta} deg, speed={args.speed}, accel={args.accel}")

    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"jb2_J{args.joint}_d{args.delta:g}_v{args.speed:g}_a{args.accel:g}_{stamp}"
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"

    records = []
    stop_event = threading.Event()
    rec_thread = threading.Thread(
        target=_record_loop,
        args=(args.robot_ip, stop_event, records, args.sample_hz),
        daemon=True,
    )
    rec_thread.start()

    robot = rb.Cobot(args.robot_ip)
    rc = rb.ResponseCollector()

    command_start = None
    command_end = None
    try:
        mode = rb.OperationMode.Real if args.real else rb.OperationMode.Simulation
        robot.set_operation_mode(rc, mode)
        robot.set_speed_bar(rc, 1.0)
        robot.flush(rc)
        _check(rc)

        time.sleep(args.pre_record)

        robot.move_jb2_clear(rc)
        robot.move_jb2_add(rc, target, args.speed, args.accel, args.blend)
        robot.flush(rc)
        _check(rc)

        command_start = records[-1]["t"] if records else 0.0
        robot.move_jb2_run(rc)
        robot.flush(rc)

        started = robot.wait_for_move_started(rc, args.start_timeout)
        if started.type() == rb.ReturnType.Success:
            robot.wait_for_move_finished(rc, args.finish_timeout)
        _check(rc)
        command_end = records[-1]["t"] if records else command_start

        time.sleep(args.post_record)
    finally:
        stop_event.set()
        rec_thread.join(timeout=2.0)

    if len(records) < 2:
        raise RuntimeError("Not enough samples recorded")

    t, q_ang, q_ref, v_ang, v_ref = _save_csv(csv_path, records, joint_idx)
    title = (
        f"MoveJB2 speed profile J{args.joint}: "
        f"delta={args.delta:g} deg, speed={args.speed:g}, accel={args.accel:g}, mode={mode_name}"
    )
    _save_plot(png_path, t, q_ang, q_ref, v_ang, v_ref, title, (command_start, command_end))

    print(f"samples: {len(records)}")
    print(f"measured peak |d(jnt_ang)/dt|: {np.max(np.abs(v_ang)):.3f} deg/s")
    print(f"reference peak |d(jnt_ref)/dt|: {np.max(np.abs(v_ref)):.3f} deg/s")
    print(f"csv: {csv_path}")
    print(f"plot: {png_path}")


if __name__ == "__main__":
    _main()
