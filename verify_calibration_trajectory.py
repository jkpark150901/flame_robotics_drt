"""
verify_calibration_trajectory.py
================================
캘리브레이션 파일을 적용한 Motive/NatNet 좌표가 로봇 TCP와 잘 동기화되는지
trajectory CSV를 실행하면서 확인합니다.

입력 trajectory CSV 형식:
  Order,J1,J2,J3,J4,J5,J6,Speed,Acceleration
"""

import argparse
import asyncio
import csv
import datetime
import ipaddress
import json
import logging
import socket
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import rbpodo as rb
from tools.NatNet.NatNetClient import NatNetClient


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

_natnet_file_log = logging.getLogger("NatNetPython.NatNetClient")
_natnet_file_log.setLevel(logging.DEBUG)
_natnet_file_log.propagate = False
_mocap_log_path = Path("mocap_frames_verify.log")
if not _natnet_file_log.handlers:
    _natnet_fh = logging.FileHandler(_mocap_log_path, encoding="utf-8")
    _natnet_fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S"))
    _natnet_file_log.addHandler(_natnet_fh)


class NatNetState:
    def __init__(self):
        self._lock = threading.Lock()
        self.rb_pos: np.ndarray | None = None
        self.rb_rot: np.ndarray | None = None
        self.rb_time: float | None = None

    def update_rb(self, pos, quat_xyzw):
        with self._lock:
            self.rb_pos = np.asarray(pos, dtype=float)
            self.rb_rot = np.asarray(quat_xyzw, dtype=float)
            self.rb_time = time.time()

    def get_rb(self):
        with self._lock:
            if self.rb_pos is None:
                return None, None, None
            return self.rb_pos.copy(), self.rb_rot.copy(), self.rb_time


_natnet = NatNetState()
_rb_id = 0


def _on_rigid_body(rb_id, position, rotation):
    if rb_id == _rb_id:
        _natnet.update_rb(position, rotation)


def _resolve_client_ip(server_ip: str, client_ip: str | None) -> str:
    if client_ip and client_ip.lower() != "auto":
        return client_ip

    server_addr = socket.gethostbyname(server_ip)
    if ipaddress.ip_address(server_addr).is_loopback:
        return "127.0.0.1"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((server_addr, 1510))
        return sock.getsockname()[0]
    finally:
        sock.close()


def _validate_ip_pair(server_ip: str, client_ip: str):
    server_addr = ipaddress.ip_address(socket.gethostbyname(server_ip))
    client_addr = ipaddress.ip_address(socket.gethostbyname(client_ip))
    if server_addr.is_loopback != client_addr.is_loopback:
        raise ValueError(
            f"server={server_ip}, client={client_ip} 조합이 맞지 않습니다. "
            "원격 Motive 서버를 쓸 때 client는 이 PC의 같은 네트워크 대역 IP여야 합니다."
        )


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    T_align = np.asarray(data.get("T_base_motive", data["T_align"]), dtype=float)
    if "T_rb_tcp" in data:
        T_rigidbody_tcp = np.asarray(data["T_rb_tcp"], dtype=float)
    elif "T_rigidbody_tcp" in data:
        T_rigidbody_tcp = np.asarray(data["T_rigidbody_tcp"], dtype=float)
    else:
        T_rigidbody_tcp = np.eye(4)
        T_rigidbody_tcp[:3, 3] = np.asarray(data.get("rb_to_tcp_offset_m", [0.0, 0.0, 0.0]), dtype=float)
    return T_align, T_rigidbody_tcp


def load_trajectory(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    waypoints = []
    for i, row in enumerate(rows, start=1):
        try:
            joints = np.array([float(row[f"J{j}"]) for j in range(1, 7)], dtype=float)
            speed = float(row.get("Speed") or 0.0)
            accel = float(row.get("Acceleration") or 0.0)
            order = int(float(row.get("Order") or i))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path} {i}번째 row 파싱 실패: {row}") from exc
        waypoints.append({"order": order, "joints": joints, "speed": speed, "accel": accel})

    if not waypoints:
        raise ValueError(f"trajectory CSV가 비어 있습니다: {path}")
    return waypoints


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=float)
    n = x * x + y * y + z * z + w * w
    if n <= 0.0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.eye(3)
    axis = rotvec / angle
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def tcp_raw_to_matrix(tcp_raw: np.ndarray, orientation_type: str = "zyx_euler_deg") -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(tcp_raw[:3], dtype=float) / 1000.0
    r = np.asarray(tcp_raw[3:6], dtype=float)
    if orientation_type == "zyx_euler_deg":
        rx, ry, rz = np.radians(r)
        R = _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)
    elif orientation_type == "xyz_euler_deg":
        rx, ry, rz = np.radians(r)
        R = _rot_x(rx) @ _rot_y(ry) @ _rot_z(rz)
    elif orientation_type == "rotvec_deg":
        R = matrix_from_rotvec(np.radians(r))
    elif orientation_type == "rotvec_rad":
        R = matrix_from_rotvec(r)
    else:
        raise ValueError(f"unknown orientation_type: {orientation_type}")
    T[:3, :3] = R
    return T


def rb_pose_to_matrix(pos_m: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(pos_m, dtype=float)
    T[:3, :3] = quat_xyzw_to_matrix(quat_xyzw)
    return T


def transform_pose(T_align: np.ndarray, point_m: np.ndarray, quat_xyzw: np.ndarray, T_rigidbody_tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T_motive_tcp = rb_pose_to_matrix(point_m, quat_xyzw) @ T_rigidbody_tcp
    return T_align @ T_motive_tcp, T_motive_tcp


class LiveTrajectoryPlot:
    def __init__(self, enabled: bool, grid_size: tuple[float, float, float], grid_center: tuple[float, float, float]):
        self.enabled = enabled
        self.grid_size = np.asarray(grid_size, dtype=float)
        self.grid_center = np.asarray(grid_center, dtype=float)
        self.fig = None
        self.ax = None
        self.err_ax = None
        self.robot_line = None
        self.mocap_line = None
        self.robot_current = None
        self.mocap_current = None
        self.error_line = None
        self.error_text = None
        if not enabled:
            return

        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        grid = self.fig.add_gridspec(3, 1, height_ratios=[2.2, 0.8, 0.05])
        self.ax = self.fig.add_subplot(grid[0], projection="3d")
        self.err_ax = self.fig.add_subplot(grid[1])
        self.robot_line, = self.ax.plot([], [], [], "b-", label="Robot TCP")
        self.mocap_line, = self.ax.plot([], [], [], "r-", label="Mocap aligned")
        self.robot_current = self.ax.scatter([], [], [], c="blue", s=35)
        self.mocap_current = self.ax.scatter([], [], [], c="red", s=35)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.legend()
        self.ax.set_title("Calibration Verification")
        self._apply_fixed_grid()

        self.error_line, = self.err_ax.plot([], [], "k-", label="error")
        self.err_ax.set_xlabel("Time [s]")
        self.err_ax.set_ylabel("Error [mm]")
        self.err_ax.grid(True, alpha=0.3)
        self.err_ax.legend(loc="upper right")
        self.error_text = self.err_ax.text(0.01, 0.92, "", transform=self.err_ax.transAxes, va="top")
        self.fig.tight_layout()

    def _apply_fixed_grid(self):
        half = self.grid_size / 2.0
        mins = self.grid_center - half
        maxs = self.grid_center + half
        self.ax.set_xlim(mins[0], maxs[0])
        self.ax.set_ylim(mins[1], maxs[1])
        self.ax.set_zlim(mins[2], maxs[2])

    def update(
        self,
        elapsed_values: list[float],
        robot_points: list[np.ndarray],
        mocap_points: list[np.ndarray],
        error_values: list[float],
    ):
        if not self.enabled or self.fig is None or self.ax is None or self.err_ax is None:
            return
        if not robot_points:
            return

        robot = np.asarray(robot_points, dtype=float)
        mocap = np.asarray(mocap_points, dtype=float)
        self.robot_line.set_data(robot[:, 0], robot[:, 1])
        self.robot_line.set_3d_properties(robot[:, 2])

        mocap_valid = mocap[~np.isnan(mocap).any(axis=1)]
        if len(mocap_valid) > 0:
            self.mocap_line.set_data(mocap_valid[:, 0], mocap_valid[:, 1])
            self.mocap_line.set_3d_properties(mocap_valid[:, 2])
            combined = np.vstack([robot, mocap_valid])
        else:
            combined = robot

        self.robot_current._offsets3d = ([robot[-1, 0]], [robot[-1, 1]], [robot[-1, 2]])
        if not np.isnan(mocap[-1]).any():
            self.mocap_current._offsets3d = ([mocap[-1, 0]], [mocap[-1, 1]], [mocap[-1, 2]])

        self._apply_fixed_grid()

        elapsed = np.asarray(elapsed_values, dtype=float)
        errors = np.asarray(error_values, dtype=float)
        valid = ~np.isnan(errors)
        if len(elapsed) > 0:
            self.error_line.set_data(elapsed[valid], errors[valid])
            self.err_ax.set_xlim(0.0, max(1.0, float(elapsed[-1])))
            if np.any(valid):
                ymax = max(1.0, float(np.nanmax(errors)) * 1.2)
                self.err_ax.set_ylim(0.0, ymax)
                rmse = float(np.sqrt(np.nanmean(errors[valid] ** 2)))
                self.error_text.set_text(f"now {errors[valid][-1]:.2f} mm | RMSE {rmse:.2f} mm")

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def save(self, path: Path):
        if self.enabled and self.fig is not None:
            self.fig.savefig(path, dpi=150)

    def hold(self):
        if self.enabled:
            plt.ioff()
            plt.show()


async def upload_trajectory(args, robot, waypoints: list[dict]):
    rc = rb.ResponseCollector()
    await robot.flush(rc)
    rc.error().throw_if_not_empty()

    log.info("%d 개 waypoint 업로드 중: %s", len(waypoints), args.trajectory)
    await robot.move_jb2_clear(rc)
    for wp in waypoints:
        speed = wp["speed"] if wp["speed"] > 0 else args.speed
        accel = wp["accel"] if wp["accel"] > 0 else args.accel
        await robot.move_jb2_add(rc, wp["joints"], speed, accel, args.blending)
    await robot.flush(rc)
    rc.error().throw_if_not_empty()


async def run_and_record(
    args,
    robot,
    data_channel,
    T_align: np.ndarray,
    T_rigidbody_tcp: np.ndarray,
    plotter: LiveTrajectoryPlot,
):
    rc = rb.ResponseCollector()
    records = []
    elapsed_values = []
    robot_points = []
    mocap_points = []
    error_values = []
    recording = True
    last_plot = 0.0
    last_mocap_warning = 0.0
    prev_tcp_pos_m = None
    prev_rb_pos = None

    async def recorder():
        nonlocal last_plot, last_mocap_warning, prev_tcp_pos_m, prev_rb_pos
        while recording:
            now = time.time()
            data = await data_channel.request_data()
            tcp_raw = np.asarray(data.sdata.tcp_ref, dtype=float)
            tcp_pos_m = tcp_raw[:3] / 1000.0
            T_tcp = tcp_raw_to_matrix(tcp_raw, args.tcp_orientation_type)

            rb_pos, rb_rot, rb_time = _natnet.get_rb()
            if prev_tcp_pos_m is None:
                tcp_step_mm = 0.0
                rb_step_mm = 0.0
            else:
                tcp_step_mm = float(np.linalg.norm(tcp_pos_m - prev_tcp_pos_m) * 1000.0)
                rb_step_mm = float(np.linalg.norm(rb_pos - prev_rb_pos) * 1000.0) if rb_pos is not None and prev_rb_pos is not None else np.nan

            mocap_age_ms = (now - rb_time) * 1000.0 if rb_time is not None else np.nan
            mocap_status = "ok"
            if rb_pos is None:
                mocap_status = "missing"
            elif mocap_age_ms > args.max_mocap_age_ms:
                mocap_status = "stale_age"
            elif (
                prev_tcp_pos_m is not None
                and tcp_step_mm >= args.stale_check_tcp_step_mm
                and rb_step_mm <= args.stale_mocap_step_mm
            ):
                mocap_status = "stale_pose"

            mocap_valid = mocap_status == "ok" or not args.reject_stale_mocap
            if mocap_status != "ok" and now - last_mocap_warning >= 1.0:
                log.warning(
                    "mocap 상태 이상: %s | age=%s ms, tcp_step=%.2f mm, rb_step=%s mm",
                    mocap_status,
                    "n/a" if np.isnan(mocap_age_ms) else f"{mocap_age_ms:.1f}",
                    tcp_step_mm,
                    "n/a" if np.isnan(rb_step_mm) else f"{rb_step_mm:.2f}",
                )
                last_mocap_warning = now

            if rb_pos is not None and mocap_valid:
                T_pred, T_motive_tcp = transform_pose(T_align, rb_pos, rb_rot, T_rigidbody_tcp)
                rb_aligned = T_pred[:3, 3]
                rb_corrected = T_motive_tcp[:3, 3]
                error_mm = float(np.linalg.norm(tcp_pos_m - rb_aligned) * 1000.0)
                rotation_error_deg = float(np.linalg.norm(rotvec_from_matrix(T_tcp[:3, :3].T @ T_pred[:3, :3])) * 180.0 / np.pi)
            else:
                if rb_pos is None:
                    rb_pos = np.full(3, np.nan)
                    rb_rot = np.full(4, np.nan)
                rb_corrected = np.full(3, np.nan)
                rb_aligned = np.full(3, np.nan)
                error_mm = np.nan
                rotation_error_deg = np.nan

            elapsed = now - t_start
            records.append({
                "elapsed_s": round(elapsed, 4),
                "tcp_x_m": round(float(tcp_pos_m[0]), 6),
                "tcp_y_m": round(float(tcp_pos_m[1]), 6),
                "tcp_z_m": round(float(tcp_pos_m[2]), 6),
                "tcp_rx_deg": round(float(tcp_raw[3]), 4),
                "tcp_ry_deg": round(float(tcp_raw[4]), 4),
                "tcp_rz_deg": round(float(tcp_raw[5]), 4),
                "rb_raw_x_m": round(float(rb_pos[0]), 6),
                "rb_raw_y_m": round(float(rb_pos[1]), 6),
                "rb_raw_z_m": round(float(rb_pos[2]), 6),
                "rb_qx": round(float(rb_rot[0]), 6),
                "rb_qy": round(float(rb_rot[1]), 6),
                "rb_qz": round(float(rb_rot[2]), 6),
                "rb_qw": round(float(rb_rot[3]), 6),
                "rb_corrected_x_m": round(float(rb_corrected[0]), 6),
                "rb_corrected_y_m": round(float(rb_corrected[1]), 6),
                "rb_corrected_z_m": round(float(rb_corrected[2]), 6),
                "rb_aligned_x_m": round(float(rb_aligned[0]), 6),
                "rb_aligned_y_m": round(float(rb_aligned[1]), 6),
                "rb_aligned_z_m": round(float(rb_aligned[2]), 6),
                "mocap_status": mocap_status,
                "mocap_age_ms": round(float(mocap_age_ms), 3),
                "tcp_step_mm": round(float(tcp_step_mm), 4),
                "rb_step_mm": round(float(rb_step_mm), 4) if not np.isnan(rb_step_mm) else "",
                "error_mm": round(float(error_mm), 4),
                "rotation_error_deg": round(float(rotation_error_deg), 4),
            })
            elapsed_values.append(elapsed)
            robot_points.append(tcp_pos_m)
            mocap_points.append(rb_aligned)
            error_values.append(error_mm)
            prev_tcp_pos_m = tcp_pos_m.copy()
            if rb_pos is not None and not np.isnan(rb_pos).any():
                prev_rb_pos = rb_pos.copy()

            if args.plot and now - last_plot >= args.plot_period:
                plotter.update(elapsed_values, robot_points, mocap_points, error_values)
                last_plot = now

            await asyncio.sleep(args.sample_period)

    t_start = time.time()
    rec_task = asyncio.create_task(recorder())

    await robot.move_jb2_run(rc)
    if (await robot.wait_for_move_started(rc, 3.0)).type() == rb.ReturnType.Success:
        log.info("모션 시작 - robot TCP/mocap 동시 기록 중 ...")
        await robot.wait_for_move_finished(rc)
        log.info("모션 완료.")
    else:
        log.warning("모션 시작 확인 타임아웃. 기록을 종료합니다.")

    recording = False
    await rec_task
    plotter.update(elapsed_values, robot_points, mocap_points, error_values)
    return records


def save_records(records: list[dict], path: Path):
    if not records:
        log.warning("저장할 기록이 없습니다.")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    errors = np.array([row["error_mm"] for row in records], dtype=float)
    errors = errors[~np.isnan(errors)]
    rot_errors = np.array([row.get("rotation_error_deg", np.nan) for row in records], dtype=float)
    rot_errors = rot_errors[~np.isnan(rot_errors)]
    if len(errors) > 0:
        log.info("결과 CSV 저장: %s", path)
        log.info("오차: RMSE %.3f mm, mean %.3f mm, max %.3f mm", np.sqrt(np.mean(errors**2)), np.mean(errors), np.max(errors))
        if len(rot_errors) > 0:
            log.info(
                "회전 오차: RMSE %.3f deg, mean %.3f deg, max %.3f deg",
                np.sqrt(np.mean(rot_errors**2)),
                np.mean(rot_errors),
                np.max(rot_errors),
            )
    else:
        log.info("결과 CSV 저장: %s (mocap 매칭 데이터 없음)", path)


async def _async_main(args):
    global _rb_id
    _rb_id = args.rigid_body_id

    trajectory_path = Path(args.trajectory)
    cal_path = Path(args.cal_file)
    waypoints = load_trajectory(trajectory_path)
    T_align, T_rigidbody_tcp = load_calibration(cal_path)
    log.info("캘리브레이션 translation: [%.3f %.3f %.3f] m", *T_align[:3, 3])
    log.info("Rigid body -> TCP translation: [%.2f %.2f %.2f] mm", *(T_rigidbody_tcp[:3, 3] * 1000.0))

    client_ip = _resolve_client_ip(args.server, args.client)
    _validate_ip_pair(args.server, client_ip)

    client = NatNetClient()
    client.set_client_address(client_ip)
    client.set_server_address(args.server)
    client.rigid_body_listener = _on_rigid_body
    client.set_use_multicast(args.multicast)
    client.set_print_level(args.mocap_log_interval)

    if not client.run("d"):
        raise RuntimeError("NatNet 스트리밍 시작 실패.")
    time.sleep(1.0)
    if not client.connected():
        raise RuntimeError("NatNet 서버 연결 실패. Motive 스트리밍 설정을 확인하세요.")
    log.info("NatNet 연결 완료 (%s -> %s), 상세 로그: %s", client_ip, args.server, _mocap_log_path)

    try:
        robot = rb.asyncio.Cobot(args.robot_ip)
        data_channel = rb.asyncio.CobotData(args.robot_ip)
        rc = rb.ResponseCollector()

        await robot.set_operation_mode(rc, rb.OperationMode.Real)
        await robot.set_speed_bar(rc, args.speed_bar)
        await robot.flush(rc)
        rc.error().throw_if_not_empty()
        log.info("로봇 연결 완료 (%s), Real 모드", args.robot_ip)

        if args.move_to_first:
            first_q = waypoints[0]["joints"]
            log.info("첫 waypoint로 이동 중: %s deg", np.round(first_q, 2).tolist())
            await robot.move_j(rc, first_q, args.speed, args.accel)
            if (await robot.wait_for_move_started(rc, 3.0)).type() == rb.ReturnType.Success:
                await robot.wait_for_move_finished(rc)
            rc.error().throw_if_not_empty()
            await asyncio.sleep(args.settle_time)

        await upload_trajectory(args, robot, waypoints)
        log.info("준비 완료. Enter 를 누르면 trajectory 실행 및 동기화 검증을 시작합니다.")
        await asyncio.get_event_loop().run_in_executor(None, input)

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output) if args.output else Path(f"calibration_verify_{stamp}.csv")
        plot_path = Path(args.plot_output) if args.plot_output else Path(f"calibration_verify_{stamp}.png")

        plotter = LiveTrajectoryPlot(
            args.plot,
            grid_size=(args.plot_x_size, args.plot_y_size, args.plot_z_size),
            grid_center=(args.plot_x_center, args.plot_y_center, args.plot_z_center),
        )
        records = await run_and_record(args, robot, data_channel, T_align, T_rigidbody_tcp, plotter)
        save_records(records, output_path)
        plotter.save(plot_path)
        if args.plot:
            log.info("플롯 이미지 저장: %s", plot_path)
        if args.hold_plot:
            plotter.hold()

    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("사용자 중단 - 종료 중 ...")
    finally:
        client.shutdown()


def main():
    p = argparse.ArgumentParser(description="캘리브레이션 적용 후 robot TCP와 mocap 궤적 동기화 검증")
    p.add_argument("trajectory", nargs="?", default="robot1_trajectory_rbpodo_ready.csv",
                   help="trajectory CSV 경로")
    p.add_argument("--cal_file", default="calibration_svd.json", help="캘리브레이션 JSON 경로")
    p.add_argument("--output", default=None, help="결과 CSV 경로")
    p.add_argument("--plot_output", default=None, help="저장할 플롯 이미지 경로")
    p.add_argument("--rigid_body_id", type=int, required=True, help="NatNet rigid body ID")
    p.add_argument("--robot_ip", default="10.0.2.7")
    p.add_argument("--server", default="127.0.0.1", help="NatNet 서버 IP")
    p.add_argument("--client", default="auto", help="NatNet 클라이언트 IP (기본값: auto)")
    p.add_argument("--multicast", action="store_true", default=False)
    p.add_argument("--mocap_log_interval", type=int, default=0, help="mocap frame 상세 파일 로그 간격. 0이면 비활성화")
    p.add_argument("--speed", type=float, default=20)
    p.add_argument("--accel", type=float, default=40)
    p.add_argument("--blending", type=float, default=5.0)
    p.add_argument("--speed_bar", type=float, default=0.3)
    p.add_argument("--sample_period", type=float, default=0.02, help="기록 주기 s")
    p.add_argument("--max_mocap_age_ms", type=float, default=250.0,
                   help="이 값보다 오래된 mocap frame은 stale_age로 처리")
    p.add_argument("--reject_stale_mocap", action=argparse.BooleanOptionalAction, default=True,
                   help="missing/stale mocap sample은 오차 계산과 플롯에서 제외")
    p.add_argument("--stale_check_tcp_step_mm", type=float, default=5.0,
                   help="stale_pose 검사를 시작할 최소 로봇 TCP 이동량 mm")
    p.add_argument("--stale_mocap_step_mm", type=float, default=0.2,
                   help="로봇은 움직였는데 mocap 이동량이 이 값 이하이면 stale_pose로 처리")
    p.add_argument(
        "--tcp_orientation_type",
        choices=["zyx_euler_deg", "xyz_euler_deg", "rotvec_deg", "rotvec_rad"],
        default="zyx_euler_deg",
    )
    p.add_argument("--plot_period", type=float, default=0.1, help="실시간 플롯 갱신 주기 s")
    p.add_argument("--plot_x_size", type=float, default=3.0, help="3D 플롯 X축 고정 크기 m")
    p.add_argument("--plot_y_size", type=float, default=3.0, help="3D 플롯 Y축 고정 크기 m")
    p.add_argument("--plot_z_size", type=float, default=2.0, help="3D 플롯 Z축 고정 크기 m")
    p.add_argument("--plot_x_center", type=float, default=0.0, help="3D 플롯 X축 중심 m")
    p.add_argument("--plot_y_center", type=float, default=0.0, help="3D 플롯 Y축 중심 m")
    p.add_argument("--plot_z_center", type=float, default=1.0, help="3D 플롯 Z축 중심 m")
    p.add_argument("--settle_time", type=float, default=0.5, help="첫 waypoint 이동 후 안정화 대기 시간 s")
    p.add_argument("--move_to_first", action=argparse.BooleanOptionalAction, default=True,
                   help="실행 전 첫 waypoint로 먼저 이동")
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                   help="실시간 3D 플롯 표시")
    p.add_argument("--hold_plot", action="store_true", help="종료 후 플롯 창 유지")
    args = p.parse_args()

    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
