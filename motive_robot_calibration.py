"""
precision_eval_svd.py
=====================
로봇 엔드 이펙터 위치 정밀도 평가 (정지 자세 샘플링 기반 SVD 캘리브레이션)

동작 순서:
  1. 지정된 관절 템플릿에서 0번 조인트를 45도 단위로 회전하며 샘플 자세 생성
  2. 각 자세로 이동 후 안정화를 위해 대기
  3. 정지 상태에서 로봇 TCP 위치와 NatNet rigid body 위치를 샘플링
  4. 수집된 두 점군(Point Clouds)을 SVD(Kabsch) 알고리즘으로 매칭하여
     최적의 T_align (NatNet World -> Robot Base) 계산 및 JSON 저장
  5. 계산된 T_align을 데이터에 적용하여 오차(RMSE) 확인 및 CSV 저장
"""

# Calibration model convention
#
# T_base_tcp_i = T_base_motive @ T_motive_rb_i @ T_rb_tcp
#
# T_base_tcp    : ^B T_TCP, robot FK/controller TCP pose
# T_motive_rb   : ^M T_RB, mocap rigid body pose
# T_base_motive : ^B T_M, transform from mocap world to robot base
# T_rb_tcp      : ^RB T_TCP, transform from mocap rigid body frame to robot TCP frame

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


import numpy as np
from calibration.solver import (
    quat_xyzw_to_matrix, average_quaternions_xyzw,
    matrix_from_rotvec, rotvec_from_matrix, average_rotations, average_transforms,
    invert_transform, se3_from_vec, vec_from_se3,
    tcp_raw_to_matrix, rb_pose_to_matrix,
    _compute_svd_transform, _rmse_mm, _residuals_mm,
    compute_T_align_svd, compute_T_align_with_rb_offset, solve_calibration,
    OPENCV_HANDEYE_METHODS, solve_handeye_opencv,
    handeye_residuals, handeye_absolute_residual, solve_handeye_absolute_ls,
    save_calibration, apply_calibration_and_save,
    _get_float, load_calibration_samples_csv,
)
_OPENCV_HANDEYE_METHODS = OPENCV_HANDEYE_METHODS

import rbpodo as rb
from tools.NatNet.NatNetClient import NatNetClient

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger(__name__)

_natnet_file_log = logging.getLogger('NatNetPython.NatNetClient')
_natnet_file_log.setLevel(logging.DEBUG)
_natnet_file_log.propagate = False
_mocap_log_path = Path('mocap_frames.log')
if not _natnet_file_log.handlers:
    _natnet_fh = logging.FileHandler(_mocap_log_path, encoding='utf-8')
    _natnet_fh.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S'))
    _natnet_file_log.addHandler(_natnet_fh)


# =============================================================================
# 정지 자세 샘플링용 관절 경로 생성
# =============================================================================
def _with_base_angle(template_deg: np.ndarray, base_angle_deg: float) -> np.ndarray:
    q = np.asarray(template_deg, dtype=float).copy()
    q[0] = base_angle_deg
    return q


def calibration_joint_poses(args) -> list[tuple[str, np.ndarray]]:
    if args.base_step_deg <= 0:
        raise ValueError('--base_step_deg 는 0보다 커야 합니다.')
    if args.sample_count <= 0:
        raise ValueError('--sample_count 는 0보다 커야 합니다.')

    base_angles = np.arange(
        args.base_start_deg,
        args.base_stop_deg,
        args.base_step_deg,
        dtype=float,
    )
    templates = [
        ('max_reach_j1_90', np.array(args.max_reach_j1_90, dtype=float)),
        ('max_reach_j1_45', np.array(args.max_reach_j1_45, dtype=float)),
        ('half_reach_down', np.array(args.half_reach_down_joints, dtype=float)),
        ('half_reach_up', np.array(args.half_reach_up_joints, dtype=float)),
    ]

    poses: list[tuple[str, np.ndarray]] = []
    for label, template in templates:
        if template.shape != (6,):
            raise ValueError(f'{label} 템플릿은 6개 joint 값을 가져야 합니다.')
        for base_angle in base_angles:
            for tool_roll in args.tool_roll_sweep_deg:
                q = _with_base_angle(template, base_angle)
                q[5] = template[5] + tool_roll
                poses.append((f'{label}_j0_{base_angle:g}_j5_{q[5]:g}', q))
    return poses


def save_calibration_plan_csv(path: str, sample_poses: list[tuple[str, np.ndarray]], speed: float, accel: float):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['pose_label', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'Speed', 'Acceleration'])
        writer.writeheader()
        for label, q_deg in sample_poses:
            writer.writerow({
                'pose_label': label,
                'J1': round(float(q_deg[0]), 6),
                'J2': round(float(q_deg[1]), 6),
                'J3': round(float(q_deg[2]), 6),
                'J4': round(float(q_deg[3]), 6),
                'J5': round(float(q_deg[4]), 6),
                'J6': round(float(q_deg[5]), 6),
                'Speed': speed,
                'Acceleration': accel,
            })
    log.info("캘리브레이션 pose plan CSV 저장 완료 → %s", path)


def load_calibration_plan_csv(path: str) -> list[tuple[str, np.ndarray]]:
    sample_poses = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for idx, row in enumerate(csv.DictReader(f), start=1):
            label = row.get('pose_label') or f'csv_pose_{idx}'
            q_deg = np.array([float(row[f'J{i}']) for i in range(1, 7)], dtype=float)
            sample_poses.append((label, q_deg))
    if not sample_poses:
        raise ValueError(f'캘리브레이션 pose plan CSV가 비어 있습니다: {path}')
    log.info("캘리브레이션 pose plan CSV 로드 완료: %s (%d poses)", path, len(sample_poses))
    return sample_poses


# =============================================================================
# NatNet 공유 상태
# =============================================================================
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

    def get_rb(self) -> tuple:
        with self._lock:
            if self.rb_pos is None: return None, None
            return self.rb_pos.copy(), self.rb_rot

    def get_rb_since(self, min_time: float) -> tuple[np.ndarray, np.ndarray, float] | None:
        with self._lock:
            if self.rb_pos is None or self.rb_time is None or self.rb_time < min_time:
                return None
            return self.rb_pos.copy(), self.rb_rot.copy(), self.rb_time

_natnet  = NatNetState()
_rb_id   = 1

def _on_rigid_body(rb_id, position, rotation):
    if rb_id == _rb_id:
        _natnet.update_rb(position, rotation)


# =============================================================================
# 정지 자세 실행 + RAW 데이터 수집
# =============================================================================
async def _wait_for_latest_rb(min_time: float, timeout_s: float = 2.0) -> tuple[np.ndarray, np.ndarray, float] | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rb_sample = _natnet.get_rb_since(min_time)
        if rb_sample is not None:
            return rb_sample
        await asyncio.sleep(0.02)
    return None


async def sample_stable_point(args, data_channel):
    tcp_samples = []
    rb_samples = []
    rb_rot_samples = []
    rb_time_samples = []

    for _ in range(args.sample_count):
        sample_time = time.time()
        data = await data_channel.request_data()
        tcp_raw = np.array(data.sdata.tcp_ref, dtype=float)
        rb_sample = await _wait_for_latest_rb(sample_time, args.mocap_timeout)
        if rb_sample is not None:
            rb_pos, rb_rot, rb_time = rb_sample
            tcp_samples.append(tcp_raw)
            rb_samples.append(rb_pos)
            rb_rot_samples.append(rb_rot)
            rb_time_samples.append(rb_time)
        await asyncio.sleep(args.sample_period)

    if len(rb_samples) < args.min_mocap_samples:
        return None, None, None, len(rb_samples)

    rb_rot = average_quaternions_xyzw(rb_rot_samples)
    unique_rb_times = len({round(t, 4) for t in rb_time_samples})
    if unique_rb_times < args.min_mocap_samples:
        log.warning(
            "정지 샘플 mocap frame timestamp가 충분히 갱신되지 않았습니다: unique=%d/%d",
            unique_rb_times,
            len(rb_time_samples),
        )

    return np.mean(tcp_samples, axis=0), np.mean(rb_samples, axis=0), rb_rot, len(rb_samples)


async def run_and_collect_stationary(args, robot, data_channel, sample_poses: list[tuple[str, np.ndarray]]):
    rc = rb.ResponseCollector()

    raw_records = []
    points_motive = []
    quat_motive = []
    points_tcp_m = []
    t_start = time.time()
    prev_tcp_pos_m = None
    prev_rb_pos = None

    log.info("%d 개 정지 자세 샘플링 시작", len(sample_poses))
    for idx, (label, q_deg) in enumerate(sample_poses, start=1):
        if args.step_confirm:
            input(f"[{idx:02d}/{len(sample_poses):02d}] {label} -> {np.round(q_deg, 1).tolist()} deg 이동하려면 Enter...")
        log.info("[%02d/%02d] 이동: %s -> %s deg", idx, len(sample_poses), label, np.round(q_deg, 1).tolist())
        await robot.move_j(rc, q_deg, args.speed, args.accel)
        if (await robot.wait_for_move_started(rc, 3.0)).type() == rb.ReturnType.Success:
            await robot.wait_for_move_finished(rc)
        else:
            log.warning("[%02d/%02d] 모션 시작 확인 타임아웃: %s", idx, len(sample_poses), label)
        rc.error().throw_if_not_empty()

        await asyncio.sleep(args.settle_time)
        tcp_raw, rb_pos, rb_rot, mocap_valid_samples = await sample_stable_point(args, data_channel)
        if tcp_raw is None or rb_pos is None:
            log.warning(
                "[%02d/%02d] mocap 유효 샘플 부족: %s (%d/%d)",
                idx,
                len(sample_poses),
                label,
                mocap_valid_samples,
                args.sample_count,
            )
            continue

        elapsed = time.time() - t_start
        tcp_pos_m = tcp_raw[:3] / 1000.0
        if prev_tcp_pos_m is None:
            tcp_step_mm = 0.0
            rb_step_mm = 0.0
            step_ratio = np.nan
            step_delta_mm = 0.0
        else:
            tcp_step_mm = float(np.linalg.norm(tcp_pos_m - prev_tcp_pos_m) * 1000.0)
            rb_step_mm = float(np.linalg.norm(rb_pos - prev_rb_pos) * 1000.0)
            step_ratio = rb_step_mm / tcp_step_mm if tcp_step_mm > 1e-9 else np.nan
            step_delta_mm = rb_step_mm - tcp_step_mm

        stale_mocap = (
            prev_tcp_pos_m is not None
            and tcp_step_mm >= args.stale_check_tcp_step_mm
            and rb_step_mm <= args.stale_mocap_step_mm
        )
        if stale_mocap:
            log.warning(
                "[%02d/%02d] mocap tracking stale 의심: %s | tcp step=%.1f mm, mocap step=%.1f mm "
                "(threshold tcp>=%.1f, mocap<=%.1f)",
                idx,
                len(sample_poses),
                label,
                tcp_step_mm,
                rb_step_mm,
                args.stale_check_tcp_step_mm,
                args.stale_mocap_step_mm,
            )
            if args.reject_stale_mocap:
                log.warning("[%02d/%02d] stale mocap 샘플 제외: %s", idx, len(sample_poses), label)
                continue
        prev_tcp_pos_m = tcp_pos_m.copy()
        prev_rb_pos = rb_pos.copy()

        points_tcp_m.append(tcp_pos_m)
        points_motive.append(rb_pos)
        quat_motive.append(rb_rot)

        raw_records.append({
            'elapsed_s': round(elapsed, 4),
            'pose_label': label,
            'joint_0_deg': round(float(q_deg[0]), 4),
            'joint_1_deg': round(float(q_deg[1]), 4),
            'joint_2_deg': round(float(q_deg[2]), 4),
            'joint_3_deg': round(float(q_deg[3]), 4),
            'joint_4_deg': round(float(q_deg[4]), 4),
            'joint_5_deg': round(float(q_deg[5]), 4),
            'tcp_x_mm': round(float(tcp_raw[0]), 4),
            'tcp_y_mm': round(float(tcp_raw[1]), 4),
            'tcp_z_mm': round(float(tcp_raw[2]), 4),
            'tcp_rx_deg': round(float(tcp_raw[3]), 4),
            'tcp_ry_deg': round(float(tcp_raw[4]), 4),
            'tcp_rz_deg': round(float(tcp_raw[5]), 4),
            'rb_raw_x_m': round(float(rb_pos[0]), 6),
            'rb_raw_y_m': round(float(rb_pos[1]), 6),
            'rb_raw_z_m': round(float(rb_pos[2]), 6),
            'rb_qx': round(float(rb_rot[0]), 6),
            'rb_qy': round(float(rb_rot[1]), 6),
            'rb_qz': round(float(rb_rot[2]), 6),
            'rb_qw': round(float(rb_rot[3]), 6),
            'mocap_valid_samples': mocap_valid_samples,
            'tcp_step_mm': round(tcp_step_mm, 4),
            'rb_step_mm': round(rb_step_mm, 4),
            'step_delta_mm': round(step_delta_mm, 4),
            'step_ratio': round(float(step_ratio), 6) if not np.isnan(step_ratio) else '',
        })
        log.info(
            "[%02d/%02d] 샘플 완료: 누적 %d 점 | step L2 tcp=%.1f mm, mocap=%.1f mm, diff=%+.1f mm, ratio=%s",
            idx,
            len(sample_poses),
            len(raw_records),
            tcp_step_mm,
            rb_step_mm,
            step_delta_mm,
            "n/a" if np.isnan(step_ratio) else f"{step_ratio:.3f}",
        )

    return raw_records, np.array(points_motive), np.array(quat_motive), np.array(points_tcp_m)


async def move_to_start(robot, rc, q_deg: np.ndarray):
    log.info("시작 자세로 이동 중: %s deg", np.round(q_deg, 1).tolist())
    await robot.move_j(rc, q_deg, 30, 60)
    if (await robot.wait_for_move_started(rc, 3.0)).type() == rb.ReturnType.Success:
        await robot.wait_for_move_finished(rc)
    rc.error().throw_if_not_empty()
    log.info("홈 자세 도달 완료.")




def calibrate_from_samples(args, raw_records, points_motive, quat_motive, points_tcp_m, source_name: str):
    if len(points_motive) < 10:
        raise RuntimeError(f'캘리브레이션 샘플이 너무 적습니다: {len(points_motive)}')

    log.info("=== SVD (Kabsch) 캘리브레이션 연산 중: %s ===", source_name)
    step_rows = [r for r in raw_records if r.get('tcp_step_mm', 0) not in ('', 0, 0.0)]
    if step_rows:
        tcp_steps = np.array([float(r['tcp_step_mm']) for r in step_rows])
        rb_steps = np.array([float(r['rb_step_mm']) for r in step_rows])
        step_diff = rb_steps - tcp_steps
        log.info(
            "Step L2 요약: tcp mean %.1f mm, mocap mean %.1f mm, diff mean %+.1f mm, diff max abs %.1f mm",
            float(np.mean(tcp_steps)),
            float(np.mean(rb_steps)),
            float(np.mean(step_diff)),
            float(np.max(np.abs(step_diff))),
        )
        worst_idx = int(np.argmax(np.abs(step_diff)))
        worst_row = step_rows[worst_idx]
        log.info(
            "Step L2 최대 차이: %s tcp=%.1f mm, mocap=%.1f mm, diff=%+.1f mm",
            worst_row['pose_label'],
            float(worst_row['tcp_step_mm']),
            float(worst_row['rb_step_mm']),
            float(worst_row['step_delta_mm']),
        )

    if args.calibration_model == 'handeye':
        T_motive_rb_list = [
            rb_pose_to_matrix(
                np.array([row['rb_raw_x_m'], row['rb_raw_y_m'], row['rb_raw_z_m']]),
                np.array([row['rb_qx'], row['rb_qy'], row['rb_qz'], row['rb_qw']]),
            )
            for row in raw_records
        ]
        T_base_tcp_list = [
            tcp_raw_to_matrix(np.array([
                row['tcp_x_mm'],
                row['tcp_y_mm'],
                row['tcp_z_mm'],
                row['tcp_rx_deg'],
                row['tcp_ry_deg'],
                row['tcp_rz_deg'],
            ]), args.tcp_orientation_type)
            for row in raw_records
        ]
        T_align, T_rigidbody_tcp = solve_handeye_opencv(
            T_motive_rb_list, T_base_tcp_list, args.opencv_handeye_method,
        )
        residuals, rot_residuals = handeye_residuals(T_align, T_rigidbody_tcp, T_motive_rb_list, T_base_tcp_list)
        inlier_mask = np.ones(len(raw_records), dtype=bool)
        if args.outlier_threshold_mm > 0.0:
            inlier_mask = residuals <= args.outlier_threshold_mm
            for idx in np.where(~inlier_mask)[0]:
                log.warning(
                    "캘리브레이션 outlier 제외: %s residual=%.1f mm",
                    raw_records[idx]['pose_label'],
                    residuals[idx],
                )
            if np.count_nonzero(inlier_mask) >= 6 and np.any(~inlier_mask):
                T_align, T_rigidbody_tcp = solve_handeye_opencv(
                    [T for T, ok in zip(T_motive_rb_list, inlier_mask) if ok],
                    [T for T, ok in zip(T_base_tcp_list, inlier_mask) if ok],
                    args.opencv_handeye_method,
                )
                residuals, rot_residuals = handeye_residuals(T_align, T_rigidbody_tcp, T_motive_rb_list, T_base_tcp_list)
            elif np.count_nonzero(inlier_mask) < 6:
                log.warning("outlier 제외 후 inlier가 %d개뿐이라 전체 샘플로 계산합니다.", np.count_nonzero(inlier_mask))
                inlier_mask[:] = True

        log.info("T_rigidbody_tcp translation: [%.2f %.2f %.2f] mm", *(T_rigidbody_tcp[:3, 3] * 1000.0))
        log.info(
            "최종 hand-eye position RMSE: inlier %.3f mm / all %.3f mm",
            float(np.sqrt(np.mean(residuals[inlier_mask] ** 2))),
            float(np.sqrt(np.mean(residuals ** 2))),
        )
        log.info(
            "최종 hand-eye rotation RMSE: inlier %.3f deg / all %.3f deg",
            float(np.sqrt(np.mean(rot_residuals[inlier_mask] ** 2))),
            float(np.sqrt(np.mean(rot_residuals ** 2))),
        )
        log.info("캘리브레이션 inlier: %d/%d", np.count_nonzero(inlier_mask), len(inlier_mask))
        for row, is_inlier in zip(raw_records, inlier_mask):
            row['calibration_inlier'] = int(bool(is_inlier))
        log.info("T_base_motive translation: [%.3f %.3f %.3f] m", *T_align[:3, 3])
        save_calibration(T_align, args.cal_file, T_rigidbody_tcp)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = args.output_csv or f"precision_eval_svd_{stamp}.csv"
        apply_calibration_and_save(raw_records, T_align, out_csv, T_rigidbody_tcp, args.tcp_orientation_type)
        return T_align, T_rigidbody_tcp

    T_origin = compute_T_align_svd(points_motive, points_tcp_m)
    origin_rmse = _rmse_mm(T_origin, points_motive, points_tcp_m)
    log.info("Rigid body origin 기준 SVD RMSE: %.3f mm", origin_rmse)

    labels = [row['pose_label'] for row in raw_records]
    T_align, rb_to_tcp_offset, corrected_motive, inlier_mask = solve_calibration(
        points_motive,
        quat_motive,
        points_tcp_m,
        args.calibrate_rigidbody_offset,
        args.outlier_threshold_mm,
        labels,
    )
    if args.calibrate_rigidbody_offset and rb_to_tcp_offset is not None:
        log.info("Rigid body local -> TCP offset 추정: [%.2f %.2f %.2f] mm", *(rb_to_tcp_offset * 1000.0))
    else:
        log.info("Rigid body offset 보정 비활성화: rigid body 원점이 TCP라고 가정합니다.")

    if args.calibrate_rigidbody_offset and rb_to_tcp_offset is not None:
        rotations = np.array([quat_xyzw_to_matrix(q) for q in quat_motive])
        corrected_all = np.array([
            p_motive + R_motive @ rb_to_tcp_offset
            for p_motive, R_motive in zip(points_motive, rotations)
        ])
    else:
        corrected_all = points_motive

    all_rmse = _rmse_mm(T_align, corrected_all, points_tcp_m)
    inlier_rmse = _rmse_mm(T_align, corrected_all[inlier_mask], points_tcp_m[inlier_mask])
    log.info("최종 SVD RMSE: inlier %.3f mm / all %.3f mm", inlier_rmse, all_rmse)
    log.info("캘리브레이션 inlier: %d/%d", np.count_nonzero(inlier_mask), len(inlier_mask))

    for row, is_inlier in zip(raw_records, inlier_mask):
        row['calibration_inlier'] = int(bool(is_inlier))

    log.info("T_align translation: [%.3f %.3f %.3f] m", *T_align[:3, 3])
    save_calibration(T_align, args.cal_file, rb_to_tcp_offset)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.output_csv or f"precision_eval_svd_{stamp}.csv"
    apply_calibration_and_save(raw_records, T_align, out_csv, rb_to_tcp_offset, args.tcp_orientation_type)
    return T_align, rb_to_tcp_offset


# =============================================================================
# Main (asyncio)
# =============================================================================

def _resolve_client_ip(server_ip: str, client_ip: str | None) -> str:
    if client_ip and client_ip.lower() != 'auto':
        return client_ip

    server_addr = socket.gethostbyname(server_ip)
    if ipaddress.ip_address(server_addr).is_loopback:
        return '127.0.0.1'

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect does not send packets; it lets the OS choose the outbound
        # interface that would reach the NatNet server.
        sock.connect((server_addr, 1510))
        return sock.getsockname()[0]
    finally:
        sock.close()


def _validate_ip_pair(server_ip: str, client_ip: str):
    server_addr = ipaddress.ip_address(socket.gethostbyname(server_ip))
    client_addr = ipaddress.ip_address(socket.gethostbyname(client_ip))

    if server_addr.is_loopback != client_addr.is_loopback:
        raise ValueError(
            f'server={server_ip}, client={client_ip} 조합이 맞지 않습니다. '
            '원격 Motive 서버를 쓸 때 client는 이 PC의 같은 네트워크 대역 IP여야 합니다.'
        )


async def _async_main(args):
    global _rb_id
    _rb_id  = args.rigid_body_id

    loop = asyncio.get_event_loop()

    # ── NatNet 연결 ───────────────────────────────────────────────────────────
    client_ip = _resolve_client_ip(args.server, args.client)
    _validate_ip_pair(args.server, client_ip)

    client = NatNetClient()
    client.set_client_address(client_ip)
    client.set_server_address(args.server)
    client.rigid_body_listener = _on_rigid_body
    client.set_use_multicast(args.multicast)
    client.set_print_level(args.mocap_log_interval)
    log.info("mocap frame 상세 로그 파일: %s (매 %d 프레임)", _mocap_log_path, args.mocap_log_interval)

    if not client.run('d'):
        raise RuntimeError("NatNet 스트리밍 시작 실패.")
    time.sleep(1.0)
    if not client.connected():
        raise RuntimeError("NatNet 서버 연결 실패. Motive 스트리밍 설정을 확인하세요.")
    log.info("NatNet 연결 완료 (%s → %s)", client_ip, args.server)

    try:
        # ── 로봇 연결 ─────────────────────────────────────────────────────────────
        robot        = rb.asyncio.Cobot(args.robot_ip)
        data_channel = rb.asyncio.CobotData(args.robot_ip)
        rc           = rb.ResponseCollector()

        await robot.set_operation_mode(rc, rb.OperationMode.Real)
        await robot.set_speed_bar(rc, args.speed_bar)
        await robot.flush(rc)
        rc.error().throw_if_not_empty()
        log.info("로봇 연결 완료 (%s), Real 모드", args.robot_ip)

        # ── 시작 자세 이동 ────────────────────────────────────────────────────────
        await move_to_start(robot, rc, np.array(args.start_joints))

        # ── 정지 샘플링 자세 생성 ────────────────────────────────────────────────
        if args.calibration_plan_csv:
            sample_poses = load_calibration_plan_csv(args.calibration_plan_csv)
        else:
            sample_poses = calibration_joint_poses(args)
        if args.save_calibration_plan_csv:
            save_calibration_plan_csv(args.save_calibration_plan_csv, sample_poses, args.speed, args.accel)
        log.info("샘플 자세 계획 완료: %d poses (0번 joint %.1f° 간격)", len(sample_poses), args.base_step_deg)

        # ── 모션 실행 및 RAW 데이터 수집 ──────────────────────────────────────────
        if args.step_confirm:
            log.info("준비 완료. 각 스텝마다 Enter 를 누르면 해당 자세로 이동합니다.")
        elif args.pause_before_start:
            log.info("준비 완료. Enter 를 눌러 정지 자세 샘플링을 시작 ...")
            await loop.run_in_executor(None, input)
        else:
            log.info("준비 완료. 정지 자세 샘플링을 자동으로 시작합니다.")

        raw_records, points_motive, quat_motive, points_tcp_m = await run_and_collect_stationary(
            args,
            robot,
            data_channel,
            sample_poses,
        )

        calibrate_from_samples(
            args,
            raw_records,
            points_motive,
            quat_motive,
            points_tcp_m,
            'live samples',
        )

        log.info("평가 완료. (저장된 CSV 파일을 플롯하여 궤적을 확인해보세요!)")

    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("사용자 중단 — 종료 중 ...")
    finally:
        client.shutdown()

# =============================================================================
# Entry point
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="로봇 정밀도 평가 (SVD 궤적 캘리브레이션 적용)")
    p.add_argument('--rigid_body_id', type=int, default=1, help='NatNet rigid body ID')
    p.add_argument('--robot_ip',      default='10.0.2.7')
    p.add_argument('--server',        default='192.168.0.241', help='NatNet 서버 IP')
    p.add_argument('--client',        default='auto', help='NatNet 클라이언트 IP (기본값: auto)')
    p.add_argument('--multicast',     action='store_true', default=False)
    p.add_argument('--mocap_log_interval', type=int, default=20, help='mocap frame 상세 파일 로그 간격. 0이면 비활성화')
    p.add_argument('--cal_file',      default='calibration_svd.json')
    p.add_argument('--recalibrate_from_csv', default=None,
                   help='기존 캘리브레이션 결과 CSV를 참조해 로봇 이동 없이 다시 캘리브레이션')
    p.add_argument('--output_csv', default=None, help='캘리브레이션 결과 CSV 저장 경로')
    p.add_argument('--calibration_plan_csv', default=None,
                   help='캘리브레이션에 사용할 pose plan CSV 경로 (pose_label,J1..J6 형식)')
    p.add_argument('--save_calibration_plan_csv', default=None,
                   help='생성/로드한 캘리브레이션 pose plan을 CSV로 저장')
    p.add_argument('--plan_only', action='store_true',
                   help='pose plan CSV만 저장하고 종료 (--save_calibration_plan_csv 필요)')
    p.add_argument('--speed',         type=float, default=400)
    p.add_argument('--accel',         type=float, default=200)
    p.add_argument('--speed_bar',     type=float, default=0.3)
    p.add_argument('--start_joints',  type=float, nargs=6, default=[0, 0, 0, 0, 0, 0])
    p.add_argument('--base_start_deg', type=float, default=0.0)
    p.add_argument('--base_stop_deg',  type=float, default=360.0)
    p.add_argument('--base_step_deg',  type=float, default=45.0)
    p.add_argument('--step_confirm', action=argparse.BooleanOptionalAction, default=False,
                   help='각 스텝 이동 전에 Enter 확인을 받음')
    p.add_argument('--pause_before_start', action=argparse.BooleanOptionalAction, default=False,
                   help='전체 샘플링 시작 전에 Enter 확인을 받음')
    p.add_argument('--settle_time',    type=float, default=0.5, help='각 모션 완료 후 샘플링 전 대기 시간 s')
    p.add_argument('--sample_count',   type=int,   default=5, help='정지 자세마다 평균낼 샘플 수')
    p.add_argument('--min_mocap_samples', type=int, default=3,
                   help='정지 자세 하나를 유효하게 인정하기 위한 최소 mocap 프레임 수')
    p.add_argument('--sample_period',  type=float, default=0.02, help='정지 자세 샘플 간격 s')
    p.add_argument('--mocap_timeout',  type=float, default=2.0, help='각 샘플에서 새 mocap 프레임 대기 시간 s')
    p.add_argument('--reject_stale_mocap', action=argparse.BooleanOptionalAction, default=True,
                   help='로봇 TCP는 움직였는데 mocap pose가 거의 고정이면 해당 샘플 제외')
    p.add_argument('--stale_check_tcp_step_mm', type=float, default=50.0,
                   help='stale mocap 검사를 시작할 최소 로봇 TCP 이동량 mm')
    p.add_argument('--stale_mocap_step_mm', type=float, default=2.0,
                   help='이 값 이하로만 mocap rigid body가 움직이면 stale로 의심할 이동량 mm')
    p.add_argument('--calibrate_rigidbody_offset', action=argparse.BooleanOptionalAction, default=True,
                   help='rigid body local 원점에서 TCP까지의 offset도 함께 추정')
    p.add_argument('--calibration_model', choices=['handeye', 'point'], default='point',
                   help='handeye: TCP pose와 rigid body pose의 R,t 전체 추정, point: 위치 점군 SVD')
    p.add_argument('--tcp_orientation_type',
                   choices=['zyx_euler_deg', 'xyz_euler_deg', 'rotvec_deg', 'rotvec_rad'],
                   default='zyx_euler_deg')
    p.add_argument('--min_relative_rotation_deg', type=float, default=10.0,
                   help='hand-eye closed-form 초기값 계산에 사용할 최소 상대 회전 deg')
    p.add_argument('--handeye_rot_weight', type=float, default=0.1,
                   help='absolute LS residual에서 rotation residual 가중치')
    p.add_argument('--handeye_loss_f_scale', type=float, default=0.01,
                   help='absolute LS Huber loss f_scale')
    p.add_argument('--outlier_threshold_mm', type=float, default=60.0,
                   help='초기 캘리브레이션 residual이 이 값보다 큰 샘플은 제외. 0이면 비활성화')
    p.add_argument('--max_reach_j1_90', type=float, nargs=6, default=[0, 90, 0, 0, 0, 0],
                   help='최대 리치/q1=90 템플릿. q0은 base sweep 값으로 대체')
    p.add_argument('--max_reach_j1_45', type=float, nargs=6, default=[0, 45, 0, 0, 0, 0],
                   help='최대 리치/q1=45 템플릿. q0은 base sweep 값으로 대체')
    p.add_argument('--half_reach_down_joints', type=float, nargs=6, default=[0, 45, -90, 0, 45, 0],
                   help='절반 리치 아래쪽 템플릿. q0은 base sweep 값으로 대체')
    p.add_argument('--half_reach_up_joints', type=float, nargs=6, default=[0, 45, 90, 0, -45, 0],
                   help='절반 리치 위쪽 템플릿. q0은 base sweep 값으로 대체')
    p.add_argument('--tool_roll_sweep_deg', type=float, nargs='+', default=[-60.0, 0.0, 60.0],
                   help='hand-eye R,t 추정을 위한 5번 joint 추가 회전 샘플 목록. 예: -60 0 60')
    p.add_argument('--opencv_handeye_method',
                   choices=list(_OPENCV_HANDEYE_METHODS.keys()),
                   default='park',
                   help='OpenCV calibrateHandEye 알고리즘 (handeye 모드 전용)')

    args = p.parse_args()
    args.opencv_handeye_method = _OPENCV_HANDEYE_METHODS[args.opencv_handeye_method]
    if args.min_mocap_samples <= 0:
        raise ValueError('--min_mocap_samples 는 0보다 커야 합니다.')
    if args.min_mocap_samples > args.sample_count:
        raise ValueError('--min_mocap_samples 는 --sample_count 보다 클 수 없습니다.')

    if args.plan_only:
        if not args.save_calibration_plan_csv:
            raise ValueError('--plan_only 는 --save_calibration_plan_csv 와 함께 사용해야 합니다.')
        sample_poses = load_calibration_plan_csv(args.calibration_plan_csv) if args.calibration_plan_csv else calibration_joint_poses(args)
        save_calibration_plan_csv(args.save_calibration_plan_csv, sample_poses, args.speed, args.accel)
        return

    if args.recalibrate_from_csv:
        raw_records, points_motive, quat_motive, points_tcp_m, has_quat = load_calibration_samples_csv(
            args.recalibrate_from_csv
        )
        if args.calibrate_rigidbody_offset and not has_quat:
            log.warning("CSV에 rb_qx/rb_qy/rb_qz/rb_qw가 없어 rigid body offset 추정을 비활성화합니다.")
            args.calibrate_rigidbody_offset = False
        calibrate_from_samples(
            args,
            raw_records,
            points_motive,
            quat_motive,
            points_tcp_m,
            args.recalibrate_from_csv,
        )
        return

    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
