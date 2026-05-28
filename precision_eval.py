"""
precision_eval.py
=================
로봇 엔드 이펙터 위치 정밀도 평가.

동작 순서:
  1. joint.yaml 에서 URDF 운동학 로드 → FK/IK 빌드
  2. 보정(calibration): 정지 자세에서 NatNet rigid body + 로봇 TCP 로
     T_align (NatNet 세계 좌표 → 로봇 베이스 좌표) 계산 후 JSON 저장
  3. 현재 TCP 중심으로 8자 (Lissajous) 경로 생성 → IK 풀기
  4. move_jb2 로 경로 실행
  5. 실행 중 로봇 TCP(rbpodo data channel) + NatNet 마커 위치를 동시 기록
  6. CSV 저장

사용법:
  python precision_eval.py --model_id 1 [--marker_id 0]
                           [--rigid_body_id 1]
                           [--robot_ip 10.0.2.7]
                           [--server 127.0.0.1] [--client 127.0.0.1]
                           [--multicast]
                           [--joint_yaml joint.yaml]
                           [--cal_file calibration.json] [--recalibrate]
                           [--radius 0.05]       # 8자 반지름 (m)
                           [--n_waypoints 60]
                           [--speed 20] [--accel 40]   # deg/s, deg/s²
                           [--blending 5.0]            # deg (joint space)
                           [--speed_bar 0.3]
"""

import argparse
import asyncio
import csv
import datetime
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

import rbpodo as rb
from NatNetClient import NatNetClient

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# =============================================================================
# URDF 기반 Forward Kinematics / IK
# =============================================================================

def _make_trans(xyz):
    T = np.eye(4)
    T[:3, 3] = np.array(xyz, dtype=float)
    return T


def _make_rpy(rpy):
    """URDF RPY (roll, pitch, yaw) → 4×4. 순서: Rz(yaw)*Ry(pitch)*Rx(roll)"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    return T


def _rot_axis(axis, angle_rad):
    """임의 축(unit vector) 중심 회전 → 4×4"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.array(axis, dtype=float) * angle_rad).as_matrix()
    return T


class URDFRobot:
    """
    joint.yaml 의 URDF 형식 운동 체인에서 FK/IK 를 제공합니다.
    단위: 위치 m, 각도 rad (내부). rbpodo 인터페이스는 deg 변환 메서드 사용.
    """

    def __init__(self, joint_yaml: str):
        with open(joint_yaml, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        self.joints = []
        for name, jd in data.items():
            origin = jd.get('origin', {})
            xyz = origin.get('xyz', [0, 0, 0])
            rpy = origin.get('rpy', [0, 0, 0])
            axis = jd.get('axis', {}).get('xyz', [0, 0, 1])
            lim  = jd.get('limit', {})
            self.joints.append({
                'name':   name,
                'T_fixed': _make_trans(xyz) @ _make_rpy(rpy),
                'axis':   np.array(axis, dtype=float),
                'lower':  float(lim.get('lower', -np.pi)),
                'upper':  float(lim.get('upper',  np.pi)),
            })
        self.n = len(self.joints)

    # ── Forward Kinematics ──────────────────────────────────────────────────

    def fk(self, q: np.ndarray) -> np.ndarray:
        """q: (n,) rad → 4×4 SE3 (위치 m)"""
        T = np.eye(4)
        for jnt, qi in zip(self.joints, q):
            T = T @ jnt['T_fixed'] @ _rot_axis(jnt['axis'], qi)
        return T

    # ── Numerical Jacobian ──────────────────────────────────────────────────

    def _jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """6×n Jacobian. 행 0-2: dp/dq (m/rad), 행 3-5: dω/dq (무차원)"""
        T0  = self.fk(q)
        p0  = T0[:3, 3]
        rv0 = Rotation.from_matrix(T0[:3, :3]).as_rotvec()
        J   = np.zeros((6, self.n))
        for i in range(self.n):
            qp = q.copy(); qp[i] += eps
            Ti  = self.fk(qp)
            J[:3, i] = (Ti[:3, 3] - p0) / eps
            J[3:, i] = (Rotation.from_matrix(Ti[:3, :3]).as_rotvec() - rv0) / eps
        return J

    # ── Damped Least-Squares IK ─────────────────────────────────────────────

    def ik(self, T_target: np.ndarray, q0: np.ndarray,
           max_iter: int = 300, pos_tol: float = 1e-4,
           ori_tol: float = 1e-3, lam: float = 0.05) -> tuple[np.ndarray, bool]:
        """
        DLS IK. 반환: (q_rad, success).
        pos_tol: m,  ori_tol: rad.
        """
        q    = q0.copy()
        R_t  = Rotation.from_matrix(T_target[:3, :3])
        dp = dr = np.zeros(3)

        for _ in range(max_iter):
            T   = self.fk(q)
            dp  = T_target[:3, 3] - T[:3, 3]
            dr  = (R_t * Rotation.from_matrix(T[:3, :3]).inv()).as_rotvec()
            if np.linalg.norm(dp) < pos_tol and np.linalg.norm(dr) < ori_tol:
                return q, True
            err = np.concatenate([dp, dr])
            J   = self._jacobian(q)
            dq  = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), err)
            q   = np.clip(q + dq,
                          [j['lower'] for j in self.joints],
                          [j['upper'] for j in self.joints])

        log.warning("IK 미수렴: pos=%.4f m  ori=%.4f rad",
                    np.linalg.norm(dp), np.linalg.norm(dr))
        return q, False

    # ── 단위 변환 ────────────────────────────────────────────────────────────

    @staticmethod
    def deg2rad(q_deg: np.ndarray) -> np.ndarray:
        return np.radians(q_deg)

    @staticmethod
    def rad2deg(q_rad: np.ndarray) -> np.ndarray:
        return np.degrees(q_rad)


# =============================================================================
# 좌표 변환 유틸
# =============================================================================

def tcp_to_matrix(tcp_mm_deg: np.ndarray) -> np.ndarray:
    """
    rbpodo TCP [x,y,z (mm), rx,ry,rz (deg, Euler ZYX intrinsic)] → 4×4 (m).
    rbpodo 문서: "Euler ZYX angles (rx,ry,rz)" with xyz vector ordering
    → scipy 'ZYX' 에 [rz, ry, rx] 순으로 전달.
    """
    T = np.eye(4)
    T[:3, 3] = tcp_mm_deg[:3] / 1000.0
    rx, ry, rz = tcp_mm_deg[3], tcp_mm_deg[4], tcp_mm_deg[5]
    T[:3, :3] = Rotation.from_euler('ZYX', [rz, ry, rx], degrees=True).as_matrix()
    return T


# =============================================================================
# 8자 (Lissajous) 경로 생성
# =============================================================================

def figure8_poses(T_center: np.ndarray, radius: float, n: int) -> list[np.ndarray]:
    """
    T_center 의 로컬 XY 평면에 8자 경로를 생성합니다.
    radius: 반지름 (m). 반환: n 개의 4×4 pose 리스트.
    """
    R = T_center[:3, :3]
    p = T_center[:3, 3]
    poses = []
    for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
        dx = radius * np.sin(t)
        dy = radius / 2.0 * np.sin(2 * t)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p + R @ np.array([dx, dy, 0.0])
        poses.append(T)
    return poses


# =============================================================================
# NatNet 공유 상태 (스레드 안전)
# =============================================================================

class NatNetState:
    def __init__(self):
        self._lock = threading.Lock()
        self.rb_pos: np.ndarray | None = None   # rigid body 위치 (m, NatNet 좌표)
        self.rb_rot: Rotation | None   = None   # rigid body 회전
        self.lm_pos: np.ndarray | None = None   # labeled marker 위치 (m, NatNet 좌표)

    def update_rb(self, pos, quat_xyzw):
        with self._lock:
            self.rb_pos = np.asarray(pos,       dtype=float)
            self.rb_rot = Rotation.from_quat(quat_xyzw)

    def update_lm(self, pos):
        with self._lock:
            self.lm_pos = np.asarray(pos, dtype=float)

    def get_rb(self) -> tuple:
        with self._lock:
            if self.rb_pos is None:
                return None, None
            return self.rb_pos.copy(), self.rb_rot

    def get_lm(self) -> np.ndarray | None:
        with self._lock:
            return self.lm_pos.copy() if self.lm_pos is not None else None


# 전역 상태 / NatNet 필터
_natnet  = NatNetState()
_mdl_id  = 0
_mkr_id  = 0
_rb_id   = 0


def _on_rigid_body(rb_id, position, rotation):
    """rotation: NatNet SDK v4 → (qx, qy, qz, qw)"""
    if rb_id == _rb_id:
        _natnet.update_rb(position, rotation)


def _on_frame(data_dict):
    if "mocap_data" not in data_dict:
        return
    lm_data = data_dict["mocap_data"].labeled_marker_data
    if lm_data is None:
        return
    for lm in lm_data.labeled_marker_list:
        mid  = lm.id_num >> 16
        mkid = lm.id_num & 0xFFFF
        if mid == _mdl_id and mkid == _mkr_id:
            _natnet.update_lm(lm.pos)   # NatNet 위치는 meters


# =============================================================================
# 보정 (Calibration)
# =============================================================================

def compute_T_align(T_tcp_robot: np.ndarray,
                    pos_natnet: np.ndarray,
                    rot_natnet: Rotation) -> np.ndarray:
    """
    T_align (4×4): NatNet 세계 좌표 → 로봇 베이스 좌표.

    정적 자세에서 marker ≡ TCP 임을 이용:
      T_align = T_tcp_robot @ inv(T_marker_natnet)
    """
    T_marker = np.eye(4)
    T_marker[:3, :3] = rot_natnet.as_matrix()
    T_marker[:3, 3]  = pos_natnet
    return T_tcp_robot @ np.linalg.inv(T_marker)


def save_calibration(T_align: np.ndarray, path: str):
    with open(path, 'w') as f:
        json.dump({'T_align': T_align.tolist()}, f, indent=2)
    log.info("보정 저장 → %s", path)


def load_calibration(path: str) -> np.ndarray:
    with open(path) as f:
        return np.array(json.load(f)['T_align'])


def _wait_rigid_body(timeout: float = 10.0) -> tuple[np.ndarray, Rotation]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        pos, rot = _natnet.get_rb()
        if pos is not None:
            return pos, rot
        time.sleep(0.05)
    raise TimeoutError("NatNet rigid body 데이터 수신 타임아웃.")


# =============================================================================
# 경로 실행 + 동시 기록
# =============================================================================

async def run_eval(args, robot, data_channel, T_align: np.ndarray,
                   waypoints_q: list[np.ndarray]) -> list[dict]:
    rc        = rb.ResponseCollector()
    records   = []
    recording = True

    async def recorder():
        while recording:
            data    = await data_channel.request_data()
            elapsed = time.time() - t_start

            tcp_raw = np.array(data.sdata.tcp_ref)  # [x,y,z mm, rx,ry,rz deg]

            lm_natnet = _natnet.get_lm()
            if lm_natnet is not None:
                lm_h      = np.append(lm_natnet, 1.0)
                lm_robot  = (T_align @ lm_h)[:3]      # m, 로봇 베이스 좌표
            else:
                lm_robot = np.full(3, np.nan)

            records.append({
                'elapsed_s':   round(elapsed, 4),
                'tcp_x_mm':    round(float(tcp_raw[0]), 4),
                'tcp_y_mm':    round(float(tcp_raw[1]), 4),
                'tcp_z_mm':    round(float(tcp_raw[2]), 4),
                'tcp_rx_deg':  round(float(tcp_raw[3]), 4),
                'tcp_ry_deg':  round(float(tcp_raw[4]), 4),
                'tcp_rz_deg':  round(float(tcp_raw[5]), 4),
                'lm_x_m':      round(float(lm_robot[0]), 6),
                'lm_y_m':      round(float(lm_robot[1]), 6),
                'lm_z_m':      round(float(lm_robot[2]), 6),
            })
            await asyncio.sleep(0.02)   # ~50 Hz

    # ── 경로 업로드 ──────────────────────────────────────────────────────────
    await robot.flush(rc)
    rc.error().throw_if_not_empty()

    log.info("%d 개 waypoint 업로드 중 (move_jb2) ...", len(waypoints_q))
    await robot.move_jb2_clear(rc)
    for q_deg in waypoints_q:
        await robot.move_jb2_add(rc, q_deg, args.speed, args.accel, args.blending)
    await robot.flush(rc)
    rc.error().throw_if_not_empty()

    # ── 실행 + 기록 ───────────────────────────────────────────────────────────
    t_start   = time.time()
    rec_task  = asyncio.create_task(recorder())

    await robot.move_jb2_run(rc)
    if (await robot.wait_for_move_started(rc, 3.0)).type() == rb.ReturnType.Success:
        log.info("모션 시작 — 기록 중 ...")
        await robot.wait_for_move_finished(rc)
        log.info("모션 완료.")
    else:
        log.warning("모션이 타임아웃 내 시작되지 않았습니다.")

    recording = False
    await rec_task

    return records


async def move_to_home(robot, rc):
    """모든 조인트를 0° 으로 이동하고 완료까지 대기."""
    log.info("홈 자세 (all joints = 0°) 로 이동 중 ...")
    await robot.move_j(rc, np.zeros(6), 30, 60)
    if (await robot.wait_for_move_started(rc, 3.0)).type() == rb.ReturnType.Success:
        await robot.wait_for_move_finished(rc)
    rc.error().throw_if_not_empty()
    log.info("홈 자세 도달 완료.")


def save_csv(records: list[dict], path: str):
    if not records:
        log.warning("기록된 데이터 없음.")
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    log.info("CSV 저장 완료 → %s  (%d rows)", path, len(records))


# =============================================================================
# Main (asyncio)
# =============================================================================

async def _async_main(args):
    global _mdl_id, _mkr_id, _rb_id
    _mdl_id = args.model_id
    _mkr_id = args.marker_id
    _rb_id  = args.rigid_body_id

    loop = asyncio.get_event_loop()  # run_in_executor 용 (실행 확인 프롬프트)

    # ── NatNet 연결 ───────────────────────────────────────────────────────────
    client = NatNetClient()
    client.set_client_address(args.client)
    client.set_server_address(args.server)
    client.new_frame_with_data_listener = _on_frame
    client.rigid_body_listener          = _on_rigid_body
    client.set_use_multicast(args.multicast)

    if not client.run('d'):
        raise RuntimeError("NatNet 스트리밍 시작 실패.")
    time.sleep(1.0)
    if not client.connected():
        raise RuntimeError("NatNet 서버 연결 실패. Motive 스트리밍 설정을 확인하세요.")
    log.info("NatNet 연결 완료 (%s → %s)", args.client, args.server)

    # ── 로봇 연결 ─────────────────────────────────────────────────────────────
    robot        = rb.asyncio.Cobot(args.robot_ip)
    data_channel = rb.asyncio.CobotData(args.robot_ip)
    rc           = rb.ResponseCollector()

    await robot.set_operation_mode(rc, rb.OperationMode.Real)
    await robot.set_speed_bar(rc, args.speed_bar)
    await robot.flush(rc)
    rc.error().throw_if_not_empty()
    log.info("로봇 연결 완료 (%s), Real 모드", args.robot_ip)

    # ── 홈 자세 이동 (all joints = 0°) ───────────────────────────────────────
    await move_to_home(robot, rc)

    # ── 보정 ─────────────────────────────────────────────────────────────────
    cal_path = Path(args.cal_file)
    if cal_path.exists() and not args.recalibrate:
        T_align = load_calibration(str(cal_path))
        log.info("보정 파일 로드 ← %s", cal_path)
    else:
        log.info("=== 좌표계 보정 (홈 자세 기준) ===")

        # 로봇이 정지 상태가 될 때까지 잠시 대기
        await asyncio.sleep(0.5)

        # 현재 TCP 읽기
        robot_data = await data_channel.request_data()
        tcp_now    = np.array(robot_data.sdata.tcp_ref)
        T_tcp      = tcp_to_matrix(tcp_now)
        log.info("보정 TCP: pos=[%.1f %.1f %.1f] mm  rot=[%.2f %.2f %.2f] deg",
                 *tcp_now[:3], *tcp_now[3:])

        # NatNet rigid body 수신 대기
        log.info("NatNet rigid body 데이터 대기 중 (rigid_body_id=%d) ...", _rb_id)
        pos_natnet, rot_natnet = _wait_rigid_body(timeout=10.0)
        log.info("Rigid body pos (NatNet): [%.4f %.4f %.4f] m", *pos_natnet)

        T_align = compute_T_align(T_tcp, pos_natnet, rot_natnet)
        save_calibration(T_align, str(cal_path))

    # ── FK/IK 모델 로드 ───────────────────────────────────────────────────────
    model = URDFRobot(args.joint_yaml)

    # 현재 관절 각도 읽기
    robot_data   = await data_channel.request_data()
    q_current_deg = np.array(robot_data.sdata.jnt_ref)
    q_current_rad = model.deg2rad(q_current_deg)

    T_center = model.fk(q_current_rad)
    log.info("현재 TCP (FK): pos=[%.1f %.1f %.1f] mm",
             *(T_center[:3, 3] * 1000))

    # ── 8자 경로 + IK ─────────────────────────────────────────────────────────
    log.info("8자 경로 계획 중 (%d points, r=%.0f mm) ...",
             args.n_waypoints, args.radius * 1000)
    path_poses = figure8_poses(T_center, args.radius, args.n_waypoints)

    waypoints_q = []
    q_seed      = q_current_rad.copy()
    n_fail      = 0

    for T_wp in path_poses:
        q_sol, ok = model.ik(T_wp, q_seed)
        if not ok:
            n_fail += 1
        waypoints_q.append(model.rad2deg(q_sol))
        q_seed = q_sol   # 이전 해로 warm-start

    log.info("IK 완료: %d/%d 수렴 (실패 %d)",
             len(waypoints_q) - n_fail, len(waypoints_q), n_fail)

    if n_fail > len(waypoints_q) // 4:
        raise RuntimeError(
            f"IK 실패가 너무 많습니다 ({n_fail}/{len(waypoints_q)}). "
            "joint.yaml 파라미터 또는 --radius 를 확인하세요."
        )

    # ── 실행 확인 ─────────────────────────────────────────────────────────────
    log.info("준비 완료. Enter 를 눌러 8자 경로 실행 시작 ...")
    await loop.run_in_executor(None, input)

    records = await run_eval(args, robot, data_channel, T_align, waypoints_q)

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"precision_eval_{stamp}.csv"
    save_csv(records, out_csv)

    client.shutdown()
    log.info("완료.")


# =============================================================================
# Entry point
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="로봇 엔드 이펙터 정밀도 평가 (rbpodo + NatNet)"
    )
    p.add_argument('--model_id',      type=int, required=True,
                   help='NatNet labeled marker model ID')
    p.add_argument('--marker_id',     type=int, default=0,
                   help='NatNet labeled marker ID (기본값: 0)')
    p.add_argument('--rigid_body_id', type=int, default=None,
                   help='NatNet rigid body ID (기본값: model_id 와 동일)')
    p.add_argument('--robot_ip',      default='10.0.2.7')
    p.add_argument('--server',        default='127.0.0.1',
                   help='NatNet 서버 IP')
    p.add_argument('--client',        default='127.0.0.1',
                   help='NatNet 클라이언트 IP')
    p.add_argument('--multicast',     action='store_true', default=False,
                   help='멀티캐스트 사용 (기본값: 유니캐스트)')
    p.add_argument('--joint_yaml',    default='joint.yaml',
                   help='URDF 형식 관절 파라미터 파일')
    p.add_argument('--cal_file',      default='calibration.json',
                   help='보정 변환 저장/로드 파일')
    p.add_argument('--recalibrate',   action='store_true',
                   help='기존 보정 파일이 있어도 다시 보정')
    p.add_argument('--radius',        type=float, default=0.05,
                   help='8자 반지름 m (기본값: 0.05)')
    p.add_argument('--n_waypoints',   type=int,   default=60,
                   help='IK waypoint 수 (기본값: 60)')
    p.add_argument('--speed',         type=float, default=20.0,
                   help='관절 속도 deg/s (기본값: 20)')
    p.add_argument('--accel',         type=float, default=40.0,
                   help='관절 가속도 deg/s² (기본값: 40)')
    p.add_argument('--blending',      type=float, default=5.0,
                   help='move_jb2 블렌딩 값 deg (기본값: 5.0)')
    p.add_argument('--speed_bar',     type=float, default=0.3,
                   help='로봇 속도 제한 0-1 (기본값: 0.3)')

    args = p.parse_args()
    if args.rigid_body_id is None:
        args.rigid_body_id = args.model_id

    asyncio.run(_async_main(args))


if __name__ == '__main__':
    main()
