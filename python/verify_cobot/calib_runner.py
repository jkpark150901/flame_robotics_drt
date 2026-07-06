'''
CalibRunner — 캘리브레이션 궤적 실행 + 샘플 수집 QThread
'''

import asyncio
import csv
import logging
import pathlib
import sys
import threading
import time
import types

import numpy as np

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    raise ImportError("PyQt6 is required.")

import rbpodo as rb

_ROOT = pathlib.Path(__file__).parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from calibration.solver import (
    average_quaternions_xyzw,
    rb_pose_to_matrix,
    tcp_raw_to_matrix,
    solve_handeye_opencv,
    handeye_residuals,
    OPENCV_HANDEYE_METHODS,
    save_calibration,
    apply_calibration_and_save,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CalibRunner
# ──────────────────────────────────────────────────────────────────────────────

class CalibRunner(QThread):
    '''
    Signals:
        pose_started(int, int, str)    - (현재 인덱스 1-based, 전체 개수, 레이블)
        pose_done(int, int, bool)      - (현재 인덱스 1-based, 전체 개수, 샘플 성공 여부)
        all_done(dict)                 - 수집 완료, 캘리브레이션 결과 dict
        progress(int)                  - 0..total 진행 값 (progress bar 용)
        log_msg(str)                   - 로그 메시지
        error(str)                     - 치명적 에러 (중단)
    '''

    pose_started     = pyqtSignal(int, int, str)
    pose_done        = pyqtSignal(int, int, bool)
    point_collected  = pyqtSignal(list, list)   # rb_pos[3] m, tcp_pos_m[3] m
    all_done         = pyqtSignal(dict)
    progress         = pyqtSignal(int)
    log_msg          = pyqtSignal(str)
    error            = pyqtSignal(str)

    def __init__(self, params: dict, natnet_state, parent=None):
        '''
        params keys (모두 필수):
            robot_ip, csv_path, speed, accel, speed_bar,
            settle_time, sample_count, min_mocap_samples, mocap_timeout,
            handeye_method_int, outlier_threshold_mm, tcp_orientation_type,
            cal_file
        natnet_state: NatNetStateProxy (rb_pos, rb_rot, rb_time 를 thread-safe하게 노출)
        '''
        super().__init__(parent)
        self._p = params
        self._natnet = natnet_state
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ──────────────────────────────────────────────
    # QThread entry point
    # ──────────────────────────────────────────────

    def run(self):
        self._stop_event.clear()
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            log.error("CalibRunner exception: %s", e)
            self.error.emit(str(e))
        finally:
            self._loop.close()
            self._loop = None

    # ──────────────────────────────────────────────
    # Async main
    # ──────────────────────────────────────────────

    async def _main(self):
        p = self._p

        # ── 1. CSV 읽기 ──────────────────────────────
        sample_poses = _load_plan_csv(p['csv_path'])  # raises on error

        # ── 2. 로봇 연결 ─────────────────────────────
        self._emit_log(f"로봇 연결 중: {p['robot_ip']} …")
        try:
            robot    = rb.asyncio.Cobot(p['robot_ip'])
            data_ch  = rb.asyncio.CobotData(p['robot_ip'])
            rc       = rb.ResponseCollector()
            await robot.set_operation_mode(rc, rb.OperationMode.Real)
            await robot.flush(rc)
            rc.error().throw_if_not_empty()
            await robot.set_speed_bar(rc, p['speed_bar'])
            await robot.flush(rc)
            rc.error().throw_if_not_empty()
        except Exception as e:
            raise RuntimeError(f"로봇 연결 실패: {e}") from e

        self._emit_log(f"로봇 연결 완료. {len(sample_poses)}개 포즈 수행 예정.")

        # ── 3. 포즈 순회 + 샘플 수집 ─────────────────
        raw_records     = []
        points_motive   = []
        quat_motive     = []
        points_tcp_m    = []
        prev_tcp_pos_m  = None
        prev_rb_pos     = None
        t_start         = time.time()
        total           = len(sample_poses)

        for idx, (label, q_deg) in enumerate(sample_poses, start=1):
            if self._stop_event.is_set():
                self._emit_log("사용자 중단.")
                break

            self.pose_started.emit(idx, total, label)
            self._emit_log(f"[{idx:02d}/{total:02d}] 이동: {label} → {np.round(q_deg,1).tolist()} °")

            # 이동
            try:
                await robot.move_j(rc, q_deg, p['speed'], p['accel'])
                if (await robot.wait_for_move_started(rc, 3.0)).type() == rb.ReturnType.Success:
                    await robot.wait_for_move_finished(rc)
                else:
                    self._emit_log(f"[{idx:02d}/{total:02d}] 모션 시작 타임아웃: {label}")
                rc.error().throw_if_not_empty()
            except Exception as e:
                self.error.emit(f"이동 실패: {label}: {e}")
                break

            # settle
            await asyncio.sleep(p['settle_time'])

            # 샘플
            tcp_raw, rb_pos, rb_rot, n_valid = await self._sample_point(data_ch, p)

            if tcp_raw is None or rb_pos is None:
                self._emit_log(
                    f"[{idx:02d}/{total:02d}] mocap 샘플 부족: {label} ({n_valid}/{p['sample_count']})"
                )
                self.pose_done.emit(idx, total, False)
                self.progress.emit(idx)
                continue

            elapsed = time.time() - t_start
            tcp_pos_m = tcp_raw[:3] / 1000.0

            tcp_step_mm, rb_step_mm, step_delta_mm, step_ratio = _calc_steps(
                tcp_pos_m, rb_pos, prev_tcp_pos_m, prev_rb_pos
            )
            prev_tcp_pos_m = tcp_pos_m.copy()
            prev_rb_pos    = rb_pos.copy()

            points_motive.append(rb_pos)
            quat_motive.append(rb_rot)
            points_tcp_m.append(tcp_pos_m)

            raw_records.append({
                'elapsed_s':   round(elapsed, 4),
                'pose_label':  label,
                **{f'joint_{i}_deg': round(float(q_deg[i]), 4) for i in range(6)},
                'tcp_x_mm':    round(float(tcp_raw[0]), 4),
                'tcp_y_mm':    round(float(tcp_raw[1]), 4),
                'tcp_z_mm':    round(float(tcp_raw[2]), 4),
                'tcp_rx_deg':  round(float(tcp_raw[3]), 4),
                'tcp_ry_deg':  round(float(tcp_raw[4]), 4),
                'tcp_rz_deg':  round(float(tcp_raw[5]), 4),
                'rb_raw_x_m':  round(float(rb_pos[0]), 6),
                'rb_raw_y_m':  round(float(rb_pos[1]), 6),
                'rb_raw_z_m':  round(float(rb_pos[2]), 6),
                'rb_qx': round(float(rb_rot[0]), 6),
                'rb_qy': round(float(rb_rot[1]), 6),
                'rb_qz': round(float(rb_rot[2]), 6),
                'rb_qw': round(float(rb_rot[3]), 6),
                'mocap_valid_samples': n_valid,
                'tcp_step_mm':   round(tcp_step_mm, 4),
                'rb_step_mm':    round(rb_step_mm, 4),
                'step_delta_mm': round(step_delta_mm, 4),
                'step_ratio':    round(float(step_ratio), 6) if not np.isnan(step_ratio) else '',
            })
            self._emit_log(
                f"[{idx:02d}/{total:02d}] 수집 완료: 누적 {len(raw_records)}점 "
                f"| tcp={tcp_step_mm:.1f} mm, mocap={rb_step_mm:.1f} mm"
            )
            self.point_collected.emit(rb_pos.tolist(), tcp_pos_m.tolist())
            self.pose_done.emit(idx, total, True)
            self.progress.emit(idx)

        # ── 4. 캘리브레이션 계산 ─────────────────────
        try:
            await robot.disconnect(rc)
        except Exception:
            pass

        n = len(raw_records)
        if n < 6:
            self.error.emit(f"수집된 유효 샘플이 {n}개뿐입니다 (최소 6개 필요). 캘리브레이션을 건너뜁니다.")
            return

        self._emit_log(f"캘리브레이션 계산 중 ({n}개 샘플) …")
        try:
            result = _compute_handeye(raw_records, p)
        except Exception as e:
            self.error.emit(f"캘리브레이션 계산 실패: {e}")
            return

        self._emit_log(
            f"캘리브레이션 완료 ✓ "
            f"position RMSE (inlier): {result['rmse_pos_mm_inlier']:.3f} mm"
        )
        self.all_done.emit(result)

    # ──────────────────────────────────────────────
    # Sampling
    # ──────────────────────────────────────────────

    async def _sample_point(self, data_ch, p: dict):
        tcp_samples    = []
        rb_samples     = []
        rb_rot_samples = []
        mocap_timeout  = p['mocap_timeout']
        sample_count   = p['sample_count']
        min_samples    = p['min_mocap_samples']

        for _ in range(sample_count):
            if self._stop_event.is_set():
                break
            sample_time = time.time()
            try:
                data    = await asyncio.wait_for(data_ch.request_data(), timeout=2.0)
                tcp_raw = np.array(data.sdata.tcp_ref, dtype=float)
            except Exception as e:
                log.warning("TCP data error: %s", e)
                continue

            rb_sample = await self._wait_for_rb(sample_time, mocap_timeout)
            if rb_sample is not None:
                rb_pos, rb_rot = rb_sample
                tcp_samples.append(tcp_raw)
                rb_samples.append(rb_pos)
                rb_rot_samples.append(rb_rot)
            await asyncio.sleep(0.05)

        if len(rb_samples) < min_samples:
            return None, None, None, len(rb_samples)

        rb_rot_avg = average_quaternions_xyzw(rb_rot_samples)
        return (
            np.mean(tcp_samples, axis=0),
            np.mean(rb_samples, axis=0),
            rb_rot_avg,
            len(rb_samples),
        )

    async def _wait_for_rb(self, min_time: float, timeout_s: float):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            sample = self._natnet.get_rb_since(min_time)
            if sample is not None:
                return sample  # (pos, rot)
            await asyncio.sleep(0.02)
        return None

    # ──────────────────────────────────────────────

    def _emit_log(self, msg: str):
        log.info(msg)
        self.log_msg.emit(msg)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_plan_csv(path: str) -> list[tuple[str, np.ndarray]]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"캘리브레이션 플랜 CSV를 찾을 수 없습니다: {path}")
    if p.stat().st_size == 0:
        raise ValueError(f"캘리브레이션 플랜 CSV가 비어 있습니다: {path}")

    poses = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'pose_label', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6'}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"CSV 컬럼 누락: {missing}")
        for row_num, row in enumerate(reader, start=2):
            try:
                label = row['pose_label'].strip()
                if not label:
                    raise ValueError("pose_label이 비어 있습니다")
                q = np.array([float(row[f'J{i}']) for i in range(1, 7)])
            except (KeyError, ValueError) as e:
                raise ValueError(f"CSV {row_num}행 파싱 오류: {e}") from e
            poses.append((label, q))

    if len(poses) == 0:
        raise ValueError(f"캘리브레이션 플랜에 유효한 포즈가 없습니다: {path}")
    return poses


def _calc_steps(tcp_pos_m, rb_pos, prev_tcp, prev_rb):
    if prev_tcp is None:
        return 0.0, 0.0, 0.0, float('nan')
    tcp_step = float(np.linalg.norm(tcp_pos_m - prev_tcp) * 1000.0)
    rb_step  = float(np.linalg.norm(rb_pos  - prev_rb)  * 1000.0)
    delta    = rb_step - tcp_step
    ratio    = rb_step / tcp_step if tcp_step > 1e-9 else float('nan')
    return tcp_step, rb_step, delta, ratio


def _compute_handeye(raw_records: list[dict], p: dict) -> dict:
    '''hand-eye 캘리브레이션 계산 후 결과 dict 반환.'''
    orient = p.get('tcp_orientation_type', 'zyx_euler_deg')
    method_int = p['handeye_method_int']
    outlier_mm = p['outlier_threshold_mm']

    T_motive_rb_list = [
        rb_pose_to_matrix(
            np.array([r['rb_raw_x_m'], r['rb_raw_y_m'], r['rb_raw_z_m']]),
            np.array([r['rb_qx'], r['rb_qy'], r['rb_qz'], r['rb_qw']]),
        )
        for r in raw_records
    ]
    T_base_tcp_list = [
        tcp_raw_to_matrix(
            np.array([r['tcp_x_mm'], r['tcp_y_mm'], r['tcp_z_mm'],
                      r['tcp_rx_deg'], r['tcp_ry_deg'], r['tcp_rz_deg']]),
            orient,
        )
        for r in raw_records
    ]

    T_align, T_rb_tcp = solve_handeye_opencv(T_motive_rb_list, T_base_tcp_list, method_int)
    residuals, rot_res = handeye_residuals(T_align, T_rb_tcp, T_motive_rb_list, T_base_tcp_list)

    inlier_mask = np.ones(len(raw_records), dtype=bool)
    if outlier_mm > 0.0:
        inlier_mask = residuals <= outlier_mm
        if np.count_nonzero(inlier_mask) >= 6 and np.any(~inlier_mask):
            T_align, T_rb_tcp = solve_handeye_opencv(
                [T for T, ok in zip(T_motive_rb_list, inlier_mask) if ok],
                [T for T, ok in zip(T_base_tcp_list, inlier_mask) if ok],
                method_int,
            )
            residuals, rot_res = handeye_residuals(
                T_align, T_rb_tcp, T_motive_rb_list, T_base_tcp_list
            )
        elif np.count_nonzero(inlier_mask) < 6:
            log.warning("outlier 제외 후 inlier %d개 → 전체 사용", np.count_nonzero(inlier_mask))
            inlier_mask[:] = True

    for row, ok in zip(raw_records, inlier_mask):
        row['calibration_inlier'] = int(bool(ok))

    rmse_pos_all    = float(np.sqrt(np.mean(residuals ** 2)))
    rmse_pos_inlier = float(np.sqrt(np.mean(residuals[inlier_mask] ** 2)))
    rmse_rot_all    = float(np.sqrt(np.mean(rot_res ** 2)))
    rmse_rot_inlier = float(np.sqrt(np.mean(rot_res[inlier_mask] ** 2)))

    return {
        'T_base_motive':   T_align.tolist(),
        'T_rb_tcp':        T_rb_tcp.tolist(),
        'raw_records':     raw_records,
        'residuals_mm':    residuals.tolist(),
        'rot_residuals_deg': rot_res.tolist(),
        'inlier_mask':     inlier_mask.tolist(),
        'rmse_pos_mm_all':    rmse_pos_all,
        'rmse_pos_mm_inlier': rmse_pos_inlier,
        'rmse_rot_deg_all':    rmse_rot_all,
        'rmse_rot_deg_inlier': rmse_rot_inlier,
        'n_total':   len(raw_records),
        'n_inlier':  int(np.count_nonzero(inlier_mask)),
        'cal_file':  p['cal_file'],
    }


# ──────────────────────────────────────────────────────────────────────────────
# NatNet 상태 프록시 (window 에서 natnet_worker 의 마지막 rb 를 공유)
# ──────────────────────────────────────────────────────────────────────────────

class NatNetStateProxy:
    '''CalibRunner 가 최신 RB 포즈를 읽을 수 있도록 window 에서 업데이트.'''

    def __init__(self):
        self._lock = threading.Lock()
        self._pos:  np.ndarray | None = None
        self._rot:  np.ndarray | None = None
        self._time: float | None = None

    def update(self, pos: list, rot: list):
        with self._lock:
            self._pos  = np.array(pos,  dtype=float)
            self._rot  = np.array(rot,  dtype=float)
            self._time = time.time()

    def get_rb_since(self, min_time: float):
        with self._lock:
            if self._pos is None or self._time is None or self._time < min_time:
                return None
            return self._pos.copy(), self._rot.copy()

    def get_time(self) -> float | None:
        with self._lock:
            return self._time
