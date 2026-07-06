'''
VerifyRunner — 검증 궤적 실행 + 오차 계산 QThread
'''

import asyncio
import csv
import logging
import pathlib
import sys
import threading
import time

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
    invert_transform,
)
from verifytool.calib_runner import _load_plan_csv, _calc_steps

log = logging.getLogger(__name__)


class VerifyRunner(QThread):
    '''
    Signals:
        pose_started(int, int, str)       - (idx 1-based, total, label)
        pose_done(int, int, float)         - (idx, total, error_mm) — -1 if skipped
        point_verified(list, list, float)  - (tcp_pos_m[3], rb_pred_tcp_m[3], error_mm)
        all_done(dict)                     - 결과 dict
        progress(int)
        log_msg(str)
        error(str)
    '''

    pose_started    = pyqtSignal(int, int, str)
    pose_done       = pyqtSignal(int, int, float, float)  # idx, total, pos_err_mm, rot_err_deg
    point_verified  = pyqtSignal(list, list, float, float)  # actual, predicted, pos_mm, rot_deg
    all_done        = pyqtSignal(dict)
    progress        = pyqtSignal(int)
    log_msg         = pyqtSignal(str)
    error           = pyqtSignal(str)

    def __init__(self, params: dict, calib: dict, natnet_state, parent=None):
        '''
        params: robot_ip, csv_path, speed, accel, speed_bar,
                settle_time, sample_count, min_mocap_samples,
                mocap_timeout, tcp_orientation_type
        calib:  T_base_motive (list[4][4]), T_rb_tcp (list[4][4])
        '''
        super().__init__(parent)
        self._p = params
        self._T_base_motive = np.array(calib['T_base_motive'])
        self._T_rb_tcp      = np.array(calib['T_rb_tcp'])
        self._natnet        = natnet_state
        self._stop_event    = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def stop(self):
        self._stop_event.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def run(self):
        self._stop_event.clear()
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            log.error("VerifyRunner exception: %s", e)
            self.error.emit(str(e))
        finally:
            self._loop.close()
            self._loop = None

    async def _main(self):
        p = self._p

        # ── 1. CSV ─────────────────────────────
        sample_poses = _load_plan_csv(p['csv_path'])

        # ── 2. 로봇 연결 ────────────────────────
        self._emit_log(f"로봇 연결 중: {p['robot_ip']} …")
        try:
            robot   = rb.asyncio.Cobot(p['robot_ip'])
            data_ch = rb.asyncio.CobotData(p['robot_ip'])
            rc      = rb.ResponseCollector()
            await robot.set_operation_mode(rc, rb.OperationMode.Real)
            await robot.flush(rc)
            rc.error().throw_if_not_empty()
            await robot.set_speed_bar(rc, p['speed_bar'])
            await robot.flush(rc)
            rc.error().throw_if_not_empty()
        except Exception as e:
            raise RuntimeError(f"로봇 연결 실패: {e}") from e

        self._emit_log(f"로봇 연결 완료. {len(sample_poses)}개 포즈 검증 예정.")

        # ── 3. 포즈 순회 ────────────────────────
        records        = []
        errors_mm      = []
        rot_errors_deg = []
        tcp_positions  = []
        rb_predictions = []
        labels_done    = []
        total          = len(sample_poses)
        t_start        = time.time()

        for idx, (label, q_deg) in enumerate(sample_poses, start=1):
            if self._stop_event.is_set():
                self._emit_log("사용자 중단.")
                break

            self.pose_started.emit(idx, total, label)
            self._emit_log(f"[{idx:02d}/{total:02d}] 이동: {label}")

            try:
                await robot.move_j(rc, q_deg, p['speed'], p['accel'])
                if (await robot.wait_for_move_started(rc, 3.0)).type() == rb.ReturnType.Success:
                    await robot.wait_for_move_finished(rc)
                else:
                    self._emit_log(f"[{idx:02d}/{total:02d}] 모션 시작 타임아웃: {label}")
                rc.error().throw_if_not_empty()
            except Exception as e:
                self.error.emit(f"이동 실패 {label}: {e}")
                break

            await asyncio.sleep(p['settle_time'])

            tcp_raw, rb_pos, rb_rot, n_valid = await self._sample_point(data_ch, p)

            if tcp_raw is None or rb_pos is None:
                self._emit_log(
                    f"[{idx:02d}/{total:02d}] mocap 샘플 부족: {label} "
                    f"({n_valid}/{p['sample_count']})"
                )
                self.pose_done.emit(idx, total, -1.0)
                self.progress.emit(idx)
                continue

            # ── 오차 계산 ────────────────────────
            T_motive_rb = rb_pose_to_matrix(rb_pos, rb_rot)
            T_pred_tcp  = self._T_base_motive @ T_motive_rb @ self._T_rb_tcp
            orient      = p.get('tcp_orientation_type', 'zyx_euler_deg')
            T_actual_tcp = tcp_raw_to_matrix(tcp_raw, orient)

            pos_err_mm = float(
                np.linalg.norm(T_pred_tcp[:3, 3] - T_actual_tcp[:3, 3]) * 1000.0
            )
            R_diff = T_pred_tcp[:3, :3].T @ T_actual_tcp[:3, :3]
            rot_err_deg = float(np.degrees(
                np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
            ))
            elapsed = time.time() - t_start

            tcp_pos_m  = T_actual_tcp[:3, 3].tolist()
            pred_pos_m = T_pred_tcp[:3, 3].tolist()
            errors_mm.append(pos_err_mm)
            rot_errors_deg.append(rot_err_deg)
            tcp_positions.append(tcp_pos_m)
            rb_predictions.append(pred_pos_m)
            labels_done.append(label)

            records.append({
                'elapsed_s':    round(elapsed, 4),
                'pose_label':   label,
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
                'pred_tcp_x_m': round(pred_pos_m[0], 6),
                'pred_tcp_y_m': round(pred_pos_m[1], 6),
                'pred_tcp_z_m': round(pred_pos_m[2], 6),
                'error_mm':    round(pos_err_mm, 4),
                'rot_err_deg': round(rot_err_deg, 4),
                'mocap_valid_samples': n_valid,
            })

            self._emit_log(
                f"[{idx:02d}/{total:02d}] pos: {pos_err_mm:.3f} mm  "
                f"rot: {rot_err_deg:.3f} °  ({label})"
            )
            self.point_verified.emit(tcp_pos_m, pred_pos_m, pos_err_mm, rot_err_deg)
            self.pose_done.emit(idx, total, pos_err_mm, rot_err_deg)
            self.progress.emit(idx)

        try:
            await robot.disconnect(rc)
        except Exception:
            pass

        if len(errors_mm) == 0:
            self.error.emit("유효 샘플이 없습니다.")
            return

        errors_arr  = np.array(errors_mm)
        rot_arr     = np.array(rot_errors_deg)
        result = {
            'labels':          labels_done,
            'errors_mm':       errors_mm,
            'rot_errors_deg':  rot_errors_deg,
            'tcp_positions':   tcp_positions,
            'rb_predictions':  rb_predictions,
            'rmse_mm':         float(np.sqrt(np.mean(errors_arr ** 2))),
            'max_mm':          float(np.max(errors_arr)),
            'mean_mm':         float(np.mean(errors_arr)),
            'rmse_rot_deg':    float(np.sqrt(np.mean(rot_arr ** 2))),
            'max_rot_deg':     float(np.max(rot_arr)),
            'mean_rot_deg':    float(np.mean(rot_arr)),
            'raw_records':     records,
        }
        self._emit_log(
            f"검증 완료 — pos RMSE: {result['rmse_mm']:.3f} mm  "
            f"Max: {result['max_mm']:.3f} mm  "
            f"| rot RMSE: {result['rmse_rot_deg']:.3f} °  "
            f"Max: {result['max_rot_deg']:.3f} °  "
            f"({len(errors_mm)}/{total} 포즈)"
        )
        self.all_done.emit(result)

    async def _sample_point(self, data_ch, p: dict):
        tcp_samples    = []
        rb_samples     = []
        rb_rot_samples = []

        for _ in range(p['sample_count']):
            if self._stop_event.is_set():
                break
            sample_time = time.time()
            try:
                data    = await asyncio.wait_for(data_ch.request_data(), timeout=2.0)
                tcp_raw = np.array(data.sdata.tcp_ref, dtype=float)
            except Exception as e:
                log.warning("TCP data error: %s", e)
                continue
            rb_sample = await self._wait_for_rb(sample_time, p['mocap_timeout'])
            if rb_sample is not None:
                rb_pos, rb_rot = rb_sample
                tcp_samples.append(tcp_raw)
                rb_samples.append(rb_pos)
                rb_rot_samples.append(rb_rot)
            await asyncio.sleep(0.05)

        if len(rb_samples) < p['min_mocap_samples']:
            return None, None, None, len(rb_samples)

        return (
            np.mean(tcp_samples, axis=0),
            np.mean(rb_samples, axis=0),
            average_quaternions_xyzw(rb_rot_samples),
            len(rb_samples),
        )

    async def _wait_for_rb(self, min_time: float, timeout_s: float):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            sample = self._natnet.get_rb_since(min_time)
            if sample is not None:
                return sample
            await asyncio.sleep(0.02)
        return None

    def _emit_log(self, msg: str):
        log.info(msg)
        self.log_msg.emit(msg)
