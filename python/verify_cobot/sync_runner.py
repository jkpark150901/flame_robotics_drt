"""
sync_runner.py
==============
로봇 TCP + NatNet Rigid Body 를 연속으로 동기 기록하는 QThread.

동작:
  1. rbpodo CobotData 채널로 ~50 Hz TCP 수신
  2. NatNetStateProxy 에서 가장 최근 RB 포즈 조회 (타임스탬프 기반 동기)
  3. 캘리브레이션 T_base_motive @ T_motive_rb @ T_rb_tcp 적용
  4. TCP 실제 위치와 RB 예측 위치 오차 계산
  5. 포인트 단위로 point_recorded 시그널 방출
  6. 정지 시 all_done 으로 전체 레코드 전달
"""

import asyncio
import logging
import sys
import threading
import time
import pathlib

import numpy as np

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    raise ImportError("PyQt6 is required.")

_ROOT = pathlib.Path(__file__).parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from calibration.solver import rb_pose_to_matrix, tcp_raw_to_matrix

log = logging.getLogger(__name__)


class SyncRunner(QThread):
    """
    Signals:
        point_recorded(dict)  — elapsed_s, tcp_x/y/z_m, rb_aligned_x/y/z_m,
                                 error_mm, mocap_age_ms
        all_done(list)        — list of all dicts (for CSV export)
        log_msg(str)
        error(str)
    """

    point_recorded = pyqtSignal(dict)
    all_done       = pyqtSignal(list)
    log_msg        = pyqtSignal(str)
    error          = pyqtSignal(str)

    def __init__(self, params: dict, calib: dict, natnet_state, parent=None):
        """
        params:
            robot_ip, speed_bar, interval_s, tcp_orientation_type
        calib:
            T_base_motive, T_rb_tcp
        """
        super().__init__(parent)
        self._p             = params
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
            log.error("SyncRunner: %s", e)
            self.error.emit(str(e))
        finally:
            self._loop.close()
            self._loop = None

    async def _main(self):
        try:
            import rbpodo as rb
        except ImportError as e:
            raise RuntimeError("rbpodo 모듈을 찾을 수 없습니다.") from e

        p = self._p
        self._emit_log(f"로봇 연결 중: {p['robot_ip']} …")
        try:
            data_ch = rb.asyncio.CobotData(p['robot_ip'])
        except Exception as e:
            raise RuntimeError(f"로봇 연결 실패: {e}") from e

        self._emit_log("기록 시작 (Stop 버튼으로 종료)")
        records  = []
        t_start  = time.time()
        interval = float(p.get('interval_s', 0.02))   # 기본 50 Hz
        orient   = p.get('tcp_orientation_type', 'zyx_euler_deg')
        prev_rb_time = 0.0

        while not self._stop_event.is_set():
            sample_time = time.time()
            elapsed = sample_time - t_start

            # ── 로봇 TCP ────────────────────────────────────────────
            try:
                data    = await asyncio.wait_for(data_ch.request_data(), timeout=1.0)
                tcp_raw = np.array(data.sdata.tcp_ref, dtype=float)
            except Exception as e:
                log.warning("TCP 수신 실패: %s", e)
                await asyncio.sleep(interval)
                continue

            # ── NatNet RB (sample_time 이후 최신 프레임) ─────────────
            rb_sample = self._natnet.get_rb_since(prev_rb_time)
            if rb_sample is None:
                # 최신 프레임이 없으면 가장 최근 것이라도 사용
                rb_sample = self._natnet.get_rb_since(0.0)
            if rb_sample is None:
                await asyncio.sleep(interval)
                continue

            rb_pos, rb_rot = rb_sample
            now_rb  = self._natnet.get_time()
            mocap_age_ms = (sample_time - now_rb) * 1000.0 if now_rb else 0.0
            prev_rb_time = now_rb or 0.0

            # ── 캘리브레이션 적용 ────────────────────────────────────
            T_motive_rb  = rb_pose_to_matrix(rb_pos, rb_rot)
            T_pred_tcp   = self._T_base_motive @ T_motive_rb @ self._T_rb_tcp
            T_actual_tcp = tcp_raw_to_matrix(tcp_raw, orient)

            rb_aligned = T_pred_tcp[:3, 3]
            tcp_pos_m  = T_actual_tcp[:3, 3]
            error_mm   = float(np.linalg.norm(rb_aligned - tcp_pos_m) * 1000.0)

            rec = {
                'elapsed_s':      round(elapsed, 4),
                'tcp_x_m':        round(float(tcp_pos_m[0]), 6),
                'tcp_y_m':        round(float(tcp_pos_m[1]), 6),
                'tcp_z_m':        round(float(tcp_pos_m[2]), 6),
                'tcp_rx_deg':     round(float(tcp_raw[3]), 4),
                'tcp_ry_deg':     round(float(tcp_raw[4]), 4),
                'tcp_rz_deg':     round(float(tcp_raw[5]), 4),
                'rb_raw_x_m':     round(float(rb_pos[0]), 6),
                'rb_raw_y_m':     round(float(rb_pos[1]), 6),
                'rb_raw_z_m':     round(float(rb_pos[2]), 6),
                'rb_aligned_x_m': round(float(rb_aligned[0]), 6),
                'rb_aligned_y_m': round(float(rb_aligned[1]), 6),
                'rb_aligned_z_m': round(float(rb_aligned[2]), 6),
                'error_mm':       round(error_mm, 4),
                'mocap_age_ms':   round(mocap_age_ms, 2),
            }
            records.append(rec)
            self.point_recorded.emit(rec)

            await asyncio.sleep(interval)

        self._emit_log(f"기록 완료 — {len(records)} 포인트")
        if records:
            self.all_done.emit(records)

    def _emit_log(self, msg: str):
        log.info(msg)
        self.log_msg.emit(msg)


# NatNetStateProxy에 get_time() 메서드가 없을 수 있으므로 안전하게 패치
def _patch_natnet_state(proxy):
    """NatNetStateProxy에 get_time() 이 없으면 추가."""
    if not hasattr(proxy, 'get_time'):
        import threading as _t
        proxy._time = None

        _orig_update = proxy.update

        def _new_update(pos, rot):
            import time as _time
            proxy._time = _time.time()
            _orig_update(pos, rot)

        proxy.update  = _new_update

        def _get_time():
            return proxy._time

        proxy.get_time = _get_time
    return proxy
