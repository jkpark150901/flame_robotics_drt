"""
BlendTestRunner — blending 값별 궤적 실행 + 기록 QThread
"""
import asyncio
import logging
import threading
import time

import numpy as np

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    raise ImportError("PyQt6 is required.")

import rbpodo as rb

log = logging.getLogger(__name__)

# 기본 waypoint 델타값 (현재 위치 기준)
DEFAULT_JB2_DELTAS = np.array([
    [ 20,  10,   0, 0, 0, 0],
    [-15,  25,   5, 0, 0, 0],
    [ 25,   5,  -5, 0, 0, 0],
    [-10,  30,   0, 0, 0, 0],
    [  0,   0,   0, 0, 0, 0],
], dtype=float)

DEFAULT_PB_DELTAS = np.array([
    [150,    0, 0, 0, 0, 0],
    [150,  120, 0, 0, 0, 0],
    [  0,  120, 0, 0, 0, 0],
    [  0,    0, 0, 0, 0, 0],
], dtype=float)


class BlendTestRunner(QThread):
    """
    Signals
    -------
    run_progress(run_idx, total, blending)  각 실행 시작 시
    run_done(blending, records)             각 실행 완료 시
    all_done(dict)                          전체 완료 시  {blending: records}
    log_msg(str)
    error(str)
    """
    run_progress = pyqtSignal(int, int, float)
    run_done     = pyqtSignal(float, list)
    all_done     = pyqtSignal(dict)
    log_msg      = pyqtSignal(str)
    error        = pyqtSignal(str)

    def __init__(self, params: dict, parent=None):
        """
        params keys
        -----------
        robot_ip, mode ('jb2'|'pb'|'lb'), blending_values (list[float]),
        waypoint_deltas (list[list[float]] — 6-DOF deltas),
        speed, accel, speed_bar, record_hz,
        blending_option ('ratio'|'distance')   [pb 전용]
        """
        super().__init__(parent)
        self._p = params
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def stop(self):
        self._stop_event.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── QThread entry ──────────────────────────────────────────────────────

    def run(self):
        self._stop_event.clear()
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            log.error("BlendTestRunner exception: %s", e)
            self.error.emit(str(e))
        finally:
            self._loop.close()
            self._loop = None

    # ── Async main ─────────────────────────────────────────────────────────

    async def _main(self):
        p = self._p
        blend_opt = (rb.BlendingOption.Ratio
                     if p.get('blending_option', 'ratio') == 'ratio'
                     else rb.BlendingOption.Distance)

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

        # 현재 상태 읽기
        data = await asyncio.wait_for(data_ch.request_data(), timeout=3.0)
        cur_tcp    = np.array(data.sdata.tcp_ref, dtype=float)
        cur_joints = np.array(data.sdata.jnt_ref, dtype=float)
        start_joints = cur_joints.copy()

        self._emit_log(f"현재 TCP   : {np.round(cur_tcp, 2).tolist()} mm/deg")
        self._emit_log(f"현재 관절  : {np.round(cur_joints, 2).tolist()} deg")

        deltas = np.array(p['waypoint_deltas'], dtype=float)
        if p['mode'] == 'jb2':
            waypoints = [cur_joints + d for d in deltas]
        else:
            waypoints = [cur_tcp + d for d in deltas]

        blending_values = p['blending_values']
        total = len(blending_values)
        all_results: dict = {}

        for run_idx, bv in enumerate(blending_values, start=1):
            if self._stop_event.is_set():
                self._emit_log("사용자 중단.")
                break

            self._emit_log(f"[{run_idx}/{total}] blending={bv:.3f} 시작")
            self.run_progress.emit(run_idx, total, bv)

            await self._move_to_start(robot, rc, start_joints,
                                      p['speed'], p['accel'])

            records = await self._execute_with_recording(
                robot, data_ch, rc, waypoints, bv, blend_opt, p)

            all_results[bv] = records
            self.run_done.emit(bv, records)
            self._emit_log(f"[{run_idx}/{total}] 완료 — {len(records)} 샘플")

        try:
            await robot.disconnect(rc)
        except Exception:
            pass

        if not self._stop_event.is_set():
            self.all_done.emit(all_results)

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _move_to_start(self, robot, rc, start_joints, speed, accel):
        await robot.move_j(rc, start_joints, speed, accel)
        await robot.flush(rc)
        res = await robot.wait_for_move_started(rc, 5.0)
        if res.type() == rb.ReturnType.Success:
            await robot.wait_for_move_finished(rc)
        rc.error().throw_if_not_empty()
        await asyncio.sleep(0.3)

    async def _execute_with_recording(self, robot, data_ch, rc,
                                       waypoints, bv, blend_opt, p):
        records: list = []
        stop_rec = asyncio.Event()
        interval = 1.0 / p.get('record_hz', 50.0)

        async def _record():
            while not stop_rec.is_set():
                t0 = time.perf_counter()
                try:
                    data = await asyncio.wait_for(
                        data_ch.request_data(), timeout=0.5)
                    records.append({
                        't':      time.time(),
                        'tcp':    list(data.sdata.tcp_ref),
                        'joints': list(data.sdata.jnt_ref),
                    })
                except Exception:
                    pass
                await asyncio.sleep(
                    max(0.0, interval - (time.perf_counter() - t0)))

        rec_task = asyncio.create_task(_record())
        t_start  = time.time()

        mode  = p['mode']
        speed = p['speed']
        accel = p['accel']

        if mode == 'jb2':
            await robot.move_jb2_clear(rc)
            await robot.flush(rc)
            for wp in waypoints:
                await robot.move_jb2_add(rc, wp, speed, accel, bv)
            await robot.flush(rc)
            await robot.move_jb2_run(rc)
            await robot.flush(rc)
        elif mode == 'pb':
            await robot.move_pb_clear(rc)
            await robot.flush(rc)
            for wp in waypoints:
                await robot.move_pb_add(rc, wp, speed, blend_opt, bv)
            await robot.flush(rc)
            await robot.move_pb_run(rc, accel, rb.MovePBOption.Intended)
            await robot.flush(rc)
        else:  # lb
            await robot.move_lb_clear(rc)
            await robot.flush(rc)
            for wp in waypoints:
                await robot.move_lb_add(rc, wp, bv)
            await robot.flush(rc)
            await robot.move_lb_run(rc, speed, accel, rb.MoveLBOption.Intended)
            await robot.flush(rc)

        res = await robot.wait_for_move_started(rc, 10.0)
        if res.type() == rb.ReturnType.Success:
            await robot.wait_for_move_finished(rc)
        else:
            log.warning("wait_for_move_started 타임아웃 (blending=%.3f)", bv)

        stop_rec.set()
        await rec_task

        for r in records:
            r['t'] -= t_start
        return records

    def _emit_log(self, msg: str):
        log.info(msg)
        self.log_msg.emit(msg)
