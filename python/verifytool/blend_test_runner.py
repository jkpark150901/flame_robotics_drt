"""
BlendTestRunner — blending 값별 궤적 실행 + 기록 QThread

두 가지 궤적 소스:
  Delta mode  : 현재 위치 기준 상대값 waypoints (jb2 / pb / lb)
  CSV mode    : 절대 관절각 + 웨이포인트별 speed/accel (jb2 전용)
               params['use_csv'] = True
               params['waypoint_data'] = [{'joints': [...], 'speed': f, 'accel': f}, ...]
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

# Delta mode 기본 waypoints
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


def parse_trajectory_csv(path: str) -> list[dict]:
    """
    'Order,J1,J2,J3,J4,J5,J6,Speed,Acceleration' 형식 CSV를 파싱.
    반환: [{'joints': np.array([6]), 'speed': float, 'accel': float}, ...]
    """
    import csv as _csv
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = _csv.DictReader(f)
        required = {'J1', 'J2', 'J3', 'J4', 'J5', 'J6'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV 컬럼 누락 (필요: {required})")
        for row in reader:
            joints = np.array([float(row[f'J{i}']) for i in range(1, 7)])
            speed  = float(row.get('Speed', 80))
            accel  = float(row.get('Acceleration', 100))
            rows.append({'joints': joints, 'speed': speed, 'accel': accel})
    if not rows:
        raise ValueError("CSV에 데이터가 없습니다")
    return rows


def compute_plan_deviation(records: list[dict],
                            waypoint_data: list[dict]) -> np.ndarray:
    """
    기록된 샘플 시계열과 계획 궤적 간 편차(deg)를 계산.

    계획 궤적은 웨이포인트 간 joint-space 거리와 speed를 이용해 시간 파라미터화한 뒤
    각 샘플 시각에서 선형 보간한다.

    반환: shape (N,) — 각 샘플의 6-DOF 평균 편차 (deg)
    """
    if not records or not waypoint_data:
        return np.array([])

    joints_plan = np.array([wd['joints'] for wd in waypoint_data])  # (M, 6)
    speeds      = np.array([wd['speed']  for wd in waypoint_data])  # (M,)

    # 각 세그먼트 이동 시간 추정 (첫 WP→둘째 WP, …)
    seg_dist = np.linalg.norm(np.diff(joints_plan, axis=0), axis=1)   # (M-1,)
    seg_spd  = speeds[:-1]
    seg_time = np.where(seg_dist > 0.01, seg_dist / seg_spd, 0.0)     # (M-1,)
    t_plan   = np.concatenate([[0.0], np.cumsum(seg_time)])            # (M,)

    t_rec    = np.array([r['t'] for r in records])
    j_rec    = np.array([r['joints'] for r in records])                # (N, 6)

    # 계획 궤적 시간 범위로 클리핑
    t_rec_clipped = np.clip(t_rec, t_plan[0], t_plan[-1])

    # 시간별 계획 관절각 선형 보간 (scipy 없이 numpy interp)
    j_plan_interp = np.column_stack([
        np.interp(t_rec_clipped, t_plan, joints_plan[:, j]) for j in range(6)
    ])  # (N, 6)

    # 6-DOF RMS
    deviation = np.sqrt(np.mean((j_rec - j_plan_interp) ** 2, axis=1))
    return deviation


class BlendTestRunner(QThread):
    """
    Signals
    -------
    run_progress(run_idx, total, blending)
    run_done(blending, records)
    all_done(dict)   {blending: records}
    log_msg(str)
    error(str)
    """
    run_progress = pyqtSignal(int, int, float)
    run_done     = pyqtSignal(float, list)
    all_done     = pyqtSignal(dict)
    log_msg      = pyqtSignal(str)
    error        = pyqtSignal(str)

    def __init__(self, params: dict,
                 natnet_state=None, calib: dict | None = None,
                 parent=None):
        """
        params 필수 키
        --------------
        robot_ip, blending_values, speed_bar, record_hz

        Delta mode (use_csv=False):
          mode, waypoint_deltas, speed, accel, blending_option

        CSV mode (use_csv=True):
          waypoint_data  [{'joints': array, 'speed': float, 'accel': float}]
          speed, accel   (시작 자세 이동용 fallback)

        natnet_state : NatNetStateProxy  — None 이면 NatNet 기록 생략
        calib        : {'T_base_motive': [[4x4]]}  — None 이면 HE 변환 생략
        """
        super().__init__(parent)
        self._p = params
        self._natnet = natnet_state          # NatNetStateProxy | None
        self._calib  = calib                 # {'T_base_motive': [...]} | None
        self._T_motive_base: np.ndarray | None = None
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
        use_csv = p.get('use_csv', False)
        blend_opt = (rb.BlendingOption.Ratio
                     if p.get('blending_option', 'ratio') == 'ratio'
                     else rb.BlendingOption.Distance)

        # HE 변환 행렬 사전 계산
        if self._calib:
            T = np.array(self._calib['T_base_motive'], dtype=float)
            self._T_motive_base = np.linalg.inv(T)
        else:
            self._T_motive_base = None

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

        # ── 웨이포인트 준비 ──────────────────────────────────────────
        if use_csv:
            waypoint_data = p['waypoint_data']   # [{'joints', 'speed', 'accel'}]
            start_joints  = np.array(waypoint_data[0]['joints'])
            self._emit_log(
                f"CSV 궤적 로드: {len(waypoint_data)}개 웨이포인트 "
                f"(WP1 → {np.round(start_joints, 1).tolist()})")
        else:
            data = await asyncio.wait_for(data_ch.request_data(), timeout=3.0)
            cur_tcp    = np.array(data.sdata.tcp_ref, dtype=float)
            cur_joints = np.array(data.sdata.jnt_ref, dtype=float)
            start_joints = cur_joints.copy()
            self._emit_log(f"현재 TCP   : {np.round(cur_tcp, 2).tolist()}")
            self._emit_log(f"현재 관절  : {np.round(cur_joints, 2).tolist()}")

            deltas = np.array(p['waypoint_deltas'], dtype=float)
            spd, acc = p['speed'], p['accel']
            mode = p['mode']
            if mode == 'jb2':
                waypoint_data = [
                    {'joints': cur_joints + d, 'speed': spd, 'accel': acc}
                    for d in deltas
                ]
            else:
                # pb / lb: TCP 좌표 사용 (joints 키 없음)
                waypoint_data = [
                    {'tcp': cur_tcp + d, 'speed': spd, 'accel': acc}
                    for d in deltas
                ]

        blending_values = p['blending_values']
        total = len(blending_values)
        all_results: dict = {}

        for run_idx, bv in enumerate(blending_values, start=1):
            if self._stop_event.is_set():
                self._emit_log("사용자 중단.")
                break

            self._emit_log(f"[{run_idx}/{total}] blending={bv:.3f} 시작")
            self.run_progress.emit(run_idx, total, bv)

            try:
                await self._move_to_start(robot, rc, start_joints,
                                          p['speed'], p['accel'])
            except Exception as e:
                self._emit_log(f"[{run_idx}/{total}] 시작 자세 복귀 실패 — skip: {e}")
                continue

            try:
                records = await self._execute_with_recording(
                    robot, data_ch, rc, waypoint_data,
                    bv, blend_opt, p, use_csv)
            except Exception as e:
                self._emit_log(f"[{run_idx}/{total}] 실행 중 오류 — skip: {e}")
                continue

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
                                       waypoint_data, bv, blend_opt, p,
                                       use_csv: bool):
        records: list = []
        stop_rec = asyncio.Event()
        interval = 1.0 / p.get('record_hz', 50.0)

        async def _record():
            while not stop_rec.is_set():
                t0 = time.perf_counter()
                try:
                    data = await asyncio.wait_for(
                        data_ch.request_data(), timeout=0.5)
                    now = time.time()

                    # NatNet RB 위치·자세 샘플 (최근 100ms 이내)
                    rb_pos = rb_rot = None
                    if self._natnet is not None:
                        sample = self._natnet.get_rb_since(now - 0.1)
                        if sample is not None:
                            rb_pos = sample[0].tolist()   # [x,y,z] m, motive frame
                            rb_rot = sample[1].tolist()   # [qx,qy,qz,qw], motive frame

                    # FK + HE 변환: TCP position → motive frame
                    tcp_motive = None
                    if self._T_motive_base is not None:
                        tcp_m = np.array(data.sdata.tcp_ref[:3]) / 1000.0
                        p_h = np.array([*tcp_m, 1.0])
                        tcp_motive = (self._T_motive_base @ p_h)[:3].tolist()

                    records.append({
                        't':          now,
                        'tcp':        list(data.sdata.tcp_ref),  # [x,y,z mm, rx,ry,rz deg]
                        'joints':     list(data.sdata.jnt_ref),  # [J1..J6 deg]
                        'rb_pos':     rb_pos,                     # [x,y,z] m  or None
                        'rb_rot':     rb_rot,                     # [qx,qy,qz,qw] or None
                        'tcp_motive': tcp_motive,                 # [x,y,z] m  or None
                    })
                except Exception:
                    pass
                await asyncio.sleep(
                    max(0.0, interval - (time.perf_counter() - t0)))

        rec_task = asyncio.create_task(_record())

        mode  = 'jb2' if use_csv else p['mode']
        speed = p['speed']
        accel = p['accel']

        if mode == 'jb2':
            await robot.move_jb2_clear(rc)
            await robot.flush(rc)
            for wd in waypoint_data:
                await robot.move_jb2_add(
                    rc, wd['joints'], wd['speed'], wd['accel'], bv)
            await robot.flush(rc)
            await robot.move_jb2_run(rc)
            await robot.flush(rc)
        elif mode == 'pb':
            await robot.move_pb_clear(rc)
            await robot.flush(rc)
            for wd in waypoint_data:
                await robot.move_pb_add(
                    rc, wd['tcp'], wd['speed'], blend_opt, bv)
            await robot.flush(rc)
            await robot.move_pb_run(rc, accel, rb.MovePBOption.Intended)
            await robot.flush(rc)
        else:  # lb
            await robot.move_lb_clear(rc)
            await robot.flush(rc)
            for wd in waypoint_data:
                await robot.move_lb_add(rc, wd['tcp'], bv)
            await robot.flush(rc)
            await robot.move_lb_run(rc, speed, accel, rb.MoveLBOption.Intended)
            await robot.flush(rc)

        # 모션이 실제로 시작된 시각을 기준으로 시간 정규화
        # (커맨드 전송 구간 · 이전 이동 잔상을 제거)
        res = await robot.wait_for_move_started(rc, 10.0)
        t_motion_start = time.time()
        if res.type() == rb.ReturnType.Success:
            await robot.wait_for_move_finished(rc)
        else:
            log.warning("wait_for_move_started timeout (blending=%.3f)", bv)

        stop_rec.set()
        await rec_task

        # 모션 시작 50ms 전부터의 데이터만 유지, t=0 = 모션 시작
        records_out = [r for r in records
                       if r['t'] >= t_motion_start - 0.05]
        for r in records_out:
            r['t'] -= t_motion_start
        return records_out

    def _emit_log(self, msg: str):
        log.info(msg)
        self.log_msg.emit(msg)
