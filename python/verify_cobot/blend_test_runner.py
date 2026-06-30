"""
BlendTestRunner — blending 값별 궤적 실행 + 기록 QThread

두 가지 궤적 소스:
  Delta mode  : 현재 위치 기준 상대값 waypoints (jb2 / pb / lb)
  CSV mode    : 절대 관절각 또는 TCP 포즈 + 웨이포인트별 speed/accel
               params['use_csv'] = True
               params['mode'] = 'jb2' | 'pb' | 'lb'
               params['waypoint_data'] = [{'joints' 또는 'tcp', 'speed': f, 'accel': f}, ...]
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
from calibration.solver import rb_pose_to_matrix

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
    CSV 궤적을 파싱한다.

    지원 형식:
      - Joint: Order,J1,J2,J3,J4,J5,J6,Speed,Acceleration
      - TCP  : Order,X,Y,Z,Rx,Ry,Rz,Speed,Acceleration

    반환: [{'joints' 또는 'tcp': np.array([6]), 'speed': float, 'accel': float}, ...]
    """
    import csv as _csv
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = _csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        field_map = {name.strip().lower(): name for name in fieldnames}

        def _cols(candidates: list[str]) -> list[str] | None:
            out = []
            for cand in candidates:
                key = cand.lower()
                if key not in field_map:
                    return None
                out.append(field_map[key])
            return out

        joint_cols = _cols(['J1', 'J2', 'J3', 'J4', 'J5', 'J6'])
        tcp_cols = (_cols(['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz'])
                    or _cols(['TCP_X', 'TCP_Y', 'TCP_Z',
                              'TCP_Rx', 'TCP_Ry', 'TCP_Rz']))
        if joint_cols is None and tcp_cols is None:
            raise ValueError(
                "CSV 컬럼 누락: J1..J6 또는 X,Y,Z,Rx,Ry,Rz 형식이 필요합니다")

        for row in reader:
            speed  = float(row.get('Speed', 80))
            accel  = float(row.get('Acceleration', 100))
            if joint_cols is not None:
                joints = np.array([float(row[c]) for c in joint_cols])
                rows.append({'joints': joints, 'speed': speed, 'accel': accel})
            else:
                tcp = np.array([float(row[c]) for c in tcp_cols])
                rows.append({'tcp': tcp, 'speed': speed, 'accel': accel})
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
          mode           'jb2' | 'pb' | 'lb'
          waypoint_data  [{'joints' or 'tcp': array, 'speed': float, 'accel': float}]
          speed, accel   (시작 자세 이동용 fallback)

        natnet_state : NatNetStateProxy  — None 이면 NatNet 기록 생략
        calib        : {'T_base_motive': [[4x4]], 'T_rb_tcp': [[4x4]]}
                       None 이면 HE 변환 생략
        """
        super().__init__(parent)
        self._p = params
        self._natnet = natnet_state          # NatNetStateProxy | None
        self._calib  = calib                 # {'T_base_motive': [...]} | None
        self._T_base_motive: np.ndarray | None = None
        self._T_motive_base: np.ndarray | None = None
        self._T_rb_tcp: np.ndarray | None = None
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
            self._T_base_motive = np.array(self._calib['T_base_motive'], dtype=float)
            self._T_motive_base = np.linalg.inv(self._T_base_motive)
            self._T_rb_tcp = np.array(
                self._calib.get('T_rb_tcp', np.eye(4)),
                dtype=float,
            )
        else:
            self._T_base_motive = None
            self._T_motive_base = None
            self._T_rb_tcp = None

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
            mode = p.get('mode', 'jb2')
            has_joints = all('joints' in wd for wd in waypoint_data)
            has_tcp = all('tcp' in wd for wd in waypoint_data)
            if mode == 'jb2' and not has_joints:
                raise RuntimeError("jb2 모드는 J1..J6 joint CSV가 필요합니다")
            if mode in ('pb', 'lb') and not has_tcp:
                if not has_joints:
                    raise RuntimeError("pb/lb 모드는 TCP CSV 또는 joint CSV가 필요합니다")
                waypoint_data = await self._joint_waypoints_to_tcp(
                    robot, rc, waypoint_data)
                self._emit_log(
                    f"CSV joint 궤적을 {mode} TCP waypoints로 변환: "
                    f"{len(waypoint_data)}개")

            start_joints = (np.array(p['waypoint_data'][0]['joints'])
                            if has_joints else None)
            first_wp = (start_joints if start_joints is not None
                        else np.array(waypoint_data[0]['tcp']))
            self._emit_log(
                f"CSV 궤적 로드: {len(waypoint_data)}개 웨이포인트, mode={mode} "
                f"(WP1 → {np.round(first_wp, 1).tolist()})")
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
            rc.clear()

            self._emit_log(f"[{run_idx}/{total}] blending={bv:.3f} 시작")
            self.run_progress.emit(run_idx, total, bv)

            # pb / lb 모드는 실행 전에 Simulation 모드로 궤적을 검증한다.
            # 시뮬레이터가 실제 로봇의 현재 관절 구성에서 시작하게 해야
            # false negative(실제로는 가능한 궤적을 실패로 판정)를 막을 수 있다.
            if mode in ('pb', 'lb'):
                try:
                    real_joints = await self._read_joints(data_ch)
                except Exception:
                    real_joints = start_joints  # fallback
                try:
                    await self._validate_task_path_in_simulation(
                        robot, rc, waypoint_data, bv, blend_opt, p,
                        start_joints=real_joints)
                except Exception as e:
                    self._emit_log(
                        f"[{run_idx}/{total}] 시뮬레이션 검증 실패 — 실제 실행 건너뜀: {e}")
                    continue

            try:
                await self._move_to_start(robot, data_ch, rc, start_joints,
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

    async def _read_joints(self, data_ch) -> np.ndarray:
        data = await asyncio.wait_for(data_ch.request_data(), timeout=1.0)
        return np.array(data.sdata.jnt_ref, dtype=float)

    async def _joint_waypoints_to_tcp(self, robot, rc,
                                      waypoint_data: list[dict]) -> list[dict]:
        converted = []
        for wd in waypoint_data:
            joints = np.array(wd['joints'], dtype=float)
            res, tcp = await robot.calc_fk_tcp(rc, joints)
            await robot.flush(rc)
            rc.error().throw_if_not_empty()
            if res.type() != rb.ReturnType.Success:
                raise RuntimeError(f"calc_fk_tcp failed for {np.round(joints, 2).tolist()}")
            converted.append({
                'joints': joints,
                'tcp': np.array(tcp, dtype=float),
                'speed': float(wd['speed']),
                'accel': float(wd['accel']),
            })
        return converted

    async def _wait_until_joints_close(self, data_ch, target_joints,
                                       timeout=30.0, tol_deg=0.5,
                                       stable_count=3) -> bool:
        target = np.array(target_joints, dtype=float)
        deadline = time.time() + timeout
        stable = 0
        last_err = None
        while time.time() < deadline and not self._stop_event.is_set():
            try:
                q = await self._read_joints(data_ch)
            except Exception:
                await asyncio.sleep(0.05)
                continue
            last_err = float(np.max(np.abs(q - target)))
            if last_err <= tol_deg:
                stable += 1
                if stable >= stable_count:
                    return True
            else:
                stable = 0
            await asyncio.sleep(0.05)
        if last_err is not None:
            self._emit_log(f"  └ start pose wait timeout: max joint err={last_err:.3f} deg")
        return False

    async def _set_operation_mode(self, robot, rc, op_mode):
        rc.clear()
        await robot.set_operation_mode(rc, op_mode)
        await robot.flush(rc)
        rc.error().throw_if_not_empty()
        rc.clear()

    async def _validate_task_path_in_simulation(self, robot, rc,
                                                waypoint_data, bv, blend_opt, p,
                                                start_joints=None):
        """
        pb / lb 궤적을 Simulation 모드에서 사전 실행해 워크스페이스 이탈 등을 검증.

        핵심 수정:
          - 시뮬레이션이 실제보다 빠르게 완료되면 wait_for_move_started 가
            타임아웃으로 반환될 수 있다. 이 경우 rc.error() 가 비어 있으면
            정상 완료로 간주한다 (기존 코드의 버그: 타임아웃만으로 에러처리).
          - 진짜 에러(워크스페이스 이탈, IK 실패 등)는 rc.error() 를 통해
            감지해 상세 메시지를 로그에 남긴다.
        """
        mode = p.get('mode', 'jb2')
        self._emit_log(f"  └ [Sim] {mode} 궤적 사전 검증 (blending={bv:.3f})")
        try:
            await self._set_operation_mode(robot, rc, rb.OperationMode.Simulation)

            # 시작 자세로 이동 (관절 정보가 있을 때만)
            if start_joints is not None:
                rc.clear()
                await robot.move_j(rc, np.array(start_joints, dtype=float),
                                   p['speed'], p['accel'])
                await robot.flush(rc)
                rc.error().throw_if_not_empty()
                res = await robot.wait_for_move_started(rc, 3.0)
                if res.type() == rb.ReturnType.Success:
                    await robot.wait_for_move_finished(rc, 30.0)
                # 빠른 완료 허용: 에러가 없으면 OK
                rc.error().throw_if_not_empty()
                rc.clear()

            # 궤적 커맨드 전송
            await self._send_trajectory_command(
                robot, rc, waypoint_data, bv, blend_opt,
                mode, p['speed'], p['accel'])

            # 시뮬레이션 실행 대기
            # wait_for_move_started 가 타임아웃해도 즉시 실패로 보지 않는다.
            res = await robot.wait_for_move_started(rc, 3.0)
            if res.type() == rb.ReturnType.Success:
                res_fin = await robot.wait_for_move_finished(rc, 60.0)
                if res_fin.type() != rb.ReturnType.Success:
                    raise RuntimeError("시뮬레이션 모션이 제한 시간 내 완료되지 않았습니다")
            else:
                # 이미 완료됐거나 아직 시작 전: 에러 여부로 판단
                self._emit_log("  └ [Sim] move_started 이벤트 미수신 (빠른 완료 또는 미시작)")

            # 에러 확인 — 워크스페이스 이탈, IK 실패 등 모두 여기서 잡힘
            rc.error().throw_if_not_empty()
            self._emit_log("  └ [Sim] 검증 통과 — 실제 실행 진행")

        except Exception as exc:
            self._emit_log(f"  └ [Sim] 검증 실패: {exc}")
            rc.clear()
            raise RuntimeError(f"simulation precheck failed: {exc}") from exc
        finally:
            try:
                rc.clear()
                await self._set_operation_mode(robot, rc, rb.OperationMode.Real)
            except Exception:
                pass
            await asyncio.sleep(0.2)

    async def _move_to_start(self, robot, data_ch, rc, start_joints, speed, accel):
        if start_joints is None:
            self._emit_log("  └ CSV has no joint start pose; skipping start joint move")
            return
        rc.clear()
        start_joints = np.array(start_joints, dtype=float)
        try:
            current = await self._read_joints(data_ch)
            err = float(np.max(np.abs(current - start_joints)))
            if err <= 0.5:
                self._emit_log(f"  └ already at start pose (max err={err:.3f} deg)")
                return
            self._emit_log(f"  └ moving to start pose (max err={err:.3f} deg)")
        except Exception as exc:
            self._emit_log(f"  └ could not read current joints before start move: {exc}")

        await robot.move_j(rc, start_joints, speed, accel)
        await robot.flush(rc)
        rc.error().throw_if_not_empty()

        res = await robot.wait_for_move_started(rc, 2.0)
        if res.type() == rb.ReturnType.Success:
            await robot.wait_for_move_finished(rc, 60.0)
        else:
            self._emit_log("  └ start move event timeout; checking joint feedback")

        ok = await self._wait_until_joints_close(data_ch, start_joints)
        rc.error().throw_if_not_empty()
        if not ok:
            raise RuntimeError("start pose was not reached")
        rc.clear()
        await asyncio.sleep(0.2)

    async def _send_trajectory_command(self, robot, rc, waypoint_data,
                                       bv, blend_opt, mode, speed, accel):
        if mode == 'jb2':
            await robot.move_jb2_clear(rc)
            await robot.flush(rc)
            for idx, wd in enumerate(waypoint_data):
                wp_blend = 0.0 if idx == 0 or idx == len(waypoint_data) - 1 else bv
                await robot.move_jb2_add(
                    rc, wd['joints'], wd['speed'], wd['accel'], wp_blend)
            await robot.flush(rc)
            await robot.move_jb2_run(rc)
            await robot.flush(rc)
        elif mode == 'pb':
            await robot.move_pb_clear(rc)
            await robot.flush(rc)
            for idx, wd in enumerate(waypoint_data):
                wp_blend = 0.0 if idx == 0 or idx == len(waypoint_data) - 1 else bv
                await robot.move_pb_add(
                    rc, wd['tcp'], wd['speed'], blend_opt, wp_blend)
            await robot.flush(rc)
            await robot.move_pb_run(rc, accel, rb.MovePBOption.Intended)
            await robot.flush(rc)
        else:  # lb
            await robot.move_lb_clear(rc)
            await robot.flush(rc)
            for idx, wd in enumerate(waypoint_data):
                wp_blend = 0.0 if idx == 0 or idx == len(waypoint_data) - 1 else bv
                await robot.move_lb_add(rc, wd['tcp'], wp_blend)
            await robot.flush(rc)
            await robot.move_lb_run(rc, speed, accel, rb.MoveLBOption.Intended)
            await robot.flush(rc)
        rc.error().throw_if_not_empty()

    async def _execute_with_recording(self, robot, data_ch, rc,
                                       waypoint_data, bv, blend_opt, p,
                                       use_csv: bool):
        mode  = p.get('mode', 'jb2')
        speed = p['speed']
        accel = p['accel']

        records: list = []
        stop_rec = asyncio.Event()
        interval = 1.0 / p.get('record_hz', 50.0)
        mocap_max_age = float(p.get('mocap_max_age_s', 0.5))

        async def _sample_once():
            data = await asyncio.wait_for(data_ch.request_data(), timeout=0.5)
            now = time.time()

            # NatNet RB 위치·자세 샘플. Blend plotting은 연속성 확인이 목적이라
            # 100ms보다 약간 느린 mocap 갱신도 놓치지 않도록 최신 프레임 fallback을 둔다.
            rb_pos = rb_rot = None
            if self._natnet is not None:
                sample = self._natnet.get_rb_since(now - mocap_max_age)
                if sample is None:
                    sample = self._natnet.get_rb_since(0.0)
                if sample is not None:
                    rb_pos = sample[0].tolist()   # [x,y,z] m, motive frame
                    rb_rot = sample[1].tolist()   # [qx,qy,qz,qw], motive frame

            # FK + HE 변환: robot TCP base → motive frame
            tcp_motive = None
            if self._T_motive_base is not None:
                tcp_m = np.array(data.sdata.tcp_ref[:3]) / 1000.0
                p_h = np.array([*tcp_m, 1.0])
                tcp_motive = (self._T_motive_base @ p_h)[:3].tolist()

            # NatNet RB + SVD calibration: motive RB → robot-base TCP.
            # This puts NatNet and robot TCP in the same coordinate frame.
            rb_tcp_base = None
            if (rb_pos is not None and rb_rot is not None
                    and self._T_base_motive is not None
                    and self._T_rb_tcp is not None):
                T_motive_rb = rb_pose_to_matrix(rb_pos, rb_rot)
                T_pred_tcp = self._T_base_motive @ T_motive_rb @ self._T_rb_tcp
                rb_tcp_base = T_pred_tcp[:3, 3].tolist()

            return {
                't':          now,
                'tcp':        list(data.sdata.tcp_ref),  # [x,y,z mm, rx,ry,rz deg]
                'joints':     list(data.sdata.jnt_ref),  # [J1..J6 deg]
                'rb_pos':     rb_pos,                     # [x,y,z] m  or None
                'rb_rot':     rb_rot,                     # [qx,qy,qz,qw] or None
                'tcp_motive': tcp_motive,                 # [x,y,z] m  or None
                'rb_tcp_base': rb_tcp_base,                # calibrated NatNet TCP in base
            }

        async def _record():
            while not stop_rec.is_set():
                t0 = time.perf_counter()
                try:
                    records.append(await _sample_once())
                except Exception:
                    pass
                await asyncio.sleep(
                    max(0.0, interval - (time.perf_counter() - t0)))

        rec_task = asyncio.create_task(_record())

        command_start = time.time()
        rc.clear()

        await self._send_trajectory_command(
            robot, rc, waypoint_data, bv, blend_opt, mode, speed, accel)

        command_sent = time.time()

        # 모션이 실제로 시작된 시각을 기준으로 시간 정규화한다.
        # 시작 감지가 timeout되면 wait 반환 시각이 아니라 command 시각을 기준으로
        # 기록을 남긴다. 그렇지 않으면 timeout 동안 모은 샘플이 전부 버려진다.
        res = await robot.wait_for_move_started(rc, 10.0)
        if res.type() == rb.ReturnType.Success:
            t_motion_start = time.time()
            await robot.wait_for_move_finished(rc)
            await asyncio.sleep(float(p.get('final_settle_s', 0.2)))
            try:
                final_rec = await _sample_once()
                final_rec['is_final_sample'] = True
                records.append(final_rec)
            except Exception as exc:
                log.warning("final sample failed (blending=%.3f): %s", bv, exc)
        else:
            t_motion_start = command_sent
            log.warning(
                "wait_for_move_started timeout (blending=%.3f); "
                "keeping samples from command time", bv)
            await asyncio.sleep(max(interval * 2.0, 0.05))

        stop_rec.set()
        await rec_task

        # 모션 시작/커맨드 직전부터의 데이터만 유지, t=0 = 기준 시각
        keep_from = min(command_start, t_motion_start - 0.05)
        records_out = [r for r in records
                       if r['t'] >= keep_from]
        if not records_out and records:
            records_out = records
        for r in records_out:
            r['t'] -= t_motion_start
        if self._calib:
            n_cal = sum(1 for r in records_out if r.get('rb_tcp_base') is not None)
            if n_cal == 0:
                self._emit_log(
                    f"[WARN] blending={bv:.3f}: calibration loaded but no calibrated NatNet samples. "
                    "Check NatNet connection/tracking and SVD T_base_motive/T_rb_tcp.")
            else:
                self._emit_log(
                    f"[INFO] blending={bv:.3f}: calibrated NatNet samples {n_cal}/{len(records_out)}")
        if records_out:
            final_tcp = np.array(records_out[-1]['tcp'][:3], dtype=float)
            final_j = np.array(records_out[-1]['joints'], dtype=float)
            last_wp = waypoint_data[-1] if waypoint_data else {}
            if mode == 'jb2' and 'joints' in last_wp:
                err = float(np.linalg.norm(final_j - np.array(last_wp['joints'], dtype=float)))
                records_out[-1]['final_joint_error_norm_deg'] = err
                self._emit_log(
                    f"[INFO] blending={bv:.3f}: final joint error norm {err:.4f} deg")
            elif mode in ('pb', 'lb') and 'tcp' in last_wp:
                err = float(np.linalg.norm(final_tcp - np.array(last_wp['tcp'][:3], dtype=float)))
                records_out[-1]['final_tcp_position_error_mm'] = err
                self._emit_log(
                    f"[INFO] blending={bv:.3f}: final TCP position error {err:.4f} mm")
        return records_out

    def _emit_log(self, msg: str):
        log.info(msg)
        self.log_msg.emit(msg)
