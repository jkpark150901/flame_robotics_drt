"""궤적을 move_jb2_add / move_servo_j 로 추종시키면서 측정각(reqdata)을 기록한다. ** 시뮬레이션 전용 **

- `pgmode real` 은 어떤 경로로도 보내지 않는다.
- 시작 전/단계 전/기록 중/대기 중에 real 모드가 감지되면 즉시 중단한다. real 모드이면 정지(idle) 상태에서만
  simulation 으로 전환한다.
- 모션 명령은 직전 모션이 끝난 것을 확인한 뒤에만 보낸다 (측정각이 목표에 도달 + move_j/jb2 는 idle).
- 로봇이 움직이는 동안에는 계속 기다린다(움직임 없이 timeout 초가 지나야 포기).
"""
import csv
import json
import os
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime

import numpy as np

from . import trajectory
from .robot import CMD_PORT, DATA_PORT, CmdPort, DataPort, NotSimulation, StopRequested


def fmt_jnt(q_deg):
    return "jnt[" + ",".join(f"{v:.4f}" for v in q_deg) + "]"


class Recorder:
    """상태 포트를 일정 주기로 읽어 측정각을 기록하는 스레드."""

    def __init__(self, ip, hz, port=DATA_PORT, expect_sim=1):
        self.expect_sim = expect_sim
        self.dp = DataPort(ip, port)
        self.period = 1.0 / hz
        self.rows = []
        self.phase = "idle"
        self.marks = {}
        self.err = None
        self._stop = False
        self.t0 = time.perf_counter()
        self.th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        nxt = time.perf_counter()
        while not self._stop:
            try:
                d = self.dp.read()
                if d["sim"] != self.expect_sim:
                    raise NotSimulation(f"기록 중 모드가 예상과 달라짐 (sim={d['sim']})")
                self.rows.append((time.perf_counter() - self.t0, self.phase, d["robot_state"], *d["jmeas"]))
            except Exception as e:  # noqa: BLE001
                self.err = e
                return
            nxt += self.period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def start(self):
        self.th.start()

    def stop(self):
        self._stop = True
        self.th.join(timeout=2)
        self.dp.close()

    def mark(self, name):
        self.marks[name] = time.perf_counter() - self.t0

    def check(self):
        if self.err is not None:
            raise self.err

    def save(self, path, meta):
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("# " + json.dumps(meta, ensure_ascii=False) + "\n")
            w = csv.writer(f)
            w.writerow(["t", "phase", "robot_state"] + [f"meas{i}" for i in range(6)])
            for r in self.rows:
                w.writerow([f"{r[0]:.4f}", r[1], r[2]] + [f"{v:.5f}" for v in r[3:]])


class Runner:
    def __init__(self, s, log=print, stop=None):
        self.s = s
        self.log = log
        self.stop = stop or threading.Event()
        self.run_dir = None
        self.results = None
        self.expect_sim = 0 if s.robot_mode == "real" else 1

    # ------------------------------------------------------------ 대기/게이트
    def _chk(self):
        if self.stop.is_set():
            raise StopRequested("사용자 중지")

    def wait_reached(self, rec, dp, target_deg, timeout, label="", need_idle=False):
        """측정각이 target 의 reach_tol 이내(+ need_idle 이면 idle)로 reach_hold 초 유지되면 도달.
        움직이는 동안은 계속 기다리고, 움직임 없이 timeout 초가 지나거나 max_wait 를 넘기면 False."""
        s = self.s
        target = np.asarray(target_deg, dtype=float)
        t0 = last_move = time.time()
        in_since, prev = None, None
        err, d = float("nan"), {"robot_state": -1}
        while time.time() - t0 < s.max_wait:
            self._chk()
            rec.check()
            d = dp.read()
            if d["sim"] != self.expect_sim:
                raise NotSimulation(f"대기 중 모드가 예상과 달라짐 (sim={d['sim']})")
            now = time.time()
            if d["robot_state"] != 1 or (prev is not None and np.abs(d["jmeas"] - prev).max() > 0.02):
                last_move = now
            prev = d["jmeas"]
            err = np.abs(d["jmeas"] - target).max()
            if err < s.reach_tol and (not need_idle or d["robot_state"] == 1):
                in_since = in_since or now
                if now - in_since >= s.reach_hold:
                    return True
            else:
                in_since = None
            if now - last_move > timeout:
                break
            time.sleep(0.05)
        self.log(f"  {label} 도달 못 함 (움직임 없이 {timeout:g}s 경과 또는 max_wait, 최대 오차 {err:.2f} deg, robot_state={d['robot_state']})")
        return False

    def _finish(self, gate, timeout):
        """직전 명령이 끝날 때까지 대기. 끝났으면 True."""
        if gate["pending"] is None:
            return True
        target, label, check_idle = gate["pending"]
        if not self.wait_reached(gate["rec"], gate["dp"], target, timeout, label, need_idle=check_idle):
            return False
        gate["dp"].require_mode(self.expect_sim)
        gate["pending"] = None
        return True

    def _send(self, gate, line, target, label, check_idle=True, wait=True):
        """직전 명령 종료 대기 -> 전송 -> (wait 면) 이 명령 종료까지 대기."""
        if not self._finish(gate, self.s.timeout):
            raise TimeoutError(f"이전 명령 종료 대기 시간 초과 -> '{line[:40]}...' 전송 안 함")
        gate["cmd"].send(line)
        gate["pending"] = (target, label, check_idle)
        if wait and not self._finish(gate, self.s.timeout):
            raise TimeoutError(f"{label} 타임아웃")

    # ------------------------------------------------------------ 모드별 실행
    def _go_home(self, gate, rec, m):
        s = self.s
        if not s.home_first:
            return
        rec.phase = f"home_{m}"
        self.log(f"[{m}] 원점 복귀 {s.home_joints}")
        self._send(gate, f"move_j({fmt_jnt(s.home_joints)}, {s.home_speed:g}, {s.home_accel:g})", s.home_joints, "원점 복귀")

    def _run_jb2(self, cmd, rec, dp, q):
        s = self.s
        gate = {"cmd": cmd, "rec": rec, "dp": dp, "pending": None}
        self._go_home(gate, rec, "jb2")
        rec.phase = "jb2_move_to_start"
        self._send(gate, f"move_j({fmt_jnt(q[0])}, {s.speed:g}, {s.accel:g})", q[0], "시작 자세 이동")

        rec.phase = "jb2"
        self._finish(gate, s.timeout)
        rec.mark("jb2_build")
        cmd.send("move_jb2_clear()", ack=True)
        for i, p in enumerate(q[1:], 1):
            cmd.send(f"move_jb2_add({fmt_jnt(p)}, {s.speed:g}, {s.accel:g}, 0, {s.blend:g})", ack=True)
            if cmd.denied:
                raise RuntimeError("move_jb2_add 가 거부됨 (not allowed)")
            if i % 20 == 0:
                self._chk()
        time.sleep(0.5)
        dp.require_mode(self.expect_sim)
        rec.mark("jb2")  # 추종 지령 시각
        self._send(gate, "move_jb2_run(0)", q[-1], "jb2 종료", wait=False)
        ok = self._finish(gate, s.timeout)
        if ok:
            rec.mark("jb2_done")
        rec.phase = "jb2_end"
        return ok

    def _run_servo(self, cmd, rec, dp, q, t):
        s = self.s
        gate = {"cmd": cmd, "rec": rec, "dp": dp, "pending": None}
        self._go_home(gate, rec, "servo")
        rec.phase = "servo_move_to_start"
        self._send(gate, f"move_j({fmt_jnt(q[0])}, {s.speed:g}, {s.accel:g})", q[0], "시작 자세 이동")

        rec.phase = "servo"
        targets, send_t, intervals = self._servo_plan(q, t)
        n_path = len(targets)
        profile = s.servo_timing == "profile"
        hold_dt = float(intervals[-1])
        # 마지막 점 뒤에 같은 목표를 final_hold 번 더 보낸다. 실측상 마지막 지령 후 로봇은 마지막 지령 속도 × t2 만큼 더 나아가 멈추므로
        # (오차 = v_last × t2, 사용자 기록에서 6관절 일치), 같은 점을 반복해 마지막 지령 속도를 0 으로 만든다.
        # final_t2: 마지막 지령(과 반복 지령)에만 작은 t2 를 줘서, 마지막 지령 이후 더 나아가는 거리(마지막 속도 × t2)를 줄인다.
        n_hold = max(int(s.final_hold), 0)
        pts = list(targets) + [q[-1]] * n_hold
        durations = list(intervals) + [hold_dt] * n_hold
        planned = list(send_t) + [float(send_t[-1] + intervals[-1] + hold_dt * h) for h in range(n_hold)]
        self.servo_log = []
        start = time.perf_counter()
        rec.mark("servo")  # 추종 지령 시각
        for i, (p, interval, t_plan) in enumerate(zip(pts, durations, planned), 1):
            self._chk()
            time.sleep(max(0.0, start + t_plan - time.perf_counter()))
            t1 = s.t1 if s.t1 else max(float(interval), 0.002)
            # t2 자동값: profile 은 전송 주기의 2배(20 ms 주기 → 0.04 s), original 은 전송 간격. 0.02~0.2 로 제한
            t2_cmd = min(max(s.t2 if s.t2 else float(interval) * (2 if profile else 1), 0.02), 0.2)
            if i >= n_path and s.final_t2:
                t2_cmd = min(max(s.final_t2, 0.02), 0.2)
            t_sent = time.perf_counter() - start
            if t_sent - t_plan > max(0.05, 2 * float(interval)):
                raise RuntimeError(f"servo 전송 지연 {1000 * (t_sent - t_plan):.0f} ms (지령 {i}): servo 전송 주기(servo_period)를 늘리세요")
            cmd.send(f"move_servo_j({fmt_jnt(p)}, {t1:g}, {t2_cmd:g}, {s.gain:g}, {s.alpha:g})")
            self.servo_log.append((t_plan, t_sent, t1, t2_cmd, *p))
            if cmd.denied:
                raise RuntimeError("move_servo_j 가 거부됨 (not allowed)")
            if i % 10 == 0:
                rec.check()
        time.sleep(max(0.0, start + planned[-1] + durations[-1] - time.perf_counter()))  # 마지막 지령의 도달 시간까지
        rec.mark("servo_sent")
        lag = max(r[1] - r[0] for r in self.servo_log)
        self.log(f"[servo] {n_path}개 목표 + 반복 {n_hold}회 전송, 계획 대비 전송 지연 최대 {lag * 1000:.1f} ms")
        gate["pending"] = (q[-1], "servo 종료", False)  # servo 는 robot_state 의미가 불명확해 idle 검사 안 함
        ok = self._finish(gate, s.settle_timeout)
        if ok:
            rec.mark("servo_done")
        rec.phase = "servo_end"
        return ok

    def _servo_profile(self, q):
        """profile 타이밍의 이상 프로파일. 경로는 조밀한 원본 궤적(run 에서 지정, 없으면 지령점)."""
        s = self.s
        path = getattr(self, "_q_ref", None)
        key = (id(path), s.speed, s.accel, s.servo_basis, s.servo_period)
        if getattr(self, "_prof_key", None) != key:
            # 실제로 보낼 지령(servo_period 샘플)의 관절 속도·가속도가 speed·accel 을 넘지 않도록 맞춘 프로파일
            self._prof = trajectory.PathProfile.fit(q if path is None else path, s.speed, s.accel, s.servo_basis, s.servo_period)
            self._prof_key = key
        return self._prof

    def _servo_plan(self, q, t):
        """servo 전송 시간표: (목표 관절각 (n,6), 전송 시각 [s, 첫 전송 기준], 도달 시간 = t1).

        profile: 원본 궤적 위에서 speed/accel(관절 속도·가속도 한계)로 만든 이상 프로파일을 servo_period 주기로 샘플링.
                 목표 k 는 시각 k·dt 의 프로파일 위치이며 (k−1)·dt 에 보낸다 (jb2_gui 의 servo 와 같은 방식).
        original: 서브샘플한 지령점을 원본 궤적의 시간 간격 그대로 보낸다.
        """
        s = self.s
        if s.servo_timing == "profile":
            return self._servo_profile(q).schedule(s.servo_period)
        if s.servo_timing != "original":
            raise ValueError("servo_timing 은 profile | original")
        return q[1:], t[:-1] - t[0], np.diff(t)

    def _method_params(self, m, dt):
        s = self.s
        if m == "jb2":
            return {"method": "move_jb2_add", "speed": s.speed, "accel": s.accel, "blend": s.blend}
        p = {"method": "move_servo_j", "timing": s.servo_timing}
        if s.servo_timing == "profile":
            p.update({"speed": s.speed, "accel": s.accel, "basis": s.servo_basis, "geometry": "reference", "period": s.servo_period})
            q = getattr(self, "_q_cmd", None)
            if q is not None:
                prof = self._servo_profile(q)
                tg, _, iv = prof.schedule(s.servo_period)
                p.update({"profile_T": round(prof.T, 4), "profile_vpeak": round(prof.vp, 4), "n_sent": int(len(tg)),
                          "period_actual": round(float(iv[0]), 5), "speed_plan": round(prof.speed_plan, 4), "accel_plan": round(prof.accel_plan, 4),
                          "cmd_speed_max": round(prof.cmd_speed_max, 3), "cmd_accel_max": round(prof.cmd_accel_max, 3)})
        auto_t2 = "2x_period" if s.servo_timing == "profile" else min(max(dt, 0.02), 0.2)
        p.update({"t1_auto": not bool(s.t1), "t2_auto": not bool(s.t2),
                  "t1": s.t1 or (s.servo_period if s.servo_timing == "profile" else dt),
                  "t2": min(max(s.t2, 0.02), 0.2) if s.t2 else auto_t2,
                  "gain": s.gain, "alpha": s.alpha, "final_t2": (min(max(s.final_t2, 0.02), 0.2) if s.final_t2 else None),
                  "final_hold": int(s.final_hold)})
        return p

    # ------------------------------------------------------------ 사전 점검 / 실기 보호
    def _effective_settings(self):
        """real 모드에서는 속도 상한(real_speed_cap)을 적용한 설정 사본을 쓴다."""
        s = self.s
        cap = s.real_speed_cap
        if s.robot_mode != "real" or not cap:
            return s
        eff = replace(s, speed=min(s.speed, cap), accel=min(s.accel, 2 * cap),
                      home_speed=min(s.home_speed, cap), home_accel=min(s.home_accel, 2 * cap))
        if (eff.speed, eff.accel, eff.home_speed, eff.home_accel) != (s.speed, s.accel, s.home_speed, s.home_accel):
            self.log(f"[실기 보호] 속도 상한 {cap:g} deg/s 적용: speed {s.speed:g}->{eff.speed:g}, accel {s.accel:g}->{eff.accel:g}"
                     f", home_speed {s.home_speed:g}->{eff.home_speed:g}")
        return eff

    def preflight(self):
        """실행 전 점검 정보(현재 모드/상태/자세, 시작 자세까지의 거리 등)를 읽어 반환. 모션 명령은 보내지 않는다."""
        s = self.s
        ip = "127.0.0.1" if s.use_mock else s.ip
        t_full, q_full = trajectory.prepare(s)
        t, q = trajectory.subsample(t_full, np.rad2deg(q_full), s.step)
        eff = self._effective_settings()
        dp = DataPort(ip)
        try:
            d = dp.read()
        finally:
            dp.close()
        cur = d["jang"] if d["sim"] == 0 else d["jref"]
        return {
            "ip": ip, "sim": d["sim"], "robot_state": d["robot_state"], "current": cur,
            "start_pose": q[0], "delta_to_start": float(np.abs(cur - q[0]).max()),
            "path_deg": float(np.abs(np.diff(q, axis=0)).sum(0).max()), "n_points": len(t),
            "speed": eff.speed, "accel": eff.accel, "home_first": s.home_first,
            "home_delta": float(np.abs(cur - np.asarray(s.home_joints)).max()) if s.home_first else None,
        }

    # ------------------------------------------------------------ 원점 복귀 (단독)
    def preflight_home(self):
        """원점 복귀 전 점검 정보 (현재 자세, 원점까지 이동량, 적용 속도). 모션 명령은 보내지 않는다."""
        s = self.s
        ip = "127.0.0.1" if s.use_mock else s.ip
        eff = self._effective_settings()
        dp = DataPort(ip)
        try:
            d = dp.read()
        finally:
            dp.close()
        cur = d["jang"] if d["sim"] == 0 else d["jref"]
        home = np.asarray(s.home_joints, dtype=float)
        return {"ip": ip, "sim": d["sim"], "robot_state": d["robot_state"], "current": cur, "start_pose": home,
                "delta_to_start": float(np.abs(cur - home).max()), "path_deg": 0.0, "n_points": 0,
                "speed": eff.home_speed, "accel": eff.home_accel, "home_first": True,
                "home_delta": float(np.abs(cur - home).max()), "home_only": True}

    def home(self):
        """원점(home_joints)으로 move_j 한 번 보내고 도달할 때까지 기다린다. 기록/분석은 하지 않는다.

        run() 과 같은 보호를 적용한다: real 은 확인(real_ack) 필요, 정지 상태(idle)에서만 시작, 속도 상한, 모드 감시, 종료 후 simulation 복귀.
        """
        class _NoRec:  # wait_reached 가 호출하는 check() 만 필요
            phase = ""

            def check(self):
                pass

        s_req = self.s
        real = s_req.robot_mode == "real"
        if real and not s_req.real_ack:
            raise PermissionError("실기 실행은 확인 절차(real_ack)를 거쳐야 한다")
        self.expect_sim = 0 if real else 1
        s, log = self._effective_settings(), self.log
        self.s = s
        ip = "127.0.0.1" if s.use_mock else s.ip
        log(f"=== 원점 복귀 | 로봇 모드: {'REAL (실제 로봇)' if real else 'simulation'} | 대상 {ip} | 목표 {list(s.home_joints)} ===")
        dp = DataPort(ip)
        cmd = None
        try:
            d0 = dp.read()
            if d0["robot_state"] != 1:
                raise NotSimulation(f"로봇이 정지 상태가 아님 (robot_state={d0['robot_state']}) -> 시작하지 않음")
            cmd = CmdPort(ip, allow_real=real)
            cmd.send("pgmode real" if real else "pgmode simulation")
            for _ in range(20):
                time.sleep(0.25)
                if dp.read()["sim"] == self.expect_sim:
                    break
            dp.require_mode(self.expect_sim)
            gate = {"cmd": cmd, "rec": _NoRec(), "dp": dp, "pending": None}
            t0 = time.time()
            self._send(gate, f"move_j({fmt_jnt(s.home_joints)}, {s.home_speed:g}, {s.home_accel:g})", s.home_joints, "원점 복귀")
            d = dp.read()
            log(f"원점 복귀 완료 ({time.time() - t0:.1f}s), 최대 오차 {np.abs(d['jmeas'] - np.asarray(s.home_joints)).max():.2f} deg")
        except (NotSimulation, StopRequested, Exception) as e:  # noqa: BLE001
            log(f"!! 중단: {e}")
            try:
                if cmd is not None:
                    cmd.send("task stop")  # 비상정지가 아님
            except OSError:
                pass
            raise
        finally:
            if cmd is not None:
                if real and s.revert_to_sim:
                    try:
                        cmd.send("pgmode simulation")
                        time.sleep(0.5)
                        log("simulation 모드로 되돌림")
                    except OSError:
                        log("simulation 복귀 명령 전송 실패 (수동 확인 필요)")
                cmd.close()
            dp.close()

    # ------------------------------------------------------------ 전체 실행
    def run(self, analyze_results=True):
        """실행하고 실행 폴더 경로를 반환. 정상 완료 시 분석(그래프/CSV/JSON)까지 저장한다."""
        from . import analysis  # 순환 import 방지

        s_req = self.s
        real = s_req.robot_mode == "real"
        if s_req.robot_mode not in ("simulation", "real"):
            raise ValueError("robot_mode 는 simulation | real")
        if real and not s_req.real_ack:
            raise PermissionError("실기 실행은 확인 절차(real_ack)를 거쳐야 한다")
        if real and s_req.use_mock is False and not s_req.ip:
            raise ValueError("실기 IP 가 비어 있음")
        self.expect_sim = 0 if real else 1
        s, log = self._effective_settings(), self.log
        self.s = s
        if s.mode not in ("jb2", "servo", "both"):
            raise ValueError("mode 는 jb2 | servo | both")
        ip = "127.0.0.1" if s.use_mock else s.ip
        log(f"=== 로봇 모드: {'REAL (실제 로봇)' if real else 'simulation'} | 대상 {ip} ===")

        t_full, q_full = trajectory.prepare(s)
        q_deg_full = np.rad2deg(q_full)
        t, q = trajectory.subsample(t_full, q_deg_full, s.step)
        dt = round(float(np.median(np.diff(t))), 6)
        self._q_cmd = q
        self._q_ref = q_deg_full
        self.servo_log = None
        if "servo" in (["jb2", "servo"] if s.mode == "both" else [s.mode]):
            tg, _, iv = self._servo_plan(q, t)  # 설정 오류(경로 길이 0, speed/accel 0 등)를 실행 전에 확인
            if s.servo_timing == "profile":
                prof = self._servo_profile(q)
                log(f"[servo] 이상 프로파일: speed {s.speed:g} deg/s, accel {s.accel:g} deg/s² ({s.servo_basis}), 경로 {prof.L:.1f} deg, "
                    f"이상 시간 {prof.T:.2f} s | 전송 {len(tg)}개, 주기 {iv[0] * 1000:.1f} ms | "
                    f"지령 차분 최대: 속도 {prof.cmd_speed_max:.1f}, 가속도 {prof.cmd_accel_max:.1f} (계획 한계 {prof.speed_plan:.1f} / {prof.accel_plan:.1f})")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "REAL_" if real else ""
        self.run_dir = run_dir = os.path.join(s.out_dir, f"{stamp}_{tag}{s.mode}_step{s.step}")
        os.makedirs(run_dir, exist_ok=True)
        log(f"실행 폴더: {run_dir}")
        trajectory.save_interp(os.path.join(run_dir, "traj_ref.csv"), t_full, q_full)
        trajectory.save_interp(os.path.join(run_dir, f"traj_sub{s.step}.csv"), t, np.deg2rad(q))
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"requested": asdict(s_req), "effective": asdict(s)}, f, ensure_ascii=False, indent=1)
        log(f"극값 보존 샘플링: {len(t_full)} -> {len(t)} 포인트, 간격 {np.diff(t).min()*1000:.1f}~{np.diff(t).max()*1000:.1f}ms (중앙값 {dt*1000:.1f}ms)")

        dp = DataPort(ip)
        cmd = None
        rec = None
        completed = False
        modes = ["jb2", "servo"] if s.mode == "both" else [s.mode]
        try:
            d0 = dp.read()
            if d0["robot_state"] != 1:
                raise NotSimulation(f"로봇이 정지 상태가 아님 (robot_state={d0['robot_state']}) -> 시작하지 않음")
            if d0["sim"] != self.expect_sim:
                log(f"현재 모드({'simulation' if d0['sim'] == 1 else 'real'}) -> {'real' if real else 'simulation'} 으로 전환")
            cmd = CmdPort(ip, allow_real=real)
            cmd.send("pgmode real" if real else "pgmode simulation")
            for _ in range(20):
                time.sleep(0.25)
                if dp.read()["sim"] == self.expect_sim:
                    break
            dp.require_mode(self.expect_sim)  # 끝내 전환되지 않으면 여기서 중단
            log(f"{'REAL' if real else '시뮬레이션'} 모드 확인됨")

            rec = Recorder(ip, s.record_hz, expect_sim=self.expect_sim)
            rec.start()
            for m in modes:
                self._chk()
                dp.require_mode(self.expect_sim)
                log(f"[{m}] 시작")
                ok = self._run_jb2(cmd, rec, dp, q) if m == "jb2" else self._run_servo(cmd, rec, dp, q, t)
                log(f"[{m}] 종료 ok={ok}, denied={cmd.denied}")
                time.sleep(1.0)
            completed = True
        except (NotSimulation, StopRequested, Exception) as e:  # noqa: BLE001
            log(f"!! 중단: {e}")
            try:
                if cmd is not None:
                    cmd.send("task stop")  # 비상정지가 아님. 작업 정지 시도일 뿐이다.
            except OSError:
                pass
            raise
        finally:
            if rec is not None:
                rec.stop()
                meta = {
                    "timestamp": stamp, "mode": s.mode, "robot_mode": s.robot_mode, "step": s.step,
                    "n_points": int(len(t)), "dt": dt, "sampling": "stride_with_extrema", "sample_intervals": np.diff(t).tolist(), "traj": os.path.basename(s.traj_path), "record_hz": s.record_hz,
                    "measured_source": "measured only: " + ("jnt_ang (real)" if real else "jnt_ref (simulation, jnt_ang 는 0 고정)"),
                    "methods": {m: self._method_params(m, dt) for m in modes},
                    "home": {"enabled": s.home_first, "joints": s.home_joints, "speed": s.home_speed, "accel": s.home_accel},
                    "marks": rec.marks,
                }
                out = os.path.join(run_dir, f"follow_{s.mode}_{stamp}.csv")
                rec.save(out, meta)
                log(f"기록 {len(rec.rows)} 샘플 -> {out}")
            if self.servo_log:
                with open(os.path.join(run_dir, "servo_schedule.csv"), "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["planned_send_s", "actual_send_s", "t1", "t2"] + [f"q{j}" for j in range(6)])
                    for r in self.servo_log:
                        w.writerow([f"{r[0]:.5f}", f"{r[1]:.5f}", f"{r[2]:.5f}", f"{r[3]:.4f}"] + [f"{v:.4f}" for v in r[4:]])
            if cmd is not None:
                if real and s.revert_to_sim:
                    try:  # 실기 실행이 끝나면 안전한 시뮬레이션 모드로 되돌린다 (best-effort)
                        cmd.send("pgmode simulation")
                        time.sleep(0.5)
                        log("simulation 모드로 되돌림")
                    except OSError:
                        log("simulation 복귀 명령 전송 실패 (수동 확인 필요)")
                cmd.close()
            dp.close()
        if completed and analyze_results:
            self.results = analysis.analyze(run_dir, s, summary_dir=s_req.out_dir, log=log)
        return run_dir
