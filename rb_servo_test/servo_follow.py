"""궤적을 move_jb2_add / move_servo_j 로 추종시키면서 실제 관절각(reqdata)을 기록한다.

** 시뮬레이션 전용 **
- 이 스크립트는 `pgmode real` 을 절대 보내지 않는다.
- 시작 전, 각 단계 전, 명령 전송 중 주기적으로 reqdata 의 real_vs_simulation_mode 를 확인하고
  1(simulation)이 아니면 즉시 중단한다.

사용법: 아래 "설정" 블록의 전역 변수를 고친 뒤 `python rb_servo_test/servo_follow.py` 실행.

JB2 와 servo 는 한 모션 시퀀스에 섞지 말라는 문서 제한 때문에 단계 사이에 정지 상태를 확인한다.
"""
import csv
import json
import os
import socket
import struct
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================ 설정 ============================
IP = "10.0.2.7"        # 제어박스 IP (LCD 표시값으로 수정)
MODE = "servo"             # "jb2" | "servo" | "both"
TRAJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out/traj_interp_50hz.csv")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

HOME_FIRST = False         # True 면 시작점 이동 전에 원점 복귀를 먼저 함 (현재: 현재 자세에서 시작점으로 바로 이동)
HOME_JOINTS = [0, 0, 0, 0, 0, 0]  # 원점 관절각 [deg] (장비의 실제 원점 자세에 맞게 수정)
HOME_SPEED = 20            # 원점 복귀 속도 deg/s
HOME_ACCEL = 40            # 원점 복귀 가속도 deg/s^2

STEP = 10                   # 균등 서브샘플링 간격 (3 = 1/3)

# move_j (시작 자세 이동) + move_jb2_add 공통
SPEED = 200                 # deg/s
ACCEL = 300                 # deg/s^2
BLEND = 50                 # jb2 블렌드율 [%]

# move_servo_j
T1      = 0                     # 도달 시간 [s]   (0 이면 샘플 간격 사용, 최소 0.002)
T2      = 0                     # 유지 시간 [s]   (0 이면 샘플 간격 사용, 0.02~0.2 로 제한)
GAIN    = 0.1                 # 속도 추종 비율 (>0)
ALPHA   = 1                # 저역통과 필터 (0<alpha<1, 작을수록 부드러움)

REACH_TOL = 0.5            # 목표 도달 판정 허용 오차 [deg] (6관절 모두)
REACH_HOLD = 0.3           # 허용 오차 안에 이 시간 동안 머물면 도달로 판정 [s]
SETTLE_TIMEOUT = 15        # servo 마지막 지령 후 '움직임 없이' 수렴 못 하면 포기하는 시간 [s]

RATE = 50                  # 상태 기록 주파수 [Hz]
TIMEOUT = 20               # 대기 중 '움직임이 없는 채로' 이 시간이 지나면 포기 [s] (움직이는 동안은 계속 기다림)
MAX_WAIT = 1800            # 한 번의 대기 절대 상한 [s]
# ==============================================================

CMD_PORT, DATA_PORT = 5000, 5001
PKT = 580
OFF_TIME, OFF_JREF, OFF_JANG, OFF_TCP = 4, 8, 32, 104
OFF_ROBOT_STATE, OFF_SIM = 340, 404  # robot_state: 1=idle, 3=executing / sim: 1=simulation


class NotSimulation(RuntimeError):
    pass


def recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("data port closed")
        buf += chunk
    return buf


class DataPort:
    def __init__(self, ip):
        self.s = socket.create_connection((ip, DATA_PORT), timeout=3)
        self.lock = threading.Lock()

    def read(self):
        with self.lock:
            self.s.sendall(b"reqdata\n")
            buf = recv_exact(self.s, PKT)
        if buf[0] != 0x24 or buf[3] != 0x03:
            raise ValueError("reqdata 헤더 불일치 (엔디안/프레임 확인 필요)")
        f = lambda fmt, off: struct.unpack_from("<" + fmt, buf, off)
        d = {
            "time": f("f", OFF_TIME)[0],
            "jref": np.array(f("6f", OFF_JREF)),
            "jang": np.array(f("6f", OFF_JANG)),
            "robot_state": f("i", OFF_ROBOT_STATE)[0],
            "sim": f("i", OFF_SIM)[0],
        }
        # 시뮬레이션 모드에서는 엔코더(jnt_ang)가 갱신되지 않고 0 으로 고정된다(실기록으로 확인).
        # 시뮬레이션 로봇의 실제 이동 상태는 jnt_ref 이므로, 도달 판정/측정각은 모드에 따라 고른다.
        d["jmeas"] = d["jref"] if d["sim"] == 1 else d["jang"]
        return d

    def require_sim(self):
        d = self.read()
        if d["sim"] != 1:
            raise NotSimulation(f"시뮬레이션 모드가 아님 (real_vs_simulation_mode={d['sim']}) -> 중단")
        return d


class CmdPort:
    """명령 전송 + 응답 드레인 스레드. 응답은 로그/카운트만 한다."""

    def __init__(self, ip):
        self.s = socket.create_connection((ip, CMD_PORT), timeout=3)
        self.s.settimeout(None)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 한 줄 = 한 패킷, 명령 합쳐짐/쪼개짐 방지
        self.replies = []
        self.denied = 0
        self.n_reply = 0
        self._stop = False
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        while not self._stop:
            try:
                b = self.s.recv(4096)
            except OSError:
                return
            if not b:
                return
            txt = b.decode(errors="replace")
            self.replies.append(txt.strip())
            self.denied += txt.count("not allowed")
            self.n_reply += txt.count("The command")

    def send(self, line, ack=False, ack_timeout=1.0):
        """ack=True 면 이 명령의 응답이 올 때까지(최대 ack_timeout) 기다린 뒤 반환 (연속 전송 시 줄 깨짐 방지)."""
        assert "pgmode real" not in line, "real 모드 명령은 보내지 않는다"
        n0 = self.n_reply
        self.s.sendall((line.strip() + "\n").encode())
        if ack:
            t0 = time.time()
            while self.n_reply <= n0 and time.time() - t0 < ack_timeout:
                time.sleep(0.002)
            if self.n_reply <= n0:
                time.sleep(0.02)  # 응답이 없는 명령이면 최소 간격만 확보

    def close(self):
        self._stop = True
        self.s.close()


class Recorder:
    def __init__(self, ip, hz):
        self.dp = DataPort(ip)
        self.period = 1.0 / hz
        self.rows = []
        self.phase = "idle"
        self.marks = {}  # 모드별 기준 시각(첫 모션 명령 직전, 기록 t 기준)
        self.err = None
        self._stop = False
        self.t0 = time.perf_counter()
        self.th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        nxt = time.perf_counter()
        while not self._stop:
            try:
                d = self.dp.read()
                if d["sim"] != 1:
                    raise NotSimulation(f"기록 중 real 모드 감지 (sim={d['sim']})")
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

    def mark(self, name):
        self.marks[name] = time.perf_counter() - self.t0

    def check(self):
        if self.err is not None:
            raise self.err

    def save(self, path, meta):
        with open(path, "w", newline="") as f:
            f.write("# " + json.dumps(meta, ensure_ascii=False) + "\n")
            w = csv.writer(f)
            w.writerow(["t", "phase", "robot_state"] + [f"meas{i}" for i in range(6)])
            for r in self.rows:
                w.writerow([f"{r[0]:.4f}", r[1], r[2]] + [f"{v:.5f}" for v in r[3:]])


def fmt_jnt(q_deg):
    return "jnt[" + ",".join(f"{v:.4f}" for v in q_deg) + "]"


def wait_reached(rec, dp, target_deg, timeout, tol=None, hold=None, label="", need_idle=False):
    """모션이 끝날 때까지 대기하면서 측정(기록 스레드는 계속 동작).

    6관절 측정각(jmeas)이 target 의 tol 이내이고 (need_idle 이면 robot_state==idle 도) hold 초 이상 유지되면
    도달로 판정. 로봇이 움직이는 동안(robot_state != idle 이거나 측정각이 변하는 중)에는 계속 기다리고,
    움직임 없이 timeout 초가 지나거나 MAX_WAIT 를 넘기면 False.
    """
    tol = REACH_TOL if tol is None else tol
    hold = REACH_HOLD if hold is None else hold
    target = np.asarray(target_deg, dtype=float)
    t0 = last_move = time.time()
    in_since = None
    prev = None
    err = float("nan")
    d = {"robot_state": -1}
    while time.time() - t0 < MAX_WAIT:
        rec.check()
        d = dp.read()
        if d["sim"] != 1:
            raise NotSimulation(f"대기 중 real 모드 감지 (sim={d['sim']})")
        now = time.time()
        if d["robot_state"] != 1 or (prev is not None and np.abs(d["jmeas"] - prev).max() > 0.02):
            last_move = now
        prev = d["jmeas"]
        err = np.abs(d["jmeas"] - target).max()
        if err < tol and (not need_idle or d["robot_state"] == 1):
            in_since = in_since or now
            if now - in_since >= hold:
                return True
        else:
            in_since = None
        if now - last_move > timeout:
            break
        time.sleep(0.05)
    print(f"  {label} 도달 못 함 (움직임 없이 {timeout:g}s 경과 또는 MAX_WAIT, 최대 오차 {err:.2f} deg, robot_state={d['robot_state']})")
    return False


class MotionGate:
    """모션 명령은 직전 모션 명령이 끝난 것을 확인한 뒤에만 보낸다.

    끝 판정: 6관절 측정각(jmeas)이 직전 명령의 목표에 도달(REACH_TOL/REACH_HOLD) + (move_j/jb2) robot_state idle.
    """

    def __init__(self, cmd, rec, dp):
        self.cmd, self.rec, self.dp = cmd, rec, dp
        self.pending = None  # (target_deg, label, check_idle)

    def finish(self, timeout):
        """직전 명령이 끝날 때까지 대기(측정은 계속). 끝났으면 True."""
        if self.pending is None:
            return True
        target, label, check_idle = self.pending
        ok = wait_reached(self.rec, self.dp, target, timeout, label=label, need_idle=check_idle)
        if not ok:
            return False
        self.dp.require_sim()
        self.pending = None
        return ok

    def send(self, line, target, label, check_idle=True, wait=True, timeout=None):
        timeout = TIMEOUT if timeout is None else timeout
        """직전 명령 종료 대기 -> 전송 -> (wait=True 면) 이 명령 종료까지 대기."""
        if not self.finish(timeout):
            raise TimeoutError(f"이전 명령 종료 대기 시간 초과 -> '{line[:40]}...' 전송 안 함")
        self.cmd.send(line)
        self.pending = (target, label, check_idle)
        if wait and not self.finish(timeout):
            raise TimeoutError(f"{label} 타임아웃")


def go_home(gate, rec, m):
    """원점 복귀 후 종료 대기. 비교 구간(phase 가 m 으로 시작)에는 포함되지 않도록 home_<m> 으로 기록."""
    if not HOME_FIRST:
        return
    rec.phase = f"home_{m}"
    print(f"[{m}] 원점 복귀 {HOME_JOINTS}")
    gate.send(f"move_j({fmt_jnt(HOME_JOINTS)}, {HOME_SPEED:g}, {HOME_ACCEL:g})", HOME_JOINTS, "원점 복귀")


def run_jb2(cmd, rec, dp, q_deg, args):
    gate = MotionGate(cmd, rec, dp)
    go_home(gate, rec, "jb2")
    rec.phase = "jb2_move_to_start"
    gate.send(f"move_j({fmt_jnt(q_deg[0])}, {args.speed:g}, {args.accel:g})", q_deg[0], "시작 자세 이동")

    rec.phase = "jb2"
    gate.finish(TIMEOUT)  # 리스트 구성 전에도 이전 모션 종료 확인
    cmd.send("move_jb2_clear()", ack=True)
    for q in q_deg[1:]:
        cmd.send(f"move_jb2_add({fmt_jnt(q)}, {args.speed:g}, {args.accel:g}, 0, {args.blend:g})", ack=True)
        if cmd.denied:
            raise RuntimeError("move_jb2_add 가 거부됨 (not allowed)")
    time.sleep(0.5)
    dp.require_sim()
    rec.mark("jb2")
    # 비블로킹으로 실행하고, 종료까지 대기하며 측정
    gate.send("move_jb2_run(0)", q_deg[-1], "jb2 종료", wait=False)
    ok = gate.finish(args.timeout)
    rec.phase = "jb2_end"
    return ok


def run_servo(cmd, rec, dp, q_deg, dt, args):
    gate = MotionGate(cmd, rec, dp)
    go_home(gate, rec, "servo")
    rec.phase = "servo_move_to_start"
    gate.send(f"move_j({fmt_jnt(q_deg[0])}, {args.speed:g}, {args.accel:g})", q_deg[0], "시작 자세 이동")

    rec.phase = "servo"
    t1 = args.t1 if args.t1 else dt
    t2 = min(max(args.t2 if args.t2 else dt, 0.02), 0.2)
    nxt = time.perf_counter()
    rec.mark("servo")
    for i, q in enumerate(q_deg[1:], 1):
        cmd.send(f"move_servo_j({fmt_jnt(q)}, {t1:g}, {t2:g}, {args.gain:g}, {args.alpha:g})")
        if cmd.denied:
            raise RuntimeError("move_servo_j 가 거부됨 (not allowed)")
        if i % 10 == 0:
            rec.check()
        nxt += dt
        time.sleep(max(0.0, nxt - time.perf_counter()))
    # 마지막 지령 이후 최종 목표에 수렴할 때까지 대기하며 계속 기록 (servo 는 robot_state 의미가 불명확해 idle 검사 안 함)
    gate.pending = (q_deg[-1], "servo 종료", False)
    ok = gate.finish(SETTLE_TIMEOUT)
    rec.phase = "servo_end"
    return ok


def method_params(m, args, dt):
    if m == "jb2":
        return {"method": "move_jb2_add", "speed": args.speed, "accel": args.accel, "blend": args.blend}
    return {"method": "move_servo_j", "t1": args.t1 or dt, "t2": min(max(args.t2 or dt, 0.02), 0.2), "gain": args.gain, "alpha": args.alpha}


def main():
    args = SimpleNamespace(ip=IP, mode=MODE, traj=TRAJ, outdir=OUTDIR, step=STEP, speed=SPEED, accel=ACCEL,
                           blend=BLEND, t1=T1, t2=T2, gain=GAIN, alpha=ALPHA, rate=RATE, timeout=TIMEOUT)
    assert args.mode in ("jb2", "servo", "both"), "MODE 는 jb2 | servo | both"
    d = np.genfromtxt(args.traj, delimiter=",", skip_header=1)
    t, q = d[:, 0], np.rad2deg(d[:, 1:])
    idx = np.unique(np.r_[np.arange(0, len(t), args.step), len(t) - 1])
    t, q = t[idx], q[idx]
    dt = round(float(np.median(np.diff(t))), 6)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"{stamp}_{args.mode}_step{args.step}")  # 실행마다 별도 폴더
    os.makedirs(run_dir, exist_ok=True)
    print(f"실행 폴더: {run_dir}")
    sub_path = os.path.join(run_dir, f"traj_sub{args.step}.csv")
    np.savetxt(sub_path, np.c_[t, np.deg2rad(q)], delimiter=",", header=f"step={args.step} src={os.path.basename(args.traj)}\nt,j0,j1,j2,j3,j4,j5", fmt="%.6f")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list))},
                  f, ensure_ascii=False, indent=1)
    print(f"서브샘플링: {len(d)} -> {len(t)} 포인트, 간격 {dt*1000:.0f}ms ({1/dt:.1f}Hz), 저장 {sub_path}")

    dp = DataPort(args.ip)
    d0 = dp.read()
    if d0["sim"] != 1:
        # real 모드면 정지(idle) 상태일 때만 simulation 으로 전환한다. 모션/작업 중이면 건드리지 않고 중단.
        if d0["robot_state"] != 1:
            raise NotSimulation(f"real 모드이고 로봇이 정지 상태가 아님 (robot_state={d0['robot_state']}) -> 중단")
        print("real 모드 감지 -> simulation 으로 전환")
    cmd = CmdPort(args.ip)
    cmd.send("pgmode simulation")
    for _ in range(20):
        time.sleep(0.25)
        if dp.read()["sim"] == 1:
            break
    dp.require_sim()  # 끝내 전환되지 않으면 여기서 중단
    print("시뮬레이션 모드 확인됨")

    rec = Recorder(args.ip, args.rate)
    rec.start()
    modes = ["jb2", "servo"] if args.mode == "both" else [args.mode]
    completed = False
    try:
        for m in modes:
            dp.require_sim()
            print(f"[{m}] 시작")
            ok = run_jb2(cmd, rec, dp, q, args) if m == "jb2" else run_servo(cmd, rec, dp, q, dt, args)
            print(f"[{m}] 종료 ok={ok}, denied={cmd.denied}")
            time.sleep(1.0)
        completed = True
    except (NotSimulation, Exception) as e:  # noqa: BLE001
        print(f"!! 중단: {e}", file=sys.stderr)
        try:
            cmd.send("task stop")
        except OSError:
            pass
        raise
    finally:
        rec.stop()
        out = os.path.join(run_dir, f"follow_{args.mode}_{stamp}.csv")
        meta = {
            "timestamp": stamp,
            "mode": args.mode,
            "step": args.step,
            "n_points": int(len(t)),
            "dt": dt,
            "traj": os.path.basename(args.traj),
            "record_hz": args.rate,
            "measured_source": "measured only: jnt_ref (simulation, jnt_ang 는 0 고정) / jnt_ang (real)",
            "methods": {m: method_params(m, args, dt) for m in modes},
            "home": {"enabled": HOME_FIRST, "joints": HOME_JOINTS, "speed": HOME_SPEED, "accel": HOME_ACCEL},
            "marks": rec.marks,
        }
        rec.save(out, meta)
        print(f"기록 {len(rec.rows)} 샘플 -> {out}")
        if completed:  # 정상 완료한 실행만 비교 그래프(PNG)와 비교 CSV 를 자동 저장
            try:
                from compare_follow import compare
                compare(out, args.traj, run_dir, summary_dir=args.outdir)
            except Exception as e:  # noqa: BLE001
                print(f"그래프 저장 실패 (기록 CSV 는 저장됨): {e}", file=sys.stderr)
        cmd.close()


if __name__ == "__main__":
    main()
