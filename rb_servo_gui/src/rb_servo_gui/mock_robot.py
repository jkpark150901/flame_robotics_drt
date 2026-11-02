"""로봇 없이 시험하기 위한 가짜 제어박스 (127.0.0.1:5000/5001 에서만 listen).

실제 시뮬레이션 기록에서 확인한 동작을 따른다: jnt_ang 는 0 고정, jnt_ref 가 목표로 이동,
이동 중 robot_state=3, 정지 시 1. move_j / move_jb2_add·run / move_servo_j / pgmode simulation·real 만 처리한다.
(real 로 전환하면 jnt_ang 가 jnt_ref 를 따라간다. 이 서버는 127.0.0.1 에서만 열려 실제 로봇과 무관하다.)
"""
import re
import socket
import struct
import threading
import time

from .robot import CMD_PORT, DATA_PORT


class MockRobot:
    def __init__(self, vmax=90.0, host="127.0.0.1"):
        self.vmax = vmax  # deg/s (관절별 최대 이동 속도)
        self.host = host
        self.sim = 1
        self.ref = [0.0] * 6
        self.queue = []          # 목표 점들 (jb2 는 순서대로)
        self.pend = []           # move_jb2_add 로 쌓인 점
        self._last_servo = None  # (시각, 마지막 servo 목표)
        self.log = []
        self._stop = threading.Event()
        self._socks = []
        self._threads = []

    # --- 물리 ---
    @property
    def moving(self):
        return bool(self.queue)

    def _physics(self):
        last = time.time()
        while not self._stop.is_set():
            time.sleep(0.01)
            now = time.time()
            budget = self.vmax * (now - last)
            last = now
            while self.queue and budget > 0:
                tgt = self.queue[0]
                d = [t - r for t, r in zip(tgt, self.ref)]
                m = max(abs(x) for x in d)
                if m <= budget:
                    self.ref = list(tgt)
                    self.queue.pop(0)
                    budget -= m
                else:
                    k = budget / m
                    self.ref = [r + x * k for r, x in zip(self.ref, d)]
                    budget = 0

    # --- 서버 ---
    def _serve(self, port, handler):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, port))
        s.listen()
        s.settimeout(0.3)
        self._socks.append(s)
        while not self._stop.is_set():
            try:
                c, _ = s.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=handler, args=(c,), daemon=True).start()

    def _cmd(self, c):
        buf = b""
        c.settimeout(0.5)
        while not self._stop.is_set():
            try:
                b = c.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if not b:
                return
            buf += b
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._handle(line.decode().strip())
                try:
                    c.sendall(b"The command was executed\n")
                except OSError:
                    return

    def _handle(self, l):
        self.log.append(l)
        m = re.search(r"jnt\[([^\]]*)\]", l)
        vals = [float(x) for x in m.group(1).split(",")] if m else None
        if l == "pgmode simulation":
            self.sim = 1
        elif l == "pgmode real":
            self.sim = 0
        elif l.startswith("move_jb2_clear"):
            self.pend = []
        elif l.startswith("move_jb2_add") and vals:
            self.pend.append(vals)
        elif l.startswith("move_jb2_run"):
            self.queue = [list(p) for p in self.pend]
        elif l.startswith("move_j(") and vals:
            self._last_servo = None
            self.queue = [vals]
        elif l.startswith("move_servo_j") and vals:
            # 실측에서 확인한 거동 모사: 목표에 도달한 뒤 마지막 지령 속도 × t2 만큼 더 나아가 멈춘다 (다음 지령이 오면 덮어씀)
            now = time.time()
            t2 = float(l.split("]", 1)[1].split(",")[2]) if "]," in l else 0.0
            if self._last_servo is not None and now - self._last_servo[0] < 1.0:
                v = [(a - b) / max(now - self._last_servo[0], 1e-3) for a, b in zip(vals, self._last_servo[1])]
            else:
                v = [0.0] * 6
            self._last_servo = (now, list(vals))
            self.queue = [vals, [a + w * t2 for a, w in zip(vals, v)]]
        elif l.startswith("task stop"):
            self.queue = []

    def _data(self, c):
        c.settimeout(0.5)
        while not self._stop.is_set():
            try:
                b = c.recv(64)
            except socket.timeout:
                continue
            except OSError:
                return
            if not b:
                return
            p = bytearray(580)
            p[0], p[1], p[2], p[3] = 0x24, 580 & 255, 580 >> 8, 3
            struct.pack_into("<6f", p, 8, *self.ref)
            struct.pack_into("<6f", p, 32, *(self.ref if self.sim == 0 else [0.0] * 6))  # real: 엔코더가 따라감 / sim: 0 고정
            struct.pack_into("<i", p, 340, 3 if self.moving else 1)
            struct.pack_into("<i", p, 404, self.sim)
            try:
                c.sendall(bytes(p))
            except OSError:
                return

    def start(self):
        for target in (self._physics, lambda: self._serve(CMD_PORT, self._cmd), lambda: self._serve(DATA_PORT, self._data)):
            th = threading.Thread(target=target, daemon=True)
            th.start()
            self._threads.append(th)
        time.sleep(0.3)
        return self

    def stop(self):
        """서버를 멈추고 포트가 완전히 풀릴 때까지 기다린다 (바로 다시 start() 해도 'Address already in use' 가 나지 않도록)."""
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2.0)
        for s in self._socks:
            try:
                s.close()
            except OSError:
                pass
        self._socks.clear()
        self._threads.clear()
