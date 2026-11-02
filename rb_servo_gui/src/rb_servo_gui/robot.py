"""제어박스 TCP 통신: 명령 포트(5000), 상태 포트(5001, reqdata). 시뮬레이션 전용 안전장치를 포함한다."""
import socket
import struct
import threading
import time

import numpy as np

CMD_PORT, DATA_PORT = 5000, 5001
PKT = 580
OFF_TIME, OFF_JREF, OFF_JANG = 4, 8, 32
OFF_ROBOT_STATE, OFF_SIM = 340, 404  # robot_state: 1=idle, 3=executing / sim: 1=simulation


class NotSimulation(RuntimeError):
    """real 모드가 감지되었을 때."""


class StopRequested(RuntimeError):
    """사용자가 중지를 눌렀을 때."""


def recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("data port closed")
        buf += chunk
    return buf


class DataPort:
    def __init__(self, ip, port=DATA_PORT):
        self.s = socket.create_connection((ip, port), timeout=3)
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
        # 시뮬레이션에서는 엔코더(jnt_ang)가 갱신되지 않고 0 으로 고정된다(실기록으로 확인).
        # 시뮬레이션 로봇의 이동 상태는 jnt_ref 이므로 측정각은 모드에 따라 고른다.
        d["jmeas"] = d["jref"] if d["sim"] == 1 else d["jang"]
        return d

    def require_mode(self, sim_expected):
        """현재 모드가 기대한 모드(1=simulation, 0=real)와 같은지 확인. 다르면 NotSimulation 으로 중단."""
        d = self.read()
        if d["sim"] != sim_expected:
            want = "simulation" if sim_expected == 1 else "real"
            raise NotSimulation(f"예상 모드({want})와 다름 (real_vs_simulation_mode={d['sim']}) -> 중단")
        return d

    def require_sim(self):
        return self.require_mode(1)

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


class CmdPort:
    """명령 전송 + 응답 드레인 스레드."""

    def __init__(self, ip, port=CMD_PORT, allow_real=False):
        self.allow_real = allow_real  # True 일 때만 `pgmode real` 전송 가능 (GUI 확인 절차를 거친 실기 실행)
        self.s = socket.create_connection((ip, port), timeout=3)
        self.s.settimeout(None)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 한 줄 = 한 패킷
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
        """ack=True 면 응답이 올 때까지(최대 ack_timeout) 기다린 뒤 반환 (연속 전송 시 줄 깨짐 방지)."""
        if "pgmode real" in line and not self.allow_real:
            raise PermissionError("real 모드 명령은 확인 절차를 거친 실기 실행에서만 보낼 수 있다")
        n0 = self.n_reply
        self.s.sendall((line.strip() + "\n").encode())
        if ack:
            t0 = time.time()
            while self.n_reply <= n0 and time.time() - t0 < ack_timeout:
                time.sleep(0.002)
            if self.n_reply <= n0:
                time.sleep(0.02)

    def close(self):
        self._stop = True
        try:
            self.s.close()
        except OSError:
            pass
