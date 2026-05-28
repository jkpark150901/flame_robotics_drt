'''
NatNetWorker — NatNet 수신 스레드
'''

import ipaddress
import logging
import socket
import sys
import threading
import time
import pathlib

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    raise ImportError("PyQt6 is required.")

# 저장소 루트를 경로에 추가 (tools/NatNet 접근)
_ROOT = pathlib.Path(__file__).parents[4]  # python/verifytool/workers/ → repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.NatNet.NatNetClient import NatNetClient

log = logging.getLogger(__name__)


class NatNetWorker(QThread):
    '''
    NatNetClient 를 별도 QThread에서 실행.

    Signals:
        connected()                  - 연결 성공
        disconnected()               - 연결 해제
        error(str)                   - 에러 메시지
        rb_updated(int, list, list)  - (rb_id, position[x,y,z] m, rotation[qx,qy,qz,qw])
        fps_updated(float)           - 프레임 수신 속도 (Hz)
    '''

    connected    = pyqtSignal()
    disconnected = pyqtSignal()
    error        = pyqtSignal(str)
    rb_updated   = pyqtSignal(int, list, list)   # rb_id, pos[3], quat[4] xyzw
    fps_updated  = pyqtSignal(float)

    def __init__(self, server_ip: str, client_ip: str = 'auto',
                 rigid_body_id: int = 1, force_version: tuple | None = None,
                 parent=None):
        super().__init__(parent)
        self._server_ip = server_ip
        self._client_ip = client_ip
        self._rb_id = rigid_body_id
        self._force_version = force_version  # (major, minor) or None
        self._stop_event = threading.Event()

        self._frame_times: list[float] = []
        self._fps = 0.0
        self._last_emit_time: float = 0.0
        self._EMIT_INTERVAL = 1.0 / 30.0   # UI 시그널 최대 30 Hz

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()

    # ──────────────────────────────────────────────
    # QThread entry point
    # ──────────────────────────────────────────────

    def run(self):
        self._stop_event.clear()

        try:
            client_ip = self._resolve_client_ip(self._server_ip, self._client_ip)
            self._validate_ip_pair(self._server_ip, client_ip)
        except Exception as e:
            log.error("NatNet IP error: %s", e)
            self.error.emit(f"NatNet IP error: {e}")
            self.disconnected.emit()
            return

        client = NatNetClient()
        client.set_server_address(self._server_ip)
        client.set_client_address(client_ip)
        client.set_use_multicast(False)
        client.rigid_body_listener = self._on_rigid_body

        if self._force_version:
            major, minor = self._force_version
            client._NatNetClient__nat_net_requested_version[0] = major
            client._NatNetClient__nat_net_requested_version[1] = minor

        log.info("NatNet: connecting to %s (client %s) …", self._server_ip, client_ip)
        try:
            started = client.run('d')
        except Exception as e:
            msg = f"NatNet client.run() raised: {e}"
            log.error(msg)
            self.error.emit(msg)
            self.disconnected.emit()
            return
        if not started:
            msg = f"NatNet client.run() failed (server={self._server_ip})"
            log.error(msg)
            self.error.emit(msg)
            self.disconnected.emit()
            return

        log.info("NatNet: connected.")
        self.connected.emit()

        # FPS 계산 타이머
        fps_last = time.time()
        fps_count = 0

        try:
            while not self._stop_event.is_set():
                time.sleep(0.1)
                # FPS 보고 (1 초 주기)
                now = time.time()
                if now - fps_last >= 1.0:
                    with threading.Lock():
                        t = self._frame_times
                        recent = [ft for ft in t if now - ft <= 1.0]
                        self._frame_times = recent
                        fps = len(recent)
                    self._fps = float(fps)
                    self.fps_updated.emit(self._fps)
                    fps_last = now
        finally:
            try:
                client.shutdown()
            except Exception:
                pass
            log.info("NatNet: disconnected.")
            self.disconnected.emit()

    # ──────────────────────────────────────────────
    # NatNet callbacks (called from NatNet threads)
    # ──────────────────────────────────────────────

    def _on_rigid_body(self, rb_id, position, rotation):
        if int(rb_id) != self._rb_id:
            return
        now = time.time()
        self._frame_times.append(now)   # FPS 집계는 모든 프레임 카운트

        # UI 시그널은 30Hz로 제한 — 메인 스레드 이벤트 큐 포화 방지
        if now - self._last_emit_time < self._EMIT_INTERVAL:
            return
        self._last_emit_time = now

        pos  = [float(v) for v in position]   # [x, y, z] m
        quat = [float(v) for v in rotation]   # [qx, qy, qz, qw]
        self.rb_updated.emit(int(rb_id), pos, quat)

    # ──────────────────────────────────────────────
    # IP helpers (copied from check_mocap_unicast)
    # ──────────────────────────────────────────────

    @staticmethod
    def _resolve_client_ip(server_ip: str, client_ip: str) -> str:
        if client_ip and client_ip.lower() != 'auto':
            return client_ip
        server_addr = socket.gethostbyname(server_ip)
        if ipaddress.ip_address(server_addr).is_loopback:
            return '127.0.0.1'
        # UDP connect lets the OS pick the outbound interface without sending packets
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((server_addr, 1510))
            return sock.getsockname()[0]
        finally:
            sock.close()

    @staticmethod
    def _validate_ip_pair(server_ip: str, client_ip: str):
        server_addr = ipaddress.ip_address(socket.gethostbyname(server_ip))
        client_addr = ipaddress.ip_address(socket.gethostbyname(client_ip))
        if server_addr.is_loopback != client_addr.is_loopback:
            raise ValueError(
                f"server={server_ip}, client={client_ip} 조합이 맞지 않습니다. "
                "원격 Motive 서버를 쓸 때 client는 이 PC의 같은 네트워크 대역 IP여야 합니다."
            )
