import json
import time
from collections import deque
from util.logger.console import ConsoleLogger
from common.zpipe import AsyncZSocket, ZPipe
from common.zapi import ZAPIBase

try:
    from PyQt6.QtCore import QObject, pyqtSignal
except ImportError:
    print("PyQt6 is required to run this application.")

class ZAPI(QObject, ZAPIBase):
    """ZMQ communication manager for SimTool.
    Uses DEALER socket to communicate with Viewer (Router).
    """
    
    # Signal for message reception: function_name, json_kwargs
    signal_message_received = pyqtSignal(str, str)

    def __init__(self, zpipe: ZPipe = None, transport: str = "ipc", channel: str = "/tmp/viewervedo"):
        QObject.__init__(self)
        ZAPIBase.__init__(self)
        
        self.__console = ConsoleLogger.get_logger()
        self._zpipe = zpipe
        self._running = False
        self._transport = transport
        self._channel = channel
        
        # Dealer Socket
        self.__dealer_socket = AsyncZSocket("ZAPI_SIMTOOL", "dealer")
        if not self.__dealer_socket.create(pipeline=zpipe):
            self.__console.error("Failed to create Simtool Socket(Dealer)")

    def run(self):
        """Start the ZAPI communication."""
        if self._running:
            self.__console.warning("[ZAPI_SIMTOOL] Already running")
            return
        self._running = True
        self.__console.debug("Starting Simtool ZAPI...")
        
        # Connect immediately (ZeroMQ handles reconnection)
        self.__dealer_socket.set_message_callback(self._on_message_received)
        if self.__dealer_socket.join(self._transport, self._channel):
            self.__console.info(f"[ZAPI SIMTOOL] Connected to {self._transport}://{self._channel}")
        else:
            self.__console.error(f"[ZAPI SIMTOOL] Failed to connect to {self._transport}://{self._channel}")

    def stop(self):
        """Stop and cleanup."""
        self._running = False
        
        if self.__dealer_socket:
            self.__dealer_socket.destroy_socket()
        
        self.__console.debug("[ZAPI] Stopped")

    def _on_message_received(self, multipart_data):
        """Callback from sockets."""
        try:
            if len(multipart_data) < 3:
                return

            socket_name = multipart_data[0]
            function_name = multipart_data[1]
            json_kwargs = multipart_data[2]
            
            # Decode components
            socket_str = socket_name.decode('utf-8') if isinstance(socket_name, bytes) else socket_name
            function_str = function_name.decode('utf-8') if isinstance(function_name, bytes) else function_name
            kwargs_str = json_kwargs.decode('utf-8') if isinstance(json_kwargs, bytes) else json_kwargs
            
            # Emit Signal to Main Thread
            self.signal_message_received.emit(function_str, kwargs_str)

        except Exception as e:
            self.__console.error(f"[ZAPI] Error receiving message: {e}")

    # ----------------------------------------------------------------
    # Supporting ZAPIs
    # ----------------------------------------------------------------
    def _ZAPI_request_load_spool(self, file_path: str):
        """Sends command to load spool."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {
                "path": file_path
            }
            # Use ZAPIBase.call
            self.call(self.__dealer_socket, "zapi_load_spool", kwargs)
            self.__console.info(f"[ZAPI] Sent load_spool request: {file_path}")
        else:
            self.__console.warning("[ZAPI] Cannot send load_spool: Socket not connected")

    def _ZAPI_request_flip_spool_x(self):
        """Sends command to flip the loaded spool in X direction."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_flip_spool_x", {})
            self.__console.info("[ZAPI] Sent flip_spool_x request")
        else:
            self.__console.warning("[ZAPI] Cannot send flip_spool_x: Socket not connected")

    def _ZAPI_request_move_spool(self, x: float, y: float, z: float,
                                 x_rotation: float = 0.0, z_rotation: float = 0.0):
        """Sends command to move the loaded spool to an absolute position and rotation."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {
                "x": x,
                "y": y,
                "z": z,
                "x_rotation": x_rotation,
                "z_rotation": z_rotation,
            }
            self.call(self.__dealer_socket, "zapi_move_spool", kwargs)
            self.__console.info(
                f"[ZAPI] Sent move_spool request: x={x} y={y} z={z} "
                f"x_rotation={x_rotation} z_rotation={z_rotation}")
        else:
            self.__console.warning("[ZAPI] Cannot send move_spool: Socket not connected")

    def _ZAPI_request_load_test_weld_point(self, file_path: str):
        """Sends command to load test weld points."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {
                "path": file_path
            }
            # Use ZAPIBase.call
            self.call(self.__dealer_socket, "zapi_load_test_weld_point", kwargs)
            self.__console.info(f"[ZAPI] Sent load_test_weld_point request: {file_path}")
        else:
            self.__console.warning("[ZAPI] Cannot send load_test_weld_point: Socket not connected")

    def _ZAPI_request_move_positioner(self, axis: str, position: float, velocity: float = 0.0,
                                       fix_f_column_r: bool = False, fix_m_column_z: bool = False):
        """Sends positioner joint move command. axis: 'x'|'z'|'r'"""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {
                "axis": axis,
                "position": position,
                "velocity": velocity,
                "fix_f_column_r": fix_f_column_r,
                "fix_m_column_z": fix_m_column_z,
            }
            self.call(self.__dealer_socket, "zapi_move_positioner", kwargs)
            self.__console.info(f"[ZAPI] Sent move_positioner: axis={axis} pos={position} vel={velocity} "
                                f"fix_f={fix_f_column_r} fix_z={fix_m_column_z}")
        else:
            self.__console.warning("[ZAPI] Cannot send move_positioner: Socket not connected")

    def _ZAPI_request_fix_spool(self):
        """스풀을 chuck에 강체 부착(고정). 뷰어가 offset(R,t)을 계산/저장."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_fix_spool", {})
            self.__console.info("[ZAPI] Sent fix_spool request")
        else:
            self.__console.warning("[ZAPI] Cannot send fix_spool: Socket not connected")

    def _ZAPI_request_unfix_spool(self):
        """스풀 고정 해제."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_unfix_spool", {})
            self.__console.info("[ZAPI] Sent unfix_spool request")
        else:
            self.__console.warning("[ZAPI] Cannot send unfix_spool: Socket not connected")

    def _ZAPI_request_set_spool_offset(self, offset_R, offset_t):
        """로드 복원: 저장된 offset(R,t)으로 강체 부착 재설정."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_set_spool_offset",
                      {"offset_R": offset_R, "offset_t": offset_t})
            self.__console.info("[ZAPI] Sent set_spool_offset request")
        else:
            self.__console.warning("[ZAPI] Cannot send set_spool_offset: Socket not connected")

    def _ZAPI_request_filter_spool(self, method: str, params: dict = None):
        """현재 로드된 스풀에 직접 노이즈 필터를 적용. method: 'sor'|'ccl'"""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {"method": method, "params": params or {}}
            self.call(self.__dealer_socket, "zapi_filter_spool", kwargs)
            self.__console.info(f"[ZAPI] Sent filter_spool request: method={method} params={params}")
        else:
            self.__console.warning("[ZAPI] Cannot send filter_spool: Socket not connected")

    def _ZAPI_request_reconstruct_mesh(self, params: dict = None):
        """현재 로드된 스풀로 메시 재건 (Marching Cubes)."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {"params": params or {}}
            self.call(self.__dealer_socket, "zapi_reconstruct_mesh", kwargs)
            self.__console.info(f"[ZAPI] Sent reconstruct_mesh request: params={params}")
        else:
            self.__console.warning("[ZAPI] Cannot send reconstruct_mesh: Socket not connected")

    def _ZAPI_request_save_spool(self, file_path: str):
        """현재 로드된 스풀(또는 재건 메시)을 파일로 저장."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_save_spool", {"path": file_path})
            self.__console.info(f"[ZAPI] Sent save_spool request: {file_path}")
        else:
            self.__console.warning("[ZAPI] Cannot send save_spool: Socket not connected")

    def _ZAPI_request_set_mode(self, mode: str):
        """Sends command to set execution mode (simulation or real)."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {
                "mode": mode
            }
            self.call(self.__dealer_socket, "zapi_set_mode", kwargs)
            self.__console.info(f"[ZAPI] Sent set_mode request: {mode}")
        else:
            self.__console.warning("[ZAPI] Cannot send set_mode: Socket not connected")
