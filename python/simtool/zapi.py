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

    def __init__(
        self,
        zpipe: ZPipe = None,
        transport: str = "ipc",
        channel: str = "/tmp/viewervedo",
        socket_id: str = "ZAPI_SIMTOOL"
    ):
        QObject.__init__(self)
        ZAPIBase.__init__(self)
        
        self.__console = ConsoleLogger.get_logger()
        self._zpipe = zpipe
        self._running = False
        self._transport = transport
        self._channel = channel
        self._socket_id = socket_id
        
        # Dealer Socket
        self.__dealer_socket = AsyncZSocket(socket_id, "dealer")
        if not self.__dealer_socket.create(pipeline=zpipe):
            self.__console.error(f"Failed to create Simtool Socket(Dealer): {socket_id}")

    def run(self):
        """Start the ZAPI communication."""
        if self._running:
            self.__console.warning("[ZAPI_SIMTOOL] Already running")
            return
        self._running = True
        self.__console.debug(f"Starting Simtool ZAPI: {self._socket_id}")
        
        # Connect immediately (ZeroMQ handles reconnection)
        self.__dealer_socket.set_message_callback(self._on_message_received)
        if self.__dealer_socket.join(self._transport, self._channel):
            self.__console.info(f"[{self._socket_id}] Connected to {self._transport}://{self._channel}")
        else:
            self.__console.error(f"[{self._socket_id}] Failed to connect to {self._transport}://{self._channel}")

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

    def _ZAPI_request_mujoco_load_model(self, model_path: str):
        """Sends command to load an MJCF model in the MuJoCo viewer."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_load_model", {"path": model_path})
            self.__console.info(f"[ZAPI] Sent mujoco load_model request: {model_path}")
        else:
            self.__console.warning("[ZAPI] Cannot send load_model: Socket not connected")

    def _ZAPI_request_mujoco_load_models(self, model_paths: list):
        """Sends command to load multiple MJCF models as one MuJoCo workspace."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_load_models", {"paths": model_paths})
            self.__console.info(f"[ZAPI] Sent mujoco load_models request: {model_paths}")
        else:
            self.__console.warning("[ZAPI] Cannot send load_models: Socket not connected")

    def _ZAPI_request_mujoco_load_urdf_workspace(self, urdf_entries: list):
        """Sends command to load URDF visual models as one MuJoCo workspace."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_load_urdf_workspace", {"urdf": urdf_entries})
            self.__console.info(f"[ZAPI] Sent mujoco load_urdf_workspace request: {urdf_entries}")
        else:
            self.__console.warning("[ZAPI] Cannot send load_urdf_workspace: Socket not connected")

    def _ZAPI_request_mujoco_reset(self):
        """Sends command to reset the MuJoCo simulation."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_reset", {})
            self.__console.info("[ZAPI] Sent mujoco reset request")
        else:
            self.__console.warning("[ZAPI] Cannot send reset: Socket not connected")

    def _ZAPI_request_mujoco_set_joint_positions(self, positions: dict):
        """Sends direct joint-position values to the MuJoCo simulation."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_set_joint_positions", {"positions": positions})
            self.__console.info(f"[ZAPI] Sent mujoco joint positions: {positions}")
        else:
            self.__console.warning("[ZAPI] Cannot send joint positions: Socket not connected")

    def _ZAPI_request_mujoco_set_joint_targets(self, targets: dict):
        """Sends actuator targets to the MuJoCo simulation when actuators exist."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_set_joint_targets", {"targets": targets})
            self.__console.info(f"[ZAPI] Sent mujoco joint targets: {targets}")
        else:
            self.__console.warning("[ZAPI] Cannot send joint targets: Socket not connected")
