import json
import threading
from collections import deque

from common.zapi import ZAPIBase
from common.zpipe import AsyncZSocket, ZPipe
from util.logger.console import ConsoleLogger


class ZAPI(ZAPIBase):
    """ROUTER-side ZAPI for the MuJoCo backend."""

    def __init__(self, config: dict = None, zpipe: ZPipe = None, simulator=None):
        super().__init__()
        self._config = config or {}
        self.__console = ConsoleLogger.get_logger()
        self._simulator = simulator
        self._running = False
        self._current_mode = self._config.get("operation_mode", "simulation")
        self.request_queue = deque(maxlen=100)
        self._queue_lock = threading.Lock()

        self.__router_socket = AsyncZSocket("ZAPI_VIEWERMUJOCO", "router")
        if not self.__router_socket.create(pipeline=zpipe):
            self.__console.error("[ZAPI_VIEWERMUJOCO] Failed to create router socket")

    def run(self):
        if self._running:
            self.__console.warning("[ZAPI_VIEWERMUJOCO] Already running")
            return

        self._running = True
        transport = self._config.get("transport", "ipc")
        channel = self._config.get("channel", "/tmp/viewermujoco")

        self.__router_socket.set_message_callback(self._on_message_received)
        if self.__router_socket.join(transport, channel):
            self.__console.info(f"[ZAPI_VIEWERMUJOCO] Router bound: {transport}://{channel}")
        else:
            self.__console.error("[ZAPI_VIEWERMUJOCO] Failed to bind router socket")

    def stop(self):
        self._running = False
        if self.__router_socket:
            self.__router_socket.destroy_socket()
        self.__console.debug("[ZAPI_VIEWERMUJOCO] Stopped")

    def _on_message_received(self, multipart_data):
        try:
            if len(multipart_data) == 2 and len(multipart_data[1]) == 0:
                identity = multipart_data[0]
                self.__console.info(f"[ZAPI_VIEWERMUJOCO] Client connected: {identity}")
                self._send_state_info(identity)
                return

            if len(multipart_data) < 4:
                self.__console.warning(f"[ZAPI_VIEWERMUJOCO] Incomplete message: {len(multipart_data)} parts")
                return

            identity = multipart_data[0]
            function_name = multipart_data[2]
            json_kwargs = multipart_data[3]
            function_name = function_name.decode("utf-8") if isinstance(function_name, bytes) else function_name

            self._dispatch_message(identity, function_name, json_kwargs)
        except Exception as exc:
            self.__console.error(f"[ZAPI_VIEWERMUJOCO] Error processing message: {exc}")

    def _send_state_info(self, identity):
        state_info = {
            "mode": self._current_mode,
            "model": self._config.get("model", ""),
            "models": self._config.get("models", []),
            "urdf": self._config.get("urdf", [])
        }
        self._reply(identity, "update_state_info", state_info)

    def _reply(self, identity, function: str, kwargs: dict):
        if self.__router_socket and self.__router_socket.is_joined and identity:
            parts = [
                identity,
                self.__router_socket.socket_id.encode("utf-8"),
                function.encode("utf-8"),
                json.dumps(kwargs).encode("utf-8")
            ]
            self.__router_socket.dispatch(parts)

    def _dispatch_message(self, identity, function_name: str, json_kwargs):
        try:
            kwargs_str = json_kwargs.decode("utf-8") if isinstance(json_kwargs, bytes) else json_kwargs
            kwargs = json.loads(kwargs_str)
            if not isinstance(kwargs, dict):
                kwargs = {"payload": kwargs}
            kwargs["_identity"] = identity

            handler_name = function_name if function_name.startswith("zapi_") else f"zapi_{function_name}"
            handler = getattr(self, handler_name, None)
            if handler and callable(handler):
                handler(kwargs)
            else:
                self.__console.warning(f"[ZAPI_VIEWERMUJOCO] Unknown function: {function_name}")
        except Exception as exc:
            self.__console.error(f"[ZAPI_VIEWERMUJOCO] Dispatch error: {exc}")

    def zapi_terminate(self, kwargs=None):
        self._push({"command": "terminate"})

    def zapi_load_model(self, kwargs=None):
        if kwargs and "path" in kwargs:
            self._push({"command": "load_model", "path": kwargs["path"]})
        else:
            self.__console.warning("[ZAPI_VIEWERMUJOCO] load_model request without path")

    def zapi_load_models(self, kwargs=None):
        if kwargs and "paths" in kwargs:
            self._push({"command": "load_models", "paths": kwargs["paths"]})
        else:
            self.__console.warning("[ZAPI_VIEWERMUJOCO] load_models request without paths")

    def zapi_load_urdf_workspace(self, kwargs=None):
        if kwargs and "urdf" in kwargs:
            self._push({"command": "load_urdf_workspace", "urdf": kwargs["urdf"]})
        else:
            self.__console.warning("[ZAPI_VIEWERMUJOCO] load_urdf_workspace request without urdf")

    def zapi_set_mode(self, kwargs=None):
        if kwargs and "mode" in kwargs:
            self._current_mode = kwargs["mode"]
            self._push({"command": "set_mode", "mode": kwargs["mode"]})

    def zapi_reset(self, kwargs=None):
        self._push({"command": "reset"})

    def zapi_set_joint_positions(self, kwargs=None):
        self._push({
            "command": "set_joint_positions",
            "positions": (kwargs or {}).get("positions", {})
        })

    def zapi_set_joint_targets(self, kwargs=None):
        self._push({
            "command": "set_joint_targets",
            "targets": (kwargs or {}).get("targets", {})
        })

    def _push(self, data):
        if self._simulator:
            self._simulator.push_request(data)
        else:
            with self._queue_lock:
                self.request_queue.append(data)
