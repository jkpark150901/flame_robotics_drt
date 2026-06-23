"""
Zapi - ZMQ Communication Module for Visualizer
@note
- Manages all ZMQ socket communication in a separate thread
- Uses ROUTER pattern for async request/reply
- Dispatches incoming messages to internal zapi_* handler methods
"""

import threading
import json
from collections import deque
from typing import Optional
from util.logger.console import ConsoleLogger
from common.zpipe import AsyncZSocket, ZPipe
from common.zapi import ZAPIBase


class ZAPI(ZAPIBase):
    """ZMQ communication manager for Visualizer.
    
    Runs a receiver thread that listens for incoming messages and
    dispatches them to zapi_* handler methods. Provides a request_queue
    for the visualizer to consume rendering commands from.
    """

    def __init__(self, config: dict = None, zpipe: ZPipe = None, visualizer=None):
        super().__init__()
        if config is None:
            config = {}

        self._config = config
        self.__console = ConsoleLogger.get_logger()
        self._visualizer = visualizer
        self._zpipe = zpipe
        self._running = False
        self._thread = None

        # Thread-safe queue: zapi handlers push commands here,
        # visualizer polls from it each frame
        self.request_queue = deque(maxlen=100)
        self._queue_lock = threading.Lock()
        
        self._current_mode = "simulation"

        # Router Socket (receive data + system commands) ---
        self.__router_socket = AsyncZSocket("ZAPI_VIEWERVEDO", "router")
        if not self.__router_socket.create(pipeline=zpipe):
            self.__console.error("[ZAPI_VIEWERVEDO] Failed to create router socket")

    def run(self):
        """Start the ZApi communication thread."""
        if self._running:
            self.__console.warning("[ZAPI_VIEWERVEDO] Already running")
            return

        self._running = True
        self.__console.debug("Starting Viewervedo ZAPI...")

        transport = self._config.get("transport", "ipc")
        channel = self._config.get("channel", "/tmp/viewervedo")

        self.__router_socket.set_message_callback(self._on_message_received)
        if self.__router_socket.join(transport, channel):
            self.__console.debug(f"[ZAPI_VIEWERVEDO] Router bound: {transport}://{channel}")
        else:
            self.__console.error("[ZAPI_VIEWERVEDO] Failed to bind router socket")

    def stop(self):
        """Stop the ZApi communication thread and cleanup sockets."""
        self._running = False

        # Destroy sockets
        if self.__router_socket:
            self.__router_socket.destroy_socket()

        self.__console.debug("[ZAPI_VIEWERVEDO] Stopped and cleaned up")

    # ----------------------------------------------------------------
    # Message reception (called from AsyncZSocket's receiver thread)
    # ----------------------------------------------------------------
    def _on_message_received(self, multipart_data):
        """Callback from AsyncZSocket receiver thread.
        ROUTER socket receives: [identity, socket_name, function, json_kwargs]
        Why?
        Sender (DEALER) calls ZAPIBase.call -> sends [socket_name, function, json_kwargs]
        ROUTER receives -> [identity, socket_name, function, json_kwargs]
        """
        self.__console.info("-------------")
        try:
            # Check for PROBE_ROUTER connection event (identity + empty message)
            if len(multipart_data) == 2 and len(multipart_data[1]) == 0:
                identity = multipart_data[0]
                self.__console.info(f"[ZAPI_VIEWERVEDO] Client connected: {identity}")
                self._send_state_info(identity)
                return

            if len(multipart_data) < 4:
                # Minimum expected for ROUTER receiving from ZAPIBase.call:
                # [identity, socket_name, function, json_kwargs] = 4 parts
                self.__console.warning(f"[ZAPI_VIEWERVEDO] Received incomplete message: {len(multipart_data)} parts")
                return

            identity = multipart_data[0]
            socket_name = multipart_data[1]
            function_name = multipart_data[2]
            json_kwargs = multipart_data[3]

            # Decode
            socket_name_str = socket_name.decode('utf-8') if isinstance(socket_name, bytes) else socket_name
            function_name_str = function_name.decode('utf-8') if isinstance(function_name, bytes) else function_name
            
            # Dispatch
            self._dispatch_message(identity, function_name_str, json_kwargs)

        except Exception as e:
            self.__console.error(f"[ZAPI_VIEWERVEDO] Error processing message: {e}")

    def _send_state_info(self, identity):
        """Send current state information to the newly connected client."""
        state_info = {
            "display_options": self._config.get("display_options", {}),
            "mode": getattr(self, "_current_mode", "simulation")
        }
        if self.__router_socket and self.__router_socket.is_joined:
            socket_name = self.__router_socket.socket_id
            function = "update_state_info"
            
            reply_parts = [
                identity,
                socket_name.encode('utf-8'),
                function.encode('utf-8'),
                json.dumps(state_info).encode('utf-8')
            ]
            self.__router_socket.dispatch(reply_parts)
            self.__console.info(f"[ZAPI_VIEWERVEDO] Sent state_info to client {identity}")

    def _dispatch_message(self, identity, function_name, json_kwargs):
        """Route incoming messages to the appropriate zapi_* handler.
        
        Args:
            identity: Sender identity (bytes)
            function_name: Function name (str)
            json_kwargs: JSON encoded kwargs (bytes/str)
        """
        try:
            # Parse kwargs
            try:
                kwargs_str = json_kwargs.decode('utf-8') if isinstance(json_kwargs, bytes) else json_kwargs
                kwargs = json.loads(kwargs_str)
                
                if not isinstance(kwargs, dict):
                    kwargs = {"payload": kwargs} # Fallback if not dict

                # Inject identity into kwargs so we know who to reply to
                kwargs["_identity"] = identity

                # Look up zapi_<function> handler
                if function_name.startswith("zapi_"):
                    handler_name = function_name
                else:
                    handler_name = f"zapi_{function_name}"
                
                handler = getattr(self, handler_name, None)
                
                if handler and callable(handler):
                    handler(kwargs)
                    return
                else:
                    self.__console.warning(f"[ZAPI_VIEWERVEDO] Unknown function: {function_name} -> {handler_name}")
                    
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.__console.error(f"[ZAPI_VIEWERVEDO] Failed to parse kwargs: {json_kwargs}")

        except Exception as e:
            self.__console.error(f"[ZAPI_VIEWERVEDO] Dispatch error: {e}")

    # ----------------------------------------------------------------
    # zapi_* handler functions
    # ----------------------------------------------------------------
    def zapi_terminate(self, payload=None):
        """Handle termination request."""
        if self._visualizer:
            self._visualizer._should_close = True
        self.__console.info("[ZAPI_VIEWERVEDO] Termination requested")

    def zapi_ping(self, kwargs=None):
        """Handle ping request — reply with pong via router."""
        self.__console.debug("[ZAPI_VIEWERVEDO] Received ping, sending pong")
        if self.__router_socket and self.__router_socket.is_joined:
            identity = kwargs.get("_identity") if kwargs else None
            if identity:
                 # Construct payload
                 socket_name = self.__router_socket.socket_id
                 function = "pong"
                 kwargs_reply = {}
                 
                 reply_parts = [
                     identity,
                     socket_name.encode('utf-8'),
                     function.encode('utf-8'),
                     json.dumps(kwargs_reply).encode('utf-8')
                 ]
                 self.__router_socket.dispatch(reply_parts)

    def zapi_load_spool(self, kwargs=None):
        """Handle load_spool request."""
        self.__console.info(f"Received zapi_load_spool with kwargs: {kwargs}")
        if kwargs and "path" in kwargs:
            self.__console.info(f"[ZAPI_VIEWERVEDO] Processing load_spool: {kwargs['path']}")
            
            # Bridge to Visualizer: Push to visualizer's thread-safe queue
            if self._visualizer:
                # Add command field for visualizer to identify action
                request_payload = kwargs.copy()
                request_payload["command"] = "load_spool"
                self._visualizer.push_request(request_payload)
            else:
                self.push_to_queue(kwargs) # Fallback if no visualizer attached
        else:
            self.__console.warning("[ZAPI_VIEWERVEDO] Received load_spool request without path")

    def reply_load_spool(self, path: str, success: bool, identity=None):
        """Send a reply for load_spool request."""
        if self.__router_socket and self.__router_socket.is_joined and identity:
            kwargs = {
                "path": path,
                "status": "success" if success else "failed"
            }
            
            # Manual dispatch for ROUTER reply
            reply_parts = [
                identity,
                self.__router_socket.socket_id.encode('utf-8'),
                "reply_load_spool".encode('utf-8'),
                json.dumps(kwargs).encode('utf-8')
            ]
            self.__router_socket.dispatch(reply_parts)
            self.__console.info(f"[ZAPI_VIEWERVEDO] Sent reply for {path}: {success}")

    def update_spool_pose(self, pose: dict, identity=None):
        """Send latest spool pose to SimTool so saved values match viewer state."""
        if self.__router_socket and self.__router_socket.is_joined and identity:
            reply_parts = [
                identity,
                self.__router_socket.socket_id.encode('utf-8'),
                "update_spool_pose".encode('utf-8'),
                json.dumps(pose).encode('utf-8')
            ]
            self.__router_socket.dispatch(reply_parts)
            self.__console.info(f"[ZAPI_VIEWERVEDO] Sent spool pose update: {pose}")

    def zapi_flip_spool_x(self, kwargs=None):
        """Handle request to flip the currently loaded spool in X direction."""
        self.__console.info(f"Received zapi_flip_spool_x with kwargs: {kwargs}")
        request_payload = {"command": "flip_spool_x"}
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_move_spool(self, kwargs=None):
        """Handle request to move the currently loaded spool to an absolute position."""
        self.__console.info(f"Received zapi_move_spool with kwargs: {kwargs}")
        if not kwargs:
            return
        request_payload = {
            "command": "move_spool",
            "x": kwargs.get("x", 0.0),
            "y": kwargs.get("y", 0.0),
            "z": kwargs.get("z", 0.0),
            "x_rotation": kwargs.get("x_rotation", 0.0),
            "z_rotation": kwargs.get("z_rotation", 0.0),
            "_identity": kwargs.get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_move_positioner(self, kwargs=None):
        """Handle positioner joint move request."""
        self.__console.info(f"Received zapi_move_positioner with kwargs: {kwargs}")
        if not kwargs:
            return
        request_payload = {
            "command": "move_positioner",
            "axis": kwargs.get("axis"),
            "position": kwargs.get("position", 0.0),
            "velocity": kwargs.get("velocity", 0.0),
            "fix_f_column_r": kwargs.get("fix_f_column_r", False),
            "fix_m_column_z": kwargs.get("fix_m_column_z", False),
            "_identity": kwargs.get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_fix_spool(self, kwargs=None):
        """Handle fix_spool request (chuck 기준 offset 저장 후 강체 부착)."""
        self.__console.info(f"Received zapi_fix_spool with kwargs: {kwargs}")
        payload = {"command": "fix_spool", "_identity": (kwargs or {}).get("_identity")}
        if self._visualizer:
            self._visualizer.push_request(payload)
        else:
            self.push_to_queue(payload)

    def zapi_unfix_spool(self, kwargs=None):
        """Handle unfix_spool request (고정 해제)."""
        self.__console.info(f"Received zapi_unfix_spool with kwargs: {kwargs}")
        payload = {"command": "unfix_spool", "_identity": (kwargs or {}).get("_identity")}
        if self._visualizer:
            self._visualizer.push_request(payload)
        else:
            self.push_to_queue(payload)

    def zapi_set_spool_offset(self, kwargs=None):
        """Handle set_spool_offset request (로드 복원: offset 적용 강체 부착)."""
        self.__console.info(f"Received zapi_set_spool_offset with kwargs: {kwargs}")
        if not kwargs:
            return
        payload = {
            "command": "set_spool_offset",
            "offset_R": kwargs.get("offset_R"),
            "offset_t": kwargs.get("offset_t"),
            "_identity": kwargs.get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(payload)
        else:
            self.push_to_queue(payload)

    def zapi_filter_spool(self, kwargs=None):
        """Handle filter_spool request (현재 로드된 스풀에 직접 적용)."""
        self.__console.info(f"Received zapi_filter_spool with kwargs: {kwargs}")
        if not kwargs:
            return
        request_payload = {
            "command": "filter_spool",
            "method": kwargs.get("method"),
            "params": kwargs.get("params", {}),
            "_identity": kwargs.get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_reconstruct_mesh(self, kwargs=None):
        """Handle reconstruct_mesh request (현재 로드된 스풀로 메시 재건)."""
        self.__console.info(f"Received zapi_reconstruct_mesh with kwargs: {kwargs}")
        if not kwargs:
            return
        request_payload = {
            "command": "reconstruct_mesh",
            "params": kwargs.get("params", {}),
            "_identity": kwargs.get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_save_spool(self, kwargs=None):
        """Handle save_spool request (현재 로드된 스풀/메시 저장)."""
        self.__console.info(f"Received zapi_save_spool with kwargs: {kwargs}")
        if not kwargs or "path" not in kwargs:
            self.__console.warning("[ZAPI_VIEWERVEDO] save_spool without path")
            return
        request_payload = {
            "command": "save_spool",
            "path": kwargs.get("path"),
            "_identity": kwargs.get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_load_test_weld_point(self, kwargs=None):
        """Handle load_test_weld_point request."""
        self.__console.info(f"Received zapi_load_test_weld_point with kwargs: {kwargs}")
        if kwargs and "path" in kwargs:
            self.__console.info(f"[ZAPI_VIEWERVEDO] Processing load_test_weld_point: {kwargs['path']}")
            
            if self._visualizer:
                request_payload = kwargs.copy()
                request_payload["command"] = "load_test_weld_point"
                self._visualizer.push_request(request_payload)
            else:
                self.push_to_queue(kwargs)
        else:
            self.__console.warning("[ZAPI_VIEWERVEDO] Received load_test_weld_point request without path")

    def zapi_set_mode(self, kwargs=None):
        """Handle set_mode request to define execution mode."""
        self.__console.info(f"Received zapi_set_mode with kwargs: {kwargs}")
        if kwargs and "mode" in kwargs:
            mode = kwargs["mode"]
            if mode in ["simulation", "real"]:
                self.__console.info(f"[ZAPI_VIEWERVEDO] Setting execution mode to: {mode}")
                self._current_mode = mode
            else:
                self.__console.warning(f"[ZAPI_VIEWERVEDO] Invalid mode received: {mode}")
        else:
            self.__console.warning("[ZAPI_VIEWERVEDO] Received set_mode request without mode parameter")

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------
    def push_to_queue(self, data):
        """Push data to the request queue (thread-safe)."""
        with self._queue_lock:
            self.request_queue.append(data)

    def pop_from_queue(self):
        """Pop data from the request queue (thread-safe). Returns None if empty."""
        with self._queue_lock:
            if self.request_queue:
                return self.request_queue.popleft()
        return None
