"""
Zapi - ZMQ Communication Module for Robot Core
@note
- Same ZAPI style as viewervedo/simtool: ZPipe + AsyncZSocket + ZAPIBase
- Uses ROUTER pattern for async request/reply
- Dispatches incoming messages to internal zapi_* handler methods
- Argument frames are pickled (not JSON) because requests/snapshots carry numpy data
"""

import pickle
import queue
import threading

from util.logger.console import ConsoleLogger
from common.zpipe import AsyncZSocket, ZPipe
from common.zapi import ZAPIBase

DEFAULT_TRANSPORT = "tcp"
DEFAULT_CHANNEL = "127.0.0.1"
DEFAULT_PORT = 5557


def resolve_zapi_config(config: dict):
    """Resolve (transport, channel, port) from a robot_core_service config block.

    Accepts the zapi style keys (transport/channel/port) and falls back to the
    legacy "endpoint" string (e.g. tcp://127.0.0.1:5557) for backward compatibility.
    """
    config = config or {}
    transport = config.get("transport")
    channel = config.get("channel")
    port = config.get("port")

    endpoint = config.get("endpoint")
    if endpoint and (transport is None or channel is None):
        parsed_transport, _, remainder = str(endpoint).partition("://")
        if remainder:
            transport = transport or parsed_transport
            if parsed_transport in ("ipc", "inproc"):
                channel = channel or remainder
            else:
                address, _, parsed_port = remainder.rpartition(":")
                channel = channel or (address or remainder)
                if port is None and parsed_port.isdigit():
                    port = int(parsed_port)

    return (
        transport or DEFAULT_TRANSPORT,
        channel or DEFAULT_CHANNEL,
        int(port if port is not None else DEFAULT_PORT),
    )


def describe_endpoint(transport: str, channel: str, port: int) -> str:
    """Human readable endpoint string for logging."""
    if transport in ("ipc", "inproc"):
        return f"{transport}://{channel}"
    return f"{transport}://{channel}:{port}"


class ZAPI(ZAPIBase):
    """ZMQ communication manager for Robot Core.

    Binds a ROUTER socket, dispatches incoming messages to zapi_* handlers and
    executes computation requests on a single worker thread so that control
    messages (ping/shutdown) stay responsive while a request is running.
    """

    def __init__(self, config: dict = None, zpipe: ZPipe = None):
        super().__init__()
        if config is None:
            config = {}

        self._config = config
        self.__console = ConsoleLogger.get_logger()
        self._zpipe = zpipe
        self._running = False
        self._worker = None
        self._request_queue = queue.Queue()
        self._terminated = threading.Event()

        service_config = (
            config.get("robot_core_service", {})
            or config.get("planner_service", {})
            or {}
        )
        self._transport, self._channel, self._port = resolve_zapi_config(service_config)

        # Router Socket (receive requests + system commands)
        self.__router_socket = AsyncZSocket("ZAPI_ROBOTCORE", "router")
        if not self.__router_socket.create(pipeline=zpipe):
            self.__console.error("[ZAPI_ROBOTCORE] Failed to create router socket")

    @property
    def endpoint(self) -> str:
        return describe_endpoint(self._transport, self._channel, self._port)

    @property
    def is_running(self) -> bool:
        return self._running

    def run(self):
        """Start the Robot Core ZAPI (bind router, start worker thread)."""
        if self._running:
            self.__console.warning("[ZAPI_ROBOTCORE] Already running")
            return

        self._running = True
        self.__console.debug("Starting Robot Core ZAPI...")

        self._worker = threading.Thread(
            target=self._worker_loop, name="robot-core-worker", daemon=True)
        self._worker.start()

        self.__router_socket.set_message_callback(self._on_message_received)
        if self.__router_socket.join(self._transport, self._channel, self._port):
            self.__console.info(f"[ZAPI_ROBOTCORE] Router bound: {self.endpoint}")
        else:
            self.__console.error(
                f"[ZAPI_ROBOTCORE] Failed to bind router socket: {self.endpoint}")
            self._running = False
            self._terminated.set()

    def stop(self):
        """Stop the worker thread and cleanup sockets."""
        self._running = False
        self._terminated.set()
        self._request_queue.put(None)

        if self._worker is not None and threading.current_thread() is not self._worker:
            self._worker.join(timeout=2)

        if self.__router_socket:
            self.__router_socket.destroy_socket()

        self.__console.debug("[ZAPI_ROBOTCORE] Stopped and cleaned up")

    def wait(self, timeout: float = None) -> bool:
        """Block until a shutdown request arrives. Returns True when terminated."""
        return self._terminated.wait(timeout)

    # ----------------------------------------------------------------
    # Message reception (called from AsyncZSocket's receiver thread)
    # ----------------------------------------------------------------
    def _on_message_received(self, multipart_data):
        """Callback from AsyncZSocket receiver thread.
        ROUTER socket receives: [identity, socket_name, function, payload]
        Sender (DEALER) calls ZAPIBase.call_raw -> [socket_name, function, payload]
        """
        try:
            # PROBE_ROUTER connection event (identity + empty message)
            if len(multipart_data) == 2 and len(multipart_data[1]) == 0:
                self.__console.info(
                    f"[ZAPI_ROBOTCORE] Client connected: {multipart_data[0]}")
                return

            if len(multipart_data) < 4:
                self.__console.warning(
                    f"[ZAPI_ROBOTCORE] Received incomplete message: {len(multipart_data)} parts")
                return

            identity = multipart_data[0]
            function_name = multipart_data[2]
            payload = multipart_data[3]

            function_name_str = (
                function_name.decode('utf-8')
                if isinstance(function_name, bytes) else function_name)

            self._dispatch_message(identity, function_name_str, payload)

        except Exception as e:
            self.__console.error(f"[ZAPI_ROBOTCORE] Error processing message: {e}")

    def _dispatch_message(self, identity, function_name, payload):
        """Route incoming messages to the appropriate zapi_* handler."""
        try:
            try:
                kwargs = pickle.loads(payload) if payload else {}
            except Exception:
                self.__console.error("[ZAPI_ROBOTCORE] Failed to parse payload")
                return

            if not isinstance(kwargs, dict):
                kwargs = {"payload": kwargs}

            # Inject identity into kwargs so we know who to reply to
            kwargs["_identity"] = identity

            handler_name = (
                function_name if function_name.startswith("zapi_")
                else f"zapi_{function_name}")
            handler = getattr(self, handler_name, None)

            if handler and callable(handler):
                handler(kwargs)
            else:
                self.__console.warning(
                    f"[ZAPI_ROBOTCORE] Unknown function: {function_name} -> {handler_name}")

        except Exception as e:
            self.__console.error(f"[ZAPI_ROBOTCORE] Dispatch error: {e}")

    def _reply(self, identity, function: str, kwargs: dict):
        """Send a reply back to the requesting peer through the router socket."""
        if not self.__router_socket or not self.__router_socket.is_joined:
            return False
        if identity is None:
            self.__console.warning(f"[ZAPI_ROBOTCORE] Cannot reply {function}: no identity")
            return False
        payload = pickle.dumps(kwargs, protocol=pickle.HIGHEST_PROTOCOL)
        return self.call_raw(self.__router_socket, function, payload, identity=identity)

    # ----------------------------------------------------------------
    # request worker
    # ----------------------------------------------------------------
    def _worker_loop(self):
        """Execute queued robot core requests one at a time."""
        from robot_core.service import execute_request

        while True:
            item = self._request_queue.get()
            if item is None:
                break
            identity = item.pop("_identity", None)
            result = execute_request(self._config, item)
            self._reply(identity, "robot_core_completed", result)

    # ----------------------------------------------------------------
    # zapi_* handler functions
    # ----------------------------------------------------------------
    def zapi_execute_request(self, kwargs=None):
        """Queue a robot core computation request."""
        kwargs = kwargs or {}
        request = kwargs.get("request") or {}
        self.__console.info(
            f"[ZAPI_ROBOTCORE] Received request: id={kwargs.get('request_id')} "
            f"operation={request.get('operation', 'plan_inspection_path')}")
        self._request_queue.put(kwargs)

    def zapi_ping(self, kwargs=None):
        """Handle ping request - reply with pong."""
        self.__console.debug("[ZAPI_ROBOTCORE] Received ping, sending pong")
        self._reply((kwargs or {}).get("_identity"), "pong", {})

    def zapi_shutdown(self, kwargs=None):
        """Handle shutdown request - releases wait() in the main thread."""
        self.__console.info("[ZAPI_ROBOTCORE] Shutdown requested")
        self._terminated.set()
