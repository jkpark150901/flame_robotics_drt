from __future__ import annotations

import multiprocessing as mp
import pickle
import queue
import threading
import time
import traceback
import uuid

from util.logger.console import ConsoleLogger
from common.zpipe import AsyncZSocket, ZPipe
from common.zapi import ZAPIBase
from robot_core.zapi import resolve_zapi_config, describe_endpoint

# Robot Core request "operation" values. Defined here (rather than in
# robot_core.worker) so both the Visualizer call sites and robot_core.worker's
# dispatch can import the same constants without introducing a circular import
# (robot_core.worker imports viewervedo.visualizer).
OPERATION_POSE_DETERMINE = "pose_determine"
# Single source_q -> target_pose planning call. Robot Core has no notion of
# target groups, positioner rotation, or multi-target sequencing - that
# orchestration lives in SimTool now (see ROBOT_CORE_DECOUPLING_PLAN.md).
OPERATION_PLAN_SINGLE_TARGET = "plan_single_target"
DEFAULT_OPERATION = OPERATION_PLAN_SINGLE_TARGET


def submit_robot_core_request(robot_core, request_data, snapshot, *, console=None,
                               not_running_message="robot core process is not running"):
    """Submit a request to a robot-core client and normalize submission failures.

    Owns the part of a Robot Core call that is identical across every request
    type (plan_inspection_path / determine_ef_pose / check_ef_pose_ik, ...):
    validating the client is alive, calling submit(), logging, and turning any
    exception into the standard {"status": "failed", ...} envelope callers reply
    with over ZAPI. Building the request/snapshot payload itself stays with the
    caller, since that requires live scene state (meshes, joint values, ...)
    only the Visualizer has.

    Returns:
        (request_id, None) on success, or (None, failure_result) on failure.
    """
    console = console or ConsoleLogger.get_logger()
    try:
        if robot_core is None or not robot_core.is_running:
            raise RuntimeError(not_running_message)
        # Submission itself isn't logged here anymore - robot core logs the
        # request on receipt (zapi_execute_request's "Received request" line
        # for the standalone WSL service, or its embedded-process equivalent),
        # so this stays the single place a normal submission is recorded.
        return robot_core.submit(request_data, snapshot), None
    except Exception as exc:
        console.error(f"Robot Core request submission failed: {exc}")
        failure = {"status": "failed", "message": str(exc), "elapsed": 0.0}
        return None, failure


def _completion(request_id, request, *, output=None, error=None):
    result = {
        "command": "robot_core_completed",
        "request_id": request_id,
        "operation": request.get("operation", DEFAULT_OPERATION),
        "_identity": request.get("_identity"),
        "_client_request_id": request.get("_client_request_id"),
        "robot_name": request.get("robot_name"),
        "optimizer": request.get("optimizer"),
        "optimize_path": bool(request.get("optimize_path", False)),
    }
    if error is None:
        result.update({"status": "completed", "output": output})
    else:
        result.update({"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
    return result


def execute_request(config, item):
    """Run one robot-core request. Owns the request's lifecycle logging
    (received/processing/result) so callers (viewer's submit path, the
    embedded worker, and the standalone ZAPI worker) don't each have to log
    it themselves - this is the one place that runs regardless of transport."""
    from robot_core.worker import execute_robot_core_request

    console = ConsoleLogger.get_logger()
    request = item.get("request") or {}
    request_id = item.get("request_id")
    operation = request.get("operation", DEFAULT_OPERATION)
    robot_name = request.get("robot_name")
    console.info(
        f"Robot Core processing request_id={request_id} operation={operation}"
        + (f" robot={robot_name}" if robot_name else ""))
    t0 = time.time()
    try:
        output = execute_robot_core_request(config, request, item.get("snapshot") or {})
        console.info(
            f"Robot Core request_id={request_id} completed in {time.time() - t0:.2f}s")
        return _completion(request_id, request, output=output)
    except BaseException as exc:
        console.error(
            f"Robot Core request_id={request_id} failed after {time.time() - t0:.2f}s: {exc}")
        return _completion(request_id, request, error=exc)


def _embedded_worker(config, request_queue, result_queue):
    while True:
        item = request_queue.get()
        if item is None:
            break
        result_queue.put(execute_request(config, item))


# How long a request may sit unanswered before a robot-core client gives up
# on it and synthesizes a failure completion (see _watch_stale_requests on
# both EmbeddedRobotCoreClient and ExternalRobotCoreClient). A crashed
# process is caught faster via is_alive() polling (EmbeddedRobotCoreClient's
# _on_process_died) or a closed socket (external), but a worker that hangs
# WITHOUT dying or disconnecting - e.g. a native OMPL planner binding that
# never honors its PlannerTerminationCondition - looks identical to "still
# working" by either of those signals, so a wall-clock timeout is the only
# way to notice and recover from it.
DEFAULT_REQUEST_TIMEOUT_SEC = 120.0
STALE_REQUEST_SWEEP_INTERVAL_SEC = 5.0
# Per-request requests may explicitly configure a longer planning_timeout
# than DEFAULT_REQUEST_TIMEOUT_SEC (e.g. a slow optimizer) - honor that
# instead of prematurely killing a legitimately long-running request, plus
# this much margin for the IK/collision-mesh/etc. work around the actual
# planner.solve() call that planning_timeout alone doesn't cover.
STALE_REQUEST_MARGIN_SEC = 30.0


class EmbeddedRobotCoreClient:
    """Spawn-safe client for the embedded robot-core worker process."""

    def __init__(self, config, completion_callback, request_timeout_sec=DEFAULT_REQUEST_TIMEOUT_SEC):
        self._config = config
        self._callback = completion_callback
        self._context = mp.get_context("spawn")
        self._request_queue = None
        self._result_queue = None
        self._process = None
        self._listener = None
        self._watchdog = None
        self._running = False
        self._request_timeout_sec = float(request_timeout_sec)
        self._console = ConsoleLogger.get_logger()
        # request_id -> (request, submit_monotonic_time), for requests
        # submitted but not yet resolved. Used to synthesize a failure
        # completion for whatever was in flight if the worker process dies
        # without replying (e.g. a native crash in a planner library, which
        # is not a catchable Python exception) - see _on_process_died() - or
        # if it hangs WITHOUT dying (e.g. an OMPL planner binding that never
        # honors its termination condition) - see _watch_stale_requests().
        self._pending = {}
        self._pending_lock = threading.Lock()

    @property
    def pid(self):
        return None if self._process is None else self._process.pid

    @property
    def is_running(self):
        return bool(self._running and self._process is not None and self._process.is_alive())

    def start(self):
        if self.is_running:
            return
        self._request_queue = self._context.Queue()
        self._result_queue = self._context.Queue()
        self._process = self._context.Process(
            target=_embedded_worker,
            args=(self._config, self._request_queue, self._result_queue),
            name="drt-robot-core",
        )
        self._process.start()
        self._running = True
        self._listener = threading.Thread(target=self._listen, name="robot-core-results", daemon=True)
        self._listener.start()
        self._watchdog = threading.Thread(
            target=self._watch_stale_requests, name="robot-core-watchdog", daemon=True)
        self._watchdog.start()

    def _listen(self):
        while self._running:
            try:
                result = self._result_queue.get(timeout=0.2)
            except queue.Empty:
                if self._process is not None and not self._process.is_alive():
                    self._on_process_died()
                continue
            if result is None:
                break
            with self._pending_lock:
                self._pending.pop(result.get("request_id"), None)
            self._console.info(
                f"Robot Core result received: request_id={result.get('request_id')} "
                f"operation={result.get('operation')} status={result.get('status')} "
                f"robot={result.get('robot_name')} - invoking completion callback")
            self._callback(result)

    def _on_process_died(self):
        """Worker process exited without us calling stop() - most likely a
        native crash (e.g. an OMPL planner throwing outside Python's
        exception machinery) rather than a normal completion. Fail out any
        request that was in flight so callers don't hang forever waiting for
        a reply that will never come, then respawn so later requests work."""
        if not self._running:
            return
        with self._pending_lock:
            stale = list(self._pending.items())
            self._pending.clear()
        exit_code = self._process.exitcode if self._process is not None else None
        for request_id, (request, _submitted_at) in stale:
            self._console.error(
                f"Robot Core process died (exitcode={exit_code}) while request_id={request_id} "
                "was in flight - reporting it as failed")
            self._callback(_completion(
                request_id, request,
                error=f"robot core process crashed (exitcode={exit_code})"))
        self._console.warning("Robot Core process died unexpectedly - restarting it")
        self._process = None
        self._running = False
        self.start()

    def _watch_stale_requests(self):
        """Recover from a worker that hangs WITHOUT dying (e.g. an OMPL
        planner binding - AORRTC is known to do this - that never returns
        from its native solve() call even after its own timeout budget). This
        looks identical to "still working" to is_alive() polling, so
        _on_process_died() never fires: the request (and every later one,
        since _embedded_worker processes its queue one item at a time on a
        single thread) would otherwise wait forever with no reply and no
        error, silently wedging the whole embedded robot core."""
        while self._running:
            time.sleep(STALE_REQUEST_SWEEP_INTERVAL_SEC)
            if not self._running:
                return
            now = time.monotonic()
            with self._pending_lock:
                stale = [
                    (request_id, request) for request_id, (request, submitted_at) in self._pending.items()
                    if now - submitted_at > max(
                        self._request_timeout_sec,
                        float(request.get("planning_timeout") or 0.0) + STALE_REQUEST_MARGIN_SEC)
                ]
                for request_id, _ in stale:
                    self._pending.pop(request_id, None)
            if not stale:
                continue
            self._console.error(
                f"Robot Core worker process (pid={self.pid}) has not replied to "
                f"{len(stale)} request(s) in time - treating it as wedged (hung without "
                "crashing), reporting them as failed and restarting the process")
            for request_id, request in stale:
                self._callback(_completion(
                    request_id, request, error="robot core worker did not reply in time (wedged)"))
            self._restart_process()

    def _restart_process(self):
        """Force-recycle a wedged worker process. Unlike _on_process_died
        (which reacts to a process that already exited), the process here is
        still alive but stuck in a blocking native call, so it has to be
        killed explicitly before a replacement can take over its queue."""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.join(2.0)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(1.0)
            except Exception:
                pass
        self._process = None
        self._running = False
        self.start()

    def submit(self, request, snapshot):
        if not self.is_running:
            raise RuntimeError("robot core process is not running")
        request_id = uuid.uuid4().hex
        with self._pending_lock:
            self._pending[request_id] = (request, time.monotonic())
        self._request_queue.put({
            "request_id": request_id,
            "request": request,
            "snapshot": snapshot,
        })
        return request_id

    def stop(self, timeout=5.0):
        self._running = False
        if self._request_queue is not None:
            self._request_queue.put(None)
        if self._process is not None:
            self._process.join(timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(2.0)
        if self._result_queue is not None:
            self._result_queue.put(None)
        return self._process is None or not self._process.is_alive()

    shutdown = stop


class ExternalRobotCoreClient(ZAPIBase):
    """ZAPI(dealer) client for a standalone robot-core service."""

    def __init__(self, config=None, completion_callback=None, zpipe=None,
                 request_timeout_sec=DEFAULT_REQUEST_TIMEOUT_SEC):
        ZAPIBase.__init__(self)

        self._console = ConsoleLogger.get_logger()
        self._callback = completion_callback
        self._zpipe = zpipe
        self._running = False
        self._request_timeout_sec = float(request_timeout_sec)
        # request_id -> (request, submit_monotonic_time). A request is
        # "stale" (assumed lost - the service died or dropped it) once it has
        # been pending longer than _request_timeout_sec with no reply.
        self._pending = {}
        self._pending_lock = threading.Lock()
        self._watchdog = None

        self._transport, self._channel, self._port = resolve_zapi_config(config or {})
        self.endpoint = describe_endpoint(self._transport, self._channel, self._port)

        # Dealer Socket
        self.__dealer_socket = AsyncZSocket("ZAPI_ROBOTCORE_CLIENT", "dealer")
        if not self.__dealer_socket.create(pipeline=zpipe):
            self._console.error("[ZAPI_ROBOTCORE_CLIENT] Failed to create dealer socket")

    @property
    def pid(self):
        return None

    @property
    def is_running(self):
        return self._running

    def start(self):
        """Connect to the standalone robot core service."""
        if self._running:
            self._console.warning("[ZAPI_ROBOTCORE_CLIENT] Already running")
            return
        self._running = True

        self.__dealer_socket.set_message_callback(self._on_message_received)
        if self.__dealer_socket.join(self._transport, self._channel, self._port):
            self._console.info(f"[ZAPI_ROBOTCORE_CLIENT] Connected to {self.endpoint}")
        else:
            self._running = False
            self._console.error(
                f"[ZAPI_ROBOTCORE_CLIENT] Failed to connect to {self.endpoint}")
            return
        self._watchdog = threading.Thread(
            target=self._watch_stale_requests, name="robot-core-client-watchdog", daemon=True)
        self._watchdog.start()

    def _watch_stale_requests(self):
        while self._running:
            time.sleep(STALE_REQUEST_SWEEP_INTERVAL_SEC)
            now = time.monotonic()
            with self._pending_lock:
                stale = [
                    (request_id, request) for request_id, (request, submitted_at) in self._pending.items()
                    if now - submitted_at > self._request_timeout_sec
                ]
                for request_id, _ in stale:
                    self._pending.pop(request_id, None)
            for request_id, request in stale:
                self._console.error(
                    f"[ZAPI_ROBOTCORE_CLIENT] request_id={request_id} got no reply within "
                    f"{self._request_timeout_sec:.0f}s (robot core service likely crashed or "
                    "dropped it) - reporting it as failed and discarding it")
                if self._callback:
                    self._callback(_completion(
                        request_id, request,
                        error=f"no reply from robot core service within {self._request_timeout_sec:.0f}s"))

    def _on_message_received(self, multipart_data):
        """DEALER receives: [socket_name, function, payload]"""
        try:
            if len(multipart_data) < 3:
                return
            payload = multipart_data[2]
            result = pickle.loads(payload)
            with self._pending_lock:
                self._pending.pop(result.get("request_id"), None)
            if self._callback:
                self._callback(result)
        except Exception as e:
            self._console.error(f"[ZAPI_ROBOTCORE_CLIENT] Error receiving message: {e}")

    def submit(self, request, snapshot):
        if not self._running:
            raise RuntimeError("standalone robot core is not connected")
        request_id = uuid.uuid4().hex
        with self._pending_lock:
            self._pending[request_id] = (request, time.monotonic())
        payload = pickle.dumps({
            "request_id": request_id,
            "request": request,
            "snapshot": snapshot,
        }, protocol=pickle.HIGHEST_PROTOCOL)
        self.call_raw(self.__dealer_socket, "zapi_execute_request", payload)
        return request_id

    def stop(self, timeout=2.0):
        self._running = False
        if self.__dealer_socket:
            self.__dealer_socket.destroy_socket()
        with self._pending_lock:
            self._pending.clear()
        self._console.debug("[ZAPI_ROBOTCORE_CLIENT] Stopped")
        return True

    def shutdown(self):
        if self._running:
            self.call_raw(
                self.__dealer_socket, "zapi_shutdown",
                pickle.dumps({}, protocol=pickle.HIGHEST_PROTOCOL))
            time.sleep(0.1)  # let the dealer flush before the socket is closed
        return self.stop()


def serve_robot_core(config, service_config=None):
    """Serve robot-core requests until Ctrl+C or an explicit shutdown command."""
    from robot_core.zapi import ZAPI

    serve_config = dict(config or {})
    if service_config:
        serve_config["robot_core_service"] = service_config

    zapi = ZAPI(config=serve_config, zpipe=ZPipe.create_pipe(
        io_threads=serve_config.get("n_io_context", 10)))
    zapi.run()
    try:
        while not zapi.wait(0.5):
            pass
    finally:
        zapi.stop()
