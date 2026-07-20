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

    def update_inspection_point(self, point: dict, identity=None):
        """Send picked pipe inspection point to SimTool."""
        if self.__router_socket and self.__router_socket.is_joined and identity:
            reply_parts = [
                identity,
                self.__router_socket.socket_id.encode('utf-8'),
                "update_inspection_point".encode('utf-8'),
                json.dumps(point).encode('utf-8')
            ]
            self.__router_socket.dispatch(reply_parts)
            self.__console.info(f"[ZAPI_VIEWERVEDO] Sent inspection point update: {point}")

    def update_positioner_pose(self, pose: dict, identity=None):
        """Send latest positioner pose to SimTool."""
        if self.__router_socket and self.__router_socket.is_joined and identity:
            reply_parts = [
                identity,
                self.__router_socket.socket_id.encode('utf-8'),
                "update_positioner_pose".encode('utf-8'),
                json.dumps(pose).encode('utf-8')
            ]
            self.__router_socket.dispatch(reply_parts)
            self.__console.info(f"[ZAPI_VIEWERVEDO] Sent positioner pose update: {pose}")

    def update_robot_joint_state(self, state: dict, identity=None):
        """Send latest robot joint states to SimTool."""
        if self.__router_socket and self.__router_socket.is_joined and identity:
            reply_parts = [
                identity,
                self.__router_socket.socket_id.encode('utf-8'),
                "update_robot_joint_state".encode('utf-8'),
                json.dumps(state).encode('utf-8')
            ]
            self.__router_socket.dispatch(reply_parts)

    def update_chuck_mount_points(self, points: dict, identity=None):
        """Send picked pipe chuck mount points to SimTool."""
        if self.__router_socket and self.__router_socket.is_joined and identity:
            reply_parts = [
                identity,
                self.__router_socket.socket_id.encode('utf-8'),
                "update_chuck_mount_points".encode('utf-8'),
                json.dumps(points).encode('utf-8')
            ]
            self.__router_socket.dispatch(reply_parts)
            self.__console.info(f"[ZAPI_VIEWERVEDO] Sent chuck mount points update: {points}")

    def update_chuck_mount_profile(self, profile: dict, identity=None):
        """Send the pipe cylinder profile used for chuck alignment to SimTool."""
        if self.__router_socket and self.__router_socket.is_joined and identity:
            reply_parts = [
                identity,
                self.__router_socket.socket_id.encode('utf-8'),
                "update_chuck_mount_profile".encode('utf-8'),
                json.dumps(profile).encode('utf-8')
            ]
            self.__router_socket.dispatch(reply_parts)
            self.__console.info(f"[ZAPI_VIEWERVEDO] Sent chuck mount profile update: {profile}")

    def reply_inspection_path(self, result: dict, identity=None):
        """Send path planning result to SimTool."""
        if self.__router_socket and self.__router_socket.is_joined and identity:
            payload = json.dumps(result).encode('utf-8')
            reply_parts = [
                identity,
                self.__router_socket.socket_id.encode('utf-8'),
                "reply_inspection_path".encode('utf-8'),
                payload
            ]
            self.__router_socket.dispatch(reply_parts)
            try:
                compact_result = self._inspection_path_log_text(result)
            except Exception:
                compact_result = str(result)
            self.__console.debug("[ZAPI_VIEWERVEDO] Sent inspection path result\n" + compact_result)

    def _fmt_float(self, value, digits=3, default="-"):
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return default

    def _fmt_q(self, values):
        if values is None:
            return "-"
        try:
            return "[" + ", ".join(f"{float(v):.4g}" for v in values) + "]"
        except Exception:
            return str(values)

    def _inspection_path_log_text(self, result: dict):
        """Human-readable inspection path log; the full JSON payload is still sent to the UI."""
        if not isinstance(result, dict):
            return str(result)

        is_ik_check = result.get("mode") == "ik_check"
        lines = [
            "Inspection IK check result" if is_ik_check else "Inspection path result",
            f"  status  : {result.get('status', '-')}",
            f"  planner : {result.get('planner', '-')}",
        ]
        timing = result.get("timing") or {}
        if timing:
            lines.append(
                "  timing  : "
                f"wall={self._fmt_float(timing.get('planning_wall', timing.get('ik_wall')))}s, "
                f"sum={self._fmt_float(timing.get('planning_sum', timing.get('ik_sum')))}s, "
                f"obstacle={self._fmt_float(timing.get('obstacle_mesh'))}s"
            )

        robots = result.get("robots")
        inspection_groups = result.get("inspection_groups")
        if isinstance(inspection_groups, list) and inspection_groups:
            for group in inspection_groups:
                if not isinstance(group, dict):
                    continue
                lines.append(f"  {group.get('name', 'inspection pose')} :")
                group_robots = group.get("robots") or {}
                for robot_name, robot_result in group_robots.items():
                    if not isinstance(robot_result, dict):
                        continue
                    ik = robot_result.get("ik_result") or {}
                    rt = robot_result.get("timing") or {}
                    verification = robot_result.get("verification") or {}
                    preview_reason = robot_result.get("collision_preview_reason") or robot_result.get("fallback_reason")
                    lines.extend([
                        f"    - {robot_name} ({robot_result.get('pose_name', '-')})",
                        f"        waypoints : {robot_result.get('waypoints', '-')}",
                        f"        preview   : collision={bool(robot_result.get('collision_preview', False))}"
                        + (f", reason={preview_reason}" if preview_reason else ""),
                        "        collision : "
                        f"edges={verification.get('colliding_edges', '-')}, "
                        f"positioner_checked={bool(verification.get('positioner_collision_checked', False))}",
                        "        ik        : "
                        f"success={bool(ik.get('success', False))}, "
                        f"fallback={bool(ik.get('fallback', False))}, "
                        f"pos_err={self._fmt_float(ik.get('position_error'), 6)}m, "
                        f"ori_err={self._fmt_float(ik.get('orientation_error'), 6)}rad, "
                        f"collision={bool(ik.get('collision', False))}, "
                        f"pairs={ik.get('collision_pair_count', 0)}, "
                        f"iter={ik.get('iterations', '-')}",
                        f"        init_q    : {self._fmt_q(robot_result.get('init_q'))}",
                        f"        target_q  : {self._fmt_q(robot_result.get('target_q'))}",
                    ])
                    if robot_result.get("ik_experiment"):
                        exp = robot_result.get("ik_experiment") or {}
                        lines.append(f"        experiment: {exp.get('meta', exp)}")
                    if is_ik_check:
                        lines.append(
                            "        timing    : "
                            f"target={self._fmt_float(rt.get('target_setup'))}s, "
                            f"pin_cache={self._fmt_float(rt.get('pin_cache_lookup'))}s, "
                            f"start_q={self._fmt_float(rt.get('start_q_setup'))}s, "
                            f"ik={self._fmt_float(rt.get('ik'))}s, "
                            f"result={self._fmt_float(rt.get('ik_result_check'))}s, "
                            f"total={self._fmt_float(rt.get('total'))}s")
                    else:
                        lines.append(
                            "        timing    : "
                            f"setup={self._fmt_float(rt.get('planner_setup'))}s, "
                            f"ik={self._fmt_float(rt.get('ik'))}s, "
                            f"planning={self._fmt_float(rt.get('planning'))}s, "
                            f"verify={self._fmt_float(rt.get('collision_verification'))}s, "
                            f"convert={self._fmt_float(rt.get('path_conversion'))}s, "
                            f"total={self._fmt_float(rt.get('total'))}s")
                failures = group.get("failures") or {}
                if failures:
                    lines.append(f"    failures : {failures}")
            return "\n".join(lines)

        if isinstance(robots, dict) and robots:
            for robot_name, robot_result in robots.items():
                if not isinstance(robot_result, dict):
                    continue
                ik = robot_result.get("ik_result") or {}
                rt = robot_result.get("timing") or {}
                verification = robot_result.get("verification") or {}
                preview_reason = robot_result.get("collision_preview_reason") or robot_result.get("fallback_reason")
                lines.extend([
                    f"  - {robot_name} ({robot_result.get('pose_name', '-')})",
                    f"      waypoints : {robot_result.get('waypoints', '-')}",
                    f"      preview   : collision={bool(robot_result.get('collision_preview', False))}"
                    + (f", reason={preview_reason}" if preview_reason else ""),
                    "      collision : "
                    f"edges={verification.get('colliding_edges', '-')}, "
                    f"positioner_checked={bool(verification.get('positioner_collision_checked', False))}",
                    "      ik        : "
                    f"success={bool(ik.get('success', False))}, "
                    f"fallback={bool(ik.get('fallback', False))}, "
                    f"pos_err={self._fmt_float(ik.get('position_error'), 6)}m, "
                    f"ori_err={self._fmt_float(ik.get('orientation_error'), 6)}rad, "
                    f"collision={bool(ik.get('collision', False))}, "
                    f"pairs={ik.get('collision_pair_count', 0)}, "
                    f"iter={ik.get('iterations', '-')}",
                    f"      init_q    : {self._fmt_q(robot_result.get('init_q'))}",
                    f"      target_q  : {self._fmt_q(robot_result.get('target_q'))}",
                ])
                if robot_result.get("ik_experiment"):
                    exp = robot_result.get("ik_experiment") or {}
                    lines.append(f"      experiment: {exp.get('meta', exp)}")
                if is_ik_check:
                    lines.append(
                        "      timing    : "
                        f"target={self._fmt_float(rt.get('target_setup'))}s, "
                        f"pin_cache={self._fmt_float(rt.get('pin_cache_lookup'))}s, "
                        f"start_q={self._fmt_float(rt.get('start_q_setup'))}s, "
                        f"ik={self._fmt_float(rt.get('ik'))}s, "
                        f"result={self._fmt_float(rt.get('ik_result_check'))}s, "
                        f"total={self._fmt_float(rt.get('total'))}s")
                else:
                    lines.append(
                        "      timing    : "
                        f"setup={self._fmt_float(rt.get('planner_setup'))}s, "
                        f"ik={self._fmt_float(rt.get('ik'))}s, "
                        f"planning={self._fmt_float(rt.get('planning'))}s, "
                        f"verify={self._fmt_float(rt.get('collision_verification'))}s, "
                        f"convert={self._fmt_float(rt.get('path_conversion'))}s, "
                        f"total={self._fmt_float(rt.get('total'))}s")
            return "\n".join(lines)

        ik = result.get("ik_result") or {}
        lines.extend([
            f"  robot   : {result.get('robot', '-')}",
            f"  waypoints: {result.get('waypoints', '-')}",
            "  ik      : "
            f"success={bool(ik.get('success', False))}, "
            f"fallback={bool(ik.get('fallback', False))}, "
            f"pos_err={self._fmt_float(ik.get('position_error'), 6)}m, "
            f"ori_err={self._fmt_float(ik.get('orientation_error'), 6)}rad, "
            f"collision={bool(ik.get('collision', False))}, "
            f"iter={ik.get('iterations', '-')}",
            f"  init_q  : {self._fmt_q(result.get('init_q'))}",
            f"  target_q: {self._fmt_q(result.get('target_q'))}",
        ])
        if result.get("message"):
            lines.append(f"  message : {result.get('message')}")
        return "\n".join(lines)

    def _inspection_path_log_summary(self, result: dict):
        """Keep inspection path logs compact; the full payload is still sent to the UI."""
        if not isinstance(result, dict):
            return result
        summary = {
            "status": result.get("status"),
            "planner": result.get("planner"),
        }
        if "timing" in result:
            summary["timing"] = result.get("timing")
        robots = result.get("robots")
        if isinstance(robots, dict):
            summary["robots"] = {
                robot_name: {
                    "pose_name": robot_result.get("pose_name"),
                    "waypoints": robot_result.get("waypoints"),
                    "collision_preview": robot_result.get("collision_preview"),
                    "collision_preview_reason": robot_result.get("collision_preview_reason"),
                    "fallback_reason": robot_result.get("fallback_reason"),
                    "ik_result": robot_result.get("ik_result"),
                    "init_q": robot_result.get("init_q"),
                    "target_q": robot_result.get("target_q"),
                    "timing": robot_result.get("timing"),
                }
                for robot_name, robot_result in robots.items()
                if isinstance(robot_result, dict)
            }
            return summary
        summary.update({
            "robot": result.get("robot"),
            "waypoints": result.get("waypoints"),
            "collision_preview": result.get("collision_preview"),
            "collision_preview_reason": result.get("collision_preview_reason"),
            "fallback_reason": result.get("fallback_reason"),
            "ik_result": result.get("ik_result"),
            "init_q": result.get("init_q"),
            "target_q": result.get("target_q"),
        })
        return summary

    def reply_ef_pose(self, result: dict, identity=None):
        """Send EF pose determination result to SimTool."""
        if self.__router_socket and self.__router_socket.is_joined and identity:
            reply_parts = [
                identity,
                self.__router_socket.socket_id.encode('utf-8'),
                "reply_ef_pose".encode('utf-8'),
                json.dumps(result).encode('utf-8')
            ]
            self.__router_socket.dispatch(reply_parts)
            self.__console.info(f"[ZAPI_VIEWERVEDO] Sent EF pose result: {result}")

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

    def zapi_move_manipulator(self, kwargs=None):
        """Handle manipulator joint move (보간 이동) request."""
        self.__console.info(f"Received zapi_move_manipulator with kwargs: {kwargs}")
        if not kwargs:
            return
        request_payload = {
            "command": "move_manipulator",
            "robot": kwargs.get("robot"),
            "joint": kwargs.get("joint"),
            "target": kwargs.get("target", 0.0),
            "speed": kwargs.get("speed", 1.0),
            "accel": kwargs.get("accel"),
            "_identity": kwargs.get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_stop_manipulator(self, kwargs=None):
        """Handle manipulator stop request."""
        self.__console.info(f"Received zapi_stop_manipulator with kwargs: {kwargs}")
        request_payload = {
            "command": "stop_manipulator",
            "robot": (kwargs or {}).get("robot"),
            "joint": (kwargs or {}).get("joint"),
            "_identity": (kwargs or {}).get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_reset_robot_base_pose(self, kwargs=None):
        """Reset collaborative robot joints to their zero/base pose."""
        self.__console.info(f"Received zapi_reset_robot_base_pose with kwargs: {kwargs}")
        request_payload = {
            "command": "reset_robot_base_pose",
            "robot": (kwargs or {}).get("robot"),
            "_identity": (kwargs or {}).get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

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

    def zapi_set_spool_fixation(self, kwargs=None):
        """Handle spool fixation flag changes without moving positioner joints."""
        kwargs = kwargs or {}
        self.__console.info(f"Received zapi_set_spool_fixation with kwargs: {kwargs}")
        request_payload = {
            "command": "set_spool_fixation",
            "fix_f_column_r": bool(kwargs.get("fix_f_column_r", False)),
            "fix_m_column_z": bool(kwargs.get("fix_m_column_z", False)),
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

    def zapi_pick_inspection_point(self, kwargs=None):
        """Handle pipe inspection point pick mode request."""
        self.__console.info(f"Received zapi_pick_inspection_point with kwargs: {kwargs}")
        request_payload = {
            "command": "pick_inspection_point",
            "enabled": (kwargs or {}).get("enabled", True),
            "_identity": (kwargs or {}).get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_plan_inspection_path(self, kwargs=None):
        """Handle EF-only path planning request to the picked inspection point."""
        self.__console.info(f"Received zapi_plan_inspection_path with kwargs: {kwargs}")
        request_payload = {
            "command": "plan_inspection_path",
            "planner": (kwargs or {}).get("planner", "rrt_connect"),
            "robot": (kwargs or {}).get("robot", "rb20_1900es"),
            "step_size": (kwargs or {}).get("step_size", 0.08),
            "max_iter": (kwargs or {}).get("max_iter", 3000),
            "ik_solver": (kwargs or {}).get("ik_solver", "normalized_dls"),
            "ik_normalize": (kwargs or {}).get("ik_normalize", True),
            "_identity": (kwargs or {}).get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_plan_ef_pose_paths(self, kwargs=None):
        """Handle simultaneous path planning to the determined EF poses."""
        self.__console.info(f"Received zapi_plan_ef_pose_paths with kwargs: {kwargs}")
        request_payload = {
            "command": "plan_ef_pose_paths",
            "planner": (kwargs or {}).get("planner", "rrt_connect"),
            "step_size": (kwargs or {}).get("step_size", 0.08),
            "max_iter": (kwargs or {}).get("max_iter", 3000),
            "max_workers": (kwargs or {}).get("max_workers", 2),
            "ik_solver": (kwargs or {}).get("ik_solver", "normalized_dls"),
            "ik_normalize": (kwargs or {}).get("ik_normalize", True),
            "_identity": (kwargs or {}).get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_check_ef_pose_ik(self, kwargs=None):
        """Handle IK-only validation request for the determined EF poses."""
        self.__console.info(f"Received zapi_check_ef_pose_ik with kwargs: {kwargs}")
        request_payload = {
            "command": "check_ef_pose_ik",
            "planner": (kwargs or {}).get("planner", "rrt_connect"),
            "step_size": (kwargs or {}).get("step_size", 0.08),
            "max_iter": (kwargs or {}).get("max_iter", 3000),
            "max_workers": (kwargs or {}).get("max_workers", 2),
            "ik_solver": (kwargs or {}).get("ik_solver", "normalized_dls"),
            "ik_normalize": (kwargs or {}).get("ik_normalize", True),
            "_identity": (kwargs or {}).get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_pick_chuck_mount_points(self, kwargs=None):
        """Handle two-point chuck mount pick mode request."""
        self.__console.info(f"Received zapi_pick_chuck_mount_points with kwargs: {kwargs}")
        request_payload = {
            "command": "pick_chuck_mount_points",
            "enabled": (kwargs or {}).get("enabled", True),
            "clear": (kwargs or {}).get("clear", True),
            "align_on_pick": (kwargs or {}).get("align_on_pick", False),
            "align_target": (kwargs or {}).get("align_target", "f"),
            "_identity": (kwargs or {}).get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_set_chuck_mount_points(self, kwargs=None):
        """Render previously stored chuck mount points in the viewer."""
        self.__console.info(f"Received zapi_set_chuck_mount_points with kwargs: {kwargs}")
        request_payload = {
            "command": "set_chuck_mount_points",
            "points": (kwargs or {}).get("points", []),
            "local_points": (kwargs or {}).get("local_points"),
            "_identity": (kwargs or {}).get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_set_chuck_mount_config(self, kwargs=None):
        """Update chuck mount center offset/axis config in the viewer."""
        self.__console.info(f"Received zapi_set_chuck_mount_config with kwargs: {kwargs}")
        request_payload = {
            "command": "set_chuck_mount_config",
            "chuck_mount": (kwargs or {}).get("chuck_mount", {}),
            "_identity": (kwargs or {}).get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_clear_chuck_mount_points(self, kwargs=None):
        """Clear rendered chuck mount points in the viewer."""
        self.__console.info(f"Received zapi_clear_chuck_mount_points with kwargs: {kwargs}")
        request_payload = {
            "command": "clear_chuck_mount_points",
            "_identity": (kwargs or {}).get("_identity") if kwargs else None,
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_determine_ef_pose(self, kwargs=None):
        """Handle EF pose determination request for the picked inspection point."""
        self.__console.info(f"Received zapi_determine_ef_pose with kwargs: {kwargs}")
        request_payload = {
            "command": "determine_ef_pose",
            "_identity": (kwargs or {}).get("_identity"),
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_clear_inspection_path(self, kwargs=None):
        """Handle clearing picked point/path visualization."""
        self.__console.info(f"Received zapi_clear_inspection_path with kwargs: {kwargs}")
        request_payload = {
            "command": "clear_inspection_path",
            "_identity": (kwargs or {}).get("_identity") if kwargs else None,
        }
        if self._visualizer:
            self._visualizer.push_request(request_payload)
        else:
            self.push_to_queue(request_payload)

    def zapi_execute_inspection_path(self, kwargs=None):
        """Handle simulation playback request for the last planned EF path."""
        self.__console.info(f"Received zapi_execute_inspection_path with kwargs: {kwargs}")
        request_payload = {
            "command": "execute_inspection_path",
            "speed": (kwargs or {}).get("speed", 0.2),
            "_identity": (kwargs or {}).get("_identity") if kwargs else None,
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
