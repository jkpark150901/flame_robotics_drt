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

    def _ZAPI_request_set_spool_fixation(self, fix_f_column_r: bool = False, fix_m_column_z: bool = False):
        """Notify viewer that spool fixation flags changed without moving the positioner."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {
                "fix_f_column_r": bool(fix_f_column_r),
                "fix_m_column_z": bool(fix_m_column_z),
            }
            self.call(self.__dealer_socket, "zapi_set_spool_fixation", kwargs)
            self.__console.info(
                f"[ZAPI] Sent set_spool_fixation: fix_f={fix_f_column_r} fix_z={fix_m_column_z}")
        else:
            self.__console.warning("[ZAPI] Cannot send set_spool_fixation: Socket not connected")

    def _ZAPI_request_filter_spool(self, method: str, params: dict = None):
        """현재 로드된 스풀에 직접 노이즈 필터를 적용. method: 'sor'|'ccl'"""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {"method": method, "params": params or {}}
            self.call(self.__dealer_socket, "zapi_filter_spool", kwargs)
            self.__console.info(f"[ZAPI] Sent filter_spool request: method={method} params={params}")
        else:
            self.__console.warning("[ZAPI] Cannot send filter_spool: Socket not connected")

## mesh reconstruction 현재 미사용
##region mesh reconstruction
    def _ZAPI_request_reconstruct_mesh(self, params: dict = None):
        """현재 로드된 스풀로 메시 재건 (Marching Cubes)."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {"params": params or {}}
            self.call(self.__dealer_socket, "zapi_reconstruct_mesh", kwargs)
            self.__console.info(f"[ZAPI] Sent reconstruct_mesh request: params={params}")
        else:
            self.__console.warning("[ZAPI] Cannot send reconstruct_mesh: Socket not connected")
##endregion

    def _ZAPI_request_save_spool(self, file_path: str):
        """현재 로드된 스풀(또는 재건 메시)을 파일로 저장."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_save_spool", {"path": file_path})
            self.__console.info(f"[ZAPI] Sent save_spool request: {file_path}")
        else:
            self.__console.warning("[ZAPI] Cannot send save_spool: Socket not connected")

    def _ZAPI_request_move_manipulator(self, robot: str, joint: str,
                                       target: float, speed: float = 1.0, accel: float = None):
        """매니퓰레이터 조인트를 target으로 사다리꼴 프로파일 이동(가감속). 예: 레일 베이스."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {"robot": robot, "joint": joint, "target": target, "speed": speed, "accel": accel}
            self.call(self.__dealer_socket, "zapi_move_manipulator", kwargs)
            self.__console.info(f"[ZAPI] Sent move_manipulator: {robot}.{joint} → {target} (speed={speed})")
        else:
            self.__console.warning("[ZAPI] Cannot send move_manipulator: Socket not connected")

    def _ZAPI_request_stop_manipulator(self, robot: str, joint: str = None):
        """매니퓰레이터 조인트 애니메이션 중지."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_stop_manipulator", {"robot": robot, "joint": joint})
            self.__console.info(f"[ZAPI] Sent stop_manipulator: {robot} {joint}")
        else:
            self.__console.warning("[ZAPI] Cannot send stop_manipulator: Socket not connected")

    def _ZAPI_request_pick_inspection_point(self, enabled: bool = True):
        """Enable/disable one-click pipe inspection point picking in the viewer."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_pick_inspection_point", {"enabled": bool(enabled)})
            self.__console.info(f"[ZAPI] Sent pick_inspection_point: enabled={enabled}")
        else:
            self.__console.warning("[ZAPI] Cannot send pick_inspection_point: Socket not connected")

    def _ZAPI_request_pick_chuck_mount_points(self, enabled: bool = True, clear: bool = True):
        """Enable/disable two-click pipe chuck mount point picking in the viewer."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {"enabled": bool(enabled), "clear": bool(clear)}
            self.call(self.__dealer_socket, "zapi_pick_chuck_mount_points", kwargs)
            self.__console.info(f"[ZAPI] Sent pick_chuck_mount_points: {kwargs}")
        else:
            self.__console.warning("[ZAPI] Cannot send pick_chuck_mount_points: Socket not connected")

    def _ZAPI_request_set_chuck_mount_points(self, points, local_points=None):
        """Render stored chuck mount points in the viewer."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {"points": points or []}
            if local_points is not None:
                kwargs["local_points"] = local_points
            self.call(self.__dealer_socket, "zapi_set_chuck_mount_points", kwargs)
            self.__console.info(f"[ZAPI] Sent set_chuck_mount_points: {kwargs}")
        else:
            self.__console.warning("[ZAPI] Cannot send set_chuck_mount_points: Socket not connected")

    def _ZAPI_request_clear_chuck_mount_points(self):
        """Clear rendered chuck mount points in the viewer."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_clear_chuck_mount_points", {})
            self.__console.info("[ZAPI] Sent clear_chuck_mount_points")
        else:
            self.__console.warning("[ZAPI] Cannot send clear_chuck_mount_points: Socket not connected")

    def _ZAPI_request_plan_inspection_path(self, planner: str, robot: str = "rb20_1900es",
                                           step_size: float = 0.08, max_iter: int = 3000):
        """Request EF-only path planning to the currently picked pipe inspection point."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {
                "planner": planner,
                "robot": robot,
                "step_size": step_size,
                "max_iter": max_iter,
            }
            self.call(self.__dealer_socket, "zapi_plan_inspection_path", kwargs)
            self.__console.info(f"[ZAPI] Sent plan_inspection_path: {kwargs}")
        else:
            self.__console.warning("[ZAPI] Cannot send plan_inspection_path: Socket not connected")

    def _ZAPI_request_determine_ef_pose(self):
        """Request EF pose determination for the currently picked inspection point."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_determine_ef_pose", {})
            self.__console.info("[ZAPI] Sent determine_ef_pose")
        else:
            self.__console.warning("[ZAPI] Cannot send determine_ef_pose: Socket not connected")

    def _ZAPI_request_clear_inspection_path(self):
        """Clear picked inspection point and planned path visuals."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            self.call(self.__dealer_socket, "zapi_clear_inspection_path", {})
            self.__console.info("[ZAPI] Sent clear_inspection_path")
        else:
            self.__console.warning("[ZAPI] Cannot send clear_inspection_path: Socket not connected")

    def _ZAPI_request_execute_inspection_path(self, speed: float = 0.2):
        """Start viewer-side simulation playback for the last planned EF path."""
        if self.__dealer_socket and self.__dealer_socket.is_joined:
            kwargs = {"speed": float(speed)}
            self.call(self.__dealer_socket, "zapi_execute_inspection_path", kwargs)
            self.__console.info(f"[ZAPI] Sent execute_inspection_path: {kwargs}")
        else:
            self.__console.warning("[ZAPI] Cannot send execute_inspection_path: Socket not connected")

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
