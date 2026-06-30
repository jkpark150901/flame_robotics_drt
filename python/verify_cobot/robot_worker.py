'''
RobotWorker — rbpodo asyncio 통신 스레드
'''

import asyncio
import logging
import threading

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    raise ImportError("PyQt6 is required.")

import rbpodo as rb

log = logging.getLogger(__name__)


class RobotWorker(QThread):
    '''
    asyncio 이벤트 루프를 별도 QThread에서 실행.

    Signals:
        connected()          - 연결 성공
        disconnected()       - 연결 해제
        error(str)           - 에러 메시지
        tcp_updated(list)    - TCP [x,y,z,rx,ry,rz] mm/deg (10 Hz)
        joints_updated(list) - 관절 [J1..J6] deg (10 Hz)
    '''

    connected    = pyqtSignal()
    disconnected = pyqtSignal()
    error        = pyqtSignal(str)
    tcp_updated     = pyqtSignal(list)
    joints_updated  = pyqtSignal(list)

    _POLL_INTERVAL = 0.1  # 10 Hz

    def __init__(self, robot_ip: str, parent=None):
        super().__init__(parent)
        self._robot_ip = robot_ip
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ──────────────────────────────────────────────
    # Public API (UI 스레드에서 호출)
    # ──────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ──────────────────────────────────────────────
    # QThread entry point
    # ──────────────────────────────────────────────

    def run(self):
        self._stop_event.clear()
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            log.error("RobotWorker loop exception: %s", e)
            self.error.emit(str(e))
        finally:
            self._loop.close()
            self._loop = None
            self.disconnected.emit()
            log.info("RobotWorker stopped.")

    # ──────────────────────────────────────────────
    # Async main
    # ──────────────────────────────────────────────

    async def _main(self):
        log.info("Robot: connecting to %s …", self._robot_ip)
        try:
            robot = rb.asyncio.Cobot(self._robot_ip)
            data_ch = rb.asyncio.CobotData(self._robot_ip)
            rc = rb.ResponseCollector()

            await robot.set_operation_mode(rc, rb.OperationMode.Real)
            await robot.flush(rc)
            rc.error().throw_if_not_empty()

        except Exception as e:
            log.error("Robot connect failed: %s", e)
            self.error.emit(f"Robot connect failed: {e}")
            return

        log.info("Robot: connected.")
        self.connected.emit()

        try:
            while not self._stop_event.is_set():
                try:
                    data = await asyncio.wait_for(data_ch.request_data(), timeout=2.0)
                    tcp    = list(data.sdata.tcp_ref)    # [x,y,z,rx,ry,rz]
                    joints = list(data.sdata.jnt_ref)   # [J1..J6]
                    self.tcp_updated.emit(tcp)
                    self.joints_updated.emit(joints)
                except asyncio.TimeoutError:
                    log.warning("Robot data timeout.")
                except Exception as e:
                    log.warning("Robot poll error: %s", e)

                await asyncio.sleep(self._POLL_INTERVAL)
        finally:
            try:
                await robot.disconnect(rc)
            except Exception:
                pass
            log.info("Robot: disconnected.")
