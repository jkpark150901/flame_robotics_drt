'''
Calibration & Verification Tool — Main Window
'''

try:
    from PyQt6.QtGui import QCloseEvent, QFontDatabase, QFont
    from PyQt6.QtWidgets import (
        QMainWindow, QApplication, QMessageBox, QFileDialog,
    )
    from PyQt6.uic import loadUi
    from PyQt6.QtCore import QTimer
except ImportError:
    raise ImportError("PyQt6 is required.")

import os
import pathlib
import json
import logging

import matplotlib.pyplot as plt

from verifytool.workers.robot_worker import RobotWorker
from verifytool.workers.natnet_worker import NatNetWorker
from verifytool.calib_runner import CalibRunner, NatNetStateProxy
from verifytool.verify_runner import VerifyRunner
from verifytool.sync_runner import SyncRunner
from verifytool.blend_test_runner import (
    BlendTestRunner, DEFAULT_JB2_DELTAS, DEFAULT_PB_DELTAS,
    parse_trajectory_csv, compute_plan_deviation,
)

log = logging.getLogger(__name__)


class VerifyCobotWindow(QMainWindow):
    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._calib_result: dict | None = None
        self._robot_worker: RobotWorker | None = None
        self._natnet_worker: NatNetWorker | None = None
        self._last_rb_pos: list | None = None
        self._last_rb_quat: list | None = None
        self._rb_origin: list | None = None
        self._natnet_state = NatNetStateProxy()
        self._calib_runner: CalibRunner | None = None
        self._scatter_rb = None
        self._scatter_tcp = None
        self._line_rb = None
        self._line_tcp = None
        self._verify_runner: VerifyRunner | None = None
        self._verify_result: dict | None = None
        self._sync_runner:  SyncRunner  | None = None
        self._sync_records: list = []
        # sync 실시간 플롯용 누적 버퍼
        self._sync_elapsed: list = []
        self._sync_errors:  list = []
        # blend test
        self._blend_runner:    BlendTestRunner | None = None
        self._blend_results:   dict = {}   # {blending_value: records}
        self._blend_csv_data:  list | None = None  # [{'joints','speed','accel'}]

        ui_path = pathlib.Path(config['app_path']) / config['gui']
        if not ui_path.is_file():
            raise FileNotFoundError(f"UI file not found: {ui_path}")
        loadUi(ui_path, self)
        self.setWindowTitle(config.get('window_title', 'Calibration & Verification Tool'))

        self._apply_config_defaults()
        self._inject_plot_widgets()
        self._connect_signals()
        self._start_monitor_timer()

        log.info("VerifyCobotWindow initialized.")

    # ──────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────

    def _apply_config_defaults(self):
        cfg = self._config
        if hasattr(self, 'edit_robot_ip'):
            self.edit_robot_ip.setText(cfg.get('robot_ip', '10.0.2.7'))
        if hasattr(self, 'edit_natnet_server'):
            self.edit_natnet_server.setText(cfg.get('natnet_server_ip', '192.168.0.241'))
        if hasattr(self, 'edit_natnet_client'):
            self.edit_natnet_client.setText(cfg.get('natnet_client_ip', 'auto'))
        if hasattr(self, 'spin_rb_id'):
            self.spin_rb_id.setValue(int(cfg.get('natnet_rigid_body_id', 1)))
        if hasattr(self, 'spin_speed'):
            self.spin_speed.setValue(float(cfg.get('speed', 400)))
        if hasattr(self, 'spin_accel'):
            self.spin_accel.setValue(float(cfg.get('accel', 200)))
        if hasattr(self, 'spin_speed_bar'):
            self.spin_speed_bar.setValue(float(cfg.get('speed_bar', 0.3)))
        if hasattr(self, 'spin_calib_speed'):
            self.spin_calib_speed.setValue(float(cfg.get('speed', 400)))
        if hasattr(self, 'spin_calib_accel'):
            self.spin_calib_accel.setValue(float(cfg.get('accel', 200)))
        if hasattr(self, 'spin_settle_time'):
            self.spin_settle_time.setValue(float(cfg.get('settle_time', 0.5)))
        if hasattr(self, 'spin_sample_count'):
            self.spin_sample_count.setValue(int(cfg.get('sample_count', 5)))
        if hasattr(self, 'spin_outlier_mm'):
            self.spin_outlier_mm.setValue(float(cfg.get('outlier_threshold_mm', 60.0)))
        if hasattr(self, 'cbx_handeye_method'):
            method = cfg.get('opencv_handeye_method', 'park')
            idx = self.cbx_handeye_method.findText(method)
            if idx >= 0:
                self.cbx_handeye_method.setCurrentIndex(idx)

    def _inject_plot_widgets(self):
        """Calibration 3D → matplotlib,  Verification/Sync → pyqtgraph 2D."""
        import numpy as np
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from PyQt6.QtWidgets import QSplitter
        from PyQt6.QtCore import Qt

        self._pts_rb:  list = []
        self._pts_tcp: list = []

        # ── Calibration: matplotlib 3D (GLX 불필요) ────────────────
        self._calib_fig = Figure(tight_layout=True)
        self._calib_fig.patch.set_facecolor('#1a1a2e')
        self._calib_canvas = FigureCanvasQTAgg(self._calib_fig)
        self.widget_calib_plot.layout().addWidget(self._calib_canvas)
        self._redraw_calib_plot()

        # ── Verification / Sync: pyqtgraph 2D ─────────────────────
        try:
            import pyqtgraph as pg

            self._verify_plot_pos = None
            self._verify_plot_rot = None

            # Verification: matplotlib 3D 궤적만 표시
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self._verify_pose_fig = Figure(tight_layout=True)
            self._verify_pose_fig.patch.set_facecolor('#1a1a2e')
            self._verify_pose_canvas = FigureCanvasQTAgg(self._verify_pose_fig)
            self.widget_verify_plot.layout().addWidget(self._verify_pose_canvas)
            self._redraw_verify_poses()

            self._inject_sync_tab(pg)
            log.info("Plot widgets injected (matplotlib 3D + pyqtgraph 2D).")
        except Exception as e:
            log.warning("pyqtgraph unavailable — verification/sync plots skipped: %s", e)
            self._verify_plot_pos = None
            self._verify_plot_rot = None
        self._inject_blend_tab()

    def _connect_signals(self):
        # Tab 1 – Connection
        self.btn_robot_connect.clicked.connect(self._on_robot_connect)
        self.btn_robot_disconnect.clicked.connect(self._on_robot_disconnect)
        self.btn_natnet_connect.clicked.connect(self._on_natnet_connect)
        self.btn_natnet_disconnect.clicked.connect(self._on_natnet_disconnect)

        # Tab 2 – Calibration
        self.btn_browse_calib_csv.clicked.connect(self._on_browse_calib_csv)
        self.btn_run_calib.clicked.connect(self._on_run_calib)
        self.btn_stop_calib.clicked.connect(self._on_stop_calib)
        self.btn_save_calib.clicked.connect(self._on_save_calib)
        self.btn_load_calib.clicked.connect(self._on_load_calib)

        # Tab 3 – Verification
        self.btn_browse_verify_csv.clicked.connect(self._on_browse_verify_csv)
        self.btn_verify_load_calib.clicked.connect(self._on_verify_load_calib)
        self.btn_run_verify.clicked.connect(self._on_run_verify)
        self.btn_stop_verify.clicked.connect(self._on_stop_verify)
        self.btn_save_verify_csv.clicked.connect(self._on_save_verify_csv)

    def _start_monitor_timer(self):
        self._monitor_timer = QTimer(self)
        self._monitor_timer.timeout.connect(self._update_monitor)
        self._monitor_timer.start(100)  # 10 Hz

    # ──────────────────────────────────────────────
    # Monitor update (called every 100 ms)
    # ──────────────────────────────────────────────

    def _update_monitor(self):
        # Populated by workers via signals once they are implemented.
        pass

    # ──────────────────────────────────────────────
    # Tab 1 handlers (stubs — workers wired in later)
    # ──────────────────────────────────────────────

    def _on_robot_connect(self):
        ip = self.edit_robot_ip.text().strip()
        if not ip:
            self._log("[ERROR] Robot IP is empty.")
            return
        if self._robot_worker and self._robot_worker.isRunning():
            return

        self._robot_worker = RobotWorker(ip, parent=self)
        self._robot_worker.connected.connect(self._on_robot_connected)
        self._robot_worker.disconnected.connect(self._on_robot_disconnected)
        self._robot_worker.error.connect(lambda msg: self._log(f"[Robot ERROR] {msg}"))
        self._robot_worker.tcp_updated.connect(self._on_tcp_updated)
        self._robot_worker.joints_updated.connect(self._on_joints_updated)

        self.btn_robot_connect.setEnabled(False)
        self.btn_robot_disconnect.setEnabled(True)
        self._set_robot_status("connecting", "#f0a500")
        self._log(f"Robot: connecting to {ip} …")
        self._robot_worker.start()

    def _on_robot_disconnect(self):
        if self._robot_worker:
            self._robot_worker.stop()
        self.btn_robot_disconnect.setEnabled(False)

    def _on_robot_connected(self):
        self._set_robot_status("connected", "#00c853")
        self.btn_robot_connect.setEnabled(False)
        self.btn_robot_disconnect.setEnabled(True)
        self._log("Robot: connected.")

    def _on_robot_disconnected(self):
        self._set_robot_status("disconnected", "#e53935")
        self.btn_robot_connect.setEnabled(True)
        self.btn_robot_disconnect.setEnabled(False)
        self._log("Robot: disconnected.")

    def _set_robot_status(self, text: str, color: str):
        style = f"color: {color}; font-weight: bold;"
        if hasattr(self, 'lbl_robot_status'):
            self.lbl_robot_status.setText(text)
            self.lbl_robot_status.setStyleSheet(style)
        if hasattr(self, 'lbl_hdr_robot_status'):
            self.lbl_hdr_robot_status.setText(f"● Robot: {text}")
            self.lbl_hdr_robot_status.setStyleSheet(style)

    def _on_tcp_updated(self, tcp: list):
        if hasattr(self, 'lbl_tcp_pos') and len(tcp) >= 3:
            self.lbl_tcp_pos.setText(f"X:{tcp[0]:.1f}  Y:{tcp[1]:.1f}  Z:{tcp[2]:.1f} mm")
        if hasattr(self, 'lbl_tcp_rot') and len(tcp) >= 6:
            self.lbl_tcp_rot.setText(f"Rx:{tcp[3]:.2f}  Ry:{tcp[4]:.2f}  Rz:{tcp[5]:.2f} °")
        if hasattr(self, 'lbl_hdr_tcp') and len(tcp) >= 3:
            self.lbl_hdr_tcp.setText(f"TCP: {tcp[0]:.0f}, {tcp[1]:.0f}, {tcp[2]:.0f} mm")

    def _on_joints_updated(self, joints: list):
        if hasattr(self, 'lbl_joints') and len(joints) >= 6:
            self.lbl_joints.setText(
                f"J1:{joints[0]:.1f}  J2:{joints[1]:.1f}  J3:{joints[2]:.1f}  "
                f"J4:{joints[3]:.1f}  J5:{joints[4]:.1f}  J6:{joints[5]:.1f} °"
            )

    def _on_natnet_connect(self):
        server_ip = self.edit_natnet_server.text().strip()
        client_ip = self.edit_natnet_client.text().strip() or 'auto'
        rb_id     = int(self.spin_rb_id.value())
        if not server_ip:
            self._log("[ERROR] NatNet server IP is empty.")
            return
        if self._natnet_worker and self._natnet_worker.isRunning():
            return

        force_version = self._config.get('natnet_force_version')  # e.g. [4, 0]
        fv = tuple(force_version) if force_version else None

        self._natnet_worker = NatNetWorker(
            server_ip=server_ip, client_ip=client_ip,
            rigid_body_id=rb_id, force_version=fv, parent=self,
        )
        self._natnet_worker.connected.connect(self._on_natnet_connected)
        self._natnet_worker.disconnected.connect(self._on_natnet_disconnected)
        self._natnet_worker.error.connect(lambda msg: self._log(f"[NatNet ERROR] {msg}"))
        self._natnet_worker.rb_updated.connect(self._on_rb_updated)
        # raw_rb_listener: Qt 시그널 쓰로틀(30Hz) 없이 모든 프레임을 state proxy에 전달
        self._natnet_worker.raw_rb_listener = (
            lambda rb_id, pos, quat, ts: self._natnet_state.update(pos, quat)
        )
        self._natnet_worker.fps_updated.connect(self._on_natnet_fps)

        self.btn_natnet_connect.setEnabled(False)
        self.btn_natnet_disconnect.setEnabled(True)
        self._set_natnet_status("connecting", "#f0a500")
        self._log(f"NatNet: connecting to {server_ip} (rb_id={rb_id}) …")
        self._natnet_worker.start()

    def _on_natnet_disconnect(self):
        if self._natnet_worker:
            self._natnet_worker.stop()
        self.btn_natnet_disconnect.setEnabled(False)

    def _on_natnet_connected(self):
        self._set_natnet_status("connected", "#00c853")
        self.btn_natnet_connect.setEnabled(False)
        self.btn_natnet_disconnect.setEnabled(True)
        self._log("NatNet: connected.")

    def _on_natnet_disconnected(self):
        self._set_natnet_status("disconnected", "#e53935")
        self.btn_natnet_connect.setEnabled(True)
        self.btn_natnet_disconnect.setEnabled(False)
        self._rb_origin = None
        self._log("NatNet: disconnected.")

    def _set_natnet_status(self, text: str, color: str):
        style = f"color: {color}; font-weight: bold;"
        if hasattr(self, 'lbl_natnet_status'):
            self.lbl_natnet_status.setText(text)
            self.lbl_natnet_status.setStyleSheet(style)
        if hasattr(self, 'lbl_hdr_natnet_status'):
            self.lbl_hdr_natnet_status.setText(f"● NatNet: {text}")
            self.lbl_hdr_natnet_status.setStyleSheet(style)

    def _on_rb_updated(self, rb_id: int, pos: list, quat: list):
        self._last_rb_pos  = pos
        self._last_rb_quat = quat
        if self._rb_origin is None:
            self._rb_origin = list(pos)
        if hasattr(self, 'lbl_rb_pos'):
            self.lbl_rb_pos.setText(
                f"X:{pos[0]*1000:.1f}  Y:{pos[1]*1000:.1f}  Z:{pos[2]*1000:.1f} mm")
        if hasattr(self, 'lbl_rb_quat'):
            self.lbl_rb_quat.setText(
                f"qx:{quat[0]:.3f}  qy:{quat[1]:.3f}  qz:{quat[2]:.3f}  qw:{quat[3]:.3f}")
        if hasattr(self, 'lbl_rb_tracking'):
            self.lbl_rb_tracking.setText("tracking")
            self.lbl_rb_tracking.setStyleSheet("color: #00c853;")
        if hasattr(self, 'lbl_hdr_rb_pos'):
            self.lbl_hdr_rb_pos.setText(
                f"RB: {pos[0]*1000:.0f}, {pos[1]*1000:.0f}, {pos[2]*1000:.0f} mm")
        if hasattr(self, 'lbl_rb_delta') and self._rb_origin is not None:
            dx = (pos[0] - self._rb_origin[0]) * 1000
            dy = (pos[1] - self._rb_origin[1]) * 1000
            dz = (pos[2] - self._rb_origin[2]) * 1000
            self.lbl_rb_delta.setText(
                f"dx:{dx:+.1f}  dy:{dy:+.1f}  dz:{dz:+.1f} mm")
        if hasattr(self, 'lbl_rb_r2_dist') and self._rb_origin is not None:
            dx = (pos[0] - self._rb_origin[0]) * 1000
            dy = (pos[1] - self._rb_origin[1]) * 1000
            dz = (pos[2] - self._rb_origin[2]) * 1000
            r = (dx**2 + dy**2 + dz**2) ** 0.5
            self.lbl_rb_r2_dist.setText(f"{r:.2f} mm")

    def _on_natnet_fps(self, fps: float):
        if hasattr(self, 'lbl_natnet_fps'):
            self.lbl_natnet_fps.setText(f"{fps:.0f} Hz")
        if hasattr(self, 'lbl_hdr_natnet_fps'):
            self.lbl_hdr_natnet_fps.setText(f"{fps:.0f} Hz")

    # ──────────────────────────────────────────────
    # Tab 2 handlers
    # ──────────────────────────────────────────────

    def _on_browse_calib_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Calibration Plan CSV",
                                              str(self._config.get('root_path', '.')),
                                              "CSV Files (*.csv)")
        if path:
            self.edit_calib_csv.setText(path)
            self._load_calib_plan_preview(path)

    def _load_calib_plan_preview(self, path: str):
        try:
            import csv
            with open(path, newline='', encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.lbl_pose_count.setText(f"Poses: {len(rows)}")
            self.progress_calib.setMaximum(len(rows))
            self._log(f"Calibration plan loaded: {len(rows)} poses  ←  {os.path.basename(path)}")
        except Exception as e:
            self._log(f"[ERROR] Could not read CSV: {e}")

    def _on_run_calib(self):
        # ── 전제 조건 검사 ──────────────────────────
        csv_path = self.edit_calib_csv.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "CSV 없음", "캘리브레이션 플랜 CSV를 먼저 선택하세요.")
            return
        import pathlib
        if not pathlib.Path(csv_path).exists():
            QMessageBox.warning(self, "파일 없음", f"CSV 파일을 찾을 수 없습니다:\n{csv_path}")
            return

        robot_ip = self.edit_robot_ip.text().strip()
        if not robot_ip:
            QMessageBox.warning(self, "IP 없음", "Robot IP를 입력하세요.")
            return

        if self._natnet_worker is None or not self._natnet_worker.isRunning():
            QMessageBox.warning(self, "NatNet 미연결", "NatNet에 먼저 연결하세요.")
            return

        if self._calib_runner and self._calib_runner.isRunning():
            return

        # ── handeye method ──────────────────────────
        from calibration.solver import OPENCV_HANDEYE_METHODS
        method_str = self.cbx_handeye_method.currentText()
        import cv2
        method_int = OPENCV_HANDEYE_METHODS.get(method_str, cv2.CALIB_HAND_EYE_PARK)

        params = {
            'robot_ip':              robot_ip,
            'csv_path':              csv_path,
            'speed':                 float(self.spin_calib_speed.value()),
            'accel':                 float(self.spin_calib_accel.value()),
            'speed_bar':             float(self.spin_speed_bar.value()),
            'settle_time':           float(self.spin_settle_time.value()),
            'sample_count':          int(self.spin_sample_count.value()),
            'min_mocap_samples':     int(self._config.get('min_mocap_samples', 3)),
            'mocap_timeout':         float(self._config.get('mocap_timeout', 2.0)),
            'handeye_method_int':    method_int,
            'outlier_threshold_mm':  float(self.spin_outlier_mm.value()),
            'tcp_orientation_type':  self._config.get('tcp_orientation_type', 'zyx_euler_deg'),
            'cal_file':              self._config.get('cal_file', 'calibration_svd.json'),
        }

        # 실시간 플롯 초기화
        self._pts_rb  = []
        self._pts_tcp = []
        if self._scatter_rb is not None:
            import numpy as np
            empty = np.zeros((1, 3), dtype=float)
            self._scatter_rb.setData(pos=empty,  color=(0.2, 0.8, 1.0, 0.0))
            self._scatter_tcp.setData(pos=empty, color=(0.2, 1.0, 0.4, 0.0))
            self._line_rb.setData(pos=empty,  color=(0.4, 0.7, 1.0, 0.0))
            self._line_tcp.setData(pos=empty, color=(0.2, 1.0, 0.5, 0.0))

        self._calib_runner = CalibRunner(params, self._natnet_state, parent=self)
        self._calib_runner.pose_started.connect(self._on_calib_pose_started)
        self._calib_runner.pose_done.connect(self._on_calib_pose_done)
        self._calib_runner.point_collected.connect(self._on_calib_point_collected)
        self._calib_runner.all_done.connect(self._on_calib_all_done)
        self._calib_runner.progress.connect(self.progress_calib.setValue)
        self._calib_runner.log_msg.connect(self._log)
        self._calib_runner.error.connect(self._on_calib_error)

        self.btn_run_calib.setEnabled(False)
        self.btn_stop_calib.setEnabled(True)
        self.btn_save_calib.setEnabled(False)
        self.progress_calib.setValue(0)
        self._log(f"캘리브레이션 시작 ({method_str}) …")
        self._calib_runner.start()

    def _on_stop_calib(self):
        if self._calib_runner and self._calib_runner.isRunning():
            self._calib_runner.stop()
        self.btn_run_calib.setEnabled(True)
        self.btn_stop_calib.setEnabled(False)
        self._log("캘리브레이션 중단 요청.")

    def _on_calib_point_collected(self, rb_pos: list, tcp_pos_m: list):
        self._pts_rb.append([v * 1000.0 for v in rb_pos])
        self._pts_tcp.append([v * 1000.0 for v in tcp_pos_m])
        # 5포인트마다 갱신 (매번 재렌더링하면 느림)
        if len(self._pts_rb) % 5 == 0:
            self._redraw_calib_plot()

    def _on_calib_pose_started(self, idx: int, total: int, label: str):
        self.progress_calib.setMaximum(total)
        self.lbl_pose_count.setText(f"Pose: {idx} / {total}  ({label})")

    def _on_calib_pose_done(self, idx: int, total: int, ok: bool):
        status = "OK" if ok else "SKIP"
        self._log(f"  └ [{idx:02d}/{total:02d}] {status}")

    def _on_calib_all_done(self, result: dict):
        import numpy as np
        self._calib_result = result
        self.btn_run_calib.setEnabled(True)
        self.btn_stop_calib.setEnabled(False)
        self.btn_save_calib.setEnabled(True)

        T = np.array(result['T_base_motive'])
        t = T[:3, 3]
        self.lbl_calib_t_base_motive.setText(
            f"T_base_motive  t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m  r={_rotvec_str(T[:3,:3])}"
        )
        T2 = np.array(result['T_rb_tcp'])
        t2 = T2[:3, 3]
        self.lbl_calib_t_rb_tcp.setText(
            f"T_rb_tcp  t=[{t2[0]*1000:.2f}, {t2[1]*1000:.2f}, {t2[2]*1000:.2f}] mm  r={_rotvec_str(T2[:3,:3])}"
        )
        self.lbl_calib_rmse.setText(
            f"pos RMSE (inlier): {result['rmse_pos_mm_inlier']:.3f} mm  "
            f"| rot: {result['rmse_rot_deg_inlier']:.3f} °  "
            f"| inlier: {result['n_inlier']}/{result['n_total']}"
        )
        self.lbl_calib_model.setText(f"method: {self.cbx_handeye_method.currentText()}")
        self._log(
            f"캘리브레이션 완료 — pos RMSE {result['rmse_pos_mm_inlier']:.3f} mm "
            f"(inlier {result['n_inlier']}/{result['n_total']})"
        )
        self._update_calib_plot(result)

        # ── 자동 저장 ─────────────────────────────
        cal_path = _resolve_cal_path(self._config)
        try:
            from calibration.solver import save_calibration
            save_calibration(T, str(cal_path), np.array(result['T_rb_tcp']))
            self._log(f"캘리브레이션 자동 저장 → {cal_path}")
        except Exception as e:
            self._log(f"[경고] 자동 저장 실패: {e}")

        # ── 검증 탭 / Sync 탭 자동 업데이트 ────────
        self._apply_calib_to_verify_tab(result, str(cal_path))
        if hasattr(self, 'btn_sync_start'):
            self.btn_sync_start.setEnabled(True)

    def _on_calib_error(self, msg: str):
        self.btn_run_calib.setEnabled(True)
        self.btn_stop_calib.setEnabled(False)
        self._log(f"[캘리브레이션 ERROR] {msg}")
        QMessageBox.critical(self, "캘리브레이션 오류", msg)

    def _update_calib_plot(self, result: dict):
        """캘리브레이션 완료 후 inlier/outlier 색상으로 최종 플롯 갱신."""
        try:
            import numpy as np
            records = result['raw_records']
            inlier  = result['inlier_mask']
            pts_rb  = np.array([[r['rb_raw_x_m'], r['rb_raw_y_m'], r['rb_raw_z_m']]
                                  for r in records]) * 1000.0
            pts_tcp = np.array([[r['tcp_x_mm'], r['tcp_y_mm'], r['tcp_z_mm']]
                                  for r in records])
            self._pts_rb  = pts_rb.tolist()
            self._pts_tcp = pts_tcp.tolist()
            self._calib_inlier = list(inlier)
            self._redraw_calib_plot()
        except Exception as e:
            log.warning("calib plot update failed: %s", e)

    def _redraw_calib_plot(self):
        """matplotlib 3D 캘리브레이션 플롯 전체 재렌더링."""
        if not hasattr(self, '_calib_fig'):
            return
        import numpy as np
        _DARK = '#1a1a2e'
        self._calib_fig.clear()
        ax = self._calib_fig.add_subplot(111, projection='3d')
        ax.set_facecolor(_DARK)
        ax.tick_params(colors='#aaa', labelsize=7)
        for sp in [ax.xaxis, ax.yaxis, ax.zaxis]:
            sp.pane.fill = False
            sp.pane.set_edgecolor('#333')

        inlier = getattr(self, '_calib_inlier', None)

        if self._pts_rb:
            pts_rb  = np.array(self._pts_rb)
            pts_tcp = np.array(self._pts_tcp)

            if inlier is not None:
                c_rb  = ['#3399ff' if ok else '#ff4444' for ok in inlier]
                c_tcp = ['#00cc66' if ok else '#ff4444' for ok in inlier]
            else:
                c_rb  = '#3399ff'
                c_tcp = '#00cc66'

            ax.scatter(pts_rb[:, 0], pts_rb[:, 1], pts_rb[:, 2],
                       c=c_rb,  s=30, label='RB (NatNet)', depthshade=False)
            ax.scatter(pts_tcp[:, 0], pts_tcp[:, 1], pts_tcp[:, 2],
                       c=c_tcp, s=30, marker='^', label='TCP (robot)', depthshade=False)
            ax.plot(pts_rb[:, 0],  pts_rb[:, 1],  pts_rb[:, 2],
                    color='#3399ff', lw=0.8, alpha=0.4)
            ax.plot(pts_tcp[:, 0], pts_tcp[:, 1], pts_tcp[:, 2],
                    color='#00cc66', lw=0.8, alpha=0.4)
            ax.legend(fontsize=8, labelcolor='white',
                      facecolor='#2a2a4e', edgecolor='#444', loc='upper left')

        ax.set_xlabel('X (mm)', color='#aaa', fontsize=8, labelpad=4)
        ax.set_ylabel('Y (mm)', color='#aaa', fontsize=8, labelpad=4)
        ax.set_zlabel('Z (mm)', color='#aaa', fontsize=8, labelpad=4)
        ax.set_title('Calibration Poses  (blue=RB  green=TCP  red=outlier)',
                     color='white', fontsize=9, pad=6)
        self._calib_canvas.draw()

    def _on_save_calib(self):
        if self._calib_result is None:
            return
        default = str(_resolve_cal_path(self._config))
        path, _ = QFileDialog.getSaveFileName(self, "Save Calibration JSON",
                                              default, "JSON Files (*.json)")
        if path:
            import numpy as np
            try:
                from calibration.solver import save_calibration
                T = np.array(self._calib_result['T_base_motive'])
                T_rb_tcp = np.array(self._calib_result['T_rb_tcp']) \
                    if 'T_rb_tcp' in self._calib_result else None
                save_calibration(T, path, T_rb_tcp)
                self._log(f"Calibration saved → {path}")
            except Exception as e:
                self._log(f"[ERROR] 저장 실패: {e}")

    def _redraw_verify_poses(self, result: dict | None = None):
        """Robot TCP 궤적 vs Motive RB 궤적을 3D로 표시. 오차는 숫자 레이블로만."""
        if not hasattr(self, '_verify_pose_fig'):
            return
        import numpy as np
        _DARK = '#1a1a2e'
        self._verify_pose_fig.clear()
        ax = self._verify_pose_fig.add_subplot(111, projection='3d')
        ax.set_facecolor(_DARK)
        ax.tick_params(colors='#aaa', labelsize=7)
        for sp in [ax.xaxis, ax.yaxis, ax.zaxis]:
            sp.pane.fill = False
            sp.pane.set_edgecolor('#333')

        if result is not None:
            tcp  = np.array(result['tcp_positions']) * 1000   # m → mm
            pred = np.array(result['rb_predictions']) * 1000

            # Robot 궤적 (파랑)
            ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2],
                    'o-', color='steelblue', lw=1.5, ms=5,
                    label='Robot TCP', alpha=0.9)
            # Motive RB 궤적 (주황)
            ax.plot(pred[:, 0], pred[:, 1], pred[:, 2],
                    '^--', color='tomato', lw=1.5, ms=5,
                    label='Motive RB (aligned)', alpha=0.9)

            ax.legend(fontsize=8, labelcolor='white',
                      facecolor='#2a2a4e', edgecolor='#444', loc='upper left')
            n = len(result['errors_mm'])
            ax.set_title(
                f'Robot TCP  vs  Motive RB    N={n}  '
                f'RMSE={result["rmse_mm"]:.3f} mm',
                color='white', fontsize=9, pad=6)
        else:
            ax.set_title('Robot TCP  vs  Motive RB\n(run verification to display)',
                         color='#555', fontsize=9, pad=6)

        ax.set_xlabel('X (mm)', color='#aaa', fontsize=8, labelpad=4)
        ax.set_ylabel('Y (mm)', color='#aaa', fontsize=8, labelpad=4)
        ax.set_zlabel('Z (mm)', color='#aaa', fontsize=8, labelpad=4)
        self._verify_pose_canvas.draw()

    def _on_load_calib(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Calibration JSON",
                                              str(self._config.get('root_path', '.')),
                                              "JSON Files (*.json)")
        if path:
            self._load_calibration_json(path)

    def _load_calibration_json(self, path: str):
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            self._log(f"[ERROR] 파일을 찾을 수 없습니다: {path}")
            return
        except Exception as e:
            self._log(f"[ERROR] Load calibration failed: {e}")
            return

        import numpy as np
        if 'T_rb_tcp' not in data:
            offset = data.get('rb_to_tcp_offset_m')
            if offset is not None and len(offset) == 3:
                T_rb_tcp = np.eye(4)
                T_rb_tcp[:3, 3] = offset
                data['T_rb_tcp'] = T_rb_tcp.tolist()
            else:
                data['T_rb_tcp'] = np.eye(4).tolist()

        self._calib_result = data
        T = np.array(data['T_base_motive'])
        t = T[:3, 3]
        T2 = np.array(data['T_rb_tcp'])
        t2 = T2[:3, 3]
        self.lbl_calib_t_base_motive.setText(
            f"T_base_motive  t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m  r={_rotvec_str(T[:3,:3])}"
        )
        self.lbl_calib_t_rb_tcp.setText(
            f"T_rb_tcp  t=[{t2[0]*1000:.2f}, {t2[1]*1000:.2f}, {t2[2]*1000:.2f}] mm  r={_rotvec_str(T2[:3,:3])}"
        )
        self.lbl_calib_model.setText(f"Loaded: {os.path.basename(path)}")
        self.btn_save_calib.setEnabled(True)
        self._apply_calib_to_verify_tab(data, os.path.basename(path))
        self._log(f"Calibration loaded ← {path}")

    # ──────────────────────────────────────────────
    # Tab 3 handlers
    # ──────────────────────────────────────────────

    def _on_browse_verify_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Verification Plan CSV",
                                              str(self._config.get('root_path', '.')),
                                              "CSV Files (*.csv)")
        if path:
            self.edit_verify_csv.setText(path)
            try:
                import csv
                with open(path, newline='', encoding='utf-8-sig') as f:
                    rows = list(csv.DictReader(f))
                self.progress_verify.setMaximum(len(rows))
                self._log(f"Verification plan: {len(rows)} poses ← {os.path.basename(path)}")
            except Exception as e:
                self._log(f"[ERROR] {e}")

    def _on_verify_load_calib(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Calibration JSON",
                                              str(self._config.get('root_path', '.')),
                                              "JSON Files (*.json)")
        if path:
            self._load_verification_calib(path)

    def _load_verification_calib(self, path: str):
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            self._log(f"[ERROR] 캘리브레이션 파일을 찾을 수 없습니다: {path}")
            QMessageBox.warning(self, "파일 없음", f"파일을 찾을 수 없습니다:\n{path}")
            return
        except Exception as e:
            self._log(f"[ERROR] {e}")
            return

        # 레거시 SVD 포맷 호환: T_rb_tcp 없으면 rb_to_tcp_offset_m 으로 4x4 생성
        import numpy as np
        if 'T_rb_tcp' not in data:
            offset = data.get('rb_to_tcp_offset_m')
            if offset is not None and len(offset) == 3:
                T_rb_tcp = np.eye(4)
                T_rb_tcp[:3, 3] = offset
                data['T_rb_tcp'] = T_rb_tcp.tolist()
                self._log(f"[정보] T_rb_tcp 없음 → rb_to_tcp_offset_m으로 대체")
            else:
                data['T_rb_tcp'] = np.eye(4).tolist()
                self._log(f"[정보] T_rb_tcp 없음 → Identity로 설정")

        self._calib_result = data
        self._apply_calib_to_verify_tab(data, os.path.basename(path))
        self._log(f"검증용 캘리브레이션 로드 ← {path}")

    def _apply_calib_to_verify_tab(self, data: dict, label: str):
        if hasattr(self, 'btn_sync_start'):
            self.btn_sync_start.setEnabled(True)
        import numpy as np
        T = np.array(data['T_base_motive'])
        t = T[:3, 3]
        self.lbl_verify_calib_file.setText(label)
        self.lbl_verify_calib_file.setStyleSheet('color: #00c853; font-weight: bold;')
        self.lbl_verify_t_base_motive.setText(
            f"t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m  r={_rotvec_str(T[:3,:3])}"
        )

    def _on_run_verify(self):
        if self._calib_result is None:
            QMessageBox.warning(self, "캘리브레이션 없음", "캘리브레이션 JSON을 먼저 불러오세요.")
            return

        csv_path = self.edit_verify_csv.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "CSV 없음", "검증 경로 CSV를 먼저 선택하세요.")
            return
        import pathlib
        if not pathlib.Path(csv_path).exists():
            QMessageBox.warning(self, "파일 없음", f"CSV 파일을 찾을 수 없습니다:\n{csv_path}")
            return

        robot_ip = self.edit_robot_ip.text().strip()
        if not robot_ip:
            QMessageBox.warning(self, "IP 없음", "Robot IP를 입력하세요.")
            return

        if self._natnet_worker is None or not self._natnet_worker.isRunning():
            QMessageBox.warning(self, "NatNet 미연결", "NatNet에 먼저 연결하세요.")
            return

        if self._verify_runner and self._verify_runner.isRunning():
            return

        if 'T_base_motive' not in self._calib_result or 'T_rb_tcp' not in self._calib_result:
            QMessageBox.warning(self, "캘리브레이션 불완전",
                                "T_base_motive 또는 T_rb_tcp가 없습니다.")
            return

        params = {
            'robot_ip':          robot_ip,
            'csv_path':          csv_path,
            'speed':             float(self.spin_calib_speed.value()),
            'accel':             float(self.spin_calib_accel.value()),
            'speed_bar':         float(self.spin_speed_bar.value()),
            'settle_time':       float(self.spin_settle_time.value()),
            'sample_count':      int(self.spin_sample_count.value()),
            'min_mocap_samples': int(self._config.get('min_mocap_samples', 3)),
            'mocap_timeout':     float(self._config.get('mocap_timeout', 2.0)),
            'tcp_orientation_type': self._config.get('tcp_orientation_type', 'zyx_euler_deg'),
        }

        if self._verify_plot_pos:
            self._verify_plot_pos.clear()
        if self._verify_plot_rot:
            self._verify_plot_rot.clear()

        self._verify_result = None
        self.progress_verify.setValue(0)

        self._verify_runner = VerifyRunner(params, self._calib_result,
                                           self._natnet_state, parent=self)
        self._verify_runner.pose_started.connect(self._on_verify_pose_started)
        self._verify_runner.pose_done.connect(self._on_verify_pose_done)
        self._verify_runner.point_verified.connect(self._on_verify_point)
        self._verify_runner.all_done.connect(self._on_verify_all_done)
        self._verify_runner.progress.connect(self.progress_verify.setValue)
        self._verify_runner.log_msg.connect(self._log)
        self._verify_runner.error.connect(self._on_verify_error)

        self.btn_run_verify.setEnabled(False)
        self.btn_stop_verify.setEnabled(True)
        self.btn_save_verify_csv.setEnabled(False)
        self._log("검증 시작 …")
        self._verify_runner.start()

    def _on_stop_verify(self):
        if self._verify_runner and self._verify_runner.isRunning():
            self._verify_runner.stop()
        self.btn_run_verify.setEnabled(True)
        self.btn_stop_verify.setEnabled(False)
        self._log("검증 중단 요청.")

    def _on_verify_pose_started(self, idx: int, total: int, label: str):
        self.progress_verify.setMaximum(total)

    def _on_verify_pose_done(self, idx: int, total: int, pos_err_mm: float, rot_err_deg: float):
        if pos_err_mm >= 0:
            self._log(f"  └ [{idx:02d}/{total:02d}] pos {pos_err_mm:.3f} mm  rot {rot_err_deg:.3f} °")

    def _on_verify_point(self, tcp_pos_m: list, pred_pos_m: list,
                         pos_err_mm: float, rot_err_deg: float):
        pass  # 누적 bar 업데이트는 all_done에서 처리

    def _on_verify_all_done(self, result: dict):
        self._verify_result = result
        self.btn_run_verify.setEnabled(True)
        self.btn_stop_verify.setEnabled(False)
        self.btn_save_verify_csv.setEnabled(True)

        self.lbl_verify_rmse.setText(
            f"Pos RMSE: {result['rmse_mm']:.3f} mm  |  Rot RMSE: {result['rmse_rot_deg']:.3f} °"
        )
        self.lbl_verify_max.setText(
            f"Pos Max: {result['max_mm']:.3f} mm  |  Rot Max: {result['max_rot_deg']:.3f} °"
        )
        self.lbl_verify_mean.setText(
            f"Pos Mean: {result['mean_mm']:.3f} mm  |  Rot Mean: {result['mean_rot_deg']:.3f} °"
        )

        self._update_verify_plot(result)
        self._update_verify_table(result)
        self._redraw_verify_poses(result)
        self._log(
            f"검증 완료 — RMSE {result['rmse_mm']:.3f} mm  "
            f"Max {result['max_mm']:.3f} mm  ({len(result['errors_mm'])}포즈)"
        )

    def _on_verify_error(self, msg: str):
        self.btn_run_verify.setEnabled(True)
        self.btn_stop_verify.setEnabled(False)
        self._log(f"[검증 ERROR] {msg}")
        QMessageBox.critical(self, "검증 오류", msg)

    def _update_verify_plot(self, result: dict):
        if self._verify_plot_pos is None:
            return
        try:
            import pyqtgraph as pg
            import numpy as np

            def _draw_bars(plot_widget, values, rmse, unit_label):
                plot_widget.clear()
                colors = ['#3399ff' if v <= rmse else '#ff7733' for v in values]
                for i, (v, col) in enumerate(zip(values, colors)):
                    bar = pg.BarGraphItem(x=[i], height=[v], width=0.7,
                                         brush=col, pen=pg.mkPen(None))
                    plot_widget.addItem(bar)
                rmse_line = pg.InfiniteLine(
                    pos=rmse, angle=0,
                    pen=pg.mkPen('#00ff88', width=1,
                                 style=pg.QtCore.Qt.PenStyle.DashLine),
                    label=f'RMSE {rmse:.2f}{unit_label}',
                    labelOpts={'color': '#00ff88', 'position': 0.95},
                )
                plot_widget.addItem(rmse_line)
                ax = plot_widget.getAxis('bottom')
                ax.setTicks([[(i, str(i+1)) for i in range(len(values))]])

            _draw_bars(self._verify_plot_pos,
                       result['errors_mm'], result['rmse_mm'], 'mm')
            _draw_bars(self._verify_plot_rot,
                       result['rot_errors_deg'], result['rmse_rot_deg'], '°')
        except Exception as e:
            log.warning("verify plot update failed: %s", e)

    def _update_verify_table(self, result: dict):
        try:
            from PyQt6.QtWidgets import QTableWidgetItem, QHeaderView
            from PyQt6.QtGui import QColor
            import numpy as np
            table    = self.table_verify
            errors   = result['errors_mm']
            rot_errs = result.get('rot_errors_deg', [0.0] * len(errors))
            labels   = result['labels']
            tcp      = result['tcp_positions']
            pred     = result['rb_predictions']
            rmse_pos = result['rmse_mm']
            rmse_rot = result.get('rmse_rot_deg', 0.0)

            table.setColumnCount(6)
            table.setHorizontalHeaderLabels(
                ['#', 'Pose', 'Pos err (mm)', 'Rot err (°)',
                 'TCP pos', 'Pred pos'])
            table.setRowCount(len(errors))

            for row, (label, err, rot, t, p) in enumerate(
                    zip(labels, errors, rot_errs, tcp, pred)):
                table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
                table.setItem(row, 1, QTableWidgetItem(label))

                item_pos = QTableWidgetItem(f"{err:.3f}")
                item_pos.setForeground(
                    QColor('#ff7733') if err > rmse_pos else QColor('#3399ff'))
                table.setItem(row, 2, item_pos)

                item_rot = QTableWidgetItem(f"{rot:.3f}")
                item_rot.setForeground(
                    QColor('#ff7733') if rot > rmse_rot else QColor('#3399ff'))
                table.setItem(row, 3, item_rot)

                table.setItem(row, 4, QTableWidgetItem(
                    f"[{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}] mm"))
                table.setItem(row, 5, QTableWidgetItem(
                    f"[{p[0]*1000:.1f}, {p[1]*1000:.1f}, {p[2]*1000:.1f}] mm"))
            table.resizeColumnsToContents()
        except Exception as e:
            log.warning("verify table update failed: %s", e)

    def _on_save_verify_csv(self):
        if self._verify_result is None:
            return
        import datetime, csv as _csv
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = str(self._config.get('root_path', '.')) + f'/verify_{stamp}.csv'
        path, _ = QFileDialog.getSaveFileName(self, "Save Verification CSV",
                                              default, "CSV Files (*.csv)")
        if not path:
            return
        try:
            records = self._verify_result['raw_records']
            if not records:
                return
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = _csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
            self._log(f"검증 결과 저장 → {path}")
        except Exception as e:
            self._log(f"[ERROR] CSV 저장 실패: {e}")

    # ──────────────────────────────────────────────
    # Tab 4 – Blend Test (blending 비교 궤적 실행)
    # ──────────────────────────────────────────────

    def _inject_blend_tab(self):
        """Blend Test 탭: CSV 궤적 or 델타 waypoints + blending 비교 플롯."""
        import numpy as np
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from PyQt6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
            QLabel, QPushButton, QComboBox, QDoubleSpinBox,
            QPlainTextEdit, QProgressBar, QSplitter, QLineEdit,
        )
        from PyQt6.QtCore import Qt

        tab = QWidget()
        main_vl = QVBoxLayout(tab)
        main_vl.setContentsMargins(6, 6, 6, 6)
        main_vl.setSpacing(6)

        hsplit = QSplitter(Qt.Orientation.Horizontal)

        # ── 왼쪽 패널 ───────────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(400)
        left_vl = QVBoxLayout(left)
        left_vl.setContentsMargins(0, 0, 4, 0)
        left_vl.setSpacing(6)

        # ① Trajectory source
        grp_traj = QGroupBox("Trajectory")
        traj_vl = QVBoxLayout(grp_traj)
        traj_vl.setSpacing(3)

        src_hl = QHBoxLayout()
        self._blend_lbl_csv = QLabel("No CSV — using delta mode")
        self._blend_lbl_csv.setStyleSheet("color:#888; font-size:10px;")
        self._blend_lbl_csv.setWordWrap(True)
        btn_load_csv = QPushButton("Load CSV")
        btn_load_csv.setFixedWidth(80)
        btn_clear_csv = QPushButton("Clear")
        btn_clear_csv.setFixedWidth(50)
        src_hl.addWidget(self._blend_lbl_csv, 1)
        src_hl.addWidget(btn_load_csv)
        src_hl.addWidget(btn_clear_csv)
        traj_vl.addLayout(src_hl)

        # delta mode waypoints (hidden when CSV loaded)
        self._blend_grp_delta = QGroupBox("Delta Waypoints  (j1,j2,j3,j4,j5,j6 per line)")
        delta_vl = QVBoxLayout(self._blend_grp_delta)
        delta_vl.setSpacing(3)
        self._blend_edit_wp = QPlainTextEdit()
        self._blend_edit_wp.setFixedHeight(110)
        self._blend_edit_wp.setStyleSheet("font-family: Courier New; font-size:10px;")
        self._blend_edit_wp.setPlainText(
            '\n'.join(','.join(f'{v:.0f}' for v in row) for row in DEFAULT_JB2_DELTAS))
        delta_vl.addWidget(self._blend_edit_wp)
        btn_reset_wp = QPushButton("Reset to Mode Defaults")
        btn_reset_wp.setStyleSheet("font-size:10px;")
        delta_vl.addWidget(btn_reset_wp)
        traj_vl.addWidget(self._blend_grp_delta)
        left_vl.addWidget(grp_traj)

        # ② Settings
        grp_set = QGroupBox("Settings")
        fl = QFormLayout(grp_set)
        fl.setSpacing(4)

        self._blend_cbx_mode = QComboBox()
        self._blend_cbx_mode.addItems(['jb2', 'pb', 'lb'])
        fl.addRow("Mode (delta only):", self._blend_cbx_mode)

        self._blend_cbx_blend_opt = QComboBox()
        self._blend_cbx_blend_opt.addItems(['ratio', 'distance'])
        fl.addRow("Blend option (pb):", self._blend_cbx_blend_opt)

        self._blend_spin_speed = QDoubleSpinBox()
        self._blend_spin_speed.setRange(1, 500)
        self._blend_spin_speed.setValue(80.0)
        self._blend_spin_speed.setSuffix("  deg/s")
        fl.addRow("Speed (start/delta):", self._blend_spin_speed)

        self._blend_spin_accel = QDoubleSpinBox()
        self._blend_spin_accel.setRange(1, 2000)
        self._blend_spin_accel.setValue(100.0)
        self._blend_spin_accel.setSuffix("  deg/s²")
        fl.addRow("Accel (start/delta):", self._blend_spin_accel)

        self._blend_spin_hz = QDoubleSpinBox()
        self._blend_spin_hz.setRange(5, 200)
        self._blend_spin_hz.setValue(50.0)
        self._blend_spin_hz.setSuffix(" Hz")
        fl.addRow("Record Hz:", self._blend_spin_hz)

        self._blend_edit_values = QLineEdit("0.0  0.3  0.7  1.0")
        self._blend_edit_values.setPlaceholderText("0.0 0.3 0.7 1.0")
        fl.addRow("Blending values:", self._blend_edit_values)

        left_vl.addWidget(grp_set)

        # ③ Joint selection
        _JC = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c']
        from PyQt6.QtWidgets import QCheckBox
        grp_joints = QGroupBox("Joints")
        joints_hl = QHBoxLayout(grp_joints)
        joints_hl.setSpacing(4)
        self._blend_joint_chk: list = []
        for i in range(6):
            chk = QCheckBox(f'J{i+1}')
            chk.setChecked(True)
            chk.setStyleSheet(
                f"color: {_JC[i]}; font-weight: bold; font-size: 10px;")
            chk.stateChanged.connect(self._redraw_blend_plot)
            self._blend_joint_chk.append(chk)
            joints_hl.addWidget(chk)
        left_vl.addWidget(grp_joints)

        # ④ Control
        grp_ctrl = QGroupBox("Control")
        ctrl_vl = QVBoxLayout(grp_ctrl)
        ctrl_vl.setSpacing(4)

        ctrl_hl = QHBoxLayout()
        self.btn_blend_run = QPushButton("▶ Run")
        self.btn_blend_run.setStyleSheet(
            "background:#1b5e20; color:white; font-weight:bold; padding:6px;")
        self.btn_blend_stop = QPushButton("■ Stop")
        self.btn_blend_stop.setStyleSheet(
            "background:#b71c1c; color:white; font-weight:bold; padding:6px;")
        self.btn_blend_stop.setEnabled(False)
        self.btn_blend_save = QPushButton("Save JSON")
        self.btn_blend_save.setEnabled(False)
        self.btn_blend_clear = QPushButton("Clear")
        ctrl_hl.addWidget(self.btn_blend_run)
        ctrl_hl.addWidget(self.btn_blend_stop)
        ctrl_hl.addWidget(self.btn_blend_save)
        ctrl_hl.addWidget(self.btn_blend_clear)
        ctrl_vl.addLayout(ctrl_hl)

        self._blend_progress = QProgressBar()
        self._blend_progress.setValue(0)
        ctrl_vl.addWidget(self._blend_progress)

        self._blend_lbl_status = QLabel("Ready.")
        self._blend_lbl_status.setStyleSheet("font-size:10px; color:#aaa;")
        ctrl_vl.addWidget(self._blend_lbl_status)
        left_vl.addWidget(grp_ctrl)
        left_vl.addStretch()
        hsplit.addWidget(left)

        # ── 오른쪽: matplotlib 캔버스 ────────────────────────────────
        self._blend_fig = Figure()
        self._blend_fig.patch.set_facecolor('white')
        self._blend_canvas = FigureCanvasQTAgg(self._blend_fig)
        hsplit.addWidget(self._blend_canvas)
        hsplit.setStretchFactor(0, 0)
        hsplit.setStretchFactor(1, 1)
        main_vl.addWidget(hsplit, 1)

        # ── 로그 ────────────────────────────────────────────────────
        self._blend_log = QPlainTextEdit()
        self._blend_log.setReadOnly(True)
        self._blend_log.setMaximumBlockCount(400)
        self._blend_log.setFixedHeight(75)
        self._blend_log.setStyleSheet(
            "background:#16213e; color:#aaa; font-size:10px; border:none;")
        main_vl.addWidget(self._blend_log)

        self.tab_widget.addTab(tab, "Blend Test")

        # 시그널 연결
        self.btn_blend_run.clicked.connect(self._on_blend_run)
        self.btn_blend_stop.clicked.connect(self._on_blend_stop)
        self.btn_blend_save.clicked.connect(self._on_blend_save)
        self.btn_blend_clear.clicked.connect(self._on_blend_clear)
        btn_reset_wp.clicked.connect(self._on_blend_reset_wp)
        btn_load_csv.clicked.connect(self._on_blend_load_csv)
        btn_clear_csv.clicked.connect(self._on_blend_clear_csv)
        self._blend_cbx_mode.currentTextChanged.connect(self._on_blend_reset_wp)

        self._redraw_blend_plot()

    # ── Blend Test 핸들러 ─────────────────────────────────────────────────────

    def _on_blend_load_csv(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Trajectory CSV", ".", "CSV Files (*.csv)")
        if not path:
            return
        try:
            data = parse_trajectory_csv(path)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "CSV 오류", str(e))
            return
        self._blend_csv_data = data
        import pathlib
        fname = pathlib.Path(path).name
        self._blend_lbl_csv.setText(f"{fname}  ({len(data)} waypoints)")
        self._blend_lbl_csv.setStyleSheet("color:#4fc3f7; font-size:10px;")
        self._blend_grp_delta.setEnabled(False)
        self._blend_cbx_mode.setEnabled(False)
        self._blend_log.appendPlainText(
            f"[CSV] {fname} loaded — {len(data)} waypoints, "
            f"speed={data[0]['speed']:.0f} deg/s")

    def _on_blend_clear_csv(self):
        self._blend_csv_data = None
        self._blend_lbl_csv.setText("No CSV — using delta mode")
        self._blend_lbl_csv.setStyleSheet("color:#888; font-size:10px;")
        self._blend_grp_delta.setEnabled(True)
        self._blend_cbx_mode.setEnabled(True)

    def _on_blend_run(self):
        robot_ip = self.edit_robot_ip.text().strip()
        if not robot_ip:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "IP 없음", "Robot IP를 입력하세요.")
            return
        if self._blend_runner and self._blend_runner.isRunning():
            return

        try:
            blending_values = [float(v) for v in
                               self._blend_edit_values.text().split()]
            if not blending_values:
                raise ValueError("empty")
        except ValueError as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Blending values error", str(e))
            return

        use_csv = self._blend_csv_data is not None
        speed   = float(self._blend_spin_speed.value())
        accel   = float(self._blend_spin_accel.value())

        if use_csv:
            params = {
                'robot_ip':        robot_ip,
                'use_csv':         True,
                'waypoint_data':   self._blend_csv_data,
                'blending_values': blending_values,
                'speed':           speed,
                'accel':           accel,
                'speed_bar':       float(self.spin_speed_bar.value()),
                'record_hz':       float(self._blend_spin_hz.value()),
            }
        else:
            try:
                deltas = []
                for line in self._blend_edit_wp.toPlainText().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    vals = [float(v) for v in line.split(',')]
                    if len(vals) != 6:
                        raise ValueError(f"need 6 values: {line}")
                    deltas.append(vals)
                if not deltas:
                    raise ValueError("no waypoints")
            except ValueError as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Waypoints error", str(e))
                return

            params = {
                'robot_ip':        robot_ip,
                'use_csv':         False,
                'mode':            self._blend_cbx_mode.currentText(),
                'waypoint_deltas': deltas,
                'blending_values': blending_values,
                'speed':           speed,
                'accel':           accel,
                'speed_bar':       float(self.spin_speed_bar.value()),
                'record_hz':       float(self._blend_spin_hz.value()),
                'blending_option': self._blend_cbx_blend_opt.currentText(),
            }

        self._blend_results = {}
        self._blend_progress.setMaximum(len(blending_values))
        self._blend_progress.setValue(0)
        self._blend_log.clear()
        self._redraw_blend_plot()

        calib = None
        if self._calib_result:
            calib = {'T_base_motive': self._calib_result['T_base_motive']}

        self._blend_runner = BlendTestRunner(
            params,
            natnet_state=self._natnet_state,
            calib=calib,
            parent=self,
        )
        self._blend_runner.run_progress.connect(self._on_blend_run_progress)
        self._blend_runner.run_done.connect(self._on_blend_run_done)
        self._blend_runner.all_done.connect(self._on_blend_all_done)
        self._blend_runner.log_msg.connect(self._blend_log.appendPlainText)
        self._blend_runner.error.connect(self._on_blend_error)

        self.btn_blend_run.setEnabled(False)
        self.btn_blend_stop.setEnabled(True)
        self.btn_blend_save.setEnabled(False)
        self._blend_lbl_status.setText("Running …")
        self._blend_runner.start()

    def _on_blend_stop(self):
        if self._blend_runner and self._blend_runner.isRunning():
            self._blend_runner.stop()
        self.btn_blend_stop.setEnabled(False)

    def _on_blend_run_progress(self, run_idx: int, total: int, bv: float):
        self._blend_lbl_status.setText(
            f"Run {run_idx}/{total}  blending={bv:.3f} …")

    def _on_blend_run_done(self, bv: float, records: list):
        self._blend_results[bv] = records
        self._blend_progress.setValue(len(self._blend_results))
        self._redraw_blend_plot()

    def _on_blend_all_done(self, all_results: dict):
        import numpy as np
        self._blend_results = all_results
        self.btn_blend_run.setEnabled(True)
        self.btn_blend_stop.setEnabled(False)
        self.btn_blend_save.setEnabled(True)

        # 편차 요약 (CSV 모드일 때만)
        if self._blend_csv_data:
            parts = []
            for bv in sorted(all_results):
                dev = compute_plan_deviation(all_results[bv], self._blend_csv_data)
                if len(dev):
                    parts.append(f"blend={bv:.2f}: rmse={dev.mean():.3f} deg")
            self._blend_lbl_status.setText(
                f"Done — {len(all_results)} runs  |  " + "  ".join(parts))
        else:
            self._blend_lbl_status.setText(
                f"Done — {len(all_results)} runs.  Save available.")
        self._redraw_blend_plot()

    def _on_blend_error(self, msg: str):
        self.btn_blend_run.setEnabled(True)
        self.btn_blend_stop.setEnabled(False)
        self._blend_lbl_status.setText(f"Error: {msg}")
        self._blend_log.appendPlainText(f"[ERROR] {msg}")

    def _on_blend_clear(self):
        self._blend_results = {}
        self._blend_progress.setValue(0)
        self._blend_lbl_status.setText("Ready.")
        self._blend_log.clear()
        self._redraw_blend_plot()

    def _on_blend_reset_wp(self):
        import numpy as np
        mode   = self._blend_cbx_mode.currentText()
        deltas = DEFAULT_JB2_DELTAS if mode == 'jb2' else DEFAULT_PB_DELTAS
        self._blend_edit_wp.setPlainText(
            '\n'.join(','.join(f'{v:.0f}' for v in row) for row in deltas))

    def _on_blend_save(self):
        if not self._blend_results:
            return
        import datetime
        from PyQt6.QtWidgets import QFileDialog
        stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = str(self._config.get('root_path', '.')) + f'/blend_{stamp}.json'
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Blend Results", default, "JSON Files (*.json)")
        if not path:
            return
        use_csv = self._blend_csv_data is not None
        payload = {
            'mode':    'jb2' if use_csv else self._blend_cbx_mode.currentText(),
            'use_csv': use_csv,
            'results': {str(bv): recs
                        for bv, recs in self._blend_results.items()},
        }
        if use_csv:
            payload['waypoint_data'] = [
                {'joints': wd['joints'].tolist(),
                 'speed':  wd['speed'],
                 'accel':  wd['accel']}
                for wd in self._blend_csv_data
            ]
        with open(path, 'w', encoding='utf-8') as f:
            import json as _json
            _json.dump(payload, f)
        self._log(f"Blend results saved → {path}")

    def _redraw_blend_plot(self):
        import numpy as np
        import matplotlib.cm as cm
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D

        if not hasattr(self, '_blend_fig'):
            return

        self._blend_fig.clear()
        self._blend_fig.patch.set_facecolor('white')
        use_csv = bool(self._blend_csv_data)

        # ── 공통 스타일 (흰 배경) ────────────────────────────────────
        BG  = 'white'
        GRD = '#dddddd'
        TC  = '#444444'
        LBL = '#333333'
        TTL = '#111111'

        def _style(ax, ylabel='', xlabel=''):
            ax.set_facecolor(BG)
            ax.tick_params(colors=TC, labelsize=8)
            ax.grid(True, alpha=0.6, color=GRD, linewidth=0.6)
            for sp in ax.spines.values():
                sp.set_edgecolor('#bbbbbb')
            if ylabel:
                ax.set_ylabel(ylabel, color=LBL, fontsize=8)
            if xlabel:
                ax.set_xlabel(xlabel, color=LBL, fontsize=8)

        if not self._blend_results:
            ax = self._blend_fig.add_subplot(111)
            _style(ax)
            ax.text(0.5, 0.5, 'Press Run to start',
                    ha='center', va='center', color='#999', fontsize=12,
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            self._blend_canvas.draw()
            return

        # ── 선택된 관절 목록 ─────────────────────────────────────────
        J_COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c']
        if hasattr(self, '_blend_joint_chk'):
            j_visible = [chk.isChecked() for chk in self._blend_joint_chk]
        else:
            j_visible = [True] * 6
        visible_idx = [i for i, v in enumerate(j_visible) if v]

        bv_list   = sorted(self._blend_results.keys())
        n_bv      = max(len(bv_list), 1)
        bv_colors = cm.tab10(np.linspace(0.0, 0.9, n_bv))
        bv_ls     = ['-', '--', ':', '-.']   # blending별 선 스타일

        # ── 계획 궤적 시간 파라미터화 (CSV 모드) ─────────────────────
        t_plan = jplan = None
        if use_csv and self._blend_csv_data:
            wd    = self._blend_csv_data
            jplan = np.array([w['joints'] for w in wd])
            spds  = np.array([w['speed']  for w in wd])
            seg_d = np.linalg.norm(np.diff(jplan, axis=0), axis=1)
            seg_t = np.where(seg_d > 0.01, seg_d / spds[:-1], 0.0)
            t_plan = np.concatenate([[0.0], np.cumsum(seg_t)])

        # has_ee: 엔드이펙터 데이터 존재 여부
        first_recs = self._blend_results[bv_list[0]]
        has_tcp_m   = any(r.get('tcp_motive') is not None for r in first_recs)
        has_rb      = any(r.get('rb_pos')     is not None for r in first_recs)
        has_rb_rot  = any(r.get('rb_rot')     is not None for r in first_recs)

        # ── GridSpec: 5행 ────────────────────────────────────────────
        # Row 0: 관절각    Row 1: 관절속도    Row 2: 관절가속도
        # Row 3: EE XY | EE Z vs time
        # Row 4: EE 선속도 | EE 각속도
        gs = GridSpec(5, 2,
                      figure=self._blend_fig,
                      height_ratios=[2.8, 1.5, 1.5, 2.2, 2.2],
                      hspace=0.60, wspace=0.32,
                      left=0.08, right=0.97, top=0.96, bottom=0.04)

        ax_j    = self._blend_fig.add_subplot(gs[0, :])
        ax_vel  = self._blend_fig.add_subplot(gs[1, :], sharex=ax_j)
        ax_acc  = self._blend_fig.add_subplot(gs[2, :], sharex=ax_j)
        ax_xy   = self._blend_fig.add_subplot(gs[3, 0])
        ax_z    = self._blend_fig.add_subplot(gs[3, 1])
        ax_lv   = self._blend_fig.add_subplot(gs[4, 0])
        ax_av   = self._blend_fig.add_subplot(gs[4, 1])

        _style(ax_j,   ylabel='Joint angle (deg)')
        _style(ax_vel, ylabel='Velocity (deg/s)')
        _style(ax_acc, ylabel='Accel (deg/s²)',  xlabel='Time (s)')
        _style(ax_xy,  ylabel='Y (m)',            xlabel='X (m)')
        _style(ax_z,   ylabel='Z (m)',            xlabel='Time (s)')
        _style(ax_lv,  ylabel='Lin. vel (mm/s)',  xlabel='Time (s)')
        _style(ax_av,  ylabel='Ang. vel (deg/s)', xlabel='Time (s)')

        ax_j.set_title('Joint angles vs time', color=TTL, fontsize=9, fontweight='bold')
        ax_vel.set_title('Joint velocity (L2 norm)', color=TTL, fontsize=8)
        ax_acc.set_title('Joint acceleration (L2 norm)', color=TTL, fontsize=8)
        ax_xy.set_title('EE XY — TCP (line) vs NatNet RB (dot)',
                        color=TTL, fontsize=8)
        ax_z.set_title('EE Z vs time', color=TTL, fontsize=8)
        ax_lv.set_title('EE linear velocity', color=TTL, fontsize=8)
        src = 'NatNet RB' if has_rb_rot else 'TCP Euler'
        ax_av.set_title(f'EE angular velocity ({src})', color=TTL, fontsize=8)

        # ── ① 계획 궤적 오버레이 (점선 + 마커) ──────────────────────
        if t_plan is not None and jplan is not None:
            for j_idx in visible_idx:
                ax_j.plot(t_plan, jplan[:, j_idx], '--',
                          color=J_COLORS[j_idx], lw=1.0, alpha=0.45,
                          zorder=2)
                ax_j.scatter(t_plan, jplan[:, j_idx],
                             color=J_COLORS[j_idx], s=14, alpha=0.55,
                             zorder=3)

        # ── ② 실제 데이터 ─────────────────────────────────────────
        def _smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
            if len(x) < w:
                return x
            return np.convolve(x, np.ones(w) / w, mode='same')

        def _clean(arr_t, arr_data):
            """단조 증가 타임스탬프만 유지. 불연속 구간은 NaN 삽입."""
            if len(arr_t) < 2:
                return arr_t, arr_data
            # 단조 증가 마스크
            mono = np.concatenate([[True], np.diff(arr_t) > 1e-6])
            t_c = arr_t[mono]
            d_c = arr_data[mono]
            # 시간 간격이 median의 5배 이상인 지점에 NaN 삽입 (갭 가시화)
            dt = np.diff(t_c)
            med_dt = np.median(dt)
            gap_idx = np.where(dt > med_dt * 5)[0] + 1
            if len(gap_idx):
                t_out = np.insert(t_c.astype(float), gap_idx, np.nan)
                d_shape = (len(t_out),) + d_c.shape[1:]
                d_out = np.full(d_shape, np.nan)
                # NaN 삽입된 인덱스 보정
                mask = ~np.isnan(t_out)
                d_out[mask] = d_c
            else:
                t_out, d_out = t_c, d_c
            return t_out, d_out

        for bv, bv_color, ls in zip(bv_list, bv_colors, bv_ls):
            recs = self._blend_results[bv]
            if not recs:
                continue

            t_raw = np.array([r['t'] for r in recs])
            j_raw = np.array([r['joints'] for r in recs])  # (N, 6)

            t, j = _clean(t_raw, j_raw)

            if np.sum(~np.isnan(t)) < 2:
                continue

            # 관절각 (선택된 축만)
            for j_idx in visible_idx:
                ax_j.plot(t, j[:, j_idx],
                          color=J_COLORS[j_idx], ls=ls, lw=1.4,
                          alpha=0.9, zorder=4)

            # 속도·가속도 (NaN 구간 제외 후 수치 미분)
            t_valid = t[~np.isnan(t)]
            j_valid = j[~np.isnan(t)]
            if len(t_valid) >= 5 and visible_idx:
                j_sel  = j_valid[:, visible_idx]
                vel    = np.gradient(j_sel, t_valid, axis=0)
                acc    = np.gradient(vel,   t_valid, axis=0)
                vel_s  = np.column_stack([_smooth(vel[:, k])    for k in range(len(visible_idx))])
                acc_s  = np.column_stack([_smooth(acc[:, k], 7) for k in range(len(visible_idx))])
                vel_mag = np.linalg.norm(vel_s, axis=1)
                acc_mag = np.linalg.norm(acc_s, axis=1)
                ax_vel.plot(t_valid, vel_mag, color=bv_color, ls=ls, lw=1.4,
                            label=f'blend={bv:.2f}')
                ax_acc.plot(t_valid, acc_mag, color=bv_color, ls=ls, lw=1.4)

            # ── 엔드이펙터 ───────────────────────────────────────────
            # tcp_motive: FK + HE 변환 위치 (실선)
            tc = tt = None
            if has_tcp_m:
                _pairs = [(r['t'], r['tcp_motive']) for r in recs
                          if r.get('tcp_motive') is not None]
                if _pairs:
                    tt = np.array([x[0] for x in _pairs])
                    tc = np.array([x[1] for x in _pairs])  # (N,3) m
                    ax_xy.plot(tc[:, 0], tc[:, 1], color=bv_color,
                               ls=ls, lw=1.8, zorder=4,
                               label=f'FK+HE  b={bv:.2f}')
                    ax_z.plot(tt, tc[:, 2], color=bv_color, ls=ls, lw=1.5,
                              label=f'FK+HE  b={bv:.2f}')

            # NatNet RB: 점 + 선으로 함께 표시
            rb = rt = rb_q = None
            if has_rb:
                _pairs_rb = [(r['t'], r['rb_pos'], r.get('rb_rot'))
                             for r in recs if r.get('rb_pos') is not None]
                if _pairs_rb:
                    rt  = np.array([x[0] for x in _pairs_rb])
                    rb  = np.array([x[1] for x in _pairs_rb])    # (N,3) m
                    rots = [x[2] for x in _pairs_rb]
                    # 회전 데이터가 있는 경우만 수집
                    if has_rb_rot and all(r is not None for r in rots):
                        rb_q = np.array(rots)                     # (N,4) xyzw
                    # XY 궤적: 얇은 선 + 반투명 마커로 tcp_motive 와 비교
                    ax_xy.plot(rb[:, 0], rb[:, 1],
                               color=bv_color, ls=':', lw=1.2, alpha=0.7,
                               zorder=3, label=f'NatNet RB  b={bv:.2f}')
                    ax_xy.scatter(rb[::3, 0], rb[::3, 1],  # 3샘플에 1개
                                  color=bv_color, s=8, alpha=0.5,
                                  marker='.', zorder=3)
                    ax_z.plot(rt, rb[:, 2], color=bv_color, ls=':', lw=1.2,
                              alpha=0.7, label=f'NatNet RB  b={bv:.2f}')

            # ── 엔드이펙터 선속도 ─────────────────────────────────────
            # 우선순위: NatNet RB > tcp_motive > 로봇 TCP raw
            if rb is not None and len(rb) >= 5:
                pos_src, t_src, lv_lbl = rb, rt, 'NatNet RB'
            elif tc is not None and len(tc) >= 5:
                pos_src, t_src, lv_lbl = tc, tt, 'FK+HE TCP'
            else:
                # 로봇 raw TCP (mm → m)
                _pairs_raw = [(r['t'], r['tcp'][:3]) for r in recs]
                t_src = np.array([x[0] for x in _pairs_raw])
                pos_src = np.array([x[1] for x in _pairs_raw]) / 1000.0
                lv_lbl = 'Robot TCP'

            if pos_src is not None and len(pos_src) >= 5:
                lv = np.gradient(pos_src, t_src, axis=0)          # m/s (N,3)
                lv_s  = np.column_stack([_smooth(lv[:, k]) for k in range(3)])
                lv_mag = np.linalg.norm(lv_s, axis=1) * 1000.0    # mm/s
                ax_lv.plot(t_src, lv_mag, color=bv_color, ls=ls, lw=1.4,
                           label=f'{lv_lbl}  b={bv:.2f}')

            # ── 엔드이펙터 각속도 ─────────────────────────────────────
            if rb_q is not None and len(rb_q) >= 5:
                # NatNet 쿼터니언으로 각속도 계산
                q = rb_q / np.linalg.norm(rb_q, axis=1, keepdims=True)
                dot = np.clip(np.abs(np.einsum('ij,ij->i', q[:-1], q[1:])), 0, 1)
                angle_deg = np.degrees(2.0 * np.arccos(dot))       # 연속 프레임 회전량
                dt_rb = np.diff(rt)
                omega = np.where(dt_rb > 1e-6, angle_deg / dt_rb, 0.0)  # deg/s
                omega_s = _smooth(omega, w=7)
                # 시간축: 중간점
                t_mid = 0.5 * (rt[:-1] + rt[1:])
                ax_av.plot(t_mid, omega_s, color=bv_color, ls=ls, lw=1.4,
                           label=f'NatNet RB  b={bv:.2f}')
            else:
                # Fallback: TCP Euler 각도 미분
                _pairs_e = [(r['t'], r['tcp'][3:6]) for r in recs]
                t_e  = np.array([x[0] for x in _pairs_e])
                euler = np.array([x[1] for x in _pairs_e])         # (N,3) deg
                if len(t_e) >= 5:
                    av = np.gradient(euler, t_e, axis=0)
                    av_s = np.column_stack([_smooth(av[:, k]) for k in range(3)])
                    av_mag = np.linalg.norm(av_s, axis=1)
                    ax_av.plot(t_e, av_mag, color=bv_color, ls=ls, lw=1.4,
                               label=f'TCP Euler  b={bv:.2f}')

        # ── ③ 범례 ───────────────────────────────────────────────────
        leg_kw = dict(fontsize=7, facecolor='white', edgecolor='#ccc',
                      framealpha=0.85, loc='best')

        # 관절각 범례: 색 = 축, 스타일 = blending
        handles_j = [
            Line2D([0],[0], color=J_COLORS[i], lw=2,
                   label=f'J{i+1}', alpha=0.9)
            for i in visible_idx
        ] + [
            Line2D([0],[0], color='#555', ls=ls, lw=1.3,
                   label=f'blend={bv:.2f}')
            for bv, ls in zip(bv_list, bv_ls)
        ]
        if handles_j:
            ax_j.legend(handles=handles_j, ncol=min(len(handles_j), 5),
                        **leg_kw)

        ax_vel.legend(**leg_kw)

        if has_tcp_m or has_rb:
            ax_xy.legend(ncol=2, fontsize=6, facecolor='white',
                         edgecolor='#ccc', framealpha=0.85)
            ax_z.legend(ncol=2, fontsize=6, facecolor='white',
                        edgecolor='#ccc', framealpha=0.85)
            ax_xy.set_aspect('equal', adjustable='datalim')
        else:
            for ax, msg in ((ax_xy, 'No HE calib — load calib.json'),
                            (ax_z,  'No NatNet / HE data')):
                ax.text(0.5, 0.5, msg, ha='center', va='center',
                        color='#aaa', fontsize=8, transform=ax.transAxes)

        ax_lv.legend(**leg_kw)
        ax_av.legend(**leg_kw)

        # sharex → 상단 축 x tick label 숨김
        ax_j.tick_params(labelbottom=False)
        ax_vel.tick_params(labelbottom=False)

        self._blend_canvas.draw()

    # ──────────────────────────────────────────────
    # Tab 5 – Sync (NatNet ↔ Robot 연속 동기 기록)
    # ──────────────────────────────────────────────

    def _inject_sync_tab(self, pg):
        """pyqtgraph 위젯으로 구성된 Sync 탭을 tab_widget에 추가."""
        from PyQt6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
            QLabel, QPushButton, QDoubleSpinBox, QGroupBox,
            QPlainTextEdit,
        )
        from PyQt6.QtCore import Qt

        tab = QWidget()
        main_vl = QVBoxLayout(tab)
        main_vl.setContentsMargins(6, 6, 6, 6)
        main_vl.setSpacing(6)

        # ── 상단 컨트롤 바 ──────────────────────────────────────────
        ctrl_hl = QHBoxLayout()

        grp_settings = QGroupBox("Recording")
        grp_settings.setMaximumWidth(300)
        sl = QHBoxLayout(grp_settings)
        sl.addWidget(QLabel("Interval (s):"))
        self._sync_spin_interval = QDoubleSpinBox()
        self._sync_spin_interval.setRange(0.01, 1.0)
        self._sync_spin_interval.setDecimals(3)
        self._sync_spin_interval.setSingleStep(0.01)
        self._sync_spin_interval.setValue(0.02)
        sl.addWidget(self._sync_spin_interval)
        ctrl_hl.addWidget(grp_settings)

        grp_ctrl = QGroupBox("Control")
        cl = QHBoxLayout(grp_ctrl)
        self.btn_sync_start = QPushButton("▶ Start Sync")
        self.btn_sync_start.setStyleSheet(
            "background:#1b5e20; color:white; font-weight:bold; padding:6px;")
        self.btn_sync_start.setEnabled(False)
        self.btn_sync_stop = QPushButton("■ Stop")
        self.btn_sync_stop.setStyleSheet(
            "background:#b71c1c; color:white; font-weight:bold; padding:6px;")
        self.btn_sync_stop.setEnabled(False)
        self.btn_sync_save = QPushButton("Save CSV")
        self.btn_sync_save.setEnabled(False)
        self.btn_sync_clear = QPushButton("Clear")
        cl.addWidget(self.btn_sync_start)
        cl.addWidget(self.btn_sync_stop)
        cl.addWidget(self.btn_sync_save)
        cl.addWidget(self.btn_sync_clear)
        ctrl_hl.addWidget(grp_ctrl)

        # 통계 레이블
        grp_stat = QGroupBox("Statistics")
        stat_l = QHBoxLayout(grp_stat)
        self.lbl_sync_rmse  = QLabel("RMSE: —")
        self.lbl_sync_mean  = QLabel("Mean: —")
        self.lbl_sync_max   = QLabel("Max: —")
        self.lbl_sync_pts   = QLabel("Points: 0")
        self.lbl_sync_lag   = QLabel("Mocap lag: —")
        for lbl in (self.lbl_sync_rmse, self.lbl_sync_mean,
                    self.lbl_sync_max, self.lbl_sync_pts, self.lbl_sync_lag):
            lbl.setStyleSheet("font-size:11px;")
            stat_l.addWidget(lbl)
        ctrl_hl.addWidget(grp_stat)
        ctrl_hl.addStretch()

        main_vl.addLayout(ctrl_hl)

        # ── 플롯 영역 ──────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 오차 시계열
        self._sync_plot_err = pg.PlotWidget(background='#1a1a2e')
        self._sync_plot_err.setLabel('left', 'Error (mm)')
        self._sync_plot_err.setLabel('bottom', 'Time (s)')
        self._sync_plot_err.showGrid(x=True, y=True, alpha=0.3)
        self._sync_curve_err  = self._sync_plot_err.plot(
            pen=pg.mkPen('#3399ff', width=1.5))
        self._sync_line_rmse  = pg.InfiniteLine(
            angle=0, pen=pg.mkPen('#00ff88', width=1,
                                  style=pg.QtCore.Qt.PenStyle.DashLine))
        self._sync_plot_err.addItem(self._sync_line_rmse)

        # mocap lag 시계열
        self._sync_plot_lag = pg.PlotWidget(background='#1a1a2e')
        self._sync_plot_lag.setLabel('left', 'Mocap lag (ms)')
        self._sync_plot_lag.setLabel('bottom', 'Time (s)')
        self._sync_plot_lag.showGrid(x=True, y=True, alpha=0.3)
        self._sync_curve_lag = self._sync_plot_lag.plot(
            pen=pg.mkPen('#fb8c00', width=1.5))

        splitter.addWidget(self._sync_plot_err)
        splitter.addWidget(self._sync_plot_lag)
        splitter.setSizes([2, 1])
        main_vl.addWidget(splitter, 1)

        # 로그
        self._sync_log = QPlainTextEdit()
        self._sync_log.setReadOnly(True)
        self._sync_log.setMaximumBlockCount(300)
        self._sync_log.setFixedHeight(80)
        self._sync_log.setStyleSheet(
            "background:#16213e; color:#aaa; font-size:10px; border:none;")
        main_vl.addWidget(self._sync_log)

        self.tab_widget.addTab(tab, "Sync")

        # 시그널 연결
        self.btn_sync_start.clicked.connect(self._on_sync_start)
        self.btn_sync_stop.clicked.connect(self._on_sync_stop)
        self.btn_sync_save.clicked.connect(self._on_sync_save)
        self.btn_sync_clear.clicked.connect(self._on_sync_clear)

    def _on_sync_start(self):
        if self._calib_result is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "캘리브레이션 없음",
                                "먼저 캘리브레이션을 수행하거나 로드하세요.")
            return
        robot_ip = self.edit_robot_ip.text().strip()
        if not robot_ip:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "IP 없음", "Robot IP를 입력하세요.")
            return
        if self._sync_runner and self._sync_runner.isRunning():
            return

        import numpy as np
        calib = {
            'T_base_motive': self._calib_result['T_base_motive'],
            'T_rb_tcp':      self._calib_result.get('T_rb_tcp', np.eye(4).tolist()),
        }
        params = {
            'robot_ip':             robot_ip,
            'speed_bar':            float(self.spin_speed_bar.value()),
            'interval_s':           float(self._sync_spin_interval.value()),
            'tcp_orientation_type': self._config.get('tcp_orientation_type', 'zyx_euler_deg'),
        }

        self._sync_records = []
        self._sync_elapsed = []
        self._sync_errors  = []
        self._sync_lags    = []

        self._sync_runner = SyncRunner(params, calib, self._natnet_state, parent=self)
        self._sync_runner.point_recorded.connect(self._on_sync_point)
        self._sync_runner.all_done.connect(self._on_sync_done)
        self._sync_runner.log_msg.connect(self._sync_log.appendPlainText)
        self._sync_runner.error.connect(self._on_sync_error)

        self.btn_sync_start.setEnabled(False)
        self.btn_sync_stop.setEnabled(True)
        self.btn_sync_save.setEnabled(False)
        self._sync_log.clear()
        self._sync_curve_err.setData([], [])
        self._sync_curve_lag.setData([], [])
        self._sync_runner.start()

    def _on_sync_stop(self):
        if self._sync_runner and self._sync_runner.isRunning():
            self._sync_runner.stop()
        self.btn_sync_stop.setEnabled(False)

    def _on_sync_point(self, rec: dict):
        self._sync_elapsed.append(rec['elapsed_s'])
        self._sync_errors.append(rec['error_mm'])
        self._sync_lags.append(rec['mocap_age_ms'])
        self._sync_records.append(rec)

        # 실시간 플롯 (50포인트마다 갱신)
        n = len(self._sync_elapsed)
        if n % 50 == 0 or n < 10:
            self._sync_curve_err.setData(self._sync_elapsed, self._sync_errors)
            self._sync_curve_lag.setData(self._sync_elapsed, self._sync_lags)

        if n % 100 == 0:
            import numpy as np
            errs = np.array(self._sync_errors)
            rmse = float(np.sqrt(np.mean(errs ** 2)))
            self._sync_line_rmse.setValue(rmse)
            self.lbl_sync_rmse.setText(f"RMSE: {rmse:.2f} mm")
            self.lbl_sync_mean.setText(f"Mean: {errs.mean():.2f} mm")
            self.lbl_sync_max.setText( f"Max: {errs.max():.2f} mm")
            self.lbl_sync_pts.setText( f"Points: {n}")
            lag = np.array(self._sync_lags)
            self.lbl_sync_lag.setText(f"Mocap lag: {lag.mean():.1f} ms (max {lag.max():.1f})")

    def _on_sync_done(self, records: list):
        import numpy as np
        self._sync_records = records
        self.btn_sync_start.setEnabled(True)
        self.btn_sync_stop.setEnabled(False)
        self.btn_sync_save.setEnabled(True)

        # 최종 플롯 갱신
        self._sync_curve_err.setData(self._sync_elapsed, self._sync_errors)
        self._sync_curve_lag.setData(self._sync_elapsed, self._sync_lags)

        if self._sync_errors:
            errs = np.array(self._sync_errors)
            lags = np.array(self._sync_lags)
            rmse = float(np.sqrt(np.mean(errs ** 2)))
            self._sync_line_rmse.setValue(rmse)
            self.lbl_sync_rmse.setText(f"RMSE: {rmse:.3f} mm")
            self.lbl_sync_mean.setText(f"Mean: {errs.mean():.3f} mm")
            self.lbl_sync_max.setText( f"Max:  {errs.max():.3f} mm")
            self.lbl_sync_pts.setText( f"Points: {len(records)}")
            self.lbl_sync_lag.setText(
                f"Mocap lag: {lags.mean():.1f} ms  max {lags.max():.1f} ms")

    def _on_sync_error(self, msg: str):
        self.btn_sync_start.setEnabled(True)
        self.btn_sync_stop.setEnabled(False)
        self._sync_log.appendPlainText(f"[ERROR] {msg}")

    def _on_sync_clear(self):
        self._sync_records = []
        self._sync_elapsed = []
        self._sync_errors  = []
        if hasattr(self, '_sync_lags'):
            self._sync_lags = []
        if hasattr(self, '_sync_curve_err'):
            self._sync_curve_err.setData([], [])
            self._sync_curve_lag.setData([], [])
            self._sync_line_rmse.setValue(0)
        for lbl in (self.lbl_sync_rmse, self.lbl_sync_mean,
                    self.lbl_sync_max, self.lbl_sync_pts, self.lbl_sync_lag):
            lbl.setText(lbl.text().split(':')[0] + ': —')

    def _on_sync_save(self):
        if not self._sync_records:
            return
        import datetime, csv as _csv
        stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = str(self._config.get('root_path', '.')) + f'/sync_{stamp}.csv'
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Sync CSV", default, "CSV Files (*.csv)")
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = _csv.DictWriter(
                f, fieldnames=list(self._sync_records[0].keys()))
            writer.writeheader()
            writer.writerows(self._sync_records)
        self._log(f"Sync CSV saved → {path}")

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _log(self, msg: str):
        log.info(msg)
        if hasattr(self, 'log_panel'):
            self.log_panel.appendPlainText(msg)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._monitor_timer.stop()

        # 시그널 수신을 먼저 끊어 종료 중 UI 업데이트 방지
        for worker in (self._calib_runner, self._verify_runner,
                       self._robot_worker, self._natnet_worker):
            if worker is not None:
                try:
                    worker.blockSignals(True)
                except Exception:
                    pass

        if self._calib_runner and self._calib_runner.isRunning():
            self._calib_runner.stop()
            self._calib_runner.wait(5000)

        if self._verify_runner and self._verify_runner.isRunning():
            self._verify_runner.stop()
            self._verify_runner.wait(5000)

        if self._sync_runner and self._sync_runner.isRunning():
            self._sync_runner.stop()
            self._sync_runner.wait(3000)

        if self._robot_worker and self._robot_worker.isRunning():
            self._robot_worker.stop()
            self._robot_worker.wait(3000)

        if self._natnet_worker and self._natnet_worker.isRunning():
            self._natnet_worker.stop()
            self._natnet_worker.wait(3000)

        log.info("VerifyCobotWindow closed.")
        super().closeEvent(event)


# ──────────────────────────────────────────────────────
# Utility helpers (module-level)
# ──────────────────────────────────────────────────────

def _replace_placeholder(placeholder, new_widget):
    """Swap *placeholder* QWidget in its parent layout with *new_widget*."""
    parent = placeholder.parentWidget()
    layout = parent.layout()
    if layout is None:
        return
    idx = layout.indexOf(placeholder)
    if idx < 0:
        return
    layout.removeWidget(placeholder)
    placeholder.deleteLater()
    layout.insertWidget(idx, new_widget)


def _resolve_cal_path(config: dict) -> pathlib.Path:
    """config의 cal_file을 root_path 기준 절대경로로 반환."""
    cal = config.get('cal_file', 'calibration_svd.json')
    p = pathlib.Path(cal)
    if not p.is_absolute():
        root = pathlib.Path(config.get('root_path', '.'))
        p = root / p
    return p


def _rotvec_str(R) -> str:
    try:
        from calibration.solver import rotvec_from_matrix
        import numpy as np
        rv = rotvec_from_matrix(R)
        return f"[{rv[0]:.3f}, {rv[1]:.3f}, {rv[2]:.3f}] rad"
    except Exception:
        return "[?, ?, ?] rad"
