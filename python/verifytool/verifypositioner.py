"""
verifypositioner.py
===================
최대 4개의 NatNet Rigid Body를 병렬로 수신하여
시작/정지 포즈 변화를 측정하고 궤적을 시각화하는 툴.
"""

import csv
import datetime
import json
import logging
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
        QSpinBox, QDoubleSpinBox, QScrollArea, QPlainTextEdit,
        QFileDialog, QMessageBox, QApplication,
    )
    from PyQt6.QtCore import QTimer, Qt
except ImportError:
    raise ImportError("PyQt6 is required.")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

_ROOT = pathlib.Path(__file__).parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verifytool.workers.natnet_worker import NatNetWorker
from verifytool.calib_runner import NatNetStateProxy

log = logging.getLogger(__name__)

_MAX_BUF  = 500
_N_RB     = 4
_DARK_BG  = '#1a1a2e'
_DARK_PNL = '#16213e'
_RB_COLOR = ['#e53935', '#1e88e5', '#43a047', '#fb8c00']
_GRP_STYLE = (
    f"QGroupBox {{ background: {_DARK_PNL}; border: 1px solid #333;"
    " padding-top: 14px; margin-top: 4px; }}"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px;"
    " color: #aaa; font-size: 11px; }"
)


class CsvMocapVerifyWindow(QMainWindow):

    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._state = 'IDLE'
        self._natnet_worker: NatNetWorker | None = None

        # RB별 데이터 (index 0~3)
        self._rolling: list[list] = [[] for _ in range(_N_RB)]  # (pos, quat)
        self._traj:    list[list] = [[] for _ in range(_N_RB)]  # (pos, quat) during motion
        self._start:   list      = [None] * _N_RB               # (pos_list, quat_list)
        self._result:  list      = [None] * _N_RB               # computed delta dict

        self.setWindowTitle("Verify Positioner")
        self.setStyleSheet(f"background-color: {_DARK_BG}; color: #eee;")
        self._build_ui()

    # ──────────────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        hl = QHBoxLayout(central)
        hl.setContentsMargins(6, 6, 6, 6)
        hl.setSpacing(8)

        # 좌측 스크롤 패널
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        inner = QWidget()
        left_vl = QVBoxLayout(inner)
        left_vl.setContentsMargins(2, 2, 6, 2)
        left_vl.setSpacing(6)

        left_vl.addWidget(self._build_natnet_grp())
        left_vl.addWidget(self._build_rb_grp())
        left_vl.addWidget(self._build_motion_grp())
        left_vl.addWidget(self._build_results_grp())
        left_vl.addWidget(self._build_scale_grp())
        left_vl.addStretch()

        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumBlockCount(400)
        self.log_panel.setFixedHeight(100)
        self.log_panel.setStyleSheet(
            f"background:{_DARK_PNL}; color:#aaa; font-size:10px; border:none;")
        left_vl.addWidget(self.log_panel)

        scroll.setWidget(inner)
        hl.addWidget(scroll)

        # 우측 플롯
        self._fig = Figure(tight_layout=True)
        self._fig.patch.set_facecolor(_DARK_BG)
        self._canvas = FigureCanvasQTAgg(self._fig)
        hl.addWidget(self._canvas, 1)

        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_save.clicked.connect(self._on_save)

        self.resize(1200, 720)
        self._redraw()

    def _build_natnet_grp(self):
        grp = QGroupBox("NatNet Connection")
        grp.setStyleSheet(_GRP_STYLE)
        vl = QVBoxLayout(grp)
        vl.setSpacing(4)
        vl.setContentsMargins(8, 4, 8, 8)
        for label, attr, key in [
            ("Server:", "edit_server", "natnet_server_ip"),
            ("Client:", "edit_client", "natnet_client_ip"),
        ]:
            hl = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(46)
            lbl.setStyleSheet("color:#888; font-size:11px;")
            w = QLineEdit(self._config.get(key, ""))
            hl.addWidget(lbl); hl.addWidget(w)
            setattr(self, attr, w)
            vl.addLayout(hl)
        hl_btn = QHBoxLayout()
        self.btn_connect    = QPushButton("Connect")
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)
        hl_btn.addWidget(self.btn_connect)
        hl_btn.addWidget(self.btn_disconnect)
        vl.addLayout(hl_btn)
        self.lbl_status = QLabel("disconnected")
        self.lbl_status.setStyleSheet("color:#e53935; font-weight:bold; font-size:11px;")
        vl.addWidget(self.lbl_status)
        self.lbl_fps = QLabel("FPS: —")
        self.lbl_fps.setStyleSheet("color:#aaa; font-size:11px;")
        vl.addWidget(self.lbl_fps)
        return grp

    def _build_rb_grp(self):
        grp = QGroupBox("Rigid Bodies")
        grp.setStyleSheet(_GRP_STYLE)
        grid = QGridLayout(grp)
        grid.setSpacing(4)
        grid.setContentsMargins(8, 4, 8, 8)
        for col, txt in enumerate(["En", "ID", "Position (m)"]):
            lbl = QLabel(txt)
            lbl.setStyleSheet("color:#666; font-size:10px;")
            grid.addWidget(lbl, 0, col)
        self._rb_checks, self._rb_spins, self._rb_lbls = [], [], []
        for i in range(_N_RB):
            chk = QCheckBox()
            chk.setChecked(i == 0)
            chk.setStyleSheet(
                f"QCheckBox::indicator{{border:1px solid {_RB_COLOR[i]}; width:12px; height:12px;}}"
                f"QCheckBox::indicator:checked{{background:{_RB_COLOR[i]};}}")
            spin = QSpinBox()
            spin.setRange(1, 99)
            spin.setValue(i + 1)
            spin.setFixedWidth(42)
            lbl = QLabel("—")
            lbl.setStyleSheet(f"color:{_RB_COLOR[i]}; font-size:10px;")
            grid.addWidget(chk, i+1, 0)
            grid.addWidget(spin, i+1, 1)
            grid.addWidget(lbl, i+1, 2)
            self._rb_checks.append(chk)
            self._rb_spins.append(spin)
            self._rb_lbls.append(lbl)
        return grp

    def _build_motion_grp(self):
        grp = QGroupBox("Motion")
        grp.setStyleSheet(_GRP_STYLE)
        vl = QVBoxLayout(grp)
        vl.setSpacing(4)
        vl.setContentsMargins(8, 4, 8, 8)
        self.lbl_state = QLabel("IDLE")
        self.lbl_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_state.setFixedHeight(30)
        self.lbl_state.setStyleSheet("font-size:16px; font-weight:bold; color:#aaa;")
        vl.addWidget(self.lbl_state)
        hl = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.setFixedHeight(32)
        self.btn_start.setStyleSheet(
            "background:#1b5e20; color:white; font-weight:bold; border-radius:3px;")
        self.btn_start.setEnabled(False)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFixedHeight(32)
        self.btn_stop.setStyleSheet(
            "background:#b71c1c; color:white; font-weight:bold; border-radius:3px;")
        self.btn_stop.setEnabled(False)
        hl.addWidget(self.btn_start); hl.addWidget(self.btn_stop)
        vl.addLayout(hl)
        hl2 = QHBoxLayout()
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedHeight(24)
        self.btn_clear.setStyleSheet("color:#aaa; border:1px solid #444; border-radius:2px;")
        self.btn_save = QPushButton("Save CSV")
        self.btn_save.setFixedHeight(24)
        self.btn_save.setStyleSheet("color:#aaa; border:1px solid #444; border-radius:2px;")
        self.btn_save.setEnabled(False)
        hl2.addWidget(self.btn_clear); hl2.addWidget(self.btn_save)
        vl.addLayout(hl2)
        return grp

    def _build_results_grp(self):
        grp = QGroupBox("Pose Delta & Eigen")
        grp.setStyleSheet(_GRP_STYLE)
        vl = QVBoxLayout(grp)
        vl.setSpacing(2)
        vl.setContentsMargins(6, 4, 6, 6)
        self._res_widgets = []
        row_defs = [
            ('dx','dx(mm)'),('dy','dy(mm)'),('dz','dz(mm)'),
            ('dr','droll'),('dp','dpitch'),('dyw','dyaw'),
            ('angle','angle'),('axis','axis'),
        ]
        for i in range(_N_RB):
            hdr = QLabel(f"── RB {i+1}")
            hdr.setStyleSheet(f"color:{_RB_COLOR[i]}; font-size:11px; font-weight:bold;")
            vl.addWidget(hdr)
            grid = QGridLayout()
            grid.setSpacing(2)
            grid.setContentsMargins(4, 0, 4, 2)
            labels = {}
            for r, (key, txt) in enumerate(row_defs):
                lk = QLabel(f"{txt}:")
                lk.setStyleSheet("color:#555; font-size:10px;")
                lk.setFixedWidth(52)
                lv = QLabel("—")
                lv.setStyleSheet("color:#ccc; font-size:10px; font-weight:bold;")
                grid.addWidget(lk, r, 0)
                grid.addWidget(lv, r, 1)
                labels[key] = lv
            vl.addLayout(grid)
            self._res_widgets.append(labels)
        return grp

    def _build_scale_grp(self):
        grp = QGroupBox("Plot Scale")
        grp.setStyleSheet(_GRP_STYLE)
        vl = QVBoxLayout(grp)
        vl.setSpacing(3)
        vl.setContentsMargins(8, 4, 8, 8)
        self._scale_spins = {}
        for label, key, default in [
            ("cx (m):", "cx", 0.0), ("cy (m):", "cy", 0.0),
            ("cz (m):", "cz", 0.0), ("Range (m):", "range", 1.0),
        ]:
            hl = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(64)
            lbl.setStyleSheet("color:#888; font-size:10px;")
            sp = QDoubleSpinBox()
            sp.setRange(-100.0, 100.0)
            sp.setDecimals(3)
            sp.setSingleStep(0.1)
            sp.setValue(default)
            sp.valueChanged.connect(self._redraw)
            hl.addWidget(lbl); hl.addWidget(sp)
            vl.addLayout(hl)
            self._scale_spins[key] = sp
        return grp

    # ──────────────────────────────────────────────────────────────────────────
    # NatNet
    # ──────────────────────────────────────────────────────────────────────────

    def _active_ids(self):
        return [self._rb_spins[i].value()
                for i in range(_N_RB) if self._rb_checks[i].isChecked()]

    def _rb_index(self, rb_id: int):
        for i in range(_N_RB):
            if self._rb_checks[i].isChecked() and self._rb_spins[i].value() == rb_id:
                return i
        return None

    def _on_connect(self):
        server = self.edit_server.text().strip()
        client = self.edit_client.text().strip() or 'auto'
        ids    = self._active_ids()
        if not server:
            QMessageBox.warning(self, "Error", "Server IP를 입력하세요.")
            return
        if not ids:
            QMessageBox.warning(self, "Error", "하나 이상의 RB를 활성화하세요.")
            return
        if self._natnet_worker and self._natnet_worker.isRunning():
            return
        fv = tuple(self._config['natnet_force_version']) \
             if self._config.get('natnet_force_version') else None
        self._natnet_worker = NatNetWorker(
            server_ip=server, client_ip=client,
            rigid_body_id=ids, force_version=fv, parent=self)
        self._natnet_worker.connected.connect(self._on_connected)
        self._natnet_worker.disconnected.connect(self._on_disconnected)
        self._natnet_worker.error.connect(lambda m: self._log(f"[NatNet] {m}"))
        self._natnet_worker.rb_updated.connect(self._on_rb_updated)
        self._natnet_worker.fps_updated.connect(self._on_fps)
        self.btn_connect.setEnabled(False)
        self.lbl_status.setText("connecting …")
        self.lbl_status.setStyleSheet("color:#f0a500; font-weight:bold; font-size:11px;")
        self._natnet_worker.start()

    def _on_disconnect(self):
        if self._natnet_worker:
            self._natnet_worker.stop()
        self.btn_disconnect.setEnabled(False)

    def _on_connected(self):
        self.lbl_status.setText("connected")
        self.lbl_status.setStyleSheet("color:#00c853; font-weight:bold; font-size:11px;")
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(True)
        self._log("NatNet connected  RBs=" + str(self._active_ids()))
        self._refresh_buttons()

    def _on_disconnected(self):
        self.lbl_status.setText("disconnected")
        self.lbl_status.setStyleSheet("color:#e53935; font-weight:bold; font-size:11px;")
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self._log("NatNet disconnected.")
        self._refresh_buttons()

    def _on_fps(self, fps: float):
        self.lbl_fps.setText(f"FPS: {fps:.1f} Hz")

    def _on_rb_updated(self, rb_id: int, pos: list, quat: list):
        idx = self._rb_index(rb_id)
        if idx is None:
            return
        buf = self._rolling[idx]
        buf.append((list(pos), list(quat)))
        if len(buf) > _MAX_BUF:
            buf.pop(0)
        if self._state == 'MOVING':
            self._traj[idx].append((list(pos), list(quat)))
        self._rb_lbls[idx].setText(
            f"{pos[0]*1000:+6.1f} {pos[1]*1000:+6.1f} {pos[2]*1000:+6.1f}")

    def _snapshot(self, i: int):
        """가장 최근 샘플을 그대로 반환. 데이터 없으면 None."""
        buf = self._rolling[i]
        if not buf:
            return None
        pos, quat = buf[-1]
        return (list(pos), list(quat))

    # ──────────────────────────────────────────────────────────────────────────
    # 모션 제어
    # ──────────────────────────────────────────────────────────────────────────

    def _refresh_buttons(self):
        connected = (self._natnet_worker is not None and
                     self._natnet_worker.isRunning())
        if self._state == 'IDLE':
            self.btn_start.setEnabled(connected)
            self.btn_stop.setEnabled(False)
        else:
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)

    def _on_start(self):
        if self._state != 'IDLE':
            return
        ts_start = datetime.datetime.now()
        for i in range(_N_RB):
            self._traj[i]  = []
            self._start[i] = None
            if self._rb_checks[i].isChecked():
                snap = self._snapshot(i)
                if snap:
                    self._start[i] = (snap[0], snap[1], ts_start)
                    self._log(f"  RB{i+1} start: "
                              f"pos=[{snap[0][0]*1000:.1f},{snap[0][1]*1000:.1f},{snap[0][2]*1000:.1f}]mm"
                              f"  @{ts_start.strftime('%H:%M:%S.%f')[:12]}")
        missing = [i for i in range(_N_RB)
                   if self._rb_checks[i].isChecked() and self._start[i] is None]
        if missing:
            QMessageBox.warning(self, "No data",
                f"RB{[i+1 for i in missing]}: NatNet 데이터 없음.")
            return
        self._state = 'MOVING'
        self.lbl_state.setText("MOVING")
        self.lbl_state.setStyleSheet("font-size:16px; font-weight:bold; color:#00c853;")
        self._log("Motion started.")
        self._refresh_buttons()

    def _on_stop(self):
        if self._state != 'MOVING':
            return
        self._state = 'IDLE'
        ts_stop = datetime.datetime.now()
        for i in range(_N_RB):
            if not self._rb_checks[i].isChecked() or self._start[i] is None:
                self._result[i] = None
                continue
            snap = self._snapshot(i)
            if snap is None:
                self._log(f"  RB{i+1}: stop 스냅샷 실패 (데이터 없음)")
                self._result[i] = None
                continue
            self._log(f"  RB{i+1} stop: traj={len(self._traj[i])}프레임  "
                      f"pos=[{snap[0][0]*1000:.1f},{snap[0][1]*1000:.1f},{snap[0][2]*1000:.1f}]mm"
                      f"  @{ts_stop.strftime('%H:%M:%S.%f')[:12]}")
            try:
                self._result[i] = self._compute_delta(
                    i, np.array(snap[0]), snap[1], ts_stop)
            except Exception as e:
                self._log(f"  RB{i+1}: delta 계산 오류 — {e}")
                self._result[i] = None
        self.lbl_state.setText("IDLE")
        self.lbl_state.setStyleSheet("font-size:16px; font-weight:bold; color:#aaa;")
        self._log("Motion stopped.")
        self.btn_save.setEnabled(any(r is not None for r in self._result))
        self._refresh_buttons()
        self._update_result_labels()
        self._redraw()

    def _on_clear(self):
        self._state   = 'IDLE'
        self._result  = [None] * _N_RB
        self._start   = [None] * _N_RB
        for i in range(_N_RB):
            self._traj[i] = []
        self.lbl_state.setText("IDLE")
        self.lbl_state.setStyleSheet("font-size:16px; font-weight:bold; color:#aaa;")
        for labels in self._res_widgets:
            for lv in labels.values():
                lv.setText("—")
        self.btn_save.setEnabled(False)
        self._refresh_buttons()
        self._redraw()

    # ──────────────────────────────────────────────────────────────────────────
    # 계산
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_delta(self, idx: int, stop_pos: np.ndarray,
                       stop_quat: list, ts_stop: datetime.datetime) -> dict:
        sp, sq, ts_start = self._start[idx]
        start_rot = Rotation.from_quat(sq)
        stop_rot  = Rotation.from_quat(stop_quat)
        dp  = (stop_pos - np.array(sp)) * 1000.0
        rel = start_rot.inv() * stop_rot
        yaw, pitch, roll = rel.as_euler('ZYX', degrees=True)

        R = rel.as_matrix()
        evals, evecs = np.linalg.eig(R)
        best = int(np.argmin(np.abs(evals - 1.0)))
        axis = evecs[:, best].real
        axis /= np.linalg.norm(axis) + 1e-12
        angle_deg = float(np.degrees(np.max(np.abs(np.angle(evals)))))

        res = {
            'rb_idx': idx,
            'timestamp_start': ts_start.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'timestamp_stop':  ts_stop.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'duration_s': (ts_stop - ts_start).total_seconds(),
            'dx_mm': float(dp[0]), 'dy_mm': float(dp[1]), 'dz_mm': float(dp[2]),
            'droll_deg': float(roll), 'dpitch_deg': float(pitch), 'dyaw_deg': float(yaw),
            'angle_deg': angle_deg,
            'axis_x': float(axis[0]), 'axis_y': float(axis[1]), 'axis_z': float(axis[2]),
            'eigenvalues': evals.tolist(),
        }
        self._log(
            f"  RB{idx+1}: Δ=[{dp[0]:+.2f},{dp[1]:+.2f},{dp[2]:+.2f}]mm  "
            f"angle={angle_deg:.3f}°  axis=[{axis[0]:+.3f},{axis[1]:+.3f},{axis[2]:+.3f}]")
        return res

    def _update_result_labels(self):
        for i, res in enumerate(self._result):
            labels = self._res_widgets[i]
            if res is None:
                for lv in labels.values():
                    lv.setText("—")
                continue
            labels['dx'].setText(f"{res['dx_mm']:+.3f} mm")
            labels['dy'].setText(f"{res['dy_mm']:+.3f} mm")
            labels['dz'].setText(f"{res['dz_mm']:+.3f} mm")
            labels['dr'].setText(f"{res['droll_deg']:+.3f}°")
            labels['dp'].setText(f"{res['dpitch_deg']:+.3f}°")
            labels['dyw'].setText(f"{res['dyaw_deg']:+.3f}°")
            labels['angle'].setText(f"{res['angle_deg']:.4f}°")
            labels['axis'].setText(
                f"[{res['axis_x']:+.3f},{res['axis_y']:+.3f},{res['axis_z']:+.3f}]")

    # ──────────────────────────────────────────────────────────────────────────
    # 플롯
    # ──────────────────────────────────────────────────────────────────────────

    _T = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)  # left-hand Y-up

    def _p(self, pos):
        return pos[0], -pos[2], pos[1]

    def _redraw(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111, projection='3d')
        ax.set_facecolor(_DARK_BG)
        ax.tick_params(colors='#aaa', labelsize=7)
        for sp in [ax.xaxis, ax.yaxis, ax.zaxis]:
            sp.pane.fill = False
            sp.pane.set_edgecolor('#333')

        cx   = self._scale_spins['cx'].value()
        cy   = self._scale_spins['cy'].value()
        cz   = self._scale_spins['cz'].value()
        half = max(self._scale_spins['range'].value() / 2.0, 0.05)
        ax.set_xlim(cx  - half, cx  + half)
        ax.set_ylim(-cz - half, -cz + half)
        ax.set_zlim(cy  - half, cy  + half)
        arrow_len = half / 4

        any_drawn = False
        for i in range(_N_RB):
            pts = self._traj[i]
            if not pts:
                continue
            color = _RB_COLOR[i]
            xs = [ p[0][0] for p in pts]
            ys = [-p[0][2] for p in pts]
            zs = [ p[0][1] for p in pts]
            ax.plot(xs, ys, zs, '-', color=color, lw=1.0, alpha=0.4,
                    label=f"RB{i+1}")

            # 좌표계 프레임 (최대 6개)
            step = max(1, len(pts) // 6)
            frames = pts[::step]
            if pts[-1] is not frames[-1]:
                frames = frames + [pts[-1]]
            t_vals = np.linspace(0.35, 1.0, len(frames))
            for k, (pos, quat) in enumerate(frames):
                R_plot = self._T @ Rotation.from_quat(quat).as_matrix()
                px, py, pz = self._p(pos)
                fade = t_vals[k]
                for col, avec in zip(('#e53935','#43a047','#1e88e5'), R_plot.T):
                    ax.quiver(px, py, pz, avec[0], avec[1], avec[2],
                              length=arrow_len, normalize=True,
                              color=col, alpha=fade,
                              linewidth=1.0, arrow_length_ratio=0.2)

            ax.scatter(*self._p(pts[0][0]),  color=color, s=50, marker='*', zorder=10)
            ax.scatter(*self._p(pts[-1][0]), color=color, s=50, marker='X', zorder=10)
            any_drawn = True

        if any_drawn:
            ax.legend(fontsize=8, labelcolor='white',
                      facecolor='#2a2a4e', edgecolor='#444', loc='upper left')

        ax.set_xlabel('X (m)', color='#aaa', fontsize=8, labelpad=4)
        ax.set_ylabel('Z (m)', color='#aaa', fontsize=8, labelpad=4)
        ax.set_zlabel('Y (m)', color='#aaa', fontsize=8, labelpad=4)
        ax.set_title('Trajectory  (Left-hand  Y-up)', color='white', fontsize=10, pad=8)
        self._canvas.draw()

    # ──────────────────────────────────────────────────────────────────────────
    # 저장
    # ──────────────────────────────────────────────────────────────────────────

    def _on_save(self):
        stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = str(self._config.get('root_path', '.')) + f'/positioner_{stamp}.csv'
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", default, "CSV Files (*.csv)")
        if not path:
            return

        rows = []
        for i, res in enumerate(self._result):
            if res is None:
                continue
            evals = res['eigenvalues']

            def _ev(v, part):
                if hasattr(v, 'real'):
                    return v.real if part == 're' else v.imag
                return v if part == 're' else 0.0

            rows.append({
                'rb': i+1, 'rb_id': self._rb_spins[i].value(),
                'timestamp_start': res['timestamp_start'],
                'timestamp_stop':  res['timestamp_stop'],
                'duration_s':      res['duration_s'],
                'dx_mm': res['dx_mm'], 'dy_mm': res['dy_mm'], 'dz_mm': res['dz_mm'],
                'droll_deg': res['droll_deg'], 'dpitch_deg': res['dpitch_deg'],
                'dyaw_deg': res['dyaw_deg'], 'angle_deg': res['angle_deg'],
                'axis_x': res['axis_x'], 'axis_y': res['axis_y'], 'axis_z': res['axis_z'],
                'eval1_re': _ev(evals[0],'re'), 'eval1_im': _ev(evals[0],'im'),
                'eval2_re': _ev(evals[1],'re'), 'eval2_im': _ev(evals[1],'im'),
                'eval3_re': _ev(evals[2],'re'), 'eval3_im': _ev(evals[2],'im'),
            })

            traj = self._traj[i]
            if traj:
                tp = path.replace('.csv', f'_rb{i+1}_traj.csv')
                ts_start_str = res['timestamp_start']
                with open(tp, 'w', newline='', encoding='utf-8-sig') as f:
                    w = csv.DictWriter(f, fieldnames=[
                        'frame', 'timestamp_start', 'x_m', 'y_m', 'z_m',
                        'qx', 'qy', 'qz', 'qw'])
                    w.writeheader()
                    for fr, (pos, quat) in enumerate(traj):
                        w.writerow({
                            'frame': fr, 'timestamp_start': ts_start_str,
                            'x_m': pos[0], 'y_m': pos[1], 'z_m': pos[2],
                            'qx': quat[0], 'qy': quat[1], 'qz': quat[2], 'qw': quat[3],
                        })
                self._log(f"Traj → {pathlib.Path(tp).name}")

        if rows:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            self._log(f"Results → {pathlib.Path(path).name}")

    # ──────────────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        log.info(msg)
        self.log_panel.appendPlainText(msg)

    def closeEvent(self, event):
        if self._natnet_worker and self._natnet_worker.isRunning():
            try:
                self._natnet_worker.blockSignals(True)
            except Exception:
                pass
            self._natnet_worker.stop()
            self._natnet_worker.wait(3000)
        super().closeEvent(event)


# ──────────────────────────────────────────────────────────────────────────────

def _avg_quat(quats: list) -> list:
    arr = np.array(quats, dtype=float)
    ref = arr[0]
    for i in range(1, len(arr)):
        if np.dot(arr[i], ref) < 0:
            arr[i] = -arr[i]
    avg = arr.mean(axis=0)
    n = np.linalg.norm(avg)
    return (avg / n).tolist() if n > 1e-9 else ref.tolist()
