'''
DRT Tool Window
@Author Byunghun Hwang<bh.hwang@iae.re.kr>
'''

try:
    # using PyQt6
    from PyQt6.QtGui import QImage, QPixmap, QCloseEvent, QStandardItem, QStandardItemModel
    from PyQt6.QtWidgets import QApplication, QFrame, QMainWindow, QLabel, QPushButton, QCheckBox, QComboBox, QDialog
    from PyQt6.QtWidgets import QMessageBox, QProgressBar, QFileDialog, QComboBox, QLineEdit, QSlider, QVBoxLayout
    from PyQt6.QtWidgets import QInputDialog
    from PyQt6.uic import loadUi
    from PyQt6.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal, QRegularExpression
except ImportError:
    print("PyQt6 is required to run this application.")

import os, sys
import pathlib
import json
import importlib
import inspect
import subprocess
import threading

from util.logger.console import ConsoleLogger
from common.config_loader import load_config
from plugins.pluginbase.plannerbase import PlannerBase
from plugins.pluginbase.optimizerbase import OptimizerBase
from plugins.pathplanner import Q_SPACE_PLANNER_MODULES
from plugins.pathplanner.ompl import SUPPORTED_ALGORITHMS as OMPL_SUPPORTED_ALGORITHMS
from plugins.robotics.inspection_workflow import InspectionWorkflowState
from plugins.robotics.plan_run_io import default_run_dir, save_plan_run
from simtool.inspection_path_handler import InspectionPathHandler, resolve_planner_timeout
from simtool.inspection_sequencer import InspectionSequencer
from simtool.param import SimParameterMap


class AppWindow(QMainWindow):
    # 모션 속도/가속 (사다리꼴 프로파일). kind별 단위: lin=m, rot=deg(전송 시 rad 변환)
    _LIN_SPEED, _LIN_ACCEL = 1.0, 2.0               # m/s, m/s^2
    _ROT_SPEED, _ROT_ACCEL = 120.0, 240.0            # deg/s, deg/s^2
    _LIN_RES, _ROT_RES = 0.01, 1.0                  # 슬라이더 분해능 (m, deg)

    _SOURCE_ROBOT = "rb20_1900es"
    _DDA_ROBOT = "dda_rb10_1300e"

    # 관절 테이블: (slider, edit, joint, kind, lo, hi)  kind: 'lin'(m) | 'rot'(deg)
    _SOURCE_JOINTS = [
        ('slider_source_base_pos',   'edit_source_base_pos',   'rt_joint_linear_track', 'lin',   0.0,  4.87),
        ('slider_source_base_pos_2', 'edit_source_base_pos_2', 'rt_joint_carriage',     'lin',   0.0,  0.7),
        ('slider_source_base_pos_3', 'edit_source_base_pos_3', 'rt_base',               'rot', -180.0, 180.0),
        ('slider_source_base_pos_4', 'edit_source_base_pos_4', 'rt_shoulder',           'rot', -180.0, 180.0),
        ('slider_source_base_pos_5', 'edit_source_base_pos_5', 'rt_elbow',              'rot', -180.0, 180.0),
        ('slider_source_base_pos_6', 'edit_source_base_pos_6', 'rt_wrist1',             'rot', -180.0, 180.0),
        ('slider_source_base_pos_7', 'edit_source_base_pos_7', 'rt_wrist2',             'rot', -180.0, 180.0),
        ('slider_source_base_pos_8', 'edit_source_base_pos_8', 'rt_wrist3',             'rot', -180.0, 180.0),
    ]
    _DDA_JOINTS = [
        ('slider_source_base_pos_9',  'edit_source_base_pos_9',  'dda_joint_linear_track', 'lin',   0.0,  4.87),
        ('slider_source_base_pos_10', 'edit_source_base_pos_10', 'dda_joint_carriage',     'lin',   0.0,  0.7),
        ('slider_source_base_pos_11', 'edit_source_base_pos_11', 'dda_joint_base',         'rot', -180.0, 180.0),
        ('slider_source_base_pos_12', 'edit_source_base_pos_12', 'dda_joint_shoulder',     'rot', -180.0, 180.0),
        ('slider_source_base_pos_13', 'edit_source_base_pos_13', 'dda_joint_elbow',        'rot', -180.0, 180.0),
        ('slider_source_base_pos_14', 'edit_source_base_pos_14', 'dda_joint_wrist1',       'rot', -180.0, 180.0),
        ('slider_source_base_pos_15', 'edit_source_base_pos_15', 'dda_joint_wrist2',       'rot', -180.0, 180.0),
        ('slider_source_base_pos_16', 'edit_source_base_pos_16', 'dda_joint_wrist3',       'rot', -180.0, 180.0),
    ]

    def __init__(self, config:dict, zpipe):
        """ initialization """
        super().__init__()
        
        self.__console = ConsoleLogger.get_logger()
        self.__config = config
        self.zpipe = zpipe
        self.zapi = None
        self._chuck_mount_points = []
        self._chuck_mount_local_points = []
        # SimTool's own config (simtool.cfg) has no ef_pose section; the EF frames (RT pipe_facing_axis) live in
        # viewervedo.cfg. Falling back to the hard-coded default [0,-1,0] made the reachable/rotation-needed split
        # use a different axis than the pose determination and the Viewer (which read viewervedo.cfg), so the
        # planning phases came out wrong. Read it from there when simtool.cfg does not define it.
        def _rt_frame(cfg):
            return (((cfg.get("ef_pose", {}) or {}).get("frames", {}) or {}).get("rt", {}) or {})
        rt_frame_cfg = _rt_frame(config)
        if "pipe_facing_axis" not in rt_frame_cfg:
            try:
                rt_frame_cfg = _rt_frame(self.__load_viewer_config())
            except Exception as exc:
                self.__console.warning(f"could not read RT pipe_facing_axis from viewervedo.cfg: {exc}")
        self._inspection_workflow = InspectionWorkflowState(
            rt_pipe_facing_axis=list(rt_frame_cfg.get("pipe_facing_axis", [0.0, -1.0, 0.0]))
        )
        self._inspection_path_handler = InspectionPathHandler(
            config, self._inspection_workflow)
        self._inspection_sequencer = InspectionSequencer(self.__console)

        try:            
            if "gui" in config:
                # load UI
                ui_path = pathlib.Path(config["app_path"]) / config["gui"]
                if os.path.isfile(ui_path):
                    loadUi(ui_path, self)
                    self.setWindowTitle(config.get("window_title", "DRT Simulation Tool"))

                    # Initialize SimParameterMap for UI sync
                    self._sim_param_map = SimParameterMap(self, self.__config, self.__console)
                    self.__hide_mesh_reconstruction_ui()

                    # Load plugins and samples
                    self.__load_plugins()

                    # connect UI componens signals
                    self.__connect_signals()

                else:
                    raise Exception(f"Cannot found UI file : {ui_path}")
            
            # Initialize ZAPI and Start ZAPI
            if self.zpipe:
                from simtool.zapi import ZAPI
                zapi_config = self.__config.get("zapi", {})
                transport = zapi_config.get("transport", "ipc")
                channel = zapi_config.get("channel", "/tmp/viewervedo")
                
                self.zapi = ZAPI(zpipe=self.zpipe, transport=transport, channel=channel)
                self.zapi.signal_message_received.connect(self._handle_message)
                self.zapi.run()
                self.__console.info("Now ZAPI is running for SimTool")
            else:
                 self.__console.error("ZPipe instance missing, ZAPI not started")
                
        except Exception as e:
            self.__console.error(f"{e}")

    def __hide_mesh_reconstruction_ui(self):
        """Hide experimental point-cloud processing controls from the main UI."""
        for widget_name in (
            'label_proc_sor',
            'label_sor_neighbors',
            'spin_pcd_sor_neighbors',
            'label_sor_std',
            'spin_pcd_sor_std_ratio',
            'btn_pcd_sor_filter',
            'label_proc_ccl',
            'label_ccl_level',
            'spin_pcd_ccl_level',
            'label_ccl_min',
            'spin_pcd_ccl_min_points',
            'btn_pcd_ccl_filter',
            'label_proc_mesh',
            'label_mesh_res',
            'spin_mesh_resolution',
            'label_mesh_sigma',
            'spin_mesh_sigma',
            'label_mesh_level',
            'spin_mesh_level',
            'btn_mesh_convert',
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.hide()

    def __load_plugins(self):
        """
        Load plugins and populate comboboxes
        """
        
        # Load Path Planners and Optimizers
        plugin_categories = [
            {
                "name": "PathPlanner",
                "path": "python/plugins/pathplanner",
                "package_prefix": "plugins.pathplanner",
                "base_class": PlannerBase,
                "combobox": getattr(self, 'cbx_plugin_pathplanner', None)
            },
            {
                "name": "Optimizer",
                "path": "python/plugins/optimizer",
                "package_prefix": "plugins.optimizer",
                "base_class": OptimizerBase,
                "combobox": getattr(self, 'cbx_plugin_optimizer', None)
            }
        ]

        for category in plugin_categories:
            combobox = category["combobox"]
            
            if combobox is not None:
                try:
                    combobox.clear()
                    if category["name"] == "Optimizer":
                        combobox.addItem("None", "")
                    elif category["name"] == "PathPlanner":
                        for algorithm in OMPL_SUPPORTED_ALGORITHMS:
                            combobox.addItem(f"OMPL: {algorithm}", algorithm)
                        # falls through below to also add the legacy per-file
                        # q-space planners (rrt, rrt_connect, direct_path, ...)

                    root_path = self.__config.get("root_path", "")
                    plugin_path = pathlib.Path(root_path) / category["path"]
                    
                    if plugin_path.exists():
                        files = list(plugin_path.glob("*.py"))
                        
                        for file_path in files:
                            if file_path.name == "__init__.py":
                                continue
                            if (
                                category["name"] == "PathPlanner"
                                and file_path.stem not in Q_SPACE_PLANNER_MODULES
                            ):
                                continue
                            module_name = f"{category['package_prefix']}.{file_path.stem}"
                            try:
                                module = importlib.import_module(module_name)
                                for name, obj in inspect.getmembers(module):
                                    if (
                                        inspect.isclass(obj)
                                        and issubclass(obj, category["base_class"])
                                        and obj.__module__ == module.__name__
                                    ):
                                        if obj is not category["base_class"]:
                                            self.__console.debug(f"Found plugin class: {obj.__name__}")
                                            combobox.addItem(obj.__name__, file_path.stem)
                            except Exception as e:
                                self.__console.error(f"Failed to load {category['name']} plugin {module_name}: {e}")
                    else:
                        self.__console.warning(f"{category['name']} plugin directory not found: {plugin_path}")

                except Exception as e:
                    import traceback
                    self.__console.error(f"Error processing category {category['name']}: {e}")
                    self.__console.error(traceback.format_exc())

        # 3. Load Pipe Spool Samples and Test Weld Points
        # Show the configured load default beside the pipe selector.
        if hasattr(self, 'label_3'):
            try:
                load_cfg = self.__load_viewer_config().get("spool_load", {}) or {}
                ratio = float(load_cfg.get("downsample_keep_ratio", 1.0) or 1.0)
                step = max(1, int(round(1.0 / ratio))) if 0.0 < ratio < 1.0 else 1
                self.label_3.setText(f"Pipe Spool — 기본 유지: 1/{step} ({100 / step:g}%)")
                self.label_3.setFixedWidth(310)
                cap = int(load_cfg.get("visual_max_points", 0))
                self.label_3.setToolTip(
                    f"viewervedo.cfg 로드 기본값: 점군 {step}개마다 1개 유지.\n"
                    f"화면 표시 상한: {cap if cap > 0 else '없음'}. "
                    "충돌 메쉬 입력에도 유지 비율 적용. 변경 후 viewer 재시작 및 배관 재로드 필요.")
            except Exception as exc:
                self.__console.warning(f"could not read spool sampling defaults: {exc}")
        sample_path = pathlib.Path(self.__config.get("root_path", "")) / "sample"
        
        if hasattr(self, 'cbx_pipe_spool'):
            self.cbx_pipe_spool.clear()
            if sample_path.exists():
                for file_path in sample_path.iterdir():
                    if file_path.suffix.lower() in ['.pcd', '.ply']:
                        self.cbx_pipe_spool.addItem(file_path.name)
            else:
                self.__console.warning(f"Sample directory not found: {sample_path}")

        if hasattr(self, 'cbx_test_weld_point'):
            self.cbx_test_weld_point.clear()
            if sample_path.exists():
                # Add a default blank item if you want, or just add all csv
                for file_path in sample_path.iterdir():
                    if file_path.suffix.lower() == '.csv':
                        self.cbx_test_weld_point.addItem(file_path.name)

    def __connect_signals(self):
        """Connect UI signals"""
        if hasattr(self, 'btn_load_spool'):
            self.btn_load_spool.clicked.connect(self.__on_btn_load_spool_clicked)
            if not hasattr(self, 'btn_spool_flip_x'):
                self.btn_flip_spool_x = QPushButton("Flip X", self.btn_load_spool.parent())
                self.btn_flip_spool_x.setObjectName("btn_flip_spool_x")
                geo = self.btn_load_spool.geometry()
                self.btn_flip_spool_x.setGeometry(geo.x() + geo.width() + 8, geo.y(), 76, geo.height())
                self.btn_flip_spool_x.clicked.connect(self.__on_btn_flip_spool_x_clicked)
                self.btn_flip_spool_x.show()
        if hasattr(self, 'btn_spool_flip_x'):
            self.btn_spool_flip_x.clicked.connect(self.__on_btn_flip_spool_x_clicked)
        if hasattr(self, 'btn_spool_position_move'):
            self.btn_spool_position_move.clicked.connect(self.__on_btn_spool_position_move_clicked)
        if hasattr(self, 'btn_spool_pose_save'):
            self.btn_spool_pose_save.clicked.connect(self.__on_btn_spool_pose_save_clicked)
        if hasattr(self, 'btn_spool_pose_load'):
            self.btn_spool_pose_load.clicked.connect(self.__on_btn_spool_pose_load_clicked)
        if hasattr(self, 'btn_load_test_weld_point'):
            self.btn_load_test_weld_point.clicked.connect(self.__on_btn_load_test_weld_point_clicked)
        if hasattr(self, 'btn_load_sim_parameters'):
            self.btn_load_sim_parameters.clicked.connect(self.__on_btn_load_sim_parameters_clicked)
        if hasattr(self, 'btn_test_async_zapi_request'):
            self.btn_test_async_zapi_request.clicked.connect(self.on_btn_test_async_zapi_request_clicked)
        if hasattr(self, 'btn_start_simulation'):
            self.btn_start_simulation.clicked.connect(self.__on_btn_start_simulation_clicked)
        self.__ensure_load_playback_button()
        if hasattr(self, 'btn_load_playback_result'):
            self.btn_load_playback_result.clicked.connect(self.__on_btn_load_playback_result_clicked)
        if hasattr(self, 'btn_save_plan_result'):
            self.btn_save_plan_result.clicked.connect(self.__on_btn_save_plan_result_clicked)
        if hasattr(self, 'btn_pick_inspection_point'):
            self.btn_pick_inspection_point.clicked.connect(self.__on_btn_pick_inspection_point_clicked)
        self.__ensure_determine_ef_pose_button()
        if hasattr(self, 'btn_determine_ef_pose'):
            self.btn_determine_ef_pose.clicked.connect(self.__on_btn_determine_ef_pose_clicked)
        if hasattr(self, 'btn_check_inspection_ik'):
            self.btn_check_inspection_ik.clicked.connect(self.__on_btn_check_inspection_ik_clicked)
        if hasattr(self, 'btn_save_inspection_points'):
            self.btn_save_inspection_points.clicked.connect(self.__on_btn_save_inspection_points_clicked)
        if hasattr(self, 'btn_load_inspection_points'):
            self.btn_load_inspection_points.clicked.connect(self.__on_btn_load_inspection_points_clicked)
        if hasattr(self, 'btn_save_planning_snapshot'):
            self.btn_save_planning_snapshot.clicked.connect(self.__on_btn_save_planning_snapshot_clicked)
        if hasattr(self, 'btn_load_planning_snapshot'):
            self.btn_load_planning_snapshot.clicked.connect(self.__on_btn_load_planning_snapshot_clicked)
        self.__ensure_chuck_mount_pick_button()
        if hasattr(self, 'btn_align_f_column'):
            self.btn_align_f_column.clicked.connect(self.__on_btn_align_f_column_clicked)
        if hasattr(self, 'btn_align_m_column'):
            self.btn_align_m_column.clicked.connect(self.__on_btn_align_m_column_clicked)
        if hasattr(self, 'btn_apply_chuck_mount_cfg'):
            self.btn_apply_chuck_mount_cfg.clicked.connect(self.__on_btn_apply_chuck_mount_cfg_clicked)
        if hasattr(self, 'btn_save_chuck_mount_cfg'):
            self.btn_save_chuck_mount_cfg.clicked.connect(self.__on_btn_save_chuck_mount_cfg_clicked)
        if hasattr(self, 'btn_fix_chuck_mount'):
            self.btn_fix_chuck_mount.clicked.connect(self.__on_btn_fix_chuck_mount_clicked)
        if hasattr(self, 'btn_release_chuck_mount'):
            self.btn_release_chuck_mount.clicked.connect(self.__on_btn_release_chuck_mount_clicked)
        self.__hide_legacy_spool_fix_controls()
        if hasattr(self, 'btn_plan_inspection_path'):
            self.btn_plan_inspection_path.clicked.connect(self.__on_btn_plan_inspection_path_clicked)
        if hasattr(self, 'btn_clear_inspection_path'):
            self.btn_clear_inspection_path.clicked.connect(self.__on_btn_clear_inspection_path_clicked)
            
        if hasattr(self, 'radio_mode_simulation'):
            self.radio_mode_simulation.toggled.connect(self.__on_radio_mode_toggled)
        if hasattr(self, 'radio_mode_real'):
            self.radio_mode_real.toggled.connect(self.__on_radio_mode_toggled)

        if hasattr(self, 'btn_positioner_x_move'):
            self.btn_positioner_x_move.clicked.connect(self.__on_btn_positioner_x_move_clicked)
        if hasattr(self, 'btn_positioner_z_move'):
            self.btn_positioner_z_move.clicked.connect(self.__on_btn_positioner_z_move_clicked)
        if hasattr(self, 'btn_positioner_r_move'):
            self.btn_positioner_r_move.clicked.connect(self.__on_btn_positioner_r_move_clicked)
        if hasattr(self, 'btn_positioner_clamp_move'):
            self.btn_positioner_clamp_move.clicked.connect(self.__on_btn_positioner_clamp_move_clicked)

        if hasattr(self, 'btn_mesh_convert'):
            self.btn_mesh_convert.clicked.connect(self.__on_btn_mesh_convert_clicked)
        if hasattr(self, 'btn_pcd_sor_filter'):
            self.btn_pcd_sor_filter.clicked.connect(self.__on_btn_pcd_sor_filter_clicked)
        if hasattr(self, 'btn_pcd_ccl_filter'):
            self.btn_pcd_ccl_filter.clicked.connect(self.__on_btn_pcd_ccl_filter_clicked)
        if hasattr(self, 'btn_pcd_save'):
            self.btn_pcd_save.clicked.connect(self.__on_btn_pcd_save_clicked)

        # 매니퓰레이터(Source/DDA) 전체 관절 = 보간 애니메이션 모션
        self.__wire_manipulator(self._SOURCE_ROBOT, self._SOURCE_JOINTS,
                                'btn_robot_source_move', 'btn_robot_source_stop')
        self.__wire_manipulator(self._DDA_ROBOT, self._DDA_JOINTS,
                                'btn_robot_dda_move', 'btn_robot_dda_stop')
        if hasattr(self, 'btn_robot_source_base'):
            self.btn_robot_source_base.clicked.connect(
                lambda _=False: self.__reset_robot_base_pose(self._SOURCE_ROBOT))
        if hasattr(self, 'btn_robot_dda_base'):
            self.btn_robot_dda_base.clicked.connect(
                lambda _=False: self.__reset_robot_base_pose(self._DDA_ROBOT))

        # 스풀 고정 체크박스 → 수동 컨트롤(슬라이더/이동) 잠금 토글
        if hasattr(self, 'chk_spool_fix_f_column_r'):
            self.chk_spool_fix_f_column_r.toggled.connect(self.__update_spool_controls_enabled)
            self.chk_spool_fix_f_column_r.toggled.connect(self.__on_spool_fixation_toggled)
        if hasattr(self, 'chk_spool_fix_m_column_z'):
            self.chk_spool_fix_m_column_z.toggled.connect(self.__update_spool_controls_enabled)
            self.chk_spool_fix_m_column_z.toggled.connect(self.__on_spool_fixation_toggled)
        self.__update_spool_controls_enabled()  # 초기 상태 반영

    def __get_spool_fix_flags(self):
        fix_f_column_r = (hasattr(self, 'chk_spool_fix_f_column_r') and
                          self.chk_spool_fix_f_column_r.isChecked())
        fix_m_column_z = (hasattr(self, 'chk_spool_fix_m_column_z') and
                          self.chk_spool_fix_m_column_z.isChecked())
        if hasattr(self, '_spool_mount_fixed'):
            fixed = bool(self._spool_mount_fixed)
            return fixed, fixed
        return fix_f_column_r, fix_m_column_z

    def __hide_legacy_spool_fix_controls(self):
        if not hasattr(self, '_spool_mount_fixed'):
            self._spool_mount_fixed = False
        for name in (
            'groupBox_spool_fix',
            'chk_spool_fix_f_column_r',
            'chk_spool_fix_m_column_z',
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.hide()

    def __spool_move_blocked_by_fix(self):
        """스풀이 링크에 고정(r 또는 m 중 하나라도)되어 있으면 스풀 수동 이동을 차단."""
        fix_f, fix_z = self.__get_spool_fix_flags()
        if fix_f or fix_z:
            self.__console.warning(
                "스풀이 링크에 고정되어 있어 스풀 이동이 차단되었습니다. "
                "(Spool Fixation 체크 해제 필요)")
            return True
        return False

    def __update_spool_controls_enabled(self, *args):
        """스풀 고정 시 수동 컨트롤(슬라이더/입력/버튼)을 비활성화."""
        fix_f, fix_z = self.__get_spool_fix_flags()
        enabled = not (fix_f or fix_z)
        widget_names = [
            'slider_spool_x_pos', 'slider_spool_y_pos', 'slider_spool_z_pos',
            'slider_spool_x_rot', 'slider_spool_z_rot',
            'edit_spool_x_pos', 'edit_spool_y_pos', 'edit_spool_z_pos',
            'edit_spool_x_rot', 'edit_spool_z_rot',
            'btn_spool_position_move', 'btn_spool_flip_x', 'btn_flip_spool_x',
        ]
        for name in widget_names:
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(enabled)
        positioner_locked_widgets = [
            'slider_positioner_x_pos',
            'edit_positioner_x_pos',
            'edit_positioner_x_vel',
            'btn_positioner_x_move',
            'btn_positioner_x_stop',
            'slider_positioner_clamp_pos',
            'edit_positioner_clamp_pos',
            'edit_positioner_clamp_vel',
            'btn_positioner_clamp_move',
            'btn_positioner_clamp_stop',
        ]
        for name in positioner_locked_widgets:
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(enabled)
        for name in (
            'slider_positioner_z_pos',
            'edit_positioner_z_pos',
            'edit_positioner_z_vel',
            'btn_positioner_z_move',
            'btn_positioner_z_stop',
            'slider_positioner_r_pos',
            'edit_positioner_r_pos',
            'edit_positioner_r_vel',
            'btn_positioner_r_move',
            'btn_positioner_r_stop',
        ):
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(True)

    def __on_spool_fixation_toggled(self, *args):
        fix_f, fix_z = self.__get_spool_fix_flags()
        if self.zapi:
            self.zapi._ZAPI_request_set_spool_fixation(fix_f, fix_z)

    def __on_btn_positioner_x_move_clicked(self):
        try:
            if bool(getattr(self, '_spool_mount_fixed', False)):
                self.__set_path_plan_status("Mount fixed: only R/Z axes can move")
                return
            pos = float(self.edit_positioner_x_pos.text() or "0")
            vel = float(self.edit_positioner_x_vel.text() or "0")
            fix_f, fix_z = self.__get_spool_fix_flags()
            if self.zapi:
                self.zapi._ZAPI_request_move_positioner("x", pos, vel, fix_f, fix_z)
        except (ValueError, AttributeError) as e:
            self.__console.error(f"Error moving positioner X: {e}")

    def __on_btn_positioner_z_move_clicked(self):
        try:
            pos = float(self.edit_positioner_z_pos.text() or "0")
            vel = float(self.edit_positioner_z_vel.text() or "0")
            fix_f, fix_z = self.__get_spool_fix_flags()
            if self.zapi:
                self.zapi._ZAPI_request_move_positioner("z", pos, vel, fix_f, fix_z)
        except (ValueError, AttributeError) as e:
            self.__console.error(f"Error moving positioner Z: {e}")

    def __on_btn_positioner_r_move_clicked(self):
        try:
            pos = float(self.edit_positioner_r_pos.text() or "0")
            vel = float(self.edit_positioner_r_vel.text() or "0")
            fix_f, fix_z = self.__get_spool_fix_flags()
            if self.zapi:
                self.zapi._ZAPI_request_move_positioner("r", pos, vel, fix_f, fix_z)
        except (ValueError, AttributeError) as e:
            self.__console.error(f"Error moving positioner R: {e}")

    # Positioner clamp joint (f_column_r_to_f_column_passive_clamp) URDF
    # limit is -0.9..0.0 m; UI convention sends this as 0..0.9 and negates it
    # before writing the joint (see visualizer.py's
    # "prismatic y-axis, range -0.9~0; UI value 0~0.9 maps to joint =
    # -position" comment). The automatic F-column alignment solver already
    # clips to this range (visualizer.py bounds[3]=(0.0, 0.9)); manual text
    # entry here had no such check, so a typo/out-of-range value went
    # straight to the joint with no rejection or clamping.
    _CLAMP_POS_MIN = 0.0
    _CLAMP_POS_MAX = 0.9

    def __clamp_positioner_clamp_value(self, pos: float) -> float:
        clamped = max(self._CLAMP_POS_MIN, min(self._CLAMP_POS_MAX, pos))
        if clamped != pos:
            self.__console.warning(
                f"positioner clamp value {pos} out of range "
                f"[{self._CLAMP_POS_MIN}, {self._CLAMP_POS_MAX}] - clamped to {clamped}")
        return clamped

    def __on_btn_positioner_clamp_move_clicked(self):
        try:
            if bool(getattr(self, '_spool_mount_fixed', False)):
                self.__set_path_plan_status("Mount fixed: only R/Z axes can move")
                return
            pos = self.__clamp_positioner_clamp_value(float(self.edit_positioner_clamp_pos.text() or "0"))
            vel = float(self.edit_positioner_clamp_vel.text() or "0")
            fix_f, fix_z = self.__get_spool_fix_flags()
            if self.zapi:
                self.zapi._ZAPI_request_move_positioner("clamp", pos, vel, fix_f, fix_z)
        except (ValueError, AttributeError) as e:
            self.__console.error(f"Error moving positioner clamp: {e}")

    def __on_btn_spool_position_move_clicked(self):
        """Handle manual spool position move button click."""
        try:
            pose = self.__get_spool_pose_from_ui()
            self.__request_spool_pose_move(pose)
        except (ValueError, AttributeError) as e:
            self.__console.error(f"Error moving spool position: {e}")

    def __get_spool_pose_from_ui(self):
        return {
            "x": float(self.edit_spool_x_pos.text() or "0"),
            "y": float(self.edit_spool_y_pos.text() or "0"),
            "z": float(self.edit_spool_z_pos.text() or "0"),
            "x_rotation": float(self.edit_spool_x_rot.text() or "0"),
            "z_rotation": float(self.edit_spool_z_rot.text() or "0"),
        }

    def __get_positioner_pose_from_ui(self):
        return {
            "x": float(self.edit_positioner_x_pos.text() or "0"),
            "z": float(self.edit_positioner_z_pos.text() or "0"),
            "r": float(self.edit_positioner_r_pos.text() or "0"),
            "clamp": self.__clamp_positioner_clamp_value(float(self.edit_positioner_clamp_pos.text() or "0")),
        }

    def __set_positioner_pose_to_ui(self, pose):
        if not pose:
            return None
        positioner = pose.get("positioner", pose)
        x = float(positioner.get("x", 0.0))
        z = float(positioner.get("z", 0.0))
        r = float(positioner.get("r", 0.0))
        clamp = float(positioner.get("clamp", 0.0))
        if hasattr(self, '_sim_param_map'):
            self._sim_param_map.set_positioner_values(x=x, z=z, r=r, clamp=clamp)
        else:
            self.edit_positioner_x_pos.setText(f"{x:.3f}")
            self.edit_positioner_z_pos.setText(f"{z:.3f}")
            self.edit_positioner_r_pos.setText(f"{r:.3f}")
            self.edit_positioner_clamp_pos.setText(f"{clamp:.3f}")
        return {
            "x": x,
            "z": z,
            "r": r,
            "clamp": clamp,
        }

    def __set_spool_pose_to_ui(self, pose):
        x = float(pose.get("x", pose.get("position", [0.0, 0.0, 0.0])[0]))
        y = float(pose.get("y", pose.get("position", [0.0, 0.0, 0.0])[1]))
        z = float(pose.get("z", pose.get("position", [0.0, 0.0, 0.0])[2]))
        x_rotation = float(pose.get("x_rotation", 0.0))
        z_rotation = float(pose.get("z_rotation", 0.0))
        if hasattr(self, '_sim_param_map'):
            self._sim_param_map.set_spool_pose_values(x, y, z, x_rotation, z_rotation)
        else:
            self.edit_spool_x_pos.setText(f"{x:.3f}")
            self.edit_spool_y_pos.setText(f"{y:.3f}")
            self.edit_spool_z_pos.setText(f"{z:.3f}")
            self.edit_spool_x_rot.setText(f"{x_rotation:.3f}")
            self.edit_spool_z_rot.setText(f"{z_rotation:.3f}")
        return {
            "x": x,
            "y": y,
            "z": z,
            "x_rotation": x_rotation,
            "z_rotation": z_rotation,
        }

    def __get_robot_joint_state_from_ui(self):
        import math
        robots = {}

        def read_table(robot_name, table):
            joints = {}
            for _slider_name, edit_name, joint_name, kind, _lo, _hi in table:
                edit = getattr(self, edit_name, None)
                if edit is None:
                    continue
                try:
                    ui_value = float(edit.text() or "0")
                except ValueError:
                    continue
                joints[joint_name] = math.radians(ui_value) if kind == 'rot' else ui_value
            if joints:
                robots[robot_name] = joints

        read_table(self._SOURCE_ROBOT, self._SOURCE_JOINTS)
        read_table(self._DDA_ROBOT, self._DDA_JOINTS)
        return robots

    def __request_spool_pose_move(self, pose, force=False):
        if not force and self.__spool_move_blocked_by_fix():
            return
        if self.zapi:
            self.zapi._ZAPI_request_move_spool(
                pose["x"],
                pose["y"],
                pose["z"],
                pose.get("x_rotation", 0.0),
                pose.get("z_rotation", 0.0),
            )
            self.__console.info(
                f"Requested to move spool pose: x={pose['x']}, y={pose['y']}, z={pose['z']}, "
                f"x_rotation={pose.get('x_rotation', 0.0)}, z_rotation={pose.get('z_rotation', 0.0)}")
        else:
            self.__console.error("ZAPI instance not available")

    def __request_positioner_pose_move(self, pose):
        if not pose:
            return
        if self.zapi:
            fix_f, fix_z = self.__get_spool_fix_flags()
            if bool(getattr(self, '_spool_mount_fixed', False)):
                self.zapi._ZAPI_request_move_positioner("z", pose["z"], 0.0, fix_f, fix_z)
                self.zapi._ZAPI_request_move_positioner("r", pose["r"], 0.0, fix_f, fix_z)
                self.__console.info(
                    f"Requested to move fixed positioner Z/R only: z={pose['z']}, r={pose['r']}")
                return
            self.zapi._ZAPI_request_move_positioner("x", pose["x"], 0.0, fix_f, fix_z)
            self.zapi._ZAPI_request_move_positioner("z", pose["z"], 0.0, fix_f, fix_z)
            self.zapi._ZAPI_request_move_positioner("r", pose["r"], 0.0, fix_f, fix_z)
            self.zapi._ZAPI_request_move_positioner("clamp", pose["clamp"], 0.0, fix_f, fix_z)
            self.__console.info(
                f"Requested to move positioner pose: x={pose['x']}, z={pose['z']}, "
                f"r={pose['r']}, clamp={pose['clamp']}")
        else:
            self.__console.error("ZAPI instance not available")

    def __get_current_spool_path(self):
        if hasattr(self, '_current_spool_path') and self._current_spool_path:
            return pathlib.Path(self._current_spool_path)
        if hasattr(self, 'cbx_pipe_spool'):
            current_text = self.cbx_pipe_spool.currentText()
            if current_text:
                return pathlib.Path(self.__config.get("root_path", "")) / "sample" / current_text
        return None

    def __get_spool_pose_path(self, spool_path=None):
        if spool_path is None:
            spool_path = self.__get_current_spool_path()
        if spool_path is None:
            return None
        return pathlib.Path(spool_path).with_suffix(".json")

    def __save_spool_pose(self, spool_path=None):
        pose_path = self.__get_spool_pose_path(spool_path)
        if pose_path is None:
            self.__console.warning("Cannot save spool pose: no spool file selected")
            return
        pose = self.__get_spool_pose_from_ui()
        positioner_pose = self.__get_positioner_pose_from_ui()
        fix_f, fix_z = self.__get_spool_fix_flags()
        payload = {
            "spool_file": pathlib.Path(spool_path or self.__get_current_spool_path()).name,
            "spool": {
                "x": pose["x"],
                "y": pose["y"],
                "z": pose["z"],
                "x_rotation": pose["x_rotation"],
                "z_rotation": pose["z_rotation"],
            },
            "positioner": {
                "x": positioner_pose["x"],
                "z": positioner_pose["z"],
                "r": positioner_pose["r"],
                "clamp": positioner_pose["clamp"],
            },
            # 고정(체크) 상태 — spool 포즈는 chuck 기준 오프셋이다
            "fix_f_column_r": bool(fix_f),
            "fix_m_column_z": bool(fix_z),
            "fixation": {
                "fixed": bool(fix_f or fix_z),
                "fix_f_column_r": bool(fix_f),
                "fix_m_column_z": bool(fix_z),
            },
            "chuck_mount_points": {
                "points": getattr(self, '_chuck_mount_points', []),
                "local_points": getattr(self, '_chuck_mount_local_points', []),
            },
            "x": pose["x"],
            "y": pose["y"],
            "z": pose["z"],
            "x_rotation": pose["x_rotation"],
            "z_rotation": pose["z_rotation"],
        }
        with open(pose_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        self.__console.info(f"Saved spool pose: {pose_path}")

    def __load_spool_pose(self, spool_path=None, apply_move=False):
        pose_path = self.__get_spool_pose_path(spool_path)
        if pose_path is None:
            self.__console.warning("Cannot load spool pose: no spool file selected")
            return False
        if not pose_path.exists():
            self.__console.info(f"No spool pose file found: {pose_path}")
            return False
        with open(pose_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        positioner_pose = self.__set_positioner_pose_to_ui(payload.get("positioner"))
        has_spool_pose = (
            "spool" in payload
            or "position" in payload
            or any(k in payload for k in ("x", "y", "z", "x_rotation", "z_rotation"))
        )
        pose = self.__set_spool_pose_to_ui(payload.get("spool", payload)) if has_spool_pose else None
        self.__set_chuck_mount_points_from_payload(payload.get("chuck_mount_points"))
        self.__console.info(f"Loaded spool pose: {pose_path}")

        # 고정 체크박스 상태 먼저 복원 → 포지셔너 이동 시 r-fix 동기화가 반영되도록
        self.__set_spool_fix_checks(
            bool(payload.get("fix_f_column_r", False)),
            bool(payload.get("fix_m_column_z", False)))
        if apply_move:
            # 1) 포지셔너 먼저 이동 → chuck 위치/회전 확정 (r-fix면 passive r 동기화)
            if positioner_pose:
                self.__request_positioner_pose_move(positioner_pose)
            # 2) 스풀 오프셋(chuck 기준) 적용 (고정 가드 무시하고 강제 적용)
            if pose is not None:
                self.__request_spool_pose_move(pose, force=True)
            self.__request_chuck_mount_points_render()
        return True

    def __set_chuck_mount_points_from_payload(self, payload):
        if not payload:
            self._chuck_mount_points = []
            self._chuck_mount_local_points = []
            return
        if isinstance(payload, dict):
            points = payload.get("points", [])
            local_points = payload.get("local_points", [])
        else:
            points = payload
            local_points = []
        self._chuck_mount_points = points or []
        self._chuck_mount_local_points = local_points or []

    def __request_chuck_mount_points_render(self):
        if not self.zapi or not getattr(self, '_chuck_mount_points', None):
            return
        self.zapi._ZAPI_request_set_chuck_mount_points(
            self._chuck_mount_points,
            getattr(self, '_chuck_mount_local_points', None))

    def __set_spool_fix_checks(self, fix_f, fix_z):
        """고정 체크박스 상태를 시그널 없이 설정하고 컨트롤 잠금만 갱신."""
        self._spool_mount_fixed = bool(fix_f or fix_z)
        for name, val in (('chk_spool_fix_f_column_r', fix_f),
                          ('chk_spool_fix_m_column_z', fix_z)):
            chk = getattr(self, name, None)
            if chk is not None:
                blocked = chk.blockSignals(True)
                chk.setChecked(bool(val))
                chk.blockSignals(blocked)
        self.__update_spool_controls_enabled()

    def __on_btn_spool_pose_save_clicked(self):
        try:
            self.__save_spool_pose()
        except Exception as e:
            self.__console.error(f"Error saving spool pose: {e}")

    def __on_btn_spool_pose_load_clicked(self):
        try:
            self.__load_spool_pose(apply_move=True)
        except Exception as e:
            self.__console.error(f"Error loading spool pose: {e}")

    def __on_radio_mode_toggled(self, checked=False):
        """Handle execution mode radio button toggle"""
        # PyQt toggle signal emits twice: once for unchecked, once for checked
        if not checked:
            return
            
        if getattr(self, 'zapi', None):
            if hasattr(self, 'radio_mode_simulation') and self.radio_mode_simulation.isChecked():
                self.zapi._ZAPI_request_set_mode("simulation")
            elif hasattr(self, 'radio_mode_real') and self.radio_mode_real.isChecked():
                self.zapi._ZAPI_request_set_mode("real")

    def on_btn_test_async_zapi_request_clicked(self):
        """Handle async ZAPI test request button click"""
        pass

    def __current_planner_module_name(self):
        if not hasattr(self, 'cbx_plugin_pathplanner'):
            return "rrt_connect"
        module_name = self.cbx_plugin_pathplanner.currentData()
        if module_name:
            planner_name = str(module_name)
        else:
            text = self.cbx_plugin_pathplanner.currentText()
            planner_name = text.strip().lower() if text else "rrt_connect"
        return planner_name

    def __current_optimizer_module_name(self):
        if not hasattr(self, 'cbx_plugin_optimizer'):
            return None
        module_name = self.cbx_plugin_optimizer.currentData()
        if module_name:
            return str(module_name)
        text = self.cbx_plugin_optimizer.currentText()
        optimizer_name = text.strip().lower() if text else ""
        if optimizer_name in {"", "none"}:
            return None
        return optimizer_name

    def __path_optimization_enabled(self):
        if hasattr(self, 'chk_path_optimization'):
            return bool(self.chk_path_optimization.isChecked())
        return self.__current_optimizer_module_name() is not None

    def __current_path_robot_name(self):
        if hasattr(self, 'cbx_path_robot'):
            data = self.cbx_path_robot.currentData()
            if data:
                return str(data)
            text = self.cbx_path_robot.currentText()
            if "DDA" in text.upper():
                return self._DDA_ROBOT
        return self._SOURCE_ROBOT

    def __current_ik_normalize_enabled(self):
        if hasattr(self, 'chk_ik_normalize'):
            return bool(self.chk_ik_normalize.isChecked())
        return True

    def __current_ik_solver_name(self):
        if hasattr(self, 'cbx_ik_solver'):
            data = self.cbx_ik_solver.currentData()
            if data:
                return str(data)
            text = self.cbx_ik_solver.currentText()
            if text:
                return str(text).strip().lower()
        return "collision_free"

    def __current_ik_request_options(self):
        solver = self.__current_ik_solver_name()
        if solver == "dls":
            return solver, False
        return solver, self.__current_ik_normalize_enabled()

    def __set_path_plan_status(self, msg):
        if hasattr(self, 'label_path_plan_status'):
            self.label_path_plan_status.setText(str(msg))
        self.__console.info(str(msg))

    def __ensure_determine_ef_pose_button(self):
        if not hasattr(self, 'btn_pick_inspection_point') or not hasattr(self, 'btn_clear_inspection_path'):
            return
        parent = self.btn_clear_inspection_path.parent()
        if parent is not None and hasattr(parent, 'geometry'):
            parent_geo = parent.geometry()
            # Must clear btn_save_planning_snapshot's bottom edge (y=522, h=32)
            # plus a bit of margin, or that button gets visually clipped by
            # the group box's border.
            target_height = 692
            if parent_geo.height() < target_height:
                parent.setGeometry(parent_geo.x(), parent_geo.y(), parent_geo.width(), target_height)
        self.btn_pick_inspection_point.setGeometry(10, 348, 101, 32)
        if not hasattr(self, 'btn_determine_ef_pose'):
            self.btn_determine_ef_pose = QPushButton("EF Pose", parent)
            self.btn_determine_ef_pose.setObjectName("btn_determine_ef_pose")
        self.btn_determine_ef_pose.setGeometry(116, 348, 101, 32)
        if not hasattr(self, 'chk_show_rejected_ef'):
            self.chk_show_rejected_ef = QCheckBox("충돌 후보", parent)
            self.chk_show_rejected_ef.setObjectName("chk_show_rejected_ef")
            self.chk_show_rejected_ef.setChecked(True)
            self.chk_show_rejected_ef.setToolTip("충돌로 탈락한 EF 자세와 충돌 지점 표시. 채택된 자세 표시는 유지합니다.")
            self.chk_show_rejected_ef.toggled.connect(self.__on_show_rejected_ef_toggled)
        self.chk_show_rejected_ef.setGeometry(10, 586, 120, 26)
        self.chk_show_rejected_ef.show()
        if not hasattr(self, 'chk_spool_collision_mesh'):
            self.chk_spool_collision_mesh = QCheckBox("파이프 충돌 메쉬", parent)
            self.chk_spool_collision_mesh.setToolTip("체크: 실제 충돌 검사 메쉬 / 해제: 원래 점군 표시")
            self.chk_spool_collision_mesh.toggled.connect(self.__on_spool_collision_mesh_toggled)
        self.chk_spool_collision_mesh.setGeometry(150, 586, 170, 26)
        self.chk_spool_collision_mesh.show()
        if not hasattr(self, 'btn_check_inspection_ik'):
            self.btn_check_inspection_ik = QPushButton("Check IK", parent)
            self.btn_check_inspection_ik.setObjectName("btn_check_inspection_ik")
        self.btn_check_inspection_ik.setGeometry(10, 386, 100, 32)
        if hasattr(self, 'btn_plan_inspection_path'):
            self.btn_plan_inspection_path.setGeometry(114, 386, 104, 32)
        if not hasattr(self, 'btn_stop_planning'):
            self.btn_stop_planning = QPushButton("Stop Planning", parent)
            self.btn_stop_planning.setObjectName("btn_stop_planning")
            self.btn_stop_planning.setToolTip(
                "진행 중인 경로 계획을 종료한다: 남은 목표는 보내지 않고, Robot Core에서 계산 중인 목표는 중단(작업 "
                "프로세스 재시작)한다. 그때까지 계획된 결과는 저장되고 플레이백할 수 있다.")
            self.btn_stop_planning.clicked.connect(self.__on_btn_stop_planning_clicked)
        self.btn_stop_planning.setGeometry(222, 386, 99, 32)
        if not hasattr(self, 'chk_path_optimization'):
            self.chk_path_optimization = QCheckBox("Optimize", parent)
            self.chk_path_optimization.setObjectName("chk_path_optimization")
            self.chk_path_optimization.setChecked(True)
        self.chk_path_optimization.setGeometry(245, 280, 80, 26)
        if not hasattr(self, 'label_ik_solver'):
            self.label_ik_solver = QLabel("IK Solver", parent)
            self.label_ik_solver.setObjectName("label_ik_solver")
        self.label_ik_solver.setGeometry(10, 424, 80, 24)
        if not hasattr(self, 'cbx_ik_solver'):
            self.cbx_ik_solver = QComboBox(parent)
            self.cbx_ik_solver.setObjectName("cbx_ik_solver")
            self.cbx_ik_solver.addItem("DLS", "dls")
            self.cbx_ik_solver.addItem("QP IK", "qp")
            self.cbx_ik_solver.addItem("PyBullet IK", "pybullet")
            self.cbx_ik_solver.addItem("Collision-free IK", "collision_free")
            self.cbx_ik_solver.setCurrentIndex(3)  # default: Collision-free IK
        self.cbx_ik_solver.setGeometry(92, 424, 125, 24)
        if not hasattr(self, 'chk_ik_normalize'):
            self.chk_ik_normalize = QCheckBox("Normalize IK joints", parent)
            self.chk_ik_normalize.setObjectName("chk_ik_normalize")
            self.chk_ik_normalize.setChecked(True)
        self.chk_ik_normalize.setGeometry(224, 424, 97, 24)
        if not hasattr(self, 'chk_lock_linear_track'):
            self.chk_lock_linear_track = QCheckBox("Lock track to target X", parent)
            self.chk_lock_linear_track.setObjectName("chk_lock_linear_track")
            self.chk_lock_linear_track.setChecked(False)
            self.chk_lock_linear_track.setToolTip(
                "체크하면 linear_track을 target pose의 world x 위치로 먼저 이동/고정한 뒤 "
                "나머지 joint만 path planning (탐색 공간 축소 - "
                "InspectionPlanningBase.plan_q_path_for_robot의 lock_linear_track). "
                "팔이 큰 자세 차이 중일 때 이 중간 자세가 실제로 충돌할 수 있으니 기본은 꺼짐.")
        # Own row (not appended to the IK-solver row) - that row already runs
        # edge-to-edge with the group box's width, so a widget tacked on to
        # its right was rendered past the visible border and never showed up.
        self.chk_lock_linear_track.setGeometry(10, 452, 165, 24)
        if not hasattr(self, 'chk_show_saved_ik'):
            self.chk_show_saved_ik = QCheckBox("Show saved IK", parent)
            self.chk_show_saved_ik.setObjectName("chk_show_saved_ik")
            self.chk_show_saved_ik.setChecked(True)
            self.chk_show_saved_ik.setToolTip(
                "스냅샷에 저장된 IK 목표 자세(precompute_ik_goals.py의 ik_goal_q)를 반투명 로봇으로 표시한다 "
                "(DDA 주황 / RT 하늘색, 충돌 링크 빨강). 포지셔너 회전 후 계획되는 group은 회전 후에 보인다. "
                "Viewer에서 F8로도 전환.")
            self.chk_show_saved_ik.toggled.connect(self.__on_chk_show_saved_ik_toggled)
        self.chk_show_saved_ik.setGeometry(180, 452, 141, 24)
        self.chk_show_saved_ik.show()
        self.btn_clear_inspection_path.setGeometry(222, 348, 99, 32)
        if hasattr(self, 'edit_inspection_point'):
            self.edit_inspection_point.setGeometry(10, 484, 171, 22)
        if hasattr(self, 'label_path_plan_status'):
            self.label_path_plan_status.setGeometry(188, 484, 133, 22)
        if not hasattr(self, 'btn_save_inspection_points'):
            self.btn_save_inspection_points = QPushButton("Save Points", parent)
            self.btn_save_inspection_points.setObjectName("btn_save_inspection_points")
        self.btn_save_inspection_points.setGeometry(10, 514, 155, 32)
        if not hasattr(self, 'btn_load_inspection_points'):
            self.btn_load_inspection_points = QPushButton("Load Points", parent)
            self.btn_load_inspection_points.setObjectName("btn_load_inspection_points")
        self.btn_load_inspection_points.setGeometry(168, 514, 153, 32)
        if not hasattr(self, 'btn_save_planning_snapshot'):
            self.btn_save_planning_snapshot = QPushButton("Save Planning Snapshot", parent)
            self.btn_save_planning_snapshot.setObjectName("btn_save_planning_snapshot")
            self.btn_save_planning_snapshot.setToolTip(
                "결정된 EF pose + collision scene을 벤치마킹용 snapshot(.pkl)으로 저장")
        self.btn_save_planning_snapshot.setText("Save Snapshot")
        self.btn_save_planning_snapshot.setGeometry(10, 550, 153, 32)
        if not hasattr(self, 'btn_load_planning_snapshot'):
            self.btn_load_planning_snapshot = QPushButton("Load Snapshot", parent)
            self.btn_load_planning_snapshot.setObjectName("btn_load_planning_snapshot")
            self.btn_load_planning_snapshot.setToolTip(
                "저장된 planning snapshot(.pkl)으로 배관/positioner/로봇 상태와 EF pose를 복원한다 "
                "(복원 후 바로 Plan Inspection Path 실행 가능). 신뢰하는 파일만 선택할 것")
        self.btn_load_planning_snapshot.setGeometry(168, 550, 153, 32)
        if not hasattr(self, 'btn_random_snapshot_points'):
            self.btn_random_snapshot_points = QPushButton("Random 5 Points", parent)
            self.btn_random_snapshot_points.setToolTip("스냅샷에서 포인트 5개를 무작위 선택해 불러옵니다. 포인트별 모든 DDA/RT 자세는 유지합니다.")
            self.btn_random_snapshot_points.clicked.connect(self.__on_random_snapshot_points)
        self.btn_random_snapshot_points.setGeometry(10, 620, 311, 30)
        self.btn_random_snapshot_points.show()
        if not hasattr(self, 'chk_auto_rotate_positioner'):
            self.chk_auto_rotate_positioner = QCheckBox("Auto rotate positioner", parent)
            self.chk_auto_rotate_positioner.setChecked(bool((self.__config.get("path_planning", {}) or {}).get("auto_rotate_positioner", True)))
            self.chk_auto_rotate_positioner.setToolTip("현재 각도에서 접근 가능한 그룹을 먼저 계획한 후, 필요한 그룹은 포지셔너를 회전해 계획합니다. 처음부터 회전이 필요한 그룹만 있으면 바로 회전합니다.")
        self.chk_auto_rotate_positioner.setGeometry(10, 654, 311, 26)
        self.chk_auto_rotate_positioner.show()
        display_group = getattr(self, 'groupBox_3', None)
        if display_group is not None:
            geo = display_group.geometry()
            target_y = 546
            if parent is not None and hasattr(parent, 'geometry'):
                parent_geo = parent.geometry()
                target_y = max(target_y, parent_geo.y() + parent_geo.height() + 14)
            if geo.y() < target_y:
                display_group.setGeometry(geo.x(), target_y, geo.width(), geo.height())
        self.btn_determine_ef_pose.show()
        self.btn_check_inspection_ik.show()
        self.label_ik_solver.show()
        self.cbx_ik_solver.show()
        self.chk_ik_normalize.show()
        self.btn_save_inspection_points.show()
        self.btn_load_inspection_points.show()
        self.btn_save_planning_snapshot.show()
        self.btn_load_planning_snapshot.show()

    def __ensure_load_playback_button(self):
        """Grow the "Operation" groupBox (btn_start_simulation's parent) to
        fit a new button below it - same pattern as __ensure_save_planning_
        snapshot_button/__ensure_chuck_mount_pick_button."""
        if not hasattr(self, 'btn_start_simulation'):
            return
        parent = self.btn_start_simulation.parent()
        sim_geo = self.btn_start_simulation.geometry()
        if parent is not None and hasattr(parent, 'geometry'):
            group_geo = parent.geometry()
            target_height = sim_geo.y() + sim_geo.height() + 8 + 36 + 8 + 36 + 10
            if group_geo.height() < target_height:
                parent.setGeometry(group_geo.x(), group_geo.y(), group_geo.width(), target_height)
        if not hasattr(self, 'btn_load_playback_result'):
            self.btn_load_playback_result = QPushButton("Load Playback Result...", parent)
            self.btn_load_playback_result.setObjectName("btn_load_playback_result")
            self.btn_load_playback_result.setToolTip(
                "test_ompl_planning.py --target all이 저장한 debug/<method>_<timestamp>/ "
                "폴더를 선택해 Start Simulation으로 재생 가능하게 불러온다")
        half_w = (sim_geo.width() - 4) // 2
        self.btn_load_playback_result.setText("Load Result...")
        self.btn_load_playback_result.setGeometry(
            sim_geo.x(), sim_geo.y() + sim_geo.height() + 8, half_w, 36)
        self.btn_load_playback_result.show()
        if not hasattr(self, 'btn_save_plan_result'):
            self.btn_save_plan_result = QPushButton("Save Result...", parent)
            self.btn_save_plan_result.setObjectName("btn_save_plan_result")
            self.btn_save_plan_result.setToolTip(
                "마지막 경로 계획 결과(경로 + 실패 원인)를 폴더로 저장한다. "
                "계획이 끝날 때마다 debug/plan_runs/ 에도 자동 저장된다")
        self.btn_save_plan_result.setGeometry(
            sim_geo.x() + half_w + 4, sim_geo.y() + sim_geo.height() + 8, half_w, 36)
        self.btn_save_plan_result.show()
        if not hasattr(self, 'btn_evaluate_plan_result'):
            self.btn_evaluate_plan_result = QPushButton("Evaluate Result...", parent)
            self.btn_evaluate_plan_result.setObjectName("btn_evaluate_plan_result")
            self.btn_evaluate_plan_result.setToolTip(
                "저장된 경로 계획 결과 폴더(debug/plan_runs/...)를 평가 지표로 평가한다 (evaluate_plan_run.py, "
                "백그라운드 실행). 결과: <폴더>/eval/summary.md, targets.csv")
            self.btn_evaluate_plan_result.clicked.connect(self.__on_btn_evaluate_plan_result_clicked)
        self.btn_evaluate_plan_result.setGeometry(
            sim_geo.x(), sim_geo.y() + sim_geo.height() + 8 + 36 + 8, half_w, 36)
        self.btn_evaluate_plan_result.show()
        if not hasattr(self, 'btn_sac_inference'):
            self.btn_sac_inference = QPushButton("SAC Inference…", parent)
            self.btn_sac_inference.setObjectName("btn_sac_inference")
            self.btn_sac_inference.setToolTip("학습 모델로 저장된 기준 경로를 보정하고 시뮬레이션 결과를 불러옵니다")
            self.btn_sac_inference.clicked.connect(self.__on_sac_inference_clicked)
        self.btn_sac_inference.setGeometry(
            sim_geo.x() + half_w + 4, sim_geo.y() + sim_geo.height() + 8 + 36 + 8, half_w, 36)
        self.btn_sac_inference.show()

    def __on_sac_inference_clicked(self):
        from simtool.sac_inference_dialog import SACInferenceDialog
        dialog = getattr(self, '_sac_inference_dialog', None)
        if dialog is None:
            dialog = SACInferenceDialog(self)
            dialog.load_result.connect(self.__on_btn_load_playback_result_clicked)
            self._sac_inference_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def __ensure_chuck_mount_pick_button(self):
        if not hasattr(self, 'btn_spool_pose_load'):
            return
        parent = self.btn_spool_pose_load.parent()
        group = parent
        if group is not None and hasattr(group, 'geometry'):
            group_geo = group.geometry()
            target_height = 430
            if group_geo.height() < target_height:
                group.setGeometry(group_geo.x(), group_geo.y(), group_geo.width(), target_height)
                fix_group = getattr(self, 'groupBox_spool_fix', None)
                if fix_group is not None:
                    fix_geo = fix_group.geometry()
                    fix_group.setGeometry(fix_geo.x(), group_geo.y() + target_height + 7, fix_geo.width(), fix_geo.height())
        geo = self.btn_spool_pose_load.geometry()
        self.btn_spool_pose_load.setGeometry(geo.x(), geo.y(), geo.width(), geo.height())
        self.btn_spool_pose_load.setText("Load")
        y = geo.y() + geo.height() + 8
        group_width = parent.geometry().width() if hasattr(parent, 'geometry') else 330
        left_x = 10
        gap = 8
        col_w = max(130, int((group_width - left_x * 2 - gap) / 2))

        if not hasattr(self, 'btn_align_f_column'):
            self.btn_align_f_column = QPushButton("Align F column", parent)
            self.btn_align_f_column.setObjectName("btn_align_f_column")
        self.btn_align_f_column.setGeometry(left_x, y, col_w, geo.height())
        self.btn_align_f_column.show()

        if not hasattr(self, 'btn_align_m_column'):
            self.btn_align_m_column = QPushButton("Align M column", parent)
            self.btn_align_m_column.setObjectName("btn_align_m_column")
        self.btn_align_m_column.setGeometry(left_x + col_w + gap, y, col_w, geo.height())
        self.btn_align_m_column.show()

        self.__ensure_chuck_mount_config_controls(parent, left_x, y + geo.height() + 10, col_w, gap)

    def __ensure_chuck_mount_config_controls(self, parent, left_x, y, col_w, gap):
        cfg = self.__load_viewer_chuck_mount_cfg()
        row_h = 24
        label_w = 52
        edit_w = max(74, col_w - label_w - 4)
        fields = (
            ("lbl_f_chuck_offset", "F off", "edit_f_chuck_offset", "f_column", "center_offset", 0),
            ("lbl_m_chuck_offset", "M off", "edit_m_chuck_offset", "m_column", "center_offset", 0),
            ("lbl_f_chuck_axis", "F axis", "edit_f_chuck_axis", "f_column", "axis", 1),
            ("lbl_m_chuck_axis", "M axis", "edit_m_chuck_axis", "m_column", "axis", 1),
        )
        for label_name, label_text, edit_name, column_name, key, row in fields:
            is_m = label_name.startswith("lbl_m")
            x = left_x + (col_w + gap if is_m else 0)
            row_y = y + row * (row_h + 6)
            if not hasattr(self, label_name):
                label = QLabel(label_text, parent)
                label.setObjectName(label_name)
                setattr(self, label_name, label)
            label = getattr(self, label_name)
            label.setGeometry(x, row_y, label_w, row_h)
            label.show()

            if not hasattr(self, edit_name):
                edit = QLineEdit(parent)
                edit.setObjectName(edit_name)
                setattr(self, edit_name, edit)
            edit = getattr(self, edit_name)
            edit.setGeometry(x + label_w, row_y, edit_w, row_h)
            edit.setText(self.__format_chuck_mount_vec(cfg, column_name, key))
            edit.show()

        button_y = y + 2 * (row_h + 6) + 2
        if not hasattr(self, 'btn_apply_chuck_mount_cfg'):
            self.btn_apply_chuck_mount_cfg = QPushButton("Apply chuck cfg", parent)
            self.btn_apply_chuck_mount_cfg.setObjectName("btn_apply_chuck_mount_cfg")
        self.btn_apply_chuck_mount_cfg.setGeometry(left_x, button_y, col_w, 30)
        self.btn_apply_chuck_mount_cfg.show()

        if not hasattr(self, 'btn_save_chuck_mount_cfg'):
            self.btn_save_chuck_mount_cfg = QPushButton("Save chuck cfg", parent)
            self.btn_save_chuck_mount_cfg.setObjectName("btn_save_chuck_mount_cfg")
        self.btn_save_chuck_mount_cfg.setGeometry(left_x + col_w + gap, button_y, col_w, 30)
        self.btn_save_chuck_mount_cfg.show()

        profile_y = button_y + 38
        if not hasattr(self, 'label_chuck_mount_profile'):
            self.label_chuck_mount_profile = QLabel("Cylinder: -", parent)
            self.label_chuck_mount_profile.setObjectName("label_chuck_mount_profile")
            self.label_chuck_mount_profile.setWordWrap(True)
        self.label_chuck_mount_profile.setGeometry(left_x, profile_y, col_w * 2 + gap, 42)
        self.label_chuck_mount_profile.show()

        fix_y = profile_y + 50
        if not hasattr(self, 'btn_fix_chuck_mount'):
            self.btn_fix_chuck_mount = QPushButton("Fix aligned mount", parent)
            self.btn_fix_chuck_mount.setObjectName("btn_fix_chuck_mount")
        self.btn_fix_chuck_mount.setGeometry(left_x, fix_y, col_w, 30)
        self.btn_fix_chuck_mount.show()

        if not hasattr(self, 'btn_release_chuck_mount'):
            self.btn_release_chuck_mount = QPushButton("Release mount", parent)
            self.btn_release_chuck_mount.setObjectName("btn_release_chuck_mount")
        self.btn_release_chuck_mount.setGeometry(left_x + col_w + gap, fix_y, col_w, 30)
        self.btn_release_chuck_mount.show()

    def __viewer_cfg_path(self):
        return pathlib.Path(self.__config.get("root_path", "")) / "python" / "viewervedo.cfg"

    def __load_viewer_config(self):
        path = self.__viewer_cfg_path()
        if path.exists():
            return load_config(path)
        return {}

    def __load_viewer_chuck_mount_cfg(self):
        cfg = self.__load_viewer_config().get("chuck_mount", {}) or {}
        defaults = {
            "f_column": {
                "center_offset": [0.0, 0.0, 0.0],
                "axis": [1.0, 0.0, 0.0],
            },
            "m_column": {
                "center_offset": [0.0, 0.0, 0.0],
                "axis": [-1.0, 0.0, 0.0],
                "profile_align_axis_sign": -1.0,
                "r_rotation_sign": -1.0,
            },
        }
        for column_name, values in defaults.items():
            cfg.setdefault(column_name, {})
            for key, default_value in values.items():
                cfg[column_name].setdefault(key, default_value)
        return cfg

    def __format_chuck_mount_vec(self, cfg, column_name, key):
        values = cfg.get(column_name, {}).get(key, [0.0, 0.0, 0.0])
        return ", ".join(f"{float(v):.6g}" for v in values)

    def __parse_chuck_mount_vec(self, widget_name):
        text = getattr(self, widget_name).text().strip()
        values = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
        if len(values) != 3:
            raise ValueError(f"{widget_name} must have 3 values: x,y,z")
        return [float(value) for value in values]

    def __chuck_mount_cfg_from_ui(self):
        cfg = self.__load_viewer_chuck_mount_cfg()
        cfg.setdefault("f_column", {})
        cfg.setdefault("m_column", {})
        cfg["f_column"]["center_offset"] = self.__parse_chuck_mount_vec("edit_f_chuck_offset")
        cfg["f_column"]["axis"] = self.__parse_chuck_mount_vec("edit_f_chuck_axis")
        cfg["m_column"]["center_offset"] = self.__parse_chuck_mount_vec("edit_m_chuck_offset")
        cfg["m_column"]["axis"] = self.__parse_chuck_mount_vec("edit_m_chuck_axis")
        return cfg

    def __apply_chuck_mount_cfg(self, save=False):
        chuck_cfg = self.__chuck_mount_cfg_from_ui()
        if save:
            viewer_cfg = self.__load_viewer_config()
            viewer_cfg["chuck_mount"] = chuck_cfg
            path = self.__viewer_cfg_path()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(viewer_cfg, f, indent=4)
                f.write("\n")
            self.__console.info(f"Saved chuck mount cfg: {path}")
        if self.zapi:
            self.zapi._ZAPI_request_set_chuck_mount_config(chuck_cfg)
        return chuck_cfg

    def __on_btn_pick_inspection_point_clicked(self):
        try:
            if not self.zapi:
                self.__set_path_plan_status("[!] ZAPI not available")
                return
            self.zapi._ZAPI_request_pick_inspection_point(True, clear=False, multi_select=True)
            self.__set_path_plan_status("Pick mode: click one or more pipe surfaces in viewer")
        except Exception as e:
            self.__console.error(f"Error requesting inspection point pick: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_align_f_column_clicked(self):
        try:
            if not self.zapi:
                self.__set_path_plan_status("[!] ZAPI not available")
                return
            self.zapi._ZAPI_request_pick_chuck_mount_points(
                True, clear=True, align_on_pick=True, align_target="f")
            self.__set_path_plan_status("Align F: click pipe point for f-column")
        except Exception as e:
            self.__console.error(f"Error requesting f-column align pick: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_align_m_column_clicked(self):
        try:
            if not self.zapi:
                self.__set_path_plan_status("[!] ZAPI not available")
                return
            self.zapi._ZAPI_request_pick_chuck_mount_points(
                True, clear=True, align_on_pick=True, align_target="m")
            self.__set_path_plan_status("Align M: click pipe point for m-column")
        except Exception as e:
            self.__console.error(f"Error requesting m-column align pick: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_fix_chuck_mount_clicked(self):
        try:
            self.__set_spool_fix_checks(True, True)
            if self.zapi:
                self.zapi._ZAPI_request_set_spool_fixation(True, True)
            self.__set_path_plan_status("Mount fixed: R/Z axes drive pipe")
        except Exception as e:
            self.__console.error(f"Error fixing chuck mount: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_release_chuck_mount_clicked(self):
        try:
            self.__set_spool_fix_checks(False, False)
            if self.zapi:
                self.zapi._ZAPI_request_set_spool_fixation(False, False)
            self.__set_path_plan_status("Mount released")
        except Exception as e:
            self.__console.error(f"Error releasing chuck mount: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_apply_chuck_mount_cfg_clicked(self):
        try:
            self.__apply_chuck_mount_cfg(save=False)
            self.__set_path_plan_status("Chuck mount cfg applied")
        except Exception as e:
            self.__console.error(f"Error applying chuck mount cfg: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_save_chuck_mount_cfg_clicked(self):
        try:
            self.__apply_chuck_mount_cfg(save=True)
            self.__set_path_plan_status("Chuck mount cfg saved")
        except Exception as e:
            self.__console.error(f"Error saving chuck mount cfg: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_save_inspection_points_clicked(self):
        try:
            if not self.zapi:
                self.__set_path_plan_status("[!] ZAPI not available")
                return
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Inspection Points",
                str(pathlib.Path(self.__config.get("root_path", "")) / "sample"),
                "JSON Files (*.json)"
            )
            if not file_name:
                return
            self.zapi._ZAPI_request_save_inspection_points(file_name)
            self.__set_path_plan_status(f"Inspection points save requested -> {pathlib.Path(file_name).name}")
        except Exception as e:
            self.__console.error(f"Error saving inspection points: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_save_planning_snapshot_clicked(self):
        try:
            if not self.zapi:
                self.__set_path_plan_status("[!] ZAPI not available")
                return
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Planning Snapshot",
                str(pathlib.Path(self.__config.get("root_path", "")) / "sample"),
                "Planning Snapshot Files (*.pkl)"
            )
            if not file_name:
                return
            if pathlib.Path(file_name).suffix.lower() not in (".pkl", ".pickle"):
                file_name += ".pkl"  # 항상 .pkl로 저장해야 불러오기 파일 선택창 필터에서 보인다
            self.zapi._ZAPI_request_save_planning_snapshot(file_name)
            self.__set_path_plan_status(f"Planning snapshot save requested -> {pathlib.Path(file_name).name}")
        except Exception as e:
            self.__console.error(f"Error saving planning snapshot: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_random_snapshot_points(self):
        import pickle
        import secrets
        import datetime
        from simtool.snapshot_sampling import sample_snapshot_points
        try:
            if not self.zapi:
                raise RuntimeError("ZAPI not available")
            root = pathlib.Path(self.__config.get("root_path", ""))
            default = root / "sample" / "planning_snapshot_p3_10(90).pkl"
            source, _ = QFileDialog.getOpenFileName(self, "Random 5 Points — trusted snapshot only",
                                                  str(default), "Planning Snapshot (*.pkl)")
            if not source:
                return
            with open(source, "rb") as stream:
                snapshot = pickle.load(stream)
            selected = sample_snapshot_points(snapshot, seed=secrets.randbits(32))
            selected["random_point_selection"]["source"] = str(pathlib.Path(source).resolve())
            folder = root / "debug" / "random_snapshots"
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output = folder / f"random5_{stamp}.pkl"
            with output.open("wb") as stream:
                pickle.dump(selected, stream)
            self._inspection_sequencer_reset_for_snapshot()
            self.zapi._ZAPI_request_load_planning_snapshot(str(output.resolve()))
            points = selected["random_point_selection"]["source_point_indices"]
            self.__set_path_plan_status(f"Random 5 Points: {[i + 1 for i in points]}")
        except Exception as exc:
            self.__console.error(f"Random snapshot load failed: {exc}")
            self.__set_path_plan_status(f"[!] {exc}")

    def __on_btn_load_planning_snapshot_clicked(self):
        """저장된 planning snapshot을 viewer로 보내 scene과 EF pose를 복원한다."""
        try:
            if not self.zapi:
                self.__set_path_plan_status("[!] ZAPI not available")
                return
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Planning Snapshot (pickle - only open files you trust)",
                str(pathlib.Path(self.__config.get("root_path", "")) / "sample"),
                "Planning Snapshot Files (*.pkl);;All Files (*)"
            )
            if not file_name:
                return
            self._inspection_sequencer_reset_for_snapshot()
            self.zapi._ZAPI_request_load_planning_snapshot(file_name)
            self.__set_path_plan_status(f"Planning snapshot load requested -> {pathlib.Path(file_name).name}")
        except Exception as e:
            self.__console.error(f"Error loading planning snapshot: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def _inspection_sequencer_reset_for_snapshot(self):
        """새 scene을 불러오기 전에 이전 계획/재생 결과를 비운다 (이전 EF pose 기준의 plan_sequence가 남지 않게)."""
        try:
            self._inspection_workflow.clear_pose_and_path()
        except Exception:
            pass

    def __on_btn_load_inspection_points_clicked(self):
        try:
            if not self.zapi:
                self.__set_path_plan_status("[!] ZAPI not available")
                return
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Inspection Points",
                str(pathlib.Path(self.__config.get("root_path", "")) / "sample"),
                "JSON Files (*.json)"
            )
            if not file_name:
                return
            self.zapi._ZAPI_request_load_inspection_points(file_name)
            self.__set_path_plan_status(f"Inspection points load requested -> {pathlib.Path(file_name).name}")
        except Exception as e:
            self.__console.error(f"Error loading inspection points: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_determine_ef_pose_clicked(self):
        try:
            if not self.zapi:
                self.__set_path_plan_status("[!] ZAPI not available")
                return
            if not self._inspection_workflow.selected_points:
                self.__set_path_plan_status("[!] Inspection point is not selected")
                return
            self.zapi._ZAPI_request_determine_ef_pose(
                inspection_points=self._inspection_workflow.selected_points)
            self.__set_path_plan_status("EF pose requested")
        except Exception as e:
            self.__console.error(f"Error requesting EF pose determination: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_spool_collision_mesh_toggled(self, checked):
        if getattr(self, 'zapi', None) is not None:
            self.zapi._ZAPI_request_show_spool_collision_mesh(bool(checked))

    def __on_show_rejected_ef_toggled(self, checked):
        if getattr(self, 'zapi', None) is not None:
            self.zapi._ZAPI_request_show_rejected_ef_candidates(bool(checked))

    def __on_chk_show_saved_ik_toggled(self, checked):
        try:
            if self.zapi:
                self.zapi._ZAPI_request_show_saved_ik_goals(bool(checked))
        except Exception as e:
            self.__console.error(f"Error toggling saved IK goals display: {e}")

    def __on_btn_check_inspection_ik_clicked(self):
        try:
            if not self.zapi:
                self.__set_path_plan_status("[!] ZAPI not available")
                return
            planner = self.__current_planner_module_name()
            ik_solver, ik_normalize = self.__current_ik_request_options()
            self.zapi._ZAPI_request_check_ef_pose_ik(
                planner=planner,
                step_size=0.08,
                max_iter=3000,
                max_workers=2,
                ik_solver=ik_solver,
                ik_normalize=ik_normalize)
            self.__set_path_plan_status(f"EF pose IK check requested: solver={ik_solver}, normalize={ik_normalize}")
        except Exception as e:
            self.__console.error(f"Error requesting inspection IK check: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_stop_planning_clicked(self):
        """End the running planning sequence (see InspectionSequencer.cancel) and abort the Robot Core request in
        flight. What was planned so far is summarised, saved to debug/plan_runs and available for playback."""
        try:
            stopped = self._inspection_sequencer.cancel("planning stopped by user")
            if self.zapi:
                self.zapi._ZAPI_request_cancel_planning()
            self.__set_path_plan_status("Planning stopped" if stopped else "No planning is running")
        except Exception as e:
            self.__console.error(f"Error stopping planning: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_plan_inspection_path_clicked(self):
        try:
            if not self.zapi:
                self.__set_path_plan_status("[!] ZAPI not available")
                return
            # No is_running guard here on purpose: InspectionSequencer.start()
            # already tolerates being called while a previous sequence is
            # still pending (e.g. wedged because robot core died mid-request
            # and never replied) - it just supersedes it, and the stale
            # reply (if it ever arrives) is dropped by request_id mismatch.
            # Blocking the retry here would leave the user with no way to
            # recover from a wedged sequence except restarting SimTool.
            planner = self.__current_planner_module_name()
            optimizer = self.__current_optimizer_module_name()
            optimize_path = self.__path_optimization_enabled()
            ik_solver, ik_normalize = self.__current_ik_request_options()
            lock_linear_track = bool(
                hasattr(self, 'chk_lock_linear_track') and self.chk_lock_linear_track.isChecked())
            target_groups = self._inspection_workflow.planner_target_groups()
            # NOTE: previously auto-cleared visuals here before starting a
            # new sequence, but clear_inspection_path also wipes the
            # determined target-pose markers (_clear_inspection_goal_pose_
            # visuals), not just the TCP path line - made target poses
            # disappear the moment planning started. Reverted - use the
            # "Clear Inspection Path" button explicitly if you want a clean
            # slate; _handle_plan_single_target_completed still appends
            # (clear=False) so this run's own targets accumulate correctly,
            # just on top of whatever was already drawn.
            # Scene of this run (pipe, alignment, positioner angle, target poses, robot postures) as it is BEFORE planning
            # starts: saved by the Viewer now (requests are handled in order, so before the first plan_single_target),
            # copied into the run folder when the run is saved - "Load Result..." restores it, evaluate_plan_run.py
            # evaluates against it.
            self._run_scene_snapshot_path = None
            self._run_initial_positioner_r_deg = float(self.__get_positioner_pose_from_ui().get("r", 0.0))
            try:
                import time as _time
                pending_dir = (pathlib.Path(self.__config.get("root_path", "")) / self.__config.get("debug_dir", "debug")
                               / "plan_runs" / "_pending")
                pending_dir.mkdir(parents=True, exist_ok=True)
                scene_path = pending_dir / f"scene_{_time.strftime('%Y%m%d_%H%M%S')}.pkl"
                self.zapi._ZAPI_request_save_planning_snapshot(str(scene_path))
                self._run_scene_snapshot_path = scene_path
            except Exception as exc:
                self.__console.warning(f"could not request the run's scene snapshot: {exc}")
            self._inspection_sequencer.start(
                self.zapi, target_groups,
                planner=planner,
                planner_options={
                    # simtool.cfg planner_options {step_size, max_iter}: max_iter caps the planner's state-validity checks
                    # (OMPL); a hard instance that hits it ends as "timeout_approximate" (only an approximate path).
                    "step_size": float((self.__config.get("planner_options", {}) or {}).get("step_size", 0.1)),
                    "max_iter": int((self.__config.get("planner_options", {}) or {}).get("max_iter", 8000)),
                    "optimizer": optimizer, "optimize_path": optimize_path,
                    "ik_solver": ik_solver, "ik_normalize": ik_normalize,
                    "lock_linear_track": lock_linear_track,
                    # simtool.cfg planner_timeouts[planner] (default 60 s): per-target planning time limit. It used to be
                    # defined but never sent, so planning ran unbounded until Robot Core's request watchdog fired.
                    "planning_timeout": resolve_planner_timeout(self.__config, planner) or None,
                },
                rt_pipe_facing_axis=self._inspection_workflow.rt_pipe_facing_axis,
                positioner_second_group_r_deg=float(
                    (self.__config.get("path_planning", {}) or {}).get(
                        "positioner_second_group_r_deg", 180.0)),
                auto_rotate_positioner=self.chk_auto_rotate_positioner.isChecked(),
                initial_positioner_r_deg=float(self.__get_positioner_pose_from_ui().get("r", 0.0)),
                on_progress=lambda text: self.__set_path_plan_status(text),
                on_finished=self.__on_inspection_sequence_finished,
            )
        except Exception as e:
            self.__console.error(f"Error requesting inspection path plan: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_inspection_sequence_finished(self, summary: dict):
        n_planned = summary.get("n_planned", 0)
        n_total = summary.get("n_total", 0)
        deferred = summary.get("deferred_groups") or []
        results = summary.get("results") or []
        n_warning = sum(1 for r in results if r.get("status_kind") == "warning")
        n_failed = sum(1 for r in results if r.get("status_kind") == "failed")
        text = (
            f"Inspection sequence {summary.get('status')}: {n_planned}/{n_total} planned"
            + (f", {n_warning} warning(s) (unreachable target - tolerated)" if n_warning else "")
            + (f", {n_failed} failed" if n_failed else "")
        )
        if deferred:
            text += f", deferred (needs positioner rotation): {deferred}"
        if summary.get("status") == "failed" or (deferred and n_total == 0):
            self.__console.warning(text)
        else:
            self.__console.info(text)
        # Per-target success/warning/failed breakdown - InspectionSequencer
        # already logged this once (same console instance, InspectionSequencer.
        # _finish()) before calling this callback, as a guaranteed fallback in
        # case this callback never ran or threw partway through. Deliberately
        # duplicated here too since __set_path_plan_status only takes the
        # one-line summary above and this is also where playback gets wired up.
        for line in summary.get("summary_lines") or []:
            self.__console.info(line)
        self.__set_path_plan_status(text)
        self._last_plan_summary = summary
        saved_dir = self.__save_plan_run_from_summary(summary, run_dir=None)
        if saved_dir:
            self.__set_path_plan_status(f"{text} | saved: {saved_dir}")

        # Wire whatever this run planned into "Start Simulation" playback -
        # reuses the pre-existing InspectionPathHandler.accept_result()/
        # workflow.plan_sequence machinery (Visualizer._start_path_playback()
        # already knows how to play a plan_sequence; only the sequencer's
        # results were never converted into that shape before now).
        plan_sequence = self._inspection_sequencer.to_plan_sequence()
        if plan_sequence:
            self._inspection_path_handler.accept_result({
                "plan_sequence": plan_sequence,
                # the positioner angle the run started from (was 0.0: the Viewer then reset the positioner to 0 deg
                # before every playback)
                "playback_initial_r_deg": float(getattr(self, "_run_initial_positioner_r_deg", None)
                                                or plan_sequence[0].get("positioner_r_deg") or 0.0),
            })
            self.__console.info(
                f"Inspection sequence: {len(plan_sequence)} group(s) available for "
                "playback (Start Simulation)")
        else:
            self._inspection_workflow.set_planning_result({})

    def __save_plan_run_from_summary(self, summary: dict, run_dir=None):
        """Persist a finished InspectionSequencer run (paths + per-target failure cause).

        run_dir=None -> <root>/<debug_dir>/plan_runs/<timestamp>_<planner>/ (auto-save).
        Returns the saved folder path, or None if nothing was saved.
        """
        results = (summary or {}).get("results") or []
        if not results:
            return None
        try:
            if run_dir is None:
                root_path = self.__config.get("root_path", "")
                base = pathlib.Path(root_path) / self.__config.get("debug_dir", "debug") / "plan_runs"
                run_dir = default_run_dir(base, summary.get("planner"))
            spool_path = self.__get_current_spool_path()
            saved = save_plan_run(
                run_dir, results,
                meta={
                    "planner": summary.get("planner"),
                    "planner_options": summary.get("planner_options") or {},
                    "status": summary.get("status"),
                    "n_planned": summary.get("n_planned"),
                    "n_total_jobs": summary.get("n_total"),
                    "spool_path": str(spool_path) if spool_path else None,
                    "initial_positioner_r_deg": getattr(self, "_run_initial_positioner_r_deg", None),
                    "positioner_second_group_r_deg": float(
                        (self.__config.get("path_planning", {}) or {}).get("positioner_second_group_r_deg", 180.0)),
                    "scene_snapshot": None,
                },
                unreached_groups=summary.get("deferred_groups") or [])
            scene = getattr(self, "_run_scene_snapshot_path", None)
            if scene is not None and pathlib.Path(scene).exists():
                import shutil
                target = pathlib.Path(saved["run_dir"]) / "scene_snapshot.pkl"
                shutil.copyfile(scene, target)
                run_json = pathlib.Path(saved["run_dir"]) / "run.json"
                try:
                    meta = json.load(open(run_json, encoding="utf-8"))
                    meta["scene_snapshot"] = target.name
                    json.dump(meta, open(run_json, "w", encoding="utf-8"), indent=2)
                except Exception:
                    pass
                self.__console.info(f"  scene snapshot of the run: {target}")
            elif scene is not None:
                self.__console.warning(f"  scene snapshot {scene} was not written by the Viewer - run saved without it")
            self.__console.info(
                f"Saved plan run: {saved['run_dir']} "
                f"({saved['n_success']}/{saved['n_total']} succeeded, {saved['n_failed']} failed)")
            for failure in saved["failures"]:
                self.__console.warning(
                    f"  [{failure['category']}] {failure['robot_name']} {failure['group_name']}:"
                    f"{failure['pose_name']} - {failure['message']}")
            self._last_saved_run_dir = saved["run_dir"]
            return saved["run_dir"]
        except Exception as e:
            self.__console.error(f"Failed to save plan run: {e}")
            return None

    def __on_btn_evaluate_plan_result_clicked(self):
        """Evaluate a saved plan run with python/evaluate_plan_run.py in a background process (it restores the run's scene
        in its own headless Viewer, so the GUI stays responsive). Default folder: the last loaded / saved run."""
        try:
            if getattr(self, "_eval_process_state", None) and self._eval_process_state.get("running"):
                self.__set_path_plan_status("[!] an evaluation is already running")
                return
            root = pathlib.Path(self.__config.get("root_path", ""))
            default_dir = getattr(self, "_last_loaded_run_dir", None) or getattr(self, "_last_saved_run_dir", None) \
                or str(root / self.__config.get("debug_dir", "debug") / "plan_runs")
            run_dir = QFileDialog.getExistingDirectory(self, "Select a saved plan run (debug/plan_runs/<run>/)",
                                                       str(default_dir))
            if not run_dir:
                return
            run_dir = pathlib.Path(run_dir)
            cmd = [sys.executable, str(root / "python" / "evaluate_plan_run.py"), "--run-dir", str(run_dir)]
            if not (run_dir / "scene_snapshot.pkl").exists():
                # runs saved before the scene was stored with the run: ask for the planning snapshot they came from
                snap, _ = QFileDialog.getOpenFileName(
                    self, f"{run_dir.name} has no scene_snapshot.pkl - select the planning snapshot it was planned from",
                    str(root / "sample"), "Planning snapshot (*.pkl)")
                if not snap:
                    return
                cmd += ["--snapshot", snap]
            state = {"running": True, "returncode": None, "lines": [], "run_dir": run_dir}
            self._eval_process_state = state

            def _run():
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                            cwd=str(root), env={**os.environ, "PYTHONNOUSERSITE": "1"})
                    for line in proc.stdout:
                        state["lines"].append(line.rstrip())
                    state["returncode"] = proc.wait()
                except Exception as exc:
                    state["lines"].append(f"evaluation failed to start: {exc}")
                    state["returncode"] = -1
                state["running"] = False

            threading.Thread(target=_run, daemon=True).start()
            self.__console.info(f"Evaluating plan run {run_dir} ...")
            self.__set_path_plan_status(f"Evaluating {run_dir.name} ... (background)")
            self._eval_poll_timer = QTimer(self)
            self._eval_poll_timer.timeout.connect(self.__poll_plan_result_evaluation)
            self._eval_poll_timer.start(1000)
        except Exception as e:
            self.__console.error(f"Error starting plan run evaluation: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __poll_plan_result_evaluation(self):
        state = getattr(self, "_eval_process_state", None) or {}
        lines = state.get("lines") or []
        progress = [l for l in lines if l.startswith("[") and "/" in l.split("]")[0]]
        if state.get("running"):
            if progress:
                self.__set_path_plan_status(f"Evaluating: {progress[-1][:160]}")
            return
        self._eval_poll_timer.stop()
        run_dir = state.get("run_dir")
        if state.get("returncode") == 0:
            for line in progress:
                self.__console.info(f"  {line}")
            md = pathlib.Path(run_dir) / "eval" / "summary.md"
            table = md.read_text(encoding="utf-8") if md.exists() else ""
            self.__console.info(f"Plan run evaluation -> {md}\n{table}")
            first_row = next((l for l in table.splitlines() if l.startswith("| ") and "Planner" not in l), "")
            self.__set_path_plan_status(f"Evaluation done ({pathlib.Path(run_dir).name}): {first_row} -> eval/summary.md")
        else:
            tail = "\n".join(lines[-15:])
            self.__console.error(f"Plan run evaluation failed (exit {state.get('returncode')}):\n{tail}")
            self.__set_path_plan_status(f"[!] evaluation failed (exit {state.get('returncode')}) - see console")

    def __on_btn_save_plan_result_clicked(self):
        """Save the last planning result to a folder chosen by the user."""
        try:
            summary = getattr(self, '_last_plan_summary', None)
            if not summary or not (summary.get("results") or []):
                self.__set_path_plan_status("[!] no planning result to save (run Plan Inspection Path first)")
                return
            root_path = self.__config.get("root_path", "")
            default_dir = str(pathlib.Path(root_path) / self.__config.get("debug_dir", "debug") / "plan_runs")
            parent_dir = QFileDialog.getExistingDirectory(self, "Select folder to save the plan result into", default_dir)
            if not parent_dir:
                return
            run_dir = default_run_dir(parent_dir, summary.get("planner"))
            saved_dir = self.__save_plan_run_from_summary(summary, run_dir=run_dir)
            self.__set_path_plan_status(
                f"Saved plan result: {saved_dir}" if saved_dir else "[!] failed to save plan result")
        except Exception as e:
            self.__console.error(f"Error saving plan result: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_clear_inspection_path_clicked(self):
        try:
            if self.zapi:
                self.zapi._ZAPI_request_clear_inspection_path()
            self._inspection_workflow.clear()
            if hasattr(self, 'edit_inspection_point'):
                self.edit_inspection_point.clear()
            self.__set_path_plan_status("Inspection path cleared")
        except Exception as e:
            self.__console.error(f"Error clearing inspection path: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_start_simulation_clicked(self):
        """Start simulation playback for the last planned inspection path."""
        try:
            if not self.zapi:
                self.__set_path_plan_status("[!] ZAPI not available")
                return
            self._inspection_path_handler.request_playback(self.zapi, speed=1.0)
            self.__set_path_plan_status("Simulation playback requested")
        except Exception as e:
            self.__console.error(f"Error starting simulation: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_load_playback_result_clicked(self, run_dir=None):
        """Load a test_ompl_planning.py run folder and wire it into the same
        plan_sequence/"Start Simulation" playback machinery a live
        InspectionSequencer run uses - see playback_loader.py and
        __on_inspection_sequence_finished's identical accept_result() call.

        Supports both output shapes test_ompl_planning.py can produce:
        - --target all: debug/<method>_<timestamp>/, has summary.csv +
          per-target <idx>_<robot>_<pose>/joint_states.csv subfolders
          (load_playback_plan_sequence).
        - --target N (a single target - e.g. checking just one or two before
          committing to a full-sequence run): joint_states.csv directly in
          the folder, no summary.csv. Falls back to
          load_playback_single_target(), which needs the robot name (not
          recoverable from the folder) - prompted for below.
        """
        try:
            from simtool.playback_loader import (load_playback_plan_sequence, load_playback_single_target,
                                                  resolve_playback_run_dir)

            root_path = self.__config.get("root_path", "")
            default_dir = str(pathlib.Path(root_path) / self.__config.get("debug_dir", "debug"))
            if not isinstance(run_dir, str) or not run_dir:
                run_dir = QFileDialog.getExistingDirectory(
                    self, "Select planning or SAC inference result folder", default_dir)
            if not run_dir:
                return
            resolved_run_dir = resolve_playback_run_dir(run_dir)
            if pathlib.Path(run_dir) != resolved_run_dir:
                self.__console.info(f"Resolved saved paths folder: {resolved_run_dir}")
                run_dir = str(resolved_run_dir)
            # benchmark_path_planners.py's --save-paths writes spool.ply (+ a
            # .json alignment sidecar, if the source snapshot had one) next to
            # summary.csv - see export_snapshot_spool.py's export_spool_from_
            # snapshot(), called from _run_once(). Auto-load it here so the
            # pipe/positioner the paths were actually planned against shows up
            # without a separate manual "Load Spool" step - and, since it's
            # the exact scene this run used, positioner rotation always
            # matches what execute_inspection_path's "needs positioner to
            # rotate but no pipe is loaded" check requires below.
            spool_path = pathlib.Path(run_dir) / "spool.ply"
            scene_snapshot = pathlib.Path(run_dir) / "scene_snapshot.pkl"
            self._pending_loaded_playback = None
            if scene_snapshot.exists() and self.zapi:
                # the exact scene the run was planned in (pipe + alignment + positioner angle + target poses + robot
                # postures), saved by SimTool at sequence start
                self.zapi._ZAPI_request_load_planning_snapshot(str(scene_snapshot.absolute()))
                self.__console.info(f"Restoring the run's scene: {scene_snapshot}")
            elif spool_path.exists() and self.zapi:
                self.zapi._ZAPI_request_load_spool(str(spool_path.absolute()))
                self.__console.info(f"Auto-loading spool from playback result: {spool_path}")
            elif self.zapi:
                # Runs saved by __save_plan_run_from_summary record which spool they were planned on.
                run_meta_path = pathlib.Path(run_dir) / "run.json"
                try:
                    recorded_spool = (json.load(open(run_meta_path, encoding="utf-8")).get("spool_path")
                                      if run_meta_path.exists() else None)
                except Exception:
                    recorded_spool = None
                if recorded_spool and pathlib.Path(recorded_spool).exists():
                    self.zapi._ZAPI_request_load_spool(str(pathlib.Path(recorded_spool).absolute()))
                    self.__console.info(f"Auto-loading spool recorded in run.json: {recorded_spool}")
            if (pathlib.Path(run_dir) / "summary.csv").exists():
                result = load_playback_plan_sequence(run_dir)
            else:
                robot_names = [str(getattr(m, "name", "")) for m in getattr(self, "_robot_models", [])
                               if getattr(m, "name", None)]
                robot_name, ok = QInputDialog.getItem(
                    self, "Single-target playback",
                    f"{run_dir} has no summary.csv (looks like a --target N single-target run).\n"
                    "Which robot was this planned for?",
                    robot_names, 0, False)
                if not ok or not robot_name:
                    return
                # Not recoverable from the folder - check this run's console
                # log for "target[N] needed a positioner rotation" (see
                # test_ompl_planning.py's _resolve_positioner_r_deg). Wrong
                # angle here still plays the robot's q_path correctly, just
                # shows the pipe/positioner at the wrong attitude.
                r_deg, ok = QInputDialog.getDouble(
                    self, "Single-target playback",
                    "Positioner angle (deg) this target was planned against "
                    "(0 if it did not need a positioner rotation):",
                    0.0, -360.0, 360.0, 2)
                if not ok:
                    return
                result = load_playback_single_target(run_dir, robot_name, positioner_r_deg=r_deg)
            plan_sequence = result["plan_sequence"]
            if not plan_sequence:
                self.__set_path_plan_status(f"[!] no playable targets found in {run_dir}")
                return
            loaded = {
                "plan_sequence": plan_sequence,
                "playback_initial_r_deg": float(plan_sequence[0].get("positioner_r_deg") or 0.0),
                "loaded_playback_result": True,
            }
            if scene_snapshot.exists() and self.zapi:
                # restoring the snapshot makes the Viewer send its EF poses again (reply_ef_pose ->
                # set_pose_result clears plan_sequence): accept the loaded paths once the snapshot load has replied
                self._pending_loaded_playback = loaded
            else:
                self._inspection_path_handler.accept_result(loaded)
                if self.zapi:
                    self.zapi._ZAPI_request_show_plan_sequence(
                        plan_sequence, playback_initial_r_deg=loaded["playback_initial_r_deg"])
            self._last_loaded_run_dir = str(run_dir)
            n_skipped = len(result["skipped"])
            text = (
                f"Loaded {result['n_loaded']}/{result['n_targets']} target(s) from {run_dir} "
                f"for playback (Start Simulation)"
                + (f" - {n_skipped} skipped (missing/empty joint_states.csv)" if n_skipped else ""))
            self.__console.info(text)
            if result["skipped"]:
                self.__console.warning(f"skipped: {result['skipped']}")
            failures = result.get("failures") or []
            if failures:
                by_category = {}
                for failure in failures:
                    by_category[failure.get("category")] = by_category.get(failure.get("category"), 0) + 1
                text += f" | {len(failures)} failed target(s): {by_category} (see console/failures.json)"
                for failure in failures:
                    self.__console.warning(
                        f"  [{failure.get('category')}] {failure.get('robot_name')} "
                        f"{failure.get('group_name')}:{failure.get('pose_name')} - {failure.get('message')}")
            self.__set_path_plan_status(text)
        except Exception as e:
            self.__console.error(f"Error loading playback result: {e}")
            self.__set_path_plan_status(f"[!] {e}")

    def __on_btn_load_spool_clicked(self):
        """Handle Load Spool button click"""
        try:
            if hasattr(self, 'cbx_pipe_spool'):
                current_text = self.cbx_pipe_spool.currentText()
                if not current_text:
                    self.__console.warning("No spool file selected")
                    return
                
                # Resolve full path
                sample_path = pathlib.Path(self.__config.get("root_path", "")) / "sample" / current_text
                if sample_path.exists():
                    self._current_spool_path = sample_path
                    if self.zapi:
                        self.zapi._ZAPI_request_load_spool(str(sample_path.absolute()))
                        self.__console.info(f"Requested to load spool: {current_text}")
                        self.__load_spool_pose(sample_path, apply_move=True)
                    else:
                        self.__console.error("ZAPI instance not available")
                else:
                    self.__console.error(f"Spool file not found: {sample_path}")
        except Exception as e:
            self.__console.error(f"Error loading spool: {e}")

    def __on_btn_flip_spool_x_clicked(self):
        """Handle spool X direction flip button click"""
        try:
            if self.__spool_move_blocked_by_fix():
                return
            if self.zapi:
                self.zapi._ZAPI_request_flip_spool_x()
                self.__console.info("Requested to flip spool X direction")
            else:
                self.__console.error("ZAPI instance not available")
        except Exception as e:
            self.__console.error(f"Error flipping spool X direction: {e}")

    def __on_btn_load_test_weld_point_clicked(self):
        """Handle Load Test Weld Point button click"""
        try:
            if hasattr(self, 'cbx_test_weld_point'):
                current_text = self.cbx_test_weld_point.currentText()
                if not current_text:
                    self.__console.warning("No test weld point file selected")
                    return
                
                # Resolve full path
                sample_path = pathlib.Path(self.__config.get("root_path", "")) / "sample" / current_text
                if sample_path.exists():
                    if self.zapi:
                        self.zapi._ZAPI_request_load_test_weld_point(str(sample_path.absolute()))
                        self.__console.info(f"Requested to load test weld point: {current_text}")
                    else:
                        self.__console.error("ZAPI instance not available")
                else:
                    self.__console.error(f"Test weld point file not found: {sample_path}")
        except Exception as e:
            self.__console.error(f"Error loading test weld point: {e}")

    def __on_btn_load_sim_parameters_clicked(self):
        """Handle Load Simulation Parameters button click"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self, 
                "Open Simulation Parameter File", 
                str(self.__config.get("root_path", "")), 
                "JSON Files (*.json)"
            )
            
            if file_name:
                self._sim_param_map.load_parameters(file_name)
        except Exception as e:
            self.__console.error(f"Error loading simulation parameters: {e}")

    def _handle_message(self, topic, msg):
        """Handle incoming ZMQ messages"""
        try:
            # Decode if bytes
            if isinstance(topic, bytes):
                topic = topic.decode('utf-8')
            if isinstance(msg, bytes):
                msg = msg.decode('utf-8')

            if topic == "update_state_info":
                self.__console.info("Received state info from viewervedo. Sending current execution mode.")
                if hasattr(self, 'radio_mode_simulation') and self.radio_mode_simulation.isChecked():
                    self.zapi._ZAPI_request_set_mode("simulation")
                elif hasattr(self, 'radio_mode_real') and self.radio_mode_real.isChecked():
                    self.zapi._ZAPI_request_set_mode("real")
                else:
                    self.zapi._ZAPI_request_set_mode("simulation")

            if topic == "update_spool_pose":
                try:
                    pose = json.loads(msg)
                    self.__set_spool_pose_to_ui(pose)
                    self.__console.info(f"Updated spool pose from viewer: {pose}")
                except json.JSONDecodeError:
                    pass

            if topic == "update_positioner_pose":
                try:
                    pose = json.loads(msg)
                    self.__set_positioner_pose_to_ui(pose)
                    self.__console.info(f"Updated positioner pose from viewer: {pose}")
                except Exception:
                    pass

            if topic == "update_robot_joint_state":
                try:
                    state = json.loads(msg)
                    self.__set_robot_joint_state_to_ui(state)
                except Exception:
                    pass

            if topic == "update_inspection_point":
                try:
                    point = json.loads(msg)
                    self._inspection_workflow.set_selected_points(point)
                    xyz = point.get("point", point)
                    points = point.get("points", []) if isinstance(point, dict) else []
                    if hasattr(self, 'edit_inspection_point'):
                        self.edit_inspection_point.setText(
                            f"{len(points) or 1} pts | "
                            f"{float(xyz[0]):.4f}, {float(xyz[1]):.4f}, {float(xyz[2]):.4f}")
                    self.__set_path_plan_status(f"Inspection point selected ({len(points) or 1})")
                except Exception:
                    pass

            if topic == "reply_plan_single_target":
                try:
                    payload = json.loads(msg)
                    self._inspection_sequencer.on_reply(payload)
                except Exception:
                    self.__console.error("Failed to process reply_plan_single_target")

            if topic == "reply_prepare_next_inspection_phase":
                try:
                    payload = json.loads(msg)
                    self._inspection_sequencer.on_phase_prepared(payload)
                except Exception:
                    self.__console.error("Failed to process reply_prepare_next_inspection_phase")

            if topic == "reply_inspection_path":
                try:
                    result = json.loads(msg)
                    self._inspection_path_handler.accept_result(result)
                    status = result.get("status", "unknown")
                    if status in ("success", "partial"):
                        mode = result.get("mode")
                        robots = result.get("robots")
                        if isinstance(robots, dict) and robots:
                            summary = ", ".join(
                                f"{name}:{info.get('ik_result', {}).get('success', False)}"
                                if mode == "ik_check"
                                else f"{name}:{int(info.get('waypoints', 0))}wp/{float(info.get('elapsed', 0.0)):.2f}s"
                                for name, info in robots.items())
                            if mode == "ik_check":
                                prefix = "IK OK" if status == "success" else "IK partial"
                            else:
                                prefix = "Paths OK" if status == "success" else "Paths partial"
                            wall_elapsed = result.get("wall_elapsed", result.get("elapsed", 0.0))
                            self.__set_path_plan_status(
                                f"{prefix}: {summary} | wall {float(wall_elapsed):.2f}s")
                        else:
                            self.__set_path_plan_status(
                                f"Path OK: {result.get('waypoints', 0)} wp, "
                                f"{float(result.get('elapsed', 0.0)):.2f}s")
                    else:
                        self.__set_path_plan_status(f"Path failed: {result.get('message', status)}")
                except Exception:
                    pass

            if topic == "reply_planning_snapshot":
                try:
                    result = json.loads(msg)
                    if result.get("action") == "load":
                        if result.get("status") == "success":
                            # viewer가 복원한 고정 상태/스풀 경로를 UI에도 반영한다. 체크박스가 해제된 채로
                            # 남으면 다음 positioner 이동 요청이 고정 해제 값을 실어 보내 정렬이 풀린다.
                            fixation = result.get("fixation") or {}
                            self.__set_spool_fix_checks(
                                bool(fixation.get("fix_f_column_r", False)),
                                bool(fixation.get("fix_m_column_z", False)))
                            restored_spool = result.get("spool_path")
                            if restored_spool:
                                self._current_spool_path = pathlib.Path(restored_spool)
                                self.__refresh_spool_combo_with_file(restored_spool)
                                if hasattr(self, 'cbx_pipe_spool'):
                                    index = self.cbx_pipe_spool.findText(pathlib.Path(restored_spool).name)
                                    if index >= 0:
                                        blocked = self.cbx_pipe_spool.blockSignals(True)
                                        self.cbx_pipe_spool.setCurrentIndex(index)
                                        self.cbx_pipe_spool.blockSignals(blocked)
                            text = (
                                f"Planning snapshot loaded: {result.get('n_target_groups')} pose group(s), "
                                f"{result.get('n_points')} point(s) <- {pathlib.Path(result.get('path', '')).name}")
                            notes = result.get("notes") or []
                            if notes:
                                text += " | " + "; ".join(notes)
                                for note in notes:
                                    self.__console.warning(f"snapshot load: {note}")
                            pending = getattr(self, "_pending_loaded_playback", None)
                            if pending:
                                self._pending_loaded_playback = None
                                self._inspection_path_handler.accept_result(pending)
                                if self.zapi:
                                    # after the snapshot restore (which redraws the EF poses): goal frames + EE paths
                                    self.zapi._ZAPI_request_show_plan_sequence(
                                        pending["plan_sequence"],
                                        playback_initial_r_deg=pending.get("playback_initial_r_deg"))
                                text += (f" | loaded plan result ready for playback: "
                                         f"{len(pending['plan_sequence'])} group(s) (Start Simulation)")
                            self.__set_path_plan_status(text)
                        else:
                            self.__set_path_plan_status(
                                f"[!] Planning snapshot load failed: {result.get('message', 'unknown')}")
                    elif result.get("status") == "success":
                        self.__set_path_plan_status(
                            f"Planning snapshot saved -> {pathlib.Path(result.get('path', '')).name}")
                    else:
                        self.__set_path_plan_status(
                            f"Planning snapshot save failed: {result.get('message', 'unknown')}")
                except Exception:
                    pass

            if topic == "update_chuck_mount_points":
                try:
                    payload = json.loads(msg)
                    points = payload.get("points", [])
                    local_points = payload.get("local_points", [])
                    self._chuck_mount_points = points
                    self._chuck_mount_local_points = local_points
                    if len(points) >= 2:
                        p0 = points[0]
                        p1 = points[1]
                        self.__set_path_plan_status(
                            "Chuck points selected: "
                            f"F({float(p0[0]):.3f},{float(p0[1]):.3f},{float(p0[2]):.3f}) "
                            f"M({float(p1[0]):.3f},{float(p1[1]):.3f},{float(p1[2]):.3f})")
                except Exception:
                    pass

            if topic == "update_chuck_mount_profile":
                try:
                    profile = json.loads(msg)
                    target = str(profile.get("target", "chuck"))
                    center = profile.get("center", [0.0, 0.0, 0.0])
                    axis = profile.get("axis", [0.0, 0.0, 0.0])
                    radius = float(profile.get("radius", 0.0))
                    text = (
                        f"{target} cylinder | "
                        f"C=({float(center[0]):.4f}, {float(center[1]):.4f}, {float(center[2]):.4f}) "
                        f"A=({float(axis[0]):.4f}, {float(axis[1]):.4f}, {float(axis[2]):.4f}) "
                        f"R={radius:.5f}"
                    )
                    if "center_error" in profile or "axis_error_deg" in profile:
                        text += (
                            f" | errC={float(profile.get('center_error', 0.0)):.5f}, "
                            f"errA={float(profile.get('axis_error_deg', 0.0)):.2f}deg"
                        )
                    if hasattr(self, 'label_chuck_mount_profile'):
                        self.label_chuck_mount_profile.setText(text)
                    self.__set_path_plan_status(text)
                except Exception:
                    pass

            if topic == "reply_ef_pose":
                try:
                    result = json.loads(msg)
                    status = result.get("status", "unknown")
                    if status == "success":
                        self._inspection_workflow.set_pose_result(result)
                        count = len(self._inspection_workflow.target_groups)
                        self.__set_path_plan_status(
                            f"EF pose ready: {count} pose(s), {float(result.get('elapsed', 0.0)):.2f}s")
                    else:
                        self.__set_path_plan_status(
                            f"EF pose failed: {result.get('message', status)} "
                            f"({float(result.get('elapsed', 0.0)):.2f}s)")
                except Exception:
                    pass

            if topic == "call":
                try:
                    payload = json.loads(msg)
                    command = payload.get("command")
                    if command == "reply_load_spool":
                        path = payload.get("path")
                        status = payload.get("status")
                        if status == "success":
                            self.__console.info(f"Viewer successfully loaded spool: {path}")
                            QMessageBox.information(self, "Load Spool", f"Successfully loaded:\n{os.path.basename(path)}")
                        else:
                            self.__console.error(f"Viewer failed to load spool: {path}")
                            QMessageBox.warning(self, "Load Spool", f"Failed to load:\n{os.path.basename(path)}")
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            self.__console.error(f"Error handling message: {e}")
    
    def __set_proc_status(self, msg):
        if hasattr(self, 'label_pcd_proc_status'):
            self.label_pcd_proc_status.setText(msg)

    def __refresh_spool_combo_with_file(self, file_path):
        if not hasattr(self, 'cbx_pipe_spool'):
            return
        sample_path = pathlib.Path(self.__config.get("root_path", "")) / "sample"
        file_path = pathlib.Path(file_path)
        if file_path.parent.resolve() != sample_path.resolve():
            return
        file_name = file_path.name
        if self.cbx_pipe_spool.findText(file_name) < 0:
            self.cbx_pipe_spool.addItem(file_name)

    def __on_btn_pcd_sor_filter_clicked(self):
        """현재 로드된 스풀에 SOR 노이즈 제거를 직접 적용."""
        try:
            if not self.zapi:
                self.__set_proc_status("[!] ZAPI not available")
                return
            neighbors = self.spin_pcd_sor_neighbors.value() if hasattr(self, 'spin_pcd_sor_neighbors') else 20
            std_ratio = self.spin_pcd_sor_std_ratio.value() if hasattr(self, 'spin_pcd_sor_std_ratio') else 2.0
            self.zapi._ZAPI_request_filter_spool(
                "sor", {"neighbors": neighbors, "std_ratio": std_ratio})
            self.__set_proc_status(f"SOR applied (n={neighbors}, std={std_ratio}) to loaded spool")
        except Exception as e:
            self.__console.error(f"Error applying SOR: {e}")
            self.__set_proc_status(f"[!] {e}")

    def __on_btn_pcd_ccl_filter_clicked(self):
        """현재 로드된 스풀에 옥트리(복셀) CCL 노이즈 제거를 직접 적용."""
        try:
            if not self.zapi:
                self.__set_proc_status("[!] ZAPI not available")
                return
            level = self.spin_pcd_ccl_level.value() if hasattr(self, 'spin_pcd_ccl_level') else 7
            min_points = self.spin_pcd_ccl_min_points.value() if hasattr(self, 'spin_pcd_ccl_min_points') else 30
            self.zapi._ZAPI_request_filter_spool(
                "ccl", {"level": level, "min_points": min_points})
            self.__set_proc_status(f"CCL applied (lv={level}, min={min_points}) to loaded spool")
        except Exception as e:
            self.__console.error(f"Error applying CCL: {e}")
            self.__set_proc_status(f"[!] {e}")

## 메쉬 재건은 현재 미사용(Marching Cubes) - 2026-07-06
## normal 이 포함된 ply 기준으로 개발 후 추후 필요시 다시 활용
#region mesh reconstruction
    def __on_btn_mesh_convert_clicked(self):
        """현재 로드된 스풀로 메시 재건(Marching Cubes)을 요청."""
        try:
            if not self.zapi:
                self.__set_proc_status("[!] ZAPI not available")
                return
            resolution = self.spin_mesh_resolution.value() if hasattr(self, 'spin_mesh_resolution') else 128
            sigma      = self.spin_mesh_sigma.value()      if hasattr(self, 'spin_mesh_sigma')      else 1.5
            level      = self.spin_mesh_level.value()      if hasattr(self, 'spin_mesh_level')      else 0.5
            self.zapi._ZAPI_request_reconstruct_mesh(
                {"resolution": resolution, "sigma": sigma, "level": level})
            self.__set_proc_status(f"Mesh reconstruct requested (res={resolution}, sigma={sigma}, lv={level})")
        except Exception as e:
            self.__console.error(f"Error requesting mesh reconstruct: {e}")
            self.__set_proc_status(f"[!] {e}")
#endregion

    def __on_btn_pcd_save_clicked(self):
        """현재 결과(필터된 스풀 또는 재건 메시)를 파일로 저장."""
        try:
            if not self.zapi:
                self.__set_proc_status("[!] ZAPI not available")
                return
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Result",
                str(pathlib.Path(self.__config.get("root_path", "")) / "sample"),
                "Point Cloud / Mesh (*.ply *.pcd *.stl *.obj)"
            )
            if not file_name:
                return
            self.zapi._ZAPI_request_save_spool(file_name)
            self.__set_proc_status(f"Save requested -> {pathlib.Path(file_name).name}")
            self.__refresh_spool_combo_with_file(file_name)
            # 포지셔너 자세가 있으면 같은 이름의 .json으로 함께 저장
            self.__save_positioner_json(file_name)
        except Exception as e:
            self.__console.error(f"Error saving result: {e}")
            self.__set_proc_status(f"[!] {e}")

    # ── 매니퓰레이터 전체 관절 모션 (source/DDA 공통, 테이블 기반) ──
    def __res_of(self, kind):
        return self._LIN_RES if kind == 'lin' else self._ROT_RES

    def __wire_manipulator(self, robot, table, move_btn, stop_btn):
        """관절 테이블의 슬라이더↔edit 동기화 + Move/Stop 버튼 연결."""
        for slider, edit, joint, kind, lo, hi in table:
            res = self.__res_of(kind)
            s = getattr(self, slider, None); e = getattr(self, edit, None)
            if s is not None:
                s.setMinimum(int(round(lo / res)))
                s.setMaximum(int(round(hi / res)))
                s.valueChanged.connect(
                    lambda v, ed=edit, r=res: self.__manip_slider_changed(ed, v, r))
            if e is not None:
                e.editingFinished.connect(
                    lambda ed=edit, sl=slider, r=res, _lo=lo, _hi=hi:
                        self.__manip_edit_changed(ed, sl, r, _lo, _hi))
        mb = getattr(self, move_btn, None); sb = getattr(self, stop_btn, None)
        if mb is not None:
            mb.clicked.connect(lambda _=False, rb=robot, tb=table: self.__manip_move(rb, tb))
        if sb is not None:
            sb.clicked.connect(lambda _=False, rb=robot: self.__manip_stop(rb))

    def __manip_slider_changed(self, edit_name, value, res):
        e = getattr(self, edit_name, None)
        if e is not None:
            e.setText(f"{value * res:.3f}")

    def __set_robot_joint_state_to_ui(self, state):
        robots = state.get("robots", {}) if isinstance(state, dict) else {}
        if not isinstance(robots, dict):
            return

        def apply_table(robot_name, table):
            joints = robots.get(robot_name)
            if not isinstance(joints, dict):
                return
            import math
            for slider_name, edit_name, joint_name, kind, lo, hi in table:
                if joint_name not in joints:
                    continue
                raw_value = float(joints[joint_name])
                ui_value = math.degrees(raw_value) if kind == 'rot' else raw_value
                ui_value = max(float(lo), min(float(ui_value), float(hi)))
                res = self._ROT_RES if kind == 'rot' else self._LIN_RES
                slider = getattr(self, slider_name, None)
                edit = getattr(self, edit_name, None)
                if slider is not None:
                    blocked = slider.blockSignals(True)
                    slider.setValue(int(round(ui_value / res)))
                    slider.blockSignals(blocked)
                if edit is not None:
                    blocked = edit.blockSignals(True)
                    edit.setText(f"{ui_value:.3f}")
                    edit.blockSignals(blocked)

        apply_table(self._SOURCE_ROBOT, self._SOURCE_JOINTS)
        apply_table(self._DDA_ROBOT, self._DDA_JOINTS)

    def __manip_edit_changed(self, edit_name, slider_name, res, lo, hi):
        e = getattr(self, edit_name, None); s = getattr(self, slider_name, None)
        if e is None or s is None:
            return
        try:
            val = float(e.text())
        except ValueError:
            return
        val = max(lo, min(val, hi))
        blocked = s.blockSignals(True)
        s.setValue(int(round(val / res)))
        s.blockSignals(blocked)

    def __manip_move(self, robot, table):
        """테이블의 모든 관절을 edit 목표값으로 보간 이동 (한 번에 팔 전체)."""
        if not self.zapi:
            return
        import math
        for _slider, edit, joint, kind, lo, hi in table:
            e = getattr(self, edit, None)
            if e is None:
                continue
            try:
                val = float(e.text() or "0")
            except ValueError:
                continue
            val = max(lo, min(val, hi))
            if kind == 'rot':
                target = math.radians(val)             # deg → rad
                speed, accel = math.radians(self._ROT_SPEED), math.radians(self._ROT_ACCEL)
            else:
                target = val                           # m
                speed, accel = self._LIN_SPEED, self._LIN_ACCEL
            self.zapi._ZAPI_request_move_manipulator(robot, joint, target, speed, accel)
        self.__console.info(f"Manipulator move requested: {robot} ({len(table)} joints)")

    def __manip_stop(self, robot):
        if self.zapi:
            self.zapi._ZAPI_request_stop_manipulator(robot, None)   # 해당 로봇 전체 정지

    def __reset_robot_base_pose(self, robot=None):
        if not self.zapi:
            return
        self.zapi._ZAPI_request_reset_robot_base_pose(robot)
        self.__console.info(f"Robot base pose reset requested: {robot or 'all'}")

    def __save_positioner_json(self, geom_path):
        """저장한 지오메트리 옆에 포지셔너/스풀 자세를 <name>.json 으로 기록."""
        try:
            pos = self.__get_positioner_pose_from_ui()
            spool = self.__get_spool_pose_from_ui()
            fix_f, fix_z = self.__get_spool_fix_flags()
        except (ValueError, AttributeError):
            return
        if pos is None:
            return
        json_path = pathlib.Path(geom_path).with_suffix(".json")
        payload = {
            "geometry_file": pathlib.Path(geom_path).name,
            "positioner": {
                "x": pos["x"],
                "z": pos["z"],
                "r": pos["r"],
                "clamp": pos["clamp"],
            },
            "spool": {
                "x": spool["x"],
                "y": spool["y"],
                "z": spool["z"],
                "x_rotation": spool["x_rotation"],
                "z_rotation": spool["z_rotation"],
            },
            "fix_f_column_r": bool(fix_f),
            "fix_m_column_z": bool(fix_z),
            "fixation": {
                "fixed": bool(fix_f or fix_z),
                "fix_f_column_r": bool(fix_f),
                "fix_m_column_z": bool(fix_z),
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        self.__console.info(f"Saved positioner/spool pose: {json_path}")

    def closeEvent(self, event:QCloseEvent) -> None:
        """ Handle close event """
        try:
            # ZAPI cleanup
            if hasattr(self, 'zapi') and self.zapi:
                self.zapi.stop()
                self.__console.info("ZAPI stopped")

        except Exception as e:
            self.__console.error(f"Error during window close: {e}")
        finally:
            self.__console.info("Successfully Closed")
            super().closeEvent(event)
