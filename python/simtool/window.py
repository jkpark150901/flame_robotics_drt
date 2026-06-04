'''
DRT Tool Window
@Author Byunghun Hwang<bh.hwang@iae.re.kr>
'''

try:
    # using PyQt6
    from PyQt6.QtGui import QImage, QPixmap, QCloseEvent, QStandardItem, QStandardItemModel
    from PyQt6.QtWidgets import QApplication, QFrame, QMainWindow, QLabel, QPushButton, QCheckBox, QComboBox, QDialog
    from PyQt6.QtWidgets import QMessageBox, QProgressBar, QFileDialog, QComboBox, QLineEdit, QSlider, QVBoxLayout
    from PyQt6.uic import loadUi
    from PyQt6.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal, QRegularExpression
except ImportError:
    print("PyQt6 is required to run this application.")

import os, sys
import pathlib
import json
import importlib
import inspect

from util.logger.console import ConsoleLogger
from plugins.pluginbase.plannerbase import PlannerBase
from plugins.pluginbase.optimizerbase import OptimizerBase
from simtool.param import SimParameterMap


class AppWindow(QMainWindow):
    def __init__(self, config:dict, zpipe):
        """ initialization """
        super().__init__()
        
        self.__console = ConsoleLogger.get_logger()
        self.__config = config
        self.zpipe = zpipe
        self.zapi = None
        self.mujoco_zapi = None

        try:            
            if "gui" in config:
                # load UI
                ui_path = pathlib.Path(config["app_path"]) / config["gui"]
                if os.path.isfile(ui_path):
                    loadUi(ui_path, self)
                    self.setWindowTitle(config.get("window_title", "DRT Simulation Tool"))

                    # Initialize SimParameterMap for UI sync
                    self._sim_param_map = SimParameterMap(self, self.__config, self.__console)

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

                mujoco_config = self.__config.get("mujoco", {})
                if mujoco_config.get("enable", False):
                    mujoco_transport = mujoco_config.get("transport", "ipc")
                    mujoco_channel = mujoco_config.get("channel", "/tmp/viewermujoco")
                    self.mujoco_zapi = ZAPI(
                        zpipe=self.zpipe,
                        transport=mujoco_transport,
                        channel=mujoco_channel,
                        socket_id="ZAPI_SIMTOOL_MUJOCO"
                    )
                    self.mujoco_zapi.signal_message_received.connect(self._handle_mujoco_message)
                    self.mujoco_zapi.run()
                    self.__console.info("Now ZAPI is running for MuJoCo")

                    urdf_entries = mujoco_config.get("urdf", [])
                    if urdf_entries:
                        root_path = pathlib.Path(self.__config.get("root_path", ""))
                        resolved_entries = []
                        for urdf_entry in urdf_entries:
                            if isinstance(urdf_entry, str):
                                resolved_entries.append(str(root_path / urdf_entry))
                            else:
                                resolved_entry = urdf_entry.copy()
                                resolved_entry["path"] = str(root_path / resolved_entry.get("path", ""))
                                resolved_entries.append(resolved_entry)
                        self.mujoco_zapi._ZAPI_request_mujoco_load_urdf_workspace(resolved_entries)
                    else:
                        model_paths = mujoco_config.get("models", [])
                        if model_paths:
                            root_path = pathlib.Path(self.__config.get("root_path", ""))
                            resolved_models = []
                            for model_entry in model_paths:
                                if isinstance(model_entry, str):
                                    resolved_models.append(str(root_path / model_entry))
                                else:
                                    resolved_entry = model_entry.copy()
                                    resolved_entry["path"] = str(root_path / resolved_entry.get("path", ""))
                                    resolved_models.append(resolved_entry)
                            self.mujoco_zapi._ZAPI_request_mujoco_load_models(resolved_models)
                        else:
                            model_path = mujoco_config.get("model", "")
                            if model_path:
                                model_path = pathlib.Path(self.__config.get("root_path", "")) / model_path
                                self.mujoco_zapi._ZAPI_request_mujoco_load_model(str(model_path))
            else:
                 self.__console.error("ZPipe instance missing, ZAPI not started")
                
        except Exception as e:
            self.__console.error(f"{e}")

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
                    
                    root_path = self.__config.get("root_path", "")
                    plugin_path = pathlib.Path(root_path) / category["path"]
                    
                    if plugin_path.exists():
                        files = list(plugin_path.glob("*.py"))
                        
                        for file_path in files:
                            if file_path.name == "__init__.py":
                                continue
                            module_name = f"{category['package_prefix']}.{file_path.stem}"
                            try:
                                module = importlib.import_module(module_name)
                                for name, obj in inspect.getmembers(module):
                                    if inspect.isclass(obj) and issubclass(obj, category["base_class"]): 
                                        if obj is not category["base_class"]:
                                            self.__console.debug(f"Found plugin class: {obj.__name__}")
                                            combobox.addItem(obj.__name__)
                            except Exception as e:
                                self.__console.error(f"Failed to load {category['name']} plugin {module_name}: {e}")
                    else:
                        self.__console.warning(f"{category['name']} plugin directory not found: {plugin_path}")

                except Exception as e:
                    import traceback
                    self.__console.error(f"Error processing category {category['name']}: {e}")
                    self.__console.error(traceback.format_exc())

        # 3. Load Pipe Spool Samples and Test Weld Points
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
        if hasattr(self, 'btn_load_test_weld_point'):
            self.btn_load_test_weld_point.clicked.connect(self.__on_btn_load_test_weld_point_clicked)
        if hasattr(self, 'btn_load_sim_parameters'):
            self.btn_load_sim_parameters.clicked.connect(self.__on_btn_load_sim_parameters_clicked)
        if hasattr(self, 'btn_test_async_zapi_request'):
            self.btn_test_async_zapi_request.clicked.connect(self.on_btn_test_async_zapi_request_clicked)
            
        if hasattr(self, 'radio_mode_simulation'):
            self.radio_mode_simulation.toggled.connect(self.__on_radio_mode_toggled)
        if hasattr(self, 'radio_mode_real'):
            self.radio_mode_real.toggled.connect(self.__on_radio_mode_toggled)

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

        if getattr(self, 'mujoco_zapi', None):
            if hasattr(self, 'radio_mode_simulation') and self.radio_mode_simulation.isChecked():
                self.mujoco_zapi._ZAPI_request_set_mode("simulation")
            elif hasattr(self, 'radio_mode_real') and self.radio_mode_real.isChecked():
                self.mujoco_zapi._ZAPI_request_set_mode("real")

    def on_btn_test_async_zapi_request_clicked(self):
        """Handle async ZAPI test request button click"""
        pass

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
                    if self.zapi:
                        self.zapi._ZAPI_request_load_spool(str(sample_path.absolute()))
                        self.__console.info(f"Requested to load spool: {current_text}")
                    else:
                        self.__console.error("ZAPI instance not available")
                else:
                    self.__console.error(f"Spool file not found: {sample_path}")
        except Exception as e:
            self.__console.error(f"Error loading spool: {e}")

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

    def _handle_mujoco_message(self, topic, msg):
        """Handle incoming ZMQ messages from the MuJoCo backend."""
        try:
            if isinstance(topic, bytes):
                topic = topic.decode('utf-8')
            if isinstance(msg, bytes):
                msg = msg.decode('utf-8')

            if topic == "update_state_info":
                self.__console.info("Received state info from MuJoCo. Sending current execution mode.")
                if hasattr(self, 'radio_mode_simulation') and self.radio_mode_simulation.isChecked():
                    self.mujoco_zapi._ZAPI_request_set_mode("simulation")
                elif hasattr(self, 'radio_mode_real') and self.radio_mode_real.isChecked():
                    self.mujoco_zapi._ZAPI_request_set_mode("real")
                else:
                    self.mujoco_zapi._ZAPI_request_set_mode("simulation")
            else:
                self.__console.debug(f"Unhandled MuJoCo message: {topic} {msg}")

        except Exception as e:
            self.__console.error(f"Error handling MuJoCo message: {e}")
    
    def closeEvent(self, event:QCloseEvent) -> None:
        """ Handle close event """
        try:
            # ZAPI cleanup
            if hasattr(self, 'zapi') and self.zapi:
                self.zapi.stop()
                self.__console.info("ZAPI stopped")
            if hasattr(self, 'mujoco_zapi') and self.mujoco_zapi:
                self.mujoco_zapi.stop()
                self.__console.info("MuJoCo ZAPI stopped")

        except Exception as e:
            self.__console.error(f"Error during window close: {e}")
        finally:
            self.__console.info("Successfully Closed")
            super().closeEvent(event)
