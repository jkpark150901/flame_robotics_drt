'''
DRT Pose Detection Module Window
@Author Byunghun Hwang<bh.hwang@iae.re.kr>
'''

try:
    # using PyQt6
    from PyQt6.QtGui import QImage, QPixmap, QCloseEvent, QStandardItem, QStandardItemModel
    from PyQt6.QtWidgets import QApplication, QFrame, QMainWindow, QLabel, QPushButton, QCheckBox, QComboBox, QDialog
    from PyQt6.QtWidgets import QMessageBox, QProgressBar, QFileDialog, QComboBox, QLineEdit, QSlider, QVBoxLayout
    from PyQt6.uic import loadUi
    from PyQt6.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal
except ImportError:
    print("PyQt6 is required to run this application.")

import zmq
import os, sys
import pathlib
import json
import importlib
import inspect

from util.logger.console import ConsoleLogger
from python.plugins.plugin_pdm.base_pd import PluginPDBase


class AppWindow(QMainWindow):
    def __init__(self, context:zmq.Context, socket, config:dict):
        """ initialization """
        super().__init__()
        
        self.__console = ConsoleLogger.get_logger() # logger
        self.__config = config  # copy configuration data

        # publisher socket
        self._publisher = socket

        try:            
            if "pdm_gui" in config:
                ui_path = pathlib.Path(config["app_path"]) / config["pdm_gui"]
                if os.path.isfile(ui_path):
                    loadUi(ui_path, self)
                    self.setWindowTitle(config.get("pdm_window_title", "Pose Detection View"))
                    self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)

                    # module list
                    for plugin_name in config["pdm_plugins"]:
                        self.cbx_pdm_plugins.addItem(plugin_name)

                    self.btn_find_pose.clicked.connect(self.on_btn_find_pose)

                    self.show()
                else:
                    raise Exception(f"Cannot found UI file : {ui_path}")
            

        except Exception as e:
            self.__console.error(f"{e}")


    def closeEvent(self, event:QCloseEvent) -> None:
        """ Handle close event """
        self.__console.info("Closing Pose Detection Window")
        return super().closeEvent(event)
    
    def on_btn_find_pose(self):
        selected_plugin = self.cbx_pdm_plugins.currentText()
        self.__call(function="API_find_pose_candidates", kwargs={"plugin":selected_plugin})
    
    def __call(self, function:str, kwargs:dict):
        topic = "call"
        message = {"function":function, "kwargs":kwargs}
        if self._publisher:
            self._publisher.send_multipart([topic.encode(), json.dumps(message).encode()])
        else:
            self.__console.warning(f"Failed send")

