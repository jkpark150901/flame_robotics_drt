'''
Utility for generating point cloud data from a 3D model.
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

import os, sys
import pathlib
import trimesh
import numpy as np
import open3d as o3d

from util.logger.console import ConsoleLogger


class AppWindow(QMainWindow):
    def __init__(self, config:dict, target:str):
        """ initialization """
        super().__init__()
        
        self.__console = ConsoleLogger.get_logger() # logger
        self.__config = config  # copy configuration data
        self.__model_file = target

        try:            
            if "util_gen_pcd_gui" in config:
                ui_path = pathlib.Path(config["app_path"]) / config["util_gen_pcd_gui"]
                if os.path.isfile(ui_path):
                    loadUi(ui_path, self)
                    self.setWindowTitle("Generate to PCD from 3D Model")

                    # interface components
                    self.btn_generate.clicked.connect(self.generate_pcd)
                    self.chk_without_noise.toggled.connect(self.__without_noise)
                    
                else:
                    raise Exception(f"Cannot found UI file : {ui_path}")

        except Exception as e:
            self.__console.error(f"{e}")

    def __without_noise(self, checked) -> None:
        """ Toggle noise addition """
        self.findChild(QLineEdit, "edit_noise").setDisabled(checked)

    def generate_pcd(self) ->bool:
        """ Generate PCD from 3D model """
        mesh = trimesh.load(self.__model_file)

        n_sample_points = int(self.findChild(QLineEdit, "edit_sample_points").text())
        points = mesh.sample(n_sample_points)
        pcd = o3d.geometry.PointCloud()

        print(n_sample_points, noise_std_dev)

        if not self.findChild(QCheckBox, name="chk_without_noise").isChecked() and self.findChild(QLineEdit, "edit_noise").text().count()>0:
            noise_std_dev = float(self.findChild(QLineEdit, "edit_noise").text())
            noise = np.random.normal(0, noise_std_dev, size=points.shape)
            noised_points = points + noise
            pcd.points = o3d.utility.Vector3dVector(noised_points)
        else:
            pcd.points = o3d.utility.Vector3dVector(points)

        filename, selected_filter = QFileDialog.getSaveFileName(self, "Save to PCD File", "", "PCD File (*.pcd);;PLY File(*.ply);;XYZ File(*.xyz);;PTS File(*.pts);;OBJ File(*.obj)")
        if filename:
            if not os.path.splitext(filename)[1]:
                if selected_filter.startswith("PCD"):
                    filename += ".pcd"
                elif selected_filter.startswith("PLY"):
                    filename += ".ply"
                elif selected_filter.startswith("XYZ"):
                    filename += ".xyz"
                elif selected_filter.startswith("PTS"):
                    filename += ".pts"
                elif selected_filter.startswith("OBJ"):
                    filename += ".obj"
                o3d.io.write_point_cloud(filename, pcd)

        # close window
        self.close()

    def closeEvent(self, event:QCloseEvent) -> None:
        """ Handle close event """
        return super().closeEvent(event)

