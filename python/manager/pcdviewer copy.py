'''
3D Visualizer for Simulation
@Author Byunghun Hwang<bh.hwang@iae.re.kr>
'''

try:
    # using PyQt6
    from PyQt6.QtGui import QImage, QPixmap, QCloseEvent, QStandardItem, QStandardItemModel
    from PyQt6.QtWidgets import QApplication, QFrame, QMainWindow, QLabel, QPushButton, QCheckBox, QComboBox, QDialog
    from PyQt6.QtWidgets import QMessageBox, QProgressBar, QFileDialog, QComboBox, QLineEdit, QSlider, QVBoxLayout
    from PyQt6.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal
except ImportError:
    print("PyQt6 is required to run this application.")

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import time
import threading
from util.logger.console import ConsoleLogger

class PCDViewer(QObject):
    def __init__(self):
        super().__init__()

        self.__console = ConsoleLogger.get_logger()

        self.gui = gui.Application.instance
        self.gui.initialize()

        self.window = self.gui.create_window("Open3D GUI Viewer", 800, 600)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        self.scene.scene.set_background([1, 1, 1, 1])  # 흰 배경
        self.scene.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0, 0, 0))

        self.geometry_map = {}
        self.counter = 0

        

        # self.__vis = o3d.visualization.Visualizer()
        # self.__vis.create_window(window_name='3D Viewer')
        # self.__geometry_list = []

        # self.__render_stop_event = threading.Event()
        # self.__render_thread = threading.Thread(target=self.run, daemon=True)
        # self.__render_thread.start()

    def close(self) -> None:
        """ close viewer """
        pass
        # self.__render_stop_event.set()
        # self.__render_thread.join()

        # self.__vis.destroy_window()

    def _add_geometry(self, geometry, name):
        # 각 지오메트리에 고유한 이름 부여
        key = f"{name}_{self.counter}"
        self.scene.scene.add_geometry(key, geometry, rendering.MaterialRecord())
        self.geometry_map[key] = geometry
        self.counter += 1
        self._fit_scene()

    def _add_box(self):
        mesh = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
        mesh.paint_uniform_color([0.2, 0.8, 0.2])
        mesh.compute_vertex_normals()
        self._add_geometry(mesh, "box")

    def _add_sphere(self):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        mesh.paint_uniform_color([1, 0.5, 0])
        mesh.compute_vertex_normals()
        self._add_geometry(mesh, "sphere")

    def _add_pcd(self):
        if not os.path.exists("example.pcd"):
            print("example.pcd 파일이 현재 폴더에 없습니다.")
            return
        pcd = o3d.io.read_point_cloud("example.pcd")
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 5
        key = f"pcd_{self.counter}"
        self.scene.scene.add_geometry(key, pcd, mat)
        self.geometry_map[key] = pcd
        self.counter += 1
        self._fit_scene()

    def _add_stl(self):
        if not os.path.exists("model.stl"):
            print("model.stl 파일이 현재 폴더에 없습니다.")
            return
        mesh = o3d.io.read_triangle_mesh("model.stl")
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.4, 0.4, 1.0])
        self._add_geometry(mesh, "stl")

    def _clear_all(self):
        for name in list(self.geometry_map.keys()):
            self.scene.scene.remove_geometry(name)
        self.geometry_map.clear()
        self.counter = 0

    def _fit_scene(self):
        bounds = self.scene.scene.bounding_box
        self.scene.setup_camera(60, bounds, bounds.get_center())

    def run(self):
        self.gui.run()

    # def run(self):
    #     """ run render """
    #     while not self.__render_stop_event.is_set():
    #         try:
    #             self.__vis.poll_events()
    #             time.sleep(0.1)  # Control the frame rate
    #             print("render")
    #         except Exception as e:
    #             self.__console.error(f"Error in PCDViewer : {e}")
    #             break

    #     # while not self.isInterruptionRequested():
    #     #     try:
    #     #         # self.__vis.poll_events()
    #     #         # self.__vis.update_renderer()
    #     #         time.sleep(0.1)  # Control the frame rate
    #     #         print("render")
    #     #     except Exception as e:
    #     #         self.__console.error(f"Error in PCDViewer : {e}")
    #     #         break

    # def add_geometry(self, geometry):
    #     if geometry is not None:
    #         self.__geometry_list.append(geometry)
    #         self.__vis.add_geometry(geometry)
    #     else:
    #         print("Geometry is None, cannot add to viewer.")

    # def remove_geometry(self, geometry):
    #     if geometry in self.__geometry_list:
    #         self.__geometry_list.remove(geometry)
    #         self.__vis.remove_geometry(geometry)
    #     else:
    #         print("Geometry not found in viewer.")

    # def add_pcd(self, pcd_file:str):
    #     pcd = o3d.io.read_point_cloud(pcd_file)
    #     self.add_geometry(pcd)

        """
        # ✅ 2. Box 생성 및 추가
box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
box.paint_uniform_color([0.2, 0.8, 0.2])
box.translate([2, 0, 0])
viewer.add_geometry(box)

# ✅ 3. STL 파일 로드 및 추가
stl = o3d.io.read_triangle_mesh("model.stl")  # 🔁 파일 경로 수정 필요
stl.compute_vertex_normals()
stl.paint_uniform_color([0.5, 0.5, 1.0])
stl.translate([-2, 0, 0])
viewer.add_geometry(stl)
"""
