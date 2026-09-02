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

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import open3d as o3d
import time
import threading
import multiprocessing
from util.logger.console import ConsoleLogger
import zmq
import json
import numpy as np
import importlib
import inspect

from python.base.plugin_pd import PluginPDMBase
from python.base.plugin_pp import PluginRCMBase


class Graphic3DWindow():
    
    # geometry container to manage the visualizing object
    _geometry_container = {}

    def __init__(self, config:dict):
        super().__init__()

        # multiprocessing.set_start_method('spawn')
        self.__console = ConsoleLogger.get_logger() # logger

        self._config = config  # copy configuration data
        self._stop_render_event = multiprocessing.Event()
        self._stop_zmq_event = threading.Event()
        self._vis = None

        # manage zmq interface
        self.context = zmq.Context(1)

        # subscriber
        self._socket_sub = self.context.socket(zmq.SUB)
        self._socket_sub.setsockopt(zmq.RCVHWM, 100)
        self._socket_sub.setsockopt(zmq.RCVTIMEO, 500)
        self._socket_sub.setsockopt(zmq.LINGER, 0)
        self._socket_sub.connect("tcp://localhost:9000")
        self._socket_sub.subscribe("call")


        try:
            # [note!!] shared queue must be passed to the multiprocessing process first
            self._msgbox_process = multiprocessing.Queue()
            self._msgbox_thread = multiprocessing.Queue()
            self._render_proc = multiprocessing.Process(target=self._render_process, args=(self._stop_render_event, self._msgbox_process,), daemon=True)
            self._render_proc.start()

            self._zmq_pipeline_thread = threading.Thread(target=self.zmq_recv_process, args=(self._stop_zmq_event, self._msgbox_process,), daemon=True)
            self._zmq_pipeline_thread.start()

        except Exception as e:
            self.__console.debug("[Graphic 3D Window] Failed to create 3D rendering process")

        self.__console.debug("[Graphic 3D Window] Start Graphic 3D Viewer")

    def zmq_recv_process(self, stop_event, msgbox:multiprocessing.Queue):
        """ zmq pipeline process for receiving data """

        poller = zmq.Poller()
        poller.register(self._socket_sub, zmq.POLLIN)
        
        while not stop_event.is_set():
            try:
                events = dict(poller.poll(1000)) # wait 1 sec
                if self._socket_sub in events:
                    if events[self._socket_sub] == zmq.POLLIN:
                        # for 'call' topic
                        topic, msg = self._socket_sub.recv_multipart()
                        if topic.decode() == "call":
                            data = json.loads(msg.decode('utf8').replace("'", '"'))
                            msgbox.put(data)  # put data into the shared queue
                print(events)
                
            except json.JSONDecodeError as e:
                print(f"[Graphic 3D Window] {e}")
                continue
            except zmq.ZMQError as e:
                print(f"[Graphic 3D Window] {e}")
                break
            except Exception as e:
                print(f"[Graphic 3D Window] {e}")
                break

        poller.unregister(self._socket_sub)

    def close(self):
        """ close graphic 3d viewer """
        # terminate process
        self._stop_render_event.set()
        self._render_proc.join(timeout=3)

        # zmq pipeline terminate
        self._stop_zmq_event.set()
        self._zmq_pipeline_thread.join(timeout=3)

        try:
            self._socket_sub.setsockopt(zmq.LINGER, 0)
            #self._poller.unregister(self.__socket)
            self._socket_sub.close()

            self.context.destroy(0)
        except zmq.ZMQError as e:
            print(f"[Graphic 3D Window] {e}")

    @staticmethod
    def _render_process_gui(stop_event, queue:multiprocessing.Queue):
        try:
            gui.Application.instance.initialize()
            w = gui.Application.instance.create_window("Transparent Box", 1024, 768)

            scene = gui.SceneWidget()
            scene.scene = rendering.Open3DScene(w.renderer)
            w.add_child(scene)

            # 박스 생성
            box = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
            box.compute_vertex_normals()

            # 박스 재질 설정 (투명도 포함)
            material = rendering.MaterialRecord()
            material.base_color = [1.0, 0.0, 0.0, 0.5]  # RGBA (0.5 = 50% 투명)
            material.shader = "defaultLitTransparency"  # 꼭 필요

            scene.scene.add_geometry("box", box, material)
            scene.setup_camera(60, box.get_axis_aligned_bounding_box(), [0, 0, 0])

        except Exception as e:
            pass


    @staticmethod
    def _render_process(stop_event, msgbox:multiprocessing.Queue):
        try:
            console = ConsoleLogger.get_logger()
            import open3d as o3d
            import zmq

            context = zmq.Context(1)
            socket_pub = context.socket(zmq.PUB)
            socket_pub.setsockopt(zmq.SNDHWM, 100)
            socket_pub.setsockopt(zmq.LINGER, 0)
            socket_pub.bind("tcp://*:9001")

            # create 3d window
            _vis = o3d.visualization.VisualizerWithKeyCallback()
            _vis.create_window(window_name='3D Graphic Viewer', width=1920, height=1920, left=50, top=50)
            _vis.get_render_option().background_color = [1.0, 1.0, 1.0]
            _vis.register_key_callback(ord("A"), Graphic3DWindow.key_press_callback)

            while not stop_event.is_set():
                while not msgbox.empty():
                    msg = msgbox.get()
                    Graphic3DWindow.geometry_manage(_vis, msg) # JSON-style message process interface
                    Graphic3DWindow._notify_geometry(socket_pub)
                _vis.poll_events()
                _vis.update_renderer()
                time.sleep(0.01)
        except Exception as e:
            console.error(f"[Graphic 3D Window] Error in rendering: {e}")

        _vis.destroy_window()
        Graphic3DWindow._geometry_container.clear()
        console.debug("[Graphic 3D Window] Render process terminated.")
    
    @staticmethod
    def key_press_callback(vis):
        print("(test)key pressed")
        return False
        
    
    @staticmethod
    def geometry_manage(visualizer:o3d.visualization.Visualizer, msg:dict) -> None:
        try:
            # static function call
            function_name = msg["function"]
            function = getattr(Graphic3DWindow, function_name)
            kwargs = msg["kwargs"]
            function(visualizer, **kwargs)
        except Exception as e:
            print(f"[3D Viewer] Exception : {e}")

    @staticmethod
    def _load_pdm_plugin(plugin_name:str) -> PluginPDMBase:
        module = importlib.import_module(f"plugin_pdm.{plugin_name}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, PluginPDMBase) and obj is not PluginPDMBase:
                return obj()
        raise Exception("No valid PDM plugin found")

    @staticmethod
    def API_find_pose(vis, plugin:str):
        print(f"Call API_find_pose {plugin}")
        module = Graphic3DWindow._load_pdm_plugin(plugin)
        if module:
            module.find_pose()

    @staticmethod
    def API_add_pcd(vis, name:str, path:str, pos:list=[0,0,0], ori:list=[0,0,0], color:list=[0,0,0]):
        print(f"Call API_add_pcd")
        obj = o3d.io.read_point_cloud(path)
        if not obj.has_points():
            print(f"[API_add_pcd] PCD has no points")
            return None
        
        R_matrix = obj.get_rotation_matrix_from_axis_angle(rotation=ori) # rotation
        obj.rotate(R=R_matrix)
        obj.translate(translation=pos, relative=False) # translation
        
        num_points = np.asarray(obj.points).shape[0]
        black_colors = np.tile(color, (num_points, 1))  # shape: (N, 3)
        obj.colors = o3d.utility.Vector3dVector(black_colors)
        obj.point_size = 1.0

        # for bbox
        aabb = obj.get_axis_aligned_bounding_box()
        aabb.color = (1,0,0)

        if not name in Graphic3DWindow._geometry_container.keys():
            Graphic3DWindow._geometry_container[name] = obj
            vis.add_geometry(obj, reset_bounding_box=True)
            Graphic3DWindow._geometry_container[f"{name}_bbox"] = aabb
            vis.add_geometry(aabb)
        else:
            print(f"[API_add_pcd] Already added {name} in geomery container")

    @staticmethod
    def API_remove_geometry(vis, name:str) -> None:
        print(f"Call API_remove_geometry")
        if name in Graphic3DWindow._geometry_container.keys():
            vis.remove_geometry(Graphic3DWindow._geometry_container[name])
            del Graphic3DWindow._geometry_container[name]
        else:
            print(f"[API_remove_geometry] {name} geometry cannot find in geometry container")
    
    @staticmethod
    def _notify_geometry(socket):
        if socket:
            try:
                topic = "call"
                message = {"msg":"test"}
                if socket:
                    socket.send_multipart([topic.encode(), json.dumps(message).encode()])
                    print("send message")
            except zmq.ZMQError as e:
                print(f"Notify Error ZMQ : {e}")
            except json.JSONDecodeError as e:
                print(f"Notify Error JSON : {e}")        
    
    @staticmethod
    def API_add_coord_frame(vis, name:str, pos:list=[0,0,0], ori:list=[0,0,0], size:float=0.1) -> None:
        print(f"Call API_add_coord_frame")
        obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=pos)
        R_matrix = obj.get_rotation_matrix_from_axis_angle(rotation=ori) # rotation
        obj.rotate(R=R_matrix)
        obj.translate(translation=pos, relative=False)
        if not name in Graphic3DWindow._geometry_container.keys():
            Graphic3DWindow._geometry_container[name] = obj
            vis.add_geometry(obj, reset_bounding_box=True)
        else:
            print(f"[api_add_coord_frame] Already added {name} in geomery container")

    @staticmethod
    def API_add_box(vis, name:str, pos:list=[0,0,0], ori:list=[0,0,0], size:list=[1.0, 1.0, 1.0], color:list=[0,0,0]):
        print(f"Call API_add_box")
        obj = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
        obj.compute_vertex_normals()
        obj.paint_uniform_color(color=color)
        R_matrix = obj.get_rotation_matrix_from_axis_angle(rotation=ori) # rotation
        obj.rotate(R=R_matrix)
        obj.translate(translation=pos, relative=False)
        if not name in Graphic3DWindow._geometry_container.keys():
            Graphic3DWindow._geometry_container[name] = obj
            vis.add_geometry(obj, reset_bounding_box=True)
        else:
            print(f"[API_add_box] Already added {name} in geomery container")

    @staticmethod
    def API_add_floor(vis, name:str, pos:list=[0,0,0], ori:list=[0,0,0], size:list=[1.0,1.0,1.0], color:list=[0,0,0]):
        print(f"Call API_add_floor")
        obj = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
        obj.compute_vertex_normals()
        R_matrix = obj.get_rotation_matrix_from_axis_angle(rotation=ori) # rotation
        obj.rotate(R=R_matrix)
        obj.translate(translation=pos, relative=False)
        Graphic3DWindow._add_geometry(vis, name, obj)

    @staticmethod
    def API_clear_geometry_all(vis):
        print(f"Call API_clear_geometry_all")
        for name in Graphic3DWindow._geometry_container.keys():
            vis.remove_geometry(Graphic3DWindow._geometry_container[name])
            print(f"Remove Geometry : {name}")
        Graphic3DWindow._geometry_container.clear()
        
    @staticmethod
    def _add_geometry(vis, name:str, obj):
        if not name in Graphic3DWindow._geometry_container.keys():
            Graphic3DWindow._geometry_container[name] = obj
            vis.add_geometry(obj, reset_bounding_box=True)
        else:
            print(f"[API] Already added {name} in geomery container")





    # def _add_sphere(self):
    #     mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    #     mesh.paint_uniform_color([1, 0.5, 0])
    #     mesh.compute_vertex_normals()
    #     self._add_geometry(mesh, "sphere")

    # def _add_pcd(self):
    #     if not os.path.exists("example.pcd"):
    #         print("example.pcd 파일이 현재 폴더에 없습니다.")
    #         return
    #     pcd = o3d.io.read_point_cloud("example.pcd")
    #     mat = rendering.MaterialRecord()
    #     mat.shader = "defaultUnlit"
    #     mat.point_size = 5
    #     key = f"pcd_{self.counter}"
    #     self.scene.scene.add_geometry(key, pcd, mat)
    #     self.geometry_map[key] = pcd
    #     self.counter += 1
    #     self._fit_scene()

    # def _add_stl(self):
    #     if not os.path.exists("model.stl"):
    #         print("model.stl 파일이 현재 폴더에 없습니다.")
    #         return
    #     mesh = o3d.io.read_triangle_mesh("model.stl")
    #     mesh.compute_vertex_normals()
    #     mesh.paint_uniform_color([0.4, 0.4, 1.0])
    #     self._add_geometry(mesh, "stl")

    # def _clear_all(self):
    #     for name in list(self.geometry_map.keys()):
    #         self.scene.scene.remove_geometry(name)
    #     self.geometry_map.clear()
    #     self.counter = 0

    # def _fit_scene(self):
    #     bounds = self.scene.scene.bounding_box
    #     self.scene.setup_camera(60, bounds, bounds.get_center())



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

