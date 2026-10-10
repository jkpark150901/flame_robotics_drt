import threading
from collections import deque
import open3d as o3d
import time
import datetime
from util.logger.console import ConsoleLogger
from common.zpipe import AsyncZSocket, ZPipe
from viewero3d.container import GeometryContainer

from common import zapi

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from common.graphic_device import GraphicDevice

class Open3DVisualizer:
    def __init__(self, config:dict=None,  zpipe:ZPipe=None):
        if config is None:
            config = {}
    
        self.__console = ConsoleLogger.get_logger()
        
        # Initialize message queue and stats
        self._message_lock = threading.Lock()
        self._pending_messages = deque(maxlen=100)
        self._message_stats = {'received': 0, 'dropped': 0}

        # Initialize GeometryContainer
        self.container = GeometryContainer()

        # Device Detection
        self.gdevice = GraphicDevice()
        self.__console.info(f"Graphic Device: Running on {self.gdevice.get_device_name()}")

        # create & join asynczsocket
        # 1. Subscriber Socket (Recv data + System commands from SimTool)
        self.__socket = AsyncZSocket("Open3DVisualizer", "subscribe")
        if self.__socket.create(pipeline=zpipe):
            transport = config.get("transport", "tcp")
            port = config.get("port", 9001)
            host = config.get("host", "localhost")
            if self.__socket.join(transport, host, port):
                self.__socket.subscribe("call")
                self.__socket.subscribe(zapi.TOPIC_SYSTEM)
                self.__socket.set_message_callback(self.__on_data_received)
                self.__console.debug(f"Socket created and joined: {transport}://{host}:{port}")
            else:
                self.__console.error("Failed to join socket")
        else:
            self.__console.error("Failed to create socket")
            
        # 2. Publisher Socket (Send System commands)
        self.sys_pub_port = config.get("sys_pub_port", 9003)
        self.__sys_socket = AsyncZSocket("Open3DVisualizerSys", "publish")
        if self.__sys_socket.create(pipeline=zpipe):
            if self.__sys_socket.join("tcp", "*", self.sys_pub_port):
                self.__console.debug(f"System Publisher bound to: *: {self.sys_pub_port}")
            else:
                 self.__console.error("Failed to bind System Publisher")

        # Initialize Open3D GUI
        self.app = gui.Application.instance
        self.app.initialize()

        window_title = config.get('window_title', f'Open3D (Optimized - {self.gdevice.get_device_name()})')
        window_size = config.get('window_size', [1920, 1080])
        self.window = self.app.create_window(window_title, window_size[0], window_size[1])
        
        # Create SceneWidget
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        
        # Background Color
        bg_color = config.get('background-color', [1.0, 1.0, 1.0, 1.0])
        self.widget3d.scene.set_background(bg_color)
        
        # Lighting
        self.widget3d.scene.set_lighting(rendering.Open3DScene.LightingProfile.MED_SHADOWS, (0.577, -0.577, -0.577))
        
        self.window.add_child(self.widget3d)

        # Basic material
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        
        # Unlit material for coordinates
        mat_unlit = rendering.MaterialRecord()
        mat_unlit.shader = "defaultUnlit"

        # Add geometries from container
        for name, data in self.container.get_geometries().items():
            if data["visible"]:
                # user rendering.Open3DScene.add_geometry
                # Note: add_geometry(name, geometry, material)
                if "coordinate" in name:
                    self.widget3d.scene.add_geometry(name, data["geometry"], mat_unlit)
                else:
                    self.widget3d.scene.add_geometry(name, data["geometry"], mat)
                
        # Set default view
        self.set_perspective()
        
        # Flag for external termination
        self._should_close = False

        # Set tick callback for rendering loop logic (ZMQ polling etc)
        self.window.set_on_tick_event(self._on_tick)
        
        self.loop_count = 0
        self.last_log_time = time.time()
        self.target_frequency_hz = 60

    def set_isometric(self):
        """Set view to isometric (Orthographic)"""
        center = [0, 0, 0]
        # Position camera at a diagonal 
        up = [0, 0, 1]
        
        # 1. Set Look At/Camera Position
        # Move camera back significantly to ensure the 10000x10000 plane is in front of the camera plane
        eye = [5, 5, 5] 
        up = [0, 0, 1]
        
        self.widget3d.look_at(center, eye, up)
        
        # 2. Set Orthographic Projection
        # Define view volume (left, right, bottom, top, near, far)
        # Plane is 10000, so we need >10000. Using 15000 for margin.
        view_width = 15.0
        view_height = 15.0
        near_plane = 0.1
        far_plane = 50000.0 # Increased to cover the new distance from eye
        
        camera = self.widget3d.scene.camera
        camera.set_projection(rendering.Camera.Projection.Ortho, 
                              -view_width/2, view_width/2, 
                              -view_height/2, view_height/2, 
                              near_plane, far_plane)
                              
        self.__console.debug("View set to Isometric (Orthogonal)")

    def set_perspective(self):
        """Set view to perspective"""
        center = [0, 0, 0]
        eye = [5, 5, 5] 
        up = [0, 0, 1]
        self.widget3d.look_at(center, eye, up)
        
        field_of_view = 60.0
        aspect_ratio = 16.0/9.0 
        near_plane = 0.1
        far_plane = 1000.0
        
        camera = self.widget3d.scene.camera
        camera.set_projection(field_of_view, aspect_ratio, near_plane, far_plane, 
                              rendering.Camera.FovType.Vertical)
        
        self.__console.debug("View set to Perspective")

    def run(self, frequency_hz: int):
        self.target_frequency_hz = frequency_hz
        self.__console.info(f"Starting GUI loop (target: {frequency_hz} Hz)")
        
        # Blocking call
        self.app.run()
        
        self.on_close()
        self.__console.info("Visualizer closed")
    
    def _on_tick(self) -> bool:
        """Called every frame by the GUI event loop."""
        if self._should_close:
            self.app.quit()
            return False

        # 1. Log Frequency
        self.loop_count, self.last_log_time = self._log_rendering_frequency(self.loop_count, self.last_log_time)
        
        # 2. Poll ZMQ Messages
        processed_count = 0
        while True:
            try:
                with self._message_lock:
                    if not self._pending_messages:
                        break
                    multipart_data = self._pending_messages.popleft()
                
                self._process_data(multipart_data)
                processed_count += 1
                if processed_count > 10: 
                    break
            except Exception as e:
                self.__console.error(f"Error processing message: {e}")
                break
        
        # Trigger redraw if needed
        if processed_count > 0:
             self.window.post_redraw()

        return True

    def _process_data(self, multipart_data):
         # Implement data processing to update geometries
         # For now, just a placeholder or move logic from original code if any.
         # The original code `__on_data_received` just put it in queue. 
         # We need to actually apply it to the scene.
         pass

    def _log_rendering_frequency(self, loop_count, last_log_time):
        """
        Calculate and log the rendering frequency every 10 loops.
        """
        loop_count += 1
        if loop_count >= 10:
            current_time = time.time()
            duration = current_time - last_log_time
            if duration > 0:
                frequency = 10.0 / duration
            
            last_log_time = current_time
            loop_count = 0
            
        return loop_count, last_log_time

    def on_close(self):
        # Send termination signal
        if hasattr(self, '_Open3DVisualizer__sys_socket') and self.__sys_socket:
             zapi.zapi_destroy(self.__sys_socket)
             # Give a moment for message to fly out
             time.sleep(0.1)
             self.__sys_socket.destroy_socket()

        # Clean up subscriber socket
        if hasattr(self, '_Open3DVisualizer__socket') and self.__socket:
            self.__socket.destroy_socket()
            self.__console.debug(f"({self.__class__.__name__}) Destroyed socket")
        

    def __on_data_received(self, multipart_data):
        """Callback function for zpipe data reception"""
        try:
            if len(multipart_data) >= 2:
                topic = multipart_data[0]
                msg = multipart_data[1]

                with self._message_lock:
                    # Check if queue is full
                    if len(self._pending_messages) >= self._pending_messages.maxlen:
                        self._message_stats['dropped'] += 1
                        # Optionally log when dropping messages frequently
                        if self._message_stats['dropped'] % 50 == 0:
                            self.__console.warning(f"Dropped {self._message_stats['dropped']} messages due to processing lag")
                    
                    # Add to queue (deque will automatically drop oldest if at maxlen)
                    self._pending_messages.append(multipart_data)
                    self._message_stats['received'] += 1
                    
        except Exception as e:
            self.__console.error(f"({self.__class__.__name__}) Error processing received data: {e}")
