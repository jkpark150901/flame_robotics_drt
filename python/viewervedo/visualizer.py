"""
3D Visualizer using Vedo
@note
- Vedo is a Python library for 3D visualization based on VTK (Visualization Toolkit).
- Visualizer is a class that renders 3D geometries
- All ZMQ communication is handled by Zapi (viewervedo/zapi.py)
"""

import threading
from collections import deque
import time
import numpy as np
import vedo
from util.logger.console import ConsoleLogger
from common.graphic_device import GraphicDevice
from viewervedo.robot import load_robots_from_config


class Visualizer:
    def __init__(self, config:dict=None):
        if config is None:
            config = {}
    
        self.__console = ConsoleLogger.get_logger()

        # Thread-safe request queue (populated by Zapi)
        self._request_queue = deque(maxlen=100)
        self._queue_lock = threading.Lock()

        # Device Detection (Reusing GraphicDevice from common)
        self.gdevice = GraphicDevice()
        self.__console.info(f"Graphic Device: Running on {self.gdevice.get_device_name()}")
        
        # GPU Acceleration check for Vedo (VTK)
        if "cuda" in self.gdevice.get_device_name().lower() or "mps" in self.gdevice.get_device_name().lower():
             self.__console.info("GPU Acceleration enabled for Vedo/VTK (if available via drivers)")
             vedo.settings.use_depth_peeling = True # Better transparency on GPU
        else:
             self.__console.info("Running on CPU mode for Vedo/VTK")

        # Initialize Vedo Plotter
        window_title = config.get('window_title', f'Vedo Viewer (Optimized - {self.gdevice.get_device_name()})')
        window_size = config.get('window_size', [1920, 1080])
        bg_color = config.get('background_color', [1.0, 1.0, 1.0])
        
        # create plotter
        self.plotter = vedo.Plotter(title=window_title, size=window_size, bg=bg_color, interactive=False)

        # Setup scene elements
        self._setup_c_space(config)
        self._setup_robots(config)

        
        # Flag for external termination (set by Zapi)
        self._should_close = False

        self.loop_count = 0
        self.last_log_time = time.time()
        self.last_frame_time = time.time()
        self.target_frequency_hz = 60
        self.fps_text = None

        display_options = config.get("display_options", {})
        if display_options.get("show_fps", False):
            self.fps_text = vedo.Text2D("FPS: 0.0", pos='top-left', s=1.0, c="black", bg="white", alpha=0.5)
            self.plotter.add(self.fps_text)

        # Register key callback
        self.plotter.add_callback("KeyPress", self._on_key_press)

    def _setup_c_space(self, config: dict):
        """Add C-Space bounding box and axes to the plotter."""
        display_options = config.get("display_options", {})
        if not display_options.get("show_c_space", False):
            return

        self.c_bounds = config.get("c_space_bound", [5.0, 8.0, 5.0])
        c_bounds = self.c_bounds
        self.c_center = [c_bounds[0]/2, c_bounds[1]/2, c_bounds[2]/2]

        c_space_box = vedo.Box(
            pos=(c_bounds[0]/2, c_bounds[1]/2, c_bounds[2]/2),
            length=c_bounds[0], width=c_bounds[1], height=c_bounds[2]
        )
        c_space_box.wireframe().c('gray').alpha(0.3)

        # Create custom axes with 1-unit intervals
        x_ticks = [(i, str(i)) for i in range(int(c_bounds[0]) + 1)]
        y_ticks = [(i, str(i)) for i in range(int(c_bounds[1]) + 1)]
        z_ticks = [(i, str(i)) for i in range(int(c_bounds[2]) + 1)]

        axes_config = dict(
            xtitle='X', x_values_and_labels=x_ticks,
            ytitle='Y', y_values_and_labels=y_ticks,
            ztitle='Z', z_values_and_labels=z_ticks,
            c='black'
        )
        c_space_axes = vedo.Axes(c_space_box, **axes_config)
        self.plotter.add(c_space_box, c_space_axes)

    def _setup_robots(self, config: dict):
        """Load robot URDF models and add their meshes to the plotter."""
        robot_actors = load_robots_from_config(config)
        if robot_actors:
            self.plotter.add(*robot_actors)
            self.__console.info(f"Added {len(robot_actors)} robot mesh actors to plotter")

    def _on_key_press(self, event):
        """Handle key press events for camera control"""
        if not event.keypress:
            return

        key = event.keypress
        
        if not hasattr(self, 'c_bounds'):
            return

        # Direction vectors for each view (will be normalized internally)
        if key == '1': # XY Plane (Top View)
            self._set_camera_view((0, 0, 1), (0, 1, 0), "XY Plane (Top View)")
        elif key == '2': # YZ Plane (Side View)
            self._set_camera_view((1, 0, 0), (0, 0, 1), "YZ Plane (Side View)")
        elif key == '3': # XZ Plane (Front View)
            self._set_camera_view((0, -1, 0), (0, 0, 1), "XZ Plane (Front View)")
        elif key == '4': # Isometric View
            self._set_camera_view((1, 1, 1), (0, 0, 1), "Isometric View")

    def _set_camera_view(self, direction, view_up, label=None):
        """Set camera view from a direction vector, preserving current zoom level.
        
        Args:
            direction: (x, y, z) direction vector from focal point to camera
            view_up: (x, y, z) camera up-direction tuple
            label: optional log label for the view
        """
        if not hasattr(self, 'c_center'):
            return
        cx, cy, cz = self.c_center

        # Get current camera distance (zoom level) from focal point
        cam_pos = np.array(self.plotter.camera.GetPosition())
        focal = np.array([cx, cy, cz])
        current_dist = np.linalg.norm(cam_pos - focal)
        if current_dist < 1e-6:
            current_dist = max(self.c_bounds) * 2.0  # fallback

        # Normalize direction and apply current distance
        d = np.array(direction, dtype=float)
        d = d / np.linalg.norm(d)
        new_pos = focal + d * current_dist

        self.plotter.camera.SetPosition(*new_pos)
        self.plotter.camera.SetFocalPoint(cx, cy, cz)
        self.plotter.camera.SetViewUp(*view_up)
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()
        if label:
            self.__console.info(f"Camera set to {label}")

    def run(self, frequency_hz: int):
        self.target_frequency_hz = frequency_hz
        self.__console.debug(f"Starting Vedo GUI loop (target: {frequency_hz} Hz)")
        
        # shape initial view to 3D perspective if C-Space is defined
        if hasattr(self, 'c_bounds'):
             max_dim = max(self.c_bounds)
             self.plotter.show(interactive=False)
             if self.plotter.camera:
                 # Set initial distance, then apply isometric direction
                 cx, cy, cz = self.c_center
                 init_dist = max_dim * 2.0
                 iso_d = init_dist / np.sqrt(3)
                 self.plotter.camera.SetPosition(cx + iso_d, cy + iso_d, cz + iso_d)
                 self.plotter.camera.SetFocalPoint(cx, cy, cz)
                 self._set_camera_view((1, 1, 1), (0, 0, 1))
        else:
             self.plotter.show(interactive=False)
        
        while not self._should_close:
            if not self.plotter.interactor or self.plotter.interactor.GetDone():
                break
            start_time = time.time()
            
            # Logic step
            if not self._on_tick():
                break
                
            # Render step
            self.plotter.render()
            
            # Event processing (interactor)
            if self.plotter.interactor:
                self.plotter.interactor.ProcessEvents()
            
            # Timing control
            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.target_frequency_hz) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        self.on_close()
        self.plotter.close()
        self.__console.info("Visualizer closed")
    
    def _on_tick(self) -> bool:
        """Called every frame by the GUI event loop."""
        
        # 1. Log Frequency
        self.loop_count, self.last_log_time = self._log_rendering_frequency(self.loop_count, self.last_log_time)
        
        # 2. Process requests from ZApi queue
        processed_count = 0
        while processed_count < 10:
            with self._queue_lock:
                if not self._request_queue:
                    break
                request_data = self._request_queue.popleft()
            
            self._process_request(request_data)
            processed_count += 1
        
        return True

    def _process_request(self, request_data):
        """Process a request from the ZApi queue."""
        try:
            # Handle dictionary payload from zapi_* handlers
            if isinstance(request_data, dict):
                command = request_data.get("command")
                if command == "load_spool":
                    path = request_data.get("path")
                    if path:
                        self.__console.info(f"Loading Spool: {path}")
                        try:
                            mesh = vedo.load(path)
                            if mesh:
                                if hasattr(self, '_loaded_spool_mesh') and self._loaded_spool_mesh:
                                    self.plotter.remove(self._loaded_spool_mesh)

                                # PLY/PCD 단위: mm → 씬 단위: m  (÷ 1000)
                                mesh.scale(1.0 / 1000.0)
                                self.__console.info(
                                    f"Spool bounds after scaling (m): "
                                    f"{[round(v, 4) for v in mesh.bounds()]}"
                                )

                                self.plotter.add(mesh)
                                self._loaded_spool_mesh = mesh # Track the currently loaded spool
                                self.plotter.render()
                                self.__console.info(f"Successfully loaded {path}")
                                
                                # Send reply
                                if hasattr(self, 'zapi') and self.zapi:
                                    identity = request_data.get("_identity")
                                    self.zapi.reply_load_spool(path, True, identity=identity)
                            else:
                                self.__console.error(f"Failed to load mesh from {path}")
                                if hasattr(self, 'zapi') and self.zapi:
                                    identity = request_data.get("_identity")
                                    self.zapi.reply_load_spool(path, False, identity=identity)
                        except Exception as e:
                            self.__console.error(f"Exception loading mesh: {e}")
                            if hasattr(self, 'zapi') and self.zapi:
                                identity = request_data.get("_identity")
                                self.zapi.reply_load_spool(path, False, identity=identity)
                elif command == "load_test_weld_point":
                    path = request_data.get("path")
                    if path:
                        self.__console.info(f"Loading Test Weld Point from CSV: {path}")
                        # We will log it for now. Actual rendering can be implemented based on CSV format.
                        # Can be implemented further as needed.
                        self.__console.info(f"Successfully handled test weld point CSV path: {path}")
            
            # Legacy/Raw handling (list/tuple)
            elif isinstance(request_data, (list, tuple)) and len(request_data) >= 2:
                 pass # Handle raw messages if any
                 
        except Exception as e:
            self.__console.error(f"Error processing request: {e}")

    def set_zapi(self, zapi):
        """Set the ZApi instance for callbacks."""
        self.zapi = zapi

    def push_request(self, data):
        """Thread-safe method for ZApi to push requests into the visualizer queue."""
        with self._queue_lock:
            self._request_queue.append(data)

    def _log_rendering_frequency(self, loop_count, last_log_time):
        current_time = time.time()
        
        # Calculate Instantaneous FPS for Text Overlay
        frame_duration = current_time - self.last_frame_time
        if frame_duration > 0:
            inst_fps = 1.0 / frame_duration
            if self.fps_text:
                self.fps_text.text(f"FPS: {inst_fps:.1f}")
        
        self.last_frame_time = current_time
            
        return loop_count, last_log_time

    def on_close(self):
        """Cleanup on visualizer close. Socket cleanup is handled by Zapi."""
        self._should_close = True
        self.__console.debug("Visualizer on_close called")
