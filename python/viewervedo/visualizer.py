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
import open3d as _o3d
from util.logger.console import ConsoleLogger
from common.graphic_device import GraphicDevice
from viewervedo.robot import RobotModel, load_robots_from_config


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
        self._robot_models = []
        root_path = config.get("root_path", "")
        for entry in config.get("urdf", []):
            import os
            name = entry.get("name", "unknown")
            path = entry.get("path", "")
            base = entry.get("base", [0, 0, 0, 0, 0, 0])
            full_path = os.path.join(str(root_path), path) if root_path else path
            if not os.path.exists(full_path):
                self.__console.error(f"[Robot] URDF file not found: {full_path}")
                continue
            model = RobotModel(name=name, urdf_path=full_path, base_pose=base)
            model.load()
            self._robot_models.append(model)

        all_actors = [a for m in self._robot_models for a in m.actors]
        if all_actors:
            self.plotter.add(*all_actors)
            self.__console.info(f"Added {len(all_actors)} robot mesh actors to plotter")

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

    def _rotate_point_about_x(self, point, angle_deg, center):
        """Rotate a point around the global X axis."""
        point = np.array(point, dtype=float)
        center = np.array(center, dtype=float)
        rad = np.deg2rad(angle_deg)
        rel = point - center
        cos_v = np.cos(rad)
        sin_v = np.sin(rad)
        rotated = np.array([
            rel[0],
            rel[1] * cos_v - rel[2] * sin_v,
            rel[1] * sin_v + rel[2] * cos_v,
        ])
        return center + rotated

    def _get_spool_pose_payload(self):
        if getattr(self, '_spool_fixed', False):
            # 고정 시 UI 포즈는 0 (offset이 배치를 담음)
            payload = {
                "x": 0.0, "y": 0.0, "z": 0.0,
                "x_rotation": 0.0, "z_rotation": 0.0,
                "fixed": True,
            }
        else:
            pos = np.array(getattr(self, '_spool_manual_pos', [0.0, 0.0, 0.0]), dtype=float)
            payload = {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "x_rotation": float(getattr(self, '_spool_manual_x_rot_deg', 0.0)),
                "z_rotation": float(getattr(self, '_spool_manual_z_rot_deg', 0.0)),
                "fixed": False,
            }
        off = getattr(self, '_spool_offset_T', None)
        if off is not None:
            payload["offset_R"] = off[:3, :3].tolist()
            payload["offset_t"] = off[:3, 3].tolist()
        return payload

    def _send_spool_pose_update(self, identity=None):
        if hasattr(self, 'zapi') and self.zapi and identity:
            self.zapi.update_spool_pose(self._get_spool_pose_payload(), identity=identity)

    def _get_spool_points(self):
        """현재 로드된 스풀 actor(들)의 월드 좌표 점을 (N,3)로 반환. 없으면 None."""
        spool = getattr(self, '_loaded_spool_mesh', None)
        if spool is None:
            return None
        actors = spool if isinstance(spool, (list, tuple)) else [spool]
        all_verts = []
        for a in actors:
            if hasattr(a, "vertices"):
                v = np.asarray(a.vertices)
                if len(v):
                    all_verts.append(v)
        if not all_verts:
            return None
        return np.vstack(all_verts)

    def _replace_spool_points(self, new_pts):
        """스풀 actor를 새 점군으로 교체 (필터 결과 반영)."""
        old = getattr(self, '_loaded_spool_mesh', None)
        if old is not None:
            self.plotter.remove(old)
        recon = getattr(self, '_spool_recon_mesh', None)
        if recon is not None:
            self.plotter.remove(recon)
            self._spool_recon_mesh = None
            self._spool_recon_mesh_o3d = None
        new_actor = vedo.Points(np.asarray(new_pts, dtype=np.float64))
        self.plotter.add(new_actor)
        self._loaded_spool_mesh = new_actor
        self.plotter.render()

    def _filter_loaded_spool(self, request_data):
        """현재 로드된 스풀에 직접 노이즈 필터(SOR/CCL)를 적용."""
        pts = self._get_spool_points()
        if pts is None:
            self.__console.warning("filter_spool: 로드된 스풀이 없습니다")
            return
        method = (request_data.get("method") or "").lower()
        params = request_data.get("params", {}) or {}
        n0 = len(pts)
        try:
            if method == "sor":
                pcd = _o3d.geometry.PointCloud()
                pcd.points = _o3d.utility.Vector3dVector(pts)
                clean, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=int(params.get("neighbors", 20)),
                    std_ratio=float(params.get("std_ratio", 2.0)))
                kept = np.asarray(clean.points)
            elif method == "ccl":
                from util.pcd_tool import voxel_ccl
                level = int(params.get("level", 7))
                min_points = int(params.get("min_points", 30))
                extent = float((pts.max(axis=0) - pts.min(axis=0)).max()) * 1.01
                voxel = extent / (2 ** level)
                _, labels = voxel_ccl(pts, voxel, min_points=min_points, connectivity=26)
                valid = labels[labels >= 0]
                if len(valid) == 0:
                    self.__console.warning("filter_spool(ccl): 연결요소 없음")
                    return
                uniq, cnts = np.unique(valid, return_counts=True)
                kept = pts[labels == uniq[np.argmax(cnts)]]
            else:
                self.__console.warning(f"filter_spool: 알 수 없는 method '{method}'")
                return
            self._replace_spool_points(kept)
            self.__console.info(f"filter_spool({method}): {n0} → {len(kept)} 점 (제거 {n0 - len(kept)})")
        except Exception as e:
            self.__console.error(f"filter_spool 실패: {e}")

    def _reconstruct_loaded_spool_mesh(self, request_data):
        """현재 로드된 스풀 점군으로 메시 재건(Marching Cubes) 후 표시."""
        pts = self._get_spool_points()
        if pts is None:
            self.__console.warning("reconstruct_mesh: 로드된 스풀이 없습니다")
            return
        params = request_data.get("params", {}) or {}
        try:
            from util.pcd_tool import reconstruct_mesh_marching_cubes
            pcd = _o3d.geometry.PointCloud()
            pcd.points = _o3d.utility.Vector3dVector(pts)
            mesh_o3d = reconstruct_mesh_marching_cubes(
                pcd,
                resolution=int(params.get("resolution", 128)),
                sigma=float(params.get("sigma", 1.5)),
                level=float(params.get("level", 0.5)))
            verts = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            if len(verts) == 0 or len(faces) == 0:
                self.__console.warning("reconstruct_mesh: 빈 메시")
                return
            vmesh = vedo.Mesh([verts, faces]).c("gray")
            old = getattr(self, '_spool_recon_mesh', None)
            if old is not None:
                self.plotter.remove(old)
            self.plotter.add(vmesh)
            self._spool_recon_mesh = vmesh
            self._spool_recon_mesh_o3d = mesh_o3d  # 저장용 원본 보관
            self.plotter.render()
            self.__console.info(f"reconstruct_mesh: 정점 {len(verts)}, 면 {len(faces)}")
        except Exception as e:
            self.__console.error(f"reconstruct_mesh 실패: {e}")

    def _save_loaded_spool(self, request_data):
        """현재 결과를 저장. 재건 메시가 있으면 메시를, 없으면 점군을 저장."""
        path = request_data.get("path")
        if not path:
            return
        try:
            recon_o3d = getattr(self, '_spool_recon_mesh_o3d', None)
            if getattr(self, '_spool_recon_mesh', None) is not None and recon_o3d is not None:
                _o3d.io.write_triangle_mesh(path, recon_o3d)
                self.__console.info(f"save_spool: 메시 저장 {path}")
            else:
                pts = self._get_spool_points()
                if pts is None:
                    self.__console.warning("save_spool: 저장할 스풀이 없습니다")
                    return
                pcd = _o3d.geometry.PointCloud()
                pcd.points = _o3d.utility.Vector3dVector(pts)
                _o3d.io.write_point_cloud(path, pcd)
                self.__console.info(f"save_spool: 점군 저장 {path} ({len(pts)} 점)")
        except Exception as e:
            self.__console.error(f"save_spool 실패: {e}")

    # --- 스풀 프레임/고정(강체 부착) 유틸 ---
    CHUCK_LINK_NAME = "m_column_passive_r"

    @staticmethod
    def _rotz(deg):
        r = np.deg2rad(deg); c, s = np.cos(r), np.sin(r)
        T = np.eye(4); T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c
        return T

    @staticmethod
    def _rotx(deg):
        r = np.deg2rad(deg); c, s = np.cos(r), np.sin(r)
        T = np.eye(4); T[1, 1] = c; T[1, 2] = -s; T[2, 1] = s; T[2, 2] = c
        return T

    @staticmethod
    def _transl(v):
        T = np.eye(4); T[:3, 3] = np.asarray(v, dtype=float)
        return T

    def _chuck_world_T(self):
        """column m chuck joint(m_column_passive_r) 링크의 4x4 월드 변환."""
        for model in getattr(self, '_robot_models', []):
            if hasattr(model, 'get_link_world_T'):
                T = model.get_link_world_T(self.CHUCK_LINK_NAME)
                if T is not None:
                    return np.asarray(T, dtype=float)
        return None

    def _render_spool_world_T(self):
        """_spool_world_T 와 _spool_local_verts 로 스풀 actor 정점을 절대 갱신."""
        local = getattr(self, '_spool_local_verts', None)
        spool = getattr(self, '_loaded_spool_mesh', None)
        T = getattr(self, '_spool_world_T', None)
        if local is None or spool is None or T is None:
            return False
        actors = spool if isinstance(spool, (list, tuple)) else [spool]
        if not actors:
            return False
        world = (T[:3, :3] @ local.T).T + T[:3, 3]
        a = actors[0]
        if hasattr(a, 'vertices'):
            a.vertices = world
            return True
        return False

    def _fix_spool(self, request_data=None):
        """현재 chuck 기준 스풀 offset(R,t)을 저장하고 강체 부착(고정)한다.
        고정 시 UI 포즈는 0으로 초기화되며 이후 포지셔너 이동에 따라 절대 재계산된다.
        """
        if getattr(self, '_spool_local_verts', None) is None or getattr(self, '_spool_world_T', None) is None:
            self.__console.warning("fix_spool: 스풀 프레임 정보가 없습니다(미로드/비PCD)")
            return
        Tc = self._chuck_world_T()
        if Tc is None:
            self.__console.warning("fix_spool: chuck 변환을 찾을 수 없습니다")
            return
        self._spool_offset_T = np.linalg.inv(Tc) @ self._spool_world_T
        self._spool_fixed = True
        # UI 포즈 0 초기화 (offset이 배치를 담으므로)
        self._spool_manual_x_rot_deg = 0.0
        self._spool_manual_z_rot_deg = 0.0
        self.__console.info("fix_spool: chuck 기준 offset 저장, 스풀 강체 부착")
        if request_data is not None:
            self._send_spool_pose_update(request_data.get("_identity"))

    def _unfix_spool(self, request_data=None):
        """고정 해제. 스풀은 현재 위치에 그대로 머문다."""
        self._spool_fixed = False
        self._spool_offset_T = None
        self.__console.info("unfix_spool: 스풀 고정 해제")
        if request_data is not None:
            self._send_spool_pose_update(request_data.get("_identity"))

    def _set_spool_offset(self, request_data):
        """로드 복원용: 저장된 offset(R,t)을 적용해 현재 chuck에 강체 부착."""
        if getattr(self, '_spool_local_verts', None) is None:
            self.__console.warning("set_spool_offset: 스풀 프레임 정보 없음")
            return
        R = np.asarray(request_data.get("offset_R"), dtype=float)
        t = np.asarray(request_data.get("offset_t"), dtype=float)
        if R.shape != (3, 3) or t.shape != (3,):
            self.__console.warning("set_spool_offset: offset 형식 오류")
            return
        off = np.eye(4); off[:3, :3] = R; off[:3, 3] = t
        Tc = self._chuck_world_T()
        if Tc is None:
            return
        self._spool_offset_T = off
        self._spool_fixed = True
        self._spool_world_T = Tc @ off
        self._render_spool_world_T()
        self.plotter.render()
        self.__console.info("set_spool_offset: offset 적용 및 강체 부착 복원")

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
                            import pathlib as _pl
                            if _pl.Path(path).suffix.lower() == ".pcd":
                                
                                _pcd = _o3d.io.read_point_cloud(path)
                                _pts = np.asarray(_pcd.points, dtype=np.float64)
                                mesh = vedo.Points(_pts)
                            else:
                                mesh = vedo.load(path)
                            if mesh is not None:
                                # 포지셔너 원점 상태에서 척 조인트 중심을 대략 계산한다.
                                # URDF 기준: base_to_m_column(8.0, 1.377, 0.328)
                                #        + m_column_z_to_m_column_passive_r(-0.247, 0, 0.885)
                                # viewervedo 설정에서 positioner base가 y=0.5만큼 이동되어 있다.
                                chuck_center = [7.753, 1.877, 1.213]
                                # m_column_passive_r visual mesh의 X 방향 길이는 약 0.442 m이다.
                                # 배관 원점을 척 길이만큼 x 방향으로 보낸 뒤 Z축 기준 -90도로 정렬한다.
                                spool_origin = [chuck_center[0] - 0.442, chuck_center[1], chuck_center[2]]
                                self._loaded_spool_origin = spool_origin
                                self._spool_manual_pos = np.array(spool_origin, dtype=float)
                                self._spool_manual_x_rot_deg = 0.0
                                self._spool_manual_z_rot_deg = 0.0
                                # 스풀 프레임 추적: local verts(배치 전, 스케일만) + world_T(배치 변환)
                                # world = R @ local + t  형태로 절대 변환 관리 (고정 시 강체 부착에 사용)
                                _is_pcd = _pl.Path(path).suffix.lower() == ".pcd"
                                self._spool_local_verts = (_pts * 1e-3) if _is_pcd else None
                                _Tb = self._rotz(-90)
                                _Tb[:3, 3] = np.asarray(spool_origin, dtype=float)
                                self._spool_world_T = _Tb
                                self._spool_fixed = False
                                self._spool_offset_T = None
                                if isinstance(mesh, (list, tuple)):
                                    for actor in mesh:
                                        if hasattr(actor, "scale"):
                                            actor.scale(1e-3, origin=False)
                                        if hasattr(actor, "shift"):
                                            actor.shift(*spool_origin)
                                        if hasattr(actor, "rotate_z"):
                                            actor.rotate_z(-90, around=spool_origin)
                                elif hasattr(mesh, "scale"):
                                    mesh.scale(1e-3, origin=False)
                                    if hasattr(mesh, "shift"):
                                        mesh.shift(*spool_origin)
                                    if hasattr(mesh, "rotate_z"):
                                        mesh.rotate_z(-90, around=spool_origin)

                                if getattr(self, '_loaded_spool_mesh', None) is not None:
                                    self.plotter.remove(self._loaded_spool_mesh)
                                
                                self.plotter.add(mesh)
                                self._loaded_spool_mesh = mesh
                                self._loaded_spool_x_flipped = False
                                self._spool_r_angle_deg = 0.0
                                # 현재 FK 기준 척 조인트 위치를 추적 기준점으로 저장
                                self._spool_chuck_pos = np.array(chuck_center)
                                for model in getattr(self, '_robot_models', []):
                                    pos = model.get_link_world_pos("m_column_passive_r")
                                    if pos is not None:
                                        self._spool_chuck_pos = pos
                                    r_rad = getattr(model, '_joint_cfg', {}).get("f_column_z_to_f_column_r")
                                    if r_rad is not None:
                                        self._spool_r_angle_deg = float(np.rad2deg(r_rad))
                                    if pos is not None:
                                        break
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
                elif command == "flip_spool_x":
                    spool = getattr(self, '_loaded_spool_mesh', None)
                    if spool is None or (isinstance(spool, (list, tuple)) and len(spool) == 0):
                        self.__console.warning("Cannot flip spool X direction: no spool loaded")
                        return True

                    actors = spool if isinstance(spool, (list, tuple)) else [spool]

                    # 스풀의 현재 bounding box 중심을 mirror origin으로 사용
                    bounds_list = [a.bounds() for a in actors if hasattr(a, "bounds")]
                    if bounds_list:
                        x_min = min(b[0] for b in bounds_list)
                        x_max = max(b[1] for b in bounds_list)
                        y_min = min(b[2] for b in bounds_list)
                        y_max = max(b[3] for b in bounds_list)
                        z_min = min(b[4] for b in bounds_list)
                        z_max = max(b[5] for b in bounds_list)
                        center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
                    else:
                        center = [0, 0, 0]

                    for actor in actors:
                        if hasattr(actor, "mirror"):
                            actor.mirror(axis="x", origin=center)

                    self._loaded_spool_x_flipped = not getattr(self, '_loaded_spool_x_flipped', False)
                    self.plotter.render()
                    self.__console.info(
                        f"Flipped spool X direction: {self._loaded_spool_x_flipped}")
                elif command == "move_spool":
                    spool = getattr(self, '_loaded_spool_mesh', None)
                    if spool is None or (isinstance(spool, (list, tuple)) and len(spool) == 0):
                        self.__console.warning("Cannot move spool position: no spool loaded")
                        return True
                    if getattr(self, '_spool_fixed', False):
                        # 고정 상태에서는 수동 이동 무시 (UI에서도 잠겨 있음)
                        self.__console.warning("move_spool 무시: 스풀이 고정되어 있음")
                        return True

                    target_pos = np.array([
                        float(request_data.get("x", 0.0)),
                        float(request_data.get("y", 0.0)),
                        float(request_data.get("z", 0.0)),
                    ], dtype=float)
                    target_x_rot = float(request_data.get("x_rotation", 0.0))
                    target_z_rot = float(request_data.get("z_rotation", 0.0))
                    current_pos = getattr(self, '_spool_manual_pos', None)
                    if current_pos is None:
                        current_pos = np.array(getattr(self, '_loaded_spool_origin', [0.0, 0.0, 0.0]), dtype=float)
                    current_x_rot = getattr(self, '_spool_manual_x_rot_deg', 0.0)
                    current_z_rot = getattr(self, '_spool_manual_z_rot_deg', 0.0)

                    delta = target_pos - current_pos
                    delta_x_rot = target_x_rot - current_x_rot
                    delta_z_rot = target_z_rot - current_z_rot
                    actors = spool if isinstance(spool, (list, tuple)) else [spool]
                    for actor in actors:
                        if hasattr(actor, "shift"):
                            actor.shift(delta.tolist())
                        if hasattr(actor, "rotate_x") and abs(delta_x_rot) > 1e-9:
                            actor.rotate_x(delta_x_rot, around=target_pos.tolist())
                        if hasattr(actor, "rotate_z") and abs(delta_z_rot) > 1e-9:
                            actor.rotate_z(delta_z_rot, around=target_pos.tolist())

                    self._spool_manual_pos = target_pos
                    self._spool_manual_x_rot_deg = target_x_rot
                    self._spool_manual_z_rot_deg = target_z_rot
                    self._loaded_spool_origin = target_pos.tolist()

                    # 프레임 추적: 동일한 delta 변환을 _spool_world_T에도 합성 (fix 시 offset 정확성)
                    if getattr(self, '_spool_world_T', None) is not None:
                        Tsh = self._transl(delta)
                        Tt = self._transl(target_pos)
                        A = Tt @ self._rotx(delta_x_rot) @ self._transl(-target_pos)
                        B = Tt @ self._rotz(delta_z_rot) @ self._transl(-target_pos)
                        self._spool_world_T = B @ A @ Tsh @ self._spool_world_T

                    self.plotter.render()
                    self._send_spool_pose_update(request_data.get("_identity"))
                    self.__console.info(
                        f"Moved spool pose to {target_pos.tolist()}, "
                        f"x_rotation={target_x_rot}, z_rotation={target_z_rot}")
                elif command == "move_positioner":
                    import math
                    axis = request_data.get("axis")
                    position = float(request_data.get("position", 0.0))
                    velocity = float(request_data.get("velocity", 0.0))

                    for model in getattr(self, '_robot_models', []):
                        joint_map = model._urdf._joint_map if model._urdf else {}
                        if axis == "x" and "base_to_m_column" in joint_map:
                            model.set_joint("base_to_m_column", -position)
                        elif axis == "z" and "base_to_f_column_z" in joint_map:
                            model.set_joint("base_to_f_column_z", position)
                            model.set_joint("m_column_to_m_column_z", position)
                        elif axis == "r" and "f_column_z_to_f_column_r" in joint_map:
                            model.set_joint("f_column_z_to_f_column_r", math.radians(position))
                        elif axis == "clamp" and "f_column_r_to_f_column_passive_clamp" in joint_map:
                            # prismatic y-axis, range -0.9~0; UI value 0~0.9 → joint = -position
                            model.set_joint("f_column_r_to_f_column_passive_clamp", -position)
                        else:
                            continue
                        model.update_fk()

                    # 스풀이 chuck(column m)에 고정되어 있으면 저장된 offset(R,t)으로
                    # spool_world = T_chuck @ offset 재계산해 따라가게 한다. 해제 시엔 정지.
                    if (getattr(self, '_spool_fixed', False)
                            and getattr(self, '_spool_offset_T', None) is not None
                            and getattr(self, '_spool_local_verts', None) is not None):
                        Tc = self._chuck_world_T()
                        if Tc is not None:
                            self._spool_world_T = Tc @ self._spool_offset_T
                            self._render_spool_world_T()
                            self._send_spool_pose_update(request_data.get("_identity"))

                    self.plotter.render()
                    self.__console.info(f"Positioner {axis} moved to {position} (vel={velocity})")
                elif command == "fix_spool":
                    self._fix_spool(request_data)
                elif command == "unfix_spool":
                    self._unfix_spool(request_data)
                elif command == "set_spool_offset":
                    self._set_spool_offset(request_data)
                elif command == "filter_spool":
                    self._filter_loaded_spool(request_data)
                elif command == "reconstruct_mesh":
                    self._reconstruct_loaded_spool_mesh(request_data)
                elif command == "save_spool":
                    self._save_loaded_spool(request_data)
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
