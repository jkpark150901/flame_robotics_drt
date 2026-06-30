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

        # 매니퓰레이터 조인트 애니메이션(보간 이동) 상태
        # 각 항목: {"model", "joint", "target", "speed"}  speed 단위/프레임당 = unit/s
        self._joint_animations = []
        self._last_anim_time = None

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

        # 3. Step manipulator joint animations (보간 이동)
        now = time.time()
        dt = 0.0 if self._last_anim_time is None else (now - self._last_anim_time)
        self._last_anim_time = now
        if self._joint_animations and dt > 0:
            self._step_joint_animations(min(dt, 0.1))   # 큰 dt는 클램프

        return True

    def _find_robot(self, name):
        for m in getattr(self, '_robot_models', []):
            if getattr(m, 'name', None) == name:
                return m
        return None

    def _step_joint_animations(self, dt):
        """활성 조인트 애니메이션을 사다리꼴 속도 프로파일로 한 스텝 진행.
        가속(accel)으로 max_speed까지 올린 뒤 순항, target 도달 전 감속해 정지.
        """
        still = []
        for anim in self._joint_animations:
            model = anim["model"]; jn = anim["joint"]
            tgt = float(anim["target"])
            vmax = max(float(anim["speed"]), 1e-6)
            accel = max(float(anim["accel"]), 1e-6)
            cur = float(model._joint_cfg.get(jn, 0.0))
            vel = float(anim.get("vel", 0.0))

            d_rem = tgt - cur
            dist = abs(d_rem)
            direction = np.sign(d_rem) if d_rem != 0 else 0.0

            # 정지 임박: 남은 거리·속도가 충분히 작으면 스냅
            if dist <= 1e-6 and vel <= accel * dt:
                model.set_joint(jn, tgt); model.update_fk()
                continue

            # 감속에 필요한 거리 = v² / (2a). 그보다 가까우면 감속, 아니면 가속/순항
            stop_dist = (vel * vel) / (2.0 * accel)
            if dist <= stop_dist:
                vel = max(0.0, vel - accel * dt)      # 감속
            else:
                vel = min(vmax, vel + accel * dt)     # 가속 후 vmax 순항

            new_cur = cur + direction * vel * dt
            # target을 지나치면 스냅하고 종료
            if (tgt - new_cur) * direction <= 0:
                model.set_joint(jn, tgt); model.update_fk()
                continue

            anim["vel"] = vel
            model.set_joint(jn, new_cur); model.update_fk()
            still.append(anim)
        self._joint_animations = still

    def _set_joint_animation(self, robot_name, joint_name, target, speed, accel=None):
        """해당 로봇/조인트의 기존 애니메이션을 교체하고 사다리꼴 프로파일로 이동 시작.
        accel 미지정 시 speed의 2배(약 0.5s 가속)로 기본 설정.
        """
        model = self._find_robot(robot_name)
        if model is None or model._urdf is None:
            self.__console.warning(f"move_manipulator: 로봇 없음 '{robot_name}'")
            return
        if joint_name not in model._urdf._joint_map:
            self.__console.warning(f"move_manipulator: 조인트 없음 '{joint_name}'")
            return
        spd = float(speed)
        acc = float(accel) if accel is not None else max(spd * 2.0, 1e-6)
        # 같은 (robot, joint)의 현재 속도는 이어받아 부드럽게 재타게팅
        prev_vel = 0.0
        for a in self._joint_animations:
            if a["model"] is model and a["joint"] == joint_name:
                prev_vel = a.get("vel", 0.0)
        self._joint_animations = [
            a for a in self._joint_animations
            if not (a["model"] is model and a["joint"] == joint_name)
        ]
        self._joint_animations.append({
            "model": model, "joint": joint_name,
            "target": float(target), "speed": spd, "accel": acc, "vel": prev_vel,
        })
        self.__console.info(
            f"move_manipulator: {robot_name}.{joint_name} → {target} (vmax={spd}, accel={acc})")

    def _stop_joint_animation(self, robot_name, joint_name=None):
        """해당 로봇(또는 특정 조인트)의 애니메이션을 즉시 중지."""
        model = self._find_robot(robot_name)
        self._joint_animations = [
            a for a in self._joint_animations
            if not (a["model"] is model and (joint_name is None or a["joint"] == joint_name))
        ]
        self.__console.info(f"stop_manipulator: {robot_name} {joint_name or '(all)'}")

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
        # 스풀 포즈 = chuck 기준 오프셋 (포지셔너가 움직여도 값은 그대로)
        x, y, z = getattr(self, '_spool_offset_xyz', (0.0, 0.0, 0.0))
        return {
            "x": float(x), "y": float(y), "z": float(z),
            "x_rotation": float(getattr(self, '_spool_offset_xrot', 0.0)),
            "z_rotation": float(getattr(self, '_spool_offset_zrot', 0.0)),
        }

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
        new_pts = np.asarray(new_pts, dtype=np.float64)
        new_actor = vedo.Points(new_pts)
        self.plotter.add(new_actor)
        self._loaded_spool_mesh = new_actor
        # 오프셋 모델 일관성: world 점을 현재 chuck@offset 기준 local로 환산
        Tc = self._chuck_world_T()
        if Tc is not None and getattr(self, '_spool_local_verts', None) is not None:
            Tinv = np.linalg.inv(Tc @ self._spool_offset_T())
            self._spool_local_verts = (Tinv[:3, :3] @ new_pts.T).T + Tinv[:3, 3]
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

            # 기존 pcd 스풀 + 이전 재건 메시 제거
            old_pcd = getattr(self, '_loaded_spool_mesh', None)
            if old_pcd is not None:
                self.plotter.remove(old_pcd)
            old_recon = getattr(self, '_spool_recon_mesh', None)
            if old_recon is not None and old_recon is not old_pcd:
                self.plotter.remove(old_recon)

            self.plotter.add(vmesh)
            # 재건 메시를 새 스풀로 삼아 오프셋 모델에 연결 → 포지셔너 추종(같이 이동)
            self._loaded_spool_mesh = vmesh
            self._spool_recon_mesh = vmesh
            self._spool_recon_mesh_o3d = mesh_o3d  # 저장용 (면 정보)
            T = self._spool_world_T if getattr(self, '_spool_world_T', None) is not None else np.eye(4)
            Tinv = np.linalg.inv(T)
            # verts(월드) → local 로 환산해 world = T @ local 유지 (현재 위치 보존 + 추종 가능)
            self._spool_local_verts = (Tinv[:3, :3] @ verts.T).T + Tinv[:3, 3]
            self._spool_world_T = T
            Tc = self._chuck_world_T()
            if Tc is not None:
                self._chuck_prev_T = Tc
            self.plotter.render()
            self.__console.info(f"reconstruct_mesh: 정점 {len(verts)}, 면 {len(faces)} (pcd 제거, 메시가 스풀로 전환)")
        except Exception as e:
            self.__console.error(f"reconstruct_mesh 실패: {e}")

    def _save_loaded_spool(self, request_data):
        """현재 결과를 저장. 재건 메시가 있으면 메시를, 없으면 점군을 저장."""
        path = request_data.get("path")
        if not path:
            return
        try:
            recon_o3d = getattr(self, '_spool_recon_mesh_o3d', None)
            recon = getattr(self, '_spool_recon_mesh', None)
            if recon is not None and recon_o3d is not None:
                # 현재(추종 반영된) 메시 정점으로 갱신해 저장 (면 정보는 유지)
                m = _o3d.geometry.TriangleMesh()
                m.vertices = _o3d.utility.Vector3dVector(np.asarray(recon.vertices))
                m.triangles = recon_o3d.triangles
                m.compute_vertex_normals()
                _o3d.io.write_triangle_mesh(path, m)
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

    @staticmethod
    def _rot_about_axis(axis, center, deg):
        """center를 지나는 axis 둘레로 deg도 회전하는 4x4 (월드)."""
        axis = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return np.eye(4)
        x, y, z = axis / n
        th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th); C = 1 - c
        R = np.array([
            [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
        ])
        center = np.asarray(center, dtype=float)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = center - R @ center
        return T

    def _chuck_world_T(self):
        """column m chuck joint(m_column_passive_r) 링크의 4x4 월드 변환."""
        for model in getattr(self, '_robot_models', []):
            if hasattr(model, 'get_link_world_T'):
                T = model.get_link_world_T(self.CHUCK_LINK_NAME)
                if T is not None:
                    return np.asarray(T, dtype=float)
        return None

    def _spool_offset_T(self):
        """UI(=chuck 기준) 오프셋 포즈를 4x4 변환으로. spool_world = T_chuck @ T_offset @ local"""
        x, y, z = getattr(self, '_spool_offset_xyz', (0.0, 0.0, 0.0))
        xrot = getattr(self, '_spool_offset_xrot', 0.0)
        zrot = getattr(self, '_spool_offset_zrot', 0.0)
        return self._transl([x, y, z]) @ self._rotz(zrot) @ self._rotx(xrot)

    def _apply_spool_world_T(self):
        """현재 _spool_world_T 로 스풀 actor 정점 갱신 (world = T @ local)."""
        local = getattr(self, '_spool_local_verts', None)
        spool = getattr(self, '_loaded_spool_mesh', None)
        T = getattr(self, '_spool_world_T', None)
        if local is None or spool is None or T is None:
            return False
        world = (T[:3, :3] @ local.T).T + T[:3, 3]
        actors = spool if isinstance(spool, (list, tuple)) else [spool]
        if actors and hasattr(actors[0], 'vertices'):
            actors[0].vertices = world
            return True
        return False

    def _render_spool_offset(self):
        """수동 배치: 현재 chuck 기준으로 스풀을 절대 배치 (spool_world = T_chuck @ T_offset)."""
        local = getattr(self, '_spool_local_verts', None)
        spool = getattr(self, '_loaded_spool_mesh', None)
        if local is None or spool is None:
            return False
        Tc = self._chuck_world_T()
        if Tc is None:
            return False
        self._spool_world_T = Tc @ self._spool_offset_T()
        self._chuck_prev_T = Tc
        return self._apply_spool_world_T()

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
                                # 스풀 위치는 chuck 조인트(m_column_passive_r)를 원점으로 본다.
                                # spool_world = T_chuck @ T_offset @ local
                                #  - local: 점군을 centroid로 중심화 + 기본정렬(chuck 기준 상수)
                                #    → 포지셔너 위치와 무관하게 reload 시 항상 현재 chuck 기준으로 배치
                                #  - T_offset: UI(chuck 기준) 위치/회전, 처음엔 0
                                _is_pcd = _pl.Path(path).suffix.lower() == ".pcd"
                                _default_x = -0.442  # 척 길이만큼 x로 (chuck 기준)

                                # 기존에 로드된 스풀/재건 메시 모두 제거 (새 파이프로 교체)
                                _old_sp = getattr(self, '_loaded_spool_mesh', None)
                                if _old_sp is not None:
                                    self.plotter.remove(_old_sp)
                                    self._loaded_spool_mesh = None
                                _old_rc = getattr(self, '_spool_recon_mesh', None)
                                if _old_rc is not None:
                                    if _old_rc is not _old_sp:
                                        self.plotter.remove(_old_rc)
                                    self._spool_recon_mesh = None
                                    self._spool_recon_mesh_o3d = None

                                self._spool_offset_xyz = [0.0, 0.0, 0.0]
                                self._spool_offset_xrot = 0.0
                                self._spool_offset_zrot = 0.0
                                self._spool_fix_r = False
                                self._positioner_r_deg = 0.0
                                self._spool_world_T = None
                                self._chuck_prev_T = None
                                self._loaded_spool_x_flipped = False

                                if _is_pcd:
                                    scaled = _pts * 1e-3
                                    centroid = scaled.mean(axis=0)
                                    Rz = self._rotz(-90)[:3, :3]
                                    # centroid 중심화 → -90도 정렬 → chuck 기준 x 오프셋(상수)
                                    self._spool_local_verts = (
                                        (Rz @ (scaled - centroid).T).T + np.array([_default_x, 0.0, 0.0]))
                                    self.plotter.add(mesh)
                                    self._loaded_spool_mesh = mesh
                                    self._render_spool_offset()
                                else:
                                    # PLY 등(저장된 메시/점군): 이미 월드(미터) 좌표이므로 스케일 없이 그대로 표시
                                    self._spool_local_verts = None
                                    self.plotter.add(mesh)
                                    self._loaded_spool_mesh = mesh

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
                    # 스풀 위치/회전을 chuck 기준 오프셋으로 설정 (x,y,z,x_rot,z_rot)
                    spool = getattr(self, '_loaded_spool_mesh', None)
                    if spool is None or getattr(self, '_spool_local_verts', None) is None:
                        self.__console.warning("move_spool: 로드된 스풀(PCD)이 없습니다")
                        return True
                    self._spool_offset_xyz = [
                        float(request_data.get("x", 0.0)),
                        float(request_data.get("y", 0.0)),
                        float(request_data.get("z", 0.0)),
                    ]
                    self._spool_offset_xrot = float(request_data.get("x_rotation", 0.0))
                    self._spool_offset_zrot = float(request_data.get("z_rotation", 0.0))
                    self._render_spool_offset()
                    self.plotter.render()
                    self.__console.info(
                        f"Spool offset set to xyz={self._spool_offset_xyz}, "
                        f"x_rot={self._spool_offset_xrot}, z_rot={self._spool_offset_zrot}")
                elif command == "move_positioner":
                    import math
                    axis = request_data.get("axis")
                    position = float(request_data.get("position", 0.0))
                    velocity = float(request_data.get("velocity", 0.0))
                    fix_m_column_z = bool(request_data.get("fix_m_column_z", False))
                    fix_f_column_r = bool(request_data.get("fix_f_column_r", False))
                    self._spool_fix_r = fix_f_column_r

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

                    # 스풀 추종은 "고정된 축"에 대해서만. (체크 안 하면 안 따라감)
                    Tc_now = self._chuck_world_T()
                    has_frame = (getattr(self, '_spool_world_T', None) is not None
                                 and getattr(self, '_spool_local_verts', None) is not None)
                    if has_frame and Tc_now is not None:
                        if axis in ("x", "z") and fix_m_column_z and getattr(self, '_chuck_prev_T', None) is not None:
                            # column m 고정: chuck 병진량만큼 스풀 평행이동
                            dt = Tc_now[:3, 3] - self._chuck_prev_T[:3, 3]
                            T = np.eye(4); T[:3, 3] = dt
                            self._spool_world_T = T @ self._spool_world_T
                            self._apply_spool_world_T()
                        elif axis == "r" and fix_f_column_r:
                            # column r 고정: chuck joint 중심·축(chuck x축) 기준으로 스풀 회전
                            r_prev = getattr(self, '_positioner_r_deg', 0.0)
                            delta_r = position - r_prev
                            center = Tc_now[:3, 3]
                            axis_w = Tc_now[:3, :3] @ np.array([1.0, 0.0, 0.0])
                            Rm = self._rot_about_axis(axis_w, center, delta_r)
                            self._spool_world_T = Rm @ self._spool_world_T
                            self._apply_spool_world_T()
                    if Tc_now is not None:
                        self._chuck_prev_T = Tc_now
                    if axis == "r":
                        self._positioner_r_deg = position

                    self.plotter.render()
                    self.__console.info(f"Positioner {axis} moved to {position} (vel={velocity})")
                elif command == "move_manipulator":
                    self._set_joint_animation(
                        request_data.get("robot"),
                        request_data.get("joint"),
                        request_data.get("target", 0.0),
                        request_data.get("speed", 1.0),
                        request_data.get("accel"))
                elif command == "stop_manipulator":
                    self._stop_joint_animation(
                        request_data.get("robot"),
                        request_data.get("joint"))
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
