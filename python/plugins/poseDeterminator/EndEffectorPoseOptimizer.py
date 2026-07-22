from dataclasses import dataclass, fields
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud  # type: ignore
import numpy as np
from numpy.typing import NDArray
try:
    from .CylinderFitting import fit_cylinder
except ImportError:
    from CylinderFitting import fit_cylinder
from urdf_parser_py.urdf import URDF
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import json
import copy
import math
from typing import Any
from util.logger.console import ConsoleLogger


@dataclass(frozen=True)
class Pose6:
    """TCP pose as [x, y, z, roll, pitch, yaw]."""

    values: tuple[float, float, float, float, float, float]

    @classmethod
    def from_list(cls, values: list[float] | tuple[float, ...]) -> "Pose6":
        if len(values) != 6:
            raise ValueError(f"Pose6 requires 6 values, got {len(values)}")
        return cls(tuple(float(v) for v in values))  # type: ignore[arg-type]

    def to_list(self) -> list[float]:
        return list(self.values)
    

@dataclass(frozen=True)
class InspectionPoseSlot:
    """A single inspection pose that contains DDA and one or more RT candidates."""

    deg             : int
    rt_pose         : Pose6
    dda_pose        : Pose6

    @property
    def selected_rt_pose(self) -> Pose6:
        for candidate in self.rt_candidates:
            if candidate.name == self.rt_selection.selected_rt:
                return candidate.pose
        raise KeyError(f"selected RT pose not found: {self.rt_selection.selected_rt}")

class EndEffectorPoseOptimizer:
    _scan_data: PointCloud

    # DDA 충돌 모델 정보
    __dda_mesh: o3d.geometry.TriangleMesh
    __dda_invers_transform_mat: np.ndarray

    # RT source 충돌 모델 정보
    __rt_mesh: o3d.geometry.TriangleMesh
    __rt_invers_transform_mat: np.ndarray

    # 현재 계산된 배관 프로파일
    __pipe_direction: np.ndarray
    __pipe_center: np.ndarray
    __pipe_radius: float

    # 디버깅 정보
    __is_debug_mode: bool
    debuging_info: dict[str, Any]

    def __init__(
        self,
        debug_mode: bool = False,
        log_path: str | Path | None = None,
        log_dir: str | Path | None = None,
        log_level: str | int = "DEBUG",
        console_level: str | int | None = None,
        file_level: str | int | None = None,
        logger_name: str = "flame_robotics",
        force_logger_config: bool | None = None,
    ):
        """엔드이펙터 자세 후보 계산기를 초기화한다.

        Args:
            debug_mode: 중간 계산 결과를 `debuging_info`에 저장할지 여부.
            log_path: 로그 파일 경로. 지정하면 ConsoleLogger 파일 핸들러를 설정한다.
            log_dir: 기본 로그 파일을 생성할 디렉터리. `log_path`가 우선한다.
            log_level: logger 기본 level.
            console_level: 터미널 출력 level. None이면 `log_level`을 따른다.
            file_level: 파일 기록 level. None이면 `log_level`을 따른다.
            logger_name: Python logger 이름.
            force_logger_config: 기존 ConsoleLogger handler를 재설정할지 여부.
                None이면 `log_path` 또는 `log_dir`가 있을 때만 재설정한다.
        """
        configure_file_logger = log_path is not None or log_dir is not None
        if force_logger_config is None:
            force_logger_config = configure_file_logger
        if configure_file_logger or force_logger_config:
            self.__console = ConsoleLogger.get_logger(
                level=log_level,
                console_level=console_level,
                file_level=file_level,
                log_path=log_path,
                log_dir=log_dir,
                name=logger_name,
                force=bool(force_logger_config),
            )
        else:
            self.__console = ConsoleLogger.get_logger(level=log_level)
        self.__is_debug_mode = debug_mode
        self.debuging_info = {}
        self.__dda_pipe_facing_axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
        self.__dda_pipe_parallel_axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
        self.__rt_pipe_facing_axis = np.asarray([0.0, -1.0, 0.0], dtype=float)
        self.__collision_checker = None

    @staticmethod
    def __normalized_config_axis(
        axis: tuple[float, float, float] | list[float] | np.ndarray,
        name: str,
    ) -> np.ndarray:
        axis_arr = np.asarray(axis, dtype=float).reshape(-1)
        if axis_arr.shape[0] != 3:
            raise ValueError(f"{name} must have 3 values, got shape={axis_arr.shape}")
        axis_norm = float(np.linalg.norm(axis_arr))
        if axis_norm < 1e-9:
            raise ValueError(f"{name} must be non-zero")
        return axis_arr / axis_norm

    def set_dda_pipe_facing_axis(
        self,
        axis: tuple[float, float, float] | list[float] | np.ndarray,
        pipe_parallel_axis: tuple[float, float, float] | list[float] | np.ndarray | None = None,
    ):
        self.__dda_pipe_facing_axis = self.__normalized_config_axis(axis, "dda pipe facing axis")
        if pipe_parallel_axis is not None:
            parallel_axis = self.__normalized_config_axis(pipe_parallel_axis, "dda pipe parallel axis")
            if abs(float(np.dot(self.__dda_pipe_facing_axis, parallel_axis))) > 0.98:
                raise ValueError("dda pipe facing axis and pipe parallel axis must not be parallel")
            self.__dda_pipe_parallel_axis = parallel_axis

    def set_rt_pipe_facing_axis(self, axis: tuple[float, float, float] | list[float] | np.ndarray):
        self.__rt_pipe_facing_axis = self.__normalized_config_axis(axis, "rt pipe facing axis")

    def set_collision_checker(self, checker):
        """Register an external mesh-vs-point-cloud collision checker."""
        self.__collision_checker = checker

    def set_DDA_geometry(
        self,
        link_mesh: o3d.geometry.TriangleMesh,
        tcp_to_link_transform: np.ndarray,
    ):
        self.__dda_mesh = copy.deepcopy(link_mesh)
        self.__dda_invers_transform_mat = self.__offset_to_transform(tcp_to_link_transform)

    def set_RT_geometry(
        self,
        link_mesh: o3d.geometry.TriangleMesh,
        tcp_to_link_transform: np.ndarray,
    ):
        self.__rt_mesh = copy.deepcopy(link_mesh)
        self.__rt_invers_transform_mat = self.__offset_to_transform(tcp_to_link_transform)

    def load_scan_data(
        self,
        file_path: str,
        scale: float = 1.0,
    ):
        self._scan_data = o3d.io.read_point_cloud(file_path)  # type: ignore

        self._scan_data.scale(scale, np.asarray([0.0, 0.0, 0.0]))  # type: ignore

    def calculate_pipe_end_profiles(
        self,
        file_path: str,
        scale: float = 1.0,
        voxel_size: float | None = None,
        max_points: int = 25000,
        max_segments: int = 6,
        min_segment_points: int | None = None,
        ransac_iterations: int = 250,
        sample_size: int = 128,
        distance_threshold: float | None = None,
        connection_threshold: float | None = None,
        profile_sample_count: int = 64,
        include_segment_points: bool = False,
    ) -> dict[str, Any]:
        """Calculate terminal end positions and circular profiles of a pipe.

        Args:
            file_path: PCD/PLY path for a single pipe.
            scale: Scale applied after loading. Use 0.001 for mm-to-m data.
            voxel_size: Optional Open3D voxel downsample size.
            max_points: Maximum number of points used for fitting.
            max_segments: Maximum number of straight cylinder sections to extract.
            min_segment_points: Minimum inlier points required for a section.
            ransac_iterations: Number of cylinder RANSAC trials per straight section.
            sample_size: Number of points sampled in each RANSAC trial.
            distance_threshold: Radial inlier threshold. If None, it is inferred
                from the point-cloud bounding-box diagonal.
            connection_threshold: Max distance for two section endpoints to be
                considered a joint. If None, it is inferred from radius.
            profile_sample_count: Number of points sampled on each end circle.
            include_segment_points: Include fitted segment point clouds,
                fit_cylinder input point clouds, and unassigned fit points in
                the returned dict for visualization/debugging.
        """
        from PipeEndProfileAnalyzer import analyze_pipe_end_profiles

        return analyze_pipe_end_profiles(
            file_path=file_path,
            scale=scale,
            voxel_size=voxel_size,
            max_points=max_points,
            max_segments=max_segments,
            min_segment_points=min_segment_points,
            ransac_iterations=ransac_iterations,
            sample_size=sample_size,
            distance_threshold=distance_threshold,
            connection_threshold=connection_threshold,
            profile_sample_count=profile_sample_count,
            include_segment_points=include_segment_points,
        )

    def calculate_l_pipe_end_profiles(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Backward-compatible alias for older L-pipe callers."""
        return self.calculate_pipe_end_profiles(*args, **kwargs)

    def load_DDA_from_urdf(
        self,
        file_path: str,
        end_link_name: str = "dda_link_end",
        tcp_joint_name: str | tuple[str, ...] = "dda_joint_tcp",
        pose_to_link_offset: np.ndarray | dict[str, Any] | None = None,
    ):
        self.__dda_mesh, self.__dda_invers_transform_mat = self.__extract_tcp_and_end(
            file_path,
            end_link_name,
            tcp_joint_name,
            pose_to_link_offset,
        )

    def load_RT_from_urdf(
        self,
        file_path: str,
        end_link_name: str = "rt_link_end",
        tcp_joint_name: str | tuple[str, ...] = "rt_joint_end",
        pose_to_link_offset: np.ndarray | dict[str, Any] | None = None,
    ):
        self.__rt_mesh, self.__rt_invers_transform_mat = self.__extract_tcp_and_end(
            file_path,
            end_link_name,
            tcp_joint_name,
            pose_to_link_offset,
        )

    def __extract_tcp_and_end(
        self,
        file_path: str,
        end_link_name: str,
        tcp_joint_name: str | tuple[str, ...],
        pose_to_link_offset: np.ndarray | dict[str, Any] | None = None,
    ):
        # URDF 파일 로드
        urdf: URDF = URDF.from_xml_file(file_path)

        # 엔드이펙터 collision mesh 경로 추출
        end_geometry_file_path = urdf.link_map[end_link_name].collision.geometry.filename
        end_geometry_file_path = Path(str(end_geometry_file_path).replace("file://", ""))
        if not end_geometry_file_path.is_absolute():
            end_geometry_file_path = Path(file_path).resolve().parent / end_geometry_file_path
        end_geometry_file_path = end_geometry_file_path.resolve()

        self.__console.debug(f"End-effector collision mesh path: {end_geometry_file_path}")

        link_mesh = o3d.io.read_triangle_mesh(end_geometry_file_path)

        end_geomtry_scale = urdf.link_map[end_link_name].collision.geometry.scale
        if isinstance(end_geomtry_scale, list):
            end_geomtry_scale = float(end_geomtry_scale[0])
        elif isinstance(end_geomtry_scale, (int, float)):
            end_geomtry_scale = float(end_geomtry_scale)
        else:
            raise ValueError("End-effector collision mesh scale must be a number or a numeric list.")

        link_mesh = link_mesh.scale(end_geomtry_scale, np.zeros(3, dtype=np.float64))  # type: ignore

        end_pose_xyz = urdf.link_map[end_link_name].collision.origin.xyz
        end_pose_rpy = urdf.link_map[end_link_name].collision.origin.rpy
        T = np.eye(4)
        T[:3, :3] = R.from_euler("xyz", end_pose_rpy).as_matrix()
        T[:3, 3] = end_pose_xyz
        link_mesh = link_mesh.transform(T)  # type: ignore

        if pose_to_link_offset is not None:
            return link_mesh, self.__offset_to_transform(pose_to_link_offset)

        # URDF link tree 기준 target link에서 TCP joint까지의 상대 변환 계산
        tcp_joint_names = (tcp_joint_name,) if isinstance(tcp_joint_name, str) else tcp_joint_name
        tcp_joint = None
        for candidate_name in tcp_joint_names:
            tcp_joint = urdf.joint_map.get(candidate_name)
            if tcp_joint is not None:
                break
        if tcp_joint is None:
            raise KeyError(f"TCP joint not found. candidates={tcp_joint_names}")

        end_to_tcp_relative_pose_xyz = tcp_joint.origin.xyz
        end_to_tcp_relative_pose_rpy = tcp_joint.origin.rpy

        direct_end_to_tcp_T = np.eye(4)
        direct_end_to_tcp_T[:3, :3] = R.from_euler("xyz", end_to_tcp_relative_pose_rpy).as_matrix()
        direct_end_to_tcp_T[:3, 3] = end_to_tcp_relative_pose_xyz
        end_to_tcp_relative_pose_T = self.__relative_link_transform(
            urdf,
            end_link_name,
            tcp_joint.child,
            fallback_T=direct_end_to_tcp_T,
        )
        tcp_to_origin_mat = np.linalg.inv(end_to_tcp_relative_pose_T)

        # ----------------------------------------------------------------------
        return link_mesh, tcp_to_origin_mat

    @staticmethod
    def __offset_to_transform(offset: np.ndarray | dict[str, Any]) -> np.ndarray:
        if isinstance(offset, dict):
            if "matrix" in offset:
                T = np.asarray(offset["matrix"], dtype=float)
            else:
                xyz = np.asarray(offset.get("xyz", [0.0, 0.0, 0.0]), dtype=float)
                rpy = np.asarray(offset.get("rpy", [0.0, 0.0, 0.0]), dtype=float)
                T = np.eye(4)
                T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
                T[:3, 3] = xyz
        else:
            T = np.asarray(offset, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"pose_to_link_offset must be 4x4, got shape={T.shape}")
        return T.astype(float, copy=True)

    @staticmethod
    def __joint_origin_T(joint) -> np.ndarray:
        T = np.eye(4)
        origin = getattr(joint, "origin", None)
        if origin is not None:
            T[:3, :3] = R.from_euler("xyz", origin.rpy).as_matrix()
            T[:3, 3] = origin.xyz
        return T

    @classmethod
    def __relative_link_transform(
        cls,
        urdf: URDF,
        source_link_name: str,
        target_link_name: str,
        fallback_T: np.ndarray,
    ) -> np.ndarray:
        child_to_joint = {joint.child: joint for joint in urdf.joints}
        cache: dict[str, np.ndarray] = {}

        def root_to_link(link_name: str) -> np.ndarray:
            if link_name in cache:
                return cache[link_name]
            joint = child_to_joint.get(link_name)
            if joint is None:
                cache[link_name] = np.eye(4)
                return cache[link_name]
            T = root_to_link(joint.parent) @ cls.__joint_origin_T(joint)
            cache[link_name] = T
            return T

        try:
            return np.linalg.inv(root_to_link(source_link_name)) @ root_to_link(target_link_name)
        except Exception:
            return fallback_T

    def calculate_DDA_pose_for_detecting_welding_point(
        self,
        target_point: tuple[float, float, float],  # x,y,z
        num_candidates: int = 8,
        distance: float = 0.3,
        distance_reference_mesh: o3d.geometry.TriangleMesh | None = None,
    ):
        """용접부 탐색용 DDA 자세 후보를 계산한다.

        DDA 후보 조건:
            - 설정된 DDA pipe-facing 축이 배관 중심을 향한다.
            - 설정된 DDA pipe-parallel 축이 배관 중심축과 최대한 평행하다.
            - 배관 표면에서 지정 거리만큼 떨어진다.
            - 배관과 충돌하지 않는 후보만 반환한다.

        Args:
            target_point: 사용자가 선택한 배관 표면점.
            num_candidates: 배관 둘레 방향으로 생성할 후보 개수.
            distance: DDA mesh와 배관 표면 사이 목표 거리.

        Returns:
            `(json_str, filtered_candidates, all_candidates)` 형식의 후보 결과.
        """
        # DDA 후보 생성
        candidate_radius = self.__dda_candidate_centerline_radius(
            distance,
            self.__dda_mesh if distance_reference_mesh is None else distance_reference_mesh,
        )
        dda_tcp_pose_candidates = self.__calculate_dda_pose_candidate(
            np.asarray(target_point),
            candidate_radius,
            num_candidates,
        )

        # mesh 표면과의 목표 거리를 만족하도록 후보 위치 보정
        dda_tcp_pose_candidates = self.__adjust_dda_candidates_for_mesh_surface_distance(
            dda_tcp_pose_candidates,
            distance,
            self.__dda_mesh if distance_reference_mesh is None else distance_reference_mesh,
        )

        mask = []
        for i in range(len(dda_tcp_pose_candidates)):
            is_collision = self.__check_collision(
                self.__dda_mesh,
                dda_tcp_pose_candidates[i],
                self.__dda_invers_transform_mat,
            )
            mask.append(not is_collision)

        dda_pose_candidates_filtered = dda_tcp_pose_candidates[mask]

        # 반환 형식 변환
        # JSON 형태: [ {dda: [x,y,z,r,p,y]}, ... ]
        pose_list = []
        for row in dda_pose_candidates_filtered:
            pose_list.append({"dda": row.tolist()})

        dda_candidates_filtered_json = json.dumps(pose_list)

        return dda_candidates_filtered_json, dda_pose_candidates_filtered, dda_tcp_pose_candidates

    def __rotate_dda_pose_around_pipe_axis(
        self,
        dda_pose: np.ndarray,
        rotation_angle_deg: float = 90.0,
    ) -> np.ndarray:
        """DDA pose를 배관 중심축 기준으로 회전한다.

        Args:
            dda_pose: 원본 DDA pose `[x, y, z, roll, pitch, yaw]`.
            rotation_angle_deg: 배관 중심축 기준 회전 각도.

        Returns:
            회전이 반영된 DDA pose `[x, y, z, roll, pitch, yaw]`.
        """
        pipe_axis_unit = self.__pipe_direction / np.linalg.norm(self.__pipe_direction)

        # DDA 위치를 배관 축 기준 회전 원 위의 점으로 해석한다.
        dda_position = dda_pose[:3]

        vec_to_dda = dda_position - self.__pipe_center
        proj_len = np.dot(vec_to_dda, pipe_axis_unit)
        rotation_center = self.__pipe_center + proj_len * pipe_axis_unit

        radius_vector = dda_position - rotation_center

        # Rodrigues 공식을 사용해 반지름 벡터를 회전한다.
        cos_angle = np.cos(np.radians(rotation_angle_deg))
        sin_angle = np.sin(np.radians(rotation_angle_deg))

        k_cross_v = np.cross(pipe_axis_unit, radius_vector)
        k_dot_v = np.dot(pipe_axis_unit, radius_vector)

        rotated_radius_vector = (
            radius_vector * cos_angle + k_cross_v * sin_angle + pipe_axis_unit * k_dot_v * (1 - cos_angle)
        )

        rotated_position = rotation_center + rotated_radius_vector

        # DDA 자세도 같은 축 회전을 적용한다.
        original_rotation = R.from_euler("xyz", dda_pose[3:])
        axis_rotation = R.from_rotvec(pipe_axis_unit * np.radians(rotation_angle_deg))
        rotated_rotation = axis_rotation * original_rotation
        rotated_rpy = rotated_rotation.as_euler("xyz")

        return np.hstack([rotated_position, rotated_rpy])

    def __calculate_rt_pose_for_angle(
        self,
        dda_tcp_pose: np.ndarray,
        angle_deg: float,
        distance_from_dda_to_rt: float,
    ) -> np.ndarray:
        """DDA TCP pose와 RT 배치 각도로 RT TCP pose를 계산한다.

        `angle_deg`는 DDA mesh/TCP 좌표계의 X축을 world로 변환한 축을 기준으로,
        DDA pipe-facing 방향을 회전시키는 기울기 각도로 사용한다.
        설정된 RT pipe-facing 축은 회전된 방향의 반대편, 즉 배관 쪽을 향한다.
        """
        if self.__is_debug_mode:
            self.__console.debug(f"\n{'=' * 60}")
            self.__console.debug("[DEBUG] __calculate_rt_pose_for_angle")
            self.__console.debug(f"  - dda_tcp_pose: {dda_tcp_pose}")
            self.__console.debug(f"  - angle_deg: {angle_deg}")
            self.__console.debug(f"  - distance_from_dda_to_rt: {distance_from_dda_to_rt}")

        dda_rot_matrix = R.from_euler("xyz", dda_tcp_pose[3:]).as_matrix()
        dda_x_axis = dda_rot_matrix[:, 0]
        dda_y_axis = dda_rot_matrix[:, 1]
        dda_z_axis = dda_rot_matrix[:, 2]
        dda_x_axis_unit = dda_x_axis / np.linalg.norm(dda_x_axis)
        dda_z_axis_unit = dda_z_axis / np.linalg.norm(dda_z_axis)

        dda_pipe_facing_axis = dda_rot_matrix @ self.__dda_pipe_facing_axis
        dda_pipe_facing_axis = dda_pipe_facing_axis / np.linalg.norm(dda_pipe_facing_axis)

        cos_angle = np.cos(np.radians(angle_deg))
        sin_angle = np.sin(np.radians(angle_deg))
        k_cross_v = np.cross(dda_x_axis_unit, dda_pipe_facing_axis)
        k_dot_v = np.dot(dda_x_axis_unit, dda_pipe_facing_axis)
        dda_to_rt_direction = (
            dda_pipe_facing_axis * cos_angle
            + k_cross_v * sin_angle
            + dda_x_axis_unit * k_dot_v * (1.0 - cos_angle)
        )
        dda_to_rt_direction = dda_to_rt_direction / np.linalg.norm(dda_to_rt_direction)

        rt_front_extent = self.__rt_candidate_origin_distance_offset()
        adjusted_distance_from_dda_to_rt = float(distance_from_dda_to_rt + rt_front_extent)
        rt_position = dda_tcp_pose[:3] + dda_to_rt_direction * adjusted_distance_from_dda_to_rt

        rt_rot_matrix = self.__rotation_from_pipe_facing_axis(
            pipe_facing_world=-dda_to_rt_direction,
            world_up_hint=dda_z_axis_unit,
            local_pipe_facing_axis=self.__rt_pipe_facing_axis,
        )
        det = np.linalg.det(rt_rot_matrix)
        if det < 0:
            rt_rot_matrix[:, 0] *= -1.0

        if self.__is_debug_mode:
            rt_x_axis = rt_rot_matrix[:, 0]
            rt_y_axis = rt_rot_matrix[:, 1]
            rt_z_axis = rt_rot_matrix[:, 2]
            self.debuging_info["rt_mesh_front_extent_along_facing_axis"] = float(rt_front_extent)
            self.debuging_info["rt_adjusted_distance_from_dda_to_rt"] = adjusted_distance_from_dda_to_rt
            self.debuging_info["rt_pipe_facing_axis_local"] = self.__rt_pipe_facing_axis.tolist()
            self.debuging_info["rt_angle_axis_world"] = dda_x_axis_unit.tolist()
            self.debuging_info["rt_dda_to_rt_direction"] = dda_to_rt_direction.tolist()
            self.__console.debug(f"  - dda_x_axis: {dda_x_axis}, norm: {np.linalg.norm(dda_x_axis)}")
            self.__console.debug(f"  - dda_y_axis: {dda_y_axis}, norm: {np.linalg.norm(dda_y_axis)}")
            self.__console.debug(f"  - dda_z_axis: {dda_z_axis}, norm: {np.linalg.norm(dda_z_axis)}")
            self.__console.debug(f"  - rt angle axis = dda_x_axis(world): {dda_x_axis_unit}")
            self.__console.debug(f"  - dda_to_rt_direction: {dda_to_rt_direction}, norm: {np.linalg.norm(dda_to_rt_direction)}")
            self.__console.debug(f"  - dda_pipe_facing_axis(world): {dda_pipe_facing_axis}")
            self.__console.debug(f"  - rt_pipe_facing_axis(local): {self.__rt_pipe_facing_axis}")
            self.__console.debug(f"  - rt_x_axis: {rt_x_axis}, norm: {np.linalg.norm(rt_x_axis)}")
            self.__console.debug(f"  - rt_y_axis: {rt_y_axis}, norm: {np.linalg.norm(rt_y_axis)}")
            self.__console.debug(f"  - rt_z_axis: {rt_z_axis}, norm: {np.linalg.norm(rt_z_axis)}")
            self.__console.debug(f"  - dot(rt_x, rt_y): {np.dot(rt_x_axis, rt_y_axis)}")
            self.__console.debug(f"  - dot(rt_x, rt_z): {np.dot(rt_x_axis, rt_z_axis)}")
            self.__console.debug(f"  - dot(rt_y, rt_z): {np.dot(rt_y_axis, rt_z_axis)}")
            self.__console.debug(f"  - dot(dda_y, rt_y): {np.dot(dda_y_axis, rt_y_axis)}")
            self.__console.debug(f"  - rt_rot_matrix:\n{rt_rot_matrix}")
            self.__console.debug(f"  - det(rt_rot_matrix): {np.linalg.det(rt_rot_matrix)}")
            self.__console.debug(f"{'=' * 60}\n")

        rt_rpy = R.from_matrix(rt_rot_matrix).as_euler("xyz")
        return np.hstack([rt_position, rt_rpy])

    @staticmethod
    def __orthogonal_reference_axis(axis: np.ndarray) -> np.ndarray:
        candidates = (
            np.asarray([0.0, 0.0, 1.0], dtype=float),
            np.asarray([0.0, 1.0, 0.0], dtype=float),
            np.asarray([1.0, 0.0, 0.0], dtype=float),
        )
        axis = axis / np.linalg.norm(axis)
        return min(candidates, key=lambda candidate: abs(float(np.dot(axis, candidate))))

    @classmethod
    def __basis_from_facing_and_up(cls, facing_axis: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
        facing_axis = np.asarray(facing_axis, dtype=float)
        facing_axis = facing_axis / np.linalg.norm(facing_axis)
        up_hint = np.asarray(up_hint, dtype=float)
        up_hint = up_hint / np.linalg.norm(up_hint)
        up_axis = up_hint - facing_axis * float(np.dot(up_hint, facing_axis))
        if np.linalg.norm(up_axis) < 1e-9:
            up_hint = cls.__orthogonal_reference_axis(facing_axis)
            up_axis = up_hint - facing_axis * float(np.dot(up_hint, facing_axis))
        up_axis = up_axis / np.linalg.norm(up_axis)
        side_axis = np.cross(up_axis, facing_axis)
        side_axis = side_axis / np.linalg.norm(side_axis)
        return np.column_stack([side_axis, up_axis, facing_axis])

    @classmethod
    def __rotation_from_pipe_facing_axis(
        cls,
        pipe_facing_world: np.ndarray,
        world_up_hint: np.ndarray,
        local_pipe_facing_axis: np.ndarray,
        local_reference_axis: np.ndarray | None = None,
    ) -> np.ndarray:
        pipe_facing_world = np.asarray(pipe_facing_world, dtype=float)
        local_pipe_facing_axis = np.asarray(local_pipe_facing_axis, dtype=float)
        local_up_hint = (
            np.asarray([0.0, 0.0, 1.0], dtype=float)
            if local_reference_axis is None
            else np.asarray(local_reference_axis, dtype=float)
        )
        local_axis_unit = local_pipe_facing_axis / np.linalg.norm(local_pipe_facing_axis)
        if np.linalg.norm(local_up_hint) < 1e-9:
            local_up_hint = cls.__orthogonal_reference_axis(local_axis_unit)
        local_up_hint = local_up_hint / np.linalg.norm(local_up_hint)
        if abs(float(np.dot(local_axis_unit, local_up_hint))) > 0.98:
            local_up_hint = cls.__orthogonal_reference_axis(local_axis_unit)

        local_basis = cls.__basis_from_facing_and_up(local_axis_unit, local_up_hint)
        world_basis = cls.__basis_from_facing_and_up(pipe_facing_world, world_up_hint)
        rot_matrix = world_basis @ local_basis.T
        if np.linalg.det(rot_matrix) < 0:
            world_basis[:, 0] *= -1.0
            rot_matrix = world_basis @ local_basis.T
        return rot_matrix

    @staticmethod
    def __pose_to_T(pose) -> np.ndarray:
        pose_arr = np.asarray(pose, dtype=float)
        if pose_arr.shape == (4, 4):
            return pose_arr.copy()
        flat = pose_arr.reshape(-1)
        if flat.size < 3:
            raise ValueError(f"pose must contain at least xyz, got shape={pose_arr.shape}")
        T = np.eye(4)
        T[:3, 3] = flat[:3]
        if flat.size >= 6:
            T[:3, :3] = R.from_euler("xyz", flat[3:6]).as_matrix()
        return T

    @staticmethod
    def __target_slot_rt_name(slot_data: dict) -> str | None:
        """slot에 있는 RT 후보(RT1/RT2) 중 하나를 고른다. 둘 다 있으면 RT1을 우선한다."""
        if slot_data.get("RT1") is not None:
            return "RT1"
        if slot_data.get("RT2") is not None:
            return "RT2"
        return None

    def __pose_groups_to_target_groups(
        self, pose_groups, target_point, pose_name_to_robot_name=None
    ) -> list[dict]:
        """pose group을 최소 정보만 담은 target group으로 변환한다.

        각 target group은 다음만 담는다:
            - name: 표시용 이름 (예: "Inspection pose 1")
            - index: 순번
            - target_point: 검사 기준 위치 [x, y, z]
            - dda_pose: DDA endeffector target pose (4x4 world transform, list)
            - rt_pose:  RT endeffector target pose (4x4 world transform, list)

        positioner 회전 필요 여부는 여기서 판단하지 않는다. base planner(path planner)가
        이 정보(특히 rt_pose)를 바탕으로 직접 판단한다.
        pose_name_to_robot_name은 하위 호환용으로 남겨두지만 사용하지 않는다
        (로봇 이름 매핑은 소비자 쪽에서 pose_name으로 처리한다).
        """
        target_point_list = np.asarray(target_point, dtype=float).reshape(3).tolist()
        target_groups = []
        for pose_group in list(pose_groups or []):
            if not isinstance(pose_group, dict):
                continue
            for slot in pose_group.values():
                if not isinstance(slot, dict) or slot.get("DDA") is None:
                    continue
                rt_name = self.__target_slot_rt_name(slot)
                if rt_name is None:
                    continue

                target_groups.append({
                    "name": f"Inspection pose {len(target_groups) + 1}",
                    "index": len(target_groups),
                    "target_point": target_point_list,
                    "dda_pose": self.__pose_to_T(slot["DDA"]).tolist(),
                    "rt_pose": self.__pose_to_T(slot[rt_name]).tolist(),
                })
        return target_groups

    def calculate_DDA_RT_pose_for_taking_xray(
        self,
        target_point,
        num_candidates: int,
        distance_from_dda_to_surface: float,
        distance_from_dda_to_rt: float,
        angle_of_rt: float,
        distance_reference_mesh: o3d.geometry.TriangleMesh | None = None,
        rt_pipe_facing_axis=(0.0, -1.0, 0.0),
        pose_name_to_robot_name=None,
        force_90_fallback: bool = False,
    ) -> list[dict]:
        """Calculate DDA/RT inspection target groups.

        Returns:
            target_groups ready for viewer visualization and path planning.
        """
        if rt_pipe_facing_axis is not None:
            self.set_rt_pipe_facing_axis(rt_pipe_facing_axis)

        candidate_count = max(1, int(num_candidates))
        candidate_step_deg = 360.0 / float(candidate_count)
        self.debuging_info["candidate_count"]           = candidate_count
        self.debuging_info["candidate_step_deg"]        = candidate_step_deg
        self.debuging_info["rt_angle_of_rt_input_deg"]  = float(angle_of_rt)
        self.debuging_info["force_90_fallback"]         = bool(force_90_fallback)
        self.debuging_info["pose_return_format"]        = "target_groups"

        _, pose_groups = self.calculate_DDA_RT_pose_for_taking_xray_3pair_120(
            target_point,
            distance_from_dda_to_surface=distance_from_dda_to_surface,
            distance_from_dda_to_rt=distance_from_dda_to_rt,
            angle_of_rt=angle_of_rt,
            candidate_step_deg=candidate_step_deg,
            allow_2pair_fallback=True,
            distance_reference_mesh=distance_reference_mesh,
            force_90_fallback=force_90_fallback,
        )
        return self.__pose_groups_to_target_groups(
            pose_groups,
            target_point,
            pose_name_to_robot_name=pose_name_to_robot_name,
        )
    
    def calculate_DDA_RT_pose_for_taking_xray_indexed_0_90(
        self,
        target_point: tuple[float, float, float] | np.ndarray,
        num_candidates: int,
        distance_from_dda_to_surface: float,
        distance_from_dda_to_rt: float,
        angle_of_rt: float,
        distance_reference_mesh: o3d.geometry.TriangleMesh | None = None,
    ):
        """0도와 90도 간격의 DDA/RT 자세 세트를 생성하는 legacy 함수다.

        `num_candidates`는 4로 나누어 떨어져야 하며, 각 후보 index에서 90도 떨어진
        후보와 짝을 지어 충돌 없는 DDA/RT 그룹을 만든다.
        """
        if num_candidates <= 0:
            raise ValueError(f"num_candidates must be > 0, got {num_candidates}")
        if num_candidates % 4 != 0:
            raise ValueError(
                "num_candidates must be divisible by 4 for indexed 0/90 pairing, "
                f"got {num_candidates}"
            )

        if self.__is_debug_mode:
            self.debuging_info = {}

        candidate_radius = self.__dda_candidate_centerline_radius(
            distance_from_dda_to_surface,
            self.__dda_mesh if distance_reference_mesh is None else distance_reference_mesh,
        )
        dda_base_candidates = self.__calculate_dda_pose_candidate(
            np.asarray(target_point),
            candidate_radius,
            num_candidates,
        )
        dda_base_candidates = self.__adjust_dda_candidates_for_mesh_surface_distance(
            dda_base_candidates,
            distance_from_dda_to_surface,
            self.__dda_mesh if distance_reference_mesh is None else distance_reference_mesh,
        )

        slot_results: list[dict[str, list[float]] | None] = []
        valid_base_dda_poses = []
        base_dda_collision_count = 0
        valid_slot_indices = []
        invalid_slot_indices = []

        for idx, dda_pose in enumerate(dda_base_candidates):
            slot = self.__process_dda_rt_slot_with_collision(
                dda_pose,
                angle_of_rt,
                distance_from_dda_to_rt,
            )
            slot_results.append(slot)
            if slot is None:
                invalid_slot_indices.append(idx)
            else:
                valid_slot_indices.append(idx)
                valid_base_dda_poses.append(dda_pose)

        if self.__is_debug_mode:
            self.debuging_info["dda_base_candidates"] = dda_base_candidates
            self.debuging_info["valid_base_dda_poses"] = valid_base_dda_poses
            self.debuging_info["indexed_0_90_slot_results"] = slot_results
            self.debuging_info["indexed_0_90_valid_slot_indices"] = valid_slot_indices
            self.debuging_info["indexed_0_90_invalid_slot_indices"] = invalid_slot_indices

        pose_groups = []
        incomplete_pose_groups = []
        quarter_offset = num_candidates // 4
        step_deg = 360.0 / num_candidates

        for idx in range(num_candidates):
            idx_90 = (idx + quarter_offset) % num_candidates
            group_0_data = slot_results[idx]
            group_90_data = slot_results[idx_90]

            if group_0_data is not None and group_90_data is not None:
                group_data = {
                    "0": dict(group_0_data),
                    "90": dict(group_90_data),
                }
                pose_groups.append(group_data)
            elif self.__is_debug_mode and (group_0_data is not None or group_90_data is not None):
                partial_group = {}
                if group_0_data is not None:
                    partial_group["0"] = dict(group_0_data)
                if group_90_data is not None:
                    partial_group["90"] = dict(group_90_data)
                incomplete_pose_groups.append(partial_group)

        if self.__is_debug_mode:
            self.debuging_info["indexed_0_90_incomplete_pose_groups"] = incomplete_pose_groups
            self.debuging_info["indexed_0_90_quarter_offset"] = quarter_offset

        return json.dumps(pose_groups), pose_groups

    def calculate_DDA_RT_pose_for_taking_xray_3pair_120(
        self,
        target_point: tuple[float, float, float] | np.ndarray,
        distance_from_dda_to_surface: float,
        distance_from_dda_to_rt: float,
        angle_of_rt: float,
        candidate_step_deg: float = 3.0,
        gap_tolerance_deg: float = 10.0,
        allow_2pair_fallback: bool = True,
        distance_reference_mesh: o3d.geometry.TriangleMesh | None = None,
        force_90_fallback: bool = False,
    ) -> tuple[str, list[dict]]:
        """120도 간격의 3개 DDA/RT 자세 세트를 우선 생성하는 legacy 함수다.

        지정한 각도 간격으로 전체 DDA 후보를 만들고, 충돌 없는 슬롯 중에서
        120도 간격 3개 조합을 먼저 찾는다. 실패하면 설정에 따라 2개 조합으로 fallback한다.

        Returns:
            `(json_str, pose_groups)` 형식. `pose_groups`는 angle label별 DDA/RT pose dict를 담는다.
        """
        # 입력값 검증: 각도/거리 값은 유한해야 하고 tolerance는 60도 미만이어야 한다.
        # tolerance가 60도 이상이면 120도 조합 판단이 모호해진다.
        for _name, _val in (
            ("distance_from_dda_to_surface", distance_from_dda_to_surface),
            ("distance_from_dda_to_rt", distance_from_dda_to_rt),
            ("angle_of_rt", angle_of_rt),
            ("candidate_step_deg", candidate_step_deg),
            ("gap_tolerance_deg", gap_tolerance_deg),
        ):
            if not math.isfinite(_val):
                raise ValueError(f"{_name} must be finite, got {_val!r}")
        if candidate_step_deg <= 0:
            raise ValueError(f"candidate_step_deg must be > 0, got {candidate_step_deg}")
        if not (0.0 <= gap_tolerance_deg < 60.0):
            raise ValueError(
                f"gap_tolerance_deg must be in [0, 60), got {gap_tolerance_deg}"
            )
        _tp = np.asarray(target_point, dtype=float)
        if _tp.shape != (3,) or not bool(np.all(np.isfinite(_tp))):
            raise ValueError(f"target_point must be 3 finite floats, got {target_point!r}")

        # DDA 후보 생성. num_candidates를 바꿔 재샘플링할 수 있도록 헬퍼로 감싼다.
        candidate_radius = self.__dda_candidate_centerline_radius(
            distance_from_dda_to_surface,
            self.__dda_mesh if distance_reference_mesh is None else distance_reference_mesh,
        )

        def _build_slot_candidates(n_candidates: int):
            _step_deg = 360.0 / n_candidates  # 실제 후보 간격
            _base = self.__calculate_dda_pose_candidate(
                np.asarray(target_point),
                candidate_radius,
                n_candidates,
            )
            # mesh 표면과의 목표 거리를 만족하도록 후보 위치 보정
            _base = self.__adjust_dda_candidates_for_mesh_surface_distance(
                _base,
                distance_from_dda_to_surface,
                self.__dda_mesh if distance_reference_mesh is None else distance_reference_mesh,
            )
            _slots: list[dict | None] = []
            for dda_pose in _base:
                if self.__check_collision(self.__dda_mesh, dda_pose, self.__dda_invers_transform_mat):
                    _slots.append(None)
                    continue
                slot = self.__process_dda_rt_combination(dda_pose, angle_of_rt, distance_from_dda_to_rt)
                _slots.append(slot)  # 충돌 등으로 실패한 후보는 None으로 남긴다.
            # 유효한 후보 index만 추려 3개/2개 자세 세트 조합을 만든다.
            _valid = sorted(i for i, s in enumerate(_slots) if s is not None)
            return _step_deg, _slots, _valid, set(_valid)

        num_candidates = int(round(360.0 / candidate_step_deg))
        step_deg, slot_results, valid_indices, valid_set = _build_slot_candidates(num_candidates)

        # 3개 자세 세트: 120도 간격과 tolerance를 만족하는 조합 탐색
        # 부동소수 오차로 인한 ceil/floor 경계 문제를 줄이기 위해 epsilon을 사용한다.
        EPS = 1e-9
        ideal_idx_gap = num_candidates / 3.0
        tol_idx = gap_tolerance_deg / step_deg
        min_gap = int(np.ceil(ideal_idx_gap - tol_idx - EPS))
        max_gap = int(np.floor(ideal_idx_gap + tol_idx + EPS))

        best_triple: tuple[int, int, int] | None = None
        best_deviation_sum: float = float("inf")

        # i < j < k 순서를 유지하면서 원형 간격 3개를 검사한다.
        # gap3는 k에서 wrap-around하여 i로 돌아오는 간격이다.
        for i in valid_indices:
            for gap1 in range(min_gap, max_gap + 1):
                j = i + gap1
                if j >= num_candidates or j not in valid_set:
                    continue
                for gap2 in range(min_gap, max_gap + 1):
                    k = j + gap2
                    if k >= num_candidates or k not in valid_set:
                        continue
                    gap3 = num_candidates - gap1 - gap2
                    if not (min_gap <= gap3 <= max_gap):
                        continue
                    # 각 gap이 120도 tolerance 안에 들어오는 조합만 후보로 둔다.
                    ang_gaps = (gap1 * step_deg, gap2 * step_deg, gap3 * step_deg)
                    if any(abs(ag - 120.0) > gap_tolerance_deg + EPS for ag in ang_gaps):
                        continue
                    dev = sum(abs(ag - 120.0) for ag in ang_gaps)
                    if best_triple is None \
                            or dev < best_deviation_sum \
                            or (dev == best_deviation_sum and (i, j, k) < best_triple):
                        best_deviation_sum = dev
                        best_triple = (i, j, k)

        # 3개 조합이 있으면 가장 작은 deviation 조합을 반환한다.
        if best_triple is not None and not force_90_fallback:
            pose_groups: list[dict] = [{}]
            group = pose_groups[0]
            for idx, ideal_label in zip(best_triple, ("0", "120", "240")):
                group[ideal_label] = dict(slot_results[idx])  # type: ignore[arg-type]
            return json.dumps(pose_groups), pose_groups

        # 2개 자세 세트 fallback
        if force_90_fallback:
            self.debuging_info["selected_pose_pair_strategy"] = "forced_fallback"
            self.debuging_info["skipped_best_triple"] = best_triple
        elif not allow_2pair_fallback:
            return "[]", []

        best_pair: tuple[int, int] | None = None
        best_pair_key: tuple | None = None
        fallback_pair_gap_deg = 90.0 if force_90_fallback else 120.0
        fallback_pair_labels = ("0", "90") if force_90_fallback else ("0", "120")
        # 90도 fallback은 두 RT(-y)축이 ±45도에 가장 가까운 쌍을 고른다.
        fallback_reference_deg = 45.0
        if self.__is_debug_mode:
            self.debuging_info["fallback_pair_gap_deg"] = fallback_pair_gap_deg

        # 90도 2쌍은 두 RT가 ±45도에 정확히 오도록 해야 한다. facing 각도 = DDA 위치각이므로
        # grid에 45도가 있어야 하고(45=360/8 → 8의 배수), ±45/±60/90/120을 모두 정확히
        # 담으려면 LCM(8,3,4,6)=24의 배수가 필요하다. 현재 후보 수 이상인 24의 배수로
        # 재샘플링한다. (예: 9->24, 15->24, 25->48). step 30도(=12개)로는 45도를 못 찍는다.
        if force_90_fallback and num_candidates % 24 != 0:
            resampled_num = int(math.ceil(num_candidates / 24.0)) * 24
            if resampled_num != num_candidates:
                num_candidates = resampled_num
                step_deg, slot_results, valid_indices, valid_set = _build_slot_candidates(
                    num_candidates
                )
                if self.__is_debug_mode:
                    self.debuging_info["fallback_resampled_num_candidates"] = num_candidates

        def circular_distance_deg(actual, desired):
            return abs((float(actual) - float(desired) + 180.0) % 360.0 - 180.0)

        # 90도 fallback일 때만 사용: 각 후보 slot의 RT(-y)축 각도를 미리 계산한다.
        # 90도 2쌍 조건: RT source가 반대 방향을 바라봐야 하므로 지향축을 뒤집어 측정한다.
        facing_angles: dict[int, list[float]] = {}
        if force_90_fallback:
            for idx in valid_indices:
                angs = self.__rt_slot_facing_angles_deg(
                    slot_results[idx], flip_facing=True  # type: ignore[arg-type]
                )
                if angs:
                    facing_angles[idx] = angs

        def pair_pm_deviation(i_idx: int, j_idx: int) -> float | None:
            """쌍(i,j)의 두 RT(-y)축을 +ref/-ref에 대칭 배정했을 때 최소 최대편차."""
            angs_i = facing_angles.get(i_idx)
            angs_j = facing_angles.get(j_idx)
            if not angs_i or not angs_j:
                return None
            best = float("inf")
            for ai in angs_i:
                for aj in angs_j:
                    # i->+ref, j->-ref  또는  i->-ref, j->+ref 중 더 나은 배정
                    best = min(
                        best,
                        max(
                            circular_distance_deg(ai, fallback_reference_deg),
                            circular_distance_deg(aj, -fallback_reference_deg),
                        ),
                        max(
                            circular_distance_deg(ai, -fallback_reference_deg),
                            circular_distance_deg(aj, fallback_reference_deg),
                        ),
                    )
            return best

        for i in valid_indices:
            for j in valid_indices:
                if j <= i:
                    continue
                gap_deg = (j - i) * step_deg
                other_deg = 360.0 - gap_deg
                # gap 또는 반대 방향 gap 중 120도 tolerance에 들어오는 arc를 선택한다.
                # tolerance가 60도 미만이면 두 조건이 동시에 성립하지 않아 해석이 명확하다.
                # tolerance 범위는 위 입력 검증에서 제한한다.
                if abs(gap_deg - fallback_pair_gap_deg) <= gap_tolerance_deg:
                    gap_dev = abs(gap_deg - fallback_pair_gap_deg)
                elif abs(other_deg - fallback_pair_gap_deg) <= gap_tolerance_deg:
                    gap_dev = abs(other_deg - fallback_pair_gap_deg)
                else:
                    continue
                # 90도 fallback: RT(-y)축이 ±45에 가장 가까운 쌍을 1순위로 선택한다.
                # 그 다음 gap 편차, 마지막으로 더 앞선 index 조합을 택한다.
                if force_90_fallback:
                    pm_dev = pair_pm_deviation(i, j)
                    if pm_dev is None:
                        continue
                    key = (round(pm_dev, 6), round(gap_dev, 6), (i, j))
                else:
                    key = (round(gap_dev, 6), (i, j))
                if best_pair_key is None or key < best_pair_key:
                    best_pair_key = key
                    best_pair = (i, j)

        if best_pair is None:
            return "[]", []

        if self.__is_debug_mode and force_90_fallback:
            self.debuging_info["fallback_pair_pm45_deviation_deg"] = best_pair_key[0]
            self.debuging_info["fallback_pair_facing_angles_deg"] = [
                facing_angles.get(idx) for idx in best_pair
            ]

        pose_groups = [{}]
        group = pose_groups[0]
        for idx, ideal_label in zip(best_pair, fallback_pair_labels):
            group[ideal_label] = dict(slot_results[idx])  # type: ignore[arg-type]
        return json.dumps(pose_groups), pose_groups

    # world X축을 회전축으로 삼는다.
    # 0도 기준 방향은 -Y이고, X축 기준 반시계방향(오른손 법칙: -Y -> -Z)이 +다.
    __RT_ANGLE_AXIS_UNIT = np.asarray([1.0, 0.0, 0.0], dtype=float)
    __RT_ANGLE_REFERENCE_AXIS = np.asarray([0.0, -1.0, 0.0], dtype=float)  # 0도 기준: -Y
    __RT_ANGLE_TANGENT_AXIS = np.asarray([0.0, 0.0, -1.0], dtype=float)  # +90도(반시계): -Z

    def __rt_pose_facing_angle_deg(self, rt_pose, flip_facing: bool = False) -> float:
        """RT pose의 배관 지향축(-y축)을 world로 변환해 world X축 회전각으로 측정한다.

        RT 위치가 아니라 자세(축) 기준이며, 0도=-Y, 반시계방향(+X 오른손)이 +다.
        flip_facing=True면 반대 방향(배관 지향축의 반대)을 기준으로 측정한다.
        (90도 2쌍 조건: RT source가 반대 방향을 바라봐야 함)
        """
        pose = np.asarray(rt_pose, dtype=float)
        facing_axis = -self.__rt_pipe_facing_axis if flip_facing else self.__rt_pipe_facing_axis
        rel = R.from_euler("xyz", pose[3:6]).as_matrix() @ facing_axis
        rel = rel - self.__RT_ANGLE_AXIS_UNIT * float(np.dot(rel, self.__RT_ANGLE_AXIS_UNIT))
        if np.linalg.norm(rel) < 1e-9:
            return 0.0
        x = float(np.dot(rel, self.__RT_ANGLE_REFERENCE_AXIS))
        y = float(np.dot(rel, self.__RT_ANGLE_TANGENT_AXIS))
        return float(np.rad2deg(np.arctan2(y, x)))

    def __rt_slot_facing_angles_deg(self, slot, flip_facing: bool = False) -> list[float]:
        """slot에서 사용 가능한 RT(RT1/RT2)들의 배관 지향축 각도 목록을 반환한다."""
        angles = []
        for rt_name in ("RT1", "RT2"):
            pose = slot.get(rt_name)
            if pose is None:
                continue
            angles.append(self.__rt_pose_facing_angle_deg(pose, flip_facing=flip_facing))
        return angles


    def __process_dda_rt_slot_with_collision(
        self,
        dda_pose: np.ndarray,
        angle_of_rt: float,
        distance_from_dda_to_rt: float,
    ) -> dict[str, Any] | None:
        """DDA 기본 후보 한 개에 대해 DDA/RT 슬롯을 만들고 충돌 여부를 확인한다."""
        is_dda_collision = self.__check_collision(
            self.__dda_mesh,
            dda_pose,
            self.__dda_invers_transform_mat,
        )
        if is_dda_collision:
            return None
        return self.__process_dda_rt_combination(
            dda_pose,
            angle_of_rt,
            distance_from_dda_to_rt,
        )

    def __process_dda_rt_combination(
        self,
        dda_pose: np.ndarray,
        angle_of_rt: float,
        distance_from_dda_to_rt: float,
    ) -> dict[str, list[float]] | None:
        """DDA/RT 후보 슬롯 한 개를 만들고 충돌하는 RT pose는 제외한다."""
        result = {"DDA": dda_pose.tolist()}

        rt1_pose = self.__calculate_rt_pose_for_angle(dda_pose, angle_of_rt, distance_from_dda_to_rt)
        is_rt1_collision = self.__check_collision(
            self.__rt_mesh,
            rt1_pose,
            self.__rt_invers_transform_mat,
        )
        if not is_rt1_collision:
            result["RT1"] = rt1_pose.tolist()

        rt2_pose = self.__calculate_rt_pose_for_angle(dda_pose, -angle_of_rt, distance_from_dda_to_rt)
        is_rt2_collision = self.__check_collision(
            self.__rt_mesh,
            rt2_pose,
            self.__rt_invers_transform_mat,
        )
        if not is_rt2_collision:
            result["RT2"] = rt2_pose.tolist()

        result["_rt_angle_of_rt_deg"] = float(angle_of_rt)
        result["_rt1_angle_input_deg"] = float(angle_of_rt)
        result["_rt2_angle_input_deg"] = float(-angle_of_rt)

        if "RT1" in result or "RT2" in result:
            return result

        rejected = {
            "DDA": dda_pose.tolist(),
            "RT1": rt1_pose.tolist(),
            "RT2": rt2_pose.tolist(),
            "_rt1_collision": bool(is_rt1_collision),
            "_rt2_collision": bool(is_rt2_collision),
        }
        self.debuging_info.setdefault("rejected_pose_groups", []).append({"rejected": rejected})
        return None
    
    def calculate_pipe_profile(
        self,
        target_point: tuple[float, float, float] | np.ndarray,  # x,y,z
        sampling_size_for_calculating_normal: float = 0.01,
        radius_offset_for_sampling_points_in_sphere: float = 0.003,
        sampling_cylinder_radius: float = 0.005,
        sampling_cylinder_height_range: tuple[float, float] = (-0.1, 0.3),
    ):
        """선택한 배관 표면점 주변에서 로컬 원통 프로파일을 계산한다.

        1. 선택점 주변의 local normal을 추정한다.
        2. normal 반대 방향의 작은 실린더 ROI에서 피팅 후보점을 얻는다.
        3. ROI 반대편 점으로 초기 중심과 반지름을 추정한다.
        4. 중심 주변 sphere에서 점을 재샘플링하고 최종 실린더를 피팅한다.

        Args:
            target_point: 사용자가 선택한 배관 표면점.
            sampling_size_for_calculating_normal: normal 추정에 사용할 주변 박스 반경.
            radius_offset_for_sampling_points_in_sphere: sphere 샘플링 반지름 여유값.
            sampling_cylinder_radius: normal 방향 샘플링 실린더 반지름.
            sampling_cylinder_height_range: normal 방향 샘플링 실린더 높이 범위.
        """

        if self.__is_debug_mode:
            self.debuging_info = {}

        # 선택점 주변의 작은 박스에서 normal 추정용 점을 추출한다.
        if not isinstance(target_point, np.ndarray):
            target_point = np.array(target_point)
        gap = np.full(3, sampling_size_for_calculating_normal, dtype=np.float64)
        min_bound = target_point - gap
        max_bound = target_point + gap
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)  # type: ignore

        if self.__is_debug_mode:
            self.debuging_info["sampling_box"] = [min_bound, max_bound]

        indices = box.get_point_indices_within_bounding_box(self._scan_data.points)
        selected_points = self._scan_data.select_by_index(indices)
        if len(selected_points.points) == 0:
            selected_points = None
            raise RuntimeError(
                "target_point 주변에 normal 추정용 점이 없습니다. target_point 또는 sampling_size_for_calculating_normal 값을 확인하세요."
            )

        if self.__is_debug_mode:
            self.debuging_info["selected_points"] = selected_points

        # 선택한 주변점 normal의 중앙값으로 local normal을 추정한다.
        normals = np.asarray(selected_points.normals)
        x_m = np.median(normals[:, 0])
        y_m = np.median(normals[:, 1])
        z_m = np.median(normals[:, 2])
        normal_m = np.array([x_m, y_m, z_m])

        if self.__is_debug_mode:
            self.debuging_info["normal_m"] = normal_m

        # normal 반대 방향의 작은 실린더 ROI에서 후보 점을 샘플링한다.
        points_in_cylinder = self.__extract_points_in_cylinder(
            np.asarray(self._scan_data.points),
            target_point,
            normal_m * -1,
            sampling_cylinder_radius,
            sampling_cylinder_height_range,
        )

        if self.__is_debug_mode:
            self.debuging_info["points_in_cylinder"] = points_in_cylinder
            self.debuging_info["pipe_profile_sampling_cylinder"] = {
                "start": target_point,
                "axis": normal_m * -1,
                "radius": sampling_cylinder_radius,
                "height_range": sampling_cylinder_height_range,
            }

        clusters = self.__cluster_points_along_line(
            points_in_cylinder,
            target_point,
            normal_m * -1,
            sampling_cylinder_radius,
        )
        self.debuging_info["pipe_profile_clusters"] = clusters
        self.debuging_info["pipe_profile_points_in_cylinder"] = points_in_cylinder
        self.debuging_info["pipe_profile_target_point"] = target_point
        self.debuging_info["pipe_profile_normal_axis"] = normal_m * -1

        if len(clusters) < 2 or len(clusters[1]) == 0:
            raise RuntimeError("Pipe profile clustering failed: need at least two point clusters.")
        estimated_opposite_point = clusters[1][-1]
        estimated_center = (target_point + estimated_opposite_point) / 2
        estimated_radius = float(np.linalg.norm(estimated_opposite_point - estimated_center))

        if self.__is_debug_mode:
            self.debuging_info["estimated_center"] = estimated_center
            self.debuging_info["estimated_radius"] = estimated_radius

        points_in_sphere = self.__extract_points_in_sphere(
            np.asarray(self._scan_data.points),
            estimated_center,
            estimated_radius + radius_offset_for_sampling_points_in_sphere,
        )

        if self.__is_debug_mode:
            self.debuging_info["points_in_sphere"] = points_in_sphere

        direction, center, radius, _ = fit_cylinder(points_in_sphere)

        # 최종 원통 피팅 결과를 현재 배관 프로파일로 저장한다.
        self.__pipe_direction = direction
        self.__pipe_center = center
        self.__pipe_radius = radius

    def __dda_candidate_centerline_radius(
        self,
        surface_clearance: float,
        distance_reference_mesh: o3d.geometry.TriangleMesh | None = None,
    ) -> float:
        mesh = self.__dda_mesh if distance_reference_mesh is None else distance_reference_mesh
        front_extent = self.__mesh_extent_along_local_axis(
            mesh,
            self.__dda_invers_transform_mat,
            self.__dda_pipe_facing_axis,
        )
        candidate_radius = float(self.__pipe_radius + surface_clearance + front_extent)
        if self.__is_debug_mode:
            self.debuging_info["dda_mesh_front_extent_along_facing_axis"] = float(front_extent)
            self.debuging_info["dda_candidate_centerline_radius"] = candidate_radius
            self.debuging_info["dda_candidate_surface_clearance"] = float(surface_clearance)
        return candidate_radius

    def __rt_candidate_origin_distance_offset(self) -> float:
        return self.__mesh_extent_along_local_axis(
            self.__rt_mesh,
            self.__rt_invers_transform_mat,
            self.__rt_pipe_facing_axis,
        )

    @staticmethod
    def __mesh_extent_along_local_axis(
        link_model: o3d.geometry.TriangleMesh,
        tcp_to_link_pose_T: np.ndarray,
        local_axis: np.ndarray,
    ) -> float:
        points = np.asarray(link_model.vertices, dtype=float)
        if len(points) == 0:
            mesh_pcd = link_model.sample_points_uniformly(number_of_points=3000)
            points = np.asarray(mesh_pcd.points, dtype=float)
        if len(points) == 0:
            return 0.0

        T = np.asarray(tcp_to_link_pose_T, dtype=float)
        points_in_pose_frame = (T[:3, :3] @ points.T).T + T[:3, 3]
        axis = np.asarray(local_axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        projected = points_in_pose_frame @ axis
        return max(0.0, float(np.max(projected)))

    def __calculate_dda_pose_candidate(
        self,
        point_on_pipe_surface: np.ndarray,
        radius: float,
        num_candidates: int,
    ) -> np.ndarray:
        """World X축을 기준축으로 DDA TCP 후보를 생성한다."""
        angles = 2.0 * np.pi * np.arange(num_candidates) / num_candidates
        return self.__calculate_dda_pose_candidates_for_angles(
            point_on_pipe_surface,
            radius,
            angles,
        )

    def __calculate_dda_pose_candidates_for_angles(
        self,
        point_on_pipe_surface: np.ndarray,
        radius: float,
        angles_rad: np.ndarray | tuple[float, ...] | list[float],
    ) -> np.ndarray:
        """World X축 기준의 지정 각도들로 DDA TCP 후보를 생성한다."""
        point_on_pipe_surface = np.asarray(point_on_pipe_surface, dtype=float).reshape(3)
        angles = np.asarray(angles_rad, dtype=float).reshape(-1)
        candidate_axis_unit = np.asarray([1.0, 0.0, 0.0], dtype=float)

        vec_to_surface = point_on_pipe_surface - self.__pipe_center
        proj_len = float(np.dot(vec_to_surface, candidate_axis_unit))
        center = self.__pipe_center + proj_len * candidate_axis_unit

        v1 = np.asarray([0.0, 1.0, 0.0], dtype=float)
        v2 = np.asarray([0.0, 0.0, 1.0], dtype=float)

        offsets = np.outer(np.cos(angles), v1) + np.outer(np.sin(angles), v2)
        positions = center + offsets * radius

        facing_axes = center - positions
        facing_norm = np.linalg.norm(facing_axes, axis=1, keepdims=True)
        facing_norm[facing_norm < 1e-12] = 1.0
        facing_axes = facing_axes / facing_norm

        rot_mats = np.stack([
            self.__rotation_from_pipe_facing_axis(
                pipe_facing_world=facing_axis,
                world_up_hint=candidate_axis_unit,
                local_pipe_facing_axis=self.__dda_pipe_facing_axis,
                local_reference_axis=self.__dda_pipe_parallel_axis,
            )
            for facing_axis in facing_axes
        ], axis=0)

        if self.__is_debug_mode:
            configured_facing_world = np.einsum("nij,j->ni", rot_mats, self.__dda_pipe_facing_axis)
            local_minus_y_world = np.einsum("nij,j->ni", rot_mats, np.asarray([0.0, -1.0, 0.0]))
            self.debuging_info["dda_pipe_facing_axis_local"] = self.__dda_pipe_facing_axis.tolist()
            self.debuging_info["dda_pipe_parallel_axis_local"] = self.__dda_pipe_parallel_axis.tolist()
            self.debuging_info["dda_candidate_axis"] = candidate_axis_unit.tolist()
            self.debuging_info["dda_candidate_section_center"] = center.tolist()
            self.debuging_info["dda_candidate_radial_reference"] = v1.tolist()
            self.debuging_info["dda_candidate_tangent_reference"] = v2.tolist()
            self.debuging_info["dda_configured_facing_dot_pipe_center"] = (
                np.sum(configured_facing_world * facing_axes, axis=1).tolist()
            )
            self.debuging_info["dda_minus_y_dot_pipe_center"] = (
                np.sum(local_minus_y_world * facing_axes, axis=1).tolist()
            )

        rpy_array = R.from_matrix(rot_mats).as_euler("xyz", degrees=False)
        return np.hstack((positions, rpy_array))

    def __adjust_dda_candidates_for_mesh_surface_distance(
        self,
        dda_poses: np.ndarray,
        desired_surface_distance: float,
        distance_reference_mesh: o3d.geometry.TriangleMesh,
    ) -> np.ndarray:
        adjusted = []
        clearances = []
        shifts = []
        for pose in np.asarray(dda_poses, dtype=float):
            clearance = self.__mesh_pipe_surface_clearance(
                distance_reference_mesh,
                pose,
                self.__dda_invers_transform_mat,
            )
            shift = float(desired_surface_distance - clearance)
            pipe_facing_axis = R.from_euler("xyz", pose[3:]).as_matrix() @ self.__dda_pipe_facing_axis
            outward_axis = -pipe_facing_axis / np.linalg.norm(pipe_facing_axis)
            corrected = pose.copy()
            corrected[:3] = corrected[:3] + outward_axis * shift
            adjusted.append(corrected)
            clearances.append(float(clearance))
            shifts.append(float(shift))

        if self.__is_debug_mode:
            self.debuging_info["mesh_surface_clearances_before_adjustment"] = clearances
            self.debuging_info["mesh_surface_distance_shifts"] = shifts
            self.debuging_info["desired_mesh_surface_distance"] = float(desired_surface_distance)

        return np.asarray(adjusted, dtype=float)

    def __mesh_pipe_surface_clearance(
        self,
        link_model: o3d.geometry.TriangleMesh,
        tcp_pose: np.ndarray,
        tcp_to_link_pose_T: np.ndarray,
        sample_count: int = 3000,
    ) -> float:
        tcp_pose_T = np.eye(4)
        tcp_pose_T[:3, :3] = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
        tcp_pose_T[:3, 3] = tcp_pose[:3]

        mesh_copy = copy.deepcopy(link_model)
        mesh_copy.transform(tcp_pose_T @ tcp_to_link_pose_T)  # type: ignore
        mesh_pcd = mesh_copy.sample_points_uniformly(number_of_points=sample_count)
        points = np.asarray(mesh_pcd.points, dtype=float)
        if len(points) == 0:
            return float("inf")

        pipe_axis = self.__pipe_direction / np.linalg.norm(self.__pipe_direction)
        rel = points - self.__pipe_center
        axial = np.outer(np.dot(rel, pipe_axis), pipe_axis)
        radial = rel - axial
        radial_distances = np.linalg.norm(radial, axis=1)
        return float(np.min(radial_distances - self.__pipe_radius))

    @staticmethod
    def __extract_points_in_cylinder(
        points: np.ndarray,
        cylinder_start_point: np.ndarray | tuple[float, float, float],
        cylinder_axis: np.ndarray | tuple[float, float, float],
        radius: float,
        height_range: list[float] | tuple[float, float],
    ) -> np.ndarray:
        """유한 실린더 영역 내부의 점들을 추출한다.

        Args:
            points: 입력 점군 `(N, 3)`.
            cylinder_start_point: 실린더 축의 기준점.
            cylinder_axis: 실린더 축 방향 벡터.
            radius: 실린더 반지름.
            height_range: 기준점에서 축 방향으로 허용할 높이 범위 `[min, max]`.

        Returns:
            실린더 영역 안에 포함된 점 배열.
        """
        # 실린더 축을 단위 벡터로 정규화하고 기준점을 배열로 변환한다.
        axis = np.asarray(cylinder_axis)
        axis = axis / np.linalg.norm(axis)
        start = np.asarray(cylinder_start_point)
        # 각 점을 실린더 축에 투영한다.
        vec = points - start
        proj = np.dot(vec, axis)

        # 높이 범위와 반지름 조건을 모두 만족하는 점만 선택한다.
        mask_height = (proj >= height_range[0]) & (proj <= height_range[1])
        radial = vec - np.outer(proj, axis)
        mask_radius = np.linalg.norm(radial, axis=1) <= radius
        mask = mask_height & mask_radius

        return points[mask]

    @staticmethod
    def __extract_points_in_sphere(
        points: np.ndarray,
        sphere_center: np.ndarray | tuple,
        radius: float,
    ) -> np.ndarray:
        """구 중심과 반지름을 기준으로 내부 점들을 추출한다.

        Args:
            points: 입력 점군 `(N, 3)`.
            sphere_center: 구 중심 좌표.
            radius: 구 반지름.

        Returns:
            구 내부에 포함된 점 배열.
        """
        # 중심에서 각 점까지의 거리 계산
        vec = points - sphere_center
        dists = np.linalg.norm(vec, axis=1)

        # 반지름 이내의 점만 선택한다.
        mask = dists <= radius

        # 구 내부 점 배열 반환
        return points[mask]

    @staticmethod
    def __cluster_points_along_line(
        points: np.ndarray,
        origin_point_of_line: np.ndarray | tuple,
        direction: np.ndarray | tuple,
        cluster_distance: float = 10.0,
    ) -> list[list[np.ndarray]]:
        """점들을 기준선 방향 투영 거리로 정렬한 뒤 가까운 점끼리 클러스터링한다."""
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            return []

        origin = np.asarray(origin_point_of_line, dtype=float).reshape(3)
        axis = np.asarray(direction, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            raise ValueError("direction must be non-zero")
        axis = axis / axis_norm

        shifted_points = points - origin
        proj_points = np.dot(shifted_points, axis)

        sort_idx = np.argsort(proj_points)
        proj_sorted = proj_points[sort_idx]
        points_sorted = points[sort_idx]

        clusters: list[list[np.ndarray]] = []
        current_cluster = [points_sorted[0]]
        for i in range(1, len(points_sorted)):
            if abs(float(proj_sorted[i] - proj_sorted[i - 1])) <= float(cluster_distance):
                current_cluster.append(points_sorted[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [points_sorted[i]]

        if current_cluster:
            clusters.append(current_cluster)
        return clusters

    def __check_collision(
        self,
        link_model: o3d.geometry.TriangleMesh,
        tcp_pose: np.ndarray,
        tcp_to_link_pose_T: np.ndarray,
        margin: float = 0.05,
        sample_count: int = 5000,
    ) -> bool:
        """배관 점군과 엔드이펙터 mesh의 충돌 여부를 검사한다.

        Args:
            link_model: 충돌 검사에 사용할 링크 TriangleMesh.
            tcp_pose: TCP world pose `[x, y, z, roll, pitch, yaw]`.
            tcp_to_link_pose_T: TCP 좌표계에서 링크 mesh 좌표계로 가는 변환 행렬.
            margin: AABB crop에 추가할 여유 거리.
            sample_count: mesh 표면 샘플링 점 개수.

        Returns:
            충돌하면 True, 충돌하지 않으면 False.
        """
        # TCP pose와 TCP-to-link 변환으로 link mesh world pose를 계산한다.
        tcp_pose_T = np.eye(4)
        tcp_pose_T[:3, :3] = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
        tcp_pose_T[:3, 3] = tcp_pose[:3]

        link_pose_T = tcp_pose_T @ tcp_to_link_pose_T

        mesh_copy = copy.deepcopy(link_model)
        mesh_copy.transform(link_pose_T)  # type: ignore

        # mesh AABB를 만들고 margin만큼 확장해 가까운 배관 점만 crop한다.
        aabb = mesh_copy.get_axis_aligned_bounding_box()

        # 배관 반지름/형상 오차를 고려하기 위한 AABB 여유값
        margin_vec = np.array([margin, margin, margin])
        min_b = aabb.min_bound - margin_vec
        max_b = aabb.max_bound + margin_vec
        crop_box = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)  # type: ignore

        idx = crop_box.get_point_indices_within_bounding_box(self._scan_data.points)
        if not idx:
            return False
        sub_pcd = self._scan_data.select_by_index(idx)

        # 엔드이펙터 mesh 표면 샘플링
        mesh_pcd = mesh_copy.sample_points_uniformly(number_of_points=sample_count)

        # 가까운 배관 점과 mesh 표면 사이 최단거리가 threshold 이하면 충돌로 판단한다.
        distances = sub_pcd.compute_point_cloud_distance(mesh_pcd)
        threshold = 0.001
        return any(d <= threshold for d in distances)
