import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud  # type: ignore
import numpy as np
from numpy.typing import NDArray
from CylinderFitting import fit_cylinder
from urdf_parser_py.urdf import URDF
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import json
import copy
import math
from typing import Any


class EndEffectorPoseOptimizer:
    _scan_data: PointCloud

    # dda 정보
    __dda_mesh: o3d.geometry.TriangleMesh
    __dda_invers_transform_mat: np.ndarray

    # rt 정보
    __rt_mesh: o3d.geometry.TriangleMesh
    __rt_invers_transform_mat: np.ndarray

    # 파이프 프로파일 정보
    __pipe_direction: np.ndarray
    __pipe_center: np.ndarray
    __pipe_radius: float

    # 디버깅용
    __is_debug_mode: bool
    debuging_info: dict[str, Any]

    def __init__(self, debug_mode: bool = False):
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
        # 데이터 로드
        self._scan_data = o3d.io.read_point_cloud(file_path)  # type: ignore

        # 스케일
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
        # urdf 파일 로드---------------------------------------------------------
        urdf: URDF = URDF.from_xml_file(file_path)

        # 엔드이펙터 형상 추출-----------------------------------------------------
        # 형상 파일 경로
        end_geometry_file_path = urdf.link_map[end_link_name].collision.geometry.filename
        end_geometry_file_path = Path(str(end_geometry_file_path).replace("file://", ""))
        if not end_geometry_file_path.is_absolute():
            end_geometry_file_path = Path(file_path).resolve().parent / end_geometry_file_path
        end_geometry_file_path = end_geometry_file_path.resolve()

        print(end_geometry_file_path)

        link_mesh = o3d.io.read_triangle_mesh(end_geometry_file_path)

        # 형상 스케일
        end_geomtry_scale = urdf.link_map[end_link_name].collision.geometry.scale
        if isinstance(end_geomtry_scale, list):
            end_geomtry_scale = float(end_geomtry_scale[0])
        elif isinstance(end_geomtry_scale, (int, float)):
            end_geomtry_scale = float(end_geomtry_scale)
        else:
            raise ValueError("엔드이펙터 형상 스케일 정보가 잘못되었습니다.")

        link_mesh = link_mesh.scale(end_geomtry_scale, np.zeros(3, dtype=np.float64))  # type: ignore

        # 자세 변환
        end_pose_xyz = urdf.link_map[end_link_name].collision.origin.xyz
        end_pose_rpy = urdf.link_map[end_link_name].collision.origin.rpy
        T = np.eye(4)
        T[:3, :3] = R.from_euler("xyz", end_pose_rpy).as_matrix()
        T[:3, 3] = end_pose_xyz
        link_mesh = link_mesh.transform(T)  # type: ignore

        if pose_to_link_offset is not None:
            return link_mesh, self.__offset_to_transform(pose_to_link_offset)

        # tcp와 엔드이펙터 형상 위치관계 정보 추출-----------------------------------
        # end to tcp 정보 추출
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

        # tcp to end 변환 행렬 계산
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
        """용접부 탐색을 위한 DDA 자세 후보 계산.

        DDA 자세 후보 조건:
            - TCP의 X축이 배관 중심을 향함
            - TCP의 Y축이 배관 길이 방향과 평행
            - 배관 표면에서 distance 거리에 위치
            - 배관과 충돌하지 않음

        Args:
            target_point: 직배관 표면 위의 한 점.
            num_candidates: 계산할 자세 후보의 수(자세별 간격은 등간격). Defaults to 8.
            distance: 배관 표면으로부터의 거리. Defaults to 0.3.

        Returns:
            tuple: DDA 자세 후보를 3가지 형태로 반환.
                - JSON str: [{dda: [x,y,z,r,p,y]}, ...]
                - filtered array: 충돌 체크를 통과한 자세 후보들
                - all candidates array: 모든 자세 후보들
        """
        # DDA 자세 후보 생성------------------------------------------------------
        candidate_radius = self.__dda_candidate_centerline_radius(
            distance,
            self.__dda_mesh if distance_reference_mesh is None else distance_reference_mesh,
        )
        dda_tcp_pose_candidates = self.__calculate_dda_pose_candidate(
            np.asarray(target_point),
            candidate_radius,
            num_candidates,
        )

        # 배관과 충돌하는 후보 제거------------------------------------------------
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

        # 출력------------------------------------------------------------------
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
        """DDA 자세를 배관 중심축 기준으로 회전시킴.

        Args:
            dda_pose: 원본 DDA 자세 [x, y, z, roll, pitch, yaw].
            rotation_angle_deg: 회전 각도 (도). Defaults to 90.0.

        Returns:
            np.ndarray: 회전된 DDA 자세 [x, y, z, roll, pitch, yaw].
        """
        # 배관 중심축 단위 벡터
        pipe_axis_unit = self.__pipe_direction / np.linalg.norm(self.__pipe_direction)

        # DDA 위치를 배관 중심축 기준으로 회전
        dda_position = dda_pose[:3]

        # 배관 축 위에 DDA 위치를 투영하여 회전 중심 계산
        vec_to_dda = dda_position - self.__pipe_center
        proj_len = np.dot(vec_to_dda, pipe_axis_unit)
        rotation_center = self.__pipe_center + proj_len * pipe_axis_unit

        # 회전 중심에서 DDA까지의 벡터
        radius_vector = dda_position - rotation_center

        # 로드리게스 회전 공식으로 위치 회전
        cos_angle = np.cos(np.radians(rotation_angle_deg))
        sin_angle = np.sin(np.radians(rotation_angle_deg))

        k_cross_v = np.cross(pipe_axis_unit, radius_vector)
        k_dot_v = np.dot(pipe_axis_unit, radius_vector)

        rotated_radius_vector = (
            radius_vector * cos_angle + k_cross_v * sin_angle + pipe_axis_unit * k_dot_v * (1 - cos_angle)
        )

        rotated_position = rotation_center + rotated_radius_vector

        # DDA 자세(회전)도 같은 각도만큼 회전
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
        """주어진 DDA 자세와 각도에 대해 RT 자세 계산.

        RT 자세 조건:
            - RT source의 -Y축이 DDA TCP와 배관을 향함
            - DDA TCP와 RT TCP 간 거리는 distance_from_dda_to_rt
            - DDA에서 RT로 향하는 방향은 DDA X축을 DDA Z축 기준으로 angle_deg만큼 회전한 방향
            - RT TCP의 Y축은 DDA Z축과 같게 두어 같은 배관 단면 기준을 공유함

        Args:
            dda_tcp_pose: DDA TCP 자세 [x, y, z, roll, pitch, yaw].
            angle_deg: DDA에서 RT로 향하는 방향을 DDA X축에서 벌리는 각도 (도).
            distance_from_dda_to_rt: DDA TCP와 RT TCP 사이의 거리 (m).

        Returns:
            np.ndarray: RT TCP 자세 [x, y, z, roll, pitch, yaw].
        """
        # [DEBUG] 입력값 출력
        if self.__is_debug_mode:
            print(f"\n{'='*60}")
            print(f"[DEBUG] __calculate_rt_pose_for_angle 호출")
            print(f"  - dda_tcp_pose: {dda_tcp_pose}")
            print(f"  - angle_deg: {angle_deg}")
            print(f"  - distance_from_dda_to_rt: {distance_from_dda_to_rt}")

        # DDA TCP 좌표계에서 회전 행렬 추출
        dda_rot_matrix = R.from_euler("xyz", dda_tcp_pose[3:]).as_matrix()
        dda_x_axis = dda_rot_matrix[:, 0]  # DDA TCP X축
        dda_y_axis = dda_rot_matrix[:, 1]  # DDA TCP Y축
        dda_z_axis = dda_rot_matrix[:, 2]  # DDA TCP Z축
        dda_pipe_facing_axis = dda_rot_matrix @ self.__dda_pipe_facing_axis
        dda_pipe_facing_axis = dda_pipe_facing_axis / np.linalg.norm(dda_pipe_facing_axis)

        # [DEBUG] DDA 좌표계 축 출력
        if self.__is_debug_mode:
            print(f"  - dda_x_axis: {dda_x_axis}, norm: {np.linalg.norm(dda_x_axis)}")
            print(f"  - dda_y_axis: {dda_y_axis}, norm: {np.linalg.norm(dda_y_axis)}")
            print(f"  - dda_z_axis: {dda_z_axis}, norm: {np.linalg.norm(dda_z_axis)}")

        # DDA TCP의 Z축 단위 벡터 (XY 평면의 법선)
        dda_z_axis_unit = dda_z_axis / np.linalg.norm(dda_z_axis)

        # Rotate the configured DDA pipe-facing axis instead of hard-coded DDA +X.
        # 이 회전된 방향이 RT가 배치될 방향 (DDA에서 RT로 향하는 방향)
        cos_angle = np.cos(np.radians(angle_deg))
        sin_angle = np.sin(np.radians(angle_deg))

        k_cross_v = np.cross(dda_z_axis_unit, dda_pipe_facing_axis)
        k_dot_v = np.dot(dda_z_axis_unit, dda_pipe_facing_axis)

        # DDA에서 RT로 향하는 방향 (DDA pipe-facing axis를 angle_deg만큼 회전)
        dda_to_rt_direction = (
            dda_pipe_facing_axis * cos_angle
            + k_cross_v * sin_angle
            + dda_z_axis_unit * k_dot_v * (1 - cos_angle)
        )

        # [DEBUG] DDA to RT 방향 출력
        if self.__is_debug_mode:
            print(f"  - dda_to_rt_direction: {dda_to_rt_direction}, norm: {np.linalg.norm(dda_to_rt_direction)}")
            print(f"  - dda_pipe_facing_axis(world): {dda_pipe_facing_axis}")

        # RT TCP 위치: DDA TCP에서 회전된 방향으로 distance_from_dda_to_rt만큼 떨어진 위치
        rt_front_extent = self.__rt_candidate_origin_distance_offset()
        adjusted_distance_from_dda_to_rt = float(distance_from_dda_to_rt + rt_front_extent)
        # Push RT pose origin out so the source mesh front, not the origin, satisfies clearance.
        rt_position = dda_tcp_pose[:3] + dda_to_rt_direction * adjusted_distance_from_dda_to_rt
        if self.__is_debug_mode:
            self.debuging_info["rt_mesh_front_extent_along_facing_axis"] = float(rt_front_extent)
            self.debuging_info["rt_adjusted_distance_from_dda_to_rt"] = adjusted_distance_from_dda_to_rt

        # RT local axis chosen in config points from the RT pose origin toward the pipe.
        rt_rot_matrix = self.__rotation_from_pipe_facing_axis(
            pipe_facing_world=-dda_to_rt_direction,
            world_up_hint=dda_z_axis_unit,
            local_pipe_facing_axis=self.__rt_pipe_facing_axis,
        )
        rt_x_axis = rt_rot_matrix[:, 0]
        rt_y_axis = rt_rot_matrix[:, 1]
        rt_z_axis = rt_rot_matrix[:, 2]

        # [DEBUG] RT X축 출력
        if self.__is_debug_mode:
            print(f"  - rt_x_axis: {rt_x_axis}, norm: {np.linalg.norm(rt_x_axis)}")
            print(f"  - rt_pipe_facing_axis(local): {self.__rt_pipe_facing_axis}")

        # [DEBUG] RT Y, Z 축 출력 및 직교 여부 확인
        if self.__is_debug_mode:
            dot_xy = np.dot(rt_x_axis, rt_y_axis)
            dot_xz = np.dot(rt_x_axis, rt_z_axis)
            dot_yz = np.dot(rt_y_axis, rt_z_axis)
            dot_dda_y_rt_y = np.dot(dda_y_axis, rt_y_axis)
            print(f"  - rt_y_axis: {rt_y_axis}, norm: {np.linalg.norm(rt_y_axis)}")
            print(f"  - rt_z_axis: {rt_z_axis}, norm: {np.linalg.norm(rt_z_axis)}")
            print(f"  - dot(rt_x, rt_y): {dot_xy} (0에 가까워야 직교)")
            print(f"  - dot(rt_x, rt_z): {dot_xz} (0에 가까워야 직교)")
            print(f"  - dot(rt_y, rt_z): {dot_yz} (0에 가까워야 직교)")
            print(f"  - dot(dda_y, rt_y): {dot_dda_y_rt_y} (±1이면 평행)")

        # 회전 행렬의 유효성 검사
        det = np.linalg.det(rt_rot_matrix)

        # [DEBUG] 회전 행렬 및 행렬식 출력
        if self.__is_debug_mode:
            print(f"  - rt_rot_matrix:\n{rt_rot_matrix}")
            print(f"  - det(rt_rot_matrix): {det}")

        if det < 0:
            # 좌수 좌표계인 경우 X축만 뒤집어 -Z축의 시선 방향은 유지한다.
            rt_x_axis = -rt_x_axis
            rt_rot_matrix = np.column_stack([rt_x_axis, rt_y_axis, rt_z_axis])
            if self.__is_debug_mode:
                print(f"  - det < 0이므로 X축 반전 적용")

        rt_rpy = R.from_matrix(rt_rot_matrix).as_euler("xyz")

        # [DEBUG] 최종 결과 출력
        if self.__is_debug_mode:
            print(f"  - rt_rpy (결과): {rt_rpy}")
            print(f"{'='*60}\n")

        # RT TCP 자세 [x, y, z, roll, pitch, yaw]
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

    def calculate_DDA_RT_pose_for_taking_xray(
        self,
        target_point: tuple[float, float, float] | np.ndarray,
        num_candidates: int,
        distance_from_dda_to_surface: float,
        distance_from_dda_to_rt: float,
        angle_of_rt: float,
        distance_reference_mesh: o3d.geometry.TriangleMesh | None = None,
    ):
        """x-ray 촬영을 위한 DDA, RT 자세 후보 계산.

        DDA 자세 후보 조건:
            - DDA TCP의 X축이 배관 중심을 향함
            - DDA TCP의 Y축이 배관 길이 방향과 평행
            - 배관 표면에서 distance_from_dda_to_surface 거리에 위치
            - 배관과 충돌하지 않음
            - 원본 자세(0도)와 배관 중심축 기준 90도 회전 자세 모두 검사

        RT 자세 후보 조건:
            - RT source의 -Y축이 DDA TCP와 배관을 향함
            - DDA TCP와 RT TCP 간 거리는 distance_from_dda_to_rt
            - DDA에서 RT로 향하는 방향은 DDA X축을 DDA Z축 기준으로 ±angle_of_rt만큼 회전한 방향
            - RT TCP의 Y축은 DDA Z축과 같게 두어 같은 배관 단면 기준을 공유함
            - 배관과 충돌하지 않음

        Args:
            target_point: 직배관 표면 위의 한 점.
            num_candidates: 계산할 자세 후보의 수(자세별 간격은 등간격).
            distance_from_dda_to_surface: DDA TCP와 배관 표면 사이의 거리 (m).
            distance_from_dda_to_rt: DDA TCP와 RT TCP 사이의 거리 (m).
            angle_of_rt: DDA에서 RT로 향하는 방향을 DDA X축에서 벌리는 각도 (degree).

        Returns:
            tuple: DDA-RT 자세 그룹을 2가지 형태로 반환.
                - JSON str 형식: 그룹화된 DDA-RT 자세 쌍
                - dict 형식: 그룹화된 DDA-RT 자세 쌍
        """
        if self.__is_debug_mode:
            self.debuging_info = {}

        # DDA 자세 후보 생성------------------------------------------------------
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

        if self.__is_debug_mode:
            self.debuging_info["dda_base_candidates"] = dda_base_candidates

        # 배관과 충돌하지 않는 DDA 기본 자세만 필터링---------------------------------
        valid_base_dda_poses = []
        base_dda_collision_count = 0
        for dda_pose in dda_base_candidates:
            is_collision = self.__check_collision(
                self.__dda_mesh,
                dda_pose,
                self.__dda_invers_transform_mat,
            )
            if not is_collision:
                valid_base_dda_poses.append(dda_pose)
            else:
                base_dda_collision_count += 1

        if self.__is_debug_mode:
            self.debuging_info["valid_base_dda_poses"] = valid_base_dda_poses
            self.debuging_info["base_dda_collision_count"] = base_dda_collision_count

        # DDA-RT 자세 그룹 생성---------------------------------------------------
        pose_groups = []
        rotated_dda_collision_count = 0
        collision_pose_groups = []  # 충돌하는 자세 그룹을 따로 저장

        for base_dda_pose in valid_base_dda_poses:
            group_data = {}

            # 0도 (원본 자세) 처리
            group_0_data = self.__process_dda_rt_combination(base_dda_pose, angle_of_rt, distance_from_dda_to_rt)
            if group_0_data:
                group_data["0"] = group_0_data

            # 90도 회전 자세 처리
            rotated_dda_pose = self.__rotate_dda_pose_around_pipe_axis(base_dda_pose, 90.0)

            # 90도 회전된 DDA 자세의 충돌 검사
            is_rotated_dda_collision = self.__check_collision(
                self.__dda_mesh,
                rotated_dda_pose,
                self.__dda_invers_transform_mat,
            )

            if not is_rotated_dda_collision:
                group_90_data = self.__process_dda_rt_combination(
                    rotated_dda_pose, angle_of_rt, distance_from_dda_to_rt
                )
                if group_90_data:
                    group_data["90"] = group_90_data
            else:
                rotated_dda_collision_count += 1

            # "0"과 "90" 모두 유효할 때만 그룹에 추가, 그렇지 않으면 충돌 그룹에 추가
            if "0" in group_data and "90" in group_data:
                pose_groups.append(group_data)
            else:
                # 부분적으로라도 데이터가 있으면 충돌 그룹에 저장
                if self.__is_debug_mode:
                    if group_data:
                        collision_pose_groups.append(group_data)

        # 디버그 모드일 때 충돌 그룹 정보 저장
        if self.__is_debug_mode:
            self.debuging_info["collision_pose_groups"] = collision_pose_groups
            self.debuging_info["complete_pose_group_count"] = len(pose_groups)
            self.debuging_info["partial_pose_group_count"] = len(collision_pose_groups)
            self.debuging_info["rotated_dda_collision_count"] = rotated_dda_collision_count
            rejected_groups = self.debuging_info.get("rejected_pose_groups", [])
            self.debuging_info["rejected_pose_group_count"] = len(rejected_groups)
            self.debuging_info["rt1_collision_count"] = sum(
                1 for item in rejected_groups if item.get("rejected", {}).get("_rt1_collision", False)
            )
            self.debuging_info["rt2_collision_count"] = sum(
                1 for item in rejected_groups if item.get("rejected", {}).get("_rt2_collision", False)
            )

        if not pose_groups and collision_pose_groups:
            pose_groups = collision_pose_groups
            if self.__is_debug_mode:
                self.debuging_info["used_partial_pose_group_fallback"] = True

        # JSON 형태 출력 생성-----------------------------------------------------
        pose_groups_json = json.dumps(pose_groups)

        return pose_groups_json, pose_groups

    def calculate_DDA_RT_pose_for_taking_xray_indexed_0_90(
        self,
        target_point: tuple[float, float, float] | np.ndarray,
        num_candidates: int,
        distance_from_dda_to_surface: float,
        distance_from_dda_to_rt: float,
        angle_of_rt: float,
        distance_reference_mesh: o3d.geometry.TriangleMesh | None = None,
    ):
        """x-ray 촬영을 위한 0°/90° DDA-RT 자세 후보 계산.

        기존 ``calculate_DDA_RT_pose_for_taking_xray`` 와 같은 0°/90° 쌍을
        찾되, DDA 자세를 매번 90° 회전시키지 않는다. 배관 둘레 후보를 먼저
        모두 만들고 각 후보의 DDA/RT 충돌 여부를 한 번씩만 계산한 뒤,
        90° 떨어진 후보 index를 참조해 쌍을 구성한다.

        ``num_candidates`` 는 90°가 정확히 후보 index에 매핑되도록 4의 배수여야
        한다. 예를 들어 8개 후보이면 90° offset은 2 index이다.
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
                group_data["0"]["_actual_deg"] = int(round(idx * step_deg))
                group_data["90"]["_actual_deg"] = int(round(idx_90 * step_deg))
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
    ) -> tuple[str, list[dict]]:
        """x-ray 촬영을 위한 DDA, RT 자세 3-쌍 (120° 간격) 조합 계산.

        배관 둘레에 candidate_step_deg 간격으로 후보 자세를 생성하고,
        인접 간격이 모두 |gap - 120°| ≤ gap_tolerance_deg인 충돌-자유 3-조합
        중 편차 합이 최소인 1개를 반환한다.

        3-조합이 존재하지 않고 allow_2pair_fallback=True이면 두 후보 사이의
        호 간격이 |gap - 120°| ≤ gap_tolerance_deg인 2-조합 중 편차 최소
        1개를 반환한다.

        Args:
            target_point: 직배관 표면 위의 한 점.
            distance_from_dda_to_surface: DDA TCP와 배관 표면 사이의 거리 (m).
            distance_from_dda_to_rt: DDA TCP와 RT TCP 사이의 거리 (m).
            angle_of_rt: RT TCP X축과 DDA TCP X축 사이의 각도 (degree).
            candidate_step_deg: 배관 둘레 후보 생성 간격 (degree). Defaults to 3.0.
                num_candidates는 int(round(360 / candidate_step_deg))로 결정됨.
            gap_tolerance_deg: 인접 간격의 이상값 120°에서 허용 편차 (degree).
                Defaults to 10.0. 박스 제약: 모든 인접 간격이 [120-tol, 120+tol] 안.
            allow_2pair_fallback: 3-조합 불가 시 2-쌍 폴백 활성화 여부.
                Defaults to True.

        Returns:
            tuple[str, list[dict]]: (json_str, pose_groups)
                pose_groups는 0개 또는 1개의 그룹을 담은 리스트.
                그룹 안의 키는 회전각 문자열 ("0", "120", "240" 또는 폴백 시 "0", "120"),
                값 슬롯 구조는 기존 90° 함수와 동일 {"DDA":[...], "RT1":[...], "RT2":[...]}에
                추가 메타 `_actual_deg: int` (실측 양자화 각도).
                폴백 시에는 추가로 `_arc_deg: int` (채택된 호의 실측 각도)도 두 슬롯 모두에 추가.
        """
        # 입력 검증 (방어적 가드, Security review L-2) --------------------------
        # NaN/inf로 인한 산술/논리 비교 무력화 방지 및 박스 제약(tol < 60°) 강제.
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

        # 후보 자세 생성 -----------------------------------------------------
        num_candidates = int(round(360.0 / candidate_step_deg))
        step_deg = 360.0 / num_candidates  # 보정된 실제 step (정수 N 보장)

        candidate_radius = self.__dda_candidate_centerline_radius(
            distance_from_dda_to_surface,
            self.__dda_mesh if distance_reference_mesh is None else distance_reference_mesh,
        )
        dda_base_candidates = self.__calculate_dda_pose_candidate(
            np.asarray(target_point),
            candidate_radius,
            num_candidates,
        )

        # jkpark 각 인덱스별 슬롯 결과 (None = 무효, dict = 유효) ----------------------
        dda_base_candidates = self.__adjust_dda_candidates_for_mesh_surface_distance(
            dda_base_candidates,
            distance_from_dda_to_surface,
            self.__dda_mesh if distance_reference_mesh is None else distance_reference_mesh,
        )

        slot_results: list[dict | None] = []
        for dda_pose in dda_base_candidates:
            if self.__check_collision(self.__dda_mesh, dda_pose, self.__dda_invers_transform_mat):
                slot_results.append(None)
                continue
            slot = self.__process_dda_rt_combination(dda_pose, angle_of_rt, distance_from_dda_to_rt)
            slot_results.append(slot)  # __process_dda_rt_combination이 None 반환 가능

        # enumerate 순서가 곧 정렬 순서이므로 i < j < k가 자연 보장.
        valid_indices = sorted(i for i, s in enumerate(slot_results) if s is not None)
        valid_set = set(valid_indices)

        # 3-조합 탐색 + 편차 최소 선택 ----------------------------------------
        # 부동소수 정밀도 안전을 위해 작은 epsilon을 ceil/floor에 적용.
        EPS = 1e-9
        ideal_idx_gap = num_candidates / 3.0
        tol_idx = gap_tolerance_deg / step_deg
        min_gap = int(np.ceil(ideal_idx_gap - tol_idx - EPS))
        max_gap = int(np.floor(ideal_idx_gap + tol_idx + EPS))

        best_triple: tuple[int, int, int] | None = None
        best_deviation_sum: float = float("inf")

        # i < j < k 정렬 순회로 회전대칭 중복 자동 제거.
        # gap3은 닫힌 호 (k → wrap → i) 길이.
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
                    # 박스 제약 사후 재확인 가드 (부동소수 안전, 두 번째 방어선).
                    ang_gaps = (gap1 * step_deg, gap2 * step_deg, gap3 * step_deg)
                    if any(abs(ag - 120.0) > gap_tolerance_deg + EPS for ag in ang_gaps):
                        continue
                    dev = sum(abs(ag - 120.0) for ag in ang_gaps)
                    if best_triple is None \
                            or dev < best_deviation_sum \
                            or (dev == best_deviation_sum and (i, j, k) < best_triple):
                        best_deviation_sum = dev
                        best_triple = (i, j, k)

        # 3-조합 결과 패키징 (이상 라벨 + 실측 메타) ---------------------------
        if best_triple is not None:
            pose_groups: list[dict] = [{}]
            group = pose_groups[0]
            for idx, ideal_label in zip(best_triple, ("0", "120", "240")):
                slot = dict(slot_results[idx])  # type: ignore[arg-type]  # 얕은 복사로 원본 보존
                slot["_actual_deg"] = int(round(idx * step_deg))
                group[ideal_label] = slot
            return json.dumps(pose_groups), pose_groups

        # 2-쌍 폴백 탐색 -----------------------------------------------------
        if not allow_2pair_fallback:
            return "[]", []

        best_pair: tuple[int, int] | None = None
        best_pair_deviation: float = float("inf")
        best_pair_arc_deg: int = 0
        for i in valid_indices:
            for j in valid_indices:
                if j <= i:
                    continue
                gap_deg = (j - i) * step_deg
                other_deg = 360.0 - gap_deg
                # 두 호 중 [120-tol, 120+tol]에 들어가는 쪽 채택.
                # 가정 (tol < 60°): 두 호의 합은 360°. 두 호 모두 [120-tol, 120+tol]에
                # 들어가려면 tol ≥ 60°가 필요하므로 기본 tol=10°에서는 서로 배타적.
                # tol ≥ 60° 호출은 spec 범위 밖이며 그 경우 if/elif에 의해
                # 짧은 쪽(더 작은 j-i)이 우선 선택됨.
                if abs(gap_deg - 120.0) <= gap_tolerance_deg:
                    chosen_dev = abs(gap_deg - 120.0)
                    chosen_arc = int(round(gap_deg))
                elif abs(other_deg - 120.0) <= gap_tolerance_deg:
                    chosen_dev = abs(other_deg - 120.0)
                    chosen_arc = int(round(other_deg))
                else:
                    continue
                if best_pair is None \
                        or chosen_dev < best_pair_deviation \
                        or (chosen_dev == best_pair_deviation and (i, j) < best_pair):
                    best_pair_deviation = chosen_dev
                    best_pair = (i, j)
                    best_pair_arc_deg = chosen_arc

        if best_pair is None:
            return "[]", []

        pose_groups = [{}]
        group = pose_groups[0]
        for idx, ideal_label in zip(best_pair, ("0", "120")):
            slot = dict(slot_results[idx])  # type: ignore[arg-type]
            slot["_actual_deg"] = int(round(idx * step_deg))
            slot["_arc_deg"] = best_pair_arc_deg
            group[ideal_label] = slot
        return json.dumps(pose_groups), pose_groups

    def __process_dda_rt_slot_with_collision(
        self,
        dda_pose: np.ndarray,
        angle_of_rt: float,
        distance_from_dda_to_rt: float,
    ) -> dict[str, list[float]] | None:
        """DDA 충돌까지 포함해 한 후보 slot의 DDA/RT 유효성을 검사."""
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
        """DDA 자세에 대해 RT1(+angle), RT2(-angle) 조합 처리.

        Args:
            dda_pose: DDA TCP 자세.
            angle_of_rt: RT 배치 각도.
            distance_from_dda_to_rt: DDA-RT 간 거리.

        Returns:
            dict | None: 유효한 RT 자세가 있으면 DDA-RT 조합 딕셔너리, 없으면 None.
        """
        result = {"DDA": dda_pose.tolist()}

        # RT1 (+angle) 계산 및 충돌 검사
        rt1_pose = self.__calculate_rt_pose_for_angle(dda_pose, angle_of_rt, distance_from_dda_to_rt)
        is_rt1_collision = self.__check_collision(
            self.__rt_mesh,
            rt1_pose,
            self.__rt_invers_transform_mat,
        )

        if not is_rt1_collision:
            result["RT1"] = rt1_pose.tolist()

        # RT2 (-angle) 계산 및 충돌 검사
        rt2_pose = self.__calculate_rt_pose_for_angle(dda_pose, -angle_of_rt, distance_from_dda_to_rt)
        is_rt2_collision = self.__check_collision(
            self.__rt_mesh,
            rt2_pose,
            self.__rt_invers_transform_mat,
        )

        if not is_rt2_collision:
            result["RT2"] = rt2_pose.tolist()

        # RT1이나 RT2 중 하나라도 유효하면 결과 반환
        if "RT1" in result or "RT2" in result:
            return result
        else:
            if self.__is_debug_mode:
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
        """직배관의 프로파일(방향벡터, 중심점, 반지름) 계산하여 멤버변수에 저장.

        Args:
            target_point: 직배관 표면 위의 한 점.
            sampling_size_for_calculating_normal: 법선 계산을 위한 샘플링 크기. Defaults to 0.01.
            radius_offset_for_sampling_points_in_sphere: 구 샘플링을 위한 반지름 오프셋. Defaults to 0.003.
        """

        if self.__is_debug_mode:
            self.debuging_info = {}

        # 검사 대상 주변 미소 점군 추출---------------------------------------------
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
                "target_point 주변에 점군이 없습니다. target_pont 또는 sampling_size_for_calculating_normal 값을 조절하세요."
            )

        if self.__is_debug_mode:
            self.debuging_info["selected_points"] = selected_points

        # 중앙 벡터 계산----------------------------------------------------------
        normals = np.asarray(selected_points.normals)
        x_m = np.median(normals[:, 0])
        y_m = np.median(normals[:, 1])
        z_m = np.median(normals[:, 2])
        normal_m = np.array([x_m, y_m, z_m])

        if self.__is_debug_mode:
            self.debuging_info["normal_m"] = normal_m

        # 직경 추정--------------------------------------------------------------
        # 가늘고 긴 실린더 ROI 생성 후 내부 점 추출
        points_in_cylinder = self.__extract_points_in_cylinder(
            np.asarray(self._scan_data.points),
            target_point,
            normal_m * -1,  # 법선 벡터의 반대 방향
            sampling_cylinder_radius,  # 배관 지름에 따라 조절 필요
            sampling_cylinder_height_range,  # 배관 직경 및 브랜치 간 거리에 따라 조절 필요
        )

        if self.__is_debug_mode:
            self.debuging_info["points_in_cylinder"] = points_in_cylinder
            self.debuging_info["pipe_profile_sampling_cylinder"] = {
                "start": target_point,
                "axis": normal_m * -1,
                "radius": sampling_cylinder_radius,
                "height_range": sampling_cylinder_height_range,
            }

        # 중앙 벡터에 투영 후 군집화
        clusters = self.__cluster_points_along_line(
            points_in_cylinder,
            target_point,
            normal_m * -1,
            sampling_cylinder_radius,  # 점군 밀도에 따라 조절 필요
        )
        self.debuging_info["pipe_profile_clusters"] = clusters
        self.debuging_info["pipe_profile_points_in_cylinder"] = points_in_cylinder
        self.debuging_info["pipe_profile_target_point"] = target_point
        self.debuging_info["pipe_profile_normal_axis"] = normal_m * -1

        # 가장 먼 군집에서 가장 먼 점의 거리
        estimated_opposite_point = clusters[1][-1]
        estimated_center = (target_point + estimated_opposite_point) / 2
        estimated_radius = float(np.linalg.norm(estimated_opposite_point - estimated_center))

        if self.__is_debug_mode:
            self.debuging_info["estimated_center"] = estimated_center
            self.debuging_info["estimated_radius"] = estimated_radius

        # 배관 중심에서 배관 점군 추출----------------------------------------------
        # 배관 중심점에서 반지름 + α 범위 내의 점 추출
        points_in_sphere = self.__extract_points_in_sphere(
            np.asarray(self._scan_data.points),
            estimated_center,
            estimated_radius + radius_offset_for_sampling_points_in_sphere,  # 배관 지름에 따라 조절 필요
        )

        # 실린더 피팅------------------------------------------------------------
        if self.__is_debug_mode:
            self.debuging_info["points_in_sphere"] = points_in_sphere

        direction, center, radius, _ = fit_cylinder(points_in_sphere)

        # 멤버변수에 파이프 프로파일 저장-------------------------------------------
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
    ):
        """배관 중심에서 radius만큼 떨어지고, 배관 중심을 바라보는 DDA의 위치 및 방향 후보 계산.

        Args:
            point_on_pipe_surface: 직배관 표면 위의 한 점.
            radius: 직배관 중심으로부터의 거리.
            num_candidates: 계산할 자세 후보의 수(자세별 간격은 등간격).

        Returns:
            np.ndarray: 각 행이 [x, y, z, roll, pitch, yaw] 형태인 numpy array of shape (num_candidates, 6).
        """

        # 동적 중심 계산: surface point를 pipe 축 위에 투영
        # pipe_direction 단위 벡터로 정규화
        direction_unit = self.__pipe_direction / np.linalg.norm(self.__pipe_direction)
        vec_to_surface = point_on_pipe_surface - self.__pipe_center
        proj_len = np.dot(vec_to_surface, direction_unit)
        center = self.__pipe_center + proj_len * direction_unit

        # 배관 축에 수직인 벡터 2개 구하기------------------------------------------
        # 배관 축에 평행하지 않는 기준 벡터 선택(x축 or y축)
        basis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(basis, self.__pipe_direction)) > 0.9:
            basis = np.array([0.0, 1.0, 0.0])

        # 수직 벡터 생성
        v1 = np.cross(self.__pipe_direction, basis)  # 배관 축에 수직인 벡터 v1
        v1 /= np.linalg.norm(v1)  # 길이로 나눠서 방향 벡터 계산
        v2 = np.cross(self.__pipe_direction, v1)  # 배관 축에 수직인 벡터 v2
        v2 /= np.linalg.norm(v2)

        # 위치 계산--------------------------------------------------------------
        # 반지름이 1인 원 위의 점 좌표 계산. 원 공식 (cos θ, sin θ)
        angles = 2 * np.pi * np.arange(num_candidates) / num_candidates
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        # v1, v2는 배관 축에 수직인 벡터, 위에서 구한 원 위의 점 좌표를 원점이 (0,0,0)이고 v1, v2로 구성된 평면위로 이동
        offsets = np.outer(cos_a, v1) + np.outer(sin_a, v2)

        # 투영된 중심 주변 원형 궤도상 위치 계산
        positions = center + offsets * radius

        # Direction pose: configured DDA local axis faces the pipe center.
        pipe_direction_unit = self.__pipe_direction / np.linalg.norm(self.__pipe_direction)
        facing_axes = center - positions
        facing_norm = np.linalg.norm(facing_axes, axis=1, keepdims=True)
        facing_norm[facing_norm < 1e-12] = 1.0
        facing_axes = facing_axes / facing_norm

        rot_mats = np.stack([
            self.__rotation_from_pipe_facing_axis(
                pipe_facing_world=facing_axis,
                world_up_hint=pipe_direction_unit,
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
            self.debuging_info["dda_configured_facing_dot_pipe_center"] = (
                np.sum(configured_facing_world * facing_axes, axis=1).tolist()
            )
            self.debuging_info["dda_minus_y_dot_pipe_center"] = (
                np.sum(local_minus_y_world * facing_axes, axis=1).tolist()
            )
        rpy_array = R.from_matrix(rot_mats).as_euler("xyz", degrees=False)

        # 출력 포맷 설정----------------------------------------------------------
        # 각 행이 [x, y, z, roll, pitch, yaw] 형태인 numpy array of shape (num_candidates, 6)
        poses = np.hstack((positions, rpy_array))
        return poses

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
        if self.__collision_checker is not None:
            return bool(self.__collision_checker(
                link_model,
                tcp_pose,
                tcp_to_link_pose_T,
                margin=margin,
                sample_count=sample_count,
            ))

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
        points: np.ndarray,  # 점군
        cylinder_start_point: np.ndarray | tuple[float, float, float],  # 실린더 시작점
        cylinder_axis: np.ndarray | tuple[float, float, float],  # 실린더 축 (단위벡터)
        radius: float,  # 실린더 반지름
        height_range: list[float] | tuple[float, float],  # 실린더 높이 범위 [min, max]
    ) -> np.ndarray:
        """실린더 내부에 있는 점들을 추출.

        Args:
            points: 점군.
            cylinder_start_point: 실린더 시작점.
            cylinder_axis: 실린더 축 (단위벡터).
            radius: 실린더 반지름.
            height_range: 실린더 높이 범위 [min, max].

        Returns:
            np.ndarray: 실린더 내부에 포함되는 점들.
        """
        # 실린더 축 단위 벡터로 정규화 및 시작점 배열로 변환
        axis = np.asarray(cylinder_axis)
        axis = axis / np.linalg.norm(axis)
        start = np.asarray(cylinder_start_point)
        # 직선에 점군 투영 (proj: 점에서 start까지의 축 성분 거리)
        vec = points - start
        proj = np.dot(vec, axis)

        # 실린더의 높이와 반지름에 대한 마스크 생성
        mask_height = (proj >= height_range[0]) & (proj <= height_range[1])
        radial = vec - np.outer(proj, axis)
        mask_radius = np.linalg.norm(radial, axis=1) <= radius
        mask = mask_height & mask_radius

        # 마스크에 해당하는 점군 반환
        return points[mask]

    @staticmethod
    def __extract_points_in_sphere(
        points: np.ndarray, sphere_center: np.ndarray | tuple, radius: float  # 점군  # 구의 중심점  # 구의 반지름
    ) -> np.ndarray:
        """구 내부에 있는 점들을 추출.

        Args:
            points: 점군.
            sphere_center: 구의 중심점.
            radius: 구의 반지름.

        Returns:
            np.ndarray: 구 내부에 포함되는 점들.
        """
        # 구에 점군 투영----------------------------------------------------------
        vec = points - sphere_center
        dists = np.linalg.norm(vec, axis=1)

        # 구의 반지름에 대한 마스크 생성--------------------------------------------
        mask = dists <= radius

        # 마스크에 해당하는 점군 반환-----------------------------------------------
        return points[mask]

    @staticmethod
    def __cluster_points_along_line(
        points: np.ndarray,  # 스캔 데이터의 일부
        origin_point_of_line: np.ndarray | tuple,  # 직선의 한 점
        direction: np.ndarray | tuple,  # 직선의 방향벡터 (단위벡터)
        # min_distance: float = 5,  # position으로부터 최소 거리,
        cluster_distance: float = 10,  # 군집화 기준 거리(투영값 기준)
    ) -> list[list[np.ndarray]]:
        """직선을 따라 점들을 거리 기준으로 군집화.

        Args:
            points: 스캔 데이터의 일부.
            origin_point_of_line: 직선의 한 점.
            direction: 직선의 방향벡터 (단위벡터).
            cluster_distance: 군집화 기준 거리(투영값 기준). Defaults to 10.

        Returns:
            list[list[np.ndarray]]: 군집화된 점들의 리스트.
        """
        # 군집화 사전 준비--------------------------------------------------------
        # 투영
        shifted_points = points - origin_point_of_line  # 투영하기 위해 원점으로 이동
        proj_points = np.dot(shifted_points, direction)  # 각 점의 position으로부터의 투영값(스칼라)
        # projected_points = np.outer(proj, direction) + position  # 직선 위 투영점

        # min_distance보다 가까운 점 제외
        # mask = proj_points > min_distance
        # proj_points = proj_points[mask]
        # points = points[mask]
        # projected_points = projected_points[mask]

        # proj 기준 정렬
        sort_idx = np.argsort(proj_points)
        proj_sorted = proj_points[sort_idx]
        points_sorted = points[sort_idx]

        # 군집화: proj 값이 cluster_distance 이내면 같은 군집-----------------------
        clusters: list[list[np.ndarray]] = []
        if len(points_sorted) == 0:
            return clusters
        # 첫번째 클러스터에 첫번째 점 추가
        current_cluster = [points_sorted[0]]

        for i in range(1, len(points_sorted)):
            # 이전 점과 거리가 기준 이하이면 클러스터에 추가
            if abs(proj_sorted[i] - proj_sorted[i - 1]) <= cluster_distance:
                current_cluster.append(points_sorted[i])

            # 이전 점과 거리가 기준 이상이면 새로운 클러스터 시작
            else:
                clusters.append(current_cluster)
                current_cluster = [points_sorted[i]]

        # 마지막 클러스터 추가
        if current_cluster:
            clusters.append(current_cluster)
        # ----------------------------------------------------------------------
        return clusters

    def __check_collision(
        self,
        link_model: o3d.geometry.TriangleMesh,
        tcp_pose: np.ndarray,
        tcp_to_link_pose_T: np.ndarray,
        margin: float = 0.05,
        sample_count: int = 5000,
    ) -> bool:
        """메쉬(변환된)와 로드된 스캔 점군 데이터 간 충돌 여부 검사.

        Args:
            link_model: 검사할 TriangleMesh.
            tcp_pose: TCP 자세 array(6) [x, y, z, roll, pitch, yaw] (라디안).
            tcp_to_link_pose_T: TCP에서 링크로의 변환 행렬.
            margin: 충돌 검사 마진. Defaults to 0.05.
            sample_count: 메쉬 샘플링 점 수. Defaults to 5000.

        Returns:
            bool: 충돌 시 True.
        """
        # 엔드이펙터를 검사 대상 위치로 이동-----------------------------------------
        tcp_pose_T = np.eye(4)
        tcp_pose_T[:3, :3] = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
        tcp_pose_T[:3, 3] = tcp_pose[:3]

        link_pose_T = tcp_pose_T @ tcp_to_link_pose_T

        mesh_copy = copy.deepcopy(link_model)
        mesh_copy.transform(link_pose_T)  # type: ignore

        # 연산량을 줄이기 위해 스캔 데이터 필터링-------------------------------------
        # 엔드이펙터의 바운딩 박스 계산
        aabb = mesh_copy.get_axis_aligned_bounding_box()

        # 바운딩 박스에 마진 추가
        margin_vec = np.array([margin, margin, margin])
        min_b = aabb.min_bound - margin_vec
        max_b = aabb.max_bound + margin_vec
        crop_box = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)  # type: ignore

        # 바운딩 박스 내 점 추출
        idx = crop_box.get_point_indices_within_bounding_box(self._scan_data.points)
        if not idx:
            return False
        sub_pcd = self._scan_data.select_by_index(idx)

        # 엔드이펙터 표면 점 추출--------------------------------------------------
        mesh_pcd = mesh_copy.sample_points_uniformly(number_of_points=sample_count)

        # 점들간 거리 계산으로 충돌 여부 확인----------------------------------------
        distances = sub_pcd.compute_point_cloud_distance(mesh_pcd)
        threshold = 0.001
        return any(d <= threshold for d in distances)
