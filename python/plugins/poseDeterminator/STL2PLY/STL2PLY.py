import numpy as np
import open3d as o3d
import trimesh
from typing import Optional, Tuple
import os


class STL2PLY:
    """
    STL 파일을 PLY 파일로 변환하는 클래스

    기능:
    - STL 파일 로드 및 스케일링
    - 메시 및 점군 재생성
    - 사용자 정의 메시 밀도로 PLY 파일 생성
    """

    def __init__(self):
        """STL2PLY 클래스 초기화"""
        self.mesh = None
        self.point_cloud = None
        self.original_mesh = None

    def load_stl(self, stl_path: str, scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> bool:
        """
        STL 파일을 로드하고 스케일링을 적용합니다.

        Args:
            stl_path (str): STL 파일 경로
            scale (Tuple[float, float, float]): X, Y, Z 축 스케일링 팩터

        Returns:
            bool: 로드 성공 여부
        """
        try:
            if not os.path.exists(stl_path):
                print(f"Error: STL 파일을 찾을 수 없습니다: {stl_path}")
                return False

            # Open3D를 사용하여 STL 파일 로드
            self.mesh = o3d.io.read_triangle_mesh(stl_path) # type: ignore

            if len(self.mesh.vertices) == 0:
                print("Error: STL 파일이 비어있거나 유효하지 않습니다.")
                return False

            # 원본 메시 백업
            self.original_mesh = o3d.geometry.TriangleMesh(self.mesh)

            # 스케일링 적용
            if scale != (1.0, 1.0, 1.0):
                scale_matrix = np.diag([scale[0], scale[1], scale[2]])
                self.mesh.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices) @ scale_matrix.T)

            # 메시 정규화 및 법선 계산
            self.mesh.compute_vertex_normals()
            self.mesh.compute_triangle_normals()

            print(f"STL 파일 로드 완료: {len(self.mesh.vertices)}개 정점, {len(self.mesh.triangles)}개 삼각형")
            print(f"스케일링 적용: X={scale[0]}, Y={scale[1]}, Z={scale[2]}")

            return True

        except Exception as e:
            print(f"STL 파일 로드 중 오류 발생: {e}")
            return False

    def regenerate_mesh(self, target_edge_length: float = 0.01) -> bool:
        """
        메시를 재생성합니다. 사용자가 원하는 메시 밀도에 맞춰 재샘플링합니다.

        Args:
            target_edge_length (float): 목표 엣지 길이 (작을수록 더 조밀한 메시)

        Returns:
            bool: 재생성 성공 여부
        """
        try:
            if self.mesh is None:
                print("Error: 먼저 STL 파일을 로드해주세요.")
                return False

            # 현재 평균 엣지 길이 계산
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)

            edge_lengths = []
            for triangle in triangles:
                v0, v1, v2 = vertices[triangle]
                edge_lengths.extend([np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v1), np.linalg.norm(v0 - v2)])

            current_avg_edge = np.mean(edge_lengths)
            print(f"현재 평균 엣지 길이: {current_avg_edge:.6f}")
            print(f"목표 엣지 길이: {target_edge_length:.6f}")

            # trimesh를 사용하여 메시 리메싱
            trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

            # 목표 엣지 길이에 따라 세분화 또는 단순화
            if target_edge_length < current_avg_edge:
                # 세분화 (더 조밀하게)
                subdivision_iterations = int(np.log2(current_avg_edge / target_edge_length))
                subdivision_iterations = min(subdivision_iterations, 3)  # 최대 3번 세분화

                for i in range(subdivision_iterations):
                    trimesh_mesh = trimesh_mesh.subdivide()
                    print(f"세분화 {i+1}/{subdivision_iterations} 완료")

            elif target_edge_length > current_avg_edge * 1.5:
                # 단순화 (더 성기게)
                target_faces = int(len(triangles) * (current_avg_edge / target_edge_length) ** 2)
                target_faces = max(target_faces, 100)  # 최소 100개 면

                trimesh_mesh = trimesh_mesh.simplify_quadric_decimation(target_faces)
                print(f"메시 단순화 완료: {len(triangles)} -> {len(trimesh_mesh.faces)} 면")

            # Open3D 메시로 변환
            self.mesh = o3d.geometry.TriangleMesh()
            self.mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
            self.mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)

            # 법선 재계산
            self.mesh.compute_vertex_normals()
            self.mesh.compute_triangle_normals()

            print(f"메시 재생성 완료: {len(self.mesh.vertices)}개 정점, {len(self.mesh.triangles)}개 삼각형")
            return True

        except Exception as e:
            print(f"메시 재생성 중 오류 발생: {e}")
            return False

    def generate_point_cloud(self, num_points: int = 10000) -> bool:
        """
        메시로부터 점군을 생성합니다.

        Args:
            num_points (int): 생성할 점의 개수

        Returns:
            bool: 점군 생성 성공 여부
        """
        try:
            if self.mesh is None:
                print("Error: 먼저 STL 파일을 로드해주세요.")
                return False

            # 메시 표면에서 점 샘플링
            self.point_cloud = self.mesh.sample_points_uniformly(number_of_points=num_points)

            # 점군 색상 설정 (선택사항)
            colors = np.tile([0.7, 0.7, 0.7], (num_points, 1))  # 회색
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

            print(f"점군 생성 완료: {len(self.point_cloud.points)}개 점")
            return True

        except Exception as e:
            print(f"점군 생성 중 오류 발생: {e}")
            return False

    def save_mesh_ply(self, output_path: str) -> bool:
        """
        메시를 PLY 파일로 저장합니다.

        Args:
            output_path (str): 출력 PLY 파일 경로

        Returns:
            bool: 저장 성공 여부
        """
        try:
            if self.mesh is None:
                print("Error: 저장할 메시가 없습니다.")
                return False

            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # PLY 파일로 저장
            success = o3d.io.write_triangle_mesh(output_path, self.mesh) # type: ignore

            if success:
                print(f"메시 PLY 파일 저장 완료: {output_path}")
                return True
            else:
                print(f"메시 PLY 파일 저장 실패: {output_path}")
                return False

        except Exception as e:
            print(f"메시 저장 중 오류 발생: {e}")
            return False

    def save_pointcloud_ply(self, output_path: str) -> bool:
        """
        점군을 PLY 파일로 저장합니다.

        Args:
            output_path (str): 출력 PLY 파일 경로

        Returns:
            bool: 저장 성공 여부
        """
        try:
            if self.point_cloud is None:
                print("Error: 저장할 점군이 없습니다.")
                return False

            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # PLY 파일로 저장
            success = o3d.io.write_point_cloud(output_path, self.point_cloud) # type: ignore

            if success:
                print(f"점군 PLY 파일 저장 완료: {output_path}")
                return True
            else:
                print(f"점군 PLY 파일 저장 실패: {output_path}")
                return False

        except Exception as e:
            print(f"점군 저장 중 오류 발생: {e}")
            return False

    def convert_stl_to_ply(
        self,
        stl_path: str,
        ply_path: str,
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_edge_length: float = 0.01,
        generate_pointcloud: bool = True,
        num_points: int = 10000,
    ) -> bool:
        """
        STL 파일을 PLY 파일로 변환하는 통합 메서드

        Args:
            stl_path (str): 입력 STL 파일 경로
            ply_path (str): 출력 PLY 파일 경로
            scale (Tuple[float, float, float]): 스케일링 팩터
            target_edge_length (float): 목표 메시 엣지 길이
            generate_pointcloud (bool): 점군 생성 여부
            num_points (int): 점군 점 개수

        Returns:
            bool: 변환 성공 여부
        """
        try:
            print(f"STL to PLY 변환 시작...")
            print(f"입력 파일: {stl_path}")
            print(f"출력 파일: {ply_path}")

            # 1. STL 파일 로드 및 스케일링
            if not self.load_stl(stl_path, scale):
                return False

            # 2. 메시 재생성
            if not self.regenerate_mesh(target_edge_length):
                return False

            # 3. 메시 PLY 저장
            mesh_ply_path = ply_path.replace(".ply", "_mesh.ply")
            if not self.save_mesh_ply(mesh_ply_path):
                return False

            # 4. 점군 생성 및 저장 (선택사항)
            if generate_pointcloud:
                if self.generate_point_cloud(num_points):
                    pointcloud_ply_path = ply_path.replace(".ply", "_pointcloud.ply")
                    self.save_pointcloud_ply(pointcloud_ply_path)

            print("STL to PLY 변환 완료!")
            return True

        except Exception as e:
            print(f"변환 중 오류 발생: {e}")
            return False

    def get_mesh_info(self) -> dict:
        """
        현재 메시의 정보를 반환합니다.

        Returns:
            dict: 메시 정보
        """
        if self.mesh is None:
            return {"error": "로드된 메시가 없습니다."}

        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)

        # 바운딩 박스 계산
        min_bound = vertices.min(axis=0)
        max_bound = vertices.max(axis=0)
        size = max_bound - min_bound

        # 평균 엣지 길이 계산
        edge_lengths = []
        for triangle in triangles:
            v0, v1, v2 = vertices[triangle]
            edge_lengths.extend([np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v1), np.linalg.norm(v0 - v2)])

        # 볼륨 계산 (메시가 watertight하지 않을 수 있음)
        try:
            volume = self.mesh.get_volume()
        except Exception:
            volume = "N/A (메시가 완전히 닫혀있지 않음)"

        return {
            "vertices": len(vertices),
            "triangles": len(triangles),
            "min_bound": min_bound.tolist(),
            "max_bound": max_bound.tolist(),
            "size": size.tolist(),
            "avg_edge_length": np.mean(edge_lengths),
            "volume": volume,
        }


# 사용 예시
if __name__ == "__main__":
    # STL2PLY 인스턴스 생성
    converter = STL2PLY()

    # 예시: STL 파일을 PLY로 변환
    # converter.convert_stl_to_ply(
    #     stl_path="input.stl",
    #     ply_path="output.ply",
    #     scale=(2.0, 2.0, 2.0),  # 2배 확대
    #     target_edge_length=0.005,  # 더 조밀한 메시
    #     generate_pointcloud=True,
    #     num_points=20000
    # )

    print("STL2PLY 클래스가 준비되었습니다.")
