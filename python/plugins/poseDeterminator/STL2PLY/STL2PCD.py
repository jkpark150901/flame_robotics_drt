from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class MeshInfo:
    vertices: int
    triangles: int
    surface_area_mm2: float
    aabb_min: tuple
    aabb_max: tuple


class STLPointCloudGenerator:
    """Open3D를 사용하여 STL 메쉬에서 균등 분포된 포인트 클라우드를 생성합니다.

    기능:
    - 사용자 지정 배율 인자로 STL을 로드합니다.
    - 샘플링 및 법선 계산에 필요한 필수 메쉬 정보를 추출합니다.
    - 균일 면적 밀도 또는 목표 간격(mm)으로 표면 포인트를 생성합니다.
    - 메쉬(삼각형 법선) 기반으로 샘플 포인트에 법선을 할당합니다.
    - 파일 확장자에 따라 이진 PLY 또는 PCD 형식으로 결과를 저장합니다.

    참고:
    - STL 제공 삼각형 법선이 있으면 사용하며, 없으면 계산으로 보완합니다.
    - 샘플링 방법은 안정적이고 빠른 균일 면적 샘플링이 기본값입니다.
    - 간격 제어는 입력 간격(mm)을 표면적을 통해 목표 포인트 수로 변환하여 구현합니다.
    - 또는 모델 크기에 독립적인 밀도를 위해 (cm^2당 포인트) 밀도를 지정할 수 있습니다.
    """

    def __init__(self) -> None:
        self._stl_mesh: o3d.geometry.TriangleMesh | None = None
        self._pcd: o3d.geometry.PointCloud | None = None
        self._mesh: o3d.geometry.TriangleMesh | None = None  # 메시 구성용

    # ---------------------- 메쉬 I/O 및 정보 ----------------------
    def load_stl(
        self,
        path: str | Path,
        scale_factor: float = 1.0,
    ):
        """STL 파일을 로드하고 사용자 지정 배율 인자를 적용합니다.

        매개변수:
        - path: STL 파일 경로입니다.
        - scale_factor: 로드된 메쉬 좌표에 곱해질 배율 인자입니다.

        반환값:
        - self를 반환합니다.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"STL 파일을 찾을 수 없습니다: {p}")

        mesh = o3d.io.read_triangle_mesh(str(p))  # type: ignore
        if mesh is None or len(mesh.triangles) == 0:
            raise ValueError(f"유효한 삼각형 메쉬를 로드하지 못했습니다: {p}")

        # 다른 단위 변환 없이 사용자 정의 배율을 적용합니다.
        if scale_factor <= 0:
            raise ValueError("scale_factor는 0보다 커야 합니다")
        if scale_factor != 1.0:
            # 원점을 기준으로 스케일링하여 상대 위치를 보존합니다; 필요 시 호출자가 변환을 제어할 수 있습니다.
            mesh.scale(scale_factor, center=(0.0, 0.0, 0.0))  # type: ignore

        # 삼각형 법선이 있는지 확인합니다; 가능하면 STL 제공 법선을 사용합니다.
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()

        self._stl_mesh = mesh
        self._pcd = None  # 이전 샘플링 초기화

    def mesh_info(self) -> MeshInfo:
        """샘플링 및 법선 계산에 필요한 필수 메쉬 정보를 반환합니다.
        주의: STL 메시 정보는 사용하지 않음."""
        # STL 메시 정보 사용 금지로 인해 기본값 반환
        return MeshInfo(
            vertices=0,
            triangles=0,
            surface_area_mm2=0.0,
            aabb_min=(0.0, 0.0, 0.0),
            aabb_max=(0.0, 0.0, 0.0),
        )

    def construct_mesh_from_pcd(
        self,
        method: str = "ball_pivoting",
        ball_radii: list[float] | None = None,
        k_neighbors: int = 10,
        alpha: float = 0.03,
    ) -> None:
        """self._pcd가 존재하는 경우 포인트 클라우드를 기반으로 메시를 구성하고 self._mesh에 저장합니다.

        다양한 메시 생성 알고리즘을 지원합니다.

        매개변수:
        - method: 메시 생성 방법 ('ball_pivoting', 'poisson', 'alpha_shape', 'knn')
        - ball_radii: Ball Pivoting 알고리즘의 구 반지름 리스트 (기본값: 자동 계산)
        - k_neighbors: KNN 방법의 최근접 이웃 개수 (기본값: 10)
        - alpha: Alpha shapes 매개변수 (기본값: 0.03)
        """
        pcd = self._require_pcd()
        points = np.asarray(pcd.points)
        n_points = len(points)

        if n_points < 3:
            raise RuntimeError("메시 생성을 위해서는 최소 3개의 포인트가 필요합니다")

        # 법선이 없으면 추정
        if not pcd.has_normals():
            pcd.estimate_normals()

        try:
            mesh = None
            method = method.lower()

            if method == "ball_pivoting":
                # 자동으로 적절한 반지름 계산
                distances = pcd.compute_nearest_neighbor_distance()
                distances_array = np.asarray(distances)
                avg_dist = float(np.mean(distances_array))
                ball_radii = [avg_dist * 1.5, avg_dist * 2.0, avg_dist * 3.0]

                radii_vector = o3d.utility.DoubleVector(ball_radii)
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_vector)

            elif method == "poisson":
                # Poisson Surface Reconstruction
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=11)

            elif method == "alpha_shape":
                # Alpha Shapes
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)

            else:
                raise ValueError(f"지원하지 않는 메시 생성 방법: {method}")

            # 메시 검증 및 저장
            if mesh is None or len(mesh.triangles) == 0:
                # 기본 방법이 실패하면 대안 시도
                if method != "alpha_shape":
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)

                if mesh is None or len(mesh.triangles) == 0:
                    raise RuntimeError("포인트 클라우드에서 유효한 메시를 생성하지 못했습니다")

            # 삼각형 법선 계산
            mesh.compute_triangle_normals()
            self._mesh = mesh

        except Exception as e:
            raise RuntimeError(f"메시 구성 중 오류 발생: {e}")

    def sample_points(
        self,
        spacing_mm: float | None = None,
        density_per_cm2: float | None = None,
        method: str = "uniform",
    ):
        """메쉬 표면에서 포인트를 샘플링합니다.

        spacing_mm(표면상의 목표 간격(mm)) 또는 density_per_cm2(cm^2당 포인트 수) 중 하나를 제공합니다.
        샘플러는 목표 포인트 수를 계산한 후 면적 균일 샘플링을 수행합니다.

        매개변수:
        - spacing_mm: 표면상의 인접 포인트 간 희망 간격(mm, 대략치)입니다.
        - density_per_cm2: 면적당 포인트 수(cm^2당). 제공 시 spacing_mm을 대체합니다.
        - method: 'uniform'(기본) 또는 'poisson'. 'uniform'은 삼각형 법선을 직접 할당할 수 있습니다.

        반환값:
        - self를 반환합니다.
        """
        if self._stl_mesh:
            mesh = self._stl_mesh
        else:
            raise RuntimeError("메쉬가 로드되지 않았습니다. load_stl()을 먼저 호출하세요.")

        area_mm2 = float(mesh.get_surface_area())

        n_points = self._compute_target_points(area_mm2, spacing_mm, density_per_cm2)

        # 면적 균일 샘플링을 사용합니다; 메쉬의 삼각형 법선을 직접 사용할 수 있습니다.
        method = (method or "uniform").lower()
        if method not in {"uniform", "poisson"}:
            raise ValueError("method는 'uniform' 또는 'poisson'이어야 합니다")

        if method == "uniform":
            # 삼각형 법선이 있으면 직접 사용합니다.
            pcd = mesh.sample_points_uniformly(number_of_points=int(n_points), use_triangle_normal=True)
        else:
            # 보다 균일한 간격을 위해 포아송 디스크 샘플링을 사용합니다; 이후 도우미 균일 샘플로 법선을 매핑합니다.
            pcd = mesh.sample_points_poisson_disk(number_of_points=int(n_points))

            # STL 삼각형 법선이 있으면 균일 샘플에서 최근접 이웃으로 법선을 전달합니다.
            if mesh.has_triangle_normals():
                helper = mesh.sample_points_uniformly(
                    number_of_points=max(1024, int(0.5 * n_points)), use_triangle_normal=True
                )
                if len(helper.points) > 0:
                    # 도우미 포인트에 KD트리를 구성하고 최근접 이웃의 법선을 복사합니다.
                    tree = o3d.geometry.KDTreeFlann(helper)
                    normals = []
                    for pt in pcd.points:
                        _, idx, _ = tree.search_knn_vector_3d(pt, 1)
                        normals.append(helper.normals[idx[0]])
                    pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals))
            # 그렇지 않으면 법선을 비워 둡니다; 필요 시 호출자가 계산할 수 있습니다.

        self._pcd = pcd

    def save_point_cloud(self, path: str | Path, include_mesh: bool = False) -> None:
        """포인트 클라우드를 파일 확장자에 따라 이진 PLY 또는 PCD 형식으로 지정된 경로에 저장합니다.

        매개변수:
        - path: 저장할 파일 경로
        - include_mesh: PLY 파일의 경우 메시 정보도 함께 저장할지 여부
        """
        pcd = self._require_pcd()
        p = Path(path)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        ext = p.suffix.lower()
        if ext not in {".ply", ".pcd"}:
            raise ValueError("출력 확장자는 .ply 또는 .pcd 여야 합니다")

        if ext == ".ply" and include_mesh:
            # PLY 파일에 메시 정보와 점군 정보를 함께 저장
            self._save_ply_with_mesh(p)
        else:
            # 기본 포인트 클라우드만 저장
            success = o3d.io.write_point_cloud(str(p), pcd, write_ascii=False)  # type: ignore
            if not success:
                raise IOError(f"포인트 클라우드를 쓰지 못했습니다: {p}")

    def _save_ply_with_mesh(self, path: Path) -> None:
        """PLY 파일에 메시 정보와 점군 정보를 함께 저장합니다."""
        pcd = self._require_pcd()

        # 메시가 없으면 포인트 클라우드에서 구성
        if self._stl_mesh is None:
            self.construct_mesh_from_pcd()

        if self._mesh is None:
            raise RuntimeError("메시가 구성되지 않았습니다. construct_mesh_from_pcd()를 먼저 호출하세요.")
        mesh = self._mesh

        # 포인트 클라우드의 점들을 메시에 추가 (색상 정보가 있는 경우)
        combined_vertices = []
        combined_colors = []

        # 메시의 정점들 추가
        mesh_vertices = np.asarray(mesh.vertices)
        combined_vertices.append(mesh_vertices)

        # 포인트 클라우드의 점들 추가 (메시 정점과 중복되지 않는 경우)
        pcd_points = np.asarray(pcd.points)
        if len(pcd_points) > 0:
            combined_vertices.append(pcd_points)

        # 색상 정보가 있으면 추가
        if pcd.has_colors():
            pcd_colors = np.asarray(pcd.colors)
            # 메시 정점에 대한 기본 색상 (회색)
            mesh_colors = np.full((len(mesh_vertices), 3), 0.5)
            combined_colors = np.vstack([mesh_colors, pcd_colors])

        # 결합된 정점으로 새로운 메시 생성
        combined_mesh = o3d.geometry.TriangleMesh()
        combined_mesh.vertices = o3d.utility.Vector3dVector(np.vstack(combined_vertices))
        combined_mesh.triangles = mesh.triangles

        if len(combined_colors) > 0:
            combined_mesh.vertex_colors = o3d.utility.Vector3dVector(combined_colors)

        # PLY 파일로 저장
        success = o3d.io.write_triangle_mesh(path, combined_mesh, write_ascii=False)  # type: ignore
        if not success:
            raise IOError(f"메시를 포함한 PLY 파일을 쓰지 못했습니다: {path}")

    def get_point_cloud(self) -> o3d.geometry.PointCloud:
        """Open3D 포인트 클라우드 객체를 반환합니다(법선 포함)."""
        return self._require_pcd()

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray | None]:
        """(points[N,3], normals[N,3] 또는 None)을 반환합니다."""
        pcd = self._require_pcd()
        pts = np.asarray(pcd.points)
        nrm = np.asarray(pcd.normals) if pcd.has_normals() else None
        return pts, nrm

    def _require_pcd(self) -> o3d.geometry.PointCloud:
        if self._pcd is None:
            raise RuntimeError("포인트가 샘플링되지 않았습니다. sample_points()를 먼저 호출하세요.")
        return self._pcd

    @staticmethod
    def _compute_target_points(
        area_mm2: float,
        spacing_mm: float | None,
        density_per_cm2: float | None,
    ) -> int:
        # 둘 다 동시에 제공되는 것은 허용하지 않습니다.
        if spacing_mm is not None and density_per_cm2 is not None:
            raise ValueError("spacing_mm과 density_per_cm2는 동시에 제공될 수 없습니다. 둘 중 하나만 지정하세요.")

        if density_per_cm2 is not None:
            if density_per_cm2 <= 0:
                raise ValueError("density_per_cm2는 0보다 커야 합니다")
            # 1 cm^2 = 100 mm^2
            points = area_mm2 * (float(density_per_cm2) / 100.0)
        elif spacing_mm is not None:
            if spacing_mm <= 0:
                raise ValueError("spacing_mm은 0보다 커야 합니다")
            # 근사: 간격^2 면적당 하나의 포인트
            points = area_mm2 / float(spacing_mm) ** 2
        else:
            raise ValueError("spacing_mm 또는 density_per_cm2 중 하나를 제공해야 합니다")

        points_i = max(1, int(round(points)))
        return points_i
    
    def save_mesh(self, path: str | Path) -> None:
        """메시를 파일 확장자에 따라 이진 PLY 형식으로 지정된 경로에 저장합니다.

        매개변수:
            path (str | Path): 저장할 파일의 경로
        """
        if self._mesh is None:
            raise RuntimeError("메시가 구성되지 않았습니다. construct_mesh_from_pcd()를 먼저 호출하세요.")

        success = o3d.io.write_triangle_mesh(path, self._mesh, write_ascii=False)  # type: ignore
        if not success:
            raise IOError(f"메시를 쓰지 못했습니다: {path}")
