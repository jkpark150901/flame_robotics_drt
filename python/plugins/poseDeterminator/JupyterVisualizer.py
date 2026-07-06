import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud  # type: ignore
from open3d.cpu.pybind.utility import Vector3dVector  # type: ignore
import numpy as np
import pyvista as pv
from pyvista.plotting.plotter import Plotter
from ipywidgets import Textarea, Layout
from IPython.display import display
from vtk import vtkPointPicker


def visualize_pointcloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    point_size: int = 3,
    bg_color: str | tuple = ("white", "gray"),
    plotter: Plotter | None = None,
) -> Plotter:
    pv_data = pv.PolyData(points)

    scalars = None
    rgb = False
    if colors is not None and len(points) == len(colors):
        pv_data["colors"] = colors
        scalars = "colors"
        rgb = True

    # Determine the grid layout based on the camera_pos
    # grid_layout = (1, 1) if camera_pos is None or len(camera_pos) == 1 else (int(len(camera_pos) / 2 + 0.5), 2)

    # Create a plotter with the specified grid layout
    if plotter is not None:
        pl = plotter
    else:
        pl = Plotter()

    if isinstance(bg_color, tuple):
        if len(bg_color) == 2:
            pl.set_background(bg_color[0], top=bg_color[1])
        else:
            pl.set_background(bg_color[0])
    else:
        pl.set_background(bg_color)  # type: ignore

    if pv_data is not None:
        pl.add_mesh(pv_data, scalars=scalars, rgb=rgb, point_size=point_size)
    pl.camera.parallel_projection = True

    return pl


def visualize_pointclouds_simply(
    pcd: PointCloud | np.ndarray | Vector3dVector | None = None,
    point_size: int = 3,
    bg_color: str | tuple = ("white", "gray"),
    visualize_normals: bool = False,
    plotter: Plotter | None = None,
):
    if isinstance(pcd, PointCloud):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    elif isinstance(pcd, np.ndarray):
        points = pcd
        colors = None
    elif isinstance(pcd, Vector3dVector):
        points = np.asarray(pcd)
        colors = None
    else:
        raise ValueError("pcd는 Open3D PointCloud, numpy array, 또는 Vector3dVector 타입이어야 합니다.")

    if plotter is not None:
        pl = plotter
    else:
        pl = Plotter()

    visualize_pointcloud(points, colors, point_size, bg_color, pl)

    if visualize_normals:
        if isinstance(pcd, PointCloud):
            pl.add_arrows(np.asarray(pcd.points), np.asarray(pcd.normals))
        else:
            raise ValueError("법선을 가시화하려면 pcd가 Open3D PointCloud 타입이어야 합니다.")

    if plotter is None:
        pl.show()
    return pl


def visualize_selectable_pointcloud(
    pcd: PointCloud | np.ndarray,
    point_size: int = 3,
    camera_pos: list | None = None,
    bg_color: str | tuple = ("white", "gray"),
) -> Plotter:
    # 3D 뷰어 생성---------------------------------------------------------------
    if isinstance(pcd, PointCloud):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    else:
        points = pcd
        colors = None
    pl = visualize_pointcloud(points, colors, point_size, bg_color)

    # 텍스트 출력 위젯 생성--------------------------------------------------------
    output = Textarea(layout=Layout(width="auto", height="80px"))

    def print_selected_point(point: np.ndarray, picker: vtkPointPicker):
        output.value = f"""
XYZ==
{point[0]}, {point[1]}, {point[2]}
ID==
{picker.GetPointId()}
""".strip()

    pl.enable_point_picking(callback=print_selected_point, use_picker=True)

    display(pl.show())
    display(output)

    return pl


def add_sphere(
    plotter: Plotter,
    center: np.ndarray | list | tuple,
    radius: float = 0.1,
    color: str | tuple = "red",
    label: str | None = None,
    label_offset: tuple = (0.1, 0.1, 0.1),
    opacity: float = 1.0,
):
    # center 타입 확인 후 numpy로 변환--------------------------------------------
    if not isinstance(center, np.ndarray):
        if isinstance(center, (list, tuple)) and len(center) == 3:
            center = np.array(center, dtype=float)
        else:
            raise ValueError("center must be a numpy array, or a list/tuple of length 3.")

    # 해당 좌표에 구 형상 추가-----------------------------------------------------
    plotter.add_mesh(pv.Sphere(center=center, radius=radius), color=color, opacity=opacity)

    # 라벨 추가------------------------------------------------------------------
    if label:
        label_position = center + np.array(label_offset)  # 라벨 위치를 구체 중심에서 약간 이동
        plotter.add_point_labels(
            [label_position],  # 라벨 위치
            [label],  # 라벨 텍스트
            point_size=1,
            font_size=12,
            text_color="black",
            always_visible=True,
        )


def add_point_clusters(
    plotter: Plotter,
    clusters: list[list[np.ndarray]] | tuple,
    point_size: int = 10,
    colors: list[str | tuple] | None = None,
) -> Plotter:
    if colors is None:
        colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "yellow",
        ]

    for idx, cluster in enumerate(clusters):
        points = np.asarray(cluster, dtype=float)
        if points.size == 0:
            continue
        if points.ndim == 1:
            points = points.reshape(1, 3)
        pv_data = pv.PolyData(points)
        plotter.add_mesh(
            pv_data,
            color=colors[idx % len(colors)],
            point_size=point_size,
            render_points_as_spheres=True,
        )
    return plotter


def add_cylinder(
    plotter: Plotter,
    center: np.ndarray | list | tuple,
    direction: np.ndarray | list | tuple,
    radius: float,
    height: float,
    color: str | tuple = "blue",
    opacity: float = 1.0,
):
    # center 타입 확인 후 numpy로 변환--------------------------------------------
    if not isinstance(center, np.ndarray):
        if isinstance(center, (list, tuple)) and len(center) == 3:
            center = np.array(center, dtype=float)
        else:
            raise ValueError("center must be a numpy array, or a list/tuple of length 3.")

    # direction 타입 확인 후 numpy로 변환-----------------------------------------
    if not isinstance(direction, np.ndarray):
        if isinstance(direction, (list, tuple)) and len(direction) == 3:
            direction = np.array(direction, dtype=float)
        else:
            raise ValueError("direction must be a numpy array, or a list/tuple of length 3.")

    # 방향 벡터 정규화------------------------------------------------------------
    direction = direction / np.linalg.norm(direction)

    # 해당 좌표에 실린더 형상 추가--------------------------------------------------
    plotter.add_mesh(
        pv.Cylinder(center=center, direction=direction, radius=radius, height=height),
        color=color,
        opacity=opacity,
    )


def calculate_arrows_from_pose_candidates(xyzrpy):
    """Convert an array of shape (n,6) [x,y,z,roll,pitch,yaw] to array of shape (n,6) [x,y,z, dx,dy,dz]."""
    # xyzrpy는 numpy 배열입니다
    xyz = xyzrpy[..., 0:3]
    roll = xyzrpy[..., 3]
    pitch = xyzrpy[..., 4]
    yaw = xyzrpy[..., 5]

    # x축 방향 벡터 컴포넌트 계산
    dx = np.cos(yaw) * np.cos(pitch)
    dy = np.sin(yaw) * np.cos(pitch)
    dz = -np.sin(pitch)

    direction = np.stack([dx, dy, dz], axis=-1)
    # xyz와 방향 벡터 결합
    return np.concatenate([xyz, direction], axis=-1)


def visualize_mesh(
    mesh: o3d.geometry.TriangleMesh,
    color: str | tuple = "lightgray",
    show_edges: bool = True,
    camera_pos: list | None = None,
    bg_color: str | tuple = ("white", "gray"),
    show: bool = True,
) -> Plotter:
    """
    Open3D TriangleMesh를 PyVista로 변환하여 시각화합니다.

    Args:
        mesh (o3d.geometry.TriangleMesh): 메시 모델
        color: 메시 색 또는 컬러 맵
        show_edges: 엣지 표시 여부
        camera_pos: 카메라 위치 (PyVista 형식)
        bg_color: 배경 색 (tuple인 경우 그라데이션)
        show: True이면 즉시 표시
    Returns:
        PyVista Plotter 인스턴스
    """
    # numpy 배열로 변환
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    # faces 배열 생성 (각 삼각형 앞에 정점 수 3 추가)
    faces = np.hstack([np.full((triangles.shape[0], 1), 3, dtype=np.int64), triangles]).astype(np.int64).flatten()
    # PyVista PolyData 생성
    pv_mesh = pv.PolyData(vertices, faces)

    # Plotter 생성 및 배경 설정
    pl = Plotter()
    if isinstance(bg_color, tuple):
        if len(bg_color) == 2:
            pl.set_background(bg_color[0], top=bg_color[1])
        else:
            pl.set_background(bg_color[0])
    else:
        pl.set_background(bg_color)  # type: ignore

    # 메시 추가
    pl.add_mesh(pv_mesh, color=color, show_edges=show_edges)
    pl.camera.parallel_projection = True

    # 카메라 위치 설정
    if camera_pos is not None:
        pl.camera_position = camera_pos

    if show:
        pl.show()
    return pl


def add_mesh(
    plotter: Plotter,
    mesh: o3d.geometry.TriangleMesh,
    pose: np.ndarray,
    color: str | tuple = "lightgray",
    show_edges: bool = True,
) -> Plotter:
    """
    PyVista Plotter에 Open3D TriangleMesh를 자세(pose)와 함께 추가하여 가시화합니다.

    Args:
        plotter: PyVista Plotter 인스턴스
        mesh: o3d.geometry.TriangleMesh 타입 메시 모델
        pose: [x, y, z, roll, pitch, yaw] 형태의 numpy 배열 (각도는 라디안)
        color: 메시 색
        show_edges: 엣지 표시 여부
    Returns:
        Plotter 인스턴스 (입력 plotter)
    """
    # 메시를 numpy 배열로 변환하고 PyVista PolyData 생성----------------------------
    mesh_copy = mesh.__copy__()
    vertices = np.asarray(mesh_copy.vertices)
    triangles = np.asarray(mesh_copy.triangles)
    faces = np.hstack([np.full((triangles.shape[0], 1), 3, dtype=np.int64), triangles]).astype(np.int64).flatten()
    pv_mesh = pv.PolyData(vertices, faces)

    # 자세 적용------------------------------------------------------------------
    pv_mesh.transform(pose, inplace=True)

    # Plotter에 추가-------------------------------------------------------------
    plotter.add_mesh(pv_mesh, color=color, show_edges=show_edges)
    return plotter


def rpy_to_direction(rpy: np.ndarray | tuple) -> np.ndarray:
    """
    np.ndarray 형태의 rpy [roll, pitch, yaw] 값을 받아 3D 방향 벡터로 변환합니다.

    Args:
        rpy (numpy.ndarray): [roll, pitch, yaw] 라디안 단위 배열.

    Returns:
        numpy.ndarray: 길이가 3인 단위 방향 벡터.
    """
    # rpy 배열에서 roll, pitch, yaw 추출
    roll, pitch, yaw = rpy
    # X, Y, Z 축에 대한 회전 행렬
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    # Z-Y-X 순서로 결합된 회전 행렬
    R = Rz @ Ry @ Rx
    # 바디 프레임에서 전방 벡터
    forward = np.array([1.0, 0.0, 0.0])
    # 월드 프레임에서의 방향 벡터
    direction = R @ forward
    return direction


def xyz_direction_to_pose(xyz: tuple | list | np.ndarray, direction: tuple | list | np.ndarray) -> np.ndarray:
    """
    xyz 좌표와 방향 벡터로부터 pose [x, y, z, roll, pitch, yaw]를 계산합니다.

    Args:
        xyz (tuple or list or numpy.ndarray): 위치 좌표 (x, y, z).
        direction (tuple or list or numpy.ndarray): 방향 벡터 (dx, dy, dz).

    Returns:
        numpy.ndarray: 길이가 6인 배열 [x, y, z, roll, pitch, yaw], roll은 0으로 설정됩니다.
    """
    # numpy 배열로 변환
    xyz_arr = np.array(xyz, dtype=float)
    dir_arr = np.array(direction, dtype=float)
    # 방향 벡터 정규화
    dir_norm = dir_arr / np.linalg.norm(dir_arr)
    dx, dy, dz = dir_norm
    # yaw(방위각)와 pitch(피치각) 계산
    yaw = np.arctan2(dy, dx)
    pitch = -np.arcsin(dz)
    roll = 0.0  # roll은 회전 방향 정보로 복원 불가능하여 0으로 설정
    # pose 반환
    return np.array([xyz_arr[0], xyz_arr[1], xyz_arr[2], roll, pitch, yaw], dtype=float)


def add_coordinate_frame(
    plotter: Plotter,
    origin: np.ndarray | list | tuple = (0, 0, 0),
    length: float = 1.0,
    size: float = 1.0,
    colors: tuple = ("red", "green", "blue"),
    opacity: float = 1.0,
    label: bool = True,
    label_offset: float = 0.01,
):
    """
    주어진 PyVista Plotter에 좌표계(X, Y, Z 축)를 추가합니다.

    매개변수:
        plotter: PyVista Plotter 인스턴스.
        origin: 좌표계의 원점 (x, y, z).
        size: 각 축의 길이.
        colors: X, Y, Z 축의 색상 튜플.
        opacity: 축 화살표의 투명도.
        label: 축 레이블 표시 여부.
    label_offset: 축 끝에서 레이블까지의 거리 (축 방향 단위).
    """
    # origin이 numpy 배열인지 확인
    if not isinstance(origin, np.ndarray):
        origin = np.array(origin, dtype=float)

    # 축 방향 정의
    directions = np.eye(3) * length

    # 각 축에 대한 화살표 그리기
    for dir_vec, col, axis_name in zip(directions, colors, ["X", "Y", "Z"]):
        # 방향 벡터 단위화
        norm = np.linalg.norm(dir_vec)
        unit_dir = dir_vec / norm if norm != 0 else dir_vec
        # 기본 화살표 생성 (단위 길이)
        arrow = pv.Arrow(
            direction=unit_dir,
            tip_length=0.2,
            tip_radius=2 * size,
            shaft_radius=size,
        )
        # 원하는 길이만큼 스케일링하고 시작 위치로 이동
        arrow.scale(float(length), inplace=True)
        arrow.translate(origin, inplace=True)
        plotter.add_mesh(arrow, color=col, opacity=opacity)

        # 축 방향으로 레이블 위치 계산 및 추가
        if label:
            # dir_vec = unit_dir * length
            # label_offset: 거리 비율이 아닌 절대 거리
            unit_dir = dir_vec / np.linalg.norm(dir_vec) if np.linalg.norm(dir_vec) != 0 else dir_vec
            label_pos = origin + unit_dir * (length + label_offset)
            plotter.add_point_labels(
                [label_pos],
                [axis_name],
                point_size=1,
                font_size=12,
                text_color=col,
                always_visible=True,
            )


def add_box(
    plotter: Plotter,
    min_bound: np.ndarray | list | tuple,
    max_bound: np.ndarray | list | tuple,
    color: str | tuple = "blue",
    opacity: float = 1.0,
):
    """
    PyVista Plotter에 박스를 추가합니다.

    Args:
        plotter (Plotter): PyVista Plotter 인스턴스.
        min_bound (np.ndarray | list | tuple): 박스의 최소 경계 좌표 [x_min, y_min, z_min].
        max_bound (np.ndarray | list | tuple): 박스의 최대 경계 좌표 [x_max, y_max, z_max].
        color (str | tuple): 박스 색상. 기본값은 "blue".
        opacity (float): 투명도 (0.0~1.0). 기본값은 1.0 (불투명).
    """
    # min_bound 타입 확인 후 numpy로 변환
    if not isinstance(min_bound, np.ndarray):
        if isinstance(min_bound, (list, tuple)) and len(min_bound) == 3:
            min_bound = np.array(min_bound, dtype=float)
        else:
            raise ValueError("min_bound must be a numpy array, or a list/tuple of length 3.")

    # max_bound 타입 확인 후 numpy로 변환
    if not isinstance(max_bound, np.ndarray):
        if isinstance(max_bound, (list, tuple)) and len(max_bound) == 3:
            max_bound = np.array(max_bound, dtype=float)
        else:
            raise ValueError("max_bound must be a numpy array, or a list/tuple of length 3.")

    # PyVista Box 생성 (bounds 형식: [x_min, x_max, y_min, y_max, z_min, z_max])
    bounds = [
        min_bound[0],
        max_bound[0],  # x_min, x_max
        min_bound[1],
        max_bound[1],  # y_min, y_max
        min_bound[2],
        max_bound[2],  # z_min, z_max
    ]
    box = pv.Box(bounds=bounds)

    # Plotter에 박스 추가
    plotter.add_mesh(box, color=color, opacity=opacity)


def add_points(
    plotter: Plotter,
    pcd: PointCloud,
    point_size: int = 3,
):
    """
    주어진 PointCloud의 포인트를 Plotter에 추가합니다.

    Args:
        plotter (Plotter): PyVista Plotter 인스턴스.
        pcd (PointCloud): 추가할 포인트 클라우드.
        point_size (int, optional): 포인트 크기. 기본값은 3입니다.
    """
    points = np.asarray(pcd.points)
    pv_data = pv.PolyData(points)

    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        # Open3D 색상이 [0,1] float인 경우 uint8 [0,255]로 변환
        if colors.dtype.kind in ("f", "c"):
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        pv_data["colors"] = colors
        plotter.add_points(pv_data, scalars="colors", rgb=True, point_size=point_size)
    else:
        plotter.add_points(pv_data, color="red", point_size=point_size)
