"""
PCD 전처리 & 메시 재건 도구 (util.pcd_tool)
==========================================

스풀 PCD를 다루는 재사용 가능한 함수 모음:

  1. load_pcd                  — PCD/PLY 로드 + 스케일 조정
  2. remove_statistical_outliers — 통계적 이상치(산발 노이즈) 제거
  3. denoise_dbscan            — DBSCAN 클러스터링으로 주변 노이즈 덩어리 제거
  4. voxel_ccl                 — 옥트리(복셀) 기반 CCL (대용량 메모리 안전)
  5. reconstruct_mesh_poisson  — Poisson 표면 재건
  6. mesh_pcd_distance         — 재건 메시 vs 원본 PCD 평균 거리(성능 지표)
  7. compare_with_reference    — 상용툴 결과와 ICP 정합 후 비교

  + open3d 인라인 시각화 헬퍼 (draw_plotly 기반)

사용 (python/ 가 sys.path에 있을 때):
    from util.pcd_tool import load_pcd, voxel_ccl, save_pcd, compare_with_reference
"""

from collections import defaultdict, deque

import numpy as np
import open3d as o3d


# ===========================================================================
# 시각화 헬퍼 (open3d draw_plotly — 노트북 안 인터랙티브 렌더링)
# ===========================================================================
RGB = {
    'gray':   [0.70, 0.70, 0.70],
    'blue':   [0.25, 0.41, 0.88],
    'orange': [1.00, 0.55, 0.00],
    'green':  [0.10, 0.70, 0.20],
    'red':    [0.90, 0.10, 0.10],
}


def downsample_for_view(arr_or_pcd, n=30000):
    """표시용 다운샘플 (plotly가 무거워지지 않게). PointCloud 또는 (N,3) 배열 입력."""
    if isinstance(arr_or_pcd, o3d.geometry.PointCloud):
        pts = np.asarray(arr_or_pcd.points)
        if len(pts) <= n:
            return arr_or_pcd
        idx = np.random.choice(len(pts), n, replace=False)
        return arr_or_pcd.select_by_index(idx)
    arr = np.asarray(arr_or_pcd)
    if len(arr) <= n:
        return arr
    return arr[np.random.choice(len(arr), n, replace=False)]


def make_pcd(points, color=None):
    """(N,3) 배열 또는 PointCloud → 색칠된 PointCloud."""
    if isinstance(points, o3d.geometry.PointCloud):
        p = points
    else:
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if color is not None:
        p.paint_uniform_color(RGB.get(color, color))
    return p


def show3d(geoms):
    """plotly로 노트북 안에 인터랙티브 렌더링 (마우스 회전/확대)."""
    if not isinstance(geoms, (list, tuple)):
        geoms = [geoms]
    o3d.visualization.draw_plotly(geoms, width=720, height=600)


# ===========================================================================
# 1. 로드
# ===========================================================================
def load_pcd(path, scale=1.0):
    """PCD/PLY 로드 후 스케일 적용.

    Args:
        path:  파일 경로
        scale: 단위 변환 계수 (mm→m면 1e-3, 이미 m면 1.0)
    Returns:
        o3d.geometry.PointCloud
    """
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"점을 불러오지 못했습니다: {path}")
    if scale != 1.0:
        pcd.scale(scale, center=(0, 0, 0))
    return pcd


# ===========================================================================
# 2. 통계적 이상치 제거
# ===========================================================================
def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """각 점의 이웃 평균 거리가 전체 분포에서 벗어난 산발 노이즈를 제거.

    Returns:
        (clean_pcd, kept_index)
    """
    clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return clean, ind


# ===========================================================================
# 3. DBSCAN 클러스터링 기반 노이즈 제거
# ===========================================================================
def cluster_dbscan(pcd, eps=0.02, min_points=10):
    """DBSCAN 클러스터링. 각 점의 클러스터 라벨 배열을 반환 (-1 = 노이즈)."""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points,
                                         print_progress=False))
    return labels


def denoise_dbscan(pcd, eps=0.02, min_points=10, keep='largest', min_cluster_size=None):
    """DBSCAN으로 클러스터링한 뒤 주요 클러스터만 남겨 주변 노이즈를 제거.

    Args:
        eps:              이웃으로 간주할 거리 (m). 점 밀도에 맞게 조정
        min_points:       코어 점이 되기 위한 최소 이웃 수
        keep:             'largest' = 가장 큰 클러스터만,
                          'all_valid' = min_cluster_size 이상 모든 클러스터
        min_cluster_size: keep='all_valid'일 때 유지할 최소 클러스터 크기
    Returns:
        (clean_pcd, labels)  labels는 원본 pcd 각 점의 클러스터 라벨
    """
    labels = cluster_dbscan(pcd, eps=eps, min_points=min_points)
    valid = labels[labels >= 0]
    if len(valid) == 0:
        raise ValueError("클러스터를 찾지 못했습니다. eps/min_points를 조정하세요.")

    unique, counts = np.unique(valid, return_counts=True)
    if keep == 'largest':
        target = {unique[np.argmax(counts)]}
    elif keep == 'all_valid':
        thr = min_cluster_size if min_cluster_size is not None else 0
        target = {c for c, n in zip(unique, counts) if n >= thr}
    else:
        raise ValueError(f"알 수 없는 keep 옵션: {keep}")

    keep_idx = np.where(np.isin(labels, list(target)))[0]
    clean = pcd.select_by_index(keep_idx)
    return clean, labels


def colorize_clusters(pcd, labels):
    """클러스터 라벨에 따라 색을 입힌 PointCloud 반환 (노이즈=-1은 검정)."""
    import matplotlib.pyplot as plt
    colored = o3d.geometry.PointCloud(pcd)
    cmap = plt.get_cmap("tab20")
    colors = cmap((labels % 20) / 19.0)[:, :3]
    colors[labels < 0] = 0.0  # 노이즈는 검정
    colored.colors = o3d.utility.Vector3dVector(colors)
    return colored


# ===========================================================================
# 3-alt. 옥트리(복셀) 기반 CCL — CloudCompare "Label Connected Components" 방식
#   대용량(수백만 점)에서 DBSCAN보다 가볍고 메모리 안전.
#   특정 복셀 크기(= 옥트리 레벨)에서 점유 복셀을 연결요소로 묶는다.
# ===========================================================================
def voxel_ccl(points, voxel_size, min_points=10, connectivity=26):
    """복셀 그리드 기반 Connected Component Labeling.

    Args:
        points:       (N, 3) 배열
        voxel_size:   복셀 한 변 길이 (m). 작을수록 분리가 잘 되지만 과분할 위험
        min_points:   유지할 연결요소의 최소 점 수
        connectivity: 6(면 인접) 또는 26(면+모서리+꼭짓점 인접)
    Returns:
        (kept_point_indices, component_labels)
        component_labels: 각 점의 컴포넌트 라벨 (버려진 점은 -1)
    """
    points = np.asarray(points)
    min_bound = points.min(axis=0)

    voxel_idx = np.floor((points - min_bound) / voxel_size).astype(np.int32)

    voxel_to_indices = defaultdict(list)
    for point_id, v in enumerate(voxel_idx):
        voxel_to_indices[tuple(v)].append(point_id)

    occupied = set(voxel_to_indices.keys())

    if connectivity == 6:
        neighbors = [(1, 0, 0), (-1, 0, 0),
                     (0, 1, 0), (0, -1, 0),
                     (0, 0, 1), (0, 0, -1)]
    else:
        neighbors = [(dx, dy, dz)
                     for dx in (-1, 0, 1)
                     for dy in (-1, 0, 1)
                     for dz in (-1, 0, 1)
                     if (dx, dy, dz) != (0, 0, 0)]

    visited = set()
    kept_point_indices = []
    component_labels = np.full(len(points), -1, dtype=np.int32)
    component_id = 0

    for start in occupied:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component_voxels = []
        while queue:
            current = queue.popleft()
            component_voxels.append(current)
            for dx, dy, dz in neighbors:
                nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
                if nxt in occupied and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        point_ids = []
        for voxel in component_voxels:
            point_ids.extend(voxel_to_indices[voxel])

        if len(point_ids) >= min_points:
            component_labels[point_ids] = component_id
            kept_point_indices.extend(point_ids)
            component_id += 1

    return np.asarray(kept_point_indices), component_labels




# ===========================================================================
# 옥트리 생성 시각화 헬퍼
# ===========================================================================
def build_octree(pcd, max_depth=6, size_expand=0.01):
    """포인트 클라우드로부터 옥트리 생성."""
    octree = o3d.geometry.Octree(max_depth=max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=size_expand)
    return octree



def octree_node_boxes(octree, depth=None):
    """옥트리를 순회하며 노드 큐브 정보를 수집.

    Args:
        depth: 특정 깊이만 수집 (None이면 전체)
    Returns:
        list of (origin(3,), size, depth)
    """
    boxes = []

    def _cb(node, info):
        if depth is None or info.depth == depth:
            boxes.append((np.asarray(info.origin, dtype=float), float(info.size), int(info.depth)))
        return False

    octree.traverse(_cb)
    return boxes


def make_wire_box(origin, size, color='gray'):
    """코너 origin, 한 변 size인 정육면체의 와이어프레임 LineSet."""
    o = np.asarray(origin, dtype=float)
    s = float(size)
    corners = o + np.array([
        [0, 0, 0], [s, 0, 0], [0, s, 0], [s, s, 0],
        [0, 0, s], [s, 0, s], [0, s, s], [s, s, s],
    ])
    edges = [[0, 1], [0, 2], [1, 3], [2, 3],
             [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    c = RGB.get(color, color)
    ls.colors = o3d.utility.Vector3dVector(np.tile(c, (len(edges), 1)))
    return ls


def octree_boxes_lineset(octree, depth):
    """특정 깊이의 모든 노드 박스를 하나의 LineSet으로 합쳐 반환 (가벼움)."""
    boxes = octree_node_boxes(octree, depth=depth)
    all_pts, all_lines, all_cols = [], [], []
    edges = [[0, 1], [0, 2], [1, 3], [2, 3],
             [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    for origin, size, _ in boxes:
        base = len(all_pts)
        o = np.asarray(origin, dtype=float); s = float(size)
        corners = o + np.array([
            [0, 0, 0], [s, 0, 0], [0, s, 0], [s, s, 0],
            [0, 0, s], [s, 0, s], [0, s, s], [s, s, s],
        ])
        all_pts.extend(corners.tolist())
        all_lines.extend([[a + base, b + base] for a, b in edges])
    ls = o3d.geometry.LineSet()
    if all_pts:
        ls.points = o3d.utility.Vector3dVector(np.asarray(all_pts))
        ls.lines = o3d.utility.Vector2iVector(np.asarray(all_lines))
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(RGB['green'], (len(all_lines), 1)))
    return ls, len(boxes)


def colorize_ccl(pcd, labels):
    """CCL 컴포넌트 라벨에 따라 색을 입힌 PointCloud (버려진 -1은 검정)."""
    return colorize_clusters(pcd, labels)


# ===========================================================================
# 4·5. 메시 재건
# ===========================================================================
def reconstruct_mesh_poisson(pcd, depth=9, normal_radius=0.05, normal_max_nn=30,
                             density_quantile=0.05):
    """Poisson 표면 재건. 법선을 먼저 추정한 뒤 수행하고,
    밀도가 낮은(데이터가 없는) 영역의 면을 제거한다.

    Returns:
        o3d.geometry.TriangleMesh
    """
    pcd = o3d.geometry.PointCloud(pcd)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=normal_max_nn)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    densities = np.asarray(densities)
    if density_quantile > 0:
        thr = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < thr)
    mesh.compute_vertex_normals()
    return mesh


def reconstruct_mesh_marching_cubes(pcd, resolution=128, sigma=1.5, level=0.5):
    """마칭 큐브 재건. util.pcd_mesh.points_to_mesh를 재사용.

    Returns:
        o3d.geometry.TriangleMesh
    """
    from util.pcd_mesh import points_to_mesh
    pts = np.asarray(pcd.points)
    tm, _ = points_to_mesh(pts, resolution=resolution, sigma=sigma, level=level)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    mesh.compute_vertex_normals()
    return mesh


# ===========================================================================
# 6. 재건 품질 평가
# ===========================================================================
def mesh_pcd_distance(mesh, pcd, n_samples=100000):
    """재건된 메시 표면과 원본 PCD 간 평균/RMS 거리(가장 가까운 점 기준).

    메시 표면을 균일 샘플링한 뒤, 원본 PCD의 각 점에서 가장 가까운
    메시 샘플 점까지의 거리를 계산한다.

    Returns:
        dict(mean, rms, max, n_samples)
    """
    n = min(n_samples, max(1000, len(mesh.triangles) * 3))
    sampled = mesh.sample_points_uniformly(number_of_points=int(n))
    d = np.asarray(pcd.compute_point_cloud_distance(sampled))
    return {
        "mean": float(d.mean()),
        "rms":  float(np.sqrt((d ** 2).mean())),
        "max":  float(d.max()),
        "n_samples": int(n),
    }


# ===========================================================================
# 저장 & 상용툴 결과 비교
# ===========================================================================
def save_pcd(pcd, path, write_ascii=False):
    """PointCloud를 파일로 저장 (확장자로 PCD/PLY 자동 판별).

    Returns:
        저장 경로(str)
    """
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=write_ascii)
    if not ok:
        raise IOError(f"저장 실패: {path}")
    return str(path)


def compare_with_reference(result_pcd, reference_pcd, icp_threshold=0.05,
                           ref_scale=1.0, init_transform=None):
    """내 결과 PCD를 상용툴(예: CloudCompare) 필터 결과와 ICP 정합 후 비교.

    ICP로 result→reference 정렬을 맞춘 뒤:
      - 점 개수 차이
      - 평균 최근접 점거리 (양방향 + 대칭)
    를 계산한다.

    Args:
        result_pcd:    내 파이프라인 결과
        reference_pcd: 상용툴 필터 결과 (기준)
        icp_threshold: ICP 대응점 최대 거리 (m)
        ref_scale:     reference 단위 변환 계수 (필요시 mm→m 등)
        init_transform: ICP 초기 변환 (None이면 단위행렬)
    Returns:
        dict(
            n_result, n_reference, count_diff, count_diff_ratio,
            icp_fitness, icp_rmse, icp_transformation,
            mean_dist_r2ref, mean_dist_ref2r, mean_dist_symmetric,
            aligned_result  # 정합된 결과 PCD (시각화용)
        )
    """
    source = o3d.geometry.PointCloud(result_pcd)
    target = o3d.geometry.PointCloud(reference_pcd)
    if ref_scale != 1.0:
        target.scale(ref_scale, center=(0, 0, 0))

    if init_transform is None:
        init_transform = np.eye(4)

    reg = o3d.pipelines.registration.registration_icp(
        source, target, icp_threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    aligned = o3d.geometry.PointCloud(source)
    aligned.transform(reg.transformation)

    d_r2ref = np.asarray(aligned.compute_point_cloud_distance(target))
    d_ref2r = np.asarray(target.compute_point_cloud_distance(aligned))

    n_res = len(aligned.points)
    n_ref = len(target.points)
    mean_r2ref = float(d_r2ref.mean())
    mean_ref2r = float(d_ref2r.mean())

    return {
        "n_result": n_res,
        "n_reference": n_ref,
        "count_diff": n_res - n_ref,
        "count_diff_ratio": (n_res - n_ref) / n_ref if n_ref else float('nan'),
        "icp_fitness": float(reg.fitness),
        "icp_rmse": float(reg.inlier_rmse),
        "icp_transformation": reg.transformation,
        "mean_dist_r2ref": mean_r2ref,
        "mean_dist_ref2r": mean_ref2r,
        "mean_dist_symmetric": 0.5 * (mean_r2ref + mean_ref2r),
        "aligned_result": aligned,
    }
