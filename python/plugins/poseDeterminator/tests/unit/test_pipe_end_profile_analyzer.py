import numpy as np
import open3d as o3d

from PipeEndProfileAnalyzer import analyze_pipe_end_profiles, analyze_pipe_endpoints_by_voxel_graph


def _cylinder_surface(axis, start, length, radius, axial_count=80, radial_count=24):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    start = np.asarray(start, dtype=float)
    basis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, basis)) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
    v1 = np.cross(axis, basis)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(axis, v1)
    v2 = v2 / np.linalg.norm(v2)

    ts = np.linspace(0.0, length, axial_count)
    angles = np.linspace(0.0, 2.0 * np.pi, radial_count, endpoint=False)
    points = []
    for t in ts:
        center = start + axis * t
        circle = center + radius * (
            np.outer(np.cos(angles), v1) + np.outer(np.sin(angles), v2)
        )
        points.append(circle)
    return np.vstack(points)


def _write_point_cloud(path, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    assert o3d.io.write_point_cloud(str(path), pcd)


def _assert_centers_match(free_centers, expected_centers, tolerance):
    distances = np.linalg.norm(free_centers[:, None, :] - expected_centers[None, :, :], axis=2)
    assert np.all(np.min(distances, axis=0) < tolerance)


def test_analyze_pipe_end_profiles_finds_straight_pipe_ends(tmp_path):
    radius = 0.05
    points = _cylinder_surface([1, 0, 0], [0, 0, 0], 1.0, radius)

    path = tmp_path / "straight_pipe.ply"
    _write_point_cloud(path, points)

    result = analyze_pipe_end_profiles(
        path,
        max_points=5000,
        max_segments=4,
        ransac_iterations=80,
        sample_size=96,
        distance_threshold=0.004,
        profile_sample_count=16,
    )

    assert len(result["segments"]) == 1
    assert len(result["terminal_end_profiles"]) == 2
    assert all(len(profile["profile_points"]) == 16 for profile in result["terminal_end_profiles"])

    free_centers = np.array([profile["center"] for profile in result["terminal_end_profiles"]])
    expected_free_centers = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    _assert_centers_match(free_centers, expected_free_centers, tolerance=0.04)


def test_analyze_pipe_end_profiles_finds_l_pipe_ends(tmp_path):
    radius = 0.05
    leg_a = _cylinder_surface([1, 0, 0], [0, 0, 0], 1.0, radius)
    leg_b = _cylinder_surface([0, 1, 0], [1, 0, 0], 0.8, radius)
    points = np.vstack([leg_a, leg_b])

    path = tmp_path / "l_pipe.ply"
    _write_point_cloud(path, points)

    result = analyze_pipe_end_profiles(
        path,
        max_points=5000,
        max_segments=4,
        ransac_iterations=80,
        sample_size=96,
        distance_threshold=0.004,
        profile_sample_count=16,
    )

    assert len(result["segments"]) == 2
    assert len(result["terminal_end_profiles"]) == 2
    assert all(len(profile["profile_points"]) == 16 for profile in result["terminal_end_profiles"])

    free_centers = np.array([profile["center"] for profile in result["terminal_end_profiles"]])
    expected_free_centers = np.array([[0, 0, 0], [1, 0.8, 0]], dtype=float)
    _assert_centers_match(free_centers, expected_free_centers, tolerance=0.04)


def test_analyze_pipe_end_profiles_finds_multi_bend_pipe_ends(tmp_path):
    radius = 0.05
    leg_a = _cylinder_surface([1, 0, 0], [0, 0, 0], 1.0, radius)
    leg_b = _cylinder_surface([0, 1, 0], [1, 0, 0], 0.8, radius)
    leg_c = _cylinder_surface([0, 0, 1], [1, 0.8, 0], 0.6, radius)
    points = np.vstack([leg_a, leg_b, leg_c])

    path = tmp_path / "multi_bend_pipe.ply"
    _write_point_cloud(path, points)

    result = analyze_pipe_end_profiles(
        path,
        max_points=8000,
        max_segments=5,
        ransac_iterations=120,
        sample_size=96,
        distance_threshold=0.004,
        profile_sample_count=16,
    )

    assert len(result["segments"]) == 3
    assert len(result["terminal_end_profiles"]) == 2

    free_centers = np.array([profile["center"] for profile in result["terminal_end_profiles"]])
    expected_free_centers = np.array([[0, 0, 0], [1, 0.8, 0.6]], dtype=float)
    _assert_centers_match(free_centers, expected_free_centers, tolerance=0.04)


def test_voxel_graph_endpoint_detector_finds_straight_pipe_ends(tmp_path):
    radius = 0.05
    points = _cylinder_surface([1, 0, 0], [0, 0, 0], 1.0, radius)

    path = tmp_path / "straight_pipe_graph.ply"
    _write_point_cloud(path, points)

    result = analyze_pipe_endpoints_by_voxel_graph(
        path,
        voxel_size=0.08,
        min_voxel_points=2,
    )

    ends = np.array(result["terminal_end_points"], dtype=float)
    expected = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    _assert_centers_match(ends, expected, tolerance=0.10)
    assert result["edge_count"] > 0
    assert len(result["graph"]["path_node_indices"]) >= 2


def test_voxel_graph_endpoint_detector_finds_l_pipe_ends(tmp_path):
    radius = 0.05
    leg_a = _cylinder_surface([1, 0, 0], [0, 0, 0], 1.0, radius)
    leg_b = _cylinder_surface([0, 1, 0], [1, 0, 0], 0.8, radius)
    points = np.vstack([leg_a, leg_b])

    path = tmp_path / "l_pipe_graph.ply"
    _write_point_cloud(path, points)

    result = analyze_pipe_endpoints_by_voxel_graph(
        path,
        voxel_size=0.08,
        min_voxel_points=2,
    )

    ends = np.array(result["terminal_end_points"], dtype=float)
    expected = np.array([[0, 0, 0], [1, 0.8, 0]], dtype=float)
    _assert_centers_match(ends, expected, tolerance=0.12)
    assert result["diameter_distance"] > 1.5
