from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import open3d as o3d

from CylinderFitting import fit_cylinder


@dataclass
class CylinderSegmentProfile:
    axis: np.ndarray
    axis_point: np.ndarray
    radius: float
    endpoints: tuple[np.ndarray, np.ndarray]
    inlier_count: int
    rms_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis.tolist(),
            "axis_point": self.axis_point.tolist(),
            "radius": float(self.radius),
            "endpoints": [self.endpoints[0].tolist(), self.endpoints[1].tolist()],
            "inlier_count": int(self.inlier_count),
            "rms_error": float(self.rms_error),
        }


def analyze_pipe_end_profiles(
    file_path: str | Path,
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
    endpoint_quantile: float = 0.01,
    target_segment_count: int | None = None,
    log_timing: bool = False,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Estimate both terminal end profiles of a piecewise-straight pipe.

    The pipe can be straight, L-shaped, or composed of several straight
    sections joined by bends. The algorithm repeatedly extracts cylindrical
    straight sections and treats unconnected section endpoints as pipe ends.
    """

    pcd = o3d.io.read_point_cloud(str(file_path))  # type: ignore
    if pcd.is_empty():
        raise ValueError(f"Point cloud has no points: {file_path}")

    if scale != 1.0:
        pcd.scale(scale, np.zeros(3, dtype=float))  # type: ignore

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    points = np.asarray(pcd.points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 20:
        raise ValueError("At least 20 valid points are required.")

    rng = np.random.default_rng(random_seed)
    fit_points = _subsample_points(points, max_points, rng)

    if distance_threshold is None:
        bbox_diag = float(np.linalg.norm(fit_points.max(axis=0) - fit_points.min(axis=0)))
        distance_threshold = max(bbox_diag * 0.005, np.finfo(float).eps)
    else:
        bbox_diag = float(np.linalg.norm(fit_points.max(axis=0) - fit_points.min(axis=0)))

    if min_segment_points is None:
        min_segment_points = max(20, sample_size)
    if target_segment_count is not None and target_segment_count <= 0:
        raise ValueError("target_segment_count must be positive when provided.")
    timing_log: list[dict[str, Any]] = []

    remaining = fit_points
    segments: list[CylinderSegmentProfile] = []
    segment_point_clouds: list[np.ndarray] = []
    cylinder_fit_point_clouds: list[np.ndarray] = []
    sampling_debug_infos: list[dict[str, Any]] = []


    print(f"Analyzing pipe end profiles from {file_path}...")
    print(f"  Total points: {len(points)}" )
    print(f"  Fit points: {len(fit_points)}" )
    print(f"  bbox diagonal: {bbox_diag:.6f}" )
    print(f"  Distance threshold: {distance_threshold:.6f}" )
    # print(f"  Connection threshold: {connection_threshold:.6f}" )

    segment_limit = target_segment_count if target_segment_count is not None else max_segments
    total_start_time = perf_counter()
    for segment_attempt_index in range(max(1, segment_limit)):
        if len(remaining) < min_segment_points:
            break

        segment_start_time = perf_counter()
        try:
            segment, inlier_mask, cylinder_fit_points, sampling_debug = _fit_one_cylinder_segment(
                remaining,
                rng=rng,
                ransac_iterations=ransac_iterations,
                sample_size=sample_size,
                distance_threshold=distance_threshold,
                endpoint_quantile=endpoint_quantile,
                bbox_diag=bbox_diag,
                log_timing=log_timing,
                segment_index=len(segments),
            )
        except RuntimeError:
            if target_segment_count is not None or not segments:
                raise
            break
        if segment.inlier_count < min_segment_points:
            break

        segments.append(segment)
        segment_elapsed = perf_counter() - segment_start_time
        timing_entry = {
            "segment_index": int(len(segments) - 1),
            "elapsed_sec": float(segment_elapsed),
            "remaining_before": int(len(remaining)),
            "inlier_count": int(segment.inlier_count),
            "remaining_after": int(len(remaining) - int(np.count_nonzero(inlier_mask))),
            "best_iteration": sampling_debug.get("best_iteration"),
            "best_score": sampling_debug.get("best_score"),
            "candidate_timings": sampling_debug.get("candidate_timings", []),
        }
        timing_log.append(timing_entry)
        if log_timing:
            _print_timing(
                f"segment {timing_entry['segment_index']} done",
                segment_elapsed,
                {
                    "remaining_before": timing_entry["remaining_before"],
                    "inliers": timing_entry["inlier_count"],
                    "remaining_after": timing_entry["remaining_after"],
                    "best_iteration": timing_entry["best_iteration"],
                    "best_score": timing_entry["best_score"],
                },
            )
        if include_segment_points:
            segment_point_clouds.append(remaining[inlier_mask])
            cylinder_fit_point_clouds.append(cylinder_fit_points)
            sampling_debug_infos.append(sampling_debug)
        remaining = remaining[~inlier_mask]

    if not segments:
        raise RuntimeError("Failed to find a cylinder segment from the pipe.")
    if target_segment_count is not None and len(segments) != target_segment_count:
        raise RuntimeError(
            f"Expected {target_segment_count} segments, but found {len(segments)}."
        )

    end_records = _classify_pipe_endpoints(segments, distance_threshold, connection_threshold)
    terminal_end_profiles = []
    for record in end_records["terminal_ends"]:
        segment = segments[record["segment_index"]]
        endpoint = segment.endpoints[record["endpoint_index"]]
        profile_points = _circle_profile_points(endpoint, segment.axis, segment.radius, profile_sample_count)
        terminal_end_profiles.append(
            {
                "segment_index": int(record["segment_index"]),
                "endpoint_index": int(record["endpoint_index"]),
                "center": endpoint.tolist(),
                "axis": segment.axis.tolist(),
                "radius": float(segment.radius),
                "profile_points": profile_points.tolist(),
            }
        )

    segment_dicts = [segment.to_dict() for segment in segments]
    result = {
        "file_path": str(file_path),
        "point_count": int(len(points)),
        "fit_point_count": int(len(fit_points)),
        "unassigned_fit_point_count": int(len(remaining)),
        "distance_threshold": float(distance_threshold),
        "target_segment_count": None if target_segment_count is None else int(target_segment_count),
        "timing_sec": float(perf_counter() - total_start_time),
        "timing_log": timing_log,
        "connection_threshold": float(end_records["connection_threshold"]),
        "segments": segment_dicts,
        "joint_endpoint_pairs": end_records["joint_endpoint_pairs"],
        "terminal_end_profiles": terminal_end_profiles,
        "free_end_profiles": terminal_end_profiles,
        "elbow_endpoints": end_records["joint_endpoint_pairs"],
    }
    if include_segment_points:
        result["segment_point_clouds"] = [segment_points.tolist() for segment_points in segment_point_clouds]
        result["cylinder_fit_point_clouds"] = [fit_points.tolist() for fit_points in cylinder_fit_point_clouds]
        result["sampling_debug_infos"] = sampling_debug_infos
        result["unassigned_fit_points"] = remaining.tolist()
    return result


def analyze_l_pipe_end_profiles(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible alias for older L-pipe notebooks/tests."""
    return analyze_pipe_end_profiles(*args, **kwargs)


def analyze_pipe_endpoints_by_voxel_graph(
    file_path: str | Path,
    scale: float = 1.0,
    voxel_size: float | None = None,
    max_points: int = 50000,
    min_voxel_points: int = 1,
    neighbor_radius_voxels: int = 1,
    include_points: bool = False,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Find pipe end candidates from a coarse voxel graph.

    This is intended as a lightweight endpoint detector for both straight and
    bent pipes. It does not try to fit cylinder segments. Instead it builds a
    graph from occupied coarse voxels and uses the graph diameter endpoints as
    the two chuck-mount candidates.
    """

    started = perf_counter()
    pcd = o3d.io.read_point_cloud(str(file_path))  # type: ignore
    if pcd.is_empty():
        raise ValueError(f"Point cloud has no points: {file_path}")

    if scale != 1.0:
        pcd.scale(scale, np.zeros(3, dtype=float))  # type: ignore

    points = np.asarray(pcd.points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 4:
        raise ValueError("At least 4 valid points are required.")

    rng = np.random.default_rng(random_seed)
    fit_points = _subsample_points(points, max_points, rng)
    bbox_min = fit_points.min(axis=0)
    bbox_max = fit_points.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))

    if voxel_size is None:
        voxel_size = max(bbox_diag * 0.025, np.finfo(float).eps)
    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")
    if min_voxel_points <= 0:
        raise ValueError("min_voxel_points must be positive.")
    if neighbor_radius_voxels <= 0:
        raise ValueError("neighbor_radius_voxels must be positive.")

    nodes, voxel_keys, voxel_counts = _voxel_graph_nodes(
        fit_points,
        voxel_size=voxel_size,
        origin=bbox_min,
        min_voxel_points=min_voxel_points,
    )
    if len(nodes) < 2:
        raise RuntimeError("Voxel graph needs at least two occupied voxels.")

    adjacency, edges = _build_voxel_graph_edges(
        nodes,
        voxel_keys,
        voxel_size=voxel_size,
        neighbor_radius_voxels=neighbor_radius_voxels,
    )
    component_indices = _largest_connected_component(adjacency)
    if len(component_indices) < 2:
        raise RuntimeError(
            "Voxel graph largest component needs at least two connected nodes. "
            f"Current voxel_size={voxel_size:.6g}, node_count={len(nodes)}, edge_count={len(edges)}. "
            "Increase voxel_size, lower min_voxel_points, or use voxel_size=None for auto sizing."
        )

    far_a_seed = component_indices[0]
    dist_from_seed, _ = _dijkstra_graph(adjacency, far_a_seed, allowed=component_indices)
    far_a = _farthest_reachable_node(dist_from_seed)
    dist_from_a, prev_from_a = _dijkstra_graph(adjacency, far_a, allowed=component_indices)
    far_b = _farthest_reachable_node(dist_from_a)
    path_indices = _reconstruct_path(prev_from_a, far_a, far_b)

    endpoint_node_indices = [int(far_a), int(far_b)]
    endpoint_points = nodes[endpoint_node_indices]
    graph_edges = [
        [int(i), int(j)]
        for i, j, _ in edges
        if i in component_indices and j in component_indices
    ]

    result = {
        "file_path": str(file_path),
        "point_count": int(len(points)),
        "fit_point_count": int(len(fit_points)),
        "voxel_size": float(voxel_size),
        "min_voxel_points": int(min_voxel_points),
        "neighbor_radius_voxels": int(neighbor_radius_voxels),
        "node_count": int(len(nodes)),
        "component_node_count": int(len(component_indices)),
        "edge_count": int(len(graph_edges)),
        "diameter_distance": float(dist_from_a[far_b]),
        "terminal_node_indices": endpoint_node_indices,
        "terminal_end_points": endpoint_points.tolist(),
        "graph": {
            "nodes": nodes.tolist(),
            "voxel_keys": voxel_keys.tolist(),
            "voxel_counts": voxel_counts.tolist(),
            "edges": graph_edges,
            "component_node_indices": [int(i) for i in component_indices],
            "path_node_indices": [int(i) for i in path_indices],
            "path_points": nodes[path_indices].tolist() if path_indices else [],
        },
        "timing_sec": float(perf_counter() - started),
    }
    if include_points:
        result["points"] = points.tolist()
        result["fit_points"] = fit_points.tolist()
    return result


def _fit_one_cylinder_segment(
    points: np.ndarray,
    rng: np.random.Generator,
    ransac_iterations: int,
    sample_size: int,
    distance_threshold: float,
    endpoint_quantile: float,
    bbox_diag: float,
    log_timing: bool = False,
    segment_index: int | None = None,
) -> tuple[CylinderSegmentProfile, np.ndarray, np.ndarray, dict[str, Any]]:
    if len(points) < 20:
        raise ValueError("Not enough points to fit a cylinder segment.")

    sample_size = min(sample_size, len(points))
    best_score = -1
    best_model: tuple[np.ndarray, np.ndarray, float] | None = None
    best_mask: np.ndarray | None = None
    best_sample: np.ndarray | None = None
    best_debug: dict[str, Any] = {}
    candidate_timings: list[dict[str, Any]] = []

    for iteration_index in range(max(1, ransac_iterations)):
        iteration_start = perf_counter()
        anchor_idx = int(rng.integers(0, len(points)))
        sample, optimizer_model, sampling_debug = _sample_profile_points_from_anchor(
            points,
            anchor_idx,
            distance_threshold,
            bbox_diag,
            min_points=max(20, min(sample_size, len(points))),
            log_timing=log_timing,
        )
        sampling_elapsed = perf_counter() - iteration_start
        if sample is None or optimizer_model is None:
            if log_timing:
                candidate_timings.append(
                    {
                        "iteration": int(iteration_index),
                        "elapsed_sec": float(sampling_elapsed),
                        "status": "sample_failed",
                        "sample_timing": sampling_debug.get("timing", {}),
                    }
                )
            continue

        axis, axis_point, radius = optimizer_model
        inlier_start = perf_counter()
        residuals = _cylinder_radial_residuals(points, axis, axis_point, radius)
        mask = residuals <= distance_threshold
        score = int(mask.sum())
        inlier_elapsed = perf_counter() - inlier_start
        iteration_elapsed = perf_counter() - iteration_start

        if log_timing:
            candidate_timings.append(
                {
                    "iteration": int(iteration_index),
                    "elapsed_sec": float(iteration_elapsed),
                    "sample_elapsed_sec": float(sampling_elapsed),
                    "inlier_eval_sec": float(inlier_elapsed),
                    "status": "ok",
                    "score": score,
                    "sample_points": int(len(sample)),
                    "sample_timing": sampling_debug.get("timing", {}),
                }
            )

        if score > best_score:
            best_score = score
            best_model = (axis, axis_point, radius)
            best_mask = mask
            best_sample = sample
            best_debug = sampling_debug
            best_debug["best_iteration"] = int(iteration_index)
            best_debug["best_score"] = int(best_score)
            if log_timing:
                label = f"segment {segment_index} iter {iteration_index}" if segment_index is not None else f"iter {iteration_index}"
                _print_timing(
                    f"{label} new best",
                    iteration_elapsed,
                    {
                        "score": best_score,
                        "sample_points": len(sample),
                        **sampling_debug.get("timing", {}),
                    },
                )

    if best_model is None or best_mask is None or best_score < 10:
        raise RuntimeError("Cylinder RANSAC failed.")

    refine_start = perf_counter()
    inlier_points = best_sample if best_sample is not None else points[best_mask]
    refined_axis, refined_axis_point, refined_radius = _refine_cylinder(inlier_points, best_model)
    refine_elapsed = perf_counter() - refine_start

    mask_start = perf_counter()
    residuals = _cylinder_radial_residuals(points, refined_axis, refined_axis_point, refined_radius)
    refined_mask = residuals <= distance_threshold
    refined_inliers = points[refined_mask]
    mask_elapsed = perf_counter() - mask_start

    endpoint_start = perf_counter()
    endpoint_a, endpoint_b = _segment_axis_endpoints(
        refined_inliers,
        refined_axis,
        refined_axis_point,
        endpoint_quantile,
    )
    segment_residuals = _cylinder_radial_residuals(refined_inliers, refined_axis, refined_axis_point, refined_radius)
    rms_error = float(np.sqrt(np.mean(segment_residuals**2)))
    endpoint_elapsed = perf_counter() - endpoint_start

    segment = CylinderSegmentProfile(
        axis=refined_axis,
        axis_point=refined_axis_point,
        radius=float(refined_radius),
        endpoints=(endpoint_a, endpoint_b),
        inlier_count=int(len(refined_inliers)),
        rms_error=rms_error,
    )
    best_debug["best_iteration"] = best_debug.get("best_iteration")
    best_debug["best_score"] = int(best_score)
    if log_timing:
        best_debug["candidate_timings"] = candidate_timings
        best_debug["postprocess_timing"] = {
            "refine_sec": float(refine_elapsed),
            "mask_sec": float(mask_elapsed),
            "endpoint_sec": float(endpoint_elapsed),
        }
    return segment, refined_mask, inlier_points, best_debug


def _sample_profile_points_from_anchor(
    points: np.ndarray,
    anchor_idx: int,
    distance_threshold: float,
    bbox_diag: float,
    min_points: int,
    log_timing: bool = False,
) -> tuple[np.ndarray | None, tuple[np.ndarray, np.ndarray, float] | None, dict[str, Any]]:
    timing: dict[str, float] = {}
    total_start = perf_counter()
    anchor = points[anchor_idx]
    normal_sample_half_size = max(distance_threshold * 4.0, bbox_diag * 0.01, np.finfo(float).eps)
    step_start = perf_counter()
    normal_points = _extract_points_in_aabb(points, anchor, normal_sample_half_size)
    timing["normal_sample_sec"] = perf_counter() - step_start
    if len(normal_points) < 3:
        debug = {
            "pipe_profile_target_point": anchor.tolist(),
            "normal_estimation_points": normal_points.tolist(),
            "normal_estimation_sampling_box": [
                (anchor - normal_sample_half_size).tolist(),
                (anchor + normal_sample_half_size).tolist(),
            ],
        }
        if log_timing:
            timing["total_sec"] = perf_counter() - total_start
            debug["timing"] = timing
        return None, None, {
            **debug,
        }

    try:
        step_start = perf_counter()
        normal_m = _estimate_local_normal(normal_points, normal_sample_half_size)
        timing["normal_estimate_sec"] = perf_counter() - step_start
    except Exception:
        debug = {
            "pipe_profile_target_point": anchor.tolist(),
            "normal_estimation_points": normal_points.tolist(),
            "normal_estimation_sampling_box": [
                (anchor - normal_sample_half_size).tolist(),
                (anchor + normal_sample_half_size).tolist(),
            ],
        }
        if log_timing:
            timing["total_sec"] = perf_counter() - total_start
            debug["timing"] = timing
        return None, None, debug

    axis = _unit(-normal_m)
    profile_params = _profile_sampling_defaults(bbox_diag, distance_threshold)
    step_start = perf_counter()
    points_in_cylinder, cylinder_projections = _extract_points_in_cylinder_with_projection(
        points,
        anchor,
        axis,
        profile_params["cylinder_radius"],
        profile_params["height_range"],
    )
    timing["cylinder_sample_sec"] = perf_counter() - step_start
    step_start = perf_counter()
    clusters = _cluster_points_with_projection(
        points_in_cylinder,
        cylinder_projections,
        profile_params["cluster_distance"],
    )
    timing["cluster_sec"] = perf_counter() - step_start
    if not clusters:
        debug = _profile_sampling_debug(
            anchor=anchor,
            normal_m=normal_m,
            normal_sample_half_size=normal_sample_half_size,
            normal_points=normal_points,
            axis=axis,
            profile_params=profile_params,
            points_in_cylinder=points_in_cylinder,
            clusters=[],
        )
        if log_timing:
            timing["total_sec"] = perf_counter() - total_start
            debug["timing"] = timing
        return None, None, debug

    step_start = perf_counter()
    farthest_point = _farthest_projected_point(clusters)
    estimated_center = (anchor + farthest_point) * 0.5
    estimated_radius = float(np.linalg.norm(farthest_point - estimated_center))
    sphere_radius = estimated_radius + profile_params["sphere_radius_offset"]
    timing["farthest_sec"] = perf_counter() - step_start
    step_start = perf_counter()
    sphere_points = _extract_points_in_sphere(points, estimated_center, sphere_radius)
    timing["sphere_sample_sec"] = perf_counter() - step_start

    debug = _profile_sampling_debug(
        anchor=anchor,
        normal_m=normal_m,
        normal_sample_half_size=normal_sample_half_size,
        normal_points=normal_points,
        axis=axis,
        profile_params=profile_params,
        points_in_cylinder=points_in_cylinder,
        clusters=[cluster_points for cluster_points, _ in clusters],
    )
    debug["estimated_center"] = estimated_center.tolist()
    debug["estimated_radius"] = estimated_radius
    debug["points_in_sphere"] = sphere_points.tolist()
    debug["estimated_opposite_point"] = farthest_point.tolist()

    if len(sphere_points) < min_points:
        if log_timing:
            timing["total_sec"] = perf_counter() - total_start
            debug["timing"] = timing
        return None, None, debug

    try:
        step_start = perf_counter()
        direction, center, radius, _ = fit_cylinder(sphere_points)
        timing["fit_cylinder_sec"] = perf_counter() - step_start
    except Exception:
        if log_timing:
            timing["total_sec"] = perf_counter() - total_start
            debug["timing"] = timing
        return None, None, debug

    if log_timing:
        timing["total_sec"] = perf_counter() - total_start
        debug["timing"] = timing
    return sphere_points, (_unit(direction), np.asarray(center, dtype=float), float(radius)), debug


def _extract_points_in_aabb(points: np.ndarray, center: np.ndarray, half_size: float) -> np.ndarray:
    min_bound = center - half_size
    max_bound = center + half_size
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    return points[mask]


def _estimate_local_normal(points: np.ndarray, sample_half_size: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(sample_half_size * 1.5, np.finfo(float).eps),
            max_nn=30,
        )
    )
    normals = np.asarray(pcd.normals, dtype=float)
    if len(normals) == 0:
        raise RuntimeError("Failed to estimate local normals.")
    return _unit(np.median(normals, axis=0))


def _profile_sampling_defaults(bbox_diag: float, distance_threshold: float) -> dict[str, Any]:
    if bbox_diag > 10.0:
        return {
            "cylinder_radius": max(5.0, distance_threshold),
            "height_range": (-100.0, 300.0),
            "cluster_distance": max(5.0, distance_threshold),
            "sphere_radius_offset": max(3.0, distance_threshold * 0.5),
        }
    return {
        "cylinder_radius": max(0.005, distance_threshold),
        "height_range": (-0.1, 0.3),
        "cluster_distance": max(0.005, distance_threshold),
        "sphere_radius_offset": max(0.003, distance_threshold * 0.5),
    }


def _extract_points_in_cylinder_with_projection(
    points: np.ndarray,
    start: np.ndarray,
    axis: np.ndarray,
    radius: float,
    height_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    axis = _unit(axis)
    vec = points - start
    projection = vec @ axis
    radial = vec - np.outer(projection, axis)
    mask = (
        (projection >= height_range[0])
        & (projection <= height_range[1])
        & (np.linalg.norm(radial, axis=1) <= radius)
    )
    return points[mask], projection[mask]


def _cluster_points_with_projection(
    points: np.ndarray,
    projections: np.ndarray,
    cluster_distance: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(points) == 0:
        return []

    order = np.argsort(projections)
    points_sorted = points[order]
    projections_sorted = projections[order]
    clusters: list[list[np.ndarray]] = [[points_sorted[0]]]
    cluster_projections: list[list[float]] = [[float(projections_sorted[0])]]

    for idx in range(1, len(points_sorted)):
        if abs(float(projections_sorted[idx] - projections_sorted[idx - 1])) <= cluster_distance:
            clusters[-1].append(points_sorted[idx])
            cluster_projections[-1].append(float(projections_sorted[idx]))
        else:
            clusters.append([points_sorted[idx]])
            cluster_projections.append([float(projections_sorted[idx])])

    return [
        (np.asarray(cluster_points, dtype=float), np.asarray(cluster_proj, dtype=float))
        for cluster_points, cluster_proj in zip(clusters, cluster_projections)
    ]


def _farthest_projected_point(clusters: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    farthest_cluster, farthest_projection = max(
        clusters,
        key=lambda item: float(np.max(np.abs(item[1]))),
    )
    farthest_index = int(np.argmax(np.abs(farthest_projection)))
    return farthest_cluster[farthest_index]


def _extract_points_in_sphere(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    distances = np.linalg.norm(points - center, axis=1)
    return points[distances <= radius]


def _print_timing(label: str, elapsed_sec: float, details: dict[str, Any] | None = None) -> None:
    detail_text = ""
    if details:
        detail_parts = []
        for key, value in details.items():
            if isinstance(value, float):
                detail_parts.append(f"{key}={value:.4f}")
            else:
                detail_parts.append(f"{key}={value}")
        detail_text = " | " + ", ".join(detail_parts)
    print(f"[timing] {label}: {elapsed_sec:.3f}s{detail_text}")


def _profile_sampling_debug(
    *,
    anchor: np.ndarray,
    normal_m: np.ndarray,
    normal_sample_half_size: float,
    normal_points: np.ndarray,
    axis: np.ndarray,
    profile_params: dict[str, Any],
    points_in_cylinder: np.ndarray,
    clusters: list[np.ndarray],
) -> dict[str, Any]:
    return {
        "pipe_profile_target_point": anchor.tolist(),
        "pipe_profile_normal_axis": axis.tolist(),
        "normal_m": normal_m.tolist(),
        "normal_estimation_points": normal_points.tolist(),
        "normal_estimation_sampling_box": [
            (anchor - normal_sample_half_size).tolist(),
            (anchor + normal_sample_half_size).tolist(),
        ],
        "pipe_profile_sampling_cylinder": {
            "start": anchor.tolist(),
            "axis": axis.tolist(),
            "radius": float(profile_params["cylinder_radius"]),
            "height_range": list(profile_params["height_range"]),
        },
        "pipe_profile_points_in_cylinder": points_in_cylinder.tolist(),
        "pipe_profile_clusters": [cluster.tolist() for cluster in clusters],
    }


def _serialize_sampling_debug(debug: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "pipe_profile_target_point",
        "pipe_profile_normal_axis",
        "normal_m",
        "estimated_center",
        "estimated_radius",
    ):
        if key not in debug:
            continue
        value = debug[key]
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (float, int, str)):
            result[key] = value

    if "pipe_profile_normal_axis" not in result and "normal_m" in result:
        normal_m = np.asarray(result["normal_m"], dtype=float)
        if normal_m.shape == (3,) and np.isfinite(normal_m).all():
            result["pipe_profile_normal_axis"] = (-normal_m).tolist()

    for key in ("pipe_profile_points_in_cylinder", "points_in_sphere"):
        if key in debug:
            result[key] = np.asarray(debug[key], dtype=float).tolist()

    selected_points = debug.get("selected_points")
    if selected_points is not None and hasattr(selected_points, "points"):
        result["normal_estimation_points"] = np.asarray(selected_points.points, dtype=float).tolist()

    sampling_box = debug.get("sampling_box")
    if sampling_box is not None:
        result["normal_estimation_sampling_box"] = [
            np.asarray(bound, dtype=float).tolist()
            for bound in sampling_box
        ]

    sampling_cylinder = debug.get("pipe_profile_sampling_cylinder")
    if sampling_cylinder is not None:
        result["pipe_profile_sampling_cylinder"] = {
            "start": np.asarray(sampling_cylinder["start"], dtype=float).tolist(),
            "axis": np.asarray(sampling_cylinder["axis"], dtype=float).tolist(),
            "radius": float(sampling_cylinder["radius"]),
            "height_range": list(sampling_cylinder["height_range"]),
        }
        result.setdefault("pipe_profile_target_point", result["pipe_profile_sampling_cylinder"]["start"])
        result.setdefault("pipe_profile_normal_axis", result["pipe_profile_sampling_cylinder"]["axis"])

    clusters = debug.get("pipe_profile_clusters")
    if clusters is not None:
        result["pipe_profile_clusters"] = [
            np.asarray(cluster, dtype=float).tolist()
            for cluster in clusters
        ]
    return result


def _refine_cylinder(
    points: np.ndarray,
    fallback_model: tuple[np.ndarray, np.ndarray, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(points) < 20:
        return fallback_model

    try:
        axis, axis_point, radius, _ = fit_cylinder(points)
        axis = _unit(np.asarray(axis, dtype=float))
        axis_point = np.asarray(axis_point, dtype=float)
        if np.dot(axis, fallback_model[0]) < 0:
            axis = -axis
        return axis, axis_point, float(radius)
    except Exception:
        return fallback_model


def _cylinder_radial_residuals(
    points: np.ndarray,
    axis: np.ndarray,
    axis_point: np.ndarray,
    radius: float,
) -> np.ndarray:
    axis = _unit(axis)
    rel = points - axis_point
    axial = np.outer(rel @ axis, axis)
    radial_distance = np.linalg.norm(rel - axial, axis=1)
    return np.abs(radial_distance - radius)


def _segment_axis_endpoints(
    points: np.ndarray,
    axis: np.ndarray,
    axis_point: np.ndarray,
    endpoint_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    axis = _unit(axis)
    t = (points - axis_point) @ axis
    lo = float(np.quantile(t, endpoint_quantile))
    hi = float(np.quantile(t, 1.0 - endpoint_quantile))
    return axis_point + lo * axis, axis_point + hi * axis


def _classify_pipe_endpoints(
    segments: list[CylinderSegmentProfile],
    distance_threshold: float,
    connection_threshold: float | None,
) -> dict[str, Any]:
    endpoint_refs = []
    for segment_index, segment in enumerate(segments):
        for endpoint_index, endpoint in enumerate(segment.endpoints):
            endpoint_refs.append((segment_index, endpoint_index, endpoint))

    if len(segments) == 1:
        return {
            "connection_threshold": 0.0,
            "joint_endpoint_pairs": [],
            "terminal_ends": [
                {"segment_index": 0, "endpoint_index": 0},
                {"segment_index": 0, "endpoint_index": 1},
            ],
        }

    if connection_threshold is None:
        median_radius = float(np.median([segment.radius for segment in segments]))
        connection_threshold = max(median_radius * 4.0, distance_threshold * 8.0)

    degrees = [0 for _ in endpoint_refs]
    joint_endpoint_pairs = []
    for i in range(len(endpoint_refs)):
        for j in range(i + 1, len(endpoint_refs)):
            if endpoint_refs[i][0] == endpoint_refs[j][0]:
                continue
            distance = float(np.linalg.norm(endpoint_refs[i][2] - endpoint_refs[j][2]))
            if distance <= connection_threshold:
                degrees[i] += 1
                degrees[j] += 1
                joint_endpoint_pairs.append(
                    {
                        "distance": distance,
                        "endpoints": [
                            {
                                "segment_index": endpoint_refs[i][0],
                                "endpoint_index": endpoint_refs[i][1],
                                "center": endpoint_refs[i][2].tolist(),
                            },
                            {
                                "segment_index": endpoint_refs[j][0],
                                "endpoint_index": endpoint_refs[j][1],
                                "center": endpoint_refs[j][2].tolist(),
                            },
                        ],
                    }
                )

    terminal_indices = [idx for idx, degree in enumerate(degrees) if degree == 0]
    if len(terminal_indices) < 2:
        terminal_indices = _two_farthest_endpoint_indices(endpoint_refs)
    elif len(terminal_indices) > 2:
        subset_pair = _two_farthest_endpoint_indices([endpoint_refs[idx] for idx in terminal_indices])
        terminal_indices = [terminal_indices[idx] for idx in subset_pair]

    terminal_ends = [
        {
            "segment_index": endpoint_refs[idx][0],
            "endpoint_index": endpoint_refs[idx][1],
        }
        for idx in terminal_indices
    ]
    return {
        "connection_threshold": float(connection_threshold),
        "joint_endpoint_pairs": joint_endpoint_pairs,
        "terminal_ends": terminal_ends,
    }


def _two_farthest_endpoint_indices(endpoint_refs: list[tuple[int, int, np.ndarray]]) -> list[int]:
    best_pair = (0, 1)
    best_distance = -1.0
    for i in range(len(endpoint_refs)):
        for j in range(i + 1, len(endpoint_refs)):
            distance = float(np.linalg.norm(endpoint_refs[i][2] - endpoint_refs[j][2]))
            if distance > best_distance:
                best_distance = distance
                best_pair = (i, j)
    return [best_pair[0], best_pair[1]]


def _voxel_graph_nodes(
    points: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
    min_voxel_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voxel_indices = np.floor((points - origin) / voxel_size).astype(np.int64)
    voxel_accum: dict[tuple[int, int, int], list[Any]] = {}
    for key_arr, point in zip(voxel_indices, points):
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
        if key not in voxel_accum:
            voxel_accum[key] = [np.zeros(3, dtype=float), 0]
        voxel_accum[key][0] += point
        voxel_accum[key][1] += 1

    keys: list[tuple[int, int, int]] = []
    centers: list[np.ndarray] = []
    counts: list[int] = []
    for key, (point_sum, count) in voxel_accum.items():
        if int(count) < min_voxel_points:
            continue
        keys.append(key)
        centers.append(np.asarray(point_sum, dtype=float) / float(count))
        counts.append(int(count))

    if not centers:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    order = np.lexsort((
        np.asarray([k[2] for k in keys]),
        np.asarray([k[1] for k in keys]),
        np.asarray([k[0] for k in keys]),
    ))
    return (
        np.asarray(centers, dtype=float)[order],
        np.asarray(keys, dtype=np.int64)[order],
        np.asarray(counts, dtype=np.int64)[order],
    )


def _build_voxel_graph_edges(
    nodes: np.ndarray,
    voxel_keys: np.ndarray,
    voxel_size: float,
    neighbor_radius_voxels: int,
) -> tuple[list[list[tuple[int, float]]], list[tuple[int, int, float]]]:
    key_to_index = {tuple(map(int, key)): i for i, key in enumerate(voxel_keys)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(nodes))]
    edges: list[tuple[int, int, float]] = []
    offsets = [
        (dx, dy, dz)
        for dx in range(-neighbor_radius_voxels, neighbor_radius_voxels + 1)
        for dy in range(-neighbor_radius_voxels, neighbor_radius_voxels + 1)
        for dz in range(-neighbor_radius_voxels, neighbor_radius_voxels + 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    max_key_distance = float(np.sqrt(3.0) * neighbor_radius_voxels) + 1e-9

    for i, key_arr in enumerate(voxel_keys):
        key = tuple(map(int, key_arr))
        for offset in offsets:
            offset_norm = float(np.linalg.norm(offset))
            if offset_norm > max_key_distance:
                continue
            neighbor_key = (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
            j = key_to_index.get(neighbor_key)
            if j is None or j <= i:
                continue
            weight = float(np.linalg.norm(nodes[i] - nodes[j]))
            if weight <= voxel_size * (neighbor_radius_voxels + 0.75):
                adjacency[i].append((j, weight))
                adjacency[j].append((i, weight))
                edges.append((i, j, weight))
    return adjacency, edges


def _largest_connected_component(adjacency: list[list[tuple[int, float]]]) -> list[int]:
    visited = np.zeros(len(adjacency), dtype=bool)
    best_component: list[int] = []
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor, _ in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if len(component) > len(best_component):
            best_component = component
    return sorted(best_component)


def _dijkstra_graph(
    adjacency: list[list[tuple[int, float]]],
    start: int,
    allowed: list[int] | set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    allowed_set = set(allowed) if allowed is not None else set(range(len(adjacency)))
    distances = np.full(len(adjacency), np.inf, dtype=float)
    previous = np.full(len(adjacency), -1, dtype=np.int64)
    if start not in allowed_set:
        return distances, previous

    distances[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > distances[node]:
            continue
        for neighbor, weight in adjacency[node]:
            if neighbor not in allowed_set:
                continue
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(heap, (candidate, neighbor))
    return distances, previous


def _farthest_reachable_node(distances: np.ndarray) -> int:
    finite = np.where(np.isfinite(distances))[0]
    if len(finite) == 0:
        raise RuntimeError("No reachable graph node found.")
    return int(finite[np.argmax(distances[finite])])


def _reconstruct_path(previous: np.ndarray, start: int, end: int) -> list[int]:
    path = [int(end)]
    current = int(end)
    while current != int(start):
        current = int(previous[current])
        if current < 0:
            return []
        path.append(current)
    path.reverse()
    return path


def _circle_profile_points(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    count: int,
) -> np.ndarray:
    axis = _unit(axis)
    basis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, basis))) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])

    v1 = _unit(np.cross(axis, basis))
    v2 = _unit(np.cross(axis, v1))
    angles = np.linspace(0.0, 2.0 * np.pi, max(3, count), endpoint=False)
    return center + radius * (np.outer(np.cos(angles), v1) + np.outer(np.sin(angles), v2))


def _subsample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < np.finfo(float).eps:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm
