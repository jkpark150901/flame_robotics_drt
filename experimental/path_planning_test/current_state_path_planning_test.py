#!/usr/bin/env python3
"""
Path planning smoke test from the current pipe/positioner state.

This script mirrors the spool transform convention used by simtool/viewervedo:
  1. load spool geometry
  2. scale from mm to m by default
  3. place it at the viewer's default spool origin
  4. apply the saved sidecar pose (<spool-stem>.json), if present

Only end-effector point collision is checked by the current planner plugins.
Robot link collision is intentionally not modeled here.

Example:
  python experimental/path_planning_test/current_state_path_planning_test.py \
      --spool sample/MERGED_SPOOL-004_0001_20260528T170812.pcd \
      --planner rrt_connect --step-size 0.08 --max-iter 3000 --vis
"""

import argparse
import importlib
import inspect
import json
import pathlib
import sys
import time
import types
from typing import Optional

import numpy as np

# Open3D 0.19 imports open3d.ml unconditionally. In this workspace the optional
# ML stack can fail due NumPy/SciPy ABI mismatch, but path planning only needs
# core geometry/raycasting. Stubbing keeps the import focused on core Open3D.
sys.modules.setdefault("open3d.ml", types.ModuleType("open3d.ml"))
import open3d as o3d


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PY_ROOT = REPO_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))


DEFAULT_SPOOL_ORIGIN = np.array([7.311, 1.877, 1.213], dtype=float)


def load_planner(plugin_name: str):
    """Load a path planner plugin instance by module name."""
    from plugins.pluginbase.plannerbase import PlannerBase

    module = importlib.import_module(f"plugins.pathplanner.{plugin_name}")
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, PlannerBase) and obj is not PlannerBase:
            return obj()
    raise RuntimeError(f"Planner plugin class not found: {plugin_name}")


def pose_path_for_spool(spool_path: pathlib.Path) -> pathlib.Path:
    return spool_path.with_suffix(".json")


def load_state(spool_path: pathlib.Path, state_path: Optional[pathlib.Path]):
    path = state_path if state_path else pose_path_for_spool(spool_path)
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), path
    return {}, path


def load_spool_geometry(path: pathlib.Path):
    suffix = path.suffix.lower()
    if suffix in (".pcd", ".xyz", ".pts"):
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            raise RuntimeError(f"Empty point cloud: {path}")
        return pcd, "pcd"

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.has_triangles():
        mesh.compute_vertex_normals()
        return mesh, "mesh"

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"Unsupported or empty geometry: {path}")
    return pcd, "pcd"


def rotate_geometry_xyz(geom, center, x_deg=0.0, z_deg=0.0):
    if abs(x_deg) > 1e-12:
        rx = o3d.geometry.get_rotation_matrix_from_axis_angle([np.deg2rad(x_deg), 0.0, 0.0])
        geom.rotate(rx, center=center)
    if abs(z_deg) > 1e-12:
        rz = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, np.deg2rad(z_deg)])
        geom.rotate(rz, center=center)


def apply_viewer_spool_transform(geom, state, scale: float):
    """Reproduce viewervedo's load_spool + move_spool transform."""
    spool_pose = state.get("spool", state)
    target = np.array([
        float(spool_pose.get("x", DEFAULT_SPOOL_ORIGIN[0])),
        float(spool_pose.get("y", DEFAULT_SPOOL_ORIGIN[1])),
        float(spool_pose.get("z", DEFAULT_SPOOL_ORIGIN[2])),
    ], dtype=float)
    x_rot = float(spool_pose.get("x_rotation", 0.0))
    z_rot = float(spool_pose.get("z_rotation", 0.0))

    geom.scale(scale, center=(0.0, 0.0, 0.0))
    geom.translate(DEFAULT_SPOOL_ORIGIN)
    rz = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, np.deg2rad(-90.0)])
    geom.rotate(rz, center=DEFAULT_SPOOL_ORIGIN)

    geom.translate(target - DEFAULT_SPOOL_ORIGIN)
    rotate_geometry_xyz(geom, target, x_deg=x_rot, z_deg=z_rot)
    return {
        "x": float(target[0]),
        "y": float(target[1]),
        "z": float(target[2]),
        "x_rotation": x_rot,
        "z_rotation": z_rot,
    }


def point_cloud_to_collision_mesh(pcd, mode: str, alpha: float, voxel_size: float):
    if mode == "convex":
        mesh, _ = pcd.compute_convex_hull()
        mesh.compute_vertex_normals()
        return mesh

    if mode == "alpha":
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            if mesh.has_triangles():
                mesh.compute_vertex_normals()
                return mesh
        except Exception as exc:
            print(f"[warn] alpha shape failed ({exc}); falling back to convex hull")
        mesh, _ = pcd.compute_convex_hull()
        mesh.compute_vertex_normals()
        return mesh

    if mode == "voxel":
        down = pcd.voxel_down_sample(voxel_size)
        pts = np.asarray(down.points)
        mesh = o3d.geometry.TriangleMesh()
        for p in pts:
            box = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)
            box.translate(p - voxel_size / 2.0)
            mesh += box
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.compute_vertex_normals()
        return mesh

    raise ValueError(f"Unknown obstacle mode: {mode}")


def build_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def verify_ef_path(scene, path, clearance: float, samples_per_segment: int):
    poses = np.asarray(path, dtype=float)
    pts = []
    ray_hits = 0
    min_distance = float("inf")

    for a_pose, b_pose in zip(poses[:-1], poses[1:]):
        a = a_pose[:3]
        b = b_pose[:3]
        d = b - a
        length = np.linalg.norm(d)
        if length > 1e-9:
            direction = d / length
            rays = o3d.core.Tensor([[*a, *direction]], dtype=o3d.core.Dtype.Float32)
            t_hit = scene.cast_rays(rays)["t_hit"][0].item()
            if np.isfinite(t_hit) and t_hit < length:
                ray_hits += 1
        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            pts.append(a + t * d)
    pts.append(poses[-1, :3])

    query = o3d.core.Tensor(np.asarray(pts), dtype=o3d.core.Dtype.Float32)
    distances = scene.compute_distance(query).numpy()
    if len(distances):
        min_distance = float(np.min(distances))
    clearance_hits = int(np.count_nonzero(distances < clearance))

    return {
        "ray_hits": ray_hits,
        "clearance_hits": clearance_hits,
        "min_distance": min_distance,
        "sample_count": len(pts),
    }


def configure_planner(planner, mesh, args):
    mn = np.asarray(mesh.get_min_bound())
    mx = np.asarray(mesh.get_max_bound())
    ext = np.maximum(mx - mn, 1e-6)
    pad = np.maximum(ext * args.bounds_margin, args.min_bounds_pad)
    bounds = {
        "x_min": float(mn[0] - pad[0]),
        "x_max": float(mx[0] + pad[0]),
        "y_min": float(mn[1] - pad[1]),
        "y_max": float(mx[1] + pad[1]),
        "z_min": float(mn[2] - pad[2]),
        "z_max": float(mx[2] + pad[2]),
        "roll_min": -np.pi,
        "roll_max": np.pi,
        "pitch_min": -np.pi,
        "pitch_max": np.pi,
        "yaw_min": -np.pi,
        "yaw_max": np.pi,
    }
    if hasattr(planner, "bounds"):
        planner.bounds = bounds
    if args.step_size is not None and hasattr(planner, "step_size"):
        planner.step_size = args.step_size
    if args.max_iter is not None:
        if hasattr(planner, "max_iter"):
            planner.max_iter = args.max_iter
        if hasattr(planner, "max_iterations"):
            planner.max_iterations = args.max_iter
    planner.add_collision_object(mesh)
    return bounds


def default_start_goal(mesh, clearance):
    mn = np.asarray(mesh.get_min_bound())
    mx = np.asarray(mesh.get_max_bound())
    center = (mn + mx) / 2.0
    ext = mx - mn
    start = np.array([mx[0] + clearance, center[1], center[2], 0.0, 0.0, 0.0])
    goal = np.array([mn[0] - clearance, center[1], center[2], 0.0, 0.0, 0.0])
    if ext[0] < ext[1]:
        start[:3] = [center[0], mn[1] - clearance, center[2]]
        goal[:3] = [center[0], mx[1] + clearance, center[2]]
    return start, goal


def write_path(path, output_path, metadata):
    payload = {
        "metadata": metadata,
        "path": [np.asarray(p, dtype=float).tolist() for p in path],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def visualize(mesh, path, start, goal):
    mesh_vis = o3d.geometry.TriangleMesh(mesh)
    mesh_vis.paint_uniform_color([0.55, 0.55, 0.55])
    pts = np.asarray([p[:3] for p in path], dtype=float)
    lines = [[i, i + 1] for i in range(max(0, len(pts) - 1))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color([0.1, 0.85, 0.15])
    geoms = [mesh_vis, line_set]
    for pose, color in ((start, [1, 0, 0]), (goal, [0, 0.2, 1])):
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        marker.translate(np.asarray(pose[:3], dtype=float))
        marker.paint_uniform_color(color)
        geoms.append(marker)
    o3d.visualization.draw_geometries(geoms, window_name="current-state path planning test")


def main():
    ap = argparse.ArgumentParser(description="Path planning test using saved pipe/positioner state")
    ap.add_argument("--spool", default=str(REPO_ROOT / "sample" / "MERGED_SPOOL-004_0001_20260528T170812.pcd"))
    ap.add_argument("--state", default=None, help="Pose JSON. Defaults to <spool-stem>.json")
    ap.add_argument("--planner", default="rrt_connect")
    ap.add_argument("--start", nargs=6, type=float, default=None, metavar="V")
    ap.add_argument("--goal", nargs=6, type=float, default=None, metavar="V")
    ap.add_argument("--scale", type=float, default=0.001, help="Scale raw spool geometry before applying saved pose")
    ap.add_argument("--obstacle-mode", choices=["alpha", "convex", "voxel"], default="alpha")
    ap.add_argument("--alpha", type=float, default=0.06, help="Alpha shape radius in planner units")
    ap.add_argument("--voxel-size", type=float, default=0.03, help="Voxel box size for voxel obstacle mode")
    ap.add_argument("--step-size", type=float, default=0.08)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--bounds-margin", type=float, default=0.8)
    ap.add_argument("--min-bounds-pad", type=float, default=0.5)
    ap.add_argument("--auto-clearance", type=float, default=0.5)
    ap.add_argument("--verify-clearance", type=float, default=0.0)
    ap.add_argument("--samples-per-segment", type=int, default=12)
    ap.add_argument("--output", default=None)
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()

    spool_path = pathlib.Path(args.spool).expanduser().resolve()
    state_path = pathlib.Path(args.state).expanduser().resolve() if args.state else None
    if not spool_path.exists():
        raise FileNotFoundError(spool_path)

    state, resolved_state_path = load_state(spool_path, state_path)
    geom, geom_type = load_spool_geometry(spool_path)
    applied_pose = apply_viewer_spool_transform(geom, state, args.scale)

    if geom_type == "mesh":
        obstacle_mesh = geom
    else:
        obstacle_mesh = point_cloud_to_collision_mesh(
            geom,
            mode=args.obstacle_mode,
            alpha=args.alpha,
            voxel_size=args.voxel_size,
        )

    obstacle_mesh.compute_vertex_normals()
    mn = np.asarray(obstacle_mesh.get_min_bound())
    mx = np.asarray(obstacle_mesh.get_max_bound())
    print(f"spool: {spool_path}")
    print(f"state: {resolved_state_path if resolved_state_path and resolved_state_path.exists() else '(none)'}")
    print(f"spool pose: {applied_pose}")
    if state.get("positioner"):
        print(f"positioner pose: {state['positioner']} (not used for robot-link collision)")
    print(f"obstacle mesh bbox min={np.round(mn, 4)} max={np.round(mx, 4)}")

    start, goal = default_start_goal(obstacle_mesh, args.auto_clearance)
    if args.start is not None:
        start = np.array(args.start, dtype=float)
    if args.goal is not None:
        goal = np.array(args.goal, dtype=float)

    planner = load_planner(args.planner)
    bounds = configure_planner(planner, obstacle_mesh, args)
    print(f"planner: {type(planner).__name__}")
    print(f"bounds: {bounds}")
    print(f"start: {np.round(start, 4)}")
    print(f"goal : {np.round(goal, 4)}")

    t0 = time.time()
    path = planner.generate(start, goal)
    elapsed = time.time() - t0
    if not path:
        print(f"[FAIL] no path found ({elapsed:.2f}s)")
        return 2

    path = [np.asarray(p, dtype=float) for p in path]
    scene = build_scene(obstacle_mesh)
    verification = verify_ef_path(scene, path, args.verify_clearance, args.samples_per_segment)
    ok = verification["ray_hits"] == 0 and verification["clearance_hits"] == 0
    print(f"[OK] path found: {len(path)} waypoints ({elapsed:.2f}s)")
    print(f"verify: {verification}")
    print(f"result: {'collision-free EF path' if ok else 'collision/clearance issue detected'}")

    output_path = pathlib.Path(args.output) if args.output else (
        REPO_ROOT / "experimental" / "path_planning_test" / f"{spool_path.stem}_{args.planner}_path.json"
    )
    metadata = {
        "spool": str(spool_path),
        "state": str(resolved_state_path) if resolved_state_path else None,
        "planner": args.planner,
        "spool_pose": applied_pose,
        "positioner": state.get("positioner"),
        "verification": verification,
        "robot_links_considered": False,
    }
    write_path(path, output_path, metadata)
    print(f"saved path: {output_path}")

    if args.vis:
        visualize(obstacle_mesh, path, start, goal)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
