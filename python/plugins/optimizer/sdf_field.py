"""Cached unsigned distance fields for IK and APF costs.

The scanned pipe uses occupied point-cloud voxels and a Euclidean distance
transform, without triangle-distance queries. CAD obstacles retain mesh-based
fields so large planar faces are represented. Robot collision surfaces are
sampled in each link frame; FK and trilinear lookup give approximate distances.
Scan holes, voxel resolution and robot surface sampling affect accuracy. These
fields do not certify collision freedom or determine whether a point is inside
an obstacle. PyBullet mesh checks remain the final collision authority.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from scipy.ndimage import map_coordinates, distance_transform_edt
from scipy.spatial.transform import Rotation

DEFAULT_RESOLUTION = 0.03    # m per voxel
DEFAULT_CAP = 0.35           # m, SDF is clipped to +-cap (must be >= the largest APF d0 in use)
_CHUNK = 1_000_000


class StaticSDF:
    """Voxel distance field of one obstacle mesh, clipped to cap.

    Unsigned (distance to the surface): the obstacle meshes are unions of overlapping primitives (positioner) or scans,
    not watertight, so the ray-parity sign of a signed field is wrong far from the surface (a rail 0.4 m from the
    positioner came out "inside" it). The APF cost only needs "how close", and a penetrating link reads ~0 -> the
    potential saturates at its MIN_DISTANCE floor anyway."""

    def __init__(self, mesh, resolution: float = DEFAULT_RESOLUTION, cap: float = DEFAULT_CAP):
        self.resolution = float(resolution)
        self.cap = float(cap)
        verts = np.asarray(mesh.vertices, dtype=float)
        lo = verts.min(axis=0) - self.cap - 2 * self.resolution
        hi = verts.max(axis=0) + self.cap + 2 * self.resolution
        self.origin = lo
        self.shape = tuple(int(np.ceil((hi[i] - lo[i]) / self.resolution)) + 1 for i in range(3))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        axes = [lo[i] + self.resolution * np.arange(self.shape[i]) for i in range(3)]
        grid = np.empty(self.shape, dtype=np.float32)
        yy, zz = np.meshgrid(axes[1], axes[2], indexing="ij")
        plane = np.stack([np.zeros_like(yy), yy, zz], axis=-1).reshape(-1, 3)
        per_slab = max(1, _CHUNK // len(plane))
        for start in range(0, self.shape[0], per_slab):
            xs = axes[0][start:start + per_slab]
            pts = np.repeat(plane[None], len(xs), axis=0)
            pts[:, :, 0] = xs[:, None]
            q = o3d.core.Tensor(pts.reshape(-1, 3).astype(np.float32))
            d = scene.compute_distance(q).numpy().reshape(len(xs), self.shape[1], self.shape[2])
            grid[start:start + len(xs)] = d
        self.grid = np.clip(grid, -self.cap, self.cap)
        self.bounds = (lo, lo + self.resolution * (np.array(self.shape) - 1))

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Distance (m) to the surface at points (N,3); cap outside the grid."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        idx = (pts - self.origin) / self.resolution
        inside = np.all((idx >= 0) & (idx <= np.array(self.shape) - 1), axis=1)
        out = np.full(len(pts), self.cap, dtype=float)
        if inside.any():
            out[inside] = map_coordinates(self.grid, idx[inside].T, order=1, mode="nearest")
        return out


class PointCloudDistanceField(StaticSDF):
    """Unsigned voxel distance to input points, with no triangle BVH/reconstruction.

    Sparse/missing scan regions can overestimate clearance. This is a cost
    approximation only; PyBullet remains the collision acceptance authority.
    """
    def __init__(self, points, resolution=DEFAULT_RESOLUTION, cap=DEFAULT_CAP):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points) or resolution <= 0 or cap <= 0:
            raise ValueError("finite obstacle points, positive resolution and cap required")
        self.resolution, self.cap = float(resolution), float(cap)
        self.origin = points.min(axis=0) - cap - 2 * resolution
        hi = points.max(axis=0) + cap + 2 * resolution
        self.shape = tuple((np.ceil((hi - self.origin) / resolution).astype(int) + 1).tolist())
        # Rasterize scan samples then compute an exact Euclidean transform of
        # occupied voxels. Avoid a KD-tree query for every grid cell.
        empty = np.ones(self.shape, dtype=bool)
        occupied = np.rint((points - self.origin) / resolution).astype(np.int64)
        empty[tuple(occupied.T)] = False
        distances = distance_transform_edt(empty, sampling=resolution)
        # Account for rounding a scan point to its voxel centre. Scan holes and
        # robot surface sampling still require exact collision verification.
        distances = np.maximum(0.0, distances - np.sqrt(3.0) * resolution / 2)
        self.grid = np.minimum(distances, cap).astype(np.float32)
        self.bounds = (self.origin, self.origin + resolution * (np.array(self.shape) - 1))


def _mesh_for_geometry(geom, urdf_dir: str):
    kind = type(geom).__name__
    if kind == "Box":
        size = [float(v) for v in geom.size]
        m = o3d.geometry.TriangleMesh.create_box(*size)
        m.translate(-np.array(size) / 2.0)
        return m
    if kind == "Cylinder":
        m = o3d.geometry.TriangleMesh.create_cylinder(radius=float(geom.radius), height=float(geom.length),
                                                      resolution=24, split=4)
        return m
    if kind == "Sphere":
        return o3d.geometry.TriangleMesh.create_sphere(radius=float(geom.radius), resolution=12)
    if kind == "Mesh":
        path = str(geom.filename).replace("file://", "")
        if not os.path.isabs(path):
            path = os.path.join(urdf_dir, path)
        m = o3d.io.read_triangle_mesh(path)
        scale = getattr(geom, "scale", None)
        if scale is not None:
            s = np.asarray(scale, dtype=float).reshape(-1)
            v = np.asarray(m.vertices) * (s if s.size == 3 else s[0])
            m.vertices = o3d.utility.Vector3dVector(v)
        return m
    return None


def link_surface_points(urdf_path: str, points_per_geometry: int = 250) -> Dict[str, np.ndarray]:
    """{link name: (N,3) points on the link's collision surface, in the link frame}."""
    from urdf_parser_py.urdf import URDF
    # bytes, not the file path: the urdf_parser_py of ROS 2 Humble (on PYTHONPATH once ROS is sourced) parses with lxml,
    # which rejects a str that carries an XML encoding declaration (our URDFs start with <?xml ... encoding='utf-8'?>)
    with open(urdf_path, "rb") as f:
        urdf = URDF.from_xml_string(f.read())
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    out: Dict[str, np.ndarray] = {}
    for link in urdf.links:
        collisions = list(getattr(link, "collisions", None) or ([link.collision] if getattr(link, "collision", None) else []))
        chunks = []
        for col in collisions:
            if col is None or col.geometry is None:
                continue
            mesh = _mesh_for_geometry(col.geometry, urdf_dir)
            if mesh is None or not mesh.has_triangles():
                continue
            if getattr(col, "origin", None) is not None:
                T = np.eye(4)
                T[:3, :3] = Rotation.from_euler("xyz", col.origin.rpy).as_matrix()
                T[:3, 3] = col.origin.xyz
                mesh.transform(T)
            n = points_per_geometry * (2 if type(col.geometry).__name__ == "Mesh" else 1)
            chunks.append(np.asarray(mesh.sample_points_uniformly(number_of_points=n).points))
        if chunks:
            out[str(link.name)] = np.vstack(chunks)
    return out


class SceneDistanceField:
    """Hybrid unsigned fields: pipe point voxels, CAD mesh distances, robot link samples."""

    def __init__(self, pin_module, model, urdf_path: str, static_meshes: Sequence[Any],
                 resolution: float = DEFAULT_RESOLUTION, cap: float = DEFAULT_CAP, pipe_points=None):
        self.pin = pin_module
        self.model = model
        self.data = model.createData()
        self.cap = float(cap)
        self.names = [f"collision_object_{i}" for i in range(len(static_meshes))]
        # Scanned pipe is obstacle 0. Older snapshots only contain mesh vertices;
        # use those as a point set without querying triangles. CAD obstacles stay
        # mesh-based (their few vertices do not describe continuous box surfaces).
        self.sdfs = [
            PointCloudDistanceField(
                np.asarray(m.vertices) if pipe_points is None else pipe_points, resolution, cap)
            if i == 0 else StaticSDF(m, resolution, cap)
            for i, m in enumerate(static_meshes)]
        self.pipe_source = "scan_points" if pipe_points is not None else "snapshot_vertices"

        self.link_points = link_surface_points(urdf_path)
        self.frame_ids = {name: model.getFrameId(name) for name in self.link_points}
        # all link samples in one array so a query is one FK, one batched transform and one lookup per obstacle
        self._links = list(self.link_points)
        self._pts = np.vstack([self.link_points[l] for l in self._links])
        counts = [len(self.link_points[l]) for l in self._links]
        self._starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
        self._link_of_point = np.repeat(np.arange(len(self._links)), counts)
        self._fids = [self.frame_ids[l] for l in self._links]

    def distances(self, q: Sequence[float]) -> List[Dict[str, Any]]:
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, np.asarray(q, dtype=float))
        pin.updateFramePlacements(self.model, self.data)
        R = np.stack([np.asarray(self.data.oMf[f].rotation) for f in self._fids])
        t = np.stack([np.asarray(self.data.oMf[f].translation) for f in self._fids])
        world = np.einsum("nij,nj->ni", R[self._link_of_point], self._pts) + t[self._link_of_point]
        entries = []
        for name, sdf in zip(self.names, self.sdfs):
            per_link = np.minimum.reduceat(sdf.distance(world), self._starts)
            for link, d in zip(self._links, per_link):
                entries.append({"link": f"{link}_0", "obstacle": name, "distance": float(d)})
        return entries
