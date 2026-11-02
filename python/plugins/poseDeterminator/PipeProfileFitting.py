"""Pipe profile (local cylinder) fitting for EF pose determination.

Two methods share one parameter set (PipeProfileParams, configured under ``ef_pose.pipe_profile``):

- ``ransac`` (default): fit a cylinder to the scan points around the selected surface point with RANSAC
  (two oriented points -> axis/radius hypothesis), then refine on the inliers. It does not depend on the
  accuracy of the local normal, uses real radius bounds, and the result is validated.
- ``cluster`` (legacy): fire a thin ROI cylinder along the inverse local normal, cluster the hit points,
  take the far end of the second cluster as the opposite wall and least-squares fit a cylinder to the
  points in a sphere around the estimated center. It only works when the normal is accurate to within a few
  degrees (ROI radius vs. pipe diameter) and has no outlier rejection.

Every fit is validated (radius bounds, inlier count/ratio, angular coverage, anchor residual); a failed
validation raises PipeProfileError with the reasons instead of silently producing poses around a wrong cylinder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PipeProfileError(RuntimeError):
    """Pipe profile fitting/validation failure. ``reasons`` lists every violated check."""

    def __init__(self, message: str, reasons: Optional[List[str]] = None):
        super().__init__(message)
        self.reasons = list(reasons or [])


@dataclass
class PipeProfileParams:
    """All pipe-profile fitting parameters (units: meters / degrees)."""

    method: str = "ransac"                      # "ransac" | "cluster"
    seed: Optional[int] = 0                     # RANSAC seed; None = non-deterministic

    # --- local normal at the selected point (both methods) ---
    normal_sampling_size: float = 0.01          # half size of the box around the surface point whose normals are median-ed

    # --- RANSAC ---
    neighborhood_radius: float = 0.25           # scan points within this distance of the surface point are fitted
    max_neighborhood_points: int = 30000        # subsample cap for RANSAC scoring (speed)
    radius_min: float = 0.01                    # accepted pipe radius range
    radius_max: float = 0.30
    ransac_iterations: int = 400
    inlier_threshold: float = 0.003             # |distance to axis - radius| below this = inlier
    normal_angle_deg: float = 25.0              # inlier normals must be within this angle of the radial direction
    min_axis_angle_deg: float = 8.0             # skip sample pairs whose normals are (nearly) parallel (flat patch)
    refine: bool = True                         # least-squares refit on the RANSAC inliers

    # --- validation (both methods) ---
    min_inliers: int = 100
    min_inlier_ratio: float = 0.05              # inliers / neighborhood points
    min_angular_coverage_deg: float = 90.0      # angular extent of inliers around the axis (single-view scans: ~120-180)
    anchor_tolerance: float = 0.01              # selected surface point must lie this close to the fitted surface
    max_radius_mismatch: float = 0.30           # cluster method: |fit r - estimated r| / estimated r
    validate: bool = True

    # --- legacy cluster method ---
    cluster_roi_radius: float = 0.005           # radius of the thin ROI cylinder along the inverse normal (also the cluster gap)
    cluster_height_range: Tuple[float, float] = (-0.1, 0.3)   # ROI extent along the inverse normal from the surface point
    sphere_radius_offset: float = 0.003         # margin added to the estimated radius for the refit sphere

    @classmethod
    def from_config(cls, ef_pose_cfg: Optional[Dict[str, Any]]) -> "PipeProfileParams":
        """Build from ``ef_pose`` config. ``ef_pose.pipe_profile`` wins; the old flat keys are still honored."""
        cfg = dict(ef_pose_cfg or {})
        section = dict(cfg.get("pipe_profile") or {})
        legacy = {
            "sampling_size_for_calculating_normal": "normal_sampling_size",
            "radius_offset_for_sampling_points_in_sphere": "sphere_radius_offset",
        }
        for old, new in legacy.items():
            if old in cfg and new not in section:
                section[new] = cfg[old]
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in section.items() if k in known}
        if "cluster_height_range" in values:
            values["cluster_height_range"] = tuple(float(x) for x in values["cluster_height_range"])
        params = cls(**values)
        params.check()
        return params

    @classmethod
    def unknown_keys(cls, ef_pose_cfg: Optional[Dict[str, Any]]) -> List[str]:
        section = dict((ef_pose_cfg or {}).get("pipe_profile") or {})
        known = {f.name for f in fields(cls)}
        return sorted(k for k in section if k not in known)

    def check(self) -> None:
        if self.method not in ("ransac", "cluster"):
            raise ValueError(f"pipe_profile.method must be 'ransac' or 'cluster', got {self.method!r}")
        if not (0.0 < self.radius_min < self.radius_max):
            raise ValueError("pipe_profile: need 0 < radius_min < radius_max")
        if self.inlier_threshold <= 0 or self.neighborhood_radius <= 0:
            raise ValueError("pipe_profile: inlier_threshold and neighborhood_radius must be > 0")


# ---------------------------------------------------------------------------------------------------------------------
# RANSAC cylinder
# ---------------------------------------------------------------------------------------------------------------------
def _axis_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def _radial(points: np.ndarray, center: np.ndarray, axis: np.ndarray):
    rel = points - center
    along = rel @ axis
    perp = rel - np.outer(along, axis)
    dist = np.linalg.norm(perp, axis=1)
    return along, perp, dist


def _inlier_mask(points, normals, center, axis, radius, tol, cos_tol):
    along, perp, dist = _radial(points, center, axis)
    mask = np.abs(dist - radius) < tol
    if normals is not None and cos_tol > 0:
        radial_dir = perp / np.maximum(dist, 1e-12)[:, None]
        mask &= np.abs(np.einsum("ij,ij->i", normals, radial_dir)) >= cos_tol
    return mask


def ransac_cylinder(points: np.ndarray, normals: Optional[np.ndarray], anchor: np.ndarray,
                    params: PipeProfileParams, rng: np.random.Generator) -> Optional[Dict[str, Any]]:
    """Best cylinder hypothesis passing through ``anchor``, or None when no sample pair produced a valid one."""
    n = len(points)
    if n < 10 or normals is None:
        return None
    tol = params.inlier_threshold
    cos_tol = math.cos(math.radians(params.normal_angle_deg))
    min_sin = math.sin(math.radians(params.min_axis_angle_deg))
    best: Optional[Dict[str, Any]] = None
    for _ in range(max(1, int(params.ransac_iterations))):
        i, j = rng.choice(n, 2, replace=False)
        n1, n2 = normals[i], normals[j]
        axis = np.cross(n1, n2)
        sin_angle = float(np.linalg.norm(axis))
        if sin_angle < min_sin:
            continue
        axis /= sin_angle
        # in the plane perpendicular to the axis the two surface normals meet at the cylinder center
        d = points[j] - points[i]
        d -= axis * (d @ axis)
        try:
            t, *_ = np.linalg.lstsq(np.stack([n1, -n2], axis=1), d, rcond=None)
        except np.linalg.LinAlgError:
            continue
        r1, r2 = abs(float(t[0])), abs(float(t[1]))
        radius = 0.5 * (r1 + r2)
        if not (params.radius_min <= radius <= params.radius_max) or abs(r1 - r2) > max(3.0 * tol, 0.15 * radius):
            continue
        center = points[i] + float(t[0]) * n1
        _, _, anchor_dist = _radial(anchor[None, :], center, axis)
        if abs(float(anchor_dist[0]) - radius) > params.anchor_tolerance:
            continue
        mask = _inlier_mask(points, normals, center, axis, radius, tol, cos_tol)
        count = int(mask.sum())
        if best is None or count > best["count"]:
            best = {"count": count, "center": center, "axis": axis, "radius": radius, "mask": mask}
    return best


def refine_cylinder(points, normals, model, params: PipeProfileParams, max_points: int = 4000) -> Dict[str, Any]:
    """Least-squares refit on the inliers; keeps the refit only if it does not lose inliers."""
    try:
        from .CylinderFitting import fit_cylinder
    except ImportError:  # poseDeterminator dir itself on sys.path (see EndEffectorPoseOptimizer)
        from CylinderFitting import fit_cylinder

    tol = params.inlier_threshold
    cos_tol = math.cos(math.radians(params.normal_angle_deg))
    inl = points[model["mask"]]
    if len(inl) < 20:
        return model
    if len(inl) > max_points:
        inl = inl[np.linspace(0, len(inl) - 1, max_points).astype(int)]
    axis = model["axis"]
    theta, phi = math.acos(float(np.clip(axis[2], -1, 1))), math.atan2(float(axis[1]), float(axis[0]))
    try:
        w, c, r, _err = fit_cylinder(inl, guess_angles=[(theta, phi)])
    except Exception:
        return model
    w = np.asarray(w, float)
    if not (params.radius_min <= float(r) <= params.radius_max):
        return model
    mask = _inlier_mask(points, normals, np.asarray(c, float), w, float(r), tol, cos_tol)
    if int(mask.sum()) < model["count"]:
        return model
    return {"count": int(mask.sum()), "center": np.asarray(c, float), "axis": w, "radius": float(r), "mask": mask}


def fit_metrics(points, model, anchor) -> Dict[str, float]:
    along, perp, dist = _radial(points[model["mask"]], model["center"], model["axis"])
    residual = dist - model["radius"]
    u, v = _axis_basis(model["axis"])
    angles = np.degrees(np.arctan2(perp @ v, perp @ u))
    bins = np.unique(np.floor((angles + 180.0) / 10.0).astype(int) % 36)
    _, _, anchor_dist = _radial(np.asarray(anchor, float)[None, :], model["center"], model["axis"])
    return {
        "n_inliers": int(model["mask"].sum()),
        "rms": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else float("nan"),
        "angular_coverage_deg": float(len(bins) * 10.0),
        "axial_extent": float(np.ptp(along)) if len(along) else 0.0,
        "anchor_residual": float(abs(anchor_dist[0] - model["radius"])),
    }


def validate_fit(radius: float, metrics: Dict[str, float], n_neighborhood: int,
                 params: PipeProfileParams, estimated_radius: Optional[float] = None) -> List[str]:
    """Reasons the fit is not trustworthy (empty list = valid)."""
    reasons: List[str] = []
    if not (params.radius_min <= radius <= params.radius_max):
        reasons.append(f"radius {radius * 1000:.1f}mm outside [{params.radius_min * 1000:.0f}, {params.radius_max * 1000:.0f}]mm")
    if metrics.get("n_inliers", 0) < params.min_inliers:
        reasons.append(f"inliers {metrics.get('n_inliers', 0)} < min_inliers {params.min_inliers}")
    ratio = metrics.get("n_inliers", 0) / max(1, n_neighborhood)
    if ratio < params.min_inlier_ratio:
        reasons.append(f"inlier ratio {ratio:.3f} < {params.min_inlier_ratio}")
    if metrics.get("angular_coverage_deg", 0.0) < params.min_angular_coverage_deg:
        reasons.append(
            f"angular coverage {metrics.get('angular_coverage_deg', 0.0):.0f}deg < {params.min_angular_coverage_deg:.0f}deg")
    if metrics.get("anchor_residual", 0.0) > params.anchor_tolerance:
        reasons.append(f"surface point is {metrics['anchor_residual'] * 1000:.1f}mm off the fitted surface (> {params.anchor_tolerance * 1000:.0f}mm)")
    if estimated_radius is not None and estimated_radius > 0 \
            and abs(radius - estimated_radius) / estimated_radius > params.max_radius_mismatch:
        reasons.append(
            f"fit radius {radius * 1000:.1f}mm disagrees with cluster estimate {estimated_radius * 1000:.1f}mm")
    return reasons
