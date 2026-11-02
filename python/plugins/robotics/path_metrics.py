"""Path-planning evaluation metrics (see path_plan_metric.md).

PathMetricEvaluator.evaluate() is called once per planned target - right after plan_single_target, while that
target's collision scene (pipe, positioner, optional positioner-rotation of the pipe) is still configured on the
backend - and returns a flat dict of

- C_nom / C_eff        minimum clearance over the continuous (densified) path, without / with the safety margin m.
                       The inspection sensor (end link) vs the pipe is excluded from the "safe" set while the sensor is
                       inside the intended approach/retreat zone; every other link-obstacle pair stays in it.
- mandatory check      all robot-vs-static pairs (sensor-pipe included) must stay above d_physical_min on the whole path.
- E_d, E_theta         inspection goal errors at q(1): |d_tool_target - d*| and the angle between the sensor axis and
                       -n_target (surface normal where the sensor ray hits the pipe). Plus pose error vs the target pose
                       and the final-approach direction error.
- P_collision / P_violation (Monte Carlo)  tracking error (joint offsets/noise) and registration error (rigid pipe
                       offset) per trial; collision = any mandatory pair <= 0, violation = C_eff<0 (no margin, to avoid
                       counting the uncertainty twice) or an inspection constraint broken.
- L_q, DTW-RMSE        joint-space length (raw and range-normalised) and EE DTW-RMSE between nominal and executed path.
- safe                 planned & C_eff >= 0 & mandatory ok & E_d/E_theta within eps & P_collision <= max rate.

All distances in metres, angles in degrees.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None

try:
    import pinocchio as pin
except Exception:  # pragma: no cover
    pin = None

PIPE_OBSTACLE = "collision_object_0"   # configure_collision order: [pipe, positioner, (virtual side wall)]
VIRTUAL_OBSTACLES = ("collision_object_2",)   # robot-side wall: counts for collisions, not for the clearance metric


@dataclass
class MetricConfig:
    margin: float = 0.005                 # m: tracking + registration error converted to distance (m_{l,o})
    physical_min: float = 0.001           # m: d_physical_min, any robot-static pair below this is a violation
    approach_zone: float = 0.10           # m: sensor-pipe distance below which the (leading/trailing) run is "intended"
    d_star: Dict[str, Optional[float]] = field(default_factory=lambda: {"dda": 0.01, "rt": None})
    eps_d: float = 0.01                   # m
    eps_theta_deg: float = 5.0
    ccd_step: float = 0.02                # q step of the continuous check (rail m / joint rad)
    ccd_max_samples: int = 250            # cap on checked samples per path (one distance query ~0.2 s with the positioner mesh)
    # exact PyBullet distances only up to this radius (near contact, where the metrics need them exact); farther pairs
    # take the scene-SDF value. An exact query with the default 0.3 m radius took 1-4 s on a scanned ~100k-triangle
    # pipe, i.e. minutes per path. 0 = always exact with the backend default radius.
    exact_distance_cap: float = 0.06
    mc_trials: int = 20
    mc_step: float = 0.05
    mc_max_samples: int = 150
    sigma_track_lin: float = 0.002        # m   rail/carriage tracking error
    sigma_track_rot: float = 0.002        # rad arm joint tracking error
    sigma_reg_t: float = 0.003            # m   pipe registration translation error (1 sigma per axis)
    sigma_reg_r: float = 0.003            # rad pipe registration rotation error (1 sigma per axis)
    max_collision_rate: float = 0.05
    approach_span: float = 0.10           # m of EE travel used for the final approach direction
    # motion-time profile: trapezoidal velocity profile over the path, governed by the largest joint displacement of
    # every segment (revolute joints in deg; prismatic joints - linear track, carriage - in m with their own limits)
    joint_vel_deg: float = 50.0           # deg/s
    joint_acc_deg: float = 30.0           # deg/s^2
    lin_vel: float = 0.5                  # m/s   (linear track / carriage)
    lin_acc: float = 0.3                  # m/s^2
    sensor_kinds: tuple = ("dda",)        # robots whose end link is the inspection sensor (E_d/E_theta/approach apply);
                                          # the others (RT source) only report their aim error to the target point
    seed: int = 0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "MetricConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# ---------------------------------------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------------------------------------
def densify(q_path, max_step):
    arr = [np.asarray(q, dtype=float) for q in q_path]
    out = [arr[0]]
    for a, b in zip(arr[:-1], arr[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / max_step)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return out


def _subsample(items, max_n):
    if len(items) <= max_n:
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, max_n).astype(int))
    return [items[i] for i in idx]


def _angle_deg(u, v):
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-15))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def dtw_rmse(a: np.ndarray, b: np.ndarray, max_points: int = 200) -> float:
    """sqrt(mean ||a_i - b_j||^2) along the optimal DTW alignment (EE paths, Nx3)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) > max_points:
        a = a[np.linspace(0, len(a) - 1, max_points).astype(int)]
    if len(b) > max_points:
        b = b[np.linspace(0, len(b) - 1, max_points).astype(int)]
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2) ** 2
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, steps = n, m, 0
    total = 0.0
    while i > 0 and j > 0:
        total += cost[i - 1, j - 1]
        steps += 1
        k = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return math.sqrt(total / max(steps, 1))


class SurfaceProbe:
    """Ray / closest-point queries on the (possibly positioner-rotated) pipe mesh in world frame."""

    def __init__(self, mesh):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.centroid = np.asarray(mesh.get_center(), dtype=float)

    def normal_along(self, origin, direction):
        """(outward unit normal, hit point, ray_hit) where the ray from origin along direction meets the surface;
        falls back to the closest surface point when the ray misses."""
        origin, direction = np.asarray(origin, float), np.asarray(direction, float)
        rays = o3d.core.Tensor([[*origin, *(direction / np.linalg.norm(direction))]], dtype=o3d.core.Dtype.Float32)
        hit = self.scene.cast_rays(rays)
        t = float(hit["t_hit"].numpy()[0])
        if np.isfinite(t):
            n = hit["primitive_normals"].numpy()[0].astype(float)
            p = origin + direction / np.linalg.norm(direction) * t
            ray_hit = True
        else:
            cp = self.scene.compute_closest_points(o3d.core.Tensor([origin], dtype=o3d.core.Dtype.Float32))
            p = cp["points"].numpy()[0].astype(float)
            n = cp["primitive_normals"].numpy()[0].astype(float)
            ray_hit = False
        n = n / (np.linalg.norm(n) + 1e-15)
        if np.dot(n, origin - p) < 0:
            n = -n
        return n, p, ray_hit


@contextlib.contextmanager
def perturbed_pipe(backend, robot_name, T_base_frame):
    """Temporarily move the pipe collision object (static 0) by T (4x4, in the robot base frame)."""
    scene = backend.pybullet_scene(robot_name) if hasattr(backend, "pybullet_scene") else None
    if scene is not None and scene.static_bodies:
        from scipy.spatial.transform import Rotation
        import pybullet as pb
        body = scene.static_bodies[0]
        T = np.asarray(T_base_frame, dtype=float)
        from plugins.robotics.pybullet_collision import move_static_body
        move_static_body(body, T[:3, 3].tolist(), Rotation.from_matrix(T[:3, :3]).as_quat().tolist(), scene.client)
        try:
            yield
        finally:
            move_static_body(body, [0, 0, 0], [0, 0, 0, 1], scene.client)
        return
    handle = backend._handle(robot_name)
    ids = list(handle.static_object_ids)
    if not ids or pin is None:
        yield
        return
    obj = handle.geom_model.geometryObjects[ids[0]]
    original = pin.SE3(obj.placement)
    obj.placement = pin.SE3(np.asarray(T_base_frame[:3, :3], float), np.asarray(T_base_frame[:3, 3], float)) * original
    try:
        yield
    finally:
        obj.placement = original


# ---------------------------------------------------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------------------------------------------------
class PathMetricEvaluator:
    def __init__(self, config: Dict[str, Any], metric_cfg: MetricConfig, rotation_T=None):
        self.config = config
        self.cfg = metric_cfg
        self.rotation_T = None if rotation_T is None else np.asarray(rotation_T, dtype=float)
        self._probes: Dict[bool, SurfaceProbe] = {}

    # -- plumbing ------------------------------------------------------------------------------------------------
    @staticmethod
    def _kind(robot_name) -> str:
        return "dda" if "dda" in str(robot_name).lower() else "rt"

    def _frame_cfg(self, kind):
        return ((self.config.get("ef_pose") or {}).get("frames") or {}).get(kind) or {}

    def _probe(self, engine, rotate) -> SurfaceProbe:
        key = bool(rotate and self.rotation_T is not None)
        if key not in self._probes:
            import copy
            mesh = copy.deepcopy(engine._current_spool_collision_mesh())
            if key:
                mesh.transform(self.rotation_T)
            self._probes[key] = SurfaceProbe(mesh)
        return self._probes[key]

    @staticmethod
    def _is_sensor(link, sensor):
        return str(link) == sensor or str(link).startswith(sensor + "_")

    def _profile(self, backend, robot_name, qs, sensor):
        """Per sample: (min safe-candidate pairs, sensor-pipe distance, all-pairs min) bookkeeping."""
        rows = []
        cap = float(getattr(self.cfg, "exact_distance_cap", 0.0) or 0.0)
        # Evaluation must not create an expensive SDF solely for far distances.
        use_sdf = (cap > 0.0 and hasattr(backend, "link_obstacle_distances_sdf")
                   and getattr(backend, "has_cached_scene_sdf", lambda _: False)(robot_name))
        for q in qs:
            if use_sdf:
                entries = backend.link_obstacle_distances(robot_name, q, max_distance=cap)
                far = {(e["link"], e["obstacle"]): e["distance"] for e in backend.link_obstacle_distances_sdf(robot_name, q)}
                for e in entries:
                    if e["distance"] >= cap - 1e-9:   # nothing within the exact radius: SDF value, at least the radius
                        e["distance"] = max(cap, float(far.get((e["link"], e["obstacle"]), cap)))
            else:
                entries = backend.link_obstacle_distances(robot_name, q)
            d_sp = [e["distance"] for e in entries if self._is_sensor(e["link"], sensor) and e["obstacle"] == PIPE_OBSTACLE]
            rows.append({"entries": entries, "d_sp": min(d_sp) if d_sp else None})
        return rows

    def _zone_mask(self, rows):
        """True where the sensor is in the intended approach (trailing run) / retreat (leading run) zone."""
        d = [r["d_sp"] for r in rows]
        n = len(d)
        zone = np.zeros(n, dtype=bool)
        if any(x is None for x in d):
            return zone
        end = n
        while end > 0 and d[end - 1] < self.cfg.approach_zone:
            end -= 1
        begin = 0
        while begin < end and d[begin] < self.cfg.approach_zone:
            begin += 1
        zone[:begin] = True
        zone[end:] = True
        return zone

    def _clearances(self, rows, zone, sensor):
        """(C_nom, C_eff-without-margin-offset i.e. same value, link, obstacle, mandatory_min, mandatory_pair)."""
        c_nom, who = math.inf, (None, None)
        mand_min, mand_who = math.inf, (None, None)
        for r, in_zone in zip(rows, zone):
            for e in r["entries"]:
                d = e["distance"]
                if d < mand_min:
                    mand_min, mand_who = d, (e["link"], e["obstacle"])
                intended = in_zone and self._is_sensor(e["link"], sensor) and e["obstacle"] == PIPE_OBSTACLE
                if not intended and e["obstacle"] not in VIRTUAL_OBSTACLES and d < c_nom:
                    c_nom, who = d, (e["link"], e["obstacle"])
        return c_nom, who, mand_min, mand_who

    def _goal_errors(self, engine, backend, robot_name, kind, q_end, target_pose, rotate, d_tool_target, frame,
                     target_point=None):
        cfg = self.cfg
        T = np.asarray(backend.frame_world_T(robot_name, q_end, frame), dtype=float)
        facing_local = np.asarray(self._frame_cfg(kind).get("pipe_facing_axis", [0.0, 0.0, 1.0]), dtype=float)
        facing_local = facing_local / np.linalg.norm(facing_local)
        axis = T[:3, :3] @ facing_local
        n, hit, ray_hit = self._probe(engine, rotate).normal_along(T[:3, 3], axis)
        out = {
            "E_theta_deg": _angle_deg(axis, -n) if kind in cfg.sensor_kinds else None,
            "sensor_ray_hit": bool(ray_hit),
            "tool_target_distance": d_tool_target,
        }
        if target_point is not None:
            out["aim_error_deg"] = _angle_deg(axis, np.asarray(target_point, float) - T[:3, 3])
        d_star = (cfg.d_star or {}).get(kind)
        out["E_d"] = abs(d_tool_target - d_star) if (d_star is not None and d_tool_target is not None
                                                     and kind in cfg.sensor_kinds) else None
        if target_pose is not None:
            Tt = np.asarray(target_pose, dtype=float)
            out["E_pos"] = float(np.linalg.norm(T[:3, 3] - Tt[:3, 3]))
            out["E_rot_deg"] = _rot_angle_deg(T[:3, :3].T @ Tt[:3, :3])
        return out, n

    def _ee_positions(self, backend, robot_name, qs, frame):
        return np.array([backend.frame_world_T(robot_name, q, frame)[:3, 3] for q in qs])

    # -- Monte Carlo -----------------------------------------------------------------------------------------------
    def _monte_carlo(self, engine, backend, robot_name, kind, q_path, rotate, sensor, frame, normal, pivot_base, nominal_ee):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        handle = backend._handle(robot_name)
        model = handle.model
        types = np.array([cfg.sigma_track_lin if model.joints[i + 1].shortname().startswith("JointModelP")
                          else cfg.sigma_track_rot for i in range(model.njoints - 1)], dtype=float)
        # per-joint sigma in q order (all joints here are 1-DoF)
        sigma = np.zeros(model.nq)
        for i in range(1, model.njoints):
            sigma[model.joints[i].idx_q] = types[i - 1]
        qs_nom = _subsample(densify(q_path, cfg.mc_step), cfg.mc_max_samples)
        d_star = (cfg.d_star or {}).get(kind)
        collisions = violations = 0
        c_actual, dtws, e_ds, e_ths = [], [], [], []
        Tc = np.eye(4)
        Tc[:3, 3] = pivot_base
        for _ in range(cfg.mc_trials):
            bias = rng.normal(0.0, 1.0, model.nq) * sigma
            qs = [q + bias + rng.normal(0.0, 0.5, model.nq) * sigma for q in qs_nom]
            qs[0] = np.asarray(qs_nom[0], float)            # the run starts from the known start state
            rv = rng.normal(0.0, cfg.sigma_reg_r, 3)
            Tr = np.eye(4)
            Tr[:3, :3] = _rotvec_to_R(rv)
            Tr[:3, 3] = rng.normal(0.0, cfg.sigma_reg_t, 3)
            T_pert = Tc @ Tr @ np.linalg.inv(Tc)
            with perturbed_pipe(backend, robot_name, T_pert):
                rows = self._profile(backend, robot_name, qs, sensor)
            zone = self._zone_mask(rows)
            c_nom, _, mand_min, _ = self._clearances(rows, zone, sensor)
            collided = mand_min <= 0.0
            d_final = rows[-1]["d_sp"]
            e_d = abs(d_final - d_star) if (d_star is not None and d_final is not None) else None
            T_end = np.asarray(backend.frame_world_T(robot_name, qs[-1], frame), dtype=float)
            fl = np.asarray(self._frame_cfg(kind).get("pipe_facing_axis", [0.0, 0.0, 1.0]), dtype=float)
            e_th = _angle_deg(T_end[:3, :3] @ (fl / np.linalg.norm(fl)), -(Tr[:3, :3] @ normal))
            is_sensor_robot = kind in cfg.sensor_kinds
            bad = (collided or c_nom < 0.0 or (e_d is not None and e_d > cfg.eps_d)
                   or (is_sensor_robot and e_th > cfg.eps_theta_deg))
            collisions += int(collided)
            violations += int(bad)
            c_actual.append(c_nom)
            e_ths.append(e_th)
            if e_d is not None:
                e_ds.append(e_d)
            ee = self._ee_positions(backend, robot_name, qs, frame)
            dtws.append(dtw_rmse(nominal_ee, ee))
        n = max(cfg.mc_trials, 1)
        return {
            "P_collision": collisions / n,
            "P_violation": violations / n,
            "mc_trials": cfg.mc_trials,
            "C_executed_min": float(np.min(c_actual)) if c_actual else None,
            "C_executed_mean": float(np.mean(c_actual)) if c_actual else None,
            "DTW_RMSE": float(np.mean(dtws)) if dtws else None,
            "mc_E_theta_deg_mean": float(np.mean(e_ths)) if e_ths else None,
            "mc_E_d_mean": float(np.mean(e_ds)) if e_ds else None,
        }

    # -- main entry --------------------------------------------------------------------------------------------------
    def evaluate(self, *, engine, robot_name, pose_name, group, target_pose, q_path, result, rotate, plan_wall, **_):
        cfg = self.cfg
        kind = self._kind(robot_name)
        out: Dict[str, Any] = {"t_plan": plan_wall, "planned": False, "safe": False}
        status = result.get("status")
        if status != "success" or not q_path or len(q_path) < 1:
            out["fail_reason"] = str(result.get("message") or status)
            return out
        backend = engine._robotics_backend
        sensor = str(self._frame_cfg(kind).get("end_link", f"{kind}_link_end"))
        frame = engine._robot_target_link_name(robot_name)

        interp = densify(q_path, cfg.ccd_step)
        dense = _subsample(interp, cfg.ccd_max_samples)
        rows = self._profile(backend, robot_name, dense, sensor)
        zone = self._zone_mask(rows)
        c_nom, (link, obstacle), mand_min, (mlink, mobst) = self._clearances(rows, zone, sensor)
        # robot-static pairs are covered by the distances above; check_collision additionally catches self pairs
        stride = max(1, len(dense) // 200)
        out_self = sum(1 for q in dense[::stride] if backend.check_collision(robot_name, q).collision)
        out.update({
            "planned": True,
            "C_nom": None if math.isinf(c_nom) else c_nom,
            "C_eff": None if math.isinf(c_nom) else c_nom - cfg.margin,
            "C_eff_link": link, "C_eff_obstacle": obstacle,
            "mandatory_min_distance": None if math.isinf(mand_min) else mand_min,
            "mandatory_ok": bool(mand_min > cfg.physical_min and out_self == 0),
            "collision_samples": int(out_self),
            "mandatory_link": mlink, "mandatory_obstacle": mobst,
            "n_dense_samples": len(dense),
            "n_waypoints": len(q_path),
            "n_interp_samples": len(interp),   # > n_dense_samples: capped at ccd_max_samples (wider sample spacing)
            "approach_zone_samples": int(zone.sum()),
        })

        # does the arm stay on the robot's side of the pipe? (reaching over the pipe from the other side)
        try:
            from plugins.robotics.collision_free_ik import robot_side_excess
            ex = [robot_side_excess(backend, robot_name, q) for q in _subsample(dense, 80)]
            ex = [e for e in ex if e is not None]
            if ex:
                out["side_excess_max"] = float(max(ex))
                out["side_cross_fraction"] = float(np.mean([e > 0.0 for e in ex]))
        except Exception:
            pass

        # trajectory length + motion time (trapezoidal profile on the governing joint displacement)
        out.update(motion_time(backend, robot_name, q_path, frame, cfg))

        # joint-space length
        arr = np.asarray(q_path, dtype=float)
        out["L_q"] = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))) if len(arr) > 1 else 0.0
        _, _, span = backend.joint_limits_for_metric(robot_name, True)
        if span is not None and len(arr) > 1:
            d = np.diff(arr, axis=0) / span
            out["L_q_normalized"] = float(np.sum(np.linalg.norm(d, axis=1)))

        # inspection goal errors + final approach direction
        d_tool_target = rows[-1]["d_sp"]
        errs, normal = self._goal_errors(engine, backend, robot_name, kind, dense[-1], target_pose, rotate,
                                         d_tool_target, frame, (group or {}).get("target_point"))
        out.update(errs)
        ee_dense = self._ee_positions(backend, robot_name, dense, frame)
        behind = np.linalg.norm(ee_dense - ee_dense[-1], axis=1) >= cfg.approach_span
        if behind.any() and kind in cfg.sensor_kinds:
            k = int(np.nonzero(behind)[0][-1])
            out["approach_dir_error_deg"] = _angle_deg(ee_dense[-1] - ee_dense[k], -normal)

        # robustness
        handle = backend._handle(robot_name)
        base_T = np.asarray(handle.description.base_T, dtype=float)
        pivot_world = self._probe(engine, rotate).centroid
        pivot_base = (np.linalg.inv(base_T) @ np.append(pivot_world, 1.0))[:3]
        if cfg.mc_trials > 0:
            nominal_ee = _subsample(list(ee_dense), 200)
            out.update(self._monte_carlo(engine, backend, robot_name, kind, q_path, rotate, sensor, frame, normal,
                                         pivot_base, np.asarray(nominal_ee)))

        # safe-plan decision
        ok = out["mandatory_ok"] and out["C_eff"] is not None and out["C_eff"] >= 0.0
        if out.get("E_d") is not None:
            ok = ok and out["E_d"] <= cfg.eps_d
        if out.get("E_theta_deg") is not None:
            ok = ok and out["E_theta_deg"] <= cfg.eps_theta_deg
        if cfg.mc_trials > 0:
            ok = ok and out["P_collision"] <= cfg.max_collision_rate
        out["safe"] = bool(ok)
        if not ok:
            reasons = []
            if not out["mandatory_ok"]:
                reasons.append(f"penetration/min-gap {out['mandatory_min_distance']:.4f} ({mlink}<->{mobst}), "
                               f"{out['collision_samples']} colliding samples")
            if out["C_eff"] is not None and out["C_eff"] < 0:
                reasons.append(f"C_eff {out['C_eff']:.4f} < 0 ({link}<->{obstacle})")
            if out.get("E_d") is not None and out["E_d"] > cfg.eps_d:
                reasons.append(f"E_d {out['E_d']:.4f} > {cfg.eps_d}")
            if out.get("E_theta_deg") is not None and out["E_theta_deg"] > cfg.eps_theta_deg:
                reasons.append(f"E_theta {out['E_theta_deg']:.1f}deg > {cfg.eps_theta_deg}")
            if cfg.mc_trials > 0 and out["P_collision"] > cfg.max_collision_rate:
                reasons.append(f"P_collision {out['P_collision']:.2f} > {cfg.max_collision_rate}")
            out["fail_reason"] = "; ".join(reasons)
        return out


def trapezoid_time(distance: float, v_max: float, a_max: float) -> float:
    """Rest-to-rest time over `distance` with velocity limit v_max and acceleration limit a_max (triangular profile
    when the distance is too short to reach v_max)."""
    d = abs(float(distance))
    if d <= 0.0:
        return 0.0
    if d >= v_max * v_max / a_max:
        return d / v_max + v_max / a_max
    return 2.0 * math.sqrt(d / a_max)


class PathTiming:
    """Time parameterization of a joint path under the velocity profile of the metric (and the playback).

    Every segment gets a cruise duration = the largest of (revolute |dq| / joint_vel, prismatic |dq| / lin_vel): the
    slowest joint of the segment governs, all joints move synchronised. The path then runs one trapezoidal profile over
    the summed cruise duration S (normalised speed 1 = every segment at its governing joint's speed limit) with the
    normalised acceleration min(joint_acc / joint_vel, lin_acc / lin_vel). No stop at intermediate waypoints."""

    def __init__(self, q_path, prismatic_mask, joint_vel_deg=50.0, joint_acc_deg=30.0, lin_vel=0.5, lin_acc=0.3):
        arr = np.asarray(q_path, dtype=float)
        mask = np.asarray(prismatic_mask, dtype=bool)
        self.q = arr
        v_rev, a_rev = math.radians(joint_vel_deg), math.radians(joint_acc_deg)
        if len(arr) < 2:
            self.seg = np.zeros(0)
        else:
            d = np.abs(np.diff(arr, axis=0))
            t_rev = np.max(np.where(mask, 0.0, d), axis=1) / v_rev
            t_lin = np.max(np.where(mask, d, 0.0), axis=1) / lin_vel if mask.any() else np.zeros(len(d))
            self.seg = np.maximum(t_rev, t_lin)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg)])
        self.S = float(self.cum[-1])
        self.a = min(a_rev / v_rev, lin_acc / lin_vel if mask.any() else a_rev / v_rev)   # 1/s (normalised)
        self.T = trapezoid_time(self.S, 1.0, self.a)
        self._t_acc = 1.0 / self.a if self.S >= 1.0 / self.a else math.sqrt(self.S / self.a)
        self._v_peak = self.a * self._t_acc

    def progress(self, t):
        """Normalised arc length s(t) in [0, S]."""
        t = min(max(float(t), 0.0), self.T)
        ta, vp, a = self._t_acc, self._v_peak, self.a
        if t <= ta:
            return 0.5 * a * t * t
        t_dec = self.T - ta
        if t <= t_dec:
            return 0.5 * a * ta * ta + vp * (t - ta)
        r = self.T - t
        return self.S - 0.5 * a * r * r

    def q_at(self, t):
        """(q, segment index) at time t."""
        if len(self.q) < 2 or self.S <= 0.0:
            return self.q[-1].copy(), max(len(self.q) - 2, 0)
        s = self.progress(t)
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg) - 1))
        r = 0.0 if self.seg[i] <= 1e-12 else (s - self.cum[i]) / self.seg[i]
        return self.q[i] + (self.q[i + 1] - self.q[i]) * min(max(r, 0.0), 1.0), i


def prismatic_mask_of(backend, robot_name, n):
    model = backend._handle(robot_name).model
    mask = np.zeros(n, dtype=bool)
    for i in range(1, model.njoints):
        idx = model.joints[i].idx_q
        if idx < n:
            mask[idx] = model.joints[i].shortname().startswith("JointModelP")
    return mask


def motion_time(backend, robot_name, q_path, frame, cfg: "MetricConfig") -> Dict[str, Any]:
    """Trajectory length and motion time of a q path.

    L_ee: TCP (end-effector) path length in world. S_rev: sum over segments of the largest revolute-joint displacement
    (deg); S_lin: same for the prismatic joints (m); T_rev / T_lin: each alone on a trapezoidal profile.
    T_motion: PathTiming (segment-wise governing joint, synchronised, one trapezoid) - the same timing the playback
    uses, so T_motion is the playback duration at speed 1."""
    arr = np.asarray(q_path, dtype=float)
    out: Dict[str, Any] = {"L_ee": 0.0, "S_rev_deg": 0.0, "S_lin": 0.0, "T_rev": 0.0, "T_lin": 0.0, "T_motion": 0.0}
    if len(arr) < 2:
        return out
    prismatic = prismatic_mask_of(backend, robot_name, arr.shape[1])
    d = np.abs(np.diff(arr, axis=0))
    s_rev = float(np.sum(np.max(np.where(prismatic, 0.0, d), axis=1))) * 180.0 / math.pi
    s_lin = float(np.sum(np.max(np.where(prismatic, d, 0.0), axis=1))) if prismatic.any() else 0.0
    ee = np.array([np.asarray(backend.frame_world_T(robot_name, q, frame))[:3, 3] for q in densify(arr, 0.02)])
    t_rev = trapezoid_time(s_rev, cfg.joint_vel_deg, cfg.joint_acc_deg)
    t_lin = trapezoid_time(s_lin, cfg.lin_vel, cfg.lin_acc)
    out.update({"L_ee": float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1))) if len(ee) > 1 else 0.0,
                "S_rev_deg": s_rev, "S_lin": s_lin, "T_rev": t_rev, "T_lin": t_lin,
                "T_motion": PathTiming(arr, prismatic, cfg.joint_vel_deg, cfg.joint_acc_deg, cfg.lin_vel, cfg.lin_acc).T})
    return out


def planning_path_summary(engine, robot_name, q_path, *, max_clearance_samples=80):
    """Metrics saved with a plan, using the configured collision scene and playback speed limits.

    Clearance follows the final approach/retreat sensor exclusion used by PathMetricEvaluator.
    This bounded summary omits Monte Carlo and target-pose errors; the full evaluator remains separate.
    """
    backend = engine._robotics_backend
    cfg_data = ((engine._config.get("path_planning") or {}).get("playback_velocity_limits") or {})
    cfg = MetricConfig(
        joint_vel_deg=float(cfg_data.get("joint_vel_deg", 50.0)),
        joint_acc_deg=float(cfg_data.get("joint_acc_deg", 30.0)),
        lin_vel=float(cfg_data.get("lin_vel", 0.5)),
        lin_acc=float(cfg_data.get("lin_acc", 0.3)),
        mc_trials=0,
    )
    arr = np.asarray(q_path, dtype=float)
    if arr.ndim != 2 or not len(arr):
        raise ValueError("q_path must contain joint vectors")
    evaluator = PathMetricEvaluator(engine._config, cfg)
    kind = evaluator._kind(robot_name)
    sensor = str(evaluator._frame_cfg(kind).get("end_link", f"{kind}_link_end"))
    if hasattr(backend, "link_obstacle_distances_sdf") and not backend.has_cached_scene_sdf(robot_name):
        try:
            backend.link_obstacle_distances_sdf(robot_name, arr[0])
        except Exception:
            pass  # exact distance queries below remain available
    dense = _subsample(densify(arr, cfg.ccd_step), max_clearance_samples)
    rows = evaluator._profile(backend, robot_name, dense, sensor)
    zone = evaluator._zone_mask(rows)
    clearance, pair, mandatory, _ = evaluator._clearances(rows, zone, sensor)
    frame = engine._robot_target_link_name(robot_name)
    motion = motion_time(backend, robot_name, arr, frame, cfg)
    _, _, span = backend.joint_limits_for_metric(robot_name, True)
    span = np.asarray(span, dtype=float)
    span = np.where(np.isfinite(span) & (span > 1e-9), span, 1.0)
    normalized = arr / span
    segments = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(segments)]
    keep = np.r_[True, np.diff(arc) > 1e-12]
    smoothness = 0.0
    if keep.sum() >= 2:
        distance = float(arc[-1])
        samples = np.linspace(0.0, distance, 32)
        sampled = np.stack([np.interp(samples, arc[keep], normalized[keep, j])
                            for j in range(arr.shape[1])], axis=1)
        ds = distance / 31.0
        smoothness = float(np.sum(np.diff(sampled, n=2, axis=0) ** 2) / max(ds ** 3, 1e-12))
    duration = float(motion["T_motion"])
    return {
        "clearance_min_m": None if math.isinf(clearance) else float(clearance),
        "clearance_margin_m": None if math.isinf(clearance) else float(clearance - cfg.margin),
        "clearance_link": pair[0], "clearance_obstacle": pair[1],
        "physical_min_distance_m": None if math.isinf(mandatory) else float(mandatory),
        "clearance_samples": len(dense), "approach_zone_samples": int(zone.sum()),
        "smoothness": smoothness, "smoothness_definition": "normalized_q_arc_second_difference",
        "n_waypoints": len(arr), "joint_path_length_normalized": float(arc[-1]),
        "ee_path_length_m": float(motion["L_ee"]), "estimated_motion_s": duration,
        "estimated_ee_mean_speed_m_s": float(motion["L_ee"]) / duration if duration > 0 else 0.0,
        "playback_speed_multiplier": 1.0,
        "joint_velocity_limit_deg_s": cfg.joint_vel_deg,
        "joint_acceleration_limit_deg_s2": cfg.joint_acc_deg,
        "linear_velocity_limit_m_s": cfg.lin_vel,
        "linear_acceleration_limit_m_s2": cfg.lin_acc,
    }


def _rotvec_to_R(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


# ---------------------------------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------------------------------
def wilson_interval(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _stat(values, fn, default=None):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return fn(v) if v else default


def aggregate(rows: Sequence[Dict[str, Any]], timeout: float) -> Dict[str, Any]:
    """Planner-level summary of per-target rows (every request counted, failures included)."""
    n = len(rows)
    safe = [r for r in rows if r.get("safe")]
    planned = [r for r in rows if r.get("planned")]
    t = [r["t_plan"] for r in rows if r.get("t_plan") is not None]
    lo, hi = wilson_interval(len(safe), n)
    timeouts = sum(1 for r in rows if "timeout" in str(r.get("message") or "").lower()
                   or (r.get("t_plan") is not None and timeout and r["t_plan"] >= timeout))
    ceff_s = [r["C_eff"] for r in safe]
    ceff_p = [r["C_eff"] for r in planned]
    return {
        "n_requests": n, "n_planned": len(planned), "n_safe": len(safe),
        "SR_safe_pct": 100.0 * len(safe) / n if n else None,
        "SR_safe_ci95_pct": (100 * lo, 100 * hi),
        "planned_rate_pct": 100.0 * len(planned) / n if n else None,
        "C_eff_mean_safe_mm": _stat(ceff_s, lambda v: 1000 * float(np.mean(v))),
        "C_eff_mean_planned_mm": _stat(ceff_p, lambda v: 1000 * float(np.mean(v))),
        "C_eff_min_planned_mm": _stat(ceff_p, lambda v: 1000 * float(np.min(v))),
        "C_nom_min_planned_mm": _stat([r["C_nom"] for r in planned], lambda v: 1000 * float(np.min(v))),
        "P_collision_pct": _stat([r.get("P_collision") for r in planned], lambda v: 100 * float(np.mean(v))),
        "P_violation_pct": _stat([r.get("P_violation") for r in planned], lambda v: 100 * float(np.mean(v))),
        "E_d_mean_mm": _stat([r.get("E_d") for r in planned], lambda v: 1000 * float(np.mean(v))),
        "E_theta_mean_deg": _stat([r.get("E_theta_deg") for r in planned], lambda v: float(np.mean(v))),
        "approach_dir_error_mean_deg": _stat([r.get("approach_dir_error_deg") for r in planned], lambda v: float(np.mean(v))),
        "L_q_normalized_mean": _stat([r.get("L_q_normalized") for r in planned], lambda v: float(np.mean(v))),
        "DTW_RMSE_mean_mm": _stat([r.get("DTW_RMSE") for r in planned], lambda v: 1000 * float(np.mean(v))),
        "T_plan_mean_s": _stat(t, lambda v: float(np.mean(v))),
        "T_plan_p50_s": _stat(t, lambda v: float(np.percentile(v, 50))),
        "T_plan_p95_s": _stat(t, lambda v: float(np.percentile(v, 95))),
        "waypoints_mean": _stat([r.get("n_waypoints") for r in planned], lambda v: float(np.mean(v))),
        "waypoints_total": _stat([r.get("n_waypoints") for r in planned], lambda v: float(np.sum(v))),
        "eval_samples_mean": _stat([r.get("n_dense_samples") for r in planned], lambda v: float(np.mean(v))),
        "n_paths_sample_capped": sum(1 for r in planned if (r.get("n_interp_samples") or 0) > (r.get("n_dense_samples") or 0)),
        "T_motion_mean_s": _stat([r.get("T_motion") for r in planned], lambda v: float(np.mean(v))),
        "T_motion_total_s": _stat([r.get("T_motion") for r in planned], lambda v: float(np.sum(v))),
        "L_ee_mean_m": _stat([r.get("L_ee") for r in planned], lambda v: float(np.mean(v))),
        "L_ee_total_m": _stat([r.get("L_ee") for r in planned], lambda v: float(np.sum(v))),
        "timeout_rate_pct": 100.0 * timeouts / n if n else None,
    }


# ---------------------------------------------------------------------------------------------------------------------
# collision-free start posture for headless evaluation
# ---------------------------------------------------------------------------------------------------------------------
def safe_start_joint_states(engine, snapshot: Dict[str, Any], robots: Sequence[str], n_track: int = 60):
    """{robot: {joint: value}} start posture with the arm at its zero pose and both external axes (linear track,
    carriage) at the collision-free combination that maximizes the clearance to the pipe/positioner.

    A headless run has no operator-chosen start; q=0 puts the track at the rail end, where the folded arm touches the
    positioner column (the APF track map shows contact there), so every plan would die with "start configuration is in
    collision" before the planner (and its use of the track axes) even starts. The search sweeps linear_track x carriage
    and prefers, among the clearest cells, the one closest to the pipe's x centre so the track has little to travel."""
    import copy
    mesh = engine._current_spool_collision_mesh()
    positioner = engine._build_positioner_collision_mesh()
    backend = engine._robotics_backend
    centre_x = float(np.asarray(mesh.get_center())[0])
    states: Dict[str, Dict[str, float]] = {}
    for robot in robots:
        handle = backend._handle(robot)
        base_T = np.asarray(handle.description.base_T, dtype=float)
        inv = np.linalg.inv(base_T)
        statics = []
        for m in (mesh, positioner):
            if m is not None:
                m = copy.deepcopy(m)
                m.transform(inv)
                statics.append(m)
        backend.configure_collision(robot, static_meshes=statics, sample_resolution=0.02)
        names = backend.joint_names(robot)
        lin = next(i for i, n in enumerate(names) if "linear_track" in n)
        car = next(i for i, n in enumerate(names) if "carriage" in n)
        model = backend.robot_model(robot)
        lo, hi = np.asarray(model.lowerPositionLimit, float), np.asarray(model.upperPositionLimit, float)
        best = None
        for x in np.linspace(lo[lin], hi[lin], n_track):
            for y in np.linspace(lo[car], hi[car], 8):
                q = np.zeros(backend.dof(robot))
                q[lin], q[car] = x, y
                entries = backend.link_obstacle_distances(robot, q)
                d = min((e["distance"] for e in entries), default=math.inf)
                if backend.check_collision(robot, q).collision:
                    d = min(d, -1.0)
                # clear first (capped at 0.3 m - beyond that clearance no longer matters), then close to the pipe
                key = (min(d, 0.3), -abs(float(backend.frame_world_T(robot, q)[0, 3]) - centre_x))
                if best is None or key > best[0]:
                    best = (key, x, y, d)
        states[robot] = {names[lin]: float(best[1]), names[car]: float(best[2])}
    return states



# ---------------------------------------------------------------------------------------------------------------------
# glossary written next to the evaluation outputs
# ---------------------------------------------------------------------------------------------------------------------
_TARGET_METRICS = [
    # (column, 지표, 단위, 설명)
    ("safe", "안전 성공", "bool", "계획됨 & C_eff >= 0 & 필수 비관통 OK & E_d <= eps_d & E_theta <= eps_theta "
                               "(MC 사용 시 P_collision <= 최대 충돌률)"),
    ("planned", "계획 성공", "bool", "경로가 만들어졌는지"),
    ("C_nom", "명목 여유거리 C_nom", "m", "연속 경로 전체의 로봇 링크-장애물 최소 거리 (접근 구간의 센서-배관 쌍, 가상 벽 제외)"),
    ("C_eff", "유효 여유거리 C_eff", "m", "C_nom - 안전 마진 m (>= 0 이어야 안전)"),
    ("mandatory_min_distance", "필수 최소 간격", "m", "센서-배관 포함 모든 로봇-정적 장애물 쌍의 경로 전체 최소 거리"),
    ("mandatory_ok", "필수 비관통", "bool", "필수 최소 간격 > d_physical_min 이고 자기충돌 없음"),
    ("collision_samples", "충돌 샘플 수", "개", "경로 샘플 중 충돌(자기충돌 포함) 수"),
    ("E_d", "검사 거리 오차 E_d", "m", "|최종 센서-배관 거리 - d*|"),
    ("E_theta_deg", "검사 방향 오차 E_theta", "deg", "최종 자세의 센서 축과 -n(센서 광선이 맞는 배관 표면 법선) 사이 각도"),
    ("aim_error_deg", "조준 오차", "deg", "센서 축과 (검사점 - TCP) 방향 사이 각도"),
    ("E_pos", "TCP 위치 오차", "m", "최종 TCP 위치 - 목표 자세 위치"),
    ("E_rot_deg", "TCP 자세 오차", "deg", "최종 TCP 자세와 목표 자세의 회전 차이"),
    ("approach_dir_error_deg", "접근 방향 오차", "deg", "마지막 approach_span 구간 EE 이동 방향과 -n 사이 각도"),
    ("side_excess_max", "반대편 침범", "m", "팔이 배관 축 너머 로봇 반대편으로 넘어간 최대 거리 (<= 0 이면 없음)"),
    ("n_waypoints", "웨이포인트 수", "개", "계획된 경로(q_path)의 웨이포인트 수"),
    ("n_interp_samples", "보간 샘플 수", "개", "웨이포인트 사이를 관절 공간 ccd_step 간격으로 보간한 점 수"),
    ("n_dense_samples", "평가 샘플 수", "개", "여유거리/충돌 평가에 쓴 점 수 = 보간 샘플 (ccd_max_samples 초과 시 균등 추출)"),
    ("L_ee", "궤적 길이 (EE)", "m", "TCP 경로 길이"),
    ("S_rev_deg", "궤적 길이 (최대 변위 각)", "deg", "구간별 최대 회전 관절 변위의 합"),
    ("S_lin", "궤적 길이 (직동 축)", "m", "구간별 최대 직동 관절(리니어 트랙/캐리지) 변위의 합"),
    ("T_motion", "모션 수행 시간", "s", "구간마다 가장 느린 관절(회전 속도/직동 속도 한계 기준)이 구간 시간을 정하고 전 관절 "
                                    "동기 이동, 경로 전체 사다리꼴 1회 (플레이백 시간과 같음)"),
    ("T_rev", "회전 축 수행 시간", "s", "S_rev 를 사다리꼴 속도 프로파일(속도/가속도 한계)로 이동하는 시간"),
    ("T_lin", "직동 축 수행 시간", "s", "S_lin 을 사다리꼴 속도 프로파일(직동 속도/가속도 한계)로 이동하는 시간"),
    ("L_q", "관절 경로 길이", "m+rad", "관절 공간 경로 길이"),
    ("L_q_normalized", "정규화 관절 경로 길이", "-", "관절 범위로 나눈 관절 공간 경로 길이"),
    ("t_plan", "계획 시간 T_plan", "s", "목표 1건 계획 소요 시간"),
    ("P_collision", "충돌 확률", "0~1", "Monte Carlo(추종/정합 오차) 시행 중 관통 비율"),
    ("P_violation", "위반 확률", "0~1", "MC 시행 중 충돌, C_nom<0 또는 검사 조건 위반 비율"),
    ("DTW_RMSE", "궤적 편차 DTW-RMSE", "m", "MC 실행 EE 궤적과 명목 궤적의 DTW-RMSE"),
]

_SUMMARY_METRICS = [
    ("SR_safe_pct", "안전 성공률 SR_safe", "%", "안전 성공 수 / 요청 수 (95% 신뢰구간: SR_safe_ci95_pct, Wilson)"),
    ("planned_rate_pct", "계획 성공률", "%", "계획 성공 수 / 요청 수"),
    ("C_eff_mean_safe_mm", "유효 여유거리 평균 (안전 성공)", "mm", ""),
    ("C_eff_min_planned_mm", "유효 여유거리 최소 (계획 성공)", "mm", ""),
    ("C_nom_min_planned_mm", "명목 여유거리 최소 (계획 성공)", "mm", ""),
    ("E_d_mean_mm", "검사 거리 오차 평균", "mm", ""),
    ("E_theta_mean_deg", "검사 방향 오차 평균", "deg", ""),
    ("approach_dir_error_mean_deg", "접근 방향 오차 평균", "deg", ""),
    ("waypoints_mean / waypoints_total", "웨이포인트 수 평균 / 합", "개", "계획 성공 경로"),
    ("eval_samples_mean / n_paths_sample_capped", "평가 샘플 수 평균 / 상한에 걸린 경로 수", "개", "ccd_max_samples 상한"),
    ("L_ee_mean_m / L_ee_total_m", "궤적 길이 평균 / 합", "m", "계획 성공 경로"),
    ("T_motion_mean_s / T_motion_total_s", "모션 수행 시간 평균 / 합", "s", "계획 성공 경로"),
    ("L_q_normalized_mean", "정규화 관절 경로 길이 평균", "-", ""),
    ("T_plan_mean_s / _p50_s / _p95_s", "계획 시간 평균 / 중앙값 / 95% 분위", "s", ""),
    ("timeout_rate_pct", "타임아웃 비율", "%", ""),
    ("P_collision_pct / P_violation_pct", "충돌률 / 위반률 평균", "%", "MC 사용 시"),
    ("DTW_RMSE_mean_mm", "궤적 편차 평균", "mm", "MC 사용 시"),
]


def write_metric_glossary(out_dir, cfg: Optional[MetricConfig] = None, filename: str = "METRICS.md") -> str:
    """Write the metric names/units/definitions (targets.csv and summary columns) to <out_dir>/METRICS.md."""
    from pathlib import Path
    cfg = cfg or MetricConfig()
    lines = ["# 평가 지표 (path_plan_metric.md)", "",
             f"설정: 안전 마진 m = {cfg.margin} m, d_physical_min = {cfg.physical_min} m, "
             f"approach_zone = {cfg.approach_zone} m, d* = {cfg.d_star}, eps_d = {cfg.eps_d} m, "
             f"eps_theta = {cfg.eps_theta_deg} deg, ccd_step = {cfg.ccd_step}, MC 시행 = {cfg.mc_trials}회, "
             f"센서 로봇 = {', '.join(cfg.sensor_kinds)}",
             f"속도 프로파일: 사다리꼴, 회전 축 {cfg.joint_vel_deg:g} deg/s, {cfg.joint_acc_deg:g} deg/s^2 "
             f"(구간별 최대 변위 각 기준), 직동 축 {cfg.lin_vel:g} m/s, {cfg.lin_acc:g} m/s^2", "",
             "## 목표별 지표 (targets.csv)", "", "| 컬럼 | 지표 | 단위 | 설명 |", "|---|---|---|---|"]
    lines += [f"| `{c}` | {n} | {u} | {d} |" for c, n, u, d in _TARGET_METRICS]
    lines += ["", "## 전체 지표 (summary.json / summary.md)", "", "| 키 | 지표 | 단위 | 설명 |", "|---|---|---|---|"]
    lines += [f"| `{c}` | {n} | {u} | {d} |" for c, n, u, d in _SUMMARY_METRICS]
    path = Path(out_dir) / filename
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
