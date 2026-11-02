"""Collision-free IK refinement on top of PyBullet distances.

An IK solve only minimizes the pose error, so the arm configuration it lands on can touch the pipe, the positioner or the
arm itself. refine_collision_free() takes such a solution and moves it, keeping the end-effector pose, to a configuration
whose link-obstacle and link-link distances are all above a safety distance:

    cost(q)  = sum over (link, obstacle) and non-adjacent (link, link) pairs of max(0, d_safe - distance)^2
    each iteration:
        dq_task = J^+ * pose_error                         (pull back to the target pose, damped pseudo-inverse)
        dq_null = (I - J^+ J) * (-grad cost)               (push away from collisions without leaving the pose)
        q      <- clip(q + dq_task + dq_null, joint limits)

Distances (and therefore the gradient, by finite differences) come from PyBullet closest points on the meshes - the same
engine that validates the result - and the Jacobian from Pinocchio kinematics. The intended sensor-pipe pair only has to stay
above `sensor_min` (a touch is a collision), the rest of the arm has to stay above `d_safe`.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None

PIPE_OBSTACLE = "collision_object_0"


class _Cost:
    def __init__(self, scene, sensor_link: str, d_safe: float, sensor_min: float, d_self: float, obstacle_distances=None):
        self.scene, self.sensor, self.d_safe, self.sensor_min, self.d_self = scene, sensor_link, d_safe, sensor_min, d_self
        self.cap = max(d_safe, d_self) + 0.02
        # obstacle_distances(q) -> [{link, obstacle, distance}]: the scene SDF when given (fast, ~voxel accurate),
        # otherwise exact PyBullet closest points. Self distances are always exact (cheap).
        self.obstacle_distances = obstacle_distances or (lambda q: self.scene.distances(q, self.cap))

    def __call__(self, q) -> Tuple[float, float]:
        """(cost, smallest margin = distance - threshold over all pairs; >= 0 means every pair is clear)."""
        cost, margin = 0.0, np.inf
        for e in self.obstacle_distances(q):
            intended = bool(self.sensor) and str(e["link"]).startswith(self.sensor) and e["obstacle"] == PIPE_OBSTACLE
            thr = self.sensor_min if intended else self.d_safe
            gap = thr - e["distance"]
            margin = min(margin, -gap)
            if gap > 0:
                cost += gap * gap
        for e in self.scene.self_distances(q, self.d_self + 0.02):
            gap = self.d_self - e["distance"]
            margin = min(margin, -gap)
            if gap > 0:
                cost += gap * gap
        return cost, margin


def refine_collision_free(backend, robot_name: str, q0: Sequence[float], target_world_T: np.ndarray, frame_name: str,
                          sensor_link: Optional[str] = None, *, d_safe: float = 0.03, sensor_min: float = 0.003,
                          d_self: float = 0.01, max_iter: int = 60, pos_tol: float = 1e-3, rot_tol: float = 0.02,
                          step: float = 0.05, damping: float = 1e-2, fd_eps: float = 1e-3,
                          time_budget: float = 4.0, use_sdf: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Move q0 to a collision-free configuration with the same end-effector pose. Returns (q, info);
    info["success"] means pose within tolerance and every pair above its threshold."""
    handle = backend._handle(robot_name)
    scene = handle.pb_scene
    if scene is None or pin is None:
        return np.asarray(q0, dtype=float), {"success": False, "reason": "pybullet scene unavailable", "iterations": 0}
    model, data = handle.model, handle.model.createData()
    fid = model.getFrameId(frame_name)
    base_T = np.asarray(handle.description.base_T, dtype=float)
    T_goal = np.linalg.inv(base_T) @ np.asarray(target_world_T, dtype=float)
    lo = np.where(np.isfinite(model.lowerPositionLimit), model.lowerPositionLimit, -np.pi)
    hi = np.where(np.isfinite(model.upperPositionLimit), model.upperPositionLimit, np.pi)
    sdf_distances = (lambda q: backend.link_obstacle_distances_sdf(robot_name, q)) if use_sdf else None
    cost_fn = _Cost(scene, sensor_link or "", d_safe, sensor_min, d_self, obstacle_distances=sdf_distances)

    def pose_error(q):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        M = data.oMf[fid]
        e_pos = T_goal[:3, 3] - np.asarray(M.translation)
        e_rot = Rotation.from_matrix(T_goal[:3, :3] @ np.asarray(M.rotation).T).as_rotvec()
        return e_pos, e_rot

    q = np.clip(np.asarray(q0, dtype=float).copy(), lo, hi)
    n = len(q)
    t0 = time.perf_counter()
    best = None   # (score, q) lowest cost among pose-accurate iterates
    info: Dict[str, Any] = {"success": False, "iterations": 0}
    for it in range(max_iter + 1):
        cost, margin = cost_fn(q)
        e_pos, e_rot = pose_error(q)
        pose_ok = np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol
        if pose_ok and (best is None or cost < best[0]):
            best = (cost, q.copy())
        info.update({"iterations": it, "cost": float(cost), "margin": float(margin),
                     "pos_err": float(np.linalg.norm(e_pos)), "rot_err": float(np.linalg.norm(e_rot))})
        if pose_ok and cost <= 0.0:
            info["success"] = True
            return q, info
        if it == max_iter or time.perf_counter() - t0 > time_budget:
            break
        J = np.asarray(pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        Jp = J.T @ np.linalg.inv(J @ J.T + damping ** 2 * np.eye(6))
        dq = Jp @ np.concatenate([e_pos, e_rot])
        if cost > 0.0:
            grad = np.zeros(n)
            for j in range(n):
                qj = q.copy()
                qj[j] += fd_eps
                grad[j] = (cost_fn(qj)[0] - cost) / fd_eps
            null = (np.eye(n) - Jp @ J) @ (-grad)
            peak = float(np.max(np.abs(null)))
            if peak > 1e-12:
                dq = dq + null * (step / peak)
        peak = float(np.max(np.abs(dq)))
        if peak > step:
            dq = dq * (step / peak)
        q = np.clip(q + dq, lo, hi)
    if best is not None and best[0] < cost_fn(np.asarray(q0, dtype=float))[0]:
        info["best_cost"] = float(best[0])
        return best[1], info
    return np.asarray(q0, dtype=float), info


def apf_goal_refine(backend, robot_name: str, q0: Sequence[float], target_world_T: np.ndarray, frame_name: str,
                    sensor_link: Optional[str] = None, *, d0: float = 0.3, eta: float = 1.0, step: float = 0.05,
                    max_iter: int = 40, time_budget: float = 3.0, fd_eps: float = 0.05, pos_tol: float = 1e-3,
                    rot_tol: float = 0.01, max_halvings: int = 3, abs_tol: float = 1e-4,
                    keep_side_margin: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Moves a collision-free goal posture away from the obstacles in the null space of the TCP Jacobian - the TCP pose,
    hence the sensor-pipe distance, stays exactly as solved; only the redundant motion (linear track, carriage, arm
    self-motion) changes the rest of the arm.

        U(q) = sum over (link, obstacle) pairs of 0.5*eta*(1/d - 1/d0)^2 for d < d0   (APF, scene SDF) - except the
               sensor-pipe pair and the links that cannot move with the pose held (null-space locked)
        each iteration:  dq = -V g scaled to `step` (V = null-space basis of the TCP Jacobian, g_k = central difference
                         of U along v_k with the pose corrected), then a damped least-squares pose correction
    A step is kept only if U decreases, the posture is collision free (exact check, self and C-space included) and - when
    keep_side_margin is given - the arm stays on the robot's side of the pipe; otherwise it is halved (max_halvings).
    Returns (q, info) - q0 unchanged when nothing improved."""
    handle = backend._handle(robot_name)
    if pin is None or not hasattr(backend, "link_obstacle_distances_sdf"):
        return np.asarray(q0, dtype=float), {"iterations": 0, "reason": "unavailable"}
    model, data = handle.model, handle.model.createData()
    fid = model.getFrameId(frame_name)
    base_T = np.asarray(handle.description.base_T, dtype=float)
    T_goal = np.linalg.inv(base_T) @ np.asarray(target_world_T, dtype=float)
    lo = np.where(np.isfinite(model.lowerPositionLimit), model.lowerPositionLimit, -np.pi)
    hi = np.where(np.isfinite(model.upperPositionLimit), model.upperPositionLimit, np.pi)
    sensor = str(sensor_link or "")
    # links that cannot move while the TCP pose is held - the TCP's rigid body, and in practice the whole wrist (p1 Point 1
    # pose 1 DDA: link4-6 do not move in the null space) - are left out of U: their constant terms swamped the gradient of
    # the links that can move (link5 at 103 mm was ~99% of U, every step was rejected on SDF noise). They are still in
    # the collision check. A link is locked when its frame velocity over the null-space basis is ~0 at q0.
    q_ref = np.asarray(q0, dtype=float)
    pin.forwardKinematics(model, data, q_ref)
    pin.updateFramePlacements(model, data)
    J0 = np.asarray(pin.computeFrameJacobian(model, data, q_ref, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
    null_basis = np.linalg.svd(J0)[2][np.linalg.matrix_rank(J0, tol=1e-4):].T
    locked = set()
    for f_id, f in enumerate(model.frames):
        Jf = np.asarray(pin.computeFrameJacobian(model, data, q_ref, f_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        if null_basis.size == 0 or np.linalg.norm(Jf @ null_basis) < 1e-3:
            locked.add(str(f.name))

    def is_locked(link):
        return re.sub(r"_\d+$", "", str(link)) in locked

    def cost(q):
        u, d_min = 0.0, np.inf
        for e in backend.link_obstacle_distances_sdf(robot_name, q, cap=max(0.35, d0 + 0.05)):
            if sensor and str(e["link"]).startswith(sensor) and e["obstacle"] == PIPE_OBSTACLE:
                continue
            if is_locked(e["link"]):
                continue
            d = float(e["distance"])
            d_min = min(d_min, d)
            if d < d0:
                dd = max(d, 1e-3)
                u += 0.5 * eta * (1.0 / dd - 1.0 / d0) ** 2
        return u, d_min

    def pose_correct(q):
        for _ in range(20):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            M = data.oMf[fid]
            e_pos = T_goal[:3, 3] - np.asarray(M.translation)
            e_rot = Rotation.from_matrix(T_goal[:3, :3] @ np.asarray(M.rotation).T).as_rotvec()
            if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
                return q
            J = np.asarray(pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), np.concatenate([e_pos, e_rot])), lo, hi)
        return None

    q = np.asarray(q0, dtype=float).copy()
    n = len(q)
    u, d_min = cost(q)
    info: Dict[str, Any] = {"locked_links": sorted(n for n in locked if "link" in n), "cost_before": float(u), "clearance_before": float(d_min), "iterations": 0, "accepted": 0,
                            "rejected": {"pose": 0, "cost": 0, "collision": 0, "side": 0}}
    t0 = time.perf_counter()
    for it in range(max_iter):
        if u <= 0.0 or time.perf_counter() - t0 > time_budget:
            break
        info["iterations"] = it + 1
        # gradient along the null-space basis only: central differences of U over pose-corrected probes q +- eps*v_k.
        # Terms of links that do not move with the pose held cancel exactly; a joint-wise finite difference leaves the TCP
        # pose and mixed in the SDF noise of the near-locked wrist links (they dominate U).
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        J = np.asarray(pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        basis = np.linalg.svd(J)[2][np.linalg.matrix_rank(J, tol=1e-4):]
        if basis.size == 0:
            break
        g = np.zeros(len(basis))
        for k, v in enumerate(basis):
            for eps in (fd_eps, 0.25 * fd_eps):   # smaller probe / one-sided when a probe leaves the pose or the limits
                qp, qm = pose_correct(np.clip(q + eps * v, lo, hi)), pose_correct(np.clip(q - eps * v, lo, hi))
                if qp is not None and qm is not None:
                    g[k] = (cost(qp)[0] - cost(qm)[0]) / (2.0 * eps)
                elif qp is not None or qm is not None:
                    g[k] = (cost(qp)[0] - u) / eps if qp is not None else (u - cost(qm)[0]) / eps
                else:
                    info["rejected"]["probe"] = info["rejected"].get("probe", 0) + 1
                    continue
                break
        # directions: the gradient step first, then +-each basis vector (pattern search fallback - the APF of the
        # nearest links is not smooth at the probe scale and the gradient step sometimes raises U at every step size)
        directions = []
        if float(np.max(np.abs(basis.T @ g))) > 1e-12:
            directions.append(-(basis.T @ g))
        directions += [sign * v for v in basis for sign in (1.0, -1.0)]
        accepted, gain = False, 0.0
        for d_i, direction in enumerate(directions):
            peak = float(np.max(np.abs(direction)))
            s = step
            for _ in range(max_halvings + 1):
                s_used, s = s, s * 0.5
                cand = pose_correct(np.clip(q + direction * (s_used / peak), lo, hi))
                if cand is None:
                    info["rejected"]["pose"] += 1
                    continue
                u_new, d_new = cost(cand)
                if u_new >= u:
                    info["rejected"]["cost"] += 1
                    continue
                if backend.check_collision(robot_name, cand).collision:
                    info["rejected"]["collision"] += 1
                    continue
                if keep_side_margin is not None and not robot_side_ok(backend, robot_name, cand, keep_side_margin):
                    info["rejected"]["side"] += 1
                    continue
                gain = u - u_new
                q, u, d_min, accepted = cand, u_new, d_new, True
                info["accepted"] += 1
                if d_i > 0:
                    info["pattern_steps"] = info.get("pattern_steps", 0) + 1
                break
            if accepted:
                break
        if not accepted or gain < abs_tol:
            break
    info["q_before"] = [float(x) for x in np.asarray(q0, dtype=float)]
    info.update({"cost_after": float(u), "clearance_after": float(d_min), "time_s": time.perf_counter() - t0,
                 "joint_change_max": float(np.max(np.abs(q - np.asarray(q0, dtype=float))))})
    return q, info


def _pipe_frame(backend, robot_name):
    """(centre, axis direction) of the pipe (first static mesh, robot base frame) from a PCA of its vertices, cached."""
    handle = backend._handle(robot_name)
    meshes = list(handle._configured_meshes or [])
    key = tuple(id(m) for m in meshes[:1])
    cache = getattr(handle, "_pipe_frame_cache", None)
    if cache is not None and cache[0] == key:
        return cache[1], cache[2]
    if not meshes:
        return None, None
    v = np.asarray(meshes[0].vertices, dtype=float)
    centre = v.mean(axis=0)
    axis = np.linalg.svd(v - centre, full_matrices=False)[2][0]
    handle._pipe_frame_cache = (key, centre, axis)
    return centre, axis


_ARM_LINK_EXCLUDE = ("linear", "base", "carriage", "slide", "end", "link0")


def robot_side_excess(backend, robot_name, q) -> float:
    """How far (m) the arm reaches beyond the pipe axis onto the far side from the robot, horizontally and perpendicular to
    the pipe axis; the end effector is exempt. <= 0: the whole arm is on the robot's side. None if there is no pipe."""
    centre, axis = _pipe_frame(backend, robot_name)
    if centre is None:
        return None
    handle = backend._handle(robot_name)
    model, data = handle.model, handle.model.createData()
    h = np.cross([0.0, 0.0, 1.0], axis)
    if np.linalg.norm(h) < 1e-6:
        return None
    h = h / np.linalg.norm(h)
    pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
    pin.updateFramePlacements(model, data)
    base_name = next((f.name for f in model.frames if str(f.name).endswith("link0")), None)
    if base_name is None:
        return None
    side = np.sign(float((np.asarray(data.oMf[model.getFrameId(base_name)].translation) - centre) @ h)) or 1.0
    worst = -np.inf
    for f in model.frames:
        name = str(f.name)
        if "link" not in name or any(t in name for t in _ARM_LINK_EXCLUDE):
            continue
        worst = max(worst, -side * float((np.asarray(data.oMf[model.getFrameId(name)].translation) - centre) @ h))
    return float(worst)


def robot_side_ok(backend, robot_name, q, margin: float = 0.25) -> bool:
    """True if the arm stays on its own side of the pipe (robot_side_excess <= margin): catches IK solutions that reach over
    the pipe from the opposite side."""
    excess = robot_side_excess(backend, robot_name, q)
    return True if excess is None else excess <= margin

# ---------------------------------------------------------------------------------------------------------------------
# solver: IKOptions.solver == "collision_free"
# ---------------------------------------------------------------------------------------------------------------------
COLLISION_FREE_SOLVERS = ("collision_free", "collision_free_ik", "cf")


def track_q_nearest_target(backend, robot_name, q, target_world_T):
    """q with the linear-track joint set so the arm base (link0) is at the target's x (along the track axis), clipped to
    the track limits. The track is prismatic, so link0's position is affine in the track value: two FKs give it."""
    handle = backend._handle(robot_name)
    model = handle.model
    names = backend.joint_names(robot_name)
    lin = [i for i, n in enumerate(names) if "linear_track" in n]
    base_name = next((f.name for f in model.frames if str(f.name).endswith("link0")), None)
    if not lin or base_name is None:
        return np.asarray(q, dtype=float)
    i = lin[0]
    data = model.createData()
    fid = model.getFrameId(base_name)
    T_goal = np.linalg.inv(np.asarray(handle.description.base_T, dtype=float)) @ np.asarray(target_world_T, dtype=float)

    def base_pos(value):
        qq = np.asarray(q, dtype=float).copy()
        qq[i] = value
        pin.forwardKinematics(model, data, qq)
        pin.updateFramePlacements(model, data)
        return np.array(data.oMf[fid].translation, dtype=float, copy=True)   # a copy: oMf is a view into `data`

    p0, p1 = base_pos(0.0), base_pos(1.0)
    direction = p1 - p0                       # base-frame displacement per metre of track
    denom = float(direction @ direction)
    if denom < 1e-12:
        return np.asarray(q, dtype=float)
    value = float((T_goal[:3, 3] - p0) @ direction / denom)
    lo, hi = float(model.lowerPositionLimit[i]), float(model.upperPositionLimit[i])
    out = np.asarray(q, dtype=float).copy()
    out[i] = float(np.clip(value, lo, hi)) if np.isfinite(lo) and np.isfinite(hi) else value
    return out


def solve_collision_free_ik(backend, robot_name, target_world_T, q_init, options, frame_name=None):
    """IK that returns a collision-free configuration with the requested pose (PyBullet distances).

    1. Several PyBullet IK solves from different seeds (q_init, then random arm configurations with the linear track kept
       near q_init and the carriage resampled) - the converged ones are the candidates.
    2. Candidates that collide, or are closer than `clearance_safe` to anything, are pushed away from the obstacles in the
       null space of the pose Jacobian (refine_collision_free), keeping the end-effector pose.
    3. Among the collision-free candidates the one with the largest smallest-distance to every obstacle (sensor-pipe pair
       excluded, it is meant to be close) wins; if several are farther than `multi_start_clearance_cap`, the one nearest
       q_init in joint space.
    4. ik_goal_apf_refine (default true): the chosen goal is moved away from the obstacles in the null space of the TCP
       Jacobian (apf_goal_refine: APF over every link-obstacle pair except sensor-pipe, TCP pose held) - the goal posture
       is settled here, the path optimizers only shape the intermediate path.
    If nothing is collision free the least-bad candidate is returned with success=False
    (failure_info["reason"] == "no_collision_free_solution").
    keep_robot_side (default true): among the collision-free candidates only those whose arm stays on the robot's own side of
    the pipe (robot_side_ok, side_margin 0.25 m) are considered, unless there is none.
    Parameters come from options.extra: multi_start (seeds, default 16), multi_start_clearance_cap (0.5 m),
    multi_start_time_budget (8 s), multi_start_seed, multi_start_max_iter (2000: PyBullet IK iterations per seed - with
    the UI's 100000 one seed took seconds and the budget ran out after 3-4 of the 16 seeds), collision_free_refine (true),
    clearance_safe (0.03 m), self_clearance (0.01 m), sensor_min_distance (0.003 m), refine_top_k (4),
    refine_time_budget (4 s).
    Self collision counts like obstacle collision everywhere: a candidate's depth is min(obstacle distance, link-link
    distance), so when nothing is collision free the returned one is the least penetrating overall (before, only the
    obstacle distance was compared and a deeply self-colliding solution far from the pipe won).
    """
    from dataclasses import replace

    cfg = dict(options.extra or {})
    n_seeds = int(cfg.get("multi_start", 16))
    cap = float(cfg.get("multi_start_clearance_cap", 0.5))
    budget = float(cfg.get("multi_start_time_budget", 8.0))
    safe = float(cfg.get("clearance_safe", 0.03))
    handle = backend._handle(robot_name)
    frame = frame_name or handle.description.target_frame
    target_world_T = np.asarray(target_world_T, dtype=float)
    q_init = np.asarray(q_init, dtype=float)
    q_start = q_init   # joint-distance reference stays the caller's start posture
    # seed_track_to_target (default true): every seed starts with the linear track at the target's x instead of the start
    # posture's track position (often the rail end, metres away from the target), and only jitters it by seed_track_jitter.
    seed_track = bool(cfg.get("seed_track_to_target", True))
    track_jitter = float(cfg.get("seed_track_jitter", 0.3 if seed_track else 1.0))
    if seed_track:
        q_init = track_q_nearest_target(backend, robot_name, q_init, target_world_T)
    per_seed_iter = int(cfg.get("multi_start_max_iter", 2000))
    base_options = replace(options, solver="pybullet", extra={},
                           max_iter=min(int(options.max_iter), per_seed_iter) if per_seed_iter > 0 else options.max_iter)
    rng = np.random.default_rng(int(cfg.get("multi_start_seed", 0)))
    t0 = time.perf_counter()

    model = handle.model
    lo = np.asarray(model.lowerPositionLimit, dtype=float)
    hi = np.asarray(model.upperPositionLimit, dtype=float)
    lo = np.where(np.isfinite(lo), lo, -np.pi)
    hi = np.where(np.isfinite(hi), hi, np.pi)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    names = backend.joint_names(robot_name)
    lin = [i for i, n in enumerate(names) if "linear_track" in n]
    car = [i for i, n in enumerate(names) if "carriage" in n]
    sensor = str(frame or "")

    t_acc = {"solve": 0.0, "score": 0.0}
    use_sdf = bool(cfg.get("score_with_sdf", True)) and hasattr(backend, "link_obstacle_distances_sdf")

    def timed_solve(q_seed):
        t = time.perf_counter()
        try:
            return backend.solve_ik(robot_name, target_world_T, q_seed, options=base_options, frame_name=frame)
        finally:
            t_acc["solve"] += time.perf_counter() - t

    def score(q):
        t = time.perf_counter()
        try:
            return _score(q)
        finally:
            t_acc["score"] += time.perf_counter() - t

    def _score(q):
        """(collision, clearance, depth): clearance = smallest obstacle distance (sensor-pipe pair exempt), used to rank
        collision-free solutions; depth = min(clearance, smallest link-link distance), used to rank colliding ones."""
        collision = bool(backend.check_collision(robot_name, q).collision)   # exact (PyBullet meshes)
        # ranking distance from the precomputed scene SDF (~4 ms) - exact PyBullet closest points with a 0.5 m radius
        # took 2.6-4 s per call against the ~100k-triangle scanned pipe and made one IK target take 20-30 s.
        # score_with_sdf=false restores the exact distances.
        if use_sdf:
            entries = backend.link_obstacle_distances_sdf(robot_name, q, cap=max(cap, 0.35))
        else:
            entries = backend.link_obstacle_distances(robot_name, q, max_distance=cap)
        dists = [e["distance"] for e in entries
                 if not (sensor and str(e["link"]).startswith(sensor) and e["obstacle"] == PIPE_OBSTACLE)]
        clearance = min(dists) if dists else cap
        self_d = cap
        if handle.pb_scene is not None:
            self_d = min((e["distance"] for e in handle.pb_scene.self_distances(q, 0.1)), default=cap)
        return collision, clearance, min(clearance, self_d)

    # joint distance to the start posture used to choose the goal: per-joint |dq| / joint range, weighted (base and wrist3
    # x2 by default: a goal on the other side of the base or with a flipped wrist makes the arm swing far around)
    weights = np.ones(len(names))
    for key, w in (cfg.get("ik_goal_joint_weights") or {"base": 2.0, "wrist3": 2.0, "joint1": 2.0}).items():
        for i, n in enumerate(names):
            if str(key) in n:
                weights[i] = float(w)

    def joint_distance(q):
        return float(np.linalg.norm(weights * (np.asarray(q, dtype=float) - q_start) / span))

    candidates = []   # (collision, clearance, weighted joint distance to the start posture, IKResult, depth)
    t_ik = time.perf_counter()
    first = timed_solve(q_init)
    t_first_solve = t_acc["solve"]
    n_tried = 1

    def consider(res):
        if res is None or res.q is None or not res.success:
            return
        q = np.asarray(res.q, dtype=float)
        if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):   # outside the URDF joint limits
            return
        collision, clearance, depth = score(q)
        candidates.append((collision, clearance, joint_distance(q), res, depth))

    consider(first)
    counters = {"tried": 1, "refined": 0, "refined_ok": 0}

    def run_seeds(n, time_budget, jitter, stratified=False):
        """n random seeds (arm joints uniform, track near the target +- jitter, carriage uniform) within time_budget.
        stratified: the track seeds are spread evenly over the whole rail instead (one per slice, random inside it)."""
        t_round = time.perf_counter()
        for k in range(n):
            if time.perf_counter() - t_round > time_budget:
                break
            counters["tried"] += 1
            seed = q_init.copy()
            free = np.ones(len(seed), dtype=bool)
            for i in lin + car:
                free[i] = False
            seed[free] = rng.uniform(lo[free], hi[free])
            for i in lin:
                if stratified:
                    seed[i] = float(lo[i] + (hi[i] - lo[i]) * (k + rng.uniform()) / max(n, 1))
                else:
                    seed[i] = float(np.clip(seed[i] + rng.uniform(-jitter, jitter), lo[i], hi[i]))
            for i in car:
                seed[i] = float(rng.uniform(lo[i], hi[i]))
            try:
                consider(timed_solve(seed))
            except Exception:
                continue

    run_seeds(n_seeds, budget, track_jitter)
    n_tried = counters["tried"]

    t_seeds = time.perf_counter() - t_ik
    # near-start search (ik_goal_selection "nearest"): if the best collision-free goal still needs an arm joint to move
    # more than ik_goal_near_trigger_deg (90), seed around the start posture (+- ik_goal_near_sigma_deg, 30) for a
    # goal on the same branch - the random seeds rarely land there (p1 Point 1 pose 2 RT: the only collision-free
    # candidate had the base turned 239 deg).
    n_near_seeds = 0
    t_near = time.perf_counter()
    arm = np.array([not (i in lin or i in car) for i in range(len(names))])

    def _max_arm_change_deg(c):
        return float(np.degrees(np.max(np.abs(np.asarray(c[3].q, dtype=float) - q_start)[arm]))) if arm.any() else 0.0

    if str(cfg.get("ik_goal_selection", "nearest")).lower() == "nearest":
        free_now = [c for c in candidates if not c[0]]
        trigger = float(cfg.get("ik_goal_near_trigger_deg", 90.0))
        if not free_now or min(_max_arm_change_deg(c) for c in free_now) > trigger:
            sigma = np.radians(float(cfg.get("ik_goal_near_sigma_deg", 30.0)))
            t_round = time.perf_counter()
            for _ in range(int(cfg.get("ik_goal_near_seeds", 48))):
                if time.perf_counter() - t_round > float(cfg.get("ik_goal_near_budget", 10.0)):
                    break
                seed = q_start.copy()
                for i in lin:
                    seed[i] = float(q_init[i])
                seed[arm] = np.clip(q_start[arm] + rng.normal(0.0, sigma, int(arm.sum())), lo[arm], hi[arm])
                for i in lin:
                    seed[i] = float(np.clip(seed[i] + rng.uniform(-track_jitter, track_jitter), lo[i], hi[i]))
                for i in car:
                    seed[i] = float(rng.uniform(lo[i], hi[i]))
                n_near_seeds += 1
                try:
                    consider(timed_solve(seed))
                except Exception:
                    continue
    t_near = time.perf_counter() - t_near

    n_refined = n_refined_ok = 0
    t_ref = time.perf_counter()

    def refine_round(pool, top_k, d_safe=None, d_self=None, time_budget=None):
        """Correct colliding IK candidates; a small positive clearance alone never triggers refinement."""
        todo = sorted((c for c in pool if c[0]), key=lambda c: -c[4])
        for _, _, _, res, _ in todo[:int(top_k)]:
            counters["refined"] += 1
            q_new, rinfo = refine_collision_free(
                backend, robot_name, res.q, target_world_T, frame, sensor,
                d_safe=safe if d_safe is None else float(d_safe),
                sensor_min=float(cfg.get("sensor_min_distance", 0.003)),
                d_self=float(cfg.get("self_clearance", 0.01)) if d_self is None else float(d_self),
                time_budget=float(cfg.get("refine_time_budget", 4.0)) if time_budget is None else float(time_budget),
                use_sdf=use_sdf)
            if not rinfo.get("success"):
                continue
            counters["refined_ok"] += 1
            new_res = replace(res, q=np.asarray(q_new, dtype=float), success=True,
                              position_error=float(rinfo["pos_err"]), orientation_error=float(rinfo["rot_err"]))
            collision, clearance, depth = score(new_res.q)
            candidates.append((collision, clearance, joint_distance(new_res.q), new_res, depth))

    # A collision-free candidate is enough. Clearance improvement is not a reason to run the expensive null-space solver.
    have_free = any(not c[0] for c in candidates)
    if bool(cfg.get("collision_free_refine", True)) and not have_free:
        refine_round(list(candidates), cfg.get("refine_top_k", 4))
    n_refined, n_refined_ok = counters["refined"], counters["refined_ok"]
    t_refine = time.perf_counter() - t_ref

    # retry while nothing is collision free (a goal_collision would follow): more seeds, a wider track search and
    # refinement of the new candidates, round after round until a collision-free solution turns up. Per round
    # (lists, the last entry repeats): collision_free_retry_seeds [32, 48, 128], _retry_time_budget [8, 10, 30] s,
    # _retry_track_jitter [0.6, 1.2, "rail"] - "rail" = track seeds spread evenly over the whole rail (a target can need
    # the arm base far from its own x: p1 Point 2 pose 2 DDA was only collision free with the track 0.8 m away, ~4% of
    # the converged solutions).
    n_retry_rounds = 0
    t_retry = time.perf_counter()
    retry_rounds = int(cfg.get("collision_free_retry_rounds", 3))

    def _per_round(key, default):
        val = cfg.get(key, default)
        vals = list(val) if isinstance(val, (list, tuple)) else [val]
        return vals[min(n_retry_rounds, len(vals) - 1)]

    while retry_rounds > n_retry_rounds and not any(not c[0] for c in candidates):
        jitter = _per_round("collision_free_retry_track_jitter", [0.6, 1.2, "rail"])
        n = int(_per_round("collision_free_retry_seeds", [32, 48, 128]))
        tb = float(_per_round("collision_free_retry_time_budget", [8.0, 10.0, 30.0]))
        n_retry_rounds += 1
        before = len(candidates)
        stratified = str(jitter).lower() == "rail"
        run_seeds(n, tb, 0.0 if stratified else float(jitter), stratified=stratified)
        if bool(cfg.get("collision_free_refine", True)) and not any(not c[0] for c in candidates[before:]):
            # relaxed targets: a candidate a few mm into a self collision cannot always reach the normal 10 mm / 30 mm
            # gaps, but a collision-free one is enough here (the result is re-checked exactly by score())
            refine_round(sorted(candidates, key=lambda c: -c[4])[:int(cfg.get("refine_top_k", 4))],
                         cfg.get("refine_top_k", 4),
                         d_safe=float(cfg.get("collision_free_retry_clearance_safe", 0.005)),
                         d_self=float(cfg.get("collision_free_retry_self_clearance", 0.002)),
                         time_budget=float(cfg.get("collision_free_retry_refine_time_budget", 8.0)))
    n_tried = counters["tried"]
    n_refined, n_refined_ok = counters["refined"], counters["refined_ok"]
    t_retry = time.perf_counter() - t_retry

    clear = [c for c in candidates if not c[0]]
    n_side_ok = None
    if clear and bool(cfg.get("keep_robot_side", True)):
        margin = float(cfg.get("side_margin", 0.25))
        same_side = [c for c in clear if robot_side_ok(backend, robot_name, c[3].q, margin)]
        n_side_ok = len(same_side)
        if same_side:
            clear = same_side
    info = {"n_side_ok": n_side_ok, "n_seeds": n_seeds, "n_tried": n_tried, "per_seed_max_iter": base_options.max_iter,
            "n_converged": len(candidates), "n_collision_free": len(clear),
            "n_refined": n_refined, "n_refined_ok": n_refined_ok, "clearance_cap": cap,
            "t_first_ik": t_first_solve, "first_ik_success": bool(first.success),
            "t_seeds": t_seeds, "t_ik_solve": t_acc["solve"], "t_score": t_acc["score"], "t_refine": t_refine,
            "n_retry_rounds": n_retry_rounds, "t_retry": t_retry, "n_near_seeds": n_near_seeds, "t_near": t_near,
            "seed_track": float(q_init[lin[0]]) if lin else None, "elapsed": time.perf_counter() - t0}
    if clear:
        # ik_goal_selection "nearest" (default): among the collision-free candidates with clearance >= ik_goal_min_clearance
        # (default clearance_safe, 30 mm) the one closest to the start posture (weighted joint distance) - the largest
        # clearance used to win, which picked e.g. a goal with the base turned 239 deg from the start (p1 Point 1 pose 2
        # RT) that the arm could only reach by swinging around outside. "clearance": the old rule. If no candidate is
        # clear enough, the largest clearance wins.
        min_clear = float(cfg.get("ik_goal_min_clearance", safe))
        roomy = [c for c in clear if c[1] >= min_clear]
        if str(cfg.get("ik_goal_selection", "nearest")).lower() == "nearest" and roomy:
            chosen = min(roomy, key=lambda c: c[2])
            info["selection"] = "nearest"
        else:
            best = max(c[1] for c in clear)
            chosen = min((c for c in clear if c[1] >= best - 1e-3), key=lambda c: c[2])
            info["selection"] = "clearance"
        info.update({"chosen_clearance": float(chosen[1]), "chosen_joint_distance": float(chosen[2]),
                     "n_roomy": len(roomy), "chosen_max_arm_change_deg": _max_arm_change_deg(chosen)})
        res = chosen[3]
        # ik_goal_apf_refine (default true): the chosen goal is pushed away from the obstacles in the null space of the
        # TCP Jacobian (APF on every pair except sensor-pipe, TCP pose kept) - the goal posture is settled here, so the
        # path optimizers (CHOMP etc.) only shape the intermediate path.
        if bool(cfg.get("ik_goal_apf_refine", True)) and use_sdf:
            q_ref, ainfo = apf_goal_refine(
                backend, robot_name, res.q, target_world_T, frame, sensor,
                d0=float(cfg.get("ik_goal_apf_d0", 0.3)), eta=float(cfg.get("ik_goal_apf_eta", 1.0)),
                step=float(cfg.get("ik_goal_apf_step", 0.05)), max_iter=int(cfg.get("ik_goal_apf_max_iter", 40)),
                time_budget=float(cfg.get("ik_goal_apf_time_budget", 3.0)),
                keep_side_margin=float(cfg.get("side_margin", 0.25)) if bool(cfg.get("keep_robot_side", True))
                and robot_side_ok(backend, robot_name, res.q, float(cfg.get("side_margin", 0.25))) else None)
            info["apf_goal_refine"] = ainfo
            if ainfo.get("accepted"):
                res = replace(res, q=np.asarray(q_ref, dtype=float))
                info["chosen_clearance"] = float(score(res.q)[1])
            info["elapsed"] = time.perf_counter() - t0
        return replace(res, solver="collision_free", elapsed=info["elapsed"], extra=info)
    if candidates:   # converged but every one collides: report the one that is least deep in (obstacles and self)
        chosen = max(candidates, key=lambda c: c[4])
        res = chosen[3]
        info["chosen_clearance"] = float(chosen[1])
        info["chosen_depth"] = float(chosen[4])
    else:
        res = first
    fail = dict(res.failure_info or {})
    fail["reason"] = "no_collision_free_solution"
    return replace(res, success=False, solver="collision_free", elapsed=info["elapsed"], failure_info=fail, extra=info)
