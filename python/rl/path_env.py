"""Gymnasium environment for SAC inspection path planning (one robot, start q -> stored goal q).

Scene and goal are exactly those of the planning benchmark: the snapshot's target groups with the goal postures stored
by precompute_ik_goals.py ("ik_goal_q", APF null-space refined), the collision scene configured by
Visualizer._plan_inspection_path_for_robot ("ik_only" + "preset_goal_q": pipe, positioner, virtual side wall; phase 2 =
pipe and positioner rotated, benchmark_path_planners.apply_phase2_positioner), the exact collision check (PyBullet
meshes, self collision, C-space) for termination and the scene distance field for observations and shaping.

Observation (float32):
    q normalized to [-1, 1] by the joint limits                         n
    (q_goal - q) / joint range                                          n
    TCP pose error to the goal (position m, rotation vector rad)        6
    per link: smallest distance to any obstacle / cap (sensor-pipe excl) L
    per link: unit direction away from that obstacle (robot base frame)  3L
    sensor-pipe distance / cap, in-approach-zone flag                   2
    previous action                                                     n
    progress (step / max_steps)                                         1
Action: [-1, 1]^n -> joint step = action * max step per dt (rotary max_rot_vel_deg * dt, prismatic max_lin_vel * dt).
Reward (rl/sac.json "reward", mirrors the path metrics - runs/METRICS.md):
    + progress * (d_prev - d_now)          d = weighted normalized joint distance to the goal     (L_q, T_motion)
    - apf * min(U, apf_clip)               APF of every link-obstacle pair except sensor-pipe     (C_nom / C_eff)
    - margin * max(0, safety_margin - d_min)                                                      (C_eff >= 0)
    - side * max(0, side_excess)           arm past the pipe axis to the far side                 (side_excess_max)
    - time                                 per step                                               (T_motion)
    - smooth * |a - a_prev|^2                                                                     (acceleration)
    - collision, terminate                 exact edge check q -> q_next                           (mandatory_ok)
    + goal, terminate                      within reach_tol of the goal and the straight segment to it is free
The finished path (info["q_path"], goal appended on success) is what the planner plugin returns and the
PathMetricEvaluator scores - the same evaluation as the benchmark.
"""
from __future__ import annotations

import collections
import copy
import json
import pathlib
import pickle
import re
import sys
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

ROOT_PATH = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_PATH / "python") not in sys.path:
    sys.path.insert(0, str(ROOT_PATH / "python"))

DEFAULT_CONFIG = pathlib.Path(__file__).with_name("sac.json")
PIPE_OBSTACLE = "collision_object_0"


def load_rl_config(path=None) -> Dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return json.load(f)


def load_planning_config():
    from common.config_loader import load_config
    config = load_config(str(ROOT_PATH / "python" / "viewervedo.cfg"))
    config.update(load_config(str(ROOT_PATH / "python" / "path_planning.cfg")))
    config["root_path"] = ROOT_PATH
    config["verbose_level"] = "ERROR"
    return config


def snapshot_targets(snapshot: Dict[str, Any], robot: str, snapshot_path: str) -> List[Dict[str, Any]]:
    """Targets of one robot with a stored goal: [{snapshot, group, phase, target_pose, goal_q, start_q}]."""
    from plugins.robotics.inspection_workflow import inspection_group_pose_items
    out = []
    for g in snapshot.get("target_groups") or []:
        goal = (g.get("ik_goal_q") or {}).get(robot)
        info = (g.get("ik_goal_info") or {}).get(robot) or {}
        if goal is None or not info.get("success"):
            continue
        pose = next((t for r, _p, t in inspection_group_pose_items(g) if r == robot), None)
        if pose is None:
            continue
        out.append({"snapshot": snapshot_path, "group": g.get("name"), "phase": int(info.get("phase", 1)),
                    "target_pose": np.asarray(pose, dtype=float), "goal_q": np.asarray(goal, dtype=float),
                    "start_q": np.asarray(info.get("start_q"), dtype=float)})
    return out


def load_reference_paths(snapshot_path: str, robot: str, runs_by_scenario: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
    """{group name: [q path, ...]} of the robot's successful plans in the run folder(s) of the snapshot's scenario
    (runs_apfgoal/<algorithm>_<optimizer>_<pN>: summary.csv + NN_<robot>_<pose>/joint_states.csv). A scenario entry is a
    folder, a glob pattern ("runs_apfgoal/*_p1": every algorithm/optimizer -> several references per target) or a list."""
    m = re.search(r"_(p\d+)_", pathlib.Path(snapshot_path).name)
    spec = runs_by_scenario.get(m.group(1), runs_by_scenario.get("*")) if m else runs_by_scenario.get("*")
    if not spec:
        return {}
    out: Dict[str, List[np.ndarray]] = {}
    for item in (spec if isinstance(spec, list) else [spec]):
        dirs = sorted(ROOT_PATH.glob(item)) if any(c in item for c in "*?[") else [ROOT_PATH / item]
        for run_dir in dirs:
            for group, path in _load_run_paths(run_dir, robot).items():
                out.setdefault(group, []).append(path)
    return out


def _load_run_paths(run_dir: pathlib.Path, robot: str) -> Dict[str, np.ndarray]:
    import csv
    out = {}
    for row in csv.DictReader(open(run_dir / "summary.csv", encoding="utf-8-sig")):
        if row.get("robot_name") != robot or row.get("status") != "success":
            continue
        sub = next(run_dir.glob(f"{int(row['index']):02d}_{robot}_*"), None)
        if sub is None or not (sub / "joint_states.csv").exists():
            continue
        rows = list(csv.reader(open(sub / "joint_states.csv")))
        out[row["group_name"]] = np.asarray([[float(x) for x in r[1:]] for r in rows[1:]], dtype=float)
    return out


class _Scene:
    """One snapshot's headless engine; switches between phase 1 / phase 2 collision scenes."""

    def __init__(self, config, snapshot_path: str):
        from robot_core.worker import RobotCoreEngine
        with open(snapshot_path, "rb") as f:
            self.snapshot = pickle.load(f)
        self.path = snapshot_path
        self.engine = RobotCoreEngine(config, self.snapshot)
        self.backend = self.engine._robotics_backend
        self._positioner_phase1 = self.engine._worker_positioner_mesh
        self._pipe_phase1 = self.engine._current_spool_collision_mesh()
        self._pipe_phase2 = None
        self.active_key = None

    def activate(self, target: Dict[str, Any], robot: str) -> None:
        """Configure the robot's collision scene for the target's phase (same call path as planning)."""
        key = (target["phase"], robot)
        if key == self.active_key:
            return
        import benchmark_path_planners as bench
        if target["phase"] == 2:
            if self._pipe_phase2 is None:
                self._pipe_phase2 = copy.deepcopy(self._pipe_phase1)
                self._pipe_phase2.transform(np.asarray(self.snapshot["second_group_rotation_T"], dtype=float))
            bench.apply_phase2_positioner(self.engine, self.snapshot)
            obstacle = self._pipe_phase2
        else:
            self.engine._worker_positioner_mesh = self._positioner_phase1
            obstacle = self._pipe_phase1
        request = {"robot_name": robot, "planner": "rrt_connect", "step_size": 0.1, "max_iter": 3000,
                   "ik_solver": "collision_free", "ik_only": True, "preset_goal_q": target["goal_q"].tolist(),
                   "_start_q_override_by_robot": {robot: target["start_q"].tolist()},
                   "context_label": f"rl:{target['group']}"}
        self.engine._plan_inspection_path_for_robot(
            request, robot, target["target_pose"], obstacle, context_label=request["context_label"])
        self.active_key = key


class PathPlanningEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, rl_config: Optional[Dict[str, Any]] = None, snapshots: Optional[List[str]] = None,
                 split: str = "train", seed: Optional[int] = None):
        super().__init__()
        self.cfg = rl_config or load_rl_config()
        ecfg, self.rcfg = self.cfg["env"], self.cfg["reward"]
        self.robot = ecfg["robot"]
        self.max_steps = int(ecfg["max_steps"])
        self.cap = float(ecfg["obs_distance_cap"])
        self.approach_zone = float(ecfg["approach_zone"])
        self.reach_tol = float(ecfg["reach_tol"])
        self.episodes_per_scene = int(ecfg["episodes_per_scene"])
        self.rng = np.random.default_rng(seed)
        planning_config = load_planning_config()
        paths = snapshots or ecfg["train_snapshots" if split == "train" else "eval_snapshots"]
        self.scenes, self.targets = {}, []
        for p in paths:
            full = str((ROOT_PATH / p).resolve()) if not pathlib.Path(p).is_absolute() else p
            self.scenes[full] = _Scene(planning_config, full)
            self.targets += snapshot_targets(self.scenes[full].snapshot, self.robot, full)
        res = dict(ecfg.get("residual") or {})
        self.residual = bool(res.get("enabled", False))
        self.res_cfg = res
        if self.residual:
            # residual RL on a planner path: keep the targets whose reference ends exactly at the stored goal
            kept = []
            ref_cache = {}
            for t in self.targets:
                if t["snapshot"] not in ref_cache:
                    ref_cache[t["snapshot"]] = load_reference_paths(t["snapshot"], self.robot, res.get("reference_runs") or {})
                refs = [r for r in ref_cache[t["snapshot"]].get(t["group"], [])
                        if len(r) >= 2 and np.max(np.abs(r[-1] - t["goal_q"])) < 1e-6]
                if refs:
                    t["references"] = refs
                    t["reference"] = refs[0]
                    t["start_q"] = refs[0][0].copy()   # the benchmark's planning start
                    kept.append(t)
            self.targets = kept
        split = ecfg.get("target_split")   # optional: {"file": split json, "split": "train" | "eval"} - (snapshot, group)
        if split:
            sp = json.loads((ROOT_PATH / split["file"]).read_text(encoding="utf-8")) if not pathlib.Path(split["file"]).is_absolute() \
                else json.loads(pathlib.Path(split["file"]).read_text(encoding="utf-8"))
            keep = {(s, g) for s, g in sp[split.get("split", "train")]}
            self.targets = [t for t in self.targets if (pathlib.Path(t["snapshot"]).name, t["group"]) in keep]
        only = ecfg.get("target_filter")   # optional: train on these group names only
        if only:
            self.targets = [t for t in self.targets if t["group"] in set(only)]
        if not self.targets:
            raise RuntimeError(f"no stored goals for {self.robot} in {paths} (run precompute_ik_goals.py)")
        scene0 = next(iter(self.scenes.values()))
        self.backend = scene0.backend
        h = self.backend._handle(self.robot)
        self.n = int(h.model.nq)
        lo, hi = np.asarray(h.model.lowerPositionLimit, float), np.asarray(h.model.upperPositionLimit, float)
        self.lo, self.hi = np.where(np.isfinite(lo), lo, -np.pi), np.where(np.isfinite(hi), hi, np.pi)
        self.span = np.where(self.hi - self.lo > 1e-9, self.hi - self.lo, 1.0)
        from plugins.robotics.path_metrics import prismatic_mask_of
        prismatic = prismatic_mask_of(self.backend, self.robot, self.n)
        dt = float(ecfg["dt"])
        self.max_step = np.where(prismatic, float(ecfg["max_lin_vel"]) * dt, np.radians(float(ecfg["max_rot_vel_deg"])) * dt)
        names = list(self.backend.joint_names(self.robot))
        self.weights = np.ones(self.n)
        for key, w in (ecfg.get("joint_weights") or {}).items():
            for i, nm in enumerate(names):
                if key in nm:
                    self.weights[i] = float(w)
        self.prismatic = prismatic
        self.fixed_joint_indices = [int(i) for i in ecfg.get("fixed_joint_indices", [])]
        if any(i < 0 or i >= self.n for i in self.fixed_joint_indices):
            raise ValueError("env.fixed_joint_indices contains an out-of-range joint")
        # exact (mesh) edge check only near obstacles - see _edge_collides
        self.link_reach = float(ecfg.get("link_reach", 2.0))
        self.near_margin = float(ecfg.get("near_margin", 0.03))
        self.frame = h.description.target_frame
        self.sensor = str(self.frame or "")
        self._links = None   # filled from the SDF on first query
        # reverse curriculum: start on the goal -> start line at fraction `level` from the goal (0 = at the goal,
        # 1 = the real start); per env, raised / lowered by the recent success rate (rl/sac.json env.curriculum)
        cur = dict(ecfg.get("curriculum") or {})
        self.cur_enabled = bool(cur.get("enabled", False))
        self.cur_level = float(cur.get("start_level", 0.05)) if self.cur_enabled else 1.0
        self.cur_cfg = cur
        self.cur_hist = collections.deque(maxlen=int(cur.get("window", 20)))
        # scene randomization: random box obstacles around the reference path (env.random_obstacles)
        self.rand_cfg = dict(ecfg.get("random_obstacles") or {})
        self._boxes = []   # (pybullet body, StaticSDF) added on top of the configured scene
        self._box_specs = {}   # body id -> reproducible augmentation geometry
        self._scene_eps = 0
        self._scene_targets: List[Dict[str, Any]] = []
        self.target = None
        # observation size needs the link count: build the SDF once on a first target
        self._activate(self.targets[0])
        self._sdf_links()
        L = len(self._links)
        obs_dim = 2 * self.n + 6 + L + 3 * L + 2 + self.n + 1 + (self.n + 1 if self.residual else 0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.n,), dtype=np.float32)

    # -- scene / distances -------------------------------------------------------------------------------------
    def _activate(self, target):
        scene = self.scenes[target["snapshot"]]
        scene.activate(target, self.robot)
        self.backend = scene.backend
        self.target = target

    def _sdf_links(self):
        self.backend.link_obstacle_distances_sdf(self.robot, self.target["goal_q"])   # builds the field if needed
        field = self.backend._handle(self.robot).sdf_field
        self._links = [str(l) for l in field._links]
        return field

    def _link_distances(self, q):
        """Per link: (min distance over obstacles except sensor-pipe, unit direction away from it), sensor-pipe distance."""
        field = self._sdf_links()
        pin, model, data = field.pin, field.model, field.data
        pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
        pin.updateFramePlacements(model, data)
        R = np.stack([np.asarray(data.oMf[f].rotation) for f in field._fids])
        t = np.stack([np.asarray(data.oMf[f].translation) for f in field._fids])
        world = np.einsum("nij,nj->ni", R[field._link_of_point], field._pts) + t[field._link_of_point]
        L = len(field._links)
        best_d = np.full(L, self.cap)
        best_p = np.zeros((L, 3))
        best_o = np.full(L, -1)
        d_sp = self.cap
        for oi, (name, sdf) in enumerate(zip(field.names, field.sdfs)):
            d = sdf.distance(world)
            for li, link in enumerate(field._links):
                s = slice(field._starts[li], field._starts[li + 1] if li + 1 < L else len(d))
                k = int(np.argmin(d[s]))
                dk = float(d[s][k])
                if self.sensor and str(link).startswith(self.sensor) and name == PIPE_OBSTACLE:
                    d_sp = min(d_sp, dk)
                    continue
                if dk < best_d[li]:
                    best_d[li], best_p[li], best_o[li] = dk, world[s][k], oi
        # direction away from the nearest obstacle = SDF gradient at the closest sample (central differences)
        dirs = np.zeros((L, 3))
        eps = 0.01
        for oi, sdf in enumerate(field.sdfs):
            idx = np.where(best_o == oi)[0]
            if len(idx) == 0:
                continue
            P = best_p[idx]
            g = np.stack([(sdf.distance(P + e) - sdf.distance(P - e)) / (2 * eps) for e in np.eye(3) * eps], axis=1)
            nrm = np.linalg.norm(g, axis=1, keepdims=True)
            dirs[idx] = np.where(nrm > 1e-9, g / np.maximum(nrm, 1e-9), 0.0)
        return np.minimum(best_d, self.cap), dirs, d_sp

    def _self_or_cspace_collides(self, q):
        """Self collision (convex hulls, fast) and C-space box only - no robot-obstacle meshes."""
        import pybullet as pb
        if self.backend.cspace_violations(self.robot, q):
            return True
        sc = self.backend._handle(self.robot).pb_scene
        sc.set_q(q)
        for a, b in sc.self_pairs:
            if pb.getClosestPoints(sc.body, sc.body, 0.0, linkIndexA=sc.collision_links[a],
                                   linkIndexB=sc.collision_links[b], physicsClientId=sc.client):
                return True
        return False

    def _edge_collides(self, q_from, q_to, dmin_from, dmin_to):
        """Exact edge check (backend.check_edge_collision: meshes, self, C-space) when an obstacle is within reach of
        this step (field distance < link_reach * max rotary step + max prismatic step + near_margin; the field
        overestimates by up to ~2 cm), otherwise only self collision / C-space at the end point (18-50 ms -> ~1 ms
        per step in free space)."""
        dq = np.abs(np.asarray(q_to) - np.asarray(q_from))
        reach = (self.link_reach * float(np.max(np.where(self.prismatic, 0.0, dq)))
                 + float(np.max(np.where(self.prismatic, dq, 0.0))) + self.near_margin)
        if min(dmin_from, dmin_to) < reach:
            self.n_exact += 1
            return bool(self.backend.check_edge_collision(self.robot, q_from, q_to).collision)
        self.n_fast += 1
        return self._self_or_cspace_collides(q_to)

    def _augment_start(self):
        """Residual training augmentation: start at a random point of the reference's first part (mid_start_prob)
        and/or perturbed (start_noise_prob, start_noise steps), collision-free only."""
        c = self.res_cfg
        q, idx = self.ref[0].copy(), 0
        if self.rng.random() < float(c.get("mid_start_prob", 0.0)):
            idx = int(self.rng.integers(0, max(1, int(0.7 * (len(self.ref) - 1)))))
            q = self.ref[idx].copy()
        if self.rng.random() < float(c.get("start_noise_prob", 0.0)):
            for _ in range(5):
                cand = np.clip(q + self.rng.normal(0.0, float(c.get("start_noise", 0.3)), self.n) * self.max_step,
                               self.lo, self.hi)
                if not self.backend.check_collision(self.robot, cand).collision:
                    q = cand
                    break
        if (idx or not np.allclose(q, self.ref[0])) and self.backend.check_collision(self.robot, q).collision:
            q, idx = self.ref[0].copy(), 0
        self.q, self.ref_idx = q, idx   # the chosen reference's start (or the augmented point on / near it)

    def _densify_reference(self, ref):
        """Reference resampled so that consecutive points differ by at most half a step on every joint."""
        pts = [ref[0]]
        for a, b in zip(ref[:-1], ref[1:]):
            k = max(1, int(np.ceil(np.max(np.abs(b - a) / (0.5 * self.max_step)))))
            pts += [a + (b - a) * (i / k) for i in range(1, k + 1)]
        return np.asarray(pts)

    def _reference_target(self):
        """Advance the progress index to the nearest reference point ahead (window), return the point one step on."""
        ref = self.ref
        hi = min(len(ref), self.ref_idx + 12)
        dist = np.max(np.abs(ref[self.ref_idx:hi] - self.q) / self.max_step, axis=1)
        self.ref_idx = self.ref_idx + int(np.argmin(dist))
        return ref[min(len(ref) - 1, self.ref_idx + 2)]

    def _base_action_at_current(self):
        """Reference-tracking action from the current progress index (no index update)."""
        return np.clip((self.ref[min(len(self.ref) - 1, self.ref_idx + 2)] - self.q) / self.max_step, -1.0, 1.0)

    def _base_action(self):
        return np.clip((self._reference_target() - self.q) / self.max_step, -1.0, 1.0)

    def _apf(self, d):
        d0 = float(self.rcfg["apf_d0"])
        dd = np.maximum(d, 1e-3)
        return float(np.sum(np.where(dd < d0, 0.5 * (1.0 / dd - 1.0 / d0) ** 2, 0.0)))

    def _joint_distance(self, a, b):
        return float(np.linalg.norm(self.weights * (np.asarray(a) - np.asarray(b)) / self.span))

    def _goal_distance(self, q):
        return float(np.linalg.norm(self.weights * (self.target["goal_q"] - q) / self.span))

    def _tcp_error(self, q):
        from scipy.spatial.transform import Rotation
        T = np.asarray(self.backend.frame_world_T(self.robot, q, self.frame), dtype=float)
        G = self._goal_T
        return np.concatenate([G[:3, 3] - T[:3, 3], Rotation.from_matrix(G[:3, :3] @ T[:3, :3].T).as_rotvec()])

    def _obs(self, q, d, dirs, d_sp):
        qn = 2.0 * (q - self.lo) / self.span - 1.0
        return np.concatenate([
            qn, (self.target["goal_q"] - q) / self.span, self._tcp_error(q), d / self.cap, dirs.reshape(-1),
            [min(d_sp, self.cap) / self.cap, float(d_sp < self.approach_zone)], self.prev_action,
            [self.steps / self.max_steps]]
            + ([(self.ref[min(len(self.ref) - 1, self.ref_idx + 2)] - q) / self.span,
                [self.ref_idx / max(1, len(self.ref) - 1)]] if self.residual else [])).astype(np.float32)

    # -- gym API -----------------------------------------------------------------------------------------------
    def _pick_target(self):
        if self._scene_eps <= 0 or not self._scene_targets:
            key = self.targets[int(self.rng.integers(len(self.targets)))]
            self._scene_targets = [t for t in self.targets
                                   if t["snapshot"] == key["snapshot"] and t["phase"] == key["phase"]]
            self._scene_eps = self.episodes_per_scene
        self._scene_eps -= 1
        return self._scene_targets[int(self.rng.integers(len(self._scene_targets)))]

    def _start_q(self, target):
        q0 = target["start_q"].copy()
        if self.backend.check_collision(self.robot, q0).collision:
            # stored start from an older precompute without the pre-rotation retreat: retreat posture (track kept)
            from plugins.robotics.inspection_workflow import zero_non_linear_track_joints
            q0 = np.asarray(zero_non_linear_track_joints(list(q0), list(self.backend.joint_names(self.robot))), dtype=float)
        if self.rng.random() >= float(self.cfg["env"]["start_perturb_prob"]):
            return q0
        scale = float(self.cfg["env"]["start_perturb_scale"])
        for _ in range(20):
            q = np.clip(q0 + self.rng.normal(0.0, scale, self.n) * self.max_step / np.max(self.max_step), self.lo, self.hi)
            if not self.backend.check_collision(self.robot, q).collision:
                return q
        return q0

    def _curriculum_start(self, target, real_start):
        goal = target["goal_q"]
        lam = self.cur_level
        noise = float(self.cur_cfg.get("noise", 0.2))
        for _ in range(8):
            q = goal + lam * (real_start - goal) + self.rng.normal(0.0, noise, self.n) * self.max_step
            q = np.clip(q, self.lo, self.hi)
            if not self.backend.check_collision(self.robot, q).collision:
                return q
            lam *= 0.7
        return goal.copy()

    def _curriculum_update(self, success):
        if not self.cur_enabled:
            return
        self.cur_hist.append(bool(success))
        if len(self.cur_hist) < self.cur_hist.maxlen:
            return
        rate = sum(self.cur_hist) / len(self.cur_hist)
        c = self.cur_cfg
        if rate >= float(c.get("up_rate", 0.5)):
            self.cur_level = min(1.0, self.cur_level + float(c.get("step", 0.05)))
            self.cur_hist.clear()
        elif rate <= float(c.get("down_rate", 0.2)):
            self.cur_level = max(float(c.get("min_level", 0.02)), self.cur_level - float(c.get("step", 0.05)))
            self.cur_hist.clear()

    # -- random box obstacles ------------------------------------------------------------------------------------
    def _clear_boxes(self):
        if not self._boxes:
            return
        from plugins.robotics.pybullet_collision import remove_static_body
        sc = self.backend._handle(self.robot).pb_scene
        field = self.backend._handle(self.robot).sdf_field
        for body, sdf in self._boxes:
            if body in sc.static_bodies:
                i = sc.static_bodies.index(body)
                sc.static_bodies.pop(i)
                sc.static_names.pop(i)
                remove_static_body(body, sc.client)
            if field is not None and sdf in field.sdfs:
                i = field.sdfs.index(sdf)
                field.sdfs.pop(i)
                field.names.pop(i)
        self._boxes = []
        self._box_specs = {}

    def _add_box(self, center, size):
        import open3d as o3d
        from plugins.robotics.pybullet_collision import make_static_mesh_body
        from plugins.optimizer.sdf_field import StaticSDF
        mesh = o3d.geometry.TriangleMesh.create_box(*[float(s) for s in size])
        mesh.translate(np.asarray(center, dtype=float) - np.asarray(size, dtype=float) / 2.0)
        mesh.compute_vertex_normals()
        sc = self.backend._handle(self.robot).pb_scene
        field = self._sdf_links()
        body = make_static_mesh_body(mesh, concave=False)
        name = f"collision_object_{len(sc.static_bodies)}"
        sc.static_bodies.append(body)
        sc.static_names.append(name)
        sdf = StaticSDF(mesh, resolution=0.02, cap=field.cap)
        field.sdfs.append(sdf)
        field.names.append(name)
        self._boxes.append((body, sdf))
        self._box_specs[body] = {"center": np.asarray(center).tolist(), "size": np.asarray(size).tolist()}
        return body, sdf

    def _remove_box(self, entry):
        # _add_box returns a new tuple, so tuple identity cannot identify the stored entry.
        keep = [b for b in self._boxes if b[0] != entry[0]]
        specs = {body: spec for body, spec in self._box_specs.items() if body != entry[0]}
        self._boxes = [entry]
        self._clear_boxes()
        self._boxes = keep
        self._box_specs = specs

    def _link_origin(self, q, link_name):
        field = self._sdf_links()
        pin, model, data = field.pin, field.model, field.data
        pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
        pin.updateFramePlacements(model, data)
        return np.asarray(data.oMf[model.getFrameId(link_name)].translation, dtype=float)

    def _box_path_clearance(self, sdf, ref, stride=2):
        """Smallest distance between the robot (link surface samples) and one box along the reference (field lookup)."""
        field = self._sdf_links()
        pin, model, data = field.pin, field.model, field.data
        best = np.inf
        for q in ref[::stride]:
            pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
            pin.updateFramePlacements(model, data)
            R = np.stack([np.asarray(data.oMf[f].rotation) for f in field._fids])
            t = np.stack([np.asarray(data.oMf[f].translation) for f in field._fids])
            world = np.einsum("nij,nj->ni", R[field._link_of_point], field._pts) + t[field._link_of_point]
            best = min(best, float(np.min(sdf.distance(world))))
        return best

    def _place_random_boxes(self, start_q, goal_q):
        """1..max_count boxes near random links along the middle of the reference, not touching start or goal."""
        c = self.rand_cfg
        if not c.get("enabled") or self.rng.random() >= float(c.get("prob", 0.7)):
            return 0
        links = [l for l in self._links if re.search(r"link[1-5]$", l.replace("_0", ""))] or self._links
        n = int(self.rng.integers(1, int(c.get("max_count", 3)) + 1))
        lo_s, hi_s = c.get("size", [0.05, 0.2])
        lo_o, hi_o = c.get("offset", [0.08, 0.25])
        placed = 0
        for _ in range(n * int(c.get("attempts_per_box", 4))):
            if placed >= n:
                break
            ref = self.ref if self.residual else np.asarray([start_q, goal_q])
            i = int(self.rng.integers(int(0.2 * (len(ref) - 1)), max(int(0.2 * (len(ref) - 1)) + 1, int(0.8 * (len(ref) - 1)))))
            p = self._link_origin(ref[i], links[int(self.rng.integers(len(links)))])
            v = self.rng.normal(size=3)
            center = p + v / max(1e-9, np.linalg.norm(v)) * self.rng.uniform(lo_o, hi_o)
            entry = self._add_box(center, self.rng.uniform(lo_s, hi_s, size=3))
            if (self.backend.check_collision(self.robot, start_q).collision
                    or self.backend.check_collision(self.robot, goal_q).collision):
                self._remove_box(entry)
                continue
            if self.residual and "block_prob" in c:
                # mix of boxes that block the reference (the policy has to go around) and boxes the reference passes
                # within near_dist of (the policy has to keep clear): accept a blocking one with block_prob
                clear = self._box_path_clearance(entry[1], ref)
                blocking = clear < 0.01
                want_block = self.rng.random() < float(c["block_prob"])
                if blocking != want_block or (not blocking and clear > float(c.get("near_dist", 0.1))):
                    self._remove_box(entry)
                    continue
            placed += 1
        return placed

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        target = (options or {}).get("target") or self._pick_target()
        self._clear_boxes()
        self._activate(target)
        self._goal_T = np.asarray(self.backend.frame_world_T(self.robot, target["goal_q"], self.frame), dtype=float)
        start = (options or {}).get("start_q")
        if start is not None:
            self.q = np.asarray(start, dtype=float)
        else:
            self.q = self._start_q(target)
            if self.cur_enabled and self.cur_level < 1.0:
                self.q = self._curriculum_start(target, self.q)
        self.steps = 0
        self.prev_action = np.zeros(self.n)
        if self.residual:
            refs = target.get("references") or [target["reference"]]
            ref = refs[int(self.rng.integers(len(refs)))] if self.res_cfg.get("random_reference") else refs[0]
            self.ref = self._densify_reference(ref)
            self.ref_idx = 0
            if start is None:
                self._augment_start()
        self.n_boxes = self._place_random_boxes(self.q, target["goal_q"])
        self.path = [self.q.copy()]
        self.fixed_joint_values = self.q[self.fixed_joint_indices].copy()
        if self.residual and self.fixed_joint_indices and not np.allclose(
                self.ref[:, self.fixed_joint_indices], self.fixed_joint_values, atol=1e-6):
            raise ValueError("RRT reference moves a joint configured as fixed")
        self.d_goal = self._goal_distance(self.q)
        self.n_exact = self.n_fast = 0
        self.n_shield = 0
        d, dirs, d_sp = self._link_distances(self.q)
        self.dmin = min(float(np.min(d)), d_sp)
        # unavoidable proximity at the start / goal postures (reward.apf_baseline): excluded near those ends
        self.start_q_ep = self.q.copy()
        self.U_start = self._apf(d)
        self.U_goal = self._apf(self._link_distances(target["goal_q"])[0])
        self.terms = {k: 0.0 for k in ("progress", "apf", "margin", "side", "time", "smooth", "length",
                                        "curvature", "collision", "goal", "residual")}
        return self._obs(self.q, d, dirs, d_sp), {"target": target["group"], "phase": target["phase"],
                                               "n_boxes": self.n_boxes}

    def step(self, action):
        from plugins.robotics.collision_free_ik import robot_side_excess
        a_pol = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.residual:
            a = np.clip(self._base_action() + float(self.res_cfg.get("scale", 0.5)) * a_pol, -1.0, 1.0)
        else:
            a = a_pol
        q_next = np.clip(self.q + a * self.max_step, self.lo, self.hi)
        if self.fixed_joint_indices:
            q_next[self.fixed_joint_indices] = self.fixed_joint_values
        self.steps += 1
        rc = self.rcfg
        terms = {}
        info: Dict[str, Any] = {}
        terminated = False
        d, dirs, d_sp = self._link_distances(q_next)
        dmin_next = min(float(np.min(d)), d_sp)
        collides = self._edge_collides(self.q, q_next, self.dmin, dmin_next)
        if collides and self.residual and bool(self.res_cfg.get("shield", False)):
            # execution-time shield: a residual step that would collide is replaced by the reference-tracking step
            a = self._base_action_at_current()
            q_next = np.clip(self.q + a * self.max_step, self.lo, self.hi)
            if self.fixed_joint_indices:
                q_next[self.fixed_joint_indices] = self.fixed_joint_values
            d, dirs, d_sp = self._link_distances(q_next)
            dmin_next = min(float(np.min(d)), d_sp)
            collides = self._edge_collides(self.q, q_next, self.dmin, dmin_next)
            self.n_shield = getattr(self, "n_shield", 0) + 1
        if collides:
            terms["collision"] = -float(rc["collision"])
            terminated = True
            info["result"] = "collision"
            d, dirs, d_sp = self._link_distances(self.q)
        else:
            previous_q = self.q.copy()
            self.q = q_next
            self.dmin = dmin_next
            self.path.append(q_next.copy())
            step_delta = (q_next - previous_q) / self.span
            terms["length"] = -float(rc.get("length", 0.0)) * float(np.linalg.norm(step_delta))
            if len(self.path) >= 3:
                prev_delta = (previous_q - self.path[-3]) / self.span
                terms["curvature"] = -float(rc.get("curvature", 0.0)) * float(
                    np.dot(step_delta - prev_delta, step_delta - prev_delta))
            d_goal = self._goal_distance(q_next)
            terms["progress"] = float(rc["progress"]) * (self.d_goal - d_goal)
            self.d_goal = d_goal
            U = self._apf(d)
            base = rc.get("apf_baseline") or {}
            if base.get("enabled"):
                sigma = float(base.get("sigma", 0.15))
                w_s = max(0.0, 1.0 - self._joint_distance(q_next, self.start_q_ep) / sigma)
                w_g = max(0.0, 1.0 - d_goal / sigma)
                U = max(0.0, U - max(self.U_start * w_s, self.U_goal * w_g))
            terms["apf"] = -float(rc["apf"]) * min(U, float(rc["apf_clip"]))
            terms["margin"] = -float(rc["margin"]) * max(0.0, float(rc["safety_margin"]) - float(np.min(d)))
            terms["side"] = (-float(rc["side"]) * max(0.0, robot_side_excess(self.backend, self.robot, q_next))
                             if float(rc["side"]) else 0.0)
            if d_goal < self.reach_tol and not self.backend.check_edge_collision(
                    self.robot, q_next, self.target["goal_q"]).collision:
                final_delta = (self.target["goal_q"] - q_next) / self.span
                terms["length"] -= float(rc.get("length", 0.0)) * float(np.linalg.norm(final_delta))
                terms["curvature"] -= float(rc.get("curvature", 0.0)) * float(
                    np.dot(final_delta - step_delta, final_delta - step_delta))
                self.path.append(self.target["goal_q"].copy())
                terms["goal"] = float(rc["goal"])
                terminated = True
                info["result"] = "success"
        terms["time"] = -float(rc["time"])
        terms["smooth"] = -float(rc["smooth"]) * float(np.sum((a - self.prev_action) ** 2))
        if self.residual:
            terms["residual"] = -float(rc.get("residual", 0.0)) * float(np.sum(a_pol ** 2))
        self.prev_action = a
        for k, v in terms.items():
            self.terms[k] += v
        truncated = not terminated and self.steps >= self.max_steps
        if truncated:
            info["result"] = "timeout"
        if terminated or truncated:
            info["is_success"] = info["result"] == "success"   # SB3 logs rollout/success_rate from this
            info["curriculum_level"] = self.cur_level
            self._curriculum_update(info["is_success"])
            info.update({"q_path": [q.tolist() for q in self.path], "reward_terms": dict(self.terms),
                         "target": self.target["group"], "phase": self.target["phase"], "steps": self.steps,
                         "exact_checks": self.n_exact, "fast_checks": self.n_fast,
                         "shield_steps": getattr(self, "n_shield", 0), "n_boxes": getattr(self, "n_boxes", 0),
                         "snapshot": pathlib.Path(self.target["snapshot"]).name,
                         "obstacles": list(self._box_specs.values())})
        return self._obs(self.q, d, dirs, d_sp), float(sum(terms.values())), terminated, truncated, info
