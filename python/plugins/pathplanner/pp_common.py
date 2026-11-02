"""Adapter that runs the sampling-based planners of yijiangh/pybullet_planning (MIT, vendored in 3rdparty/pybullet_planning,
derived from Caelan Garrett's motion-planners) on this project's robot q-space and PyBullet collision engine.

The library's planners are generic: they only need distance / sample / extend / collision callables. Those come from
PlannerBase here, so every planner uses the same joint-space metric (normalized by joint ranges), the same joint limits and
the same PyBullet mesh collision check (PlannerBase.check_robot_collision -> backend.check_collision) as the rest of the
project - differences in a benchmark are then only the algorithms.

Planner plugins (plugins/pathplanner/pp_*.py) just pick ALGORITHM; their .json holds the algorithm parameters.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time
from typing import List

import numpy as np

from plugins.pluginbase.plannerbase import PlannerBase

_PP_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "3rdparty", "pybullet_planning", "src"))


def load_motion_planners():
    """pybullet_planning.motion_planners from the vendored checkout (imported lazily; needs only numpy + pybullet)."""
    if not os.path.isdir(_PP_SRC):
        raise RuntimeError(f"pybullet_planning is not checked out: expected {_PP_SRC} "
                           "(git clone https://github.com/yijiangh/pybullet_planning 3rdparty/pybullet_planning)")
    if _PP_SRC not in sys.path:
        sys.path.insert(0, _PP_SRC)
    import pybullet_planning.motion_planners as mp
    return mp


class PPMotionPlannerBase(PlannerBase):
    use_joint_space_planning = True
    ALGORITHM = "birrt"
    LABEL = "pybullet_planning"

    def __init__(self, config_path: str = None):
        super().__init__()
        if config_path is None:
            config_path = os.path.splitext(inspect.getfile(type(self)))[0] + ".json"
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.step_size = float(self.config.get("step_size", 0.1))
        self.max_iter = int(self.config.get("max_iter", 5000))
        self.configure_fixed_joints(
            fixed_joints=self.config.get("fixed_joints"),
            fixed_joint_indices=self.config.get("fixed_joint_indices"),
            fixed_joint_values=self.config.get("fixed_joint_values"),
        )
        self.configure_collision(self.config, default_sample_resolution=self.step_size)

    # -- callables handed to the library --------------------------------------------------------------------------
    def _extend_fn(self, a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = max(1, int(np.ceil(float(np.linalg.norm(b - a)) / max(self.step_size, 1e-9))))
        return [a + (b - a) * (k / n) for k in range(1, n + 1)]

    def _remaining_time(self):
        deadline = getattr(self, "planning_deadline", None)
        if deadline is None:
            return float("inf")
        return max(0.05, 0.95 * (deadline - time.monotonic()))

    def generate(self, current_pose, target_pose, step_callback=None) -> List[np.ndarray]:
        start = np.array(current_pose, dtype=float)
        goal = np.array(target_pose, dtype=float)
        dof = self._robot_dof()
        if not (self._has_robot_q_space_model() and start.shape[0] == dof and goal.shape[0] == dof):
            raise ValueError(f"{self.LABEL} plans in robot q-space only (dof={dof}); got {start.shape[0]}->{goal.shape[0]}")
        start, goal = self._prepare_fixed_joint_constraints(start, goal)
        self.last_planning_status = None
        self.last_returned_path_reaches_goal = True
        if self.check_robot_collision(start):
            print(f"{self.LABEL}[{self.ALGORITHM}] failed: start configuration is in collision")
            return []
        if self.check_robot_collision(goal):
            print(f"{self.LABEL}[{self.ALGORITHM}] failed: goal configuration is in collision")
            return []

        mp = load_motion_planners()
        distance_fn = lambda a, b: self._joint_distance(np.asarray(a, float), np.asarray(b, float))
        sample_fn = lambda: np.asarray(self._sample_robot_configuration(), dtype=float)
        collision_fn = lambda q: bool(self.check_robot_collision(np.asarray(q, dtype=float)))
        extend_fn = self._extend_fn
        common = dict(max_time=self._remaining_time())
        cfg, alg = self.config, self.ALGORITHM

        # a free straight line is the answer for every algorithm (the library's solve_motion_plan does the same)
        direct = mp.check_direct(start, goal, extend_fn, collision_fn)
        if direct is not None:
            return [start] + [np.asarray(q, dtype=float) for q in direct]

        if alg == "birrt":
            path = mp.birrt(start, goal, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=self.max_iter,
                            restarts=int(cfg.get("restarts", 2)), **common)
        elif alg == "rrt":
            path = mp.rrt(start, goal, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=self.max_iter,
                          goal_probability=float(cfg.get("goal_probability", 0.2)), **common)
        elif alg == "rrt_star":
            path = mp.rrt_star(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                               radius=float(cfg.get("radius", 2.0)), max_iterations=self.max_iter,
                               informed=bool(cfg.get("informed", True)), **common)
        elif alg == "prm":
            path = mp.prm(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                          target_degree=int(cfg.get("target_degree", 4)), num_samples=int(cfg.get("num_samples", 300)))
        elif alg == "lazy_prm":
            result = mp.lazy_prm(start, goal, sample_fn, extend_fn, collision_fn,
                                 num_samples=int(cfg.get("num_samples", 300)), **common)
            path = result[0] if isinstance(result, tuple) else result
        else:
            raise NotImplementedError(alg)
        if not path:
            print(f"{self.LABEL}[{alg}] failed: no path found")
            self.last_planning_status = "no_path"
            return []
        path = [np.asarray(q, dtype=float) for q in path]
        if not np.allclose(path[0], start):
            path.insert(0, start)
        return path
