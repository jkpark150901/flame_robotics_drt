"""Random shortcut smoothing from pybullet_planning (smooth_path, Geraerts & Overmars 2007) as an optimizer plugin.

Takes any planner's path and repeatedly replaces random sub-paths by straight segments when that is shorter and
collision free (PlannerBase.check_robot_collision = the project's PyBullet engine). Cost = normalized joint-space length.
"""
import json
import os

import numpy as np

from plugins.pathplanner.pp_common import load_motion_planners
from plugins.pluginbase.optimizerbase import OptimizerBase


class PPShortcut(OptimizerBase):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)
        config_path = config_path or os.path.splitext(__file__)[0] + ".json"
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.iterations = int(self.config.get("iterations", 200))
        self.max_time = float(self.config.get("max_time", 5.0))
        self.step_size = float(self.config.get("step_size", 0.05))
        self.seed = self.config.get("seed", 0)

    def optimize(self, path: list, planner) -> list:
        if not path or len(path) < 3:
            return path
        mp = load_motion_planners()
        if self.seed is not None:
            np.random.seed(int(self.seed))
        step = self.step_size

        def extend_fn(a, b):
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            n = max(1, int(np.ceil(float(np.linalg.norm(b - a)) / step)))
            return [a + (b - a) * (k / n) for k in range(1, n + 1)]

        smoothed = mp.smooth_path(
            [np.asarray(q, dtype=float) for q in path], extend_fn,
            lambda q: bool(planner.check_robot_collision(np.asarray(q, dtype=float))),
            distance_fn=lambda a, b: planner._joint_distance(np.asarray(a, float), np.asarray(b, float)),
            max_smooth_iterations=self.iterations, max_time=self.max_time, coarse_waypoints=True)
        return [np.asarray(q, dtype=float) for q in smoothed] if smoothed else path
