import json
import os

import numpy as np

from plugins.pluginbase.plannerbase import PlannerBase


class RRT(PlannerBase):
    """Workspace RRT planner."""

    def __init__(self, config_path=None):
        super().__init__()
        if config_path is None:
            config_path = os.path.splitext(__file__)[0] + ".json"
        with open(config_path, "r") as file:
            self.config = json.load(file)
        self.step_size = float(self.config.get("step_size", 1.0))
        self.max_iter = int(self.config.get("max_iter", 1000))
        self.goal_bias = float(self.config.get("goal_bias", 0.05))
        self.normalize_joint_space = True
        self.debug_convergence = bool(self.config.get("debug_convergence", True))
        self.debug_output_dir = self.config.get("debug_output_dir", "debug/rrt")
        self.bounds = self.config.get("workspace_bounds", {})
        self.configure_fixed_joints(
            fixed_joints=self.config.get("fixed_joints"),
            fixed_joint_indices=self.config.get("fixed_joint_indices"),
            fixed_joint_values=self.config.get("fixed_joint_values"),
        )
        self.configure_collision(self.config, default_sample_resolution=self.step_size)

    def _generate_workspace(self, current_pose, target_pose, step_callback=None):
        current_pose = np.asarray(current_pose, dtype=float)
        target_pose = np.asarray(target_pose, dtype=float)
        start_pos = current_pose[:3]
        goal_pos = target_pose[:3]
        nodes = [start_pos]
        parents = {0: None}
        goal_node_idx = None

        for _ in range(self.max_iter):
            if np.random.random() < self.goal_bias:
                sample = goal_pos
            else:
                sample = np.asarray([
                    np.random.uniform(self.bounds["x_min"], self.bounds["x_max"]),
                    np.random.uniform(self.bounds["y_min"], self.bounds["y_max"]),
                    np.random.uniform(self.bounds["z_min"], self.bounds["z_max"]),
                ])
            distances = np.linalg.norm(np.asarray(nodes) - sample, axis=1)
            nearest_idx = int(np.argmin(distances))
            nearest = nodes[nearest_idx]
            direction = sample - nearest
            length = float(np.linalg.norm(direction))
            if length == 0.0:
                continue
            new_point = nearest + direction / length * min(self.step_size, length)
            if self._check_collision(nearest, new_point):
                continue
            nodes.append(new_point)
            new_idx = len(nodes) - 1
            parents[new_idx] = nearest_idx
            if step_callback is not None:
                step_callback(nodes, parents)
            if np.linalg.norm(new_point - goal_pos) < self.step_size:
                if not self._check_collision(new_point, goal_pos):
                    nodes.append(goal_pos)
                    goal_node_idx = len(nodes) - 1
                    parents[goal_node_idx] = new_idx
                    break

        if goal_node_idx is None:
            print("Path not found within max_iter")
            return []
        path = []
        current_idx = goal_node_idx
        while current_idx is not None:
            pose = current_pose.copy()
            pose[:3] = nodes[current_idx]
            if current_idx == goal_node_idx:
                pose[3:] = np.where(
                    np.isnan(target_pose[3:]), current_pose[3:], target_pose[3:]
                )
            else:
                pose[3:] = current_pose[3:]
            path.append(pose)
            current_idx = parents[current_idx]
        return path[::-1]
