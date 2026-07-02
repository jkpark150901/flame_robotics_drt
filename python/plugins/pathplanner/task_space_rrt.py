import numpy as np
import json
import os
from typing import List, Union, Optional
import sys
import logging

# Adjust path to import PlannerBase
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.plannerbase import PlannerBase

class TaskSpaceRRT(PlannerBase):
    def __init__(self, config_path: str = None):
        super().__init__()
        if config_path is None:
            config_path = os.path.splitext(__file__)[0] + '.json'
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.step_size = self.config.get("step_size", 1.0)
        self.max_iter = self.config.get("max_iter", 1000)
        self.goal_bias = self.config.get("goal_bias", 0.1)
        self.weights = self.config.get("weights", {"pos": 1.0, "orient": 0.5})
        self.bounds = self.config.get("workspace_bounds", {
            "x_min": -10.0, "x_max": 10.0,
            "y_min": -10.0, "y_max": 10.0,
            "z_min": -10.0, "z_max": 10.0,
            "roll_min": -np.pi, "roll_max": np.pi,
            "pitch_min": -np.pi, "pitch_max": np.pi,
            "yaw_min": -np.pi, "yaw_max": np.pi
        })
        self.configure_collision(self.config, default_sample_resolution=self.step_size)

    def generate(self, current_pose: Union[List[float], np.ndarray], target_pose: Union[List[float], np.ndarray], step_callback: Optional[callable] = None) -> List[np.ndarray]:
        current_pose = np.array(current_pose)
        target_pose = np.array(target_pose)
        
        # Handle Don't Care (NaN) in target
        mask = np.isnan(target_pose)
        if np.any(mask):
            target_pose[mask] = current_pose[mask]
            
        nodes = [current_pose]
        parents = {0: None}
        
        w_pos = self.weights['pos']
        w_ori = self.weights['orient']
        
        min_dist_to_goal = float('inf')
        
        for i in range(self.max_iter):
            logging.info(f"Iteration {i+1}/{self.max_iter} | Tree Size: {len(nodes)} | Min Dist: {min_dist_to_goal:.2f}")
            # 1. Sample
            if np.random.random() < self.goal_bias:
                rnd_point = target_pose
            else:
                rnd_point = np.zeros(6)
                rnd_point[0] = np.random.uniform(self.bounds['x_min'], self.bounds['x_max'])
                rnd_point[1] = np.random.uniform(self.bounds['y_min'], self.bounds['y_max'])
                rnd_point[2] = np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
                rnd_point[3] = np.random.uniform(self.bounds['roll_min'], self.bounds['roll_max'])
                rnd_point[4] = np.random.uniform(self.bounds['pitch_min'], self.bounds['pitch_max'])
                rnd_point[5] = np.random.uniform(self.bounds['yaw_min'], self.bounds['yaw_max'])
                
            # 2. Nearest
            diffs = np.array(nodes) - rnd_point
            pos_diff = diffs[:, :3]
            orient_diff = diffs[:, 3:]
            
            # Weighted Distance
            weighted_sq_dists = w_pos * np.sum(pos_diff**2, axis=1) + w_ori * np.sum(orient_diff**2, axis=1)
            nearest_idx = np.argmin(weighted_sq_dists)
            nearest_node = nodes[nearest_idx]
            
            # 3. Steer
            direction = rnd_point - nearest_node
            dist = np.sqrt(w_pos * np.sum(direction[:3]**2) + w_ori * np.sum(direction[3:]**2))
            
            if dist == 0:
                continue
                
            # Cap at step_size
            ratio = min(1.0, self.step_size / dist)
            new_point = nearest_node + direction * ratio
            
            # 4. Collision Check
            if not self._check_collision(nearest_node, new_point):
                nodes.append(new_point)
                new_idx = len(nodes) - 1
                parents[new_idx] = nearest_idx
                
                # Check Callback
                if step_callback:
                    step_callback(nodes, parents)
                
                # Update min distance to goal
                delta = target_pose - new_point
                d_goal_current = np.sqrt(w_pos * np.sum(delta[:3]**2) + w_ori * np.sum(delta[3:]**2))
                if d_goal_current < min_dist_to_goal:
                    min_dist_to_goal = d_goal_current
                
                # 5. Check Goal
                if d_goal_current < self.step_size: # Close enough
                    if not self._check_collision(new_point, target_pose):
                        nodes.append(target_pose)
                        goal_idx = len(nodes) - 1
                        parents[goal_idx] = new_idx
                        
                        # Reconstruct
                        path = []
                        curr = goal_idx
                        while curr is not None:
                            path.append(nodes[curr])
                            curr = parents[curr]
                        return path[::-1]
                        
        logging.error(f"Task Space RRT failed to find path. Max iterations ({self.max_iter}) reached.")
        logging.error(f"Closest distance to goal achieved: {min_dist_to_goal:.4f}")
        return [] 
