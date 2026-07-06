import numpy as np
import json
import os
from typing import List, Union, Optional
import sys
import logging

# Adjust path to import PlannerBase
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.plannerbase import PlannerBase

class TaskSpaceRRTConnect(PlannerBase):
    def __init__(self, config_path: str = None):
        super().__init__()
        if config_path is None:
            config_path = os.path.splitext(__file__)[0] + '.json'
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.step_size = self.config.get("step_size", 1.0)
        self.max_iter = self.config.get("max_iter", 1000)
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
        current_pose = np.array(current_pose, dtype=float)
        target_pose = np.array(target_pose, dtype=float)
        
        # Determine strict goal from target_pose (handle NaNs for Don't Care)
        # For RRT-Connect, dual tree needs explicit start and goal.
        # If Goal has NaNs, we can't easily grow a tree from it unless we pick a concrete goal.
        # Strategy: Sample a concrete goal pose consistent with target_pose (fill NaNs with random or heuristics).
        # Since this is a planner, let's substitute NaNs with current_pose values or 0?
        # Better: Sample one concrete goal and try to connect. 
        # Or better: Just use current_pose values for NaNs (maintain orientation etc.)
        
        concrete_goal = np.copy(target_pose)
        mask_goal = np.isnan(concrete_goal)
        # Using current pose orientation for goal if unspecified seems safe for "Mainly position" task
        concrete_goal[mask_goal] = current_pose[mask_goal]
        
        # Tree A (Start), Tree B (Goal)
        tree_a = [current_pose]
        parents_a = {0: None}
        
        tree_b = [concrete_goal]
        parents_b = {0: None}
        
        path_found = False
        connect_node_a_idx = -1
        connect_node_b_idx = -1
        
        w_pos = self.weights['pos']
        w_ori = self.weights['orient']
        
        min_dist_between_trees = float('inf')
        
        for i in range(self.max_iter):
            logging.info(f"Iteration {i+1}/{self.max_iter} | Tree A: {len(tree_a)} | Tree B: {len(tree_b)} | Min Gap: {min_dist_between_trees:.2f}")
            # Sample
            if np.random.random() < 0.1: # Small bias just in case (though Connect is greedy)
                rnd_point = np.copy(concrete_goal) # Or swap
            else:
                rnd_point = np.zeros(6)
                rnd_point[0] = np.random.uniform(self.bounds['x_min'], self.bounds['x_max'])
                rnd_point[1] = np.random.uniform(self.bounds['y_min'], self.bounds['y_max'])
                rnd_point[2] = np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
                rnd_point[3] = np.random.uniform(self.bounds['roll_min'], self.bounds['roll_max'])
                rnd_point[4] = np.random.uniform(self.bounds['pitch_min'], self.bounds['pitch_max'])
                rnd_point[5] = np.random.uniform(self.bounds['yaw_min'], self.bounds['yaw_max'])
                
            # Extend A
            new_idx_a = self._extend(tree_a, parents_a, rnd_point, w_pos, w_ori)
            
            if new_idx_a is not None:
                new_node_a = tree_a[new_idx_a]
                
                # Check Gap
                diffs_b = np.array(tree_b) - new_node_a
                dists_b = w_pos * np.sum(diffs_b[:, :3]**2, axis=1) + w_ori * np.sum(diffs_b[:, 3:]**2, axis=1)
                min_gap = np.sqrt(np.min(dists_b))
                if min_gap < min_dist_between_trees:
                    min_dist_between_trees = min_gap
                
                # Connect B to new_node_a
                new_idx_b = self._connect(tree_b, parents_b, new_node_a, w_pos, w_ori)
                
                if new_idx_b is not None:
                    connect_node_a_idx = new_idx_a
                    connect_node_b_idx = new_idx_b
                    path_found = True
                    break
            
            # Callback
            if step_callback:
                step_callback(tree_a, parents_a, tree_b, parents_b)
                
            # Swap
            tree_a, tree_b = tree_b, tree_a
            parents_a, parents_b = parents_b, parents_a
            
        if path_found:
            # Check which tree is start
            root_a = tree_a[0]
            if np.allclose(root_a, current_pose):
                path_start = self._trace_path(tree_a, parents_a, connect_node_a_idx)[::-1]
                path_goal = self._trace_path(tree_b, parents_b, connect_node_b_idx)
                return path_start + path_goal
            else:
                path_start = self._trace_path(tree_b, parents_b, connect_node_b_idx)[::-1]
                path_goal = self._trace_path(tree_a, parents_a, connect_node_a_idx)
                return path_start + path_goal
                
        logging.error(f"Task Space RRT-Connect failed. Max iterations ({self.max_iter}) reached.")
        logging.error(f"Smallest gap between trees: {min_dist_between_trees:.4f}")
        return []

    def _extend(self, nodes, parents, target, w_pos, w_ori):
        diffs = np.array(nodes) - target
        pos_diff = diffs[:, :3]
        ori_diff = diffs[:, 3:]
        dists = w_pos * np.sum(pos_diff**2, axis=1) + w_ori * np.sum(ori_diff**2, axis=1)
        nearest_idx = np.argmin(dists)
        nearest_node = nodes[nearest_idx]
        
        direction = target - nearest_node
        length = np.sqrt(w_pos * np.sum(direction[:3]**2) + w_ori * np.sum(direction[3:]**2))
        
        if length == 0: return None
        
        ratio = min(1.0, self.step_size / length)
        new_point = nearest_node + direction * ratio
        
        if not self._check_collision(nearest_node, new_point):
            nodes.append(new_point)
            new_idx = len(nodes) - 1
            parents[new_idx] = nearest_idx
            return new_idx
        return None

    def _connect(self, nodes, parents, target, w_pos, w_ori):
        # Greedy connect: repeatedly extend
        curr_idx = -1
        # First, find nearest
        diffs = np.array(nodes) - target
        dists = w_pos * np.sum(diffs[:, :3]**2, axis=1) + w_ori * np.sum(diffs[:, 3:]**2, axis=1)
        nearest_idx = np.argmin(dists)
        
        curr_node = nodes[nearest_idx]
        curr_idx_in_tree = nearest_idx
        
        while True:
            direction = target - curr_node
            length = np.sqrt(w_pos * np.sum(direction[:3]**2) + w_ori * np.sum(direction[3:]**2))
            
            if length < self.step_size:
                # Reachable
                if not self._check_collision(curr_node, target):
                    nodes.append(target)
                    new_idx = len(nodes) - 1
                    parents[new_idx] = curr_idx_in_tree
                    return new_idx
                return None
                
            # Step
            ratio = self.step_size / length
            new_point = curr_node + direction * ratio
            
            if not self._check_collision(curr_node, new_point):
                nodes.append(new_point)
                new_idx = len(nodes) - 1
                parents[new_idx] = curr_idx_in_tree
                
                curr_node = new_point
                curr_idx_in_tree = new_idx
            else:
                return None # Blocked

    def _trace_path(self, nodes, parents, idx):
        path = []
        while idx is not None:
            path.append(nodes[idx])
            idx = parents[idx]
        return path
