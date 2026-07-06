import numpy as np
import os
import json
import logging
from typing import List, Union, Optional
import sys

# Adjust path to import PlannerBase
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.plannerbase import PlannerBase

class TaskSpaceMSBiRRTStar(PlannerBase):
    def __init__(self, config_path: str = None):
        super().__init__()
        if config_path is None:
            config_path = os.path.splitext(__file__)[0] + '.json'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
            
        self.base_step_size = self.config.get("base_step_size", 5.0)
        self.max_step_size = self.config.get("max_step_size", 30.0)
        self.max_iter = self.config.get("max_iter", 2000)
        self.goal_bias = self.config.get("goal_bias", 0.05)
        self.weights = self.config.get("weights", {"pos": 1.0, "orient": 0.5})
        self.dynamic_step_factor = self.config.get("dynamic_step_factor", 0.5)
        
        self.bounds = self.config.get("workspace_bounds", {
            "x_min": -500.0, "x_max": 500.0,
            "y_min": -1500.0, "y_max": 500.0,
            "z_min": -500.0, "z_max": 500.0,
            "roll_min": -3.14, "roll_max": 3.14,
            "pitch_min": -3.14, "pitch_max": 3.14,
            "yaw_min": -3.14, "yaw_max": 3.14
        })
        self.step_size = self.base_step_size
        self.configure_collision(self.config, default_sample_resolution=self.base_step_size)
            
    def _random_state(self):
        rnd_point = np.zeros(6)
        rnd_point[0] = np.random.uniform(self.bounds['x_min'], self.bounds['x_max'])
        rnd_point[1] = np.random.uniform(self.bounds['y_min'], self.bounds['y_max'])
        rnd_point[2] = np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
        rnd_point[3] = np.random.uniform(self.bounds['roll_min'], self.bounds['roll_max'])
        rnd_point[4] = np.random.uniform(self.bounds['pitch_min'], self.bounds['pitch_max'])
        rnd_point[5] = np.random.uniform(self.bounds['yaw_min'], self.bounds['yaw_max'])
        return rnd_point

    def _nearest_neighbor(self, tree_nodes, q_rand):
        nodes_arr = np.array(tree_nodes)
        diffs = nodes_arr - q_rand
        pos_diff = diffs[:, :3]
        orient_diff = diffs[:, 3:]
        w_pos = self.weights['pos']
        w_ori = self.weights['orient']
        weighted_sq_dists = w_pos * np.sum(pos_diff**2, axis=1) + w_ori * np.sum(orient_diff**2, axis=1)
        nearest_idx = np.argmin(weighted_sq_dists)
        return nearest_idx

    def _get_dynamic_step_size(self, pos):
        return self.base_step_size

    def _steer(self, q_near, q_rand):
        direction = q_rand - q_near
        w_pos = self.weights['pos']
        w_ori = self.weights['orient']
        dist = np.sqrt(w_pos * np.sum(direction[:3]**2) + w_ori * np.sum(direction[3:]**2))
        
        if dist == 0:
            return q_near
        
        # Calculate dynamic step size based on q_near (local density)
        current_step_size = self._get_dynamic_step_size(q_near[:3])
            
        ratio = min(1.0, current_step_size / dist)
        return q_near + direction * ratio

    def generate(self, start, goal, step_callback=None):
        start = np.array(start)
        goal = np.array(goal)
        
        # Tree A (Start) and Tree B (Goal)
        self.tree_a = [start]
        self.parents_a = {0: None}
        self.tree_b = [goal]
        self.parents_b = {0: None}
        
        print(f"MS-Bi-RRT* Start. Base Step: {self.base_step_size}")
        
        for i in range(self.max_iter):
            # Swap trees for bidirectional growth
            if len(self.tree_a) > len(self.tree_b):
                self.tree_a, self.tree_b = self.tree_b, self.tree_a
                self.parents_a, self.parents_b = self.parents_b, self.parents_a
                
            # 1. Sample
            q_rand = self._random_state()
            
            # 2. Extend Tree A
            idx_near_a = self._nearest_neighbor(self.tree_a, q_rand)
            q_near_a = self.tree_a[idx_near_a]
            q_new_a = self._steer(q_near_a, q_rand)
            
            if not self._check_collision(q_near_a, q_new_a):
                idx_new_a = len(self.tree_a)
                self.tree_a.append(q_new_a)
                self.parents_a[idx_new_a] = idx_near_a
                
                # 3. Connect Tree B to q_new_a
                idx_near_b = self._nearest_neighbor(self.tree_b, q_new_a)
                q_near_b = self.tree_b[idx_near_b]
                
                # Steer towards q_new_a, trying to reach it exactly
                curr_node_b = q_near_b
                connected = False
                prev_idx_b = idx_near_b
                
                # Greedy connect loop (RRT-Connect style)
                while True:
                    q_new_b = self._steer(curr_node_b, q_new_a)
                    if self._check_collision(curr_node_b, q_new_b):
                        break # Collision, stop extending
                        
                    idx_new_b = len(self.tree_b)
                    self.tree_b.append(q_new_b)
                    self.parents_b[idx_new_b] = prev_idx_b
                    prev_idx_b = idx_new_b
                    curr_node_b = q_new_b
                    
                    # Check connection
                    dist_to_connect = np.linalg.norm(q_new_b[:3] - q_new_a[:3])
                    if dist_to_connect < 1.0: # Connected
                        connected = True
                        break
                        
                    # Also stop if we reached q_new_a (approx equal)
                    if np.allclose(q_new_b, q_new_a, atol=1e-3):
                        connected = True
                        break
                        
                if step_callback:
                    # Callback expects (tree_a, parents_a, tree_b, parents_b) for 4 args if defined in planner_main?
                    # Or generic (nodes, parents). planner_main handles 2 or 4 args.
                    # Let's verify planner_main support.
                    # It supports 4 args: t1, p1, t2, p2.
                    # We need to make sure we pass them in consistent order.
                    # Since we swap, we might confusing visualization colors. 
                    # Ideally pass (tree_start, parents_start, tree_goal, parents_goal) consistently.
                    # We can track which tree is which using a flag or just pass A/B and let colors swap (it looks dynamic and cool).
                    step_callback(self.tree_a, self.parents_a, self.tree_b, self.parents_b)

                if connected:
                    # Construct Path
                    # q_new_a is the bridge node in Tree A
                    # q_new_b is the bridge node in Tree B (which is very close to q_new_a)
                    
                    # If we swapped, we need to know which is Start tree.
                    # Let's trace back from connection points.
                    
                    path_a = []
                    curr = idx_new_a
                    while curr is not None:
                        path_a.append(self.tree_a[curr])
                        curr = self.parents_a[curr]
                    path_a = path_a[::-1] # Root to Connect
                    
                    path_b = []
                    curr = idx_new_b # Connect node in B
                    while curr is not None:
                        path_b.append(self.tree_b[curr])
                        curr = self.parents_b[curr]
                    # Root of B to Connect.
                    
                    # Determine which is start
                    # Standard RRT-Connect: Start Tree Root is Start Pose.
                    # We check distance to self.start
                    dist_a_start = np.linalg.norm(path_a[0][:3] - np.array(start)[:3])
                    dist_b_start = np.linalg.norm(path_b[0][:3] - np.array(start)[:3])
                    
                    if dist_a_start < dist_b_start:
                        # A is Start, B is Goal
                        # Path = PathA + Reverse(PathB)
                        full_path = path_a + path_b[::-1]
                    else:
                        # B is Start, A is Goal
                        # Path = PathB + Reverse(PathA)
                        full_path = path_b + path_a[::-1]
                        
                    return full_path

        print("MS-Bi-RRT* Max Iterations Reached")
        return None
