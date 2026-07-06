import numpy as np
import json
import os
from typing import List, Union
import sys

# Adjust path to import PlannerBase if not installed as package
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.plannerbase import PlannerBase

class RRT(PlannerBase):
    def __init__(self, config_path: str = None):
        super().__init__()
        if config_path is None:
            # Default to json with same name
            config_path = os.path.splitext(__file__)[0] + '.json'
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.step_size = self.config.get("step_size", 1.0)
        self.max_iter = self.config.get("max_iter", 1000)
        self.goal_bias = self.config.get("goal_bias", 0.1)
        self.bounds = self.config.get("workspace_bounds", {
            "x_min": -10.0, "x_max": 10.0,
            "y_min": -10.0, "y_max": 10.0,
            "z_min": -10.0, "z_max": 10.0
        })
        self.configure_collision(self.config, default_sample_resolution=self.step_size)

    def generate(self, current_pose: Union[List[float], np.ndarray], target_pose: Union[List[float], np.ndarray]) -> List[np.ndarray]:
        current_pose = np.array(current_pose, dtype=float)
        target_pose = np.array(target_pose, dtype=float)
        
        # Extract position (first 3 elements) for RRT growth
        start_pos = current_pose[:3]
        goal_pos = target_pose[:3]
        
        nodes = [start_pos]
        parents = {0: None} # child_index: parent_index
        
        path_found = False
        goal_node_idx = -1
        
        for i in range(self.max_iter):
            # Sample random point (with goal bias)
            if np.random.random() < self.goal_bias:
                rnd_point = goal_pos
            else:
                rnd_point = np.array([
                    np.random.uniform(self.bounds['x_min'], self.bounds['x_max']),
                    np.random.uniform(self.bounds['y_min'], self.bounds['y_max']),
                    np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
                ])
                
            # Find nearest node
            dists = np.linalg.norm(np.array(nodes) - rnd_point, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_node = nodes[nearest_idx]
            
            # Steer
            direction = rnd_point - nearest_node
            length = np.linalg.norm(direction)
            if length == 0:
                continue
                
            direction = direction / length
            new_point = nearest_node + direction * min(self.step_size, length)
            
            # Collision Check
            if not self._check_collision(nearest_node, new_point):
                # Add node
                nodes.append(new_point)
                new_idx = len(nodes) - 1
                parents[new_idx] = nearest_idx
                
                # Check if close to goal
                if np.linalg.norm(new_point - goal_pos) < self.step_size:
                    # Try to connect directly to goal
                    if not self._check_collision(new_point, goal_pos):
                        nodes.append(goal_pos)
                        goal_node_idx = len(nodes) - 1
                        parents[goal_node_idx] = new_idx
                        path_found = True
                        break
                        
        if path_found:
            # Reconstruct path
            path = []
            curr_idx = goal_node_idx
            while curr_idx is not None:
                pose = np.copy(current_pose) # Copy structure
                pose[:3] = nodes[curr_idx] # Update position
                # For orientation, we can interpolate or just keep start/goal depending on req.
                # Here simply copying start orientation for intermediate nodes, 
                # and setting goal orientation for the last one is simplest, 
                # but user asked for waypoints. 
                # Let's interpolate orientation or just assign dont-care for now?
                # User said: "return list of waypoints... data is 3D coord and orientation"
                # For simplicity in basic RRT, we just linearly interpolate or set. 
                # For now, let's just use the target orientation for the final node 
                # and interpolated or start orientation for others.
                # Actually, standard RRT is only position. 
                # Let's just use start orientation for all except last?
                if curr_idx == goal_node_idx:
                    # Handle NaN in target_pose for Don't Care
                    final_orient = target_pose[3:]
                    current_orient = current_pose[3:]
                    # If NaN, use current orientation or keep it 0? 
                    # User says NaN means don't care.
                    # We will output a concrete value.
                    pose[3:] = np.where(np.isnan(final_orient), current_orient, final_orient)
                else:
                    pose[3:] = current_pose[3:]
                
                path.append(pose)
                curr_idx = parents[curr_idx]
            return path[::-1] # Reverse
        else:
            print("Path not found within max_iter")
            return []
