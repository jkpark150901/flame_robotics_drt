import numpy as np
import json
import os
import sys
import networkx as nx
from scipy.spatial import KDTree
from typing import List, Union

# Adjust path to import PlannerBase
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.plannerbase import PlannerBase

class PRM(PlannerBase):
    def __init__(self, config_path: str = None):
        super().__init__()
        if config_path is None:
            config_path = os.path.splitext(__file__)[0] + '.json'
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.num_samples = self.config.get("num_samples", 500)
        self.k_neighbors = self.config.get("k_neighbors", 10)
        self.step_size = self.config.get("step_size", 2.0)
        self.bounds = self.config.get("workspace_bounds", {
            "x_min": -50.0, "x_max": 50.0,
            "y_min": -50.0, "y_max": 50.0,
            "z_min": -50.0, "z_max": 50.0
        })
        self.graph = nx.Graph()
        self.samples = []
        self.configure_collision(self.config, default_sample_resolution=self.step_size)
        
    def _is_valid(self, p):
        # Point validity is intentionally left to the shared edge collision check.
        return True

    def generate(self, current_pose: Union[List[float], np.ndarray], target_pose: Union[List[float], np.ndarray]) -> List[np.ndarray]:
        current_pose = np.array(current_pose, dtype=float)
        target_pose = np.array(target_pose, dtype=float)
        
        start_pos = current_pose[:3]
        goal_pos = target_pose[:3]
        
        self.graph = nx.Graph()
        self.samples = [start_pos, goal_pos]
        
        # 1. Sampling Phase
        print(f"[PRM] Sampling {self.num_samples} points...")
        for _ in range(self.num_samples):
            # Sample Position
            rnd_point = np.array([
                np.random.uniform(self.bounds['x_min'], self.bounds['x_max']),
                np.random.uniform(self.bounds['y_min'], self.bounds['y_max']),
                np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
            ])
            
            if self._is_valid(rnd_point):
                self.samples.append(rnd_point)
                
        # 2. Connection Phase (Roadmap Construction)
        print(f"[PRM] Building roadmap with {len(self.samples)} nodes...")
        
        # Use KDTree for efficient Nearest Neighbor search
        tree = KDTree(self.samples)
        
        # Query k+1 neighbors (point itself is included)
        k = min(self.k_neighbors + 1, len(self.samples))
        dists, indices = tree.query(self.samples, k=k)
        
        for i, neighbors in enumerate(indices):
            p1 = self.samples[i]
            for j in neighbors[1:]: # Skip self (index 0)
                p2 = self.samples[j]
                
                # Distance check (already done by KDTree implicitly/explicitly, but if we want max connection dist?)
                # We trust KDTree k-neighbors.
                
                # Edge Collision Check
                if not self._check_collision(p1, p2):
                     distance = np.linalg.norm(p1 - p2)
                     self.graph.add_edge(i, j, weight=distance)
                     
        # 3. Query Phase
        print("[PRM] Searching for path...")
        start_idx = 0
        goal_idx = 1
        
        try:
            path_indices = nx.shortest_path(self.graph, source=start_idx, target=goal_idx, weight='weight')
            
            # Reconstruct Path
            path = []
            for idx in path_indices:
                pos = self.samples[idx]
                
                # Orientation Handling
                # Interpolate or stick to target/start?
                # Simple: Interpolate RPY from start to goal based on progress?
                # Or just keep start orientation?
                # Let's use target orientation for goal, start for others.
                full_pose = np.concatenate((pos, current_pose[3:])) # Default to start orient
                if idx == goal_idx:
                     full_pose[3:] = np.where(np.isnan(target_pose[3:]), current_pose[3:], target_pose[3:])
                
                path.append(full_pose)
                
            return path
        except nx.NetworkXNoPath:
            print("[PRM] No path found!")
            return []
