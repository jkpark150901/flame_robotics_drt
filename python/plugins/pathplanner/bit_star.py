import numpy as np
import json
import os
import sys
from typing import List, Union, Tuple
import heapq

# Adjust path to import PlannerBase
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.plannerbase import PlannerBase

class BITStar(PlannerBase):
    def __init__(self, config_path: str = None):
        super().__init__()
        if config_path is None:
            config_path = os.path.splitext(__file__)[0] + '.json'
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.batch_size = self.config.get("batch_size", 100)
        self.eta = self.config.get("eta", 1.1) # Pruning factor? Or radius scalar? RGG radius often uses eta * (log n / n)^(1/d)
        self.max_iter = self.config.get("max_iter", 1000)
        self.bounds = self.config.get("workspace_bounds", {
            "x_min": -50.0, "x_max": 50.0,
            "y_min": -50.0, "y_max": 50.0,
            "z_min": -50.0, "z_max": 50.0
        })
        # BIT* State
        self.samples = [] # Implicit RGG vertices
        self.V = set() # Tree vertices (indices in samples)
        self.QE = [] # Edge Queue (heap)
        self.QV = [] # Vertex Queue (heap)
        
        self.r = float('inf') # RGG radius
        self.g = {} # Cost to come
        self.parents = {} # Parent index
        self.c_best = float('inf')
        self.goal_idx = -1
        self.start_idx = -1
        self.step_size = self.config.get("step_size", 1.0)
        self.configure_collision(self.config, default_sample_resolution=self.step_size)

    def _calc_heuristic(self, p1, p2):
        return np.linalg.norm(p1 - p2)
    
    def _prune(self, c_best):
        # Remove nodes and edges that cannot improve solution
        # In this simplified version, we just clear queues or filter samples
        # Ideally, we remove v from V if g(v) + h(v) > c_best
        # And remove unconnected samples if heuristic > c_best
        
        # Filter samples
        to_keep_indices = []
        old_to_new_idx = {}
        
        # Keep start and goal
        # For samples not in V: check heuristic < c_best
        # For samples in V: check g[v] + h[v] < c_best
        
        # Actually pruning implementation is complex due to re-indexing.
        # Simplified BIT*: Just don't expand nodes that exceed c_best
        pass
        
    def generate(self, current_pose: Union[List[float], np.ndarray], target_pose: Union[List[float], np.ndarray]) -> List[np.ndarray]:
        current_pose = np.array(current_pose, dtype=float)
        target_pose = np.array(target_pose, dtype=float)
        start_pos = current_pose[:3]
        goal_pos = target_pose[:3]
        
        # Initialization
        self.samples = [start_pos, goal_pos]
        self.start_idx = 0
        self.goal_idx = 1
        
        self.V = {self.start_idx}
        self.g = {self.start_idx: 0.0, self.goal_idx: float('inf')}
        self.parents = {self.start_idx: None}
        self.c_best = float('inf')
        
        # Batches
        batch_id = 0
        
        while batch_id < self.max_iter:
            if not self.QE and not self.QV:
                # Prune
                if self.c_best < float('inf'):
                    # Restart with higher density / batch? 
                    # BIT* adds new batch of samples
                    pass
                    
                # Sample Batch
                new_samples = []
                c_min = np.linalg.norm(start_pos - goal_pos)
                
                # Ellipsoid rotation calculate (same as Informed RRT*)
                dir_vector = goal_pos - start_pos
                if c_min > 0: dir_vector /= c_min
                else: dir_vector = np.array([1, 0, 0])
                
                if np.abs(dir_vector[0]) < 0.9: temp_vec = np.array([1, 0, 0])
                else: temp_vec = np.array([0, 1, 0])
                
                a2 = np.cross(dir_vector, temp_vec)
                a2 /= np.linalg.norm(a2)
                a3 = np.cross(dir_vector, a2)
                C_rot = np.column_stack((dir_vector, a2, a3))
                x_center = (start_pos + goal_pos) / 2.0
                
                for _ in range(self.batch_size):
                    # Sample informed
                    if self.c_best < float('inf'):
                        # Inside ellipsoid
                        while True:
                            x_ball = np.random.uniform(-1, 1, 3)
                            if np.linalg.norm(x_ball) <= 1.0: break
                        
                        r1 = self.c_best / 2.0
                        r_other = np.sqrt(max(0, self.c_best**2 - c_min**2)) / 2.0
                        L = np.diag([r1, r_other, r_other])
                        rnd = np.dot(C_rot, np.dot(L, x_ball)) + x_center
                    else:
                        # Uniform
                        rnd = np.array([
                            np.random.uniform(self.bounds['x_min'], self.bounds['x_max']),
                            np.random.uniform(self.bounds['y_min'], self.bounds['y_max']),
                            np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
                        ])
                    new_samples.append(rnd)
                    
                start_idx_of_new = len(self.samples)
                self.samples.extend(new_samples)
                
                # RGG Radius
                # r = eta * (1 + 1/d)^(1/d) * (vol / unit_ball)^(1/d) * (log q / q)^(1/d)
                # Simplified:
                q = len(self.samples)
                r = self.eta * 50.0 * (np.log(q) / q)**(1.0/3.0) # Heuristic radius scaling
                if r > 200.0: r = 200.0 # Clamp
                self.r = r
                
                # Update Tree
                # Find edges in RGG: (v, x) where v in V, x in Samples_new, dist <= r
                # Add to QE
                # Efficient search: KDTree
                
                # Current Tree Vertices
                tree_indices = list(self.V)
                if not tree_indices: break
                
                from scipy.spatial import KDTree
                
                # All samples
                all_pts = np.array(self.samples)
                kdtree = KDTree(all_pts)
                
                # For each node in V, find neighbors in entire sample set (simplification: usually V_old -> V_new, etc)
                # Proper BIT* expands selectively.
                # Simplified: 
                # Re-calculate edges from V to X_unconnected?
                
                # Let's populate QE with potential edges from V to ALL samples within r
                # Filter those that are useful (heuristic check)
                
                # Query pairs within distance r
                # We can Query for each v in V
                valid_tree_pts = all_pts[tree_indices]
                
                # KDTree query_ball_point
                indices_list = kdtree.query_ball_point(valid_tree_pts, r)
                
                for i, v_idx in enumerate(tree_indices):
                    neighbors = indices_list[i]
                    for x_idx in neighbors:
                        if x_idx == v_idx: continue
                        
                        # Add edge (v, x) to QE
                        # If x not in V (expansion) OR x in V (rewiring - handled by QV?)
                        # BIT* puts edges in QE.
                        
                        # Simplified: Sort by g(v) + h(v, x) + h(x, goal)
                        # Heuristic queue
                        
                        dist = np.linalg.norm(self.samples[v_idx] - self.samples[x_idx])
                        potential_cost = self.g[v_idx] + dist + self._calc_heuristic(self.samples[x_idx], goal_pos)
                        
                        if potential_cost < self.c_best:
                             heapq.heappush(self.QE, (potential_cost, v_idx, x_idx))
                             
                batch_id += 1
                
            # Process Queues
            while self.QE or self.QV:
                # Get best
                # Compare top of QE and QV
                
                # If QV empty, must expand QE
                # If QE empty, must expand QV? 
                # Actually QE -> QV expansion
                
                # Standard Loop:
                if not self.QV:
                    if not self.QE: break
                    current_cost, v, x = heapq.heappop(self.QE)
                else:
                    if not self.QE:
                         # Should not happen in standard flow if QV derived from QE?
                         # QV stores nodes to be expanded?
                         # Actually BIT* uses QE to find best edge to expand tree.
                         # Let's stick to QE-only simplified logic (like lazy PRM/RRT*)
                         pass
                    
                    # Compare
                    if self.QE and self.QE[0][0] < self.QV[0][0]:
                         current_cost, v, x = heapq.heappop(self.QE)
                    else:
                         # Expand vertex?
                         # Simplified: Only use QE.
                         current_cost, v, x = heapq.heappop(self.QE)

                if current_cost > self.c_best:
                    self.QE = [] # Prune rest
                    break
                    
                # Attempt to connect v -> x
                if x in self.V:
                    # Rewiring check
                    dist = np.linalg.norm(self.samples[v] - self.samples[x])
                    new_g = self.g[v] + dist
                    if new_g < self.g.get(x, float('inf')):
                        if not self._check_collision(self.samples[v], self.samples[x]):
                            # Rewire
                            self.g[x] = new_g
                            self.parents[x] = v
                            # Propagate?
                else:
                    # Expansion
                    # g(x) = g(v) + dist
                    dist = np.linalg.norm(self.samples[v] - self.samples[x])
                    new_g = self.g[v] + dist
                    
                    # Update if better
                    if new_g < self.g.get(x, float('inf')):
                         if not self._check_collision(self.samples[v], self.samples[x]):
                             self.V.add(x)
                             self.g[x] = new_g
                             self.parents[x] = v
                             
                             # Check goal
                             dist_to_goal = np.linalg.norm(self.samples[x] - goal_pos)
                             if dist_to_goal < 1e-3 or x == self.goal_idx:
                                 if self.g[x] < self.c_best:
                                     self.c_best = self.g[x]
                                     print(f"[BIT*] New Solution Found: Cost {self.c_best:.2f}")
                                     
                             # Add visible neighbors of x to QE?
                             # In Sample-based, we usually add all edges from x
                             # But we did batch addition.
                             # For proper BIT*, we should look at edges FROM x to others.
                             # Lazy addition:
                             # Query KDTree for neighbors of x
                             
                             # Optimization: If we just added x to V, we should look for potential children y not in V
                             # (and parents for rewiring)
                             
                             # Here we rely on batch-level QE population mostly, 
                             # but strictly should dynamic add.
                             pass

        # Reconstruct
        if self.c_best < float('inf') and self.goal_idx in self.parents:
            path = []
            curr = self.goal_idx
            while curr is not None:
                pose = np.copy(current_pose)
                pose[:3] = self.samples[curr]
                if curr == self.goal_idx:
                     pose[3:] = np.where(np.isnan(target_pose[3:]), current_pose[3:], target_pose[3:])
                else:
                     pose[3:] = current_pose[3:]
                path.append(pose)
                curr = self.parents[curr]
            return path[::-1]
            
        print("[BIT*] No path found")
        return []
