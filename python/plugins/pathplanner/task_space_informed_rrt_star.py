import numpy as np
import json
import os
from typing import List, Union
import sys

# Adjust path to import PlannerBase and TaskSpaceRRTStar
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from plugins.pluginbase.plannerbase import PlannerBase
from task_space_rrt_star import TaskSpaceRRTStar

class TaskSpaceInformedRRTStar(TaskSpaceRRTStar):
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.splitext(__file__)[0] + '.json'
        super().__init__(config_path)
        
    def generate(self, current_pose: Union[List[float], np.ndarray], target_pose: Union[List[float], np.ndarray]) -> List[np.ndarray]:
        current_pose = np.array(current_pose, dtype=float)
        target_pose = np.array(target_pose, dtype=float)
        
        # State
        nodes = [current_pose]
        parents = {0: None}
        costs = {0: 0.0}
        
        w_pos = self.weights['pos']
        w_ori = self.weights['orient']
        
        c_best = float('inf')
        solution_node_idx_list = [] # Keep track of improvements
        
        # For ellipsoid sampling, we focus on POSITION checks against c_best.
        # But wait, c_best is weighted cost. 
        # If we only sample position in ellipsoid determined by c_best, 
        # we might miss solutions if orientation cost is significant?
        # Actually, since weighted_dist = sqrt(w_pos*dpos^2 + w_ori*dori^2),
        # minimal dist is when dori=0.
        # So sqrt(w_pos)*dpos <= cost.
        # dpos <= cost / sqrt(w_pos).
        # So effective c_best for position is c_best_pos_equivalent.
        # We can effectively sample position in ellipsoid defined by Start_pos, Goal_pos, and max_length = c_best / sqrt(w_pos).
        
        start_pos = current_pose[:3]
        goal_pos = target_pose[:3]
        d_min_pos = np.linalg.norm(goal_pos - start_pos)
        
        center = (start_pos + goal_pos) / 2.0
        
        # Rotation for Ellipsoid
        dir_vector = goal_pos - start_pos
        if np.linalg.norm(dir_vector) > 1e-6:
            dir_vector /= np.linalg.norm(dir_vector)
        else:
            dir_vector = np.array([1, 0, 0])
            
        if np.abs(dir_vector[0]) < 0.9:
            temp_vec = np.array([1, 0, 0])
        else:
            temp_vec = np.array([0, 1, 0])
        a2 = np.cross(dir_vector, temp_vec)
        a2 /= np.linalg.norm(a2)
        a3 = np.cross(dir_vector, a2)
        C_rot = np.column_stack((dir_vector, a2, a3))
        
        for i in range(self.max_iter):
            # Sample
            if c_best < float('inf'):
                # Informed Sampling
                # Max length allowed for position path is derived from current best total cost.
                # weighted_cost >= sqrt(w_pos)*mean_pos_dist? Not exactly.
                # Worst case for orientation is 0 cost? Best case.
                # Basically, if we ignore orientation cost, we can bound position.
                # c_pos_max = c_best / sqrt(w_pos)
                
                c_max_pos = c_best / np.sqrt(w_pos)
                
                if c_max_pos > d_min_pos + 1e-6:
                    r1 = c_max_pos / 2.0
                    r_other = np.sqrt(max(0, c_max_pos**2 - d_min_pos**2)) / 2.0
                    L = np.diag([r1, r_other, r_other])
                    
                    while True:
                        x_ball = np.random.uniform(-1, 1, 3)
                        if np.linalg.norm(x_ball) <= 1.0:
                            break
                    
                    rnd_pos = np.dot(C_rot, np.dot(L, x_ball)) + center
                    
                    # Orientation: Uniform
                    rnd_ori = np.array([
                        np.random.uniform(self.bounds['roll_min'], self.bounds['roll_max']),
                        np.random.uniform(self.bounds['pitch_min'], self.bounds['pitch_max']),
                        np.random.uniform(self.bounds['yaw_min'], self.bounds['yaw_max'])
                    ])
                    rnd_point = np.concatenate((rnd_pos, rnd_ori))
                else:
                    # c_best is very tight, sample near line? or just keep sampling
                     rnd_point = np.zeros(6) 
                     # Fallback to rejection sampling or uniform if ellipsoid is virtually line
                     rnd_point[:3] = start_pos # Degenerate
                     # Actually if c_max_pos <= d_min_pos, we are at optimum for position.
                     # Just sample uniform for exploration or goal.
                     if np.random.random() < self.goal_bias:
                         rnd_point = np.copy(target_pose)
                         mask = np.isnan(rnd_point)
                         rnd_point[mask] = 0 # Dummy
                     else:
                        # Uniform
                         rnd_point[0] = np.random.uniform(self.bounds['x_min'], self.bounds['x_max'])
                         rnd_point[1] = np.random.uniform(self.bounds['y_min'], self.bounds['y_max'])
                         rnd_point[2] = np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
                         rnd_point[3] = np.random.uniform(self.bounds['roll_min'], self.bounds['roll_max'])
                         rnd_point[4] = np.random.uniform(self.bounds['pitch_min'], self.bounds['pitch_max'])
                         rnd_point[5] = np.random.uniform(self.bounds['yaw_min'], self.bounds['yaw_max'])
            else:
                 # Standard RRT* Sampling
                if np.random.random() < self.goal_bias:
                    rnd_point = np.copy(target_pose)
                    mask = np.isnan(rnd_point)
                    rnd_point[mask] = np.random.uniform(-np.pi, np.pi, size=np.sum(mask)) 
                else:
                    rnd_point = np.zeros(6)
                    rnd_point[0] = np.random.uniform(self.bounds['x_min'], self.bounds['x_max'])
                    rnd_point[1] = np.random.uniform(self.bounds['y_min'], self.bounds['y_max'])
                    rnd_point[2] = np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
                    rnd_point[3] = np.random.uniform(self.bounds['roll_min'], self.bounds['roll_max'])
                    rnd_point[4] = np.random.uniform(self.bounds['pitch_min'], self.bounds['pitch_max'])
                    rnd_point[5] = np.random.uniform(self.bounds['yaw_min'], self.bounds['yaw_max'])

            # Nearest
            diffs = np.array(nodes) - rnd_point
            pos_diff = diffs[:, :3]
            ori_diff = diffs[:, 3:]
            weighted_sq_dists = w_pos * np.sum(pos_diff**2, axis=1) + w_ori * np.sum(ori_diff**2, axis=1)
            nearest_idx = np.argmin(weighted_sq_dists)
            nearest_node = nodes[nearest_idx]
            
            # Steer
            direction = rnd_point - nearest_node
            length = np.sqrt(w_pos * np.sum(direction[:3]**2) + w_ori * np.sum(direction[3:]**2))
            
            if length == 0: continue
            
            ratio = min(1.0, self.step_size / length)
            new_point = nearest_node + direction * ratio
            
            # Collision
            if self._check_collision(nearest_node, new_point):
                continue
                
            # Helper
            def calc_dist(p1, p2):
                d = p1 - p2
                return np.sqrt(w_pos * np.sum(d[:3]**2) + w_ori * np.sum(d[3:]**2))

            # Neighbors
            diffs_new = np.array(nodes) - new_point
            dists_all = np.sqrt(w_pos * np.sum(diffs_new[:, :3]**2, axis=1) + w_ori * np.sum(diffs_new[:, 3:]**2, axis=1))
            neighbor_indices = np.where(dists_all < self.search_radius)[0]
            
            min_cost = costs[nearest_idx] + calc_dist(new_point, nearest_node)
            best_parent_idx = nearest_idx
            
            for nb_idx in neighbor_indices:
                if nb_idx == nearest_idx: continue
                if not self._check_collision(nodes[nb_idx], new_point):
                    cost = costs[nb_idx] + calc_dist(new_point, nodes[nb_idx])
                    if cost < min_cost:
                        min_cost = cost
                        best_parent_idx = nb_idx
            
            nodes.append(new_point)
            new_idx = len(nodes) - 1
            parents[new_idx] = best_parent_idx
            costs[new_idx] = min_cost
            
            # Rewire
            for nb_idx in neighbor_indices:
                if nb_idx == best_parent_idx: continue
                dist_to_nb = calc_dist(new_point, nodes[nb_idx])
                new_cost_to_nb = min_cost + dist_to_nb
                if new_cost_to_nb < costs[nb_idx]:
                    if not self._check_collision(new_point, nodes[nb_idx]):
                        parents[nb_idx] = new_idx
                        costs[nb_idx] = new_cost_to_nb
                        
            # Check Goal
            mask = np.isnan(target_pose)
            delta = new_point - target_pose
            delta[mask] = 0.0
            dist_to_goal = np.sqrt(w_pos * np.sum(delta[:3]**2) + w_ori * np.sum(delta[3:]**2))
            
            if dist_to_goal < self.step_size:
                final_node = np.copy(target_pose)
                final_node[mask] = new_point[mask]
                
                if not self._check_collision(new_point, final_node):
                    total_cost = min_cost + dist_to_goal
                    if total_cost < c_best:
                        c_best = total_cost
                        solution_node_idx_list.append((new_idx, total_cost, final_node))
                        
        # Reconstruct Best
        if solution_node_idx_list:
            solution_node_idx_list.sort(key=lambda x: x[1])
            best_idx, _, final_pose = solution_node_idx_list[0]
            
            path = [final_pose]
            curr_idx = best_idx
            while curr_idx is not None:
                path.append(nodes[curr_idx])
                curr_idx = parents[curr_idx]
            return path[::-1]
            
        print("Task Space Informed RRT* failed.")
        return []
