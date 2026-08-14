import numpy as np
import os
from typing import List, Union

from plugins.pathplanner.rrt_star import RRTStar

class InformedRRTStar(RRTStar):
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.splitext(__file__)[0] + '.json'
        super().__init__(config_path)
        if "debug_output_dir" not in self.config:
            self.debug_output_dir = os.path.join(os.getcwd(), "debug", "informed_rrt_star")

    def _generate_workspace(
        self,
        current_pose: Union[List[float], np.ndarray],
        target_pose: Union[List[float], np.ndarray],
        step_callback=None,
    ) -> List[np.ndarray]:
        current_pose = np.array(current_pose, dtype=float)
        target_pose = np.array(target_pose, dtype=float)
        
        start_pos = current_pose[:3]
        goal_pos = target_pose[:3]
        
        # RRT* state
        nodes = [start_pos]
        parents = {0: None}
        costs = {0: 0.0}
        
        # Best solution cost found so far
        c_best = float('inf')
        solution_node_idx = -1
        
        # For Informed RRT*, we need to handle sampling differently once a solution is found.
        # We need rotation matrix from world to ellipse frame
        # C_min = dist(start, goal)
        c_min = np.linalg.norm(goal_pos - start_pos)
        x_center = (start_pos + goal_pos) / 2.0
        
        # Rotation matrix calculation
        # a1 (major axis) is direction from start to goal
        dir_vector = goal_pos - start_pos
        if c_min > 0:
            dir_vector = dir_vector / c_min
        else:
            dir_vector = np.array([1, 0, 0])
            
        # Gram-Schmidt to find rotation matrix C
        # First column is dir_vector
        # We need to find orthogonal basis in 3D
        # id_mat = np.eye(3)
        # One simple way is using SVD or just constructing basis manually
        # Let's use simple logic:
        # a1 = dir_vector.
        # find a2 orthogonal to a1.
        # a3 = a1 x a2
        
        # Propose a random vector, cross product
        # If dir_vector is parallel to X, use Y.
        if np.abs(dir_vector[0]) < 0.9:
            temp_vec = np.array([1, 0, 0])
        else:
            temp_vec = np.array([0, 1, 0])
            
        a2 = np.cross(dir_vector, temp_vec)
        a2 /= np.linalg.norm(a2)
        a3 = np.cross(dir_vector, a2)
        
        C_rot = np.column_stack((dir_vector, a2, a3))
        
        for i in range(self.max_iter):
            # Sampling
            if c_best < float('inf'):
                # Informed Sampling (Ellipsoid)
                # Sample inside unit ball
                while True:
                    x_ball = np.random.uniform(-1, 1, 3)
                    if np.linalg.norm(x_ball) <= 1.0:
                        break
                        
                # Scale
                # r1 = c_best / 2
                # r2 = r3 = sqrt(c_best^2 - c_min^2) / 2
                r1 = c_best / 2.0
                r_other = np.sqrt(max(0, c_best**2 - c_min**2)) / 2.0
                L = np.diag([r1, r_other, r_other])
                
                # Transform
                rnd_point = np.dot(C_rot, np.dot(L, x_ball)) + x_center
            
            else:
                # Standard Sampling (RRT*)
                if np.random.random() < self.goal_bias:
                    rnd_point = goal_pos
                else:
                    rnd_point = np.array([
                        np.random.uniform(self.bounds['x_min'], self.bounds['x_max']),
                        np.random.uniform(self.bounds['y_min'], self.bounds['y_max']),
                        np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
                    ])
            
            # --- Below is similar to RRT* Logic ---
            
            # Nearest
            dists = np.linalg.norm(np.array(nodes) - rnd_point, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_node = nodes[nearest_idx]
            
            # Steer
            direction = rnd_point - nearest_node
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            
            direction /= length
            new_point = nearest_node + direction * min(self.step_size, length)
            
            # Additional constraint: if using informed, check if theoretically possible to improve
            # dist(start, new) + dist(new, goal) < c_best
            # But we don't know parent yet.
            # dist(start via nearest, new) + dist(new, goal) checks?
            if c_best < float('inf'):
                # Heuristic check
                # Note: This is an optimization, not strictly required for correctness, but good for speed
                pass

            if self._check_collision(nearest_node, new_point):
                continue
                
            # Choose Parent
            new_idx = len(nodes)
            dists_all = np.linalg.norm(np.array(nodes) - new_point, axis=1)
            neighbor_indices = np.where(dists_all < self.search_radius)[0]
            
            min_cost = costs[nearest_idx] + np.linalg.norm(new_point - nearest_node)
            best_parent_idx = nearest_idx
            
            for nb_idx in neighbor_indices:
                if nb_idx == nearest_idx: continue
                if not self._check_collision(nodes[nb_idx], new_point):
                    cost = costs[nb_idx] + np.linalg.norm(new_point - nodes[nb_idx])
                    if cost < min_cost:
                        min_cost = cost
                        best_parent_idx = nb_idx
            
            # Improve solution check
            # if min_cost + heuristic(new, goal) < c_best?
            
            nodes.append(new_point)
            parents[new_idx] = best_parent_idx
            costs[new_idx] = min_cost
            
            # Rewire
            for nb_idx in neighbor_indices:
                if nb_idx == best_parent_idx: continue
                dist_to_nb = np.linalg.norm(nodes[nb_idx] - new_point)
                new_cost_to_nb = min_cost + dist_to_nb
                if new_cost_to_nb < costs[nb_idx]:
                    if not self._check_collision(new_point, nodes[nb_idx]):
                        parents[nb_idx] = new_idx
                        costs[nb_idx] = new_cost_to_nb
            
            # Check if we can reach goal from new node
            dist_to_goal = np.linalg.norm(new_point - goal_pos)
            if dist_to_goal < self.step_size:
                if not self._check_collision(new_point, goal_pos):
                    total_cost = min_cost + dist_to_goal
                    if total_cost < c_best:
                        c_best = total_cost
                        solution_node_idx = new_idx
                        # We don't stop, we continue to optimize!
        
        # Reconstruct path
        if solution_node_idx != -1:
            path = []
            
            # Add goal
            pose = np.copy(target_pose)
            final_orient = target_pose[3:]
            current_orient = current_pose[3:]
            pose[3:] = np.where(np.isnan(final_orient), current_orient, final_orient)
            path.append(pose)
            
            curr_idx = solution_node_idx
            while curr_idx is not None:
                pose = np.copy(current_pose)
                pose[:3] = nodes[curr_idx]
                pose[3:] = current_pose[3:]
                path.append(pose)
                curr_idx = parents[curr_idx]
            return path[::-1]
            
        print("Informed RRT* failed to find path")
        return []
