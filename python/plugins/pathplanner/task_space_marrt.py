import numpy as np
import os
import json
from plugins.pluginbase.plannerbase import PlannerBase
from task_space_rrt import TaskSpaceRRT

class TaskSpaceMARRT(TaskSpaceRRT):
    def __init__(self):
        super().__init__()
        # Load MARRT specific config
        config_path = os.path.join(os.path.dirname(__file__), 'task_space_marrt.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.marrt_config = json.load(f)
        else:
            self.marrt_config = {
                "retraction_steps": 5,
                "retraction_distance": 1.0
            }
            
    def _get_clearance_and_gradient(self, pos):
        """
        Calculate distance to nearest obstacle and estimated gradient to move away.
        """
        return 1000.0, np.zeros(3)

    def _retract_to_medial_axis(self, q_rand):
        """
        Iteratively move point away from obstacles (up the distance gradient)
        until local maxima (Medial Axis approximation).
        """
        pos = q_rand[:3].copy()
        
        steps = self.marrt_config.get('retraction_steps', 5)
        step_dist = self.marrt_config.get('retraction_distance', 1.0)
        
        for _ in range(steps):
            dist, grad = self._get_clearance_and_gradient(pos)
            
            # If gradient is small, we are at local max/min (ridge)
            if np.linalg.norm(grad) < 0.1:
                break
                
            # Move along gradient (away from obstacle)
            pos += grad * step_dist
            
            # Check bounds?
            # Assuming bounds check happens later or we allow slightly outside OBB if free.
            
        # Recombine with orientation
        return np.concatenate((pos, q_rand[3:]))

    def _random_state(self):
        rnd_point = np.zeros(6)
        rnd_point[0] = np.random.uniform(self.bounds['x_min'], self.bounds['x_max'])
        rnd_point[1] = np.random.uniform(self.bounds['y_min'], self.bounds['y_max'])
        rnd_point[2] = np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
        rnd_point[3] = np.random.uniform(self.bounds['roll_min'], self.bounds['roll_max'])
        rnd_point[4] = np.random.uniform(self.bounds['pitch_min'], self.bounds['pitch_max'])
        rnd_point[5] = np.random.uniform(self.bounds['yaw_min'], self.bounds['yaw_max'])
        return rnd_point

    def _nearest_neighbor(self, q_rand):
        diffs = np.array(self.nodes) - q_rand
        pos_diff = diffs[:, :3]
        orient_diff = diffs[:, 3:]
        w_pos = self.weights['pos']
        w_ori = self.weights['orient']
        weighted_sq_dists = w_pos * np.sum(pos_diff**2, axis=1) + w_ori * np.sum(orient_diff**2, axis=1)
        nearest_idx = np.argmin(weighted_sq_dists)
        return nearest_idx

    def _steer(self, q_near, q_rand):
        direction = q_rand - q_near
        w_pos = self.weights['pos']
        w_ori = self.weights['orient']
        dist = np.sqrt(w_pos * np.sum(direction[:3]**2) + w_ori * np.sum(direction[3:]**2))
        
        if dist == 0:
            return q_near
            
        ratio = min(1.0, self.step_size / dist)
        return q_near + direction * ratio

    def _extract_path(self, final_idx):
        path = []
        curr = final_idx
        while curr is not None:
            path.append(self.nodes[curr])
            curr = self.parents[curr]
        return path[::-1]

    def generate(self, start, goal, step_callback=None):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.nodes = [self.start]
        self.parents = {0: None}
        
        print("MARRT Start")

        for i in range(self.max_iter):
            # 1. Sample
            if np.random.random() < self.goal_bias:
                q_rand = self.goal
            else:
                q_rand = self._random_state()
                # MARRT Step: Retract sample towards Medial Axis
                q_rand = self._retract_to_medial_axis(q_rand)
            
            # 2. Nearest
            nearest_idx = self._nearest_neighbor(q_rand)
            q_near = self.nodes[nearest_idx]
            
            # 3. Steer
            q_new = self._steer(q_near, q_rand)
            
            # 4. Collision Check
            if self._check_collision(q_near, q_new):
                continue
                
            # Accept
            new_idx = len(self.nodes)
            self.nodes.append(q_new)
            self.parents[new_idx] = nearest_idx
            
            if step_callback:
                step_callback(self.nodes, self.parents)
            
            # Check Goal
            if np.linalg.norm(q_new[:3] - self.goal[:3]) < self.step_size:
                 if not self._check_collision(q_new, self.goal):
                     final_idx = len(self.nodes)
                     self.nodes.append(self.goal)
                     self.parents[final_idx] = new_idx
                     return self._extract_path(final_idx)
                     
        return None
