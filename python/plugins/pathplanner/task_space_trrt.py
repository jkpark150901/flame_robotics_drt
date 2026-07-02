import numpy as np
import os
import json
import math
from plugins.pluginbase.plannerbase import PlannerBase
from task_space_rrt import TaskSpaceRRT

class TaskSpaceTRRT(TaskSpaceRRT):
    def __init__(self):
        super().__init__()
        # Load TRRT specific config
        config_path = os.path.join(os.path.dirname(__file__), 'task_space_trrt.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.trrt_config = json.load(f)
        else:
            self.trrt_config = {
                "initial_temperature": 0.1,
                "temp_change_factor": 0.1,
                "temp_increase_rate": 2.0,
                "min_temperature": 1e-6
            }
            
        self.temperature = self.trrt_config.get('initial_temperature', 0.1)
        self.temp_change_factor = self.trrt_config.get('temp_change_factor', 0.1)
        self.n_fail = 0
        self.n_fail_max = 5 # Small constant buffer
        
    def _compute_cost(self, pose):
        """
        Compute cost for a given pose.
        For now, we use inverse of distance to obstacle as cost (maximize clearance).
        Cost = 1.0 / (distance + epsilon)
        """
        return 0.0

    def _transition_test(self, cost_near, cost_new, distance):
        """
        Transition test based on T-RRT logic.
        """
        if cost_new <= cost_near:
            return True
        else:
            delta_cost = cost_new - cost_near
            probability = math.exp(-delta_cost / (self.temperature * 1.0)) # k=1 for simplicity
            if np.random.random() < probability:
                # Accept uphill move, but lower temperature slightly (getting stricter?)
                # Standard T-RRT: Increase T if we fail many times, Decrease T if we accept uphill?
                # Actually usually:
                # If accepted uphill: T increase? No, typically T increases when we FAIL to find nodes.
                # T decreases when we successfully extend (refinement).
                
                # Let's stick to basic:
                # If we accept, we might cool down.
                self.temperature *= (1.0 - self.temp_change_factor)
                self.temperature = max(self.temperature, self.trrt_config.get('min_temperature', 1e-6))
                
                return True
            else:
                return False

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
        self.costs = {0: self._compute_cost(self.start)}
        
        self.temperature = self.trrt_config.get('initial_temperature', 0.1)
        self.n_fail = 0
        
        print(f"TRRT Start. Temp: {self.temperature}")

        for i in range(self.max_iter):
            # 1. Sample
            if np.random.random() < self.goal_bias:
                q_rand = self.goal
            else:
                q_rand = self._random_state()
            
            # 2. Nearest
            nearest_idx = self._nearest_neighbor(q_rand)
            q_near = self.nodes[nearest_idx]
            
            # 3. Steer
            q_new = self._steer(q_near, q_rand)
            
            # 4. Collision Check (Hard Constraint)
            if self._check_collision(q_near, q_new):
                self.n_fail += 1
                if self.n_fail > self.n_fail_max:
                    self.temperature *= self.trrt_config.get('temp_increase_rate', 2.0)
                    self.n_fail = 0
                continue
                
            # 5. Transition Test (Soft Constraint / Optimization)
            cost_near = self.costs[nearest_idx]
            cost_new = self._compute_cost(q_new)
            dist = np.linalg.norm(q_new[:3] - q_near[:3]) # Move distance
            
            if self._transition_test(cost_near, cost_new, dist):
                # Accept
                new_idx = len(self.nodes)
                self.nodes.append(q_new)
                self.parents[new_idx] = nearest_idx
                self.costs[new_idx] = cost_new
                
                if step_callback:
                    step_callback(self.nodes, self.parents)
                
                # Check Goal
                if np.linalg.norm(q_new[:3] - self.goal[:3]) < self.step_size:
                     if not self._check_collision(q_new, self.goal):
                         final_idx = len(self.nodes)
                         self.nodes.append(self.goal)
                         self.parents[final_idx] = new_idx
                         return self._extract_path(final_idx)
            else:
                # Rejected
                self.n_fail += 1
                if self.n_fail > self.n_fail_max:
                    self.temperature *= self.trrt_config.get('temp_increase_rate', 2.0)
                    self.n_fail = 0
                    
        return None
