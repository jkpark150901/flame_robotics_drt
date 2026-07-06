import numpy as np
import json
import os
from typing import List, Union
import sys

# Adjust path to import PlannerBase
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.plannerbase import PlannerBase

class RRTConnect(PlannerBase):
    def __init__(self, config_path: str = None):
        super().__init__()
        if config_path is None:
            config_path = os.path.splitext(__file__)[0] + '.json'
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.step_size = self.config.get("step_size", 1.0)
        self.max_iter = self.config.get("max_iter", 1000)
        self.bounds = self.config.get("workspace_bounds", {
            "x_min": -10.0, "x_max": 10.0,
            "y_min": -10.0, "y_max": 10.0,
            "z_min": -10.0, "z_max": 10.0
        })

        self.configure_collision(self.config, default_sample_resolution=self.step_size)

    def generate(self, current_pose: Union[List[float], np.ndarray], target_pose: Union[List[float], np.ndarray]) -> List[np.ndarray]:
        current_pose = np.array(current_pose, dtype=float)
        target_pose = np.array(target_pose, dtype=float)

        if (
            self.pin_model is not None
            and current_pose.shape[0] == self.pin_model.nq
            and target_pose.shape[0] == self.pin_model.nq
        ):
            return self._generate_joint_space(current_pose, target_pose)
        if self.pin_model is not None:
            raise ValueError(
                "RRTConnect is configured for Pinocchio collision, so generate() "
                f"must receive q-space states with nq={self.pin_model.nq}; "
                f"got {current_pose.shape[0]}->{target_pose.shape[0]}"
            )
        
        start_pos = current_pose[:3]
        goal_pos = target_pose[:3]
        
        # Two trees: Start Tree (A) and Goal Tree (B)
        tree_a = [start_pos]
        parents_a = {0: None}
        
        tree_b = [goal_pos]
        parents_b = {0: None}
        
        path_found = False
        connect_node_a_idx = -1
        connect_node_b_idx = -1
        
        for i in range(self.max_iter):
            # Sample random point
            rnd_point = np.array([
                np.random.uniform(self.bounds['x_min'], self.bounds['x_max']),
                np.random.uniform(self.bounds['y_min'], self.bounds['y_max']),
                np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
            ])
            
            # Extend Tree A
            new_idx_a = self._extend(tree_a, parents_a, rnd_point)
            
            if new_idx_a is not None:
                # Try to connect Tree B to the new node in A
                new_node_a = tree_a[new_idx_a]
                new_idx_b = self._connect(tree_b, parents_b, new_node_a)
                
                if new_idx_b is not None:
                    # Connected!
                    connect_node_a_idx = new_idx_a
                    connect_node_b_idx = new_idx_b
                    path_found = True
                    break
                    
            # Swap trees
            tree_a, tree_b = tree_b, tree_a
            parents_a, parents_b = parents_b, parents_a
            
        if path_found:
            # Reconstruct path
            # Current tree_a is the one that just connected (or was swapped)
            # We need to trace back both trees.
            # Ideally we keep track of which is start and which is goal.
            
            # To simplify, let's keep track of "start_tree" and "goal_tree" reference
            # But since we swapped, we need to know which is which.
            # Method: Start Search always extends A first.
            # If swapped, A is now Goal Tree.
            
            # Let's verify which tree is which.
            # We can check root of tree_a using parents_a[root] == None
            # The root of tree_a will either be start_pos or goal_pos.
            
            root_a = tree_a[0]
            is_tree_a_start = np.allclose(root_a, start_pos)
            
            path_a = self._trace_path(tree_a, parents_a, connect_node_a_idx)
            path_b = self._trace_path(tree_b, parents_b, connect_node_b_idx)
            
            if is_tree_a_start:
                full_path_pos = path_a[::-1] + path_b
            else:
                full_path_pos = path_b[::-1] + path_a
                
            # Construct full 6D path
            full_path = []
            for pos in full_path_pos:
                pose = np.copy(current_pose)
                pose[:3] = pos
                # Orientation handling similar to RRT (simple keep or target)
                # Just keeping start orientation for now
                full_path.append(pose)
            
            # Set final orientation
            final_orient = target_pose[3:]
            current_orient = current_pose[3:]
            full_path[-1][3:] = np.where(np.isnan(final_orient), current_orient, final_orient)
            
            return full_path
            
        return []

    def _extend(self, nodes, parents, target_point):
        """Extend tree towards target_point. Returns new node index or None."""
        dists = np.linalg.norm(np.array(nodes) - target_point, axis=1)
        nearest_idx = np.argmin(dists)
        nearest_node = nodes[nearest_idx]
        
        direction = target_point - nearest_node
        length = np.linalg.norm(direction)
        if length == 0:
            return None
            
        direction /= length
        new_point = nearest_node + direction * min(self.step_size, length)
        
        if not self._check_collision(nearest_node, new_point):
            nodes.append(new_point)
            new_idx = len(nodes) - 1
            parents[new_idx] = nearest_idx
            return new_idx
        return None

    def _connect(self, nodes, parents, target_point):
        """Try to connect tree to target_point (repeat extend). Returns new node index if reached, else None/Last."""
        # In RRT-Connect, 'Connect' means repeated extension until blocked or reached.
        # But for simplicity here, we can just try one extension or strict connect.
        # Standard RRT-Connect: Extend until obstacle or reached.
        
        last_idx = -1
        while True:
            # Find nearest in this tree to target (target is static here)
            dists = np.linalg.norm(np.array(nodes) - target_point, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_node = nodes[nearest_idx]
            
            direction = target_point - nearest_node
            dist = np.linalg.norm(direction)
            
            if dist < self.step_size:
                # Can reach directly
                if not self._check_collision(nearest_node, target_point):
                    nodes.append(target_point)
                    new_idx = len(nodes) - 1
                    parents[new_idx] = nearest_idx
                    return new_idx
                else:
                    return None # Blocked
            
            # Step towards
            direction /= dist
            new_point = nearest_node + direction * self.step_size
            
            if not self._check_collision(nearest_node, new_point):
                nodes.append(new_point)
                new_idx = len(nodes) - 1
                parents[new_idx] = nearest_idx
                last_idx = new_idx
            else:
                return None # Blocked
            
            # If we made progress, calculate new dist. 
            # In standard Connect, we just keep going from the NEW node.
            # My logic above recalculates nearest from whole tree. 
            # Optimization: just continue from new_idx.
            
            # Let's implement strict Greedy Connect from finding nearest ONCE:
            # 1. nearest_node = Nearest(tree, target)
            # 2. While True: step from nearest_node to target.
            
            # Since I already implemented step in loop, let's fix the logic:
            # We continue extending FROM THE LAST ADDED NODE.
            pass
            break # Break loop to re-impl below properly for brevity:
            
        # Re-implementation of Greedy Connect
        dists = np.linalg.norm(np.array(nodes) - target_point, axis=1)
        curr_idx = np.argmin(dists)
        curr_node = nodes[curr_idx]
        
        while True:
            direction = target_point - curr_node
            dist = np.linalg.norm(direction)
            
            if dist < self.step_size:
                if not self._check_collision(curr_node, target_point):
                    nodes.append(target_point)
                    new_idx = len(nodes) - 1
                    parents[new_idx] = curr_idx
                    return new_idx
                return None
            
            direction /= dist
            new_point = curr_node + direction * self.step_size
            
            if not self._check_collision(curr_node, new_point):
                nodes.append(new_point)
                new_idx = len(nodes) - 1
                parents[new_idx] = curr_idx
                curr_idx = new_idx
                curr_node = new_point
            else:
                return None

    def _trace_path(self, nodes, parents, idx):
        path = []
        while idx is not None:
            path.append(nodes[idx])
            idx = parents[idx]
        return path

    def _generate_joint_space(self, start_q, goal_q):
        if self.check_pinocchio_collision(start_q):
            print("RRT-Connect failed: start configuration is in collision")
            return []
        if self.check_pinocchio_collision(goal_q):
            print("RRT-Connect failed: goal configuration is in collision")
            return []

        tree_a = [start_q]
        parents_a = {0: None}
        tree_b = [goal_q]
        parents_b = {0: None}
        path_found = False
        connect_node_a_idx = -1
        connect_node_b_idx = -1

        for _ in range(self.max_iter):
            rnd_point = goal_q if np.random.random() < 0.1 else self._sample_pinocchio_configuration()
            new_idx_a = self._extend_q(tree_a, parents_a, rnd_point)
            if new_idx_a is not None:
                new_idx_b = self._connect_q(tree_b, parents_b, tree_a[new_idx_a])
                if new_idx_b is not None:
                    connect_node_a_idx = new_idx_a
                    connect_node_b_idx = new_idx_b
                    path_found = True
                    break
            tree_a, tree_b = tree_b, tree_a
            parents_a, parents_b = parents_b, parents_a

        if not path_found:
            return []

        is_tree_a_start = np.allclose(tree_a[0], start_q)
        path_a = self._trace_path(tree_a, parents_a, connect_node_a_idx)
        path_b = self._trace_path(tree_b, parents_b, connect_node_b_idx)
        if is_tree_a_start:
            return path_a[::-1] + path_b
        return path_b[::-1] + path_a

    def _extend_q(self, nodes, parents, target):
        dists = np.linalg.norm(np.array(nodes) - target, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_node = nodes[nearest_idx]
        new_point = self._steer_state(nearest_node, target, self.step_size)
        if not self._check_collision(nearest_node, new_point):
            nodes.append(new_point)
            new_idx = len(nodes) - 1
            parents[new_idx] = nearest_idx
            return new_idx
        return None

    def _connect_q(self, nodes, parents, target):
        dists = np.linalg.norm(np.array(nodes) - target, axis=1)
        curr_idx = int(np.argmin(dists))
        curr_node = nodes[curr_idx]

        while True:
            dist = float(np.linalg.norm(target - curr_node))
            if dist < self.step_size:
                if not self._check_collision(curr_node, target):
                    nodes.append(target)
                    new_idx = len(nodes) - 1
                    parents[new_idx] = curr_idx
                    return new_idx
                return None

            new_point = self._steer_state(curr_node, target, self.step_size)
            if self._check_collision(curr_node, new_point):
                return None
            nodes.append(new_point)
            new_idx = len(nodes) - 1
            parents[new_idx] = curr_idx
            curr_idx = new_idx
            curr_node = new_point
