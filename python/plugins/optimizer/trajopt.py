import numpy as np
import sys
import os
import json
from scipy.optimize import minimize

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.optimizerbase import OptimizerBase
from plugins.optimizer import apf

class TrajOpt(OptimizerBase):
    def __init__(self, config_path: str = None):
        if config_path is None:
             config_path = os.path.splitext(__file__)[0] + '.json'
        super().__init__(config_path)
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"[TrajOpt] Warning: Could not load config: {e}")
            self.config = {}
            
        self.safety_margin = self.config.get("safety_margin", 0.1)  # real meters (APF d0)
        self.w_smooth = self.config.get("w_smooth", 1.0)
        self.w_collision = self.config.get("w_collision", 20.0)  # APF eta
        self.k_att = self.config.get("k_att", apf.DEFAULT_K_ATT)  # attractive gain (goal pull)
        self.max_iter = self.config.get("max_iter", 20)
        self.last_cost_breakdown = None

    def optimize(self, path: list, planner) -> list:
        if not path or len(path) < 3:
            return path
            
        path_arr = np.array(path)
        n_waypoints = path_arr.shape[0]
        dim = path_arr.shape[1]
        
        start_conf = path_arr[0]
        goal_conf = path_arr[-1]
        
        x0 = path_arr[1:-1].flatten()
        
        def reconstruct_path(x):
            inner = x.reshape((n_waypoints - 2, dim))
            return np.vstack([start_conf, inner, goal_conf])
            
        # Cost Function: Smoothness + APF obstacle penalty (Khatib repulsive
        # potential, plugins/optimizer/apf.py, from real robot-link/obstacle
        # distances - joint-space q, not the dead o3d.RaycastingScene-based
        # xyz constraint this used to rely on via planner.scene, an attribute
        # no planner in this codebase ever sets).
        def obj_fn(x):
            curr_path = reconstruct_path(x)
            # Smoothness: Sum of squared distances (shortest path) or accelerations
            # TrajOpt usually does min Sum ||v||^2 or ||a||^2
            # Let's do ||a||^2 (smoothness)
            acc = curr_path[2:] - 2*curr_path[1:-1] + curr_path[:-2]
            smooth_cost = 0.5 * np.sum(acc**2) * self.w_smooth
            obs_cost = float(np.sum(apf.path_repulsive_cost(
                curr_path[1:-1], planner, self.safety_margin, self.w_collision)))
            # Attractive Cost - pulls each free waypoint toward its
            # straight-line target between start/goal (see gpmp2.py's
            # identical term / apf.straight_line_targets for why not the
            # literal goal_conf).
            att_cost = float(np.sum(apf.path_attractive_cost(curr_path, self.k_att)))
            return smooth_cost + obs_cost + att_cost

        # Optimize using SLSQP
        try:
             res = minimize(obj_fn, x0, method='SLSQP',
                            options={'maxiter': self.max_iter, 'disp': True})
             final_x = res.x
        except Exception as e:
             # SLSQP might fail if memory issue or singular
             print(f"[TrajOpt] Optimization failed: {e}")
             final_x = x0

        optimized_path_arr = reconstruct_path(final_x)
        self.last_cost_breakdown = apf.path_cost_breakdown(
            optimized_path_arr, planner, d0=self.safety_margin, eta=self.w_collision,
            w_smooth=self.w_smooth, k_att=self.k_att)
        return [p for p in optimized_path_arr]
