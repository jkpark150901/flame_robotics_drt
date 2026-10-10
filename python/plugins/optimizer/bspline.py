import numpy as np
import sys
import os
import json
from scipy.interpolate import splprep, splev

# Adjust path to import OptimizerBase
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.optimizerbase import OptimizerBase

class BSpline(OptimizerBase):
    def __init__(self, config_path: str = None):
        if config_path is None:
             config_path = os.path.splitext(__file__)[0] + '.json'
        super().__init__(config_path)
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"[BSpline] Warning: Could not load config: {e}")
            self.config = {}
            
        self.k = self.config.get("degree", 3)
        self.s = self.config.get("smoothing_factor", 0.5)
        self.num_samples = self.config.get("num_samples", 100)

    def optimize(self, path: list, planner) -> list:
        if not path or len(path) < 3:
            return path
            
        # Convert to numpy array
        path_arr = np.array(path)
        
        # Transpose for splprep (requires list of coordinate arrays: [x, y, z, ...])
        # curve_pts = [path_arr[:, i] for i in range(path_arr.shape[1])]
        
        # Check for duplicates (splprep dislikes duplicates)
        # Filter out consecutive duplicates
        unique_path = [path_arr[0]]
        for i in range(1, len(path_arr)):
            if not np.allclose(path_arr[i], path_arr[i-1]):
                unique_path.append(path_arr[i])
        unique_path_arr = np.array(unique_path)
        
        if len(unique_path_arr) <= self.k:
            print(f"[BSpline] Path too short for degree {self.k}. Returning original.")
            return path
            
        try:
            # splprep
            # weights can be used for "clamping" start/end?
            # splprep returns tck (knots, coefficients, degree), u (parameter values)
            tck, u = splprep(unique_path_arr.T, s=self.s, k=self.k)
            
            # Evaluate at uniform intervals
            u_new = np.linspace(0, 1, self.num_samples)
            new_points = splev(u_new, tck)
            
            # Format back to list of poses
            optimized_path = np.array(new_points).T
            
            # Ensure Start/Goal are preserved (splprep might smooth them if s > 0)
            # Actually with s>0, endpoints might drift?
            # splprep usually respects endpoints if not weighted down?
            # We can force set them.
            optimized_path[0] = path_arr[0]
            optimized_path[-1] = path_arr[-1]
            
            return [p for p in optimized_path]
            
        except Exception as e:
            print(f"[BSpline] Optimization failed: {e}")
            return path
