import numpy as np
import sys
import os
import json
from scipy.interpolate import interp1d

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.optimizerbase import OptimizerBase

class ToppRA(OptimizerBase):
    def __init__(self, config_path: str = None):
        if config_path is None:
             config_path = os.path.splitext(__file__)[0] + '.json'
        super().__init__(config_path)
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"[TOPP-RA] Warning: Could not load config: {e}")
            self.config = {}
            
        self.max_vel = self.config.get("max_vel", 5.0)
        self.max_acc = self.config.get("max_acc", 2.0)
        self.dt = self.config.get("dt", 0.1)

    def optimize(self, path: list, planner) -> list:
        """
        Simplified Time-Optimal Path Parameterization.
        Resamples the path to satisfy velocity and acceleration constraints at fixed time step dt.
        Effectively generates a trajectory q(t) sampled at dt.
        """
        if not path or len(path) < 3:
            return path
            
        path_arr = np.array(path)
        
        # 1. Parameterize by Arc Length
        # Calculate distances between waypoints
        diffs = np.diff(path_arr, axis=0) # (N-1, D)
        dists = np.linalg.norm(diffs, axis=1) # (N-1,)
        # Cumulative distance (s)
        s = np.zeros(len(path_arr))
        s[1:] = np.cumsum(dists)
        total_length = s[-1]
        
        if total_length < 1e-3:
            return path
            
        # 2. Simple Trapezoidal Velocity Profile in s-domain
        # We treat the entire path as 1D (s) and find time to traverse it.
        # This is a simplification: assumes curvature doesn't limit velocity (not true for full TOPP-RA)
        # But good enough for a basic resampler.
        
        # Max velocity reachable?
        # V_peak = min(max_vel, sqrt(total_length * max_acc)) (Triangle profile check)
        
        v_limit_by_acc = np.sqrt(total_length * self.max_acc) # If purely triangular
        v_peak = min(self.max_vel, v_limit_by_acc)
        
        # Time to accelerate to v_peak
        t_acc = v_peak / self.max_acc
        s_acc = 0.5 * self.max_acc * t_acc**2
        
        # Check if we have constant velocity phase
        # 2 * s_acc <= total_length?
        if 2 * s_acc <= total_length + 1e-6:
            # Trapezoid
            s_const = total_length - 2 * s_acc
            t_const = s_const / v_peak
            total_time = 2 * t_acc + t_const
        else:
            # Triangle (should be covered by v_peak calc, but just in case)
            t_acc = np.sqrt(total_length / self.max_acc)
            v_peak = self.max_acc * t_acc
            s_acc = total_length / 2.0
            t_const = 0
            total_time = 2 * t_acc
            
        # 3. Generate Time Steps
        num_steps = int(np.ceil(total_time / self.dt))
        times = np.linspace(0, total_time, num_steps)
        
        # 4. Map Time -> s(t)
        s_t = np.zeros_like(times)
        
        for i, t in enumerate(times):
            if t <= t_acc:
                # Accelerating
                s_t[i] = 0.5 * self.max_acc * t**2
            elif t <= t_acc + t_const:
                # Constant velocity
                s_t[i] = s_acc + v_peak * (t - t_acc)
            else:
                # Decelerating
                t_dec = t - (t_acc + t_const)
                # Distance covered from start of decel
                # v(t) = v_peak - a*t
                # s(t) = v_peak*t - 0.5*a*t^2
                s_curr = v_peak * t_dec - 0.5 * self.max_acc * t_dec**2
                s_t[i] = s_acc + s_const + s_curr
                
        # Clamp s_t
        s_t = np.clip(s_t, 0, total_length)
        
        # 5. Interpolate Path at s(t)
        # Function q(s)
        # We use linear interpolation between waypoints for q(s)
        # dim components
        optimized_path = np.zeros((num_steps, path_arr.shape[1]))
        
        for d in range(path_arr.shape[1]):
            # interp1d maps s -> q_d
            interpolator = interp1d(s, path_arr[:, d], kind='linear')
            optimized_path[:, d] = interpolator(s_t)
            
        # Optional: Check collision for new points? 
        # Since we just resampled the original path (which was collision free hopefully), 
        # it should be fine assuming sufficient resolution.
        # But this implementation assumes the input path is geometric and we just re-time it.
        
        return [p for p in optimized_path]
