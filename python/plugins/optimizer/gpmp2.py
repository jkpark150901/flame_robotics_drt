import numpy as np
import sys
import os
import json

# Adjust path to import OptimizerBase
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.optimizerbase import OptimizerBase

class GPMP2(OptimizerBase):
    def __init__(self, config_path: str = None):
        if config_path is None:
             config_path = os.path.splitext(__file__)[0] + '.json'
        super().__init__(config_path)
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"[GPMP2] Warning: Could not load config: {e}")
            self.config = {}
            
        self.epsilon = self.config.get("epsilon", 0.1) # Prediction error weight
        self.sigma_obs = self.config.get("sigma_obs", 0.02) # Obstacle variance
        self.w_gp = self.config.get("w_gp", 1.0) # Smoothness weight
        self.w_obs = self.config.get("w_obs", 10.0) # Obstacle weight
        self.num_iterations = self.config.get("num_iterations", 20)
        self.verbose = bool(self.config.get("verbose", False))
        self.last_optimization_status = None

    def optimize(self, path: list, planner) -> list:
        if not path or len(path) < 3:
            self.last_optimization_status = "skipped_short_path"
            return path
        try:
            from scipy.optimize import minimize
        except ImportError:
            self.last_optimization_status = "scipy_unavailable"
            return path
            
        # Convert path to numpy flat array for optimization
        # Path is List[np.ndarray(dim)]
        path_arr = np.array(path)
        n_waypoints = path_arr.shape[0]
        dim = path_arr.shape[1]
        
        # State: We only optimize positions here for simplicity (6D)
        # GPMP2 usually optimizes pos + vel, but we stick to pos for compatibility with this framework's path format
        # Prior: Constant Velocity Model on 6D Poistions
        # Cost = ||Prior|| + ||Obstacle||
        
        # Fixed Start and Goal
        start_conf = path_arr[0]
        goal_conf = path_arr[-1]
        
        # Initial Guess (Inner waypoints)
        # x shape: (n-2) * dim
        x0 = path_arr[1:-1].flatten()
        
        # Helper to reconstruct full path from optimization variables
        def reconstruct_path(x):
            inner = x.reshape((n_waypoints - 2, dim))
            return np.vstack([start_conf, inner, goal_conf])
            
        # Prior Matrix (Inverse covariance / Precision matrix for Constant Velocity)
        # Simplified: Sum of squared differences (like spring system)
        # K*x approx (x_t+1 - 2x_t + x_t-1) for acceleration
        # Actually GP prior for const vel is minimizing acceleration (smoothness)
        # We can implement simple finite diff accel cost.
        
        dt = 1.0 # Assume unit time step
        
        def cost_fn(x):
            curr_path = reconstruct_path(x)
            
            # 1. GP Prior (Smoothness / Acceleration)
            # Vel = (x_i+1 - x_i) / dt
            # Acc = (v_i+1 - v_i) / dt = (x_i+2 - 2x_i+1 + x_i) / dt^2
            # Cost = Sum ||Acc||^2
            if n_waypoints < 3: return 0.0
            
            # Vectorized acceleration calculation
            # Path shape: (N, D)
            # Acc shape: (N-2, D)
            # acc[i] = path[i+2] - 2*path[i+1] + path[i]
            acc = curr_path[2:] - 2*curr_path[1:-1] + curr_path[:-2]
            gp_cost = 0.5 * np.sum(acc**2) * self.w_gp
            
            # 2. Obstacle Cost
            # Signed Distance Field or discrete checks.
            # Since we don't have SDF, we use a simple penalty if collision.
            # To define a gradient, we need distance. 
            # planner._check_collision is boolean. 
            # We can't use gradient descent effectively on boolean.
            # However, `scipy.optimize.minimize` with 'Nelder-Mead' or 'Powell' is derivative free.
            # But high dim...
            # If we utilize `o3d.RaycastingScene`, we CAN get distance!
            # Let's assume planner has `scene` which is `o3d.t.geometry.RaycastingScene`
            
            obs_cost = 0.0
            
            # Check if planner has scene attribute
            has_scene = hasattr(planner, 'scene') and planner.scene is not None
            
            if has_scene:
                # Query SD (Signed Distance)
                # Need tensor points
                # curr_path shape (N, D). Raycasting only supports 3D pos.
                pos_3d = curr_path[:, :3] # (N, 3)
                
                # Convert to o3d Tensor
                import open3d as o3d
                query_points = o3d.core.Tensor(pos_3d, dtype=o3d.core.Dtype.Float32)
                
                # Compute distance
                # compute_distance returns unsigned distance? compute_signed_distance?
                # RaycastingScene usually has compute_distance.
                dist = planner.scene.compute_distance(query_points)
                dist_np = dist.numpy() # (N,)
                
                # Hinge loss or Gaussian potential
                # Cost increases as dist -> 0
                # c(d) = exp(- d^2 / 2sigma^2) ? 
                # or c(d) = max(epsilon - d, 0)
                
                # Let's use Hinge-like: if dist < obstacle_margin, penalize.
                obstacle_margin = 10.0 # Safety radius
                
                # Identify points inside or close
                # Open3D compute_distance is usually distance to nearest surface.
                # If internal, it might be 0.
                
                # Simple repelling field
                penalty = np.zeros_like(dist_np)
                mask = dist_np < obstacle_margin
                if np.any(mask):
                    # Cost = (margin - dist)^2
                    penalty[mask] = (obstacle_margin - dist_np[mask])**2
                    
                obs_cost = np.sum(penalty) * self.w_obs
                
            else:
                # Fallback: Discrete collision checks (no gradient -> flat regions)
                # This will make optimization fail if using gradient methods.
                # For this assignment, we assume we can get distance or just rely on smoothness.
                pass
                
            return gp_cost + obs_cost

        # Optimize
        # 'BFGS' requires gradient. We don't have explicit gradient unless we approximate.
        # 'L-BFGS-B' supports approx_grad=True
        res = minimize(
            cost_fn,
            x0,
            method='L-BFGS-B',
            options={'maxiter': self.num_iterations, 'disp': self.verbose},
        )
        self.last_optimization_status = "success" if bool(res.success) else "optimizer_not_converged"
        
        final_x = res.x
        optimized_path_arr = reconstruct_path(final_x)
        
        return [p for p in optimized_path_arr]
