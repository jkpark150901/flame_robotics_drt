"""CHOMP (Ratliff et al., "CHOMP: Gradient optimization techniques for efficient motion planning", ICRA 2009) with the
project's APF obstacle cost - same code structure, settings style, logs and debug outputs as stomp.py.

Trajectory: the planner's path is resampled to num_points waypoints evenly spaced along its joint-space length (CHOMP
assumes a uniform time discretisation); start and goal stay fixed, the interior waypoints xi are optimised.

Objective (all terms evaluated on the full path including the fixed ends):
    U(xi) = w_smooth * 1/2 * sum ||x[i+1] - 2 x[i] + x[i-1]||^2          smoothness (finite-difference acceleration)
          + sum_i U_apf(x_i)                                              obstacle: APF 0.5*w_obs*(1/d - 1/d0)^2 per
                                                                          (link, obstacle) pair (plugins/optimizer/apf.py,
                                                                          scene SDF distances - same as STOMP/GPMP2)
          + sum_i 0.5*k_att*||s*(x_i - line_i)||^2                        attraction to the start-goal straight line (apf)
Update (covariant gradient descent - the core of CHOMP):
    xi <- xi - learning_rate * A^-1 grad U(xi),   A = K^T K  (K = finite-difference acceleration operator)
The A^-1 preconditioning spreads every obstacle push smoothly over the neighbouring waypoints instead of moving a single
waypoint (the reason CHOMP keeps trajectories smooth). The step is capped at joint_update_limit per joint and clipped to
the joint limits. The obstacle gradient is a finite difference of the APF cost per waypoint and joint, computed only at
waypoints inside the APF influence distance d0 (the gradient is 0 elsewhere).

Inspection constraints (path_planning requirements):
1. Goal posture: settled before planning by the collision-free IK (collision_free_ik.apf_goal_refine, path_planning.cfg
   ik_goal_apf_*): the TCP pose - hence the DDA-pipe distance - is kept exactly while the redundant joints move the rest
   of the arm away from the obstacles. CHOMP therefore keeps the goal fixed (optimize_goal false, default). With
   optimize_goal true it can still move the goal the same way here: the goal waypoint is updated only in the null space
   of the TCP Jacobian and its pose error is corrected back after every step. At the goal the sensor (end link) - pipe
   pair is not penalised; every other link is.
2. Intermediate trajectory: every link, the sensor included, is repelled from every obstacle. The sensor-pipe pair is
   exempt only on the final approach run (contiguous waypoints before the goal whose sensor-pipe distance is below
   approach_zone, 0.10 m - the sensor has to get there) and the symmetric leaving run at the start (the start posture is the
   previous inspection posture); same rule as the evaluation metrics (path_metrics._zone_mask).

Safety: like STOMP, an update is only accepted when the whole path is collision free with the project's collision check
(planner._check_collision per segment); otherwise the step is halved and retried (up to max_step_halvings), else the
iteration keeps the previous path. The best collision-free path is returned.
"""
import json
import os
import time

import numpy as np

from plugins.optimizer import apf
from plugins.optimizer.stomp import Stomp


def apf_pipe_name():
    from plugins.robotics.path_metrics import PIPE_OBSTACLE
    return PIPE_OBSTACLE


def _acceleration_operator(n_total: int) -> np.ndarray:
    """K (n_total-2 x n_total): (K x)[i] = x[i] - 2 x[i+1] + x[i+2]."""
    K = np.zeros((n_total - 2, n_total))
    for i in range(n_total - 2):
        K[i, i], K[i, i + 1], K[i, i + 2] = 1.0, -2.0, 1.0
    return K


def _resample_by_length(path: np.ndarray, n: int) -> np.ndarray:
    """n waypoints evenly spaced along the polyline's joint-space length (ends kept)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-12:
        return np.repeat(path[:1], n, axis=0)
    targets = np.linspace(0.0, s[-1], n)
    out = np.empty((n, path.shape[1]))
    for d in range(path.shape[1]):
        out[:, d] = np.interp(targets, s, path[:, d])
    return out


class Chomp(Stomp):
    TAG = "CHOMP"

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.splitext(__file__)[0] + '.json'
        super().__init__(config_path)   # loads the JSON into self.config and the shared STOMP-style settings
        c = self.config
        self.num_iterations = int(c.get("num_iterations", 100))
        self.num_points = int(c.get("num_points", 40))
        self.learning_rate = float(c.get("learning_rate", 0.05))
        self.w_smooth = float(c.get("w_smooth", 1.0))
        self.d0 = float(c.get("d0", apf.DEFAULT_D0))
        self.w_obs = float(c.get("w_obs", 1.0))
        self.k_att = float(c.get("k_att", 0.0))
        self.fd_eps = float(c.get("fd_eps", 0.01))
        self.joint_update_limit = float(c.get("joint_update_limit", 0.1))
        self.max_step_halvings = int(c.get("max_step_halvings", 4))
        self.debug_output_dir = c.get("debug_output_dir", os.path.join(os.getcwd(), "debug", "chomp"))
        self.time_budget_s = c.get("time_budget_s", 30.0)
        self.early_stop_patience = c.get("early_stop_patience", 10)
        self.early_stop_rel_tol = c.get("early_stop_rel_tol", 0.005)
        self.smoothing_factor = self.w_smooth   # reported by apf.path_cost_breakdown as the smoothness weight
        self.approach_zone = float(c.get("approach_zone", 0.10))
        self.optimize_goal = bool(c.get("optimize_goal", False))
        self.goal_learning_rate = float(c.get("goal_learning_rate", 0.05))
        self.goal_pos_tol = float(c.get("goal_pos_tol", 1e-3))
        self.goal_rot_tol = float(c.get("goal_rot_tol", 0.01))
        self.sensor_link = str(c.get("sensor_link", "link_end"))   # substring of the end link (DDA / RT sensor)
        self.last_stats = {}

    # -- cost terms ------------------------------------------------------------------------------------------------
    def _is_sensor_pipe(self, e):
        return self.sensor_link in str(e["link"]) and e["obstacle"] == apf_pipe_name()

    def _waypoint_cost(self, planner, q, exempt):
        """(APF cost at q, sensor-pipe distance); exempt=True drops the sensor-pipe pair from the cost."""
        entries = apf.link_distances(planner, q)
        d_sp = min((e["distance"] for e in entries if self._is_sensor_pipe(e)), default=None)
        if exempt:
            entries = [e for e in entries if not self._is_sensor_pipe(e)]
        return apf.repulsive_cost(entries, self.d0, self.w_obs), d_sp

    def _exempt_mask(self, d_sp):
        """True where the sensor-pipe pair is not penalised: the goal, the start, and the contiguous runs next to them
        whose sensor-pipe distance is below approach_zone (final approach / leaving the previous inspection pose)."""
        n = len(d_sp)
        mask = np.zeros(n, dtype=bool)
        mask[0] = mask[-1] = True
        i = n - 2
        while i > 0 and d_sp[i] is not None and d_sp[i] < self.approach_zone:
            mask[i] = True
            i -= 1
        j = 1
        while j < i and d_sp[j] is not None and d_sp[j] < self.approach_zone:
            mask[j] = True
            j += 1
        return mask

    def _obstacle_costs(self, path, planner, mask=None):
        """Per-waypoint APF costs (with the sensor-pipe exemption mask, recomputed from this path when None)."""
        raw = [self._waypoint_cost(planner, q, False) for q in path]
        d_sp = [r[1] for r in raw]
        if mask is None:
            mask = self._exempt_mask(d_sp)
        costs = np.array([self._waypoint_cost(planner, q, True)[0] if mask[i] else raw[i][0]
                          for i, q in enumerate(path)])
        return costs, mask

    def _obstacle_gradient(self, path, planner, per_waypoint_cost, mask, indices):
        """Finite-difference gradient of the APF cost at the given waypoints (zero outside the influence distance)."""
        grad = np.zeros_like(path)
        n_queries = 0
        for i in indices:
            if per_waypoint_cost[i] <= 0.0:
                continue
            base = per_waypoint_cost[i]
            for d in range(path.shape[1]):
                q = path[i].copy()
                q[d] += self.fd_eps
                grad[i, d] = (self._waypoint_cost(planner, q, bool(mask[i]))[0] - base) / self.fd_eps
                n_queries += 1
        return grad, n_queries

    # -- goal posture: null-space motion with the TCP pose held ----------------------------------------------------
    def _goal_tools(self, backend, robot_name):
        import pinocchio as pin
        handle = backend._handle(robot_name)
        model = handle.model
        return pin, model, model.createData(), backend.frame_id(robot_name)

    def _goal_step(self, tools, q_goal, grad_goal, T_goal):
        """q_goal moved by -goal_learning_rate * N grad (N = null-space projector of the TCP Jacobian), then pulled back
        onto the TCP pose T_goal by damped least squares. None if the pose cannot be restored within tolerance."""
        from scipy.spatial.transform import Rotation
        pin, model, data, fid = tools
        q = np.asarray(q_goal, dtype=float).copy()
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        J = np.asarray(pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        Jp = np.linalg.pinv(J, rcond=1e-4)
        dq = -self.goal_learning_rate * ((np.eye(len(q)) - Jp @ J) @ grad_goal)
        peak = float(np.max(np.abs(dq))) if dq.size else 0.0
        if peak > self.joint_update_limit:
            dq *= self.joint_update_limit / peak
        q = q + dq
        for _ in range(20):   # pose correction
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            M = data.oMf[fid]
            e_pos = T_goal[:3, 3] - np.asarray(M.translation)
            e_rot = Rotation.from_matrix(T_goal[:3, :3] @ np.asarray(M.rotation).T).as_rotvec()
            if np.linalg.norm(e_pos) < self.goal_pos_tol and np.linalg.norm(e_rot) < self.goal_rot_tol:
                return q
            J = np.asarray(pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), np.concatenate([e_pos, e_rot]))
        return None

    def _attractive_gradient(self, path):
        if not self.k_att:
            return np.zeros_like(path)
        targets = apf.straight_line_targets(path)
        s = apf.DEFAULT_ATTRACTIVE_SCALE
        return self.k_att * s * s * (path - targets)

    def _total_cost(self, path, K, per_waypoint_obs):
        smooth = 0.5 * self.w_smooth * float(np.sum((K @ path) ** 2))
        obs = float(np.sum(per_waypoint_obs))
        att = float(np.sum(apf.path_attractive_cost(path, self.k_att))) if self.k_att else 0.0
        return smooth + obs + att, smooth, obs, att

    # -- main loop -------------------------------------------------------------------------------------------------
    def optimize(self, path: list, planner) -> list:
        if not path or len(path) < 2:
            print("[CHOMP] Path too short to optimize.")
            return path
        seed = np.asarray(path, dtype=float)
        current_path = _resample_by_length(seed, max(self.num_points, 3))
        n_total, dim = current_path.shape
        start_pose, goal_pose = current_path[0].copy(), current_path[-1].copy()
        fixed_indices = [int(i) for i in (getattr(planner, "fixed_joint_indices", None) or [])
                         if 0 <= int(i) < dim and abs(start_pose[int(i)] - goal_pose[int(i)]) <= 1e-9]

        backend, robot_name = apf.robot_backend_and_name(planner)
        lo = hi = None
        if backend is not None and robot_name:
            try:
                lo, hi, _ = backend.joint_limits_for_metric(robot_name, normalize=False)
            except Exception:
                lo = hi = None

        K = _acceleration_operator(n_total)
        A = (K.T @ K)[1:-1, 1:-1]                     # interior block: the metric of the covariant update
        A_inv = np.linalg.inv(A + 1e-9 * np.eye(n_total - 2))
        # normalise so that A^-1's largest diagonal entry is 1: learning_rate is then a step in joint units
        A_inv = A_inv / np.max(np.diag(A_inv))

        best_valid_path = current_path.copy() if self._is_collision_free(current_path, planner) else None
        if best_valid_path is None:
            # the resampled seed can cut a corner of the original into an obstacle: fall back to the original seed
            if self._is_collision_free(seed, planner):
                print("[CHOMP] Resampled seed collides - keeping the planner path as the fallback result.")

        phase_time = {"cost_eval": 0.0, "gradient": 0.0, "update": 0.0, "safety_check": 0.0}
        apf_call_count = 0
        cost_history = []
        path_history = [(-1, current_path.copy())]
        best_cost_so_far = float("inf")
        stall_count = 0
        stopped_early_at = None
        stopped_for_time_budget = False
        accepted_count = rejected_count = 0
        optimize_t0 = time.perf_counter()

        def _time_budget_exceeded():
            return bool(self.time_budget_s) and self.time_budget_s > 0 and \
                (time.perf_counter() - optimize_t0) > self.time_budget_s

        goal_tools = T_goal = None
        if self.optimize_goal and backend is not None and robot_name:
            try:
                goal_tools = self._goal_tools(backend, robot_name)
                pin_, model_, data_, fid_ = goal_tools
                pin_.forwardKinematics(model_, data_, goal_pose)
                pin_.updateFramePlacements(model_, data_)
                T_goal = np.asarray(data_.oMf[fid_].homogeneous, dtype=float)
            except Exception as exc:
                print(f"[CHOMP] goal null-space motion disabled: {exc}")
                goal_tools = None
        goal_moves = 0

        t = time.perf_counter()
        obs_costs, mask = self._obstacle_costs(current_path, planner)
        apf_call_count += 2 * n_total
        phase_time["cost_eval"] += time.perf_counter() - t
        total, smooth, obs, att = self._total_cost(current_path, K, obs_costs)
        # the budget covers the optimisation only: the first APF query builds the scene SDF (seconds to tens of seconds
        # for a large scanned pipe), which used to eat the whole budget before iteration 0
        optimize_t0 = time.perf_counter()
        print(f"[CHOMP] {n_total} points (seed {len(seed)}), initial cost {total:.2f} "
              f"(smooth={smooth:.2f}, obstacle={obs:.2f}, attractive={att:.2f})")

        for iter_num in range(self.num_iterations):
            if _time_budget_exceeded():
                stopped_for_time_budget = True
                print(f"[CHOMP] Time budget ({self.time_budget_s}s) exceeded before iter {iter_num} - "
                      f"stopping with best path found so far.")
                break
            iter_t0 = time.perf_counter()

            # 1. gradient of U at the interior waypoints
            g_t0 = time.perf_counter()
            g_smooth = self.w_smooth * (K.T @ (K @ current_path))
            grad_idx = list(range(1, n_total - 1)) + ([n_total - 1] if goal_tools is not None else [])
            g_obs, nq = self._obstacle_gradient(current_path, planner, obs_costs, mask, grad_idx)
            apf_call_count += nq
            g_full = g_smooth + g_obs + self._attractive_gradient(current_path)
            g = g_full[1:-1]
            g_goal = g_full[-1]
            phase_time["gradient"] += time.perf_counter() - g_t0

            # 2. covariant step, capped per joint
            u_t0 = time.perf_counter()
            step = -self.learning_rate * (A_inv @ g)
            peak = float(np.max(np.abs(step))) if step.size else 0.0
            if peak > self.joint_update_limit:
                step *= self.joint_update_limit / peak
            phase_time["update"] += time.perf_counter() - u_t0

            # 3. accept only collision-free updates that do not increase the cost (step halving otherwise)
            accepted = False
            goal_scale = 1.0
            for _ in range(self.max_step_halvings + 1):
                cand = current_path.copy()
                cand[1:-1] += step
                if lo is not None:
                    cand = np.clip(cand, lo, hi)
                if fixed_indices:
                    cand[:, fixed_indices] = start_pose[fixed_indices]
                cand[0] = start_pose
                cand[-1] = current_path[-1]
                if goal_tools is not None and obs_costs[-1] > 0.0:
                    lr0 = self.goal_learning_rate
                    self.goal_learning_rate = lr0 * goal_scale
                    new_goal = self._goal_step(goal_tools, current_path[-1], g_goal, T_goal)
                    self.goal_learning_rate = lr0
                    if new_goal is not None and (lo is None or (np.all(new_goal >= lo - 1e-9) and np.all(new_goal <= hi + 1e-9))):
                        # carry the goal's change into the interior with a linear ramp (0 at the start, 1 at the goal):
                        # moving only the last waypoint leaves a kink there (worse smoothness)
                        delta = new_goal - current_path[-1]
                        ramp = np.linspace(0.0, 1.0, n_total)[1:-1, None]
                        cand[1:-1] += ramp * delta
                        if lo is not None:
                            cand[1:-1] = np.clip(cand[1:-1], lo, hi)
                        cand[-1] = new_goal
                goal_scale *= 0.5
                c_t0 = time.perf_counter()
                cand_obs, cand_mask = self._obstacle_costs(cand, planner)
                apf_call_count += 2 * n_total
                phase_time["cost_eval"] += time.perf_counter() - c_t0
                cand_total, c_smooth, c_obs, c_att = self._total_cost(cand, K, cand_obs)
                s_t0 = time.perf_counter()
                free = cand_total <= total and self._is_collision_free(cand, planner)
                phase_time["safety_check"] += time.perf_counter() - s_t0
                if free:
                    goal_moves += int(not np.allclose(cand[-1], current_path[-1]))
                    current_path, obs_costs, mask = cand, cand_obs, cand_mask
                    total, smooth, obs, att = cand_total, c_smooth, c_obs, c_att
                    best_valid_path = current_path.copy()
                    accepted = True
                    break
                step *= 0.5
            accepted_count += int(accepted)
            rejected_count += int(not accepted)

            min_clearance = min((d for d in (apf.min_distance(apf.link_distances(planner, q)) for q in current_path)
                                 if d is not None), default=None) if iter_num % 10 == 0 else None
            cost_history.append((iter_num, total, smooth, obs, att, min_clearance))
            path_history.append((iter_num, current_path.copy()))
            if iter_num % 10 == 0 or iter_num == self.num_iterations - 1:
                print(f"[CHOMP] Iter {iter_num}: cost={total:.2f} (smooth={smooth:.2f}, obstacle={obs:.2f}, "
                      f"attractive={att:.2f}), accepted={accepted}, iter_time={time.perf_counter() - iter_t0:.3f}s")

            if self.early_stop_patience:
                if best_cost_so_far == float("inf") or \
                        total < best_cost_so_far - self.early_stop_rel_tol * abs(best_cost_so_far):
                    best_cost_so_far = total
                    stall_count = 0
                else:
                    stall_count += 1
                if stall_count >= self.early_stop_patience:
                    stopped_early_at = iter_num
                    print(f"[CHOMP] Early stop at iter {iter_num}: no >{self.early_stop_rel_tol * 100:.1f}% "
                          f"improvement in {self.early_stop_patience} iterations (cost={total:.2f})")
                    break

        ran = len(cost_history)
        total_time = sum(phase_time.values())
        if total_time > 0:
            stop_note = (f" (time budget exceeded, configured for {self.num_iterations})" if stopped_for_time_budget
                         else f" (early-stopped, configured for {self.num_iterations})" if stopped_early_at is not None
                         else "")
            print(f"[CHOMP] Timing breakdown over {ran} iterations{stop_note} ({apf_call_count} apf distance queries, "
                  f"{accepted_count} accepted / {rejected_count} rejected updates):")
            for phase, t in sorted(phase_time.items(), key=lambda kv: -kv[1]):
                print(f"[CHOMP]   {phase:24s} {t:8.3f}s ({100.0 * t / total_time:5.1f}%)")
        if len(cost_history) >= 2:
            first, last = cost_history[0], cost_history[-1]
            print(f"[CHOMP] Cost trend: iter {first[0]} -> {last[0]}: total {first[1]:.2f} -> {last[1]:.2f}, "
                  f"smooth {first[2]:.2f} -> {last[2]:.2f}, obstacle {first[3]:.2f} -> {last[3]:.2f}")

        if self.save_convergence_plot and cost_history:
            self._save_convergence_history(cost_history)
        if self.save_task_space_plot and path_history:
            self._save_task_space_history(path_history, planner)
        if self.save_playback_trajectory and path_history:
            self._save_playback_trajectory(path_history, planner)

        if best_valid_path is not None:
            final_path = best_valid_path
            self.last_optimization_status = "time_budget_exceeded" if stopped_for_time_budget else "success"
        else:
            print("[CHOMP] Warning: no collision-free path - returning the planner path unchanged.")
            final_path = seed
            self.last_optimization_status = "no_collision_free_path_found"
        self.last_stats = {"iterations": ran, "accepted": accepted_count, "rejected": rejected_count,
                           "goal_moves": goal_moves,
                           "goal_joint_change": float(np.max(np.abs(np.asarray(final_path[-1]) - goal_pose)))
                           if best_valid_path is not None else 0.0,
                           "apf_queries": apf_call_count, "time_s": time.perf_counter() - optimize_t0}
        self.last_cost_breakdown = apf.path_cost_breakdown(
            final_path, planner, d0=self.d0, eta=self.w_obs, w_smooth=self.w_smooth, k_att=self.k_att)
        return [p for p in final_path]
