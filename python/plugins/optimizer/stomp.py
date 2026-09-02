import numpy as np
import sys
import os
import time

# Adjust path to import OptimizerBase
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.optimizerbase import OptimizerBase
from plugins.optimizer import apf

class Stomp(OptimizerBase):
    def __init__(self, config_path: str = None):
        import json
        if config_path is None:
             config_path = os.path.splitext(__file__)[0] + '.json'
        super().__init__(config_path)
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"[STOMP] Warning: Could not load config: {e}")
            self.config = {}

        self.num_iterations = self.config.get("num_iterations", 50)
        self.num_samples = self.config.get("num_samples", 20)
        self.std_dev_factor = self.config.get("std_dev_factor", 0.5)
        self.smoothing_factor = self.config.get("smoothing_factor", 0.1)
        self.d0 = self.config.get("d0", apf.DEFAULT_D0)  # APF influence distance (m)
        self.w_obs = self.config.get("w_obs", 10.0)  # APF repulsive gain (eta)
        self.k_att = self.config.get("k_att", apf.DEFAULT_K_ATT)  # attractive gain (goal pull)
        self.debug_output_dir = self.config.get(
            "debug_output_dir", os.path.join(os.getcwd(), "debug", "stomp"))
        self.save_convergence_plot = bool(self.config.get("save_convergence_plot", True))
        # Plateau-based early stop, NOT an absolute cost threshold - the goal
        # itself can sit close to an obstacle by design (inspection scanning
        # near a pipe), so obstacle_cost has an irreducible floor near it and
        # "cost < X" is meaningless. Stop once the best cost seen hasn't
        # improved by more than early_stop_rel_tol (relative) for
        # early_stop_patience consecutive iterations. 0/None patience
        # disables early stopping (always run num_iterations, old behavior).
        self.early_stop_patience = self.config.get("early_stop_patience", 10)
        self.early_stop_rel_tol = self.config.get("early_stop_rel_tol", 0.01)
        self.save_playback_trajectory = bool(self.config.get("save_playback_trajectory", True))
        self.save_task_space_plot = bool(self.config.get("save_task_space_plot", True))
        # Which world axes to plot the task-space view on - default x/y.
        self.task_space_axes = list(self.config.get("task_space_axes", [0, 1]))
        self.last_cost_breakdown = None

    def _is_collision_free(self, path, planner):
        if hasattr(planner, 'check_collision_on_path'): # If planner has efficient bulk check
             return not planner.check_collision_on_path(path)
        
        # Fallback to segment check
        for i in range(len(path) - 1):
            if planner._check_collision(path[i], path[i+1]):
                return False
        return True

    def optimize(self, path: list, planner) -> list:
        if not path or len(path) < 3:
            print("[STOMP] Path too short to optimize.")
            return path

        current_path = np.array(path)
        n_waypoints = len(current_path)
        dim = current_path.shape[1]
        
        start_pose = current_path[0].copy()
        goal_pose = current_path[-1].copy()
        
        best_valid_path = None
        if self._is_collision_free(current_path, planner):
            best_valid_path = current_path.copy()

        # Adaptive Noise Scale
        diffs = np.diff(current_path, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        avg_dist = np.mean(dists)
        current_std_dev = avg_dist * self.std_dev_factor
        print(f"[STOMP] Initializing with avg_step_dist={avg_dist:.2f}, std_dev={current_std_dev:.2f}")

        # Smoothing Kernel
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size

        # Per-phase wall-clock totals across the whole optimize() call, so
        # "it's slow" turns into "X% of the time is cost evaluation, calling
        # apf N times per sample" instead of a guess - see the summary
        # printed after the loop. call counters track how many times the
        # expensive per-waypoint distance query actually runs, since that's
        # what scales with n_waypoints * num_samples * num_iterations.
        phase_time = {"noise": 0.0, "cost_eval": 0.0, "update": 0.0, "smoothing_safety_check": 0.0}
        apf_call_count = 0
        # (iter_num, total, length, obstacle, attractive, min_clearance) of the
        # best candidate each iteration - min_clearance is the closest
        # link-to-obstacle distance anywhere on that candidate (None if the
        # backend has no distance data), for the convergence plot's obstacle
        # panel (see _save_convergence_plot).
        cost_history = []
        best_cost_so_far = float("inf")
        stall_count = 0
        stopped_early_at = None
        # (iter_num, path) snapshot of the actual working path at the END of
        # each iteration (post accept/reject) - NOT the best sampled
        # candidate, which may have been rejected for colliding. This is
        # what a playback should show: how the real trajectory evolved.
        path_history = [(-1, current_path.copy())]

        for iter_num in range(self.num_iterations):
            iter_t0 = time.perf_counter()

            # 1. Generate Correlated Noise
            raw_noise = np.random.normal(0, current_std_dev, (self.num_samples, n_waypoints, dim))
            noise = np.zeros_like(raw_noise)
            for i in range(self.num_samples):
                for d in range(dim):
                    noise[i, :, d] = np.convolve(raw_noise[i, :, d], kernel, mode='same')

            noise[:, 0, :] = 0
            noise[:, -1, :] = 0

            # 2. Candidates
            candidates = np.tile(current_path, (self.num_samples, 1, 1))
            candidates += noise
            phase_time["noise"] += time.perf_counter() - iter_t0

            # 3. Evaluate Costs
            cost_eval_t0 = time.perf_counter()
            costs = np.zeros(self.num_samples)
            best_breakdown = None
            for i in range(self.num_samples):
                cand = candidates[i]
                diffs = np.diff(cand, axis=0)
                length_cost = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

                # Obstacle Cost - Artificial Potential Field (Khatib repulsive
                # potential) built from real robot-link/obstacle distances
                # (plugins/optimizer/apf.py), rather than a binary in/out
                # collision flag, so STOMP can grade samples by how close
                # they pass obstacles instead of only rejecting hard hits.
                # path_repulsive_cost() calls link_obstacle_distances() once
                # per waypoint - this is almost always the dominant cost of
                # the whole optimize() call (see the timing summary).
                obstacle_cost = float(np.sum(apf.path_repulsive_cost(cand, planner, self.d0, self.w_obs)))
                apf_call_count += n_waypoints
                collision_free = self._is_collision_free(cand, planner)
                if not collision_free:
                    obstacle_cost += 1e9

                # Attractive Cost - pulls each free waypoint toward its
                # straight-line target between start/goal (see gpmp2.py's
                # identical term / apf.straight_line_targets), so samples
                # that wander off the direct path get penalized on top of
                # the length_cost that already discourages long detours.
                att_cost = float(np.sum(apf.path_attractive_cost(cand, self.k_att)))

                costs[i] = length_cost + obstacle_cost + att_cost
                if best_breakdown is None or costs[i] < best_breakdown[0]:
                    best_breakdown = (costs[i], float(length_cost), float(obstacle_cost), att_cost)
            phase_time["cost_eval"] += time.perf_counter() - cost_eval_t0

            # 4. Update Path
            best_idx = np.argmin(costs)
            min_cost = costs[best_idx]
            min_clearance = min(
                (d for d in (apf.min_distance(apf.link_distances(planner, q)) for q in candidates[best_idx])
                 if d is not None),
                default=None,
            )
            cost_history.append((iter_num, *best_breakdown, min_clearance))

            if min_cost < 1e9:
                 update_t0 = time.perf_counter()
                 exp_cost = np.exp(-10.0 * (costs - min_cost) / (np.max(costs) - min_cost + 1e-6))
                 weights = exp_cost / np.sum(exp_cost)

                 weighted_change = np.zeros_like(current_path)
                 for i in range(self.num_samples):
                     weighted_change += (candidates[i] - current_path) * weights[i]

                 # Proposed update
                 updated_path = current_path + weighted_change
                 updated_path[0] = start_pose
                 updated_path[-1] = goal_pose

                 # 5. Smoothing (with safety check)
                 smoothed_path = updated_path.copy()
                 pad_size = kernel_size // 2
                 for d in range(dim):
                     col = smoothed_path[:, d]
                     padded = np.pad(col, (pad_size, pad_size), mode='edge')
                     smoothed = np.convolve(padded, kernel, mode='valid')
                     smoothed_path[:, d] = smoothed
                 smoothed_path[0] = start_pose
                 smoothed_path[-1] = goal_pose
                 phase_time["update"] += time.perf_counter() - update_t0

                 # Safety Checks
                 safety_t0 = time.perf_counter()
                 accepted = False

                 # Try Smoothed
                 if self._is_collision_free(smoothed_path, planner):
                     current_path = smoothed_path
                     best_valid_path = current_path.copy()
                     accepted = True
                 # Try Unsmoothed
                 elif self._is_collision_free(updated_path, planner):
                     print(f"[STOMP] Smoothing caused collision. Using unsmoothed update.")
                     current_path = updated_path
                     best_valid_path = current_path.copy()
                     accepted = True
                 else:
                     # Both collided.
                     # We can either reject the update or accept strictly better cost even if invalid?
                     # For safety, rejection is better, but might get stuck.
                     # Let's revert to previous valid if available, OR keep updated if it's "less bad"?
                     # But cost function is binary 1e9.
                     print(f"[STOMP] Update caused collision. Skipping update for this iter.")
                     # current_path remains unchanged
                     pass
                 phase_time["smoothing_safety_check"] += time.perf_counter() - safety_t0

                 if iter_num % 10 == 0 or iter_num == self.num_iterations - 1:
                     _, l, o, a = best_breakdown
                     print(
                         f"[STOMP] Iter {iter_num}: Min Cost={min_cost:.2f} "
                         f"(length={l:.2f}, obstacle={o:.2f}, attractive={a:.2f}), "
                         f"accepted={accepted}, iter_time={time.perf_counter() - iter_t0:.3f}s")

                 current_std_dev *= 0.95
            else:
                 current_std_dev *= 0.5

            path_history.append((iter_num, current_path.copy()))

            # Plateau early stop (see __init__ docstring on early_stop_* -
            # relative improvement, not an absolute cost floor, since the
            # goal can legitimately sit at a high, irreducible obstacle_cost).
            if self.early_stop_patience:
                # best_cost_so_far starts at inf, so this branch's threshold
                # (inf - rel_tol*abs(inf) = inf - inf = nan) would silently
                # never fire on the first iteration - min_cost < nan is
                # always False in Python. Treat "no baseline yet" as an
                # unconditional improvement instead of comparing against it.
                if best_cost_so_far == float("inf") or \
                        min_cost < best_cost_so_far - self.early_stop_rel_tol * abs(best_cost_so_far):
                    best_cost_so_far = min_cost
                    stall_count = 0
                else:
                    stall_count += 1
                if stall_count >= self.early_stop_patience:
                    stopped_early_at = iter_num
                    print(
                        f"[STOMP] Early stop at iter {iter_num}: no >{self.early_stop_rel_tol * 100:.1f}% "
                        f"improvement in {self.early_stop_patience} iterations "
                        f"(best_cost={best_cost_so_far:.2f})")
                    break

        ran_iterations = len(cost_history)
        total_time = sum(phase_time.values())
        if total_time > 0:
            stop_note = f" (early-stopped, configured for {self.num_iterations})" if stopped_early_at is not None else ""
            print(f"[STOMP] Timing breakdown over {ran_iterations} iterations{stop_note} "
                  f"({apf_call_count} apf distance queries total):")
            for phase, t in sorted(phase_time.items(), key=lambda kv: -kv[1]):
                print(f"[STOMP]   {phase:24s} {t:8.3f}s ({100.0 * t / total_time:5.1f}%)")
        if len(cost_history) >= 2:
            first = cost_history[0]
            last = cost_history[-1]
            print(
                f"[STOMP] Best-candidate cost trend: iter {first[0]} -> {last[0]}: "
                f"total {first[1]:.2f} -> {last[1]:.2f}, "
                f"length {first[2]:.2f} -> {last[2]:.2f}, "
                f"obstacle {first[3]:.2f} -> {last[3]:.2f}, "
                f"attractive {first[4]:.2f} -> {last[4]:.2f}")

        if self.save_convergence_plot and cost_history:
            self._save_convergence_history(cost_history)
        if self.save_task_space_plot and path_history:
            self._save_task_space_history(path_history, planner)
        if self.save_playback_trajectory and path_history:
            self._save_playback_trajectory(path_history, planner)

        # Return best valid path if we have one, otherwise current (even if invalid)
        final_path = best_valid_path if best_valid_path is not None else current_path
        if best_valid_path is None:
             print("[STOMP] Warning: Could not find any collision-free path.")
        self.last_cost_breakdown = apf.path_cost_breakdown(
            final_path, planner, d0=self.d0, eta=self.w_obs, w_smooth=self.smoothing_factor, k_att=self.k_att)
        return [p for p in final_path]

    def _save_convergence_history(self, cost_history):
        """Write per-iteration best-candidate cost breakdown to CSV, plus a
        2-panel PNG: top = length/obstacle/attractive/total cost vs
        iteration, bottom = min link-to-obstacle clearance vs iteration with
        self.d0 marked (the APF influence boundary - repulsive_potential() is
        0 above this line, this is "the obstacle" in the sense that matters
        for the cost function) and 0 marked (actual penetration)."""
        import csv as csv_module

        out_dir = self.debug_output_dir
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            print(f"[STOMP] Warning: could not create debug_output_dir {out_dir!r}: {e}")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"stomp_convergence_{timestamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.writer(f)
            writer.writerow(["iter", "total", "length", "obstacle", "attractive", "min_clearance"])
            for row in cost_history:
                writer.writerow(row)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(f"[STOMP] Wrote {csv_path} (matplotlib unavailable, skipping plot)")
            return

        iters = [row[0] for row in cost_history]
        total = [row[1] for row in cost_history]
        length = [row[2] for row in cost_history]
        obstacle = [row[3] for row in cost_history]
        attractive = [row[4] for row in cost_history]
        min_clearance = [row[5] for row in cost_history]

        fig, (ax_cost, ax_obs) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        ax_cost.plot(iters, total, label="total", color="black", linewidth=2)
        ax_cost.plot(iters, length, label="length", linestyle="--")
        ax_cost.plot(iters, obstacle, label="obstacle (repulsive)", linestyle="--")
        ax_cost.plot(iters, attractive, label="attractive", linestyle="--")
        ax_cost.set_yscale("symlog")
        ax_cost.set_ylabel("cost (symlog)")
        ax_cost.set_title("STOMP best-candidate cost per iteration")
        ax_cost.legend(loc="upper right", fontsize="small")
        ax_cost.grid(True, alpha=0.3)

        # This is the "장애물 표시" panel - the obstacle isn't a 2D shape here
        # (best candidate is an N-waypoint, D-joint path, not a point in a
        # plane), so it's shown as the clearance number that actually drives
        # obstacle_cost, against the two thresholds that matter: self.d0
        # (repulsive potential turns on below this) and 0 (actual collision).
        valid_clearance = [(it, mc) for it, mc in zip(iters, min_clearance) if mc is not None]
        if valid_clearance:
            vc_iters, vc_vals = zip(*valid_clearance)
            ax_obs.plot(vc_iters, vc_vals, label="min link-obstacle clearance", color="tab:red")
            ax_obs.axhline(self.d0, color="orange", linestyle="--",
                            label=f"d0={self.d0} (APF influence boundary)")
            ax_obs.axhline(0.0, color="black", linestyle=":", label="0 (collision)")
            ax_obs.fill_between(vc_iters, vc_vals, 0.0,
                                 where=[v < 0 for v in vc_vals], color="red", alpha=0.2)
        else:
            ax_obs.text(0.5, 0.5, "no clearance data (backend unavailable)",
                        transform=ax_obs.transAxes, ha="center", va="center")
        ax_obs.set_xlabel("iteration")
        ax_obs.set_ylabel("distance (m)")
        ax_obs.legend(loc="upper right", fontsize="small")
        ax_obs.grid(True, alpha=0.3)

        fig.tight_layout()
        png_path = os.path.join(out_dir, f"stomp_convergence_{timestamp}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"[STOMP] Wrote {csv_path}, {png_path}")

    def _save_task_space_history(self, path_history, planner):
        """Task-space (world xy by default, see self.task_space_axes) view of
        how the EE trajectory itself moved across iterations - the joint-
        space convergence plot shows cost going down, but not what that
        looks like relative to the obstacle in the space that actually
        matters for collision. One line per (subsampled) iteration, colored
        light->dark by iteration; marker size scales with that waypoint's
        APF repulsive cost, so a shrinking/moving "big dots" pattern across
        iterations is directly visible - not just a smaller cost number.

        Subsamples iterations (not every single one) because a run can have
        50+ iterations and the point is to see the overall trend, not render
        an illegible pile of overlapping lines.
        """
        backend, robot_name = apf.robot_backend_and_name(planner)
        if backend is None or not robot_name or not hasattr(backend, "frame_world_T"):
            print("[STOMP] Skipping task-space plot: robotics backend has no frame_world_T()")
            return

        max_lines = 8
        n = len(path_history)
        if n <= max_lines:
            sample_indices = list(range(n))
        else:
            sample_indices = sorted(set(np.linspace(0, n - 1, max_lines).round().astype(int).tolist()))

        out_dir = self.debug_output_dir
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            print(f"[STOMP] Warning: could not create debug_output_dir {out_dir!r}: {e}")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        ax_i, ax_j = self.task_space_axes[0], self.task_space_axes[1]
        rows = []  # (iter_num, waypoint, x, y, z, repulsive_cost, min_clearance)
        series = []  # (iter_num, xs, ys, costs)
        for idx in sample_indices:
            iter_num, path = path_history[idx]
            xs, ys, costs = [], [], []
            for wp_idx, q in enumerate(path):
                world_T = np.asarray(backend.frame_world_T(robot_name, q), dtype=float)
                xyz = world_T[:3, 3]
                entries = apf.link_distances(planner, q)
                cost = apf.repulsive_cost(entries, self.d0, self.w_obs)
                clearance = apf.min_distance(entries)
                xs.append(float(xyz[ax_i]))
                ys.append(float(xyz[ax_j]))
                costs.append(float(cost))
                rows.append((iter_num, wp_idx, float(xyz[0]), float(xyz[1]), float(xyz[2]), float(cost), clearance))
            series.append((iter_num, xs, ys, costs))

        import csv as csv_module
        csv_path = os.path.join(out_dir, f"stomp_taskspace_{timestamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.writer(f)
            writer.writerow(["iter", "waypoint", "x", "y", "z", "repulsive_cost", "min_clearance"])
            writer.writerows(rows)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(f"[STOMP] Wrote {csv_path} (matplotlib unavailable, skipping plot)")
            return

        axis_labels = ["x", "y", "z"]
        fig, ax = plt.subplots(figsize=(8, 7))
        cmap = plt.get_cmap("viridis")
        for plot_idx, (iter_num, xs, ys, costs) in enumerate(series):
            color = cmap(plot_idx / max(1, len(series) - 1))
            label = "initial" if iter_num < 0 else f"iter {iter_num}"
            ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.8, label=label, zorder=2)
            sizes = 20.0 + 400.0 * (np.asarray(costs) / max(1e-9, max(costs)))
            ax.scatter(xs, ys, s=sizes, color=color, edgecolor="black", linewidth=0.3, zorder=3)
        ax.set_xlabel(f"world {axis_labels[ax_i]}")
        ax.set_ylabel(f"world {axis_labels[ax_j]}")
        ax.set_title(f"STOMP EE task-space trajectory per iteration (robot={robot_name})\n"
                     f"marker size = APF repulsive cost at that waypoint")
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"stomp_taskspace_{timestamp}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"[STOMP] Wrote {csv_path}, {png_path}")

    def _save_playback_trajectory(self, path_history, planner):
        """Save each iteration's working path as its own "target" in exactly
        test_ompl_planning.py's --target all output shape (summary.csv +
        NN_<robot>_<pose>/joint_states.csv), so SimTool's "Load Playback
        Result" (simtool/playback_loader.py) can load this run and step
        through it iteration-by-iteration exactly like a live planning
        sequence - one plan_sequence entry per iteration, same robot each
        time. iter -1 is the initial (pre-optimization) path."""
        import csv as csv_module

        backend, robot_name = apf.robot_backend_and_name(planner)
        dim = path_history[0][1].shape[1]
        if backend is not None and robot_name:
            try:
                joint_names = [str(n) for n in backend.joint_names(robot_name)]
            except Exception:
                joint_names = [f"joint_{i}" for i in range(dim)]
        else:
            robot_name = "robot"
            joint_names = [f"joint_{i}" for i in range(dim)]
        if len(joint_names) != dim:
            joint_names = [f"joint_{i}" for i in range(dim)]

        # Positioner attitude this path was actually collision-checked
        # against - set by visualizer.py's _plan_inspection_path_for_robot
        # right before calling the optimizer (planner.debug_positioner_r_deg).
        # Without this, playback_loader.py would default to 0deg and render
        # the pipe/positioner unrotated even for a target that needed
        # rotation - the robot's q_path would still be right, just the scene
        # around it wrong.
        positioner_r_deg = float(getattr(planner, "debug_positioner_r_deg", 0.0) or 0.0)
        needs_rotation = bool(getattr(planner, "debug_obstacle_rotated", False))

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.debug_output_dir, f"stomp_playback_{timestamp}")
        try:
            os.makedirs(run_dir, exist_ok=True)
        except Exception as e:
            print(f"[STOMP] Warning: could not create playback dir {run_dir!r}: {e}")
            return

        summary_rows = []
        for row_index, (iter_num, path) in enumerate(path_history):
            pose_name = "initial" if iter_num < 0 else f"iter{iter_num:04d}"
            subdir_name = f"{row_index:02d}_{robot_name}_{pose_name}"
            subdir = os.path.join(run_dir, subdir_name)
            os.makedirs(subdir, exist_ok=True)
            with open(os.path.join(subdir, "joint_states.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv_module.writer(f)
                writer.writerow(["waypoint", *joint_names])
                for wp_idx, q in enumerate(path):
                    writer.writerow([wp_idx, *[float(v) for v in q]])
            summary_rows.append({
                "index": row_index, "group_name": pose_name, "robot_name": robot_name,
                "pose_name": pose_name, "status": "success", "message": "", "n_waypoints": len(path),
                "needs_rotation": needs_rotation, "positioner_r_deg": positioner_r_deg,
            })

        summary_path = os.path.join(run_dir, "summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.DictWriter(
                f, fieldnames=["index", "group_name", "robot_name", "pose_name", "status", "message",
                               "n_waypoints", "needs_rotation", "positioner_r_deg"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[STOMP] Wrote {len(summary_rows)}-iteration playback trajectory to {run_dir} "
              f"(load in SimTool via 'Load Playback Result' -> {summary_path})")

