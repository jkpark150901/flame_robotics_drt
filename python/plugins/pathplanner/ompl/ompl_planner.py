import json
import math
import os
import time
from pathlib import Path

import numpy as np

from plugins.pluginbase.plannerbase import PlannerBase

from .objective_factory import create_optimization_objective
from .ompl_metrics import OMPLRunMetrics
from .path_extractor import ensure_exact_endpoints, extract_full_q_path
from .planner_factory import create_ompl_planner, supported_algorithms
from .state_codec import JointStateCodec
from .validity_adapter import StateValidityAdapter


class OMPLPlanner(PlannerBase):
    """Common OMPL joint-space backend with an explicit legacy rollback hook."""

    use_joint_space_planning = True

    def __init__(self):
        super().__init__()
        self.planner_backend = "ompl"
        self.algorithm = "rrt_connect"
        self.ompl_config = {}
        self.last_planning_status = None
        self.last_returned_path_reaches_goal = False
        self.last_ompl_stats = {}
        self.last_planner_data = None
        self.last_approximate_path = []
        self.last_ompl_summary_json = None

    def configure_ompl(self, config, default_algorithm=None):
        config = dict(config or {})
        self.ompl_config = config
        self.planner_backend = str(
            config.get("planner_backend", config.get("backend", "ompl"))
        ).strip().lower()
        self.algorithm = str(
            config.get("algorithm", default_algorithm or self.algorithm)
        ).strip().lower()
        if self.algorithm not in supported_algorithms():
            raise ValueError(f"unsupported OMPL algorithm: {self.algorithm}")

    def _generate_joint_space(self, start_q, goal_q, step_callback=None):
        backend = str(getattr(self, "planner_backend", "ompl")).lower()
        if backend == "legacy":
            legacy = getattr(self, "_generate_joint_space_legacy", None)
            if legacy is None:
                raise RuntimeError(f"{self.__class__.__name__} has no legacy joint-space backend")
            return legacy(start_q, goal_q, step_callback=step_callback)
        if backend != "ompl":
            raise ValueError(f"unsupported planner backend: {backend}")
        return self._generate_joint_space_ompl(start_q, goal_q, step_callback=step_callback)

    def _generate_joint_space_ompl(self, start_q, goal_q, step_callback=None):
        ob, og, ompl_version = self._import_ompl()
        start_q = np.asarray(start_q, dtype=float).copy()
        goal_q = np.asarray(goal_q, dtype=float).copy()
        self.last_planning_status = "running"
        self.last_returned_path_reaches_goal = False
        self.last_approximate_path = []
        self._begin_convergence_debug("q_space", start_q, goal_q)

        precheck = self._precheck_endpoints(start_q, goal_q)
        if precheck is not None:
            self.last_planning_status = precheck
            metrics = OMPLRunMetrics(self.algorithm, status=precheck)
            self._finish_ompl_run(metrics, ompl_version, 0.0, [])
            return []

        lower, upper = self._full_joint_limits()
        fixed_indices = set(int(i) for i in (getattr(self, "fixed_joint_indices", []) or []))
        active_indices = [i for i in range(start_q.size) if i not in fixed_indices]
        if not active_indices:
            if np.allclose(start_q, goal_q):
                verification = self.verify_path([start_q])
                if self._verification_ok(verification):
                    self.last_planning_status = "success"
                    self.last_returned_path_reaches_goal = True
                    metrics = OMPLRunMetrics(
                        self.algorithm,
                        exact=True,
                        status="success",
                        final_verification=verification,
                    )
                    self._finish_ompl_run(metrics, ompl_version, 0.0, [start_q])
                    return [start_q]
            self.last_planning_status = "no_active_joints"
            metrics = OMPLRunMetrics(self.algorithm, status="no_active_joints")
            self._finish_ompl_run(metrics, ompl_version, 0.0, [])
            return []

        fixed_values = {index: float(start_q[index]) for index in fixed_indices}
        codec = JointStateCodec(
            start_q.size,
            lower,
            upper,
            active_indices,
            fixed_values=fixed_values,
            reference_q=start_q,
        )
        space = ob.RealVectorStateSpace(codec.dimension)
        bounds = ob.RealVectorBounds(codec.dimension)
        for index in range(codec.dimension):
            bounds.setLow(index, 0.0)
            bounds.setHigh(index, 1.0)
        space.setBounds(bounds)

        collision_config = self.ompl_config.get("collision", {}) or {}
        motion_validator = str(collision_config.get("motion_validator", "discrete")).lower()
        if motion_validator != "discrete":
            raise ValueError(
                f"unsupported OMPL motion validator: {motion_validator}; phase 1 supports discrete only"
            )
        normalized_resolution = float(collision_config.get("normalized_resolution", 0.02))
        max_extent = math.sqrt(codec.dimension)
        resolution_fraction = min(1.0, max(1e-9, normalized_resolution / max_extent))
        space.setLongestValidSegmentFraction(resolution_fraction)

        si = ob.SpaceInformation(space)
        counters = {
            "state_validity_calls": 0,
            "state_collision_rejects": 0,
            "workspace_rejects": 0,
            "deadline_rejects": 0,
            "callback_errors": 0,
        }
        validity = StateValidityAdapter(self, codec, counters)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(validity))
        if hasattr(si, "setStateValidityCheckingResolution"):
            si.setStateValidityCheckingResolution(resolution_fraction)
        si.setup()

        start_state = self._make_state(ob, space, codec.full_q_to_state_values(start_q))
        goal_state = self._make_state(ob, space, codec.full_q_to_state_values(goal_q))
        state_config = self.ompl_config.get("state_space", {}) or {}
        goal_tolerance = float(state_config.get("goal_tolerance", 1e-6))
        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start_state, goal_state, goal_tolerance)
        objective_config = self.ompl_config.get("objective", {}) or {}
        objective = create_optimization_objective(
            objective_config.get("type", "path_length"), si, ob
        )
        pdef.setOptimizationObjective(objective)

        factory_config = dict(self.ompl_config)
        factory_config["runtime_step_size"] = getattr(self, "step_size", None)
        factory_config["runtime_goal_bias"] = getattr(self, "goal_bias", None)
        factory_config.setdefault("batch_size", getattr(self, "batch_size", None))
        planner = create_ompl_planner(self.algorithm, si, factory_config, og)
        planner.setProblemDefinition(pdef)
        planner.setup()

        metrics = OMPLRunMetrics(self.algorithm, counters=counters)
        solve_config = self.ompl_config.get("solve", {}) or {}
        timeout = self._solve_timeout(float(solve_config.get("timeout_sec", 5.0)))
        slice_sec = max(0.001, float(solve_config.get("convergence_slice_sec", 0.1)))
        stop_on_first = bool(solve_config.get(
            "stop_on_first_solution", self.algorithm in {"rrt", "rrt_connect"}
        ))
        started = time.perf_counter()
        status = None
        slice_index = 0
        while time.perf_counter() - started < timeout:
            remaining = timeout - (time.perf_counter() - started)
            status = planner.solve(min(slice_sec, max(remaining, 1e-6)))
            exact = self._has_exact_solution(pdef, status, ob)
            approximate = self._has_approximate_solution(pdef, status, ob)
            if exact and metrics.first_solution_time is None:
                metrics.first_solution_time = time.perf_counter() - started
            if exact or approximate:
                path_so_far = pdef.getSolutionPath()
                extracted = extract_full_q_path(path_so_far, codec)
                if extracted:
                    self._record_convergence(
                        self._convergence_rows,
                        iteration=slice_index,
                        phase="exact_solution" if exact else "approximate_solution",
                        state=extracted[-1],
                    )
                metrics.best_cost = self._path_cost(path_so_far, objective)
            if exact and stop_on_first:
                break
            if validity.last_exception is not None or remaining <= slice_sec:
                break
            slice_index += 1

        metrics.solve_time = time.perf_counter() - started
        metrics.exact = self._has_exact_solution(pdef, status, ob)
        metrics.approximate = self._has_approximate_solution(pdef, status, ob)
        self._collect_planner_data(planner, si, ob, metrics)

        if validity.last_exception is not None and not isinstance(validity.last_exception, TimeoutError):
            metrics.status = "validity_callback_error"
            self._finish_ompl_run(metrics, ompl_version, resolution_fraction, [])
            raise RuntimeError("OMPL state validity callback failed") from validity.last_exception

        if not metrics.exact:
            if metrics.approximate:
                self.last_approximate_path = extract_full_q_path(pdef.getSolutionPath(), codec)
                metrics.status = "timeout_approximate"
            else:
                metrics.status = "timeout"
            self.last_planning_status = metrics.status
            self._finish_ompl_run(metrics, ompl_version, resolution_fraction, [])
            return []

        path = extract_full_q_path(pdef.getSolutionPath(), codec)
        path = ensure_exact_endpoints(path, start_q, goal_q, self)
        if not path:
            metrics.status = "goal_connection_failed"
            self.last_planning_status = metrics.status
            self._finish_ompl_run(metrics, ompl_version, resolution_fraction, [])
            return []

        verification = self.verify_path(path)
        metrics.final_verification = verification
        if not self._verification_ok(verification):
            metrics.status = "final_verification_failed"
            self.last_planning_status = metrics.status
            self._finish_ompl_run(metrics, ompl_version, resolution_fraction, path)
            return []

        metrics.status = "success"
        self.last_planning_status = "success"
        self.last_returned_path_reaches_goal = True
        self._record_convergence_from_path("q_space", path, status="success")
        self._finish_ompl_run(metrics, ompl_version, resolution_fraction, path)
        return path

    def _precheck_endpoints(self, start_q, goal_q):
        start_collision, start_pairs = self.check_robot_collision(start_q, return_pairs=True)
        if start_collision:
            self.last_collision_pairs = start_pairs
            return "start_collision"
        goal_collision, goal_pairs = self.check_robot_collision(goal_q, return_pairs=True)
        if goal_collision:
            self.last_collision_pairs = goal_pairs
            return "goal_collision"
        if not self._workspace_position_ok(start_q):
            return "start_out_of_workspace"
        if not self._workspace_position_ok(goal_q):
            return "goal_out_of_workspace"
        return None

    def _full_joint_limits(self):
        backend, robot_name = self._robotics_collision_backend()
        if backend is not None:
            lower, upper, _ = backend.joint_limits_for_metric(robot_name, normalize=True)
        elif self.pin_model is not None:
            lower = np.asarray(self.pin_model.lowerPositionLimit, dtype=float)
            upper = np.asarray(self.pin_model.upperPositionLimit, dtype=float)
        else:
            raise RuntimeError("robot joint limits are not configured")
        lower = np.asarray(lower, dtype=float).copy()
        upper = np.asarray(upper, dtype=float).copy()
        invalid = ~np.isfinite(lower) | ~np.isfinite(upper) | (upper <= lower)
        overrides = self.ompl_config.get("joint_limit_overrides", {}) or {}
        names = list(getattr(self, "fixed_joint_names", []) or [])
        for key, value in overrides.items():
            index = names.index(key) if isinstance(key, str) and key in names else int(key)
            lower[index], upper[index] = float(value[0]), float(value[1])
            invalid[index] = False
        if np.any(invalid):
            indices = np.where(invalid)[0].tolist()
            raise ValueError(
                f"finite OMPL joint bounds are required; configure joint_limit_overrides for {indices}"
            )
        return lower, upper

    def _solve_timeout(self, configured_timeout):
        timeout = max(float(configured_timeout), 1e-6)
        deadline = getattr(self, "planning_deadline", None)
        if deadline is not None:
            timeout = min(timeout, max(float(deadline) - time.monotonic(), 1e-6))
        return timeout

    @staticmethod
    def _make_state(ob, space, values):
        state = ob.State(space)
        for index, value in enumerate(values):
            state[index] = float(value)
        return state

    @staticmethod
    def _status_text(status):
        if status is None:
            return ""
        if hasattr(status, "asString"):
            return str(status.asString()).lower()
        return str(status).lower()

    def _has_exact_solution(self, pdef, status, ob):
        if hasattr(pdef, "hasExactSolution") and pdef.hasExactSolution():
            return True
        return "exact" in self._status_text(status)

    def _has_approximate_solution(self, pdef, status, ob):
        if self._has_exact_solution(pdef, status, ob):
            return False
        if hasattr(pdef, "hasApproximateSolution") and pdef.hasApproximateSolution():
            return True
        return "approximate" in self._status_text(status)

    @staticmethod
    def _path_cost(path_geometric, objective):
        try:
            return float(path_geometric.cost(objective).value())
        except Exception:
            return None

    def _collect_planner_data(self, planner, si, ob, metrics):
        try:
            data = ob.PlannerData(si)
            planner.getPlannerData(data)
            self.last_planner_data = data
            metrics.vertex_count = int(data.numVertices())
            metrics.edge_count = int(data.numEdges())
        except Exception:
            self.last_planner_data = None

    @staticmethod
    def _verification_ok(verification):
        return (
            int(verification.get("colliding_edges", 0)) == 0
            and int(verification.get("colliding_waypoints", 0)) == 0
        )

    def _finish_ompl_run(self, metrics, ompl_version, resolution_fraction, path):
        self.last_ompl_stats = metrics.as_dict()
        self.last_ompl_stats.update({
            "backend": "ompl",
            "ompl_version": ompl_version,
            "motion_resolution_fraction": float(resolution_fraction),
            "path_waypoints": len(path),
        })
        self._save_convergence_debug(
            self._convergence_rows,
            "q_space",
            metrics.status,
            path_waypoints=path,
        )
        debug_config = self.ompl_config.get("debug", {}) or {}
        if not bool(debug_config.get("save_summary", True)):
            return
        out_dir = self._debug_output_path(f"ompl_{self.algorithm}")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = Path(out_dir) / f"ompl_{self.algorithm}_{stamp}_{metrics.status}.json"
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(self.last_ompl_stats, file, ensure_ascii=False, indent=2, default=str)
        self.last_ompl_summary_json = str(summary_path)

    @staticmethod
    def _import_ompl():
        try:
            import ompl
            from ompl import base as ob
            from ompl import geometric as og
        except ImportError as exc:
            raise RuntimeError(
                "OMPL Python binding is required for this planner. "
                "Install the pinned OMPL 2.0.0 binding or select planner_backend='legacy'."
            ) from exc
        version = getattr(ompl, "__version__", "unknown")
        return ob, og, str(version)

    def _generate_workspace(self, current_pose, target_pose, step_callback=None):
        raise NotImplementedError("OMPLPlanner supports joint-space planning only")
