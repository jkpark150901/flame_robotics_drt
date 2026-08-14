"""Joint-space planner backed by the native OMPL Python bindings.

This replaces the previous tesseract-robotics-nanobind based OMPL integration
(now removed). It requires the official OMPL Python bindings to be importable:

    from ompl import base as ob
    from ompl import geometric as og

Install via conda-forge (`conda install -c conda-forge ompl`) or by building
OMPL from source with its Python bindings on PYTHONPATH - NOT via
`pip install tesseract-robotics-nanobind`, which is unrelated and no longer
used anywhere in this project.

State space design follows OMPL_STANDALONE_MIGRATION_PLAN.md section 6:
active (non-fixed) joints are mapped to a RealVectorStateSpace normalized to
[0,1]^N via JointStateCodec, so the OMPL L2 distance matches PlannerBase's
existing normalized joint-space metric. Collision checking reuses
PlannerBase.check_robot_collision()/verify_path() (Pinocchio/backend-based),
so no separate collision environment (Tesseract or otherwise) is needed.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from plugins.pluginbase.plannerbase import PlannerBase
from .state_codec import JointStateCodec
from util.logger.console import ConsoleLogger

_console = ConsoleLogger.get_logger()
# Logged at most once per process - if this binding lacks the iteration-PTC
# API, it'll lack it on every call, so repeating the warning every target
# would just be noise.
_iteration_ptc_warned = False

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    ob = None
    og = None

# Canonical algorithm names - kept identical to the ompl.geometric class names
# (see listavailableplanners.py) so there is no separate name-mapping table to
# keep in sync with the installed OMPL version.
SUPPORTED_ALGORITHMS = (
    "AORRTC", "BFMT", "BITstar", "BKPIECE1", "FMT", "InformedRRTstar",
    "KPIECE1", "LBKPIECE1", "PRM", "PRMstar", "RRT", "RRTConnect",
    "RRTstar", "SORRTstar",
)

# KPIECE-family planners require a projection evaluator on the state space
# before si.setup(), otherwise OMPL raises at planner.setup() time.
_PROJECTION_REQUIRED = {"BKPIECE1", "KPIECE1", "LBKPIECE1"}

DEFAULT_TIMEOUT_SEC = 5.0
DEFAULT_NORMALIZED_RESOLUTION = 0.02
DEFAULT_GOAL_TOLERANCE = 1e-3

# Sidecar config file (same pattern as the legacy per-file planners, e.g.
# rrt_connect.py -> rrt_connect.json): lets timeout_sec/normalized_resolution/
# goal_tolerance/simplify be tuned without touching code, with optional
# per-algorithm overrides. Unlike the legacy planners this one class serves
# all 14 algorithms, so there's one shared file rather than one per algorithm.
_CONFIG_PATH = os.path.splitext(__file__)[0] + ".json"


def _load_ompl_config(path=_CONFIG_PATH):
    """Read ompl_planner_base.json. Missing/invalid file -> hardcoded
    defaults (never fatal - this is tuning, not a required resource)."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    defaults = {
        "timeout_sec": float(data.get("timeout_sec", DEFAULT_TIMEOUT_SEC)),
        "normalized_resolution": float(data.get("normalized_resolution", DEFAULT_NORMALIZED_RESOLUTION)),
        "goal_tolerance": float(data.get("goal_tolerance", DEFAULT_GOAL_TOLERANCE)),
        "simplify": bool(data.get("simplify", False)),
        # 0 = no iteration cap (timeout_sec alone bounds solve()). See
        # _generate_joint_space()'s termination-condition setup.
        "max_iter": int(data.get("max_iter", 0) or 0),
        # None = auto-derive from normalized_resolution and the robot's
        # joint ranges (see configure_ompl()). Set a number here (raw joint
        # radians) to override that derivation directly instead.
        "collision_sample_resolution": (
            float(data["collision_sample_resolution"])
            if data.get("collision_sample_resolution") is not None else None
        ),
    }
    algorithms = {
        str(name): dict(overrides or {})
        for name, overrides in (data.get("algorithms") or {}).items()
    }
    return defaults, algorithms


def _make_validity_checker(si, is_valid_fn):
    """Build an ob.StateValidityChecker from a plain Python callable.

    ob.StateValidityCheckerFn (a convenience function-wrapper) isn't exported
    by every OMPL Python binding build. Subclassing ob.StateValidityChecker
    and overriding isValid() is supported by every binding variant, so use
    that unconditionally instead of trying the convenience wrapper first.
    """
    class _ValidityChecker(ob.StateValidityChecker):
        def __init__(self, si, fn):
            super().__init__(si)
            self._fn = fn

        def isValid(self, state):
            return self._fn(state)

    return _ValidityChecker(si, is_valid_fn)


class OMPLPlannerBase(PlannerBase):
    """Selectable-algorithm OMPL planner. Pick the algorithm via
    configure_ompl({"algorithm": <name in SUPPORTED_ALGORITHMS>})."""

    use_joint_space_planning = True

    def __init__(self):
        super().__init__()
        if ob is None or og is None:
            raise RuntimeError(
                "OMPL python bindings are not installed in this environment. "
                "Install via `conda install -c conda-forge ompl` (or ensure the "
                "OMPL build's python bindings are on PYTHONPATH). This project no "
                "longer uses tesseract-robotics-nanobind.")
        # PlannerBase.__init__() sets an *instance* attribute
        # self.use_joint_space_planning = False, which shadows the class
        # attribute above (instance attrs win over class attrs in Python).
        # Legacy planners (rrt_connect.py etc.) dodge this by overriding
        # generate() entirely and never consulting the flag; this class relies
        # on PlannerBase.generate()'s base dispatch, so it must be re-armed
        # here or every call falls through to the unimplemented
        # _generate_workspace().
        self.use_joint_space_planning = True
        self.algorithm = "RRTConnect"
        self.ompl_config = {}
        # File defaults (ompl_planner_base.json), reloaded fresh per instance
        # so editing the file takes effect without restarting the process -
        # this file lives on disk in the dev/WSL environment and gets edited
        # between runs, not baked into a build.
        self._file_defaults, self._file_algorithm_overrides = _load_ompl_config()
        # Populated by PlannerBase.configure(step_size=...) (hasattr-gated) and
        # used as the RRT-family "range" parameter - see _create_ompl_planner().
        self.step_size = 0.0
        self.timeout_sec = self._file_defaults["timeout_sec"]
        self.normalized_resolution = self._file_defaults["normalized_resolution"]
        self.goal_tolerance = self._file_defaults["goal_tolerance"]
        self.simplify = self._file_defaults["simplify"]
        # Must exist as an instance attribute before PlannerBase.configure()
        # runs, or its `if hasattr(self, "max_iter"): self.max_iter = ...`
        # gate silently skips it - which is exactly why max_iter used to have
        # no effect on OMPL planners at all (see _generate_joint_space()).
        # 0 = no iteration cap, timeout_sec alone bounds solve().
        self.max_iter = int(self._file_defaults.get("max_iter", 0) or 0)
        self.last_planning_status = None
        self.last_returned_path_reaches_goal = False
        self.last_ompl_stats = {}

    def configure_ompl(self, config: dict):
        """Apply OMPL-specific settings. Called by Visualizer after the shared
        PlannerBase.configure() (bounds/step_size/robot model/...).

        Precedence (lowest to highest): hardcoded DEFAULT_* constants <
        ompl_planner_base.json top-level defaults < that file's per-algorithm
        "algorithms" override < explicit keys in `config` (the caller - a
        plan_single_target request, or a script's CLI args - always wins).
        """
        config = dict(config or {})
        algorithm = str(config.get("algorithm", self.algorithm))
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"unsupported OMPL algorithm: {algorithm!r}. "
                f"supported={SUPPORTED_ALGORITHMS}")
        self.algorithm = algorithm
        self.ompl_config = config
        merged = dict(self._file_defaults)
        merged.update(self._file_algorithm_overrides.get(algorithm, {}))
        self.timeout_sec = float(
            config.get("timeout_sec", config.get("planning_timeout", merged["timeout_sec"]))
            or merged["timeout_sec"])
        self.normalized_resolution = float(
            config.get("normalized_resolution", merged["normalized_resolution"]))
        self.goal_tolerance = float(config.get("goal_tolerance", merged["goal_tolerance"]))
        self.simplify = bool(config.get("simplify", merged["simplify"]))

        # Fallback-only now: _check_robot_edge_collision()/
        # collision_pairs_along_edge() are overridden below to sample edges
        # in the *same normalized-space metric* OMPL's own internal motion
        # validator uses, so verify_path() genuinely matches OMPL's own
        # accept/reject decision instead of approximating it. This raw
        # resolution is only used in the (expected to be rare) case where
        # _joint_limits_for_metric() can't produce joint ranges (e.g. no
        # robotics backend configured) and the override falls back to it.
        explicit_resolution = config.get("collision_sample_resolution", merged.get("collision_sample_resolution"))
        if explicit_resolution is not None:
            self.pin_collision_sample_resolution = max(float(explicit_resolution), 1e-6)

    def _normalized_edge_step_count(self, q1, q2):
        """Number of interpolation steps for the p1->p2 edge, computed the
        same way OMPL's own DiscreteMotionValidator does: distance in the
        *normalized* [0,1]^N space (Euclidean, matching RealVectorStateSpace's
        default distance()) divided by the space's actual
        longestValidSegmentLength.

        Per-dimension normalization ((q - lo) / span) is affine, and affine
        maps commute with linear interpolation - so (1-a)*q1 + a*q2 in raw
        space lands on exactly the same point as normalizing then
        interpolating in normalized space. Only the *step count* actually
        needs to change to match OMPL; the interpolated raw q values at a
        given alpha are already identical either way. That's the whole fix:
        _check_robot_edge_collision()/collision_pairs_along_edge() below
        still interpolate in raw q, just with this step count instead of
        one derived from a raw-space distance/resolution that has no
        real relationship to OMPL's normalized-space one (see
        configure_ompl()'s docstring history for the approximation this
        replaced).

        si.setStateValidityCheckingResolution() (configure_ompl(), where
        normalized_resolution is applied) does NOT set an absolute per-axis
        step size - OMPL treats it as a *fraction of the state space's
        maximumExtent* (the bounding-box diagonal), i.e. the real segment
        length OMPL checks against is
        `normalized_resolution * maximumExtent`. Every active joint's bounds
        here are normalized to exactly [0, 1] (see space.setBounds() above),
        so maximumExtent = sqrt(active_dof) - the diagonal of the unit
        hypercube. Leaving that factor out (as an earlier version of this
        method did) makes this replica ~sqrt(active_dof) times finer than
        what OMPL's own solve()/PathSimplifier actually checked, so
        verify_path() ends up rejecting edges OMPL itself already accepted
        (final_verification_failed on a path solve() reported as an exact
        solution).
        """
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        _, _, span = self._joint_limits_for_metric()
        if span is None:
            # No joint ranges available (no robotics backend/model
            # configured) - fall back to the raw-space approximation.
            distance = float(np.linalg.norm(q2 - q1))
            resolution = max(float(self.pin_collision_sample_resolution), 1e-9)
        else:
            distance = float(np.linalg.norm((q2 - q1) / span))
            resolution = max(float(self.normalized_resolution), 1e-9)
            dof = self._robot_dof()
            if dof is not None:
                fixed_count = len(set(int(i) for i in getattr(self, "fixed_joint_indices", []) or []))
                active_dof = max(int(dof) - fixed_count, 1)
                resolution *= float(np.sqrt(active_dof))
        return max(1, int(np.ceil(distance / resolution)))

    def _check_robot_edge_collision(self, p1, p2):
        """Override PlannerBase's raw-space edge sampling (see
        _normalized_edge_step_count's docstring for why)."""
        steps = self._normalized_edge_step_count(p1, p2)
        q1 = np.asarray(p1, dtype=float)
        q2 = np.asarray(p2, dtype=float)
        for i in range(steps + 1):
            self._check_planning_deadline()
            alpha = i / steps
            q = (1.0 - alpha) * q1 + alpha * q2
            if self.check_robot_collision(q):
                return True
        return False

    def collision_pairs_along_edge(self, p1, p2):
        """Override PlannerBase's raw-space edge sampling (see
        _normalized_edge_step_count's docstring for why)."""
        self._check_planning_deadline()
        steps = self._normalized_edge_step_count(p1, p2)
        q1 = np.asarray(p1, dtype=float)
        q2 = np.asarray(p2, dtype=float)
        pairs = []
        seen = set()
        for i in range(steps + 1):
            self._check_planning_deadline()
            alpha = i / steps
            q = (1.0 - alpha) * q1 + alpha * q2
            hit, hit_pairs = self.check_robot_collision(q, return_pairs=True)
            if not hit:
                continue
            for pair in hit_pairs:
                key = tuple(pair)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(pair)
        return pairs

    def _create_ompl_planner(self, space_information):
        factory = getattr(og, self.algorithm, None)
        if factory is None:
            raise RuntimeError(
                f"installed OMPL binding does not export planner class: {self.algorithm}")
        planner = factory(space_information)
        # migration-plan mapping: step_size -> RRT-family "range". Best-effort;
        # not every algorithm exposes setRange().
        range_value = float(getattr(self, "step_size", 0.0) or 0.0)
        if range_value > 0.0 and hasattr(planner, "setRange"):
            try:
                planner.setRange(range_value)
            except Exception:
                pass
        return planner

    def _generate_joint_space(self, start_q, goal_q, step_callback=None):
        start_q = np.asarray(start_q, dtype=float)
        goal_q = np.asarray(goal_q, dtype=float)
        dof = self._robot_dof()
        if dof is None:
            raise RuntimeError("OMPLPlannerBase requires a robot q-space model")

        self.last_planning_status = None
        self.last_returned_path_reaches_goal = False
        self.last_ompl_stats = {}
        # -1 (not "0 iterations") matches rrt_star.py's convention for "not
        # run yet" - _plan_inspection_path_for_robot's timing log
        # (`iteration={planner_iteration_count}`) already reads this
        # attribute; OMPL planners just never set it before now.
        self.last_iteration_count = -1

        # Section 13.1 pre-checks.
        self.last_collision_pairs = []
        self.last_verification = None
        # The actual q_path a failure happened on, for diagnostics/playback -
        # normally a caller only sees "status=failed, message=...(pairs=...)"
        # with no way to see the path itself, since the real caller
        # (path_planning_service.py's plan_single_target) always returns
        # "q_path": [] on any failure (see its except-block). Every early
        # return below sets this so a failure exception can carry it out
        # (visualizer.py attaches it the same way it already does for
        # planner_stats - see the "planning failed for target" raise site).
        self.last_failed_q_path = []
        start_hit, start_pairs = self.check_robot_collision(start_q, return_pairs=True)
        if start_hit:
            self.last_planning_status = "start_collision"
            self.last_collision_pairs = [list(pair) for pair in start_pairs]
            self.last_failed_q_path = [start_q.copy()]
            _console.debug(
                f"OMPLPlannerBase: start_collision, algorithm={self.algorithm} "
                f"start_q={np.round(start_q, 8).tolist()} pairs={self.last_collision_pairs}")
            return []
        goal_hit, goal_pairs = self.check_robot_collision(goal_q, return_pairs=True)
        if goal_hit:
            self.last_planning_status = "goal_collision"
            self.last_collision_pairs = [list(pair) for pair in goal_pairs]
            self.last_failed_q_path = [start_q.copy(), goal_q.copy()]
            _console.debug(
                f"OMPLPlannerBase: goal_collision, algorithm={self.algorithm} "
                f"goal_q={np.round(goal_q, 8).tolist()} pairs={self.last_collision_pairs}")
            return []
        if not self._workspace_position_ok(start_q) or not self._workspace_position_ok(goal_q):
            self.last_planning_status = "workspace_reject"
            return []

        fixed_indices = set(int(i) for i in getattr(self, "fixed_joint_indices", []) or [])
        active_indices = [i for i in range(dof) if i not in fixed_indices]
        if not active_indices:
            raise RuntimeError("OMPL planning requires at least one active (non-fixed) joint")

        lo, hi, _ = self._joint_limits_for_metric()
        if lo is None or hi is None:
            lo = np.full(dof, -np.pi, dtype=float)
            hi = np.full(dof, np.pi, dtype=float)

        codec = JointStateCodec(
            full_dof=dof, lower=lo, upper=hi, active_indices=active_indices,
            fixed_values={i: float(start_q[i]) for i in fixed_indices},
            reference_q=start_q,
        )

        # Some OMPL planners (e.g. AORRTC) throw instead of returning a clean
        # zero-length path when start and goal already coincide in the active
        # joint space. Short-circuit that degenerate case ourselves rather
        # than relying on every algorithm to handle it gracefully.
        start_values = codec.full_q_to_state_values(start_q)
        goal_values = codec.full_q_to_state_values(goal_q)
        already_at_goal_distance = float(np.linalg.norm(goal_values - start_values))
        if already_at_goal_distance <= float(self.goal_tolerance):
            self.last_planning_status = "already_at_goal"
            self.last_returned_path_reaches_goal = True
            self.last_iteration_count = 0
            self.last_ompl_stats = {"iterations": 0, "solve_time": 0.0, "algorithm": self.algorithm}
            _console.debug(
                f"OMPLPlannerBase: already_at_goal, algorithm={self.algorithm} "
                f"distance={already_at_goal_distance:.8f} (tolerance={self.goal_tolerance}) "
                f"start_q={np.round(start_q, 8).tolist()} goal_q={np.round(goal_q, 8).tolist()}")
            return [start_q.copy(), goal_q.copy()]

        space = ob.RealVectorStateSpace(codec.dimension)
        bounds = ob.RealVectorBounds(codec.dimension)
        for i in range(codec.dimension):
            bounds.setLow(i, 0.0)
            bounds.setHigh(i, 1.0)
        space.setBounds(bounds)
        if self.algorithm in _PROJECTION_REQUIRED:
            projection_dim = min(2, codec.dimension)
            try:
                space.registerDefaultProjection(
                    ob.RealVectorRandomLinearProjectionEvaluator(space, projection_dim))
            except AttributeError as exc:
                raise RuntimeError(
                    f"OMPL binding is missing the projection evaluator required by "
                    f"{self.algorithm}: {exc}") from exc

        si = ob.SpaceInformation(space)
        stats = {"state_validity_calls": 0, "collision_rejects": 0, "workspace_rejects": 0}

        def is_valid(state):
            self._check_planning_deadline()
            stats["state_validity_calls"] += 1
            q = codec.state_to_full_q(state)
            if not self._workspace_position_ok(q):
                stats["workspace_rejects"] += 1
                return False
            if self.check_robot_collision(q):
                stats["collision_rejects"] += 1
                return False
            return True

        # Keep a strong Python reference to the validity checker instance for
        # the lifetime of this call. si.setStateValidityChecker() only needs
        # to receive it once, but if this binding's shared_ptr holder doesn't
        # keep the wrapped Python object alive on its own, nothing else here
        # references it after this line and it becomes eligible for GC while
        # OMPL's C++ side still holds a raw pointer to it - a classic
        # pybind11 lifetime bug that manifests as a segfault mid-planning
        # (observed: crashes shortly after OMPL starts calling into
        # isValid(), independent of which algorithm is used).
        validity_checker = _make_validity_checker(si, is_valid)
        si.setStateValidityChecker(validity_checker)
        si.setStateValidityCheckingResolution(max(float(self.normalized_resolution), 1e-4))
        si.setup()

        pdef = ob.ProblemDefinition(si)
        # This binding has neither a public ob.State constructor nor
        # ob.ScopedState - states are only created via alloc on
        # SpaceInformation/StateSpace, and RealVectorStateType supports plain
        # index access (state[i]). setStartAndGoalStates() clones internally
        # (OMPL's ProblemDefinition always clones start/goal states), so we
        # don't need these afterward - but do NOT call si.freeState() on them:
        # this binding's allocState() result is a Python-owned wrapper that
        # frees its underlying C++ state itself when garbage collected: an
        # explicit si.freeState() here frees the same memory a second time
        # once that GC happens, corrupting the heap (observed as "double free
        # or corruption" crashing planner.solve() later, independent of which
        # algorithm was running). Just let start_state/goal_state go out of
        # scope naturally.
        start_state = si.allocState()
        goal_state = si.allocState()
        for i in range(codec.dimension):
            start_state[i] = float(start_values[i])
            goal_state[i] = float(goal_values[i])
        pdef.setStartAndGoalStates(start_state, goal_state, float(self.goal_tolerance))
        pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

        planner = self._create_ompl_planner(si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        deadline = getattr(self, "planning_deadline", None)
        budget = float(self.timeout_sec)
        if deadline is not None:
            budget = max(0.01, min(budget, deadline - time.monotonic()))

        # Wire max_iter into a real termination condition (previously
        # accepted by the API but silently ignored - only timeout_sec ever
        # bounded solve()). This binding doesn't export
        # ob.IterationTerminationCondition/timedPlannerTerminationCondition/
        # plannerOrTerminationCondition (confirmed via the real exception,
        # not guessed - see iteration_ptc_error below), so instead of OMPL's
        # own iteration counter we build a plain-Python termination
        # condition wrapped in ob.PlannerTerminationCondition(callable) - the
        # one PTC construction path that's a core class rather than a
        # convenience free function, so it's the most likely to exist. It
        # closes over `stats["state_validity_calls"]` (already incremented by
        # is_valid() on every call OMPL makes) as the iteration proxy, and
        # checks the time budget itself so no separate time-PTC is needed.
        max_iter = int(getattr(self, "max_iter", 0) or 0)
        deadline_t = time.perf_counter() + budget
        ptc = budget
        iteration_ptc_error = None
        iteration_cap_active = False
        if max_iter > 0:
            global _iteration_ptc_warned

            def _terminate():
                return stats["state_validity_calls"] >= max_iter or time.perf_counter() >= deadline_t

            try:
                ptc = ob.PlannerTerminationCondition(_terminate)
                iteration_cap_active = True
            except Exception as exc:
                # This binding doesn't support custom PTCs either - fall back
                # to the plain time-budget float solve() always supported.
                # Log the *actual* error instead of silently discarding it -
                # this binding has repeatedly turned out to be missing APIs
                # (ob.State(), ob.ScopedState, StateValidityCheckerFn) that
                # exist in the standard OMPL bindings, and guessing instead
                # of looking at the real exception has cost a lot of time.
                ptc = budget
                iteration_ptc_error = str(exc)
                if not _iteration_ptc_warned:
                    _iteration_ptc_warned = True
                    _console.warning(
                        "OMPLPlannerBase: max_iter requested but this OMPL binding doesn't "
                        f"support a custom PlannerTerminationCondition either: {exc!r} - state "
                        "validity checks are still counted (see state_validity_calls) but "
                        "nothing actually caps them at max_iter for this and every subsequent "
                        "call; only timeout_sec bounds solve().")

        solve_t0 = time.perf_counter()
        solved = planner.solve(ptc)
        solve_elapsed = time.perf_counter() - solve_t0
        iterations = int(stats["state_validity_calls"]) if iteration_cap_active else None
        self.last_ompl_stats = {
            **stats,
            "solve_time": solve_elapsed,
            "algorithm": self.algorithm,
            "max_iter": max_iter or None,
            "iterations": iterations,
            "timeout_sec": budget,
            "iteration_ptc_error": iteration_ptc_error,
        }
        # No IterationTerminationCondition (max_iter unset or binding lacks
        # the API) -> state_validity_calls is the closest available proxy for
        # "how much work the search did", still better than leaving the
        # existing timing log's `iteration=` field permanently blank.
        self.last_iteration_count = iterations if iterations is not None else stats["state_validity_calls"]

        # Section 13.2/13.4: only exact solutions are treated as success in
        # production; approximate solutions are surfaced via status, not path.
        if not solved:
            self.last_planning_status = "timeout"
            return []
        if pdef.hasApproximateSolution():
            self.last_planning_status = "timeout_approximate"
            return []

        self.last_planning_status = "exact_solution"
        path = pdef.getSolutionPath()
        if self.simplify:
            try:
                og.PathSimplifier(si).simplifyMax(path)
            except Exception:
                pass

        q_path = []
        for i in range(path.getStateCount()):
            q = codec.state_to_full_q(path.getState(i))
            if not q_path or not np.allclose(q, q_path[-1], atol=1e-9):
                q_path.append(q)

        # Section 13.3: guarantee the returned path actually reaches goal_q.
        if q_path and not np.allclose(q_path[-1], goal_q, atol=1e-6):
            if not self._check_collision(q_path[-1], goal_q):
                q_path.append(goal_q.copy())
            else:
                self.last_planning_status = "goal_not_connected"
                self.last_failed_q_path = [q.copy() for q in q_path] + [goal_q.copy()]
                return []
        if q_path:
            q_path[0] = start_q.copy()

        verification = self.verify_path(q_path)
        if verification["colliding_edges"] or verification["colliding_waypoints"]:
            self.last_planning_status = "final_verification_failed"
            self.last_collision_pairs = verification.get("collision_pairs") or []
            # This q_path (the one that actually collided) gets thrown away
            # by returning [] - the caller (plan_q_path_for_robot) then falls
            # back to a single-point [q_start] path and re-runs verify_path()
            # on THAT instead, which trivially reports 0 colliding waypoints/
            # edges (nothing to check with only one point and no motion) and
            # silently overwrites this real verification with a meaningless
            # one. Save it here so the caller can use the real breakdown.
            self.last_verification = verification
            self.last_failed_q_path = [q.copy() for q in q_path]
            return []

        self.last_returned_path_reaches_goal = True
        return q_path
