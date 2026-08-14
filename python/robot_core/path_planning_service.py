from __future__ import annotations

import copy
import time

import numpy as np

from util.logger.console import ConsoleLogger

console = ConsoleLogger.get_logger()


def plan_single_target(engine, request):
    """Plan one robot's q-space path from a given start_q to a given target_pose.

    Robot Core no longer knows about target groups, positioner rotation, or
    multi-target sequencing (start_q chaining across targets, first/second
    phase splitting, retreat-to-safe-pose) - all of that orchestration now
    lives in SimTool (see ROBOT_CORE_DECOUPLING_PLAN.md). This function is the
    entire Robot Core planning surface: source_q -> target_pose, one call.

    Args:
        engine: RobotCoreEngine (headless Visualizer subclass) already built
            from the request's snapshot (collision meshes + other robots'
            current joint states, for multi-robot collision avoidance).
        request: dict with:
            robot_name (str, required)
            start_q (list[float], required) - the caller (SimTool, via
                Visualizer) always resolves this explicitly; Robot Core never
                guesses "current" state.
            target_pose (4x4 or 6-vector, required)
            planner (str, default "rrt_connect")
            step_size, max_iter (planner tuning, same meaning as before)
            fixed_joints / fixed_joint_indices / fixed_joint_values (optional)
            planning_timeout (float, optional)
            context_label (str, optional, for logs)
            obstacle_rotation_T (4x4, optional) - if target_pose was resolved
                against a positioner rotation (see
                inspection_workflow.resolve_target_groups_with_rotation), the
                collision obstacle (the pipe) must be rotated to match, or
                start/goal collision checks are validated against a pipe
                position the robot isn't actually moving next to. The caller
                (whoever built target_pose) knows whether a rotation applies
                and must supply the same transform here.

    Returns:
        {"result": {...}, "q_path": [...]} - result.status is "success" or
        "failed"; q_path is the full raw-q waypoint list on success, [].
    """
    started = time.perf_counter()
    robot_name = request.get("robot_name")
    start_q = request.get("start_q")
    target_pose = request.get("target_pose")
    if not robot_name:
        raise ValueError("robot_name is required")
    if start_q is None:
        raise ValueError("start_q is required")
    if target_pose is None:
        raise ValueError("target_pose is required")

    context_label = request.get("context_label") or robot_name

    try:
        obstacle_mesh = engine._current_spool_collision_mesh()
        if obstacle_mesh is None:
            raise RuntimeError("collision scene is not available")
        obstacle_rotation_T = request.get("obstacle_rotation_T")
        if obstacle_rotation_T is not None:
            obstacle_mesh = copy.deepcopy(obstacle_mesh)
            obstacle_mesh.transform(np.asarray(obstacle_rotation_T, dtype=float))

        target_request = dict(request)
        target_request["_start_q_override_by_robot"] = {
            robot_name: np.asarray(start_q, dtype=float).tolist()
        }

        plan = engine._plan_inspection_path_for_robot(
            target_request,
            robot_name,
            np.asarray(target_pose, dtype=float),
            obstacle_mesh,
            context_label=context_label,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        # start_collision/goal_collision are an expected, routine outcome of
        # sweeping many targets (some just aren't reachable from wherever
        # this robot currently is, or the target pose itself sits too close
        # to the pipe) - not a crash. Logged as a warning, not an error, and
        # tagged "warning" (vs "failed" for anything else) so a caller
        # summarizing a whole sequence's results (InspectionSequencer) can
        # tell "expected, tolerated miss" apart from a real failure.
        reason = str(exc)
        is_expected_collision = "start_collision" in reason or "goal_collision" in reason
        if is_expected_collision:
            console.warning(f"plan_single_target: start/goal collision after {elapsed:.3f}s: {exc}")
        else:
            console.error(f"plan_single_target failed after {elapsed:.3f}s: {exc}")
        return {
            "result": {
                "status": "failed",
                "status_kind": "warning" if is_expected_collision else "failed",
                "message": str(exc),
                "elapsed": elapsed,
                # Some failures (e.g. "planning failed for target: ..." from
                # _plan_inspection_path_for_robot's TCP-path-conversion
                # check) raise instead of returning a normal plan dict, which
                # would otherwise leave this permanently empty - see where
                # that exception is raised for why it's attached here.
                "planner_stats": dict(getattr(exc, "planner_stats", None) or {}),
                # Which waypoint/edge index (into "q_path" below) actually
                # collided, if the planner determined one (final_verification_
                # failed) - see OMPLPlannerBase.last_verification/plannerbase.
                # verify_path()'s waypoint_collisions/edge_collisions shape.
                "verification": dict(getattr(exc, "verification", None) or {}),
            },
            # The colliding path itself, if the planner captured one before
            # discarding it (see where visualizer.py's "planning failed for
            # target" exception attaches exc.q_path) - lets a failed result
            # still be played back / inspected to see which waypoint or edge
            # actually collided, instead of only a status string.
            "q_path": list(getattr(exc, "q_path", None) or []),
        }

    q_path = plan.get("q_path") or []
    # collision_preview=True means the returned q_path is *not actually
    # collision-free* (e.g. an optimizer - STOMP et al - gave up and
    # returned its best-effort invalid path, or the raw planner's path
    # failed final verification but wasn't discarded via an exception this
    # time). Without checking it here, such a result was reported as
    # "success" with a genuinely colliding q_path - silently wrong for any
    # consumer (benchmark success-rate stats, SimTool playback, etc).
    success = (
        bool(q_path)
        and not bool(plan.get("ik_failure"))
        and not bool(plan.get("collision_preview"))
    )
    result = {
        "status": "success" if success else "failed",
        "message": plan.get("planning_error") or (
            plan.get("collision_preview_reason") if plan.get("collision_preview") else None
        ),
        "ik_failure": plan.get("ik_failure"),
        "elapsed": float(plan.get("elapsed", time.perf_counter() - started)),
        "robot": engine._inspection_plan_result_for_robot(plan),
        # iterations/solve_time/max_iter/timeout_sec/state_validity_calls/
        # collision_rejects for OMPL-backed planners (empty for legacy ones -
        # see OMPLPlannerBase._generate_joint_space's last_ompl_stats).
        "planner_stats": plan.get("planner_stats") or {},
    }
    return {
        "result": result,
        "q_path": [np.asarray(q, dtype=float).tolist() for q in q_path],
    }
