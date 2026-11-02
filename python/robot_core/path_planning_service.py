from __future__ import annotations

import copy
import logging
import re
import time

import numpy as np

from util.logger.console import ConsoleLogger

console = ConsoleLogger.get_logger()
# matplotlib's colorbar/tick DEBUG lines (IK experiment plots) drown the planning log
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _plan_single_target_impl(engine, request):
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
                "edge_collisions": list((getattr(exc, "verification", None) or {}).get("edge_collisions", [])),
                "collision_preview": True,
                # collision-free IK goal of the failed target (None if there is none) - see InspectionSequencer
                "goal_q": getattr(exc, "goal_q", None),
                # the IK goal even when it collides (playback / visualization of the failed target)
                "ik_goal_q": getattr(exc, "ik_goal_q", None),
            },
            # The colliding path itself, if the planner captured one before
            # discarding it (see where visualizer.py's "planning failed for
            # target" exception attaches exc.q_path) - lets a failed result
            # still be played back / inspected to see which waypoint or edge
            # actually collided, instead of only a status string.
            "q_path": list(getattr(exc, "q_path", None) or []),
            "diag": {"ik": getattr(exc, "ik_multi_start", None), "timing": getattr(exc, "timing", None)},
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
        # DDA 수직 진입을 쓴 경우 구간 인덱스/수직도/clearance 수치(아니면 None) - 벤치마크 지표용.
        "approach": plan.get("approach"),
        # Which waypoint/edge actually collided (plannerbase.verify_path's
        # shape) and whether the returned q_path is a known-colliding best-
        # effort result - previously dropped here even though `plan` already
        # has them (visualizer.py always sets them), so nothing downstream
        # (a live "preview" collision warning, a saved benchmark run) could
        # ever tell a genuinely clean path from a colliding one without
        # re-deriving it. See simtool/window.py's playback-result collision
        # warning and benchmark_path_planners.py's --save-paths, both of
        # which now persist/reuse this instead of going silent.
        "verification": plan.get("verification") or {},
        "edge_collisions": plan.get("edge_collisions") or [],
        "collision_preview": bool(plan.get("collision_preview")),
        "goal_q": (None if plan.get("goal_q") is None
                   else np.asarray(plan.get("goal_q"), dtype=float).tolist()),
    }
    if success:
        try:
            from plugins.robotics.path_metrics import planning_path_summary
            result["path_metrics"] = planning_path_summary(engine, robot_name, q_path)
        except Exception as exc:
            result["path_metrics_error"] = f"{type(exc).__name__}: {exc}"
            console.warning(f"plan_single_target: path metric summary failed for {robot_name}: {exc}")
    return {
        "result": result,
        "q_path": [np.asarray(q, dtype=float).tolist() for q in q_path],
        "diag": {"ik": plan.get("ik_multi_start"), "timing": plan.get("timing"), "ik_result": plan.get("ik_result")},
    }


def _format_result_summary(request, output):
    """One readable line per target: outcome, failure category + reason, IK search, planner counters, timing."""
    from plugins.robotics.plan_run_io import classify_failure

    result = output.get("result") or {}
    diag = output.get("diag") or {}
    ok = result.get("status") == "success"
    label = request.get("context_label") or request.get("robot_name")
    parts = [f"[PLAN {'OK' if ok else 'FAIL'}] {label} ({request.get('robot_name')})"]
    if request.get("positioner_r_deg") is not None:
        parts.append(f"positioner_r={float(request['positioner_r_deg']):.0f}deg")
    if not ok:
        message = str(result.get("message") or "")
        pairs = re.search(r"pairs=(\[\[.*?\]\])", message)
        reason = re.sub(r"\(pairs=.*\)$", "", message).replace("planning failed for target: ", "")
        category = classify_failure(result) or 'unknown'
        parts.append(category if reason.strip() in (category, '') else f"{category}: {reason}")
        if pairs:
            parts.append("colliding pairs " + pairs.group(1).replace("'", "").replace("_0", ""))
    parts.append(f"waypoints={len(output.get('q_path') or [])}")
    ik = diag.get("ik")
    if ik:
        text = (f"IK seeds={ik.get('n_seeds')} converged={ik.get('n_converged')} "
                f"collision_free={ik.get('n_collision_free')}")
        if ik.get("n_side_ok") is not None:
            text += f" own_side={ik.get('n_side_ok')}"
        if ik.get("n_refined"):
            text += f" refined={ik.get('n_refined_ok')}/{ik.get('n_refined')}"
        if ik.get("chosen_clearance") is not None:
            text += f" clearance={float(ik['chosen_clearance']) * 1000:.0f}mm"
        parts.append(text)
    elif (diag.get("ik_result") or {}).get("collision"):
        parts.append("IK goal collides")
    stats = result.get("planner_stats") or {}
    if stats.get("iterations") is not None:
        parts.append(f"planner iterations={stats.get('iterations')}/{stats.get('max_iter')} "
                     f"solve={float(stats.get('solve_time') or 0.0):.1f}s")
    timing = diag.get("timing") or {}
    stage = ", ".join(f"{k}={float(timing[k]):.1f}s" for k in
                      ("target_setup", "planner_setup", "ik_target_setup", "ik",
                       "ik_result_check", "planning", "optimization",
                       "collision_verification", "path_conversion") if k in timing)
    parts.append(f"plan_elapsed={float(result.get('elapsed') or 0.0):.1f}s")
    parts.append(f"request_total={float(result.get('request_total') or 0.0):.1f}s"
                 + (f" ({stage})" if stage else ""))
    return " | ".join(parts)


def plan_single_target(engine, request):
    """plan_single_target() + one summary log line for the target (see _format_result_summary)."""
    started = time.perf_counter()
    output = _plan_single_target_impl(engine, request)
    output["result"]["request_total"] = time.perf_counter() - started
    try:
        line = _format_result_summary(request, output)
        (console.info if (output.get("result") or {}).get("status") == "success" else console.warning)(line)
    except Exception as exc:  # never let logging break a plan
        console.debug(f"plan summary log failed: {exc}")
    return output
