"""SimTool-owned orchestration for multi-target inspection path planning.

Robot Core's interface is now limited to a single source_q -> target_pose
plan per request (robot_core.service.OPERATION_PLAN_SINGLE_TARGET - see
ROBOT_CORE_DECOUPLING_PLAN.md). This module owns what used to live in
robot_core.path_planning_service.InspectionPathPlanningService: splitting/
sorting target groups into phases, deciding which groups are reachable
without a positioner rotation, and chaining each robot's start_q from one
target to the next as replies come back over ZAPI.

Both phases are automated: the "reachable" phase (no positioner rotation
needed) runs first; if rotation-needed groups remain once it finishes, this
sequencer asks the Viewer to retreat the involved robots to a safe posture
and rotate the positioner (prepare_next_inspection_phase - see
Visualizer._handle_request_prepare_next_inspection_phase, which does the
actual zero_non_linear_track_joints/_move_positioner_r_to work since it's
the side with robot-model/joint-name access), then continues dispatching
the rotated-phase's jobs the same way as the first phase.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

from plugins.robotics.inspection_workflow import (
    inspection_group_pose_items,
    partition_and_sort_target_groups,
)

# _ZAPI_request_plan_single_target's accepted planner-tuning kwargs.
_PLANNER_OPTION_KEYS = (
    "step_size", "max_iter", "planning_timeout",
    "fixed_joints", "fixed_joint_indices", "fixed_joint_values",
    "ik_solver", "ik_normalize", "optimizer", "optimize_path",
    "lock_linear_track",
)


def _build_jobs(groups):
    return [
        {
            "group_name": group.get("name"), "robot": robot_name,
            "pose_name": pose_name, "target_pose": target_T.tolist(),
        }
        for group in groups
        for robot_name, pose_name, target_T in inspection_group_pose_items(group)
    ]


class InspectionSequencer:
    """Runs one target at a time via plan_single_target, chaining start_q."""

    def __init__(self, console):
        self._console = console
        self._zapi = None
        self._jobs: List[Dict[str, Any]] = []
        self._job_index = 0
        self._pending_request_id: Optional[str] = None
        self._start_q_by_robot: Dict[str, list] = {}
        self._results: List[Dict[str, Any]] = []
        self._deferred_groups: List[Dict[str, Any]] = []
        self._all_robots: List[str] = []
        self._planner = "rrt_connect"
        self._planner_options: Dict[str, Any] = {}
        self._on_progress: Optional[Callable[[str], None]] = None
        self._on_finished: Optional[Callable[[dict], None]] = None
        self._active = False
        self._total_job_count = 0
        # Positioner angle (deg) the CURRENT batch of self._jobs was planned
        # against - 0.0 for the reachable phase, updated once
        # prepare_next_inspection_phase's reply reports the rotated angle.
        # Recorded per-result so to_plan_sequence() can give each group its
        # actual planned-against angle instead of assuming 0.0 for everyone.
        self._current_phase_r_deg = 0.0
        self._r_deg_delta = 180.0
        self._preparing_phase = False
        self._pending_phase_request_id: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._active

    def start(self, zapi, target_groups, *, planner, planner_options=None,
              rt_pipe_facing_axis=(0.0, -1.0, 0.0),
              positioner_second_group_r_deg: float = 180.0,
              on_progress=None, on_finished=None):
        if zapi is None:
            raise RuntimeError("ZAPI is not available")
        if not target_groups:
            raise RuntimeError("EF poses are not determined")
        if self._active:
            # Don't block a retry on a wedged/still-in-flight previous sequence
            # (e.g. robot core died mid-request and never replied). Starting a
            # new one supersedes it - _pending_request_id is about to be
            # overwritten by the first dispatch below, so any late reply for
            # the old request_id will simply fail the on_reply() match in
            # _pending_request_id and be dropped as stale.
            self._console.warning(
                "InspectionSequencer: starting a new sequence while a previous "
                f"one (request_id={self._pending_request_id}) is still pending - "
                "its reply (if it ever arrives) will be ignored as stale")

        phases = partition_and_sort_target_groups(
            target_groups, rt_pipe_facing_axis=rt_pipe_facing_axis)
        reachable_groups = phases[0]["groups"]
        self._deferred_groups = phases[1]["groups"]
        # Every robot with ANY target anywhere in this sequence (both
        # phases) - used for the retreat-before-rotation robot list (see
        # _advance_or_finish). NOT just the robots with a target in the
        # rotation-needed phase: the positioner physically rotates near
        # every robot in the scene regardless of which one has the next
        # target, so a robot with no phase-2 target still needs to be
        # retreated to a safe pose before the rotation, not left wherever
        # its last phase-1 target put it.
        self._all_robots = sorted({
            robot_name
            for group in target_groups
            for robot_name, _pose_name, _target_T in inspection_group_pose_items(group)
        })

        self._jobs = _build_jobs(reachable_groups)
        self._zapi = zapi
        self._job_index = 0
        self._start_q_by_robot = {}
        self._results = []
        self._total_job_count = len(self._jobs)
        self._current_phase_r_deg = 0.0
        self._r_deg_delta = float(positioner_second_group_r_deg)
        self._preparing_phase = False
        self._pending_phase_request_id = None
        self._planner = str(planner or "rrt_connect")
        options = dict(planner_options or {})
        self._planner_options = {k: v for k, v in options.items() if k in _PLANNER_OPTION_KEYS}
        unknown = set(options) - set(_PLANNER_OPTION_KEYS)
        if unknown:
            self._console.warning(f"InspectionSequencer: ignoring unsupported option(s): {unknown}")
        self._on_progress = on_progress
        self._on_finished = on_finished
        self._active = True

        if self._deferred_groups:
            self._console.info(
                f"InspectionSequencer: {len(self._deferred_groups)} group(s) need a "
                "positioner rotation - will be planned automatically after the "
                f"reachable phase: {[g.get('name') for g in self._deferred_groups]}")

        if not self._jobs:
            self._advance_or_finish()
            return
        self._dispatch_next()

    def _dispatch_next(self):
        job = self._jobs[self._job_index]
        request_id = uuid.uuid4().hex
        self._pending_request_id = request_id
        if self._on_progress:
            self._on_progress(
                f"[{self._job_index + 1}/{len(self._jobs)}] "
                f"{job['group_name']}:{job['pose_name']} ({job['robot']})")
        start_q = self._start_q_by_robot.get(job["robot"])
        self._console.info(
            "InspectionSequencer: dispatch "
            f"{job['group_name']}:{job['pose_name']} ({job['robot']}) "
            f"request_id={request_id} "
            f"start_q={'(live pose)' if start_q is None else [round(v, 5) for v in start_q]} "
            f"target_pose={job['target_pose']}")
        self._zapi._ZAPI_request_plan_single_target(
            robot=job["robot"],
            target_pose=job["target_pose"],
            planner=self._planner,
            request_id=request_id,
            start_q=self._start_q_by_robot.get(job["robot"]),
            context_label=f"{job['group_name']}:{job['pose_name']}",
            **self._planner_options,
        )

    def on_reply(self, payload: dict) -> bool:
        """Feed a reply_plan_single_target payload in. Returns True if it was
        consumed (matched the currently pending request)."""
        if not self._active:
            self._console.warning(
                "InspectionSequencer: dropping reply_plan_single_target - no sequence is "
                f"active (request_id={payload.get('request_id')}) - if a sequence was "
                "expected to still be running, this reply never advanced/finished it")
            return False
        if payload.get("request_id") != self._pending_request_id:
            # Silent before this log existed - a mismatched/late/duplicate
            # reply was just dropped with no trace, so if it happened to be
            # the LAST job's reply, _job_index never reached len(self._jobs),
            # _advance_or_finish()/_finish() never ran, and the sequence sat
            # "finished" from every individual target's own log but with no
            # final [OK]/[FAIL] summary and no error anywhere to explain why.
            self._console.warning(
                "InspectionSequencer: dropping reply_plan_single_target - request_id mismatch "
                f"(got {payload.get('request_id')}, expected {self._pending_request_id}) - "
                f"job {self._job_index + 1}/{len(self._jobs)} is still pending and will never "
                "be retried; the sequence is now stuck with no final summary unless a matching "
                "reply eventually arrives")
            return False

        job = self._jobs[self._job_index]
        status = payload.get("status")
        q_path = payload.get("q_path") or []
        # status_kind (path_planning_service.py) distinguishes an expected/
        # tolerated start_collision or goal_collision ("warning" - this
        # target just wasn't reachable from here, nothing else is wrong)
        # from any other failure ("failed") - see summary()/_finish().
        status_kind = payload.get("status_kind") or ("success" if status == "success" else "failed")
        self._results.append({
            "group_name": job["group_name"], "robot": job["robot"],
            "pose_name": job["pose_name"], "status": status, "status_kind": status_kind,
            "message": payload.get("message"), "elapsed": payload.get("elapsed"),
            "q_path": q_path, "positioner_r_deg": self._current_phase_r_deg,
            # Which waypoint/edge actually collided (OMPLPlannerBase.
            # verify_path()'s shape) - carried through so to_plan_sequence()
            # can feed it to the Viewer's existing collision-pair-highlight
            # playback machinery (_warn_collision_preview_playback/
            # _highlight_collision_pairs), not just animate the path blindly.
            "verification": payload.get("verification") or {},
        })
        if status == "success" and q_path:
            self._console.info(
                "InspectionSequencer: result "
                f"{job['group_name']}:{job['pose_name']} ({job['robot']}) "
                f"request_id={self._pending_request_id} status={status} "
                f"start_q={[round(v, 5) for v in q_path[0]]} "
                f"goal_q={[round(v, 5) for v in q_path[-1]]} n_waypoints={len(q_path)}")
            self._start_q_by_robot[job["robot"]] = q_path[-1]
        else:
            # Continue past a failed target instead of aborting the whole
            # sequence - a single unreachable/colliding pose used to stop
            # every remaining target (including in the OTHER phase) from
            # ever being planned, which is what "target pose 전체에 대해
            # plan이 안 나옴" traced back to. start_q_by_robot is left
            # untouched (this robot didn't actually move), so the next job
            # for the same robot still chains from wherever it last
            # succeeded - matching test_ompl_planning.py's per-target
            # try/except loop instead of a planner.generate()-style
            # all-or-nothing sequence.
            self._console.warning(
                "InspectionSequencer: target failed, continuing with remaining targets: "
                f"{job['group_name']}:{job['pose_name']} ({job['robot']}) - "
                f"{payload.get('message')}")

        self._job_index += 1
        if self._job_index >= len(self._jobs):
            self._advance_or_finish()
            return True
        self._dispatch_next()
        return True

    def _advance_or_finish(self):
        """Called once the current batch of jobs (self._jobs) has all
        succeeded (or was empty to begin with). If a rotation-needed phase is
        still waiting, kick off retreat+rotate; otherwise the sequence is
        actually done."""
        if not self._deferred_groups:
            self._finish()
            return
        next_groups = self._deferred_groups
        # Unconditionally retreat every robot in the sequence, not just the
        # ones with a target in next_groups - see start()'s _all_robots
        # comment for why a robot with no phase-2 target still needs to be
        # folded to a safe pose before the positioner rotates near it.
        robots = list(self._all_robots) or sorted({
            robot_name
            for group in next_groups
            for robot_name, _pose_name, _target_T in inspection_group_pose_items(group)
        })
        request_id = uuid.uuid4().hex
        self._pending_phase_request_id = request_id
        self._preparing_phase = True
        if self._on_progress:
            self._on_progress(
                f"Retreating {robots} and rotating positioner "
                f"(+{self._r_deg_delta:.0f}deg) for next phase...")
        self._console.info(
            f"InspectionSequencer: preparing next phase - retreating {robots}, "
            f"rotating positioner by {self._r_deg_delta:+.1f}deg, request_id={request_id}")
        self._zapi._ZAPI_request_prepare_next_inspection_phase(
            robots=robots, r_deg_delta=self._r_deg_delta, request_id=request_id)

    def on_phase_prepared(self, payload: dict) -> bool:
        """Feed a reply_prepare_next_inspection_phase payload in. Returns
        True if it was consumed (matched the currently pending phase-prep
        request)."""
        if not self._active or not self._preparing_phase \
                or payload.get("request_id") != self._pending_phase_request_id:
            # See on_reply()'s matching warning - a dropped/mismatched reply
            # here means the retreat+rotate step never completes and phase 2
            # is never dispatched, with nothing but silence to show for it.
            self._console.warning(
                "InspectionSequencer: dropping reply_prepare_next_inspection_phase - "
                f"active={self._active}, preparing_phase={self._preparing_phase}, "
                f"request_id got={payload.get('request_id')} "
                f"expected={self._pending_phase_request_id} - phase 2 will never be "
                "dispatched unless a matching reply eventually arrives")
            return False
        self._preparing_phase = False

        if payload.get("status") != "success":
            self._console.warning(
                "InspectionSequencer: prepare_next_inspection_phase failed, stopping sequence - "
                f"{payload.get('message')}")
            self._finish()
            return True

        start_q_by_robot = payload.get("start_q_by_robot") or {}
        self._start_q_by_robot.update(start_q_by_robot)
        self._current_phase_r_deg = float(payload.get("positioner_r_deg", self._current_phase_r_deg))
        self._console.info(
            f"InspectionSequencer: phase prepared - positioner now at "
            f"{self._current_phase_r_deg:.1f}deg, retreated start_q for {list(start_q_by_robot)}")

        next_groups = self._deferred_groups
        self._deferred_groups = []
        # Swap in the Viewer's post-rotation target groups (see
        # _handle_request_prepare_next_inspection_phase's "target_groups" -
        # SimTool's own next_groups here is a stale, never-rotated snapshot
        # from EF pose determination time; without this substitution, phase-2
        # plan_single_target requests would carry pre-rotation target poses
        # against a collision scene whose pipe/positioner has actually
        # rotated - a real geometric mismatch, not just a display issue).
        rotated_by_name = {
            g.get("name"): g for g in (payload.get("target_groups") or [])
        }
        next_groups = [rotated_by_name.get(g.get("name"), g) for g in next_groups]
        self._jobs = _build_jobs(next_groups)
        self._job_index = 0
        self._total_job_count += len(self._jobs)

        if not self._jobs:
            self._finish()
            return True
        self._dispatch_next()
        return True

    def _human_readable_lines(self):
        """One line per planned target - "<robot> <pose> -> success/warning/
        failed[: reason]" - so the outcome of a whole sequence is readable
        at a glance without scrolling back through every dispatch/result log
        line above it."""
        icons = {"success": "OK", "warning": "WARN", "failed": "FAIL"}
        lines = []
        for r in self._results:
            kind = r.get("status_kind", "failed")
            line = f"[{icons.get(kind, kind.upper())}] {r['robot']} {r['group_name']}:{r['pose_name']}"
            if kind != "success":
                line += f" -> {r.get('message')}"
            lines.append(line)
        for name in [g.get("name") for g in self._deferred_groups]:
            lines.append(f"[SKIP] {name} -> never reached (positioner-rotation prep failed or sequence stopped)")
        return lines

    def _finish(self):
        self._active = False
        self._preparing_phase = False
        n_ok = sum(1 for r in self._results if r["status"] == "success")
        summary_lines = self._human_readable_lines()
        status = (
            "success" if n_ok == self._total_job_count and self._total_job_count
            else "partial" if n_ok else "failed"
        )
        summary = {
            "status": status,
            "results": self._results,
            "deferred_groups": [g.get("name") for g in self._deferred_groups],
            "n_planned": n_ok,
            "n_total": self._total_job_count,
            "summary_lines": summary_lines,
        }
        # Log the summary here unconditionally, THEN hand it to on_finished
        # (window.py's __on_inspection_sequence_finished, which prints its
        # own copy - a deliberate duplicate now, not a bug). This used to be
        # the only place the summary was ever logged; if on_finished was
        # never wired up, threw partway through, or its output went
        # somewhere the user wasn't looking, the whole sequence went totally
        # silent at the very end with no error anywhere to explain why -
        # exactly the "no summary, no error" symptom this was chasing.
        self._console.info(
            f"InspectionSequencer: sequence {status} - {n_ok}/{self._total_job_count} planned")
        for line in summary_lines:
            self._console.info(line)
        on_finished, self._on_finished = self._on_finished, None
        if on_finished:
            try:
                on_finished(summary)
            except Exception as exc:
                self._console.error(f"InspectionSequencer: on_finished callback failed: {exc!r}")

    def to_plan_sequence(self):
        """Convert self._results (flat per-target list) into the group-keyed
        "plan_sequence" shape Visualizer._start_inspection_sequence_path_
        playback() expects: [{"name", "positioner_r_deg", "plans": {robot:
        {"q_path": [...], "status": ..., "status_kind": ...}}}, ...], in
        group order. Feed the return value into InspectionPathHandler.
        accept_result({"plan_sequence": ...}) to wire up "Start Simulation"
        playback for whatever this sequencer planned.

        Failed targets are included too, as long as they carry a q_path -
        goal_collision/start_collision/goal_not_connected/
        final_verification_failed all attach the actual (colliding) q_path
        the planner found (see OMPLPlannerBase.last_failed_q_path and
        plan_single_target's except-block in path_planning_service.py), so
        playing it back is exactly how to SEE where/what it collided with,
        instead of only reading a text reason in the log. Only a result with
        an empty q_path (e.g. iteration_cap_reached/timeout/planner_failed -
        solve() never found anything, not even an invalid one, to report) is
        skipped, since there's nothing to play back. A group missing a robot
        here (never reached at all) just won't move that robot during
        playback for that group.

        positioner_r_deg comes from each result's own recorded value (0.0 for
        the reachable phase, the rotated angle for a rotation-needed phase -
        see on_reply()/on_phase_prepared()), so a group planned after
        rotation plays back with the positioner shown at the angle it was
        actually planned against.
        """
        order = []
        groups = {}
        for r in self._results:
            if not r.get("q_path"):
                continue
            name = r["group_name"]
            if name not in groups:
                groups[name] = {
                    "name": name, "positioner_r_deg": float(r.get("positioner_r_deg", 0.0)),
                    "plans": {},
                }
                order.append(name)
            verification = r.get("verification") or {}
            is_failure = r.get("status") != "success"
            groups[name]["plans"][r["robot"]] = {
                "q_path": r["q_path"],
                "status": r.get("status"),
                "status_kind": r.get("status_kind"),
                "verification": verification,
                # Feeds Visualizer._inspection_plan_collision_reason()/
                # _warn_collision_preview_playback(), the SAME machinery a
                # collision_preview=True optimizer result already uses to
                # highlight the actual colliding geometry pairs during
                # playback (_highlight_collision_pairs) instead of just
                # animating the path with no visual indication of what
                # went wrong.
                "collision_preview": is_failure,
                "edge_collisions": verification.get("edge_collisions") or [],
                "planning_error": r.get("message") if is_failure else None,
            }
        return [groups[name] for name in order]
