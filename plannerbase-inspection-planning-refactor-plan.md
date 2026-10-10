# PlannerBase Inspection Planning Refactor Plan

## 1. Objective

Move reusable `target_poses`-based path-planning responsibilities from `Visualizer` into `PlannerBase`, including:

- Inspection-group partitioning and sorting
- Single-target robot planning
- Sequential planning across multiple targets for one robot
- Parallel planning across independent robots
- Failure-policy handling
- Timeout and cooperative cancellation
- Executor lifecycle management
- Thread-safe planner and collision-model ownership
- Structured batch results

`Visualizer` should remain responsible for UI state, scene-specific data extraction, visualization, and API response formatting.

## 2. Responsibility Boundaries

### Move to PlannerBase

- The algorithm currently implemented by `_partition_and_sort_inspection_groups`
- The parallel scheduling currently implemented inside `_plan_inspection_group_sequence`
- Sequential propagation of the final joint state between targets
- Generic target-pose transformation helpers
- Planner configuration that does not depend on Visualizer state
- IK and joint-space planning orchestration
- Joint-path-to-TCP-path conversion
- Collision verification and planning timing
- Failure, timeout, and cancellation handling
- Structured planning result construction

### Keep in Visualizer

- `_current_spool_collision_mesh`
- Reading live robot, end-effector, and positioner state
- Building inspection groups from `_inspection_target_groups`
- Determining application-specific group reachability
- Calculating deferred-group positioner rotations
- Transforming scene inputs before job construction
- Vedo actors and path rendering
- Updating `_last_*` visualization state
- ZApi request and response formatting

The existing `_plan_inspection_path_for_robot` should be split rather than moved as one method because it currently mixes planner logic with Visualizer state, logging, rendering, and response bookkeeping.

## 3. Proposed Data Models

```python
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PlanningTarget:
    name: str
    target_pose: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InspectionGroup:
    name: str
    targets_by_robot: dict[str, list[PlanningTarget]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotPlanningJob:
    robot_name: str
    start_q: np.ndarray | None
    targets: list[PlanningTarget]
    obstacle_mesh: object | None = None
    planner_name: str = "rrt_connect"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetPlanningResult:
    target_name: str
    success: bool
    q_path: list[np.ndarray] = field(default_factory=list)
    tcp_path: list[np.ndarray] = field(default_factory=list)
    goal_q: np.ndarray | None = None
    error: str | None = None
    ik_failure: dict[str, Any] | None = None
    verification: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class RobotPlanningResult:
    robot_name: str
    success: bool
    target_results: list[TargetPlanningResult]
    final_q: np.ndarray | None
    error: str | None = None
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class BatchPlanningResult:
    success: bool
    robot_results: dict[str, RobotPlanningResult]
    failures: dict[str, str]
    ik_failures: dict[str, dict[str, Any]]
    wall_elapsed: float
    cancelled: bool = False
    timing: dict[str, float] = field(default_factory=dict)
```

## 4. Group Partitioning and Sorting

Move the partitioning algorithm into `PlannerBase`, but inject the application-specific reachability rule as a callback.

```python
@dataclass
class GroupPartitionResult:
    reachable: list[InspectionGroup]
    deferred: list[InspectionGroup]
    evaluation_errors: dict[str, str] = field(default_factory=dict)


class PlannerBase:
    def partition_and_sort_groups(
        self,
        groups: Sequence[InspectionGroup],
        *,
        is_reachable: Callable[[InspectionGroup], bool],
        reachable_sort_key: Callable[[InspectionGroup], object] | None = None,
        deferred_sort_key: Callable[[InspectionGroup], object] | None = None,
        reachability_error_policy: Literal["defer", "raise"] = "defer",
    ) -> GroupPartitionResult:
        ...
```

The current inspection ordering can be supplied as a domain-specific key:

```python
def inspection_group_sort_key(group: InspectionGroup) -> tuple[float, float]:
    position = np.asarray(group.metadata["rt_position"], dtype=float)
    return float(position[0]), -float(position[2])
```

Design rules:

- `PlannerBase` owns deterministic partitioning and sorting.
- `Visualizer` supplies `is_reachable` because the decision depends on current robot and positioner state.
- Inspection-specific values such as `rt_position` remain metadata instead of becoming hard-coded `PlannerBase` fields.
- A reachability evaluation error either defers the group or propagates immediately according to `reachability_error_policy`.
- Input ordering should be used as a stable tie-breaker.

## 5. Single-Target Planning API

Extract the planning part of `_plan_inspection_path_for_robot` into a planner-layer operation:

```python
class PlannerBase:
    def plan_robot_target(
        self,
        job: RobotPlanningJob,
        target: PlanningTarget,
        start_q: np.ndarray | None,
        *,
        timeout_sec: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> TargetPlanningResult:
        ...
```

This operation should cover:

- Planner and robot-model configuration
- IK candidate generation and validation
- Joint-space path generation
- Joint-path-to-TCP-path conversion
- Collision verification
- Timing collection
- Conversion of expected failures into structured results

Scene extraction, display updates, and API response construction remain outside this method.

## 6. Sequential Planning per Robot

Targets assigned to the same robot have a state dependency and must be planned sequentially.

```python
class PlannerBase:
    def plan_target_sequence(
        self,
        job: RobotPlanningJob,
        *,
        fail_policy: Literal[
            "stop_robot",
            "skip_target",
            "raise",
        ] = "stop_robot",
        timeout_sec: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RobotPlanningResult:
        ...
```

Required behavior:

1. Start with `job.start_q`.
2. Plan targets in declared order.
3. After a success, use `q_path[-1]` as the next target's start state.
4. Preserve every target result, including failures and skipped targets.
5. Apply the selected failure policy without corrupting the last valid joint state.

`skip_target` should continue from the last successful joint state, not from a failed target's tentative state.

## 7. Parallel Batch Planning

Parallelism should occur across independent robot sequences.

```python
class PlannerBase:
    def plan_batch(
        self,
        jobs: Sequence[RobotPlanningJob],
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        fail_policy: Literal[
            "stop_robot",
            "skip_target",
            "stop_all",
            "raise",
        ] = "stop_robot",
        timeout_sec: float | None = None,
        executor: Executor | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> BatchPlanningResult:
        ...
```

Execution model:

```text
Inspection groups: sequential
    Robots inside one group: parallel
        Targets assigned to one robot: sequential
```

This keeps state-dependent motion for one robot ordered while allowing independent robots to plan concurrently.

## 8. Failure Policies

Recommended meanings:

| Policy | Behavior |
| --- | --- |
| `stop_robot` | Stop the failed robot's remaining targets; continue other robots. |
| `skip_target` | Record the failed target and continue that robot from its last successful state. |
| `stop_all` | Signal cancellation after the first failure and stop other work cooperatively. |
| `raise` | Re-raise the first failure after requesting cancellation of unfinished work. |

The default should be `stop_robot`, which provides useful partial results without allowing a failed target to invalidate another robot's independent plan.

## 9. Executor Management

- Use `concurrent.futures.ThreadPoolExecutor` as the initial implementation.
- When `executor` is `None`, `PlannerBase` creates and shuts down the executor.
- When the caller supplies an executor, the caller retains lifecycle ownership.
- Create one executor per batch, not one executor for every target or group entry.
- Limit workers to `min(max_workers, len(jobs))`.
- Do not expose raw `Future` instances in public result objects.
- Avoid `shutdown(wait=False)` as the main cancellation mechanism because running threads continue executing.

A process pool should only be considered after robot models, collision objects, meshes, and planner configuration can be serialized reliably.

## 10. Thread Safety and Collision Models

The default mode should isolate mutable planning state:

```python
collision_model_policy: Literal[
    "clone_per_job",
    "shared_readonly",
] = "clone_per_job"
```

Rules:

- Create a separate planner instance for each concurrently executing robot job.
- Never share mutable Pinocchio `Data`, geometry data, planning deadlines, debug rows, or temporary collision state.
- Clone transformed obstacles, HPP-FCL objects, and positioner collision geometry per job.
- Immutable mesh vertex and triangle arrays may later use `shared_readonly` after thread-safety is verified.
- Prefer a planner factory over copying an already configured planner instance.

Recommended factory contract:

```python
planner_factory: Callable[[RobotPlanningJob], PlannerBase]
```

If `plan_batch` remains an instance method, that instance should act as the coordinator. Worker threads should operate on planner instances returned by `planner_factory`, not concurrently mutate the coordinator.

## 11. Timeout and Cancellation

Cancelling a Python `Future` cannot stop work that is already running. Planning loops therefore need cooperative checks:

```python
self._check_planning_deadline()
cancellation_token.raise_if_cancelled()
```

Checks should occur during:

- IK candidate evaluation
- RRT iterations
- Edge interpolation
- Collision checks
- Transitions between sequential targets

Support separate scopes where practical:

```python
target_timeout_sec: float | None
robot_timeout_sec: float | None
batch_timeout_sec: float | None
```

The effective deadline for an operation should be the earliest applicable deadline. A `stop_all` failure should set the shared cancellation token so running workers can exit at their next check.

## 12. Visualizer Integration

The revised Visualizer flow should be:

```python
partition = coordinator.partition_and_sort_groups(
    groups,
    is_reachable=self._inspection_group_is_reachable_now,
    reachable_sort_key=self._inspection_group_sort_key,
    deferred_sort_key=self._inspection_group_sort_key,
)

for group_set in (partition.reachable, partition.deferred):
    jobs = self._build_robot_planning_jobs(group_set, ...)
    result = coordinator.plan_batch(
        jobs,
        max_workers=request_data.get("max_workers"),
        fail_policy=request_data.get("fail_policy", "stop_robot"),
        timeout_sec=planning_timeout,
    )
    self._render_planning_result(result)
```

For deferred groups, `Visualizer` should still calculate positioner rotation and create the transformed target poses and obstacle input before constructing jobs.

## 13. Target Architecture

```text
Visualizer request handler
    -> Build InspectionGroup objects
    -> PlannerBase.partition_and_sort_groups(...)
    -> Apply scene-specific deferred-group transform
    -> Build RobotPlanningJob objects
    -> PlannerBase.plan_batch(...)
        -> One isolated planner per robot job
        -> Robot jobs execute in parallel
        -> Each robot's targets execute sequentially
        -> Final q propagates to the next target
    -> Render paths and build ZApi response
```

The Visualizer-owned `_plan_inspection_group_sequence` and its direct `ThreadPoolExecutor` usage can be removed after migration.

## 14. Implementation Sequence

1. Add the request, result, and group dataclasses without changing runtime behavior.
2. Move partitioning and stable sorting into `PlannerBase`.
3. Add tests for reachable/deferred partitioning, sorting, ties, and callback errors.
4. Extract single-target planning from `_plan_inspection_path_for_robot`.
5. Implement sequential target planning and final-joint-state propagation.
6. Implement robot-level parallel batch planning and executor ownership rules.
7. Add cooperative cancellation and hierarchical deadlines.
8. Introduce planner factories and collision-model cloning policies.
9. Convert the Visualizer request handler to construct jobs and consume structured results.
10. Remove `_plan_inspection_group_sequence` and obsolete Visualizer executor imports.
11. Add integration tests for complete, partial, failed, timed-out, and cancelled batches.

## 15. Required Tests

- Reachable and deferred groups are partitioned correctly.
- Sorting is deterministic and preserves input order for equal keys.
- Reachability callback errors follow the configured policy.
- Different robots run concurrently.
- Targets for one robot remain sequential.
- A successful target's final `q` becomes the next target's start `q`.
- `stop_robot`, `skip_target`, `stop_all`, and `raise` behave as documented.
- External executors are not shut down by `PlannerBase`.
- Internally owned executors are shut down reliably.
- Target, robot, and batch timeouts use the earliest deadline.
- Cancellation exits IK, planning, and collision loops cooperatively.
- Concurrent jobs do not share mutable Pinocchio or collision state.
- Partial results retain successful paths and structured failure details.
- Existing Visualizer output and ZApi response semantics remain compatible.

## 16. Recommended Initial Scope

For the first implementation, use:

- `ThreadPoolExecutor`
- One planner and collision model per robot job
- Sequential groups
- Parallel robots within each group
- Sequential targets within each robot
- `stop_robot` as the default failure policy
- Cooperative deadline checks using the existing `planning_deadline`
- Structured dataclass results

More aggressive collision-data sharing or process-based parallelism should be treated as later optimizations supported by profiling and thread-safety tests.
