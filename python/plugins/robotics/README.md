# Robotics Backend

This package isolates robot-library-specific code from viewer, IK checking, and
path planning orchestration.

## Current shape

- `backend.py`
  - Library-neutral dataclasses and `RoboticsBackend` abstract interface.
  - Viewer/planner code should depend on these types.
- `pinocchio_backend.py`
  - First concrete backend using Pinocchio and hpp-fcl/coal.
  - Also owns the IK step implementations (`damped_least_squares_step`, `solve_qp_ik_step`);
    external QP IK dependencies (pink) are imported only inside `solve_qp_ik_step`.
  - `solver="pybullet"` delegates the IK solve itself to `pybullet_ik.py` (PyBullet's
    native IK, much faster per call than the Python DLS loop) while collision/FK/error
    metrics stay on Pinocchio for consistency with the other solvers.
- `pybullet_ik.py`
  - Optional, faster IK solver. Lazily imports `pybullet`; caches one DIRECT physics
    client and one loaded URDF body per robot (reused across calls, like the
    Pinocchio collision-model cache).

## Backend responsibilities

Every backend should implement:

- `register_robot(description)`
- `joint_names(robot_name)`
- `neutral_q(robot_name)`
- `frame_world_T(robot_name, q, frame_name)`
- `solve_ik(robot_name, target_world_T, q_init, options, frame_name)`
- `configure_collision(robot_name, static_meshes, sample_resolution)`
- `check_collision(robot_name, q, return_pairs)`
- `check_edge_collision(robot_name, q_from, q_to, return_pairs)`

## Why this layer exists

The viewer should not care whether the robot math comes from Pinocchio,
PyBullet, or another solver stack. The UI can expose generic choices such as
`dls` and `qp`; the backend owns the concrete library calls.

## PyBullet backend plan

A future `pybullet_backend.py` can implement the same interface by mapping:

- URDF loading: `loadURDF`
- FK/frame pose: `resetJointState` + `getLinkState`
- collision: `performCollisionDetection` + `getContactPoints`
- edge collision: interpolate q and call `check_collision`

It should not replace the interface. It should only be another implementation
behind the same `RoboticsBackend` contract.
