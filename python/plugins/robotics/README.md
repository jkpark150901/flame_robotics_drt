# Robotics Backend

This package isolates robot-library-specific code from viewer, IK checking, and
path planning orchestration.

## Current shape

- `backend.py`
  - Library-neutral dataclasses and `RoboticsBackend` abstract interface.
  - Viewer/planner code should depend on these types.
- `ik_solver.py`
  - Shared IK step implementations.
  - External QP IK dependencies are imported only in this module.
- `pinocchio_backend.py`
  - First concrete backend using Pinocchio and hpp-fcl/coal.

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
`dls`, `normalized_dls`, and `qp`; the backend owns the concrete library calls.

## PyBullet backend plan

A future `pybullet_backend.py` can implement the same interface by mapping:

- URDF loading: `loadURDF`
- FK/frame pose: `resetJointState` + `getLinkState`
- collision: `performCollisionDetection` + `getContactPoints`
- edge collision: interpolate q and call `check_collision`

It should not replace the interface. It should only be another implementation
behind the same `RoboticsBackend` contract.
