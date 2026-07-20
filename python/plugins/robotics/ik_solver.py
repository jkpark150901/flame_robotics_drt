import numpy as np


def _finite_joint_limits(pin_model, q):
    lo = np.asarray(pin_model.lowerPositionLimit, dtype=float).copy()
    hi = np.asarray(pin_model.upperPositionLimit, dtype=float).copy()
    q = np.asarray(q, dtype=float)

    invalid = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
    lo[invalid] = q[invalid] - np.pi
    hi[invalid] = q[invalid] + np.pi
    return lo, hi


def normalized_damped_least_squares_step(
    pin,
    pin_model,
    data,
    q,
    frame_id,
    err,
    damping=1e-3,
    dt=0.35,
):
    """Compute one normalized DLS IK step.

    Args:
        pin: pinocchio module.
        pin_model: Pinocchio model.
        data: Pinocchio data.
        q: Current joint vector.
        frame_id: Target frame id.
        err: 6D error from the current frame to the target frame.
        damping: task-space damping coefficient.
        dt: update gain.

    Returns:
        Next q clamped inside joint limits.

    Formula:
        delta_q = S_q (J S_q).T [(J S_q)(J S_q).T + mu I]^-1 e
    """
    J = pin.computeFrameJacobian(pin_model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
    q = np.asarray(q, dtype=float)
    lo, hi = _finite_joint_limits(pin_model, q)
    span = hi - lo
    span[span < 1e-9] = 1.0

    J_sq = J * span.reshape(1, -1)
    damped_task_matrix = J_sq @ J_sq.T + float(damping) * np.eye(6)
    delta_q = span * (J_sq.T @ np.linalg.solve(damped_task_matrix, err))

    q_next = q + float(dt) * delta_q
    return np.minimum(np.maximum(q_next, lo), hi)


def damped_least_squares_step(
    pin,
    pin_model,
    data,
    q,
    frame_id,
    err,
    damping=1e-3,
    dt=0.35,
):
    """Compute one classic DLS IK step.

    Args:
        pin: pinocchio module.
        pin_model: Pinocchio model.
        data: Pinocchio data.
        q: Current joint vector.
        frame_id: Target frame id.
        err: 6D error from the current frame to the target frame.
        damping: task-space damping coefficient.
        dt: update gain.

    Returns:
        Next q clamped inside joint limits.
    """
    J = pin.computeFrameJacobian(pin_model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
    JJt = J @ J.T
    dq = J.T @ np.linalg.solve(JJt + float(damping) * np.eye(6), err)

    q_next = pin.integrate(pin_model, np.asarray(q, dtype=float), float(dt) * dq)
    return np.minimum(np.maximum(q_next, pin_model.lowerPositionLimit), pin_model.upperPositionLimit)


def solve_qp_ik_step(
    pin_model,
    data,
    q,
    frame_name,
    target_se3,
    dt=0.35,
    solver="quadprog",
):
    """Compute one IK step with the QP backend.

    External QP IK dependencies are imported only here. Viewer/UI code only
    selects the generic `qp` solver and does not depend on backend package names.
    """
    try:
        import pink
        from pink import solve_ik
        from pink.tasks import FrameTask
    except ImportError as exc:
        raise RuntimeError(
            "QP IK backend is not available. Install backend dependencies: "
            "pip uninstall -y pink; pip install pin-pink qpsolvers quadprog"
        ) from exc

    if not hasattr(pink, "Configuration"):
        raise RuntimeError(
            "The installed QP IK backend is not the robotics package expected by this project. "
            "Run: pip uninstall -y pink; pip install pin-pink qpsolvers quadprog"
        )

    configuration = pink.Configuration(pin_model, data, np.asarray(q, dtype=float).copy())
    task = FrameTask(
        frame_name,
        position_cost=1.0,
        orientation_cost=1.0,
        gain=1.0,
    )
    task.set_target(target_se3)
    velocity = solve_ik(configuration, [task], float(dt), solver=solver)
    configuration.integrate_inplace(velocity, float(dt))
    return np.asarray(configuration.q, dtype=float).copy()
