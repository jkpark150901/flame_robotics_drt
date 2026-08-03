import numpy as np

from plugins.optimizer.gpmp2 import GPMP2
from plugins.pathplanner.direct_path import DirectPath


def test_direct_pose_path_can_be_optimized_with_gpmp2():
    planner = DirectPath()
    planner.step_size = 0.25

    start_pose = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    goal_pose = np.asarray([1.0, 0.5, 0.25, 0.2, -0.1, 0.3], dtype=float)

    seed_path = planner.generate(start_pose, goal_pose)
    optimizer = GPMP2()
    optimizer.num_iterations = 5
    optimizer.verbose = False

    optimized_path = optimizer.optimize(seed_path, planner)

    assert len(seed_path) >= 3
    assert len(optimized_path) == len(seed_path)
    assert all(np.asarray(p).shape == start_pose.shape for p in optimized_path)
    assert np.allclose(optimized_path[0], start_pose)
    assert np.allclose(optimized_path[-1], goal_pose)
    assert optimizer.last_optimization_status in {
        "success",
        "optimizer_not_converged",
        "scipy_unavailable",
    }

    verification = planner.verify_path(optimized_path)
    assert verification["colliding_edges"] == 0
    assert verification["colliding_waypoints"] == 0


def test_direct_pose_path_keeps_dont_care_orientation_before_gpmp2():
    planner = DirectPath()
    planner.step_size = 0.5

    start_pose = np.asarray([0.0, 0.0, 0.0, 0.1, 0.2, 0.3], dtype=float)
    goal_pose = np.asarray([1.0, 0.0, 0.0, np.nan, -0.2, np.nan], dtype=float)

    seed_path = planner.generate(start_pose, goal_pose)
    optimized_path = GPMP2().optimize(seed_path, planner)

    expected_goal = np.asarray([1.0, 0.0, 0.0, 0.1, -0.2, 0.3], dtype=float)
    assert np.allclose(seed_path[-1], expected_goal)
    assert np.allclose(optimized_path[-1], expected_goal)
