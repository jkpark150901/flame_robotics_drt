import numpy as np
import pytest
import csv

from plugins.pathplanner.direct_path import DirectPath
from plugins.pathplanner.bit_star import BITStar
from plugins.pathplanner.informed_rrt_star import InformedRRTStar
from plugins.pathplanner.rrt_connect import RRTConnect
from plugins.pathplanner.rrt_star import RRTStar


def _configure_test_q_space(planner):
    start_q = np.zeros(6, dtype=float)
    goal_q = np.asarray([0.3, -0.2, 0.15, 0.1, -0.1, 0.2], dtype=float)
    planner.step_size = 0.5
    planner.max_iter = 50
    if hasattr(planner, "search_radius"):
        planner.search_radius = 2.0
    if hasattr(planner, "goal_bias"):
        planner.goal_bias = 1.0
    if hasattr(planner, "goal_check_interval"):
        planner.goal_check_interval = 1
    if hasattr(planner, "early_stop_on_goal"):
        planner.early_stop_on_goal = True
    if hasattr(planner, "debug_exploration"):
        planner.debug_exploration = False
    if hasattr(planner, "debug_convergence"):
        planner.debug_convergence = False
    if hasattr(planner, "debug_solution_paths"):
        planner.debug_solution_paths = False
    planner.configure_joint_space_test_environment(
        dof=6,
        lower_limits=np.full(6, -1.0),
        upper_limits=np.full(6, 1.0),
        collision_fn=lambda q: False,
        sample_fn=lambda: goal_q.copy(),
        sample_resolution=0.2,
    )
    return start_q, goal_q


@pytest.mark.parametrize("planner_cls", [RRTStar, InformedRRTStar, BITStar])
def test_rrt_star_family_defaults_to_joint_space(planner_cls):
    planner = planner_cls()

    assert planner.use_joint_space_planning is True


@pytest.mark.parametrize("planner_cls", [DirectPath, RRTConnect, RRTStar, InformedRRTStar, BITStar])
def test_q_space_planners_accept_raw_q_and_return_q_path(planner_cls):
    planner = planner_cls()
    start_q, goal_q = _configure_test_q_space(planner)

    q_path = planner.generate(start_q, goal_q)

    assert q_path
    assert all(np.asarray(q).shape == start_q.shape for q in q_path)
    assert np.allclose(q_path[0], start_q)
    assert np.allclose(q_path[-1], goal_q)

    verification = planner.verify_path(q_path)
    assert verification["colliding_edges"] == 0
    assert verification["colliding_waypoints"] == 0


def test_q_space_test_environment_reports_collision_pairs():
    planner = DirectPath()
    start_q = np.zeros(2, dtype=float)
    goal_q = np.ones(2, dtype=float)

    def collision_fn(q):
        if q[0] > 0.5:
            return True, [("robot_link", "test_obstacle")]
        return False

    planner.configure_joint_space_test_environment(
        dof=2,
        lower_limits=[0.0, 0.0],
        upper_limits=[1.0, 1.0],
        collision_fn=collision_fn,
        sample_resolution=0.1,
    )

    verification = planner.verify_path([start_q, goal_q])

    assert verification["colliding_edges"] == 1
    assert verification["collision_pairs"] == [["robot_link", "test_obstacle"]]


def test_exploration_log_clamps_negative_start_iteration_to_zero():
    planner = RRTStar()
    planner.debug_exploration = True
    rows = planner._new_exploration_rows()

    planner._record_exploration(rows, iteration=-1, phase="start_collision")

    assert rows[0]["iteration"] == 0


@pytest.mark.parametrize("planner_cls", [DirectPath, RRTConnect, RRTStar, InformedRRTStar, BITStar])
def test_q_space_planners_keep_fixed_joint_at_start_value(planner_cls):
    planner = planner_cls()
    start_q = np.asarray([0.0, 0.4, 0.0], dtype=float)
    goal_q = np.asarray([0.6, -0.6, 0.3], dtype=float)
    planner.step_size = 0.5
    planner.max_iter = 80
    if hasattr(planner, "search_radius"):
        planner.search_radius = 2.0
    if hasattr(planner, "goal_bias"):
        planner.goal_bias = 1.0
    if hasattr(planner, "goal_check_interval"):
        planner.goal_check_interval = 1
    if hasattr(planner, "early_stop_on_goal"):
        planner.early_stop_on_goal = True
    planner.debug_convergence = False
    planner.debug_exploration = False
    planner.configure_joint_space_test_environment(
        dof=3,
        lower_limits=[-1.0, -1.0, -1.0],
        upper_limits=[1.0, 1.0, 1.0],
        collision_fn=lambda q: False,
        sample_fn=lambda: goal_q.copy(),
    )
    planner.configure_fixed_joints(fixed_joint_indices=[1])

    q_path = planner.generate(start_q, goal_q)

    assert q_path
    assert all(np.isclose(np.asarray(q)[1], start_q[1]) for q in q_path)
    assert np.allclose(q_path[-1], np.asarray([goal_q[0], start_q[1], goal_q[2]]))


def test_direct_path_writes_q_and_task_space_convergence_csv(tmp_path):
    planner = DirectPath()
    planner.debug_output_dir = str(tmp_path / "debug")
    planner.debug_convergence = True
    planner.configure_joint_space_test_environment(
        dof=2,
        lower_limits=[-1.0, -1.0],
        upper_limits=[1.0, 1.0],
        collision_fn=lambda q: False,
    )

    planner.generate(np.array([0.0, 0.0]), np.array([0.5, 0.25]))
    q_csv = planner.last_convergence_csv
    assert q_csv is not None

    task_planner = DirectPath()
    task_planner.debug_output_dir = str(tmp_path / "debug")
    task_planner.debug_convergence = True
    task_planner.generate(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, np.nan, np.nan, np.nan]),
    )
    task_csv = task_planner.last_convergence_csv
    assert task_csv is not None

    for csv_path, expected_space in [(q_csv, "q_space"), (task_csv, "task_space")]:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows
        iterations = [int(row["iteration"]) for row in rows]
        assert min(iterations) == 0
        assert all(iteration >= 0 for iteration in iterations)
        assert rows[-1]["space"] == expected_space
        assert rows[-1]["distance_to_goal"] == "0.0"
        assert "best_distance_to_goal" in rows[-1]
        assert "node_count" not in rows[-1]
        assert "cost" not in rows[-1]
        assert "best_cost" not in rows[-1]
        assert "debug" in csv_path.lower()
