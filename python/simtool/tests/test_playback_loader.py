import csv

import pytest

from simtool.playback_loader import load_playback_plan_sequence


def _write_joint_states(path, q_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["waypoint", "j0", "j1"])
        for i, q in enumerate(q_path):
            writer.writerow([i, *q])


def _write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "group_name", "robot_name", "pose_name", "status", "message", "n_waypoints",
            "iterations", "max_iter", "solve_time", "iteration_ptc_error"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_playback_plan_sequence_groups_by_group_name(tmp_path):
    run_dir = tmp_path / "RRTConnect_20260101_000000"
    _write_summary(run_dir / "summary.csv", [
        {"index": 0, "group_name": "Point 1 - Inspection pose 1", "robot_name": "dda_rb10_1300e",
         "pose_name": "DDA", "status": "success", "message": "", "n_waypoints": 2,
         "iterations": 5, "max_iter": 5000, "solve_time": 0.01, "iteration_ptc_error": ""},
        {"index": 1, "group_name": "Point 1 - Inspection pose 1", "robot_name": "rb20_1900es",
         "pose_name": "RT", "status": "success", "message": "", "n_waypoints": 2,
         "iterations": 3, "max_iter": 5000, "solve_time": 0.01, "iteration_ptc_error": ""},
    ])
    _write_joint_states(run_dir / "00_dda_rb10_1300e_DDA" / "joint_states.csv", [[0.0, 0.0], [1.0, 2.0]])
    _write_joint_states(run_dir / "01_rb20_1900es_RT" / "joint_states.csv", [[0.0, 0.0], [3.0, 4.0]])

    result = load_playback_plan_sequence(run_dir)

    assert result["n_targets"] == 2
    assert result["n_loaded"] == 2
    assert result["skipped"] == []
    assert result["plan_sequence"] == [{
        "name": "Point 1 - Inspection pose 1",
        "positioner_r_deg": 0.0,
        "plans": {
            "dda_rb10_1300e": {"q_path": [[0.0, 0.0], [1.0, 2.0]], "status": "success"},
            "rb20_1900es": {"q_path": [[0.0, 0.0], [3.0, 4.0]], "status": "success"},
        },
    }]


def test_load_playback_plan_sequence_skips_missing_csv(tmp_path):
    run_dir = tmp_path / "RRTConnect_20260101_000000"
    _write_summary(run_dir / "summary.csv", [
        {"index": 0, "group_name": "A", "robot_name": "dda_rb10_1300e", "pose_name": "DDA",
         "status": "failed", "message": "start_collision", "n_waypoints": 0,
         "iterations": "", "max_iter": "", "solve_time": "", "iteration_ptc_error": ""},
    ])
    # no joint_states.csv written for this target

    result = load_playback_plan_sequence(run_dir)

    assert result["n_targets"] == 1
    assert result["n_loaded"] == 0
    assert result["plan_sequence"] == []
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["robot_name"] == "dda_rb10_1300e"


def test_load_playback_plan_sequence_requires_summary_csv(tmp_path):
    run_dir = tmp_path / "RRTConnect_20260101_000000"
    run_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        load_playback_plan_sequence(run_dir)
