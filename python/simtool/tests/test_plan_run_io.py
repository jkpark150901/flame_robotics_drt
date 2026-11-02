import json
import csv

from plugins.robotics.plan_run_io import classify_failure, save_plan_run
from simtool.playback_loader import load_playback_plan_sequence, resolve_playback_run_dir


def _results():
    return [
        {"group_name": "G1", "robot": "dda_rb10_1300e", "pose_name": "DDA", "status": "success",
         "status_kind": "success", "q_path": [[0.0, 0.0], [1.0, 2.0]], "positioner_r_deg": 0.0,
         "elapsed": 0.5, "planner_stats": {"iterations": 7, "max_iter": 5000, "solve_time": 0.1}},
        # failed but the planner still returned its (colliding) best-effort path
        {"group_name": "G1", "robot": "rb20_1900es", "pose_name": "RT", "status": "failed",
         "status_kind": "failed", "message": "final_verification_failed: edge 3 collides",
         "q_path": [[0.0, 0.0], [3.0, 4.0]], "positioner_r_deg": 0.0, "collision_preview": True,
         "verification": {"edge_collisions": [{"edge": 3, "pairs": [["rt_link_end", "pipe"]]}]}},
        # failed with NO path at all - used to leave no trace on disk
        {"group_name": "G2", "robot": "dda_rb10_1300e", "pose_name": "DDA", "status": "failed",
         "status_kind": "warning", "message": "goal_collision: target pose collides with pipe",
         "q_path": [], "positioner_r_deg": 180.0},
        {"group_name": "G2", "robot": "rb20_1900es", "pose_name": "RT", "status": "failed",
         "status_kind": "failed", "message": "iteration_cap_reached", "q_path": [],
         "positioner_r_deg": 180.0, "planner_stats": {"iterations": 5000, "max_iter": 5000}},
    ]


def test_classify_failure_categories():
    assert classify_failure({"status": "success"}) == ""
    assert classify_failure({"status": "failed", "message": "goal_collision: x"}) == "goal_collision"
    assert classify_failure({"status": "failed", "message": "whatever", "ik_failure": {"x": 1}}) == "ik_failure"
    assert classify_failure({"status": "failed", "message": "OMPL python bindings are not installed"}) == "environment_error"
    # "ik" must not match inside unrelated words
    assert classify_failure({"status": "failed", "message": "quick pick failed"}) == "unknown"


def test_save_then_load_round_trip_keeps_paths_and_failure_causes(tmp_path):
    run_dir = tmp_path / "run"
    saved = save_plan_run(run_dir, _results(), meta={"planner": "rrt_connect", "spool_path": "/x/spool.pcd"},
                          unreached_groups=["G3"])
    assert saved["n_total"] == 4 and saved["n_success"] == 1 and saved["n_failed"] == 3

    # every target has a summary row, including the two with no path
    failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))
    assert failures["by_category"] == {
        "final_verification_failed": 1, "goal_collision": 1, "iteration_cap_reached": 1}
    assert failures["unreached_groups"] == ["G3"]
    no_path = [f for f in failures["failures"] if not f["has_path"]]
    assert {f["category"] for f in no_path} == {"goal_collision", "iteration_cap_reached"}
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["planner"] == "rrt_connect"

    loaded = load_playback_plan_sequence(run_dir)
    assert loaded["n_targets"] == 4 and loaded["n_loaded"] == 2
    g1 = next(g for g in loaded["plan_sequence"] if g["name"] == "G1")
    assert g1["plans"]["dda_rb10_1300e"]["q_path"] == [[0.0, 0.0], [1.0, 2.0]]
    # a failed-but-pathed target loads with its failure info + collision data
    rt = g1["plans"]["rb20_1900es"]
    assert rt["status"] == "failed" and "final_verification_failed" in rt["planning_error"]
    assert rt["edge_collisions"] and rt["collision_preview"] is True
    # targets without a path are reported (with the reason) instead of silently vanishing
    assert len(loaded["skipped"]) == 2
    assert all("planning failed" in item["reason"] for item in loaded["skipped"])
    assert len(loaded["failures"]) == 3
    assert loaded["unreached_groups"] == ["G3"]
    assert loaded["run"]["spool_path"] == "/x/spool.pcd"


def test_load_old_run_without_failures_json_still_reports_failures(tmp_path):
    run_dir = tmp_path / "old"
    save_plan_run(run_dir, _results())
    (run_dir / "failures.json").unlink()
    (run_dir / "run.json").unlink()
    loaded = load_playback_plan_sequence(run_dir)
    assert len(loaded["failures"]) == 3  # derived from summary.csv


def test_saved_success_includes_path_metrics_and_planning_time(tmp_path):
    result = _results()[0]
    result["path_metrics"] = {
        "clearance_min_m": 0.042, "clearance_margin_m": 0.037,
        "smoothness": 1.25, "n_waypoints": 2, "clearance_samples": 12,
        "estimated_motion_s": 3.4, "estimated_ee_mean_speed_m_s": 0.2,
        "joint_velocity_limit_deg_s": 50.0, "linear_velocity_limit_m_s": 0.5,
    }
    result["request_total"] = 0.7
    save_plan_run(tmp_path, [result])
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["planning_time_s"] == "0.5"
    assert row["request_total"] == "0.7"
    assert row["clearance_margin_m"] == "0.037"
    assert row["smoothness"] == "1.25"
    assert row["estimated_motion_s"] == "3.4"
    assert json.loads((tmp_path / "00_dda_rb10_1300e_DDA" / "metrics.json").read_text()) == result["path_metrics"]


def test_resolve_benchmark_parent_to_single_saved_run(tmp_path):
    saved = tmp_path / "result_paths" / "rrt_connect+chomp_r0"
    save_plan_run(saved, [_results()[0]])
    assert resolve_playback_run_dir(tmp_path) == saved
    assert resolve_playback_run_dir(tmp_path / "result_paths") == saved
    assert resolve_playback_run_dir(saved) == saved
