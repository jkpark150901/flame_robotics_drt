"""Plan a snapshot (benchmark flow, safe start posture) and save every target's IK goal with its collision pairs.

The planning result of a failed target only keeps the start posture, so the colliding goal configuration is not available
afterwards. This wraps InspectionPlanningBase.check_inspection_ik_for_robot to record it; render_collision_failures.py draws
the records with vedo.

    python python/capture_ik_failures.py --snapshot debug/snapshots/planning_snapshot_new_urdf.pkl --out debug/ik_goals.pkl
"""
import argparse
import pathlib
import pickle
import sys

import numpy as np

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from common.config_loader import load_config
from util.logger.console import ConsoleLogger
import benchmark_path_planners as bench
from plugins.robotics.inspection_planning_base import InspectionPlanningBase
from plugins.robotics.path_metrics import safe_start_joint_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(pathlib.Path(__file__).with_name("viewervedo.cfg")))
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--planner", default="rrt_connect")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ik-solver", default="collision_free")
    a = ap.parse_args()
    config = load_config(a.config)
    config.update(load_config(pathlib.Path(a.config).resolve().parent / "path_planning.cfg"))
    config["root_path"] = ROOT_PATH
    config["verbose_level"] = "WARNING"
    config.setdefault("inspection_ik", {})["solver"] = a.ik_solver
    ConsoleLogger.configure(config.get("logging", {}) or {}, force=True)
    snapshot = bench._load_snapshot(a.snapshot, ConsoleLogger.get_logger())

    from robot_core.worker import RobotCoreEngine
    probe = RobotCoreEngine(config, snapshot)
    robots = [str(m.name) for m in probe._robot_models if "positioner" not in str(m.name)]
    states = safe_start_joint_states(probe, snapshot, robots)
    snapshot = dict(snapshot)
    merged = {k: dict(v) for k, v in (snapshot.get("robot_joint_states") or {}).items()}
    for r, j in states.items():
        merged.setdefault(r, {}).update(j)
    snapshot["robot_joint_states"] = merged

    records = []
    original = InspectionPlanningBase.check_inspection_ik_for_robot

    def wrapped(self, request):
        out = original(self, request)
        records.append({
            "robot": request.robot_name, "start_q": np.asarray(request.start_q, float).tolist(),
            "goal_q": np.asarray(out["goal_q"], float).tolist(), "pose": np.asarray(request.target_pose, float).tolist(),
            "collision_free": bool(out.get("collision_free")), "pairs": out["ik_result"].get("collision_pairs", []),
            "ik_fallback": bool(out.get("ik_fallback")), "multi_start": out.get("ik_multi_start")})
        return out

    InspectionPlanningBase.check_inspection_ik_for_robot = wrapped

    # the planner's goal is not always the IK goal: DDA plans to the pre-approach point (standoff along the facing axis)
    original_approach = InspectionPlanningBase._build_approach_segments

    def wrapped_approach(self, request, q_start, goal_q, approach):
        out = original_approach(self, request, q_start, goal_q, approach)
        if records:
            pre_q = np.asarray(out["pre_q"], dtype=float)
            hit = self.backend.check_collision(request.robot_name, pre_q, return_pairs=True)
            records[-1].update({"pre_q": pre_q.tolist(), "pre_collision": bool(hit.collision),
                                "pre_pairs": [list(p) for p in hit.pairs]})
        return out

    InspectionPlanningBase._build_approach_segments = wrapped_approach
    row, groups, target_rows = bench._run_once(config, snapshot, a.planner, timeout=30.0, seed=0)
    # one IK call per planned target, in the same order as target_rows
    for rec, t in zip(records, target_rows):
        rec.update({"group": t["group_name"], "pose_name": t["pose_name"], "status": t["status"],
                    "positioner_r_deg": t["positioner_r_deg"], "message": t.get("message")})
    pickle.dump({"records": records, "rotation_T": snapshot.get("second_group_rotation_T")}, open(a.out, "wb"))
    print(f"saved {len(records)} IK records ({sum(r['status'] != 'success' for r in records)} failed targets) -> {a.out}")


if __name__ == "__main__":
    main()
