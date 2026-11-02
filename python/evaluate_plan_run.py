"""Evaluate a saved trajectory-planning result (debug/plan_runs/<run>/) with the metrics of path_plan_metric.md.

The run folder is what SimTool saves after every planning sequence (plan_run_io.save_plan_run): summary.csv,
<idx>_<robot>_<pose>/joint_states.csv (the planned q path), failures.json and, for runs saved from 2026-10-29 on,
scene_snapshot.pkl (the scene at the start of the sequence). The scene is restored in a headless Viewer, the positioner
is turned to each group's recorded angle exactly like the live run did (the pipe follows it), and every planned path is
evaluated against that scene with plugins/robotics/path_metrics.PathMetricEvaluator - the same metrics as
evaluate_path_plans.py, without planning again.

    python python/evaluate_plan_run.py --run-dir debug/plan_runs/20261029_170316_RRTConnect \
        [--snapshot sample/planning_snapshot_261029_1.pkl]      # needed only for runs without scene_snapshot.pkl

Writes <run-dir>/eval/targets.csv, summary.json, summary.md and prints the summary table.
"""
import argparse
import copy
import csv
import json
import pathlib
import sys
import time

import numpy as np
import vedo

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_PATH / "python"))

_Plotter = vedo.Plotter


class _OffscreenPlotter(_Plotter):
    def __init__(self, *args, **kwargs):
        kwargs["offscreen"] = True
        super().__init__(*args, **kwargs)


vedo.Plotter = _OffscreenPlotter

from common.config_loader import load_config  # noqa: E402
from plugins.robotics.path_metrics import MetricConfig, PathMetricEvaluator, aggregate, write_metric_glossary  # noqa: E402
from simtool.playback_loader import load_playback_plan_sequence  # noqa: E402
from viewervedo.visualizer import Visualizer  # noqa: E402
from evaluate_path_plans import _markdown, _parse_d_star  # noqa: E402


def _pose_key(robot_name):
    return "dda_pose" if "dda" in str(robot_name).lower() else "rt_pose"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--snapshot", default=None, help="scene snapshot (default: <run-dir>/scene_snapshot.pkl)")
    ap.add_argument("--config", default=str(ROOT_PATH / "python" / "viewervedo.cfg"))
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/eval")
    ap.add_argument("--timeout", type=float, default=180.0, help="planning time limit the run used (timeout rate)")
    ap.add_argument("--track-shift", type=float, default=0.0,
                    help="subtract this from the linear-track values of the saved paths: 0.5 for runs planned before the "
                         "track zero was moved +0.5 m (2026-10-29 evening)")
    g = ap.add_argument_group("metric parameters (path_plan_metric.md)")
    g.add_argument("--margin", type=float, default=MetricConfig.margin)
    g.add_argument("--physical-min", type=float, default=MetricConfig.physical_min)
    g.add_argument("--approach-zone", type=float, default=MetricConfig.approach_zone)
    g.add_argument("--d-star", default="dda=0.01,rt=none")
    g.add_argument("--eps-d", type=float, default=MetricConfig.eps_d)
    g.add_argument("--eps-theta", type=float, default=MetricConfig.eps_theta_deg)
    g.add_argument("--ccd-step", type=float, default=MetricConfig.ccd_step)
    g.add_argument("--mc-trials", type=int, default=0, help="Monte Carlo robustness trials (0 = off)")
    g.add_argument("--sensor-kinds", default="dda")
    g.add_argument("--joint-vel", type=float, default=MetricConfig.joint_vel_deg, help="deg/s (motion time)")
    g.add_argument("--joint-acc", type=float, default=MetricConfig.joint_acc_deg, help="deg/s^2 (motion time)")
    g.add_argument("--lin-vel", type=float, default=MetricConfig.lin_vel, help="m/s, linear track/carriage")
    g.add_argument("--lin-acc", type=float, default=MetricConfig.lin_acc, help="m/s^2, linear track/carriage")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).resolve()
    snapshot = pathlib.Path(args.snapshot).resolve() if args.snapshot else run_dir / "scene_snapshot.pkl"
    if not snapshot.exists():
        raise SystemExit(f"{snapshot} not found: this run has no scene_snapshot.pkl (saved before 2026-10-29) - pass "
                         "the planning snapshot it was planned from with --snapshot")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_playback_plan_sequence(run_dir)
    sequence = loaded["plan_sequence"]
    if args.track_shift:
        for group in sequence:
            for plan in (group.get("plans") or {}).values():
                for key in ("q_path",):
                    plan[key] = [[q[0] - args.track_shift] + list(q[1:]) for q in plan.get(key) or []]
                if plan.get("goal_q") is not None:
                    plan["goal_q"] = [plan["goal_q"][0] - args.track_shift] + list(plan["goal_q"][1:])
    summary_rows = list(csv.DictReader(open(run_dir / "summary.csv", newline="", encoding="utf-8")))

    config = load_config(args.config)
    extra = pathlib.Path(args.config).resolve().parent / "path_planning.cfg"
    if extra.exists():
        config.update(load_config(extra))
    config["root_path"] = ROOT_PATH
    config["app_path"] = pathlib.Path(args.config).resolve().parent / "viewervedo"
    viewer = Visualizer(config=config)
    viewer._handle_request_load_planning_snapshot({"path": str(snapshot)})
    initial_r = float(getattr(viewer, "_positioner_r_deg", 0.0))
    groups_by_name = {str(g.get("name")): g for g in (viewer._inspection_target_groups or [])}

    cfg = MetricConfig(
        margin=args.margin, physical_min=args.physical_min, approach_zone=args.approach_zone,
        d_star=_parse_d_star(args.d_star), eps_d=args.eps_d, eps_theta_deg=args.eps_theta, ccd_step=args.ccd_step,
        sensor_kinds=tuple(k.strip() for k in args.sensor_kinds.split(",") if k.strip()), mc_trials=args.mc_trials,
        joint_vel_deg=args.joint_vel, joint_acc_deg=args.joint_acc, lin_vel=args.lin_vel, lin_acc=args.lin_acc)
    # the scene is rotated for real below (positioner move), so the evaluator never rotates anything itself
    evaluator = PathMetricEvaluator(config, cfg, rotation_T=None)

    # target poses / points of every group at the angle it was planned at - computed BEFORE the positioner moves
    targets = {}
    for group in sequence:
        name = str(group["name"])
        source = groups_by_name.get(name)
        if source is None:
            continue
        delta = float(group.get("positioner_r_deg") or initial_r) - initial_r
        T = viewer._positioner_r_rotation_transform(delta) if abs(delta) > 1e-9 else np.eye(4)
        g2 = copy.deepcopy(source)
        if g2.get("target_point") is not None:
            g2["target_point"] = (T @ np.append(np.asarray(g2["target_point"], dtype=float)[:3], 1.0))[:3].tolist()
        targets[name] = (g2, T)

    elapsed_by_key = {(r.get("group_name"), r.get("robot_name")): r for r in summary_rows}
    rows = []
    t0 = time.perf_counter()
    for group in sequence:
        name = str(group["name"])
        r_deg = float(group.get("positioner_r_deg") or initial_r)
        if abs(r_deg - float(getattr(viewer, "_positioner_r_deg", 0.0))) > 1e-9:
            viewer._move_positioner_r_to(r_deg, visualize_verification=False)
            evaluator._probes.clear()   # pipe surface probe (E_d / E_theta normals) must follow the rotated pipe
        for robot_name, plan in (group.get("plans") or {}).items():
            srow = elapsed_by_key.get((name, robot_name), {})
            row = {"group_name": name, "robot": robot_name, "pose_name": srow.get("pose_name"),
                   "positioner_r_deg": r_deg, "status": plan.get("status"), "message": srow.get("message"),
                   "t_plan": float(srow["elapsed"]) if srow.get("elapsed") not in (None, "") else None}
            if name not in targets:
                row["metric_error"] = "group not in the scene snapshot"
                rows.append(row)
                continue
            g2, T = targets[name]
            raw = g2.get(_pose_key(robot_name))
            if raw is None:
                row["metric_error"] = f"no {_pose_key(robot_name)} in the snapshot group"
                rows.append(row)
                continue
            target_pose = T @ np.asarray(raw, dtype=float)
            viewer._playback_collision_scene(robot_name)   # live pipe (at r_deg) + positioner
            q_path = plan.get("q_path") or []
            try:
                row.update(evaluator.evaluate(
                    engine=viewer, robot_name=robot_name, pose_name=row["pose_name"], group=g2,
                    target_pose=target_pose, q_path=q_path,
                    result={"status": plan.get("status"), "message": srow.get("message")},
                    rotate=False, plan_wall=row["t_plan"]))
            except Exception as exc:
                row["metric_error"] = f"{type(exc).__name__}: {exc}"
            row["t_plan"] = row.get("t_plan")
            rows.append(row)
            print(f"[{len(rows)}/{len(summary_rows)}] {name}:{row['pose_name']} ({robot_name}) r={r_deg:.1f} "
                  f"status={row['status']} safe={row.get('safe')} C_eff={row.get('C_eff')} "
                  f"E_d={row.get('E_d')} E_theta={row.get('E_theta_deg')} L_ee={row.get('L_ee')} "
                  f"T_motion={row.get('T_motion')} "
                  f"{row.get('fail_reason') or row.get('metric_error') or ''}", flush=True)

    planner = str((loaded.get("run") or {}).get("planner") or run_dir.name)
    summaries = {planner: aggregate(rows, args.timeout)}
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_dir / "targets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"run_dir": str(run_dir), "snapshot": str(snapshot), "metric_config": cfg.__dict__,
                   "eval_wall_s": time.perf_counter() - t0, "summaries": summaries}, f, indent=2, default=str)
    md = _markdown(summaries, cfg, args.timeout)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    write_metric_glossary(out_dir, cfg)   # METRICS.md: what every column / summary key means
    print(md)
    print(f"\nwritten to {out_dir} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
