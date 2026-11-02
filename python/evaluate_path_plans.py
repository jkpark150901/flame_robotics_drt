"""Plan a snapshot with several planners and evaluate the paths with the metrics of path_plan_metric.md.

Usage:
    python python/evaluate_path_plans.py \
        --snapshot debug/snapshots/planning_snapshot_new_urdf.pkl \
        --planners rrt_connect,direct_path+stomp --repeats 3 --timeout 30 \
        --margin 0.005 --mc-trials 20 --output-dir debug/eval_<timestamp>

Planning itself is benchmark_path_planners._run_once - the same per-target plan_single_target calls, start_q
chaining and positioner-rotation phase (retreat to the zero arm posture with the track kept, pipe rotated by
second_group_rotation_T, rotated-group targets planned against the rotated pipe). Only a metric hook is added, so
every metric is measured on the collision scene that target was actually planned in. See
plugins/robotics/path_metrics.py for the definitions (C_eff, E_d, E_theta, P_collision, P_violation, T_plan, SR_safe...).

Outputs in --output-dir: targets.csv (one row per planned target, all metrics + failure reasons), summary.json,
summary.md (result table in the format of path_plan_metric.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from common.config_loader import load_config
from util.logger.console import ConsoleLogger
from plugins.robotics.path_metrics import (MetricConfig, PathMetricEvaluator, aggregate, safe_start_joint_states,
                                           write_metric_glossary)
import benchmark_path_planners as bench


def _parse_d_star(text):
    out = {"dda": None, "rt": None}
    for item in filter(None, (s.strip() for s in (text or "").split(","))):
        key, _, val = item.partition("=")
        out[key.strip().lower()] = None if val.strip().lower() in ("none", "") else float(val)
    return out


def _fmt(v, nd=1, none="—"):
    return none if v is None else f"{v:.{nd}f}"


def _markdown(summaries, cfg, timeout):
    head = ("| Planner | Safe Success Rate ↑ (%) | Effective Clearance ↑ (mm) | Collision Rate ↓ (%) | "
            "검사 거리 오차 ↓ (mm) | 검사 방향 오차 ↓ (°) | Planning P95 ↓ (s) |\n|---|---:|---:|---:|---:|---:|---:|\n")
    rows = []
    for name, s in summaries.items():
        lo, hi = s["SR_safe_ci95_pct"]
        rows.append(
            f"| {name} | {_fmt(s['SR_safe_pct'])} ({lo:.0f}–{hi:.0f}) | {_fmt(s['C_eff_mean_safe_mm'])} | "
            f"{_fmt(s['P_collision_pct'])} | {_fmt(s['E_d_mean_mm'])} | {_fmt(s['E_theta_mean_deg'])} | "
            f"{_fmt(s['T_plan_p95_s'], 2)} |")
    extra = ("\n\n| Planner | n 요청 | n 계획됨 | n safe | C_nom,min (mm) | C_eff,min (mm) | P_violation (%) | "
             "접근방향 오차 (°) | L_q 정규화 | 궤적 길이 평균 (m) | 모션 시간 평균/합 (s) | DTW-RMSE (mm) | "
             "T_plan 평균/P50 (s) | timeout (%) |\n"
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, s in summaries.items():
        extra += (
            f"| {name} | {s['n_requests']} | {s['n_planned']} | {s['n_safe']} | {_fmt(s['C_nom_min_planned_mm'])} | "
            f"{_fmt(s['C_eff_min_planned_mm'])} | {_fmt(s['P_violation_pct'])} | "
            f"{_fmt(s['approach_dir_error_mean_deg'])} | {_fmt(s['L_q_normalized_mean'], 2)} | "
            f"{_fmt(s.get('L_ee_mean_m'), 2)} | {_fmt(s.get('T_motion_mean_s'), 1)}/{_fmt(s.get('T_motion_total_s'), 1)} | "
            f"{_fmt(s['DTW_RMSE_mean_mm'], 2)} | {_fmt(s['T_plan_mean_s'], 2)}/{_fmt(s['T_plan_p50_s'], 2)} | "
            f"{_fmt(s['timeout_rate_pct'])} |\n")
    params = (f"\n\n설정: margin={cfg.margin} m, d_physical_min={cfg.physical_min} m, approach_zone={cfg.approach_zone} m, "
              f"d*={cfg.d_star}, eps_d={cfg.eps_d} m, eps_theta={cfg.eps_theta_deg}°, ccd_step={cfg.ccd_step}, "
              f"MC {cfg.mc_trials}회 (추종 σ lin={cfg.sigma_track_lin} m / rot={cfg.sigma_track_rot} rad, "
              f"정합 σ t={cfg.sigma_reg_t} m / r={cfg.sigma_reg_r} rad), max collision rate={cfg.max_collision_rate}, "
              f"timeout={timeout} s, 속도 프로파일: 사다리꼴 회전 축 {cfg.joint_vel_deg:g} deg/s / "
              f"{cfg.joint_acc_deg:g} deg/s^2 (최대 변위 각 기준), 직동 축 {cfg.lin_vel:g} m/s / {cfg.lin_acc:g} m/s^2\n")
    return head + "\n".join(rows) + extra + params


def _save_run(run_dir, records, joint_names, target_rows, planner, args, cfg, initial_r, second_delta):
    """One planner/seed run as a plan-run folder (plan_run_io layout) + the scene + its evaluation (eval/)."""
    import shutil
    from plugins.robotics.plan_run_io import save_plan_run
    saved = save_plan_run(run_dir, records, joint_names_by_robot=joint_names, meta={
        "planner": planner, "planner_options": {"step_size": args.step_size, "planning_timeout": args.timeout},
        "source": "evaluate_path_plans.py", "snapshot": str(pathlib.Path(args.snapshot).resolve()),
        "initial_positioner_r_deg": initial_r, "positioner_second_group_r_deg": second_delta,
        "scene_snapshot": "scene_snapshot.pkl"})
    shutil.copyfile(args.snapshot, pathlib.Path(run_dir) / "scene_snapshot.pkl")
    eval_dir = pathlib.Path(run_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary = {planner: aggregate(target_rows, args.timeout)}
    keys = []
    for r in target_rows:
        keys += [k for k in r if k not in keys]
    with open(eval_dir / "targets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(target_rows)
    with open(eval_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"metric_config": cfg.__dict__, "summaries": summary}, f, indent=2, default=str)
    (eval_dir / "summary.md").write_text(_markdown(summary, cfg, args.timeout), encoding="utf-8")
    write_metric_glossary(eval_dir, cfg)
    print(f"saved run {saved['run_dir']} ({saved['n_success']}/{saved['n_total']} planned)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(pathlib.Path(__file__).with_name("viewervedo.cfg")))
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--planners", default="rrt_connect", help="comma list; '<planner>+<optimizer>' allowed")
    ap.add_argument("--repeats", type=int, default=3, help="seeds per planner")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=30.0, help="T_max per target (s)")
    ap.add_argument("--step-size", type=float, default=0.1, help="planner step (SimTool: simtool.cfg planner_options)")
    ap.add_argument("--max-iter", type=int, default=8000, help="planner validity-check cap (SimTool: 8000)")
    ap.add_argument("--ik-solver", default=None, help="override path_planning.cfg inspection_ik.solver (pybullet/dls/qp)")
    ap.add_argument("--start", choices=["safe", "snapshot"], default="safe",
                    help="safe: start every robot from the collision-free linear-track/carriage position nearest the pipe "
                         "(arm at zero); snapshot: the joint states stored in the snapshot (q=0 puts the track at the rail "
                         "end where the folded arm touches the positioner)")
    ap.add_argument("--seed-carriage-track", type=int, choices=[0, 1], default=None,
                    help="override path_planning.seed_carriage_track (IK initial guess for the carriage axis)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--save-runs", action="store_true",
                    help="also save every planner/seed run as a plan-run folder (<output-dir>/runs/<planner>_s<seed>/: "
                         "summary.csv, joint_states.csv per target, failures.json with the IK goal, scene_snapshot.pkl, "
                         "eval/) - loadable with SimTool 'Load Result...' for playback")
    ap.add_argument("--verbose-level", default="WARNING")
    g = ap.add_argument_group("metric parameters (path_plan_metric.md)")
    g.add_argument("--margin", type=float, default=MetricConfig.margin, help="safety margin m (m)")
    g.add_argument("--physical-min", type=float, default=MetricConfig.physical_min)
    g.add_argument("--approach-zone", type=float, default=MetricConfig.approach_zone,
                   help="sensor-pipe distance (m) inside which the leading/trailing run is the intended approach")
    g.add_argument("--d-star", default="dda=0.01,rt=none", help="target sensor distance per robot, e.g. dda=0.01,rt=0.03")
    g.add_argument("--eps-d", type=float, default=MetricConfig.eps_d)
    g.add_argument("--eps-theta", type=float, default=MetricConfig.eps_theta_deg)
    g.add_argument("--ccd-step", type=float, default=MetricConfig.ccd_step)
    g.add_argument("--mc-trials", type=int, default=MetricConfig.mc_trials, help="0 disables the Monte Carlo robustness")
    g.add_argument("--sigma-track-lin", type=float, default=MetricConfig.sigma_track_lin)
    g.add_argument("--sigma-track-rot", type=float, default=MetricConfig.sigma_track_rot)
    g.add_argument("--sigma-reg-t", type=float, default=MetricConfig.sigma_reg_t)
    g.add_argument("--sigma-reg-r", type=float, default=MetricConfig.sigma_reg_r)
    g.add_argument("--sensor-kinds", default="dda", help="robots whose end link is the inspection sensor (E_d/E_theta apply)")
    g.add_argument("--max-collision-rate", type=float, default=MetricConfig.max_collision_rate)
    g.add_argument("--joint-vel", type=float, default=MetricConfig.joint_vel_deg, help="deg/s (motion time)")
    g.add_argument("--joint-acc", type=float, default=MetricConfig.joint_acc_deg, help="deg/s^2 (motion time)")
    g.add_argument("--lin-vel", type=float, default=MetricConfig.lin_vel, help="m/s, linear track/carriage")
    g.add_argument("--lin-acc", type=float, default=MetricConfig.lin_acc, help="m/s^2, linear track/carriage")
    args = ap.parse_args()

    config = load_config(args.config)
    extra = pathlib.Path(args.config).resolve().parent / "path_planning.cfg"
    if extra.exists():
        config.update(load_config(extra))
    config["root_path"] = ROOT_PATH
    config.setdefault("planner_options", {})["max_iter"] = int(args.max_iter)
    if args.seed_carriage_track is not None:
        config.setdefault("path_planning", {})["seed_carriage_track"] = bool(args.seed_carriage_track)
    if args.ik_solver:
        config.setdefault("inspection_ik", {})["solver"] = args.ik_solver
    config["verbose_level"] = args.verbose_level.upper()
    ConsoleLogger.configure(config.get("logging", {}) or {}, force=True)
    console = ConsoleLogger.get_logger()
    snapshot = bench._load_snapshot(args.snapshot, console)

    cfg = MetricConfig(
        margin=args.margin, physical_min=args.physical_min, approach_zone=args.approach_zone,
        d_star=_parse_d_star(args.d_star), eps_d=args.eps_d, eps_theta_deg=args.eps_theta, ccd_step=args.ccd_step,
        sensor_kinds=tuple(k.strip() for k in args.sensor_kinds.split(",") if k.strip()), mc_trials=args.mc_trials, sigma_track_lin=args.sigma_track_lin, sigma_track_rot=args.sigma_track_rot,
        sigma_reg_t=args.sigma_reg_t, sigma_reg_r=args.sigma_reg_r, max_collision_rate=args.max_collision_rate,
        joint_vel_deg=args.joint_vel, joint_acc_deg=args.joint_acc, lin_vel=args.lin_vel, lin_acc=args.lin_acc,
        seed=args.seed)
    if args.start == "safe":
        from robot_core.worker import RobotCoreEngine
        probe = RobotCoreEngine(config, snapshot)
        robots = [str(getattr(m, "name", "")) for m in probe._robot_models if "positioner" not in str(getattr(m, "name", ""))]
        states = safe_start_joint_states(probe, snapshot, robots)
        snapshot = dict(snapshot)
        merged = {k: dict(v) for k, v in (snapshot.get("robot_joint_states") or {}).items()}
        for robot, joints in states.items():
            merged.setdefault(robot, {}).update(joints)
        snapshot["robot_joint_states"] = merged
        console.info(f"safe start posture (tracks): {states}")
    evaluator = PathMetricEvaluator(config, cfg, rotation_T=snapshot.get("second_group_rotation_T"))

    out_dir = pathlib.Path(args.output_dir or ROOT_PATH / "debug" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, per_planner = [], {}
    initial_r = float(snapshot.get("positioner_r_deg") or 0.0)
    second_delta = float((config.get("path_planning", {}) or {}).get("positioner_second_group_r_deg", 180.0))
    for planner in bench._split_methods(args.planners):
        rows = []
        for r in range(args.repeats):
            seed = args.seed + r
            console.info(f"[{planner}] seed {seed} ...")
            records, joint_names = [], {}

            def hook(**kw):
                out = evaluator.evaluate(**kw)
                if args.save_runs:   # one plan_run_io record per target, in planning order
                    res, engine, robot = kw["result"], kw["engine"], kw["robot_name"]
                    try:
                        bm = engine._robotics_backend.robot_model(robot)
                        joint_names.setdefault(robot, list(engine._robot_joint_names(robot, bm)))
                    except Exception:
                        pass
                    records.append({
                        "group_name": (kw.get("group") or {}).get("name"), "robot": robot,
                        "pose_name": kw.get("pose_name"), "status": res.get("status"),
                        "status_kind": res.get("status_kind"), "message": res.get("message"),
                        "q_path": [list(map(float, q)) for q in (kw.get("q_path") or [])],
                        "positioner_r_deg": initial_r + (second_delta if kw.get("rotate") else 0.0),
                        "elapsed": kw.get("plan_wall"), "planner_stats": res.get("planner_stats") or {},
                        "verification": res.get("verification") or {}, "edge_collisions": res.get("edge_collisions") or [],
                        "collision_preview": bool(res.get("collision_preview")), "ik_failure": res.get("ik_failure"),
                        "goal_q": res.get("goal_q"), "ik_goal_q": res.get("ik_goal_q"), "diag": res.get("diag")})
                return out

            try:
                run_row, _groups, target_rows = bench._run_once(
                    config, snapshot, planner, timeout=args.timeout, seed=seed, step_size=args.step_size,
                    metric_hook=hook)
            except Exception as exc:
                console.error(f"[{planner}] seed {seed} crashed: {exc}")
                continue
            for t in target_rows:
                t["planner"], t["seed"] = planner, seed
                t.setdefault("t_plan", t.get("plan_wall_s"))
                t.setdefault("planned", False)
                t.setdefault("safe", False)
                t["message"] = t.get("message")
                rows.append(t)
            if args.save_runs and records:
                _save_run(out_dir / "runs" / f"{planner}_s{seed}", records, joint_names, target_rows, planner, args,
                          cfg, initial_r, second_delta)
        per_planner[planner] = rows
        all_rows.extend(rows)

    summaries = {name: aggregate(rows, args.timeout) for name, rows in per_planner.items()}
    keys = []
    for r in all_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_dir / "targets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "metric_config": cfg.__dict__, "summaries": summaries}, f, indent=2, default=str)
    md = _markdown(summaries, cfg, args.timeout)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    write_metric_glossary(out_dir, cfg)   # METRICS.md: what every column / summary key means
    print(md)
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
