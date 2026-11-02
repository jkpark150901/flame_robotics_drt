"""Evaluate a trained SAC policy with the benchmark's path metrics (PathMetricEvaluator), from the real start posture.

For every target: one deterministic rollout (its path is scored like a benchmark plan: C_eff, mandatory_ok, E_d,
E_theta, L_ee, T_motion, safe, t_plan) and N stochastic rollouts (success rate); side by side with the same target's
rows of runs_apfgoal/<algorithm>_<optimizer>_<scenario>/eval/targets.csv.

    python python/rl/eval_sac.py --model debug/rl/<name>/final_model.zip --snapshots sample/planning_snapshot_p2_3.pkl \
        --targets "Point 1 - Inspection pose 2" --compare RRTConnect_none,RRTConnect_chomp
"""
import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rl.path_env import PathPlanningEnv, load_rl_config, load_planning_config, ROOT_PATH  # noqa: E402


def rollout(env, model, target, deterministic, seed=None):
    obs, _ = env.reset(seed=seed, options={"target": target})
    t0 = time.perf_counter()
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            return info, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    ap.add_argument("--config", default=None, help="rl config used for training (e.g. python/rl/sac_residual.json)")
    ap.add_argument("--robot", default=None)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--stochastic", type=int, default=20)
    ap.add_argument("--compare", default="RRTConnect_none,RRTConnect_chomp")
    ap.add_argument("--out", default=None, help="json with the per-target results")
    ap.add_argument("--seed", type=int, default=0, help="Reproducible per-target scene/augmentation seed")
    args = ap.parse_args()

    from stable_baselines3 import SAC
    from plugins.robotics.path_metrics import MetricConfig, PathMetricEvaluator
    from plugins.robotics.inspection_workflow import inspection_group_pose_items

    cfg = load_rl_config(args.config)
    if args.robot:
        cfg["env"]["robot"] = args.robot
    if args.targets:
        cfg["env"]["target_filter"] = [t.strip() for t in args.targets.split(",") if t.strip()]
    cfg["env"]["start_perturb_prob"] = 0.0
    cfg["env"]["curriculum"] = {"enabled": False}
    zero_policy = args.model == "zero"   # residual baseline: follow the reference path only (policy action 0)

    class _Zero:
        def predict(self, obs, deterministic=True):
            return np.zeros(env.action_space.shape, dtype=np.float32), None
    snapshots = [s.strip() for s in args.snapshots.split(",")]
    env = PathPlanningEnv(cfg, snapshots=snapshots, seed=args.seed)
    model = _Zero() if zero_policy else SAC.load(args.model, device=args.device)
    planning_config = load_planning_config()
    metric_cfg = MetricConfig(mc_trials=0, d_star={"dda": 0.01, "rt": None})

    results = []
    for target_index, target in enumerate(env.targets):
        scene = env.scenes[target["snapshot"]]
        evaluator = PathMetricEvaluator(planning_config, metric_cfg,
                                        rotation_T=scene.snapshot.get("second_group_rotation_T"))
        scene_seed = args.seed + 1000 * target_index
        info, wall = rollout(env, model, target, deterministic=True, seed=scene_seed)
        group = next(g for g in scene.snapshot["target_groups"] if g.get("name") == target["group"])
        pose_name = next(p for r, p, _t in inspection_group_pose_items(group) if r == env.robot)
        ok = info["result"] == "success"
        metrics = evaluator.evaluate(
            engine=scene.engine, robot_name=env.robot, pose_name=pose_name, group=group,
            target_pose=target["target_pose"], q_path=info["q_path"],
            result={"status": "success" if ok else "failed", "message": info["result"]},
            rotate=target["phase"] == 2, plan_wall=wall)
        outcomes = ([] if zero_policy else
                    [rollout(env, model, target, deterministic=False, seed=scene_seed)[0]["result"]
                     for _ in range(args.stochastic)])
        row = {"snapshot": pathlib.Path(target["snapshot"]).stem, "group": target["group"], "phase": target["phase"],
               "result": info["result"], "steps": info["steps"], "stochastic_success": outcomes.count("success"),
               "stochastic_n": len(outcomes), "q_path": info["q_path"],
               "scene_seed": scene_seed, "n_boxes": info.get("n_boxes", 0),
               "obstacles": info.get("obstacles", []), "shield_steps": info.get("shield_steps", 0),
               **{k: metrics.get(k) for k in ("safe", "C_eff", "mandatory_ok", "E_d", "E_theta_deg", "L_ee",
                                             "T_motion", "t_plan", "fail_reason")}}
        results.append(row)
        if args.out:
            partial = pathlib.Path(args.out).with_suffix(".partial.json")
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def fmt(v, scale=1.0, n=1):
        try:
            return f"{float(v) * scale:.{n}f}"
        except (TypeError, ValueError):
            return "-"

    scen = {"p1_2": "p1", "p2_3": "p2", "p3_6": "p3"}
    print("\n| target | method | result | safe | C_eff (mm) | E_d (mm) | E_theta (deg) | L_ee (m) | T_motion (s) | t_plan (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        name = f"{r['snapshot'][-4:]} {r['group']} ph{r['phase']}"
        print(f"| {name} | SAC (det.; stochastic {r['stochastic_success']}/{r['stochastic_n']}) | {r['result']} | {r['safe']} | "
              f"{fmt(r['C_eff'], 1000)} | {fmt(r['E_d'], 1000)} | {fmt(r['E_theta_deg'])} | {fmt(r['L_ee'], 1, 2)} | "
              f"{fmt(r['T_motion'])} | {fmt(r['t_plan'], 1, 2)} |")
        sc = scen.get(r["snapshot"].replace("planning_snapshot_", "").replace(".apfgoal", ""), "")
        for run in [c.strip() for c in args.compare.split(",") if c.strip()]:
            f = ROOT_PATH / "runs_apfgoal" / f"{run}_{sc}" / "eval" / "targets.csv"
            if not f.exists():
                continue
            for b in csv.DictReader(open(f, encoding="utf-8-sig")):
                if b.get("group_name") == r["group"] and env.robot in b.get("robot", ""):
                    print(f"| | {run} | {b.get('status')} | {b.get('safe')} | {fmt(b.get('C_eff'), 1000)} | {fmt(b.get('E_d'), 1000)} | "
                          f"{fmt(b.get('E_theta_deg'))} | {fmt(b.get('L_ee'), 1, 2)} | {fmt(b.get('T_motion'))} | {fmt(b.get('t_plan'), 1, 2)} |")
    # aggregate over the evaluated targets: SAC vs the compared runs on the same targets
    agg = {"SAC": []}
    for r in results:
        agg["SAC"].append({"ok": r["result"] == "success", "C_eff": r["C_eff"], "path_safe": bool(
            r["result"] == "success" and r["C_eff"] is not None and r["C_eff"] >= 0 and r["mandatory_ok"]),
            "L_ee": r["L_ee"], "T_motion": r["T_motion"], "t_plan": r["t_plan"]})
        sc = scen.get(r["snapshot"].replace("planning_snapshot_", "").replace(".apfgoal", ""), "")
        for run in [c.strip() for c in args.compare.split(",") if c.strip()]:
            f = ROOT_PATH / "runs_apfgoal" / f"{run}_{sc}" / "eval" / "targets.csv"
            if not f.exists():
                continue
            for b in csv.DictReader(open(f, encoding="utf-8-sig")):
                if b.get("group_name") == r["group"] and env.robot in b.get("robot", ""):
                    g = lambda k: float(b[k]) if b.get(k) not in (None, "", "None") else None
                    ok = b.get("status") == "success"
                    agg.setdefault(run, []).append({"ok": ok, "C_eff": g("C_eff"), "path_safe": bool(
                        ok and g("C_eff") is not None and g("C_eff") >= 0 and b.get("mandatory_ok") == "True"),
                        "L_ee": g("L_ee"), "T_motion": g("T_motion"), "t_plan": g("t_plan")})
    print("\n| method | targets | success | path safe (C_eff>=0 & no penetration) | C_eff mean (mm) | C_eff min (mm) | L_ee mean (m) | T_motion mean (s) | t_plan mean (s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, rows in agg.items():
        okr = [x for x in rows if x["ok"]]
        mean = lambda k, s=1.0: (sum(x[k] for x in okr if x[k] is not None) / max(1, sum(1 for x in okr if x[k] is not None))) * s
        ce = [x["C_eff"] for x in okr if x["C_eff"] is not None]
        print(f"| {name} | {len(rows)} | {len(okr)} | {sum(x['path_safe'] for x in rows)} | {mean('C_eff', 1000):.1f} | "
              f"{(min(ce) * 1000 if ce else float('nan')):.1f} | {mean('L_ee'):.2f} | {mean('T_motion'):.1f} | {mean('t_plan'):.2f} |")
    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.model).with_name("eval.json")
    out.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
