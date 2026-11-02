"""Export SAC evaluation trajectories (rl/eval_sac.py json, "q_path" per target) as SimTool playback folders.

For every scenario in the json, the scenario's RRTConnect run (runs_apfgoal/RRTConnect_none_<pN>) is copied - scene
snapshot, run.json, summary.csv, every target's joint_states.csv - and the evaluated targets of the robot get the
SAC path instead (summary.csv status/message/n_waypoints updated; a failed SAC rollout keeps its path up to the
collision, so playback shows where it hit). Other targets keep the RRTConnect path, so the whole sequence plays.

    python python/rl/export_playback.py --eval debug/rl/dda_residual_apf5_base/eval_p3.json --name apf5_base_p3
    -> runs_rl/apf5_base_p3_p3/  (SimTool "Load Result...")
"""
import argparse
import csv
import json
import pathlib
import re
import shutil
import sys

ROOT_PATH = pathlib.Path(__file__).resolve().parents[2]


def export(eval_json: pathlib.Path, name: str, out_root: pathlib.Path, robot: str, base_runs: pathlib.Path):
    rows = json.loads(eval_json.read_text(encoding="utf-8"))
    by_scen = {}
    for r in rows:
        m = re.search(r"_(p\d+)_", r["snapshot"] + "_")
        by_scen.setdefault(m.group(1), []).append(r)
    written = []
    for scen, items in sorted(by_scen.items()):
        src = base_runs / f"RRTConnect_none_{scen}"
        dst = out_root / f"{name}_{scen}"
        if dst.exists():
            print(f"skip {dst} (exists)")
            continue
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("eval"))
        with open(dst / "summary.csv", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields, summary = reader.fieldnames, list(reader)
        by_group = {r["group"]: r for r in items}
        replaced = 0
        for row in summary:
            r = by_group.get(row["group_name"])
            if r is None or row["robot_name"] != robot:
                continue
            sub = next(dst.glob(f"{int(row['index']):02d}_{robot}_*"), None)
            if sub is None:
                sub = dst / f"{int(row['index']):02d}_{robot}_{row['pose_name']}"
                sub.mkdir(parents=True, exist_ok=True)
            header = None
            js = sub / "joint_states.csv"
            if js.exists():
                with open(js, encoding="utf-8") as f:
                    header = next(csv.reader(f))
            header = header or ["waypoint"] + [f"q{i}" for i in range(len(r["q_path"][0]))]
            with open(js, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                for i, q in enumerate(r["q_path"]):
                    w.writerow([i] + [float(v) for v in q])
            ok = r["result"] == "success"
            row["status"] = "success" if ok else "failed"
            row["status_kind"] = "success" if ok else "failed"
            row["message"] = "" if ok else f"SAC rollout: {r['result']}"
            row["n_waypoints"] = str(len(r["q_path"]))
            replaced += 1
        with open(dst / "summary.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary)
        run = json.loads((dst / "run.json").read_text(encoding="utf-8"))
        run.update({"planner": f"RRTConnect+SAC ({name})", "source": "rl/export_playback.py",
                    "sac_eval": str(eval_json), "sac_targets_replaced": replaced,
                    "n_success": sum(1 for r in summary if r["status"] == "success"),
                    "n_failed": sum(1 for r in summary if r["status"] != "success")})
        (dst / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append((dst, replaced, sum(1 for r in items if r["result"] == "success"), len(items)))
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, action="append")
    ap.add_argument("--name", required=True, action="append", help="one per --eval")
    ap.add_argument("--out", default=str(ROOT_PATH / "runs_rl"))
    ap.add_argument("--robot", default="dda_rb10_1300e")
    ap.add_argument("--base-runs", default=str(ROOT_PATH / "runs_apfgoal"))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ev, name in zip(args.eval, args.name):
        for dst, replaced, ok, n in export(pathlib.Path(ev), name, out, args.robot, pathlib.Path(args.base_runs)):
            print(f"{dst.relative_to(ROOT_PATH)}: {replaced} target(s) replaced with SAC paths ({ok}/{n} SAC success)")


if __name__ == "__main__":
    main()
