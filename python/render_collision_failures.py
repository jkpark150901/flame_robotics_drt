"""Render the IK goal configurations that collide (records from capture_ik_failures.py) with vedo.

For every target whose planning failed (or that is listed with --all) it shows the viewer scene - positioner, both robots,
the pipe - with the robot at the IK goal configuration, the colliding links in red (positioner links in orange when the
positioner is part of a pair), the pipe red when it is part of a pair, and the target TCP frame. Two views per target
(close-up iso and top) are written as PNG.

The scene is what the planner collision-checked: positioner mesh static at rotation 0; for targets planned after the
positioner rotation the pipe is rotated by second_group_rotation_T.

    python python/render_collision_failures.py --snapshot debug/snapshots/planning_snapshot_new_urdf.pkl \
        --records debug/ik_goals.pkl --out-dir reports/2026-10-28/collision_failures
"""
import argparse
import os
import pathlib
import pickle
import sys

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
from viewervedo.visualizer import Visualizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT_PATH / "python" / "viewervedo.cfg"))
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--all", action="store_true", help="also render targets whose IK goal is collision free")
    ap.add_argument("--only-colliding-goal", action="store_true", default=True)
    args = ap.parse_args()

    config = load_config(args.config)
    extra = pathlib.Path(args.config).resolve().parent / "path_planning.cfg"
    if extra.exists():
        config.update(load_config(extra))
    config["root_path"] = ROOT_PATH
    config["app_path"] = pathlib.Path(args.config).resolve().parent / "viewervedo"
    viewer = Visualizer(config=config)
    plotter = viewer.plotter
    backend = viewer._robotics_backend

    snapshot = pickle.load(open(args.snapshot, "rb"))
    data = pickle.load(open(args.records, "rb"))
    rotation_T = data.get("rotation_T")
    records = data["records"]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verts = np.asarray(snapshot["spool_vertices"], dtype=float)
    faces = np.asarray(snapshot["spool_triangles"], dtype=int)
    pipe_actors = {}
    for rotated in (False, True):
        v = verts.copy()
        if rotated and rotation_T is not None:
            T = np.asarray(rotation_T, dtype=float)
            v = (np.c_[v, np.ones(len(v))] @ T.T)[:, :3]
        actor = vedo.Mesh([v, faces]).c("gray").alpha(0.45)
        actor.pickable(False)
        pipe_actors[rotated] = actor

    models = {str(m.name): m for m in viewer._robot_models}
    n_done = 0
    for idx, rec in enumerate(records):
        failed = rec["status"] != "success"
        shots = []   # (what, q, pairs)
        if args.all or (failed and not rec["collision_free"]):
            shots.append(("IK goal", rec["goal_q"], rec["pairs"]))
        if rec.get("pre_collision") and (args.all or failed):
            shots.append(("PRE-APPROACH point", rec["pre_q"], rec["pre_pairs"]))
        for what, q_goal, pairs in shots:
            robot = rec["robot"]
            model = models[robot]
            names = backend.joint_names(robot)
            for jn, value in zip(names, q_goal):
                model.set_joint(jn, float(value))
            model.update_fk()
            rotated = abs(float(rec["positioner_r_deg"] or 0.0)) > 1e-6
            plotter.remove(pipe_actors[not rotated])
            plotter.add(pipe_actors[rotated])
            viewer._clear_collision_highlights()
            viewer._highlight_collision_pairs(
                [[a for a in p if not str(a).startswith("collision_object_")] for p in pairs])
            pipe_hit = any(str(a) == "collision_object_0" for p in pairs for a in p)
            pos_hit = any(str(a) == "collision_object_1" for p in pairs for a in p)
            pipe_actors[rotated].c("red" if pipe_hit else "gray").alpha(0.7 if pipe_hit else 0.45)
            pos_model = models.get("positioner")
            pos_saved = []
            if pos_hit and pos_model is not None:
                for actors in pos_model._link_actors.values():
                    for a in actors:
                        pos_saved.append((a, tuple(a.color())))
                        a.c("orange")

            target = np.asarray(rec["pose"], dtype=float)
            tcp = target[:3, 3]
            frame = [vedo.Arrow(tcp, tcp + 0.25 * target[:3, k], c=c, s=0.003)
                     for k, c in enumerate(("red", "green", "blue"))]
            for f in frame:
                f.pickable(False)
                plotter.add(f)
            label = (f"[{idx}] {robot}  {rec['group']}:{rec['pose_name']}  rot={rec['positioner_r_deg']}deg  "
                     f"{'PLAN FAILED' if failed else 'planned'}  -  {what} collides\n"
                     + "colliding: " + "; ".join(f"{a} <-> {b}" for a, b in pairs[:5]) + (" ..." if len(pairs) > 5 else ""))
            text = vedo.Text2D(label, pos="top-left", s=0.7, c="black")
            plotter.add(text)
            base = f"{idx:02d}_{robot}_{rec['pose_name']}{'_rot' if rotated else ''}_{'pre' if 'PRE' in what else 'goal'}"
            views = {"iso": (tcp + np.array([1.6, -1.8, 1.2]), (0, 0, 1)),
                     "top": (tcp + np.array([0.0, 0.0, 3.2]), (0, 1, 0))}
            for name, (cam, up) in views.items():
                plotter.camera.SetPosition(*cam)
                plotter.camera.SetFocalPoint(*tcp)
                plotter.camera.SetViewUp(*up)
                plotter.render()
                plotter.screenshot(str(out_dir / f"{base}_{name}.png"))
            n_done += 1
            plotter.remove(text)
            for f in frame:
                plotter.remove(f)
            viewer._clear_collision_highlights()
            for a, c in pos_saved:
                a.c(c)
    print(f"rendered {n_done} target(s) -> {out_dir}")


if __name__ == "__main__":
    main()
