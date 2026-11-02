"""Collision-free IK check of a planning snapshot only (no path planning), rendered with vedo.

Runs the viewer's "Check IK" handler (_handle_request_check_ef_pose_ik) headless, one target group at a time, so the result is
exactly what the SimTool "Check IK" button shows: same IK solver, same collision scene (pipe + positioner + virtual side
wall, PyBullet meshes), same phase split (groups that need the positioner rotation are checked against the rotated pipe).
For each group a PNG shows the IK goal postures; colliding links are red, colliding obstacles are drawn as a red
translucent mesh.

    python python/check_ik_poses.py --snapshot sample/planning_snapshot_261029_2.pkl --out-dir reports/ik_check
"""
import argparse
import pathlib
import re
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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ik-solver", default="collision_free")
    ap.add_argument("--groups", default="", help="comma list of group indices to check (default: all)")
    args = ap.parse_args()

    config = load_config(args.config)
    extra = pathlib.Path(args.config).resolve().parent / "path_planning.cfg"
    if extra.exists():
        config.update(load_config(extra))
    config["root_path"] = ROOT_PATH
    config["app_path"] = pathlib.Path(args.config).resolve().parent / "viewervedo"
    viewer = Visualizer(config=config)
    viewer._handle_request_load_planning_snapshot({"path": str(pathlib.Path(args.snapshot).resolve())})
    groups = list(viewer._inspection_target_groups or [])
    if not groups:
        raise SystemExit("snapshot has no target groups")
    if args.groups:
        keep = {int(i) for i in args.groups.split(",")}
        groups = [g for i, g in enumerate(groups) if i in keep]

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter = viewer.plotter
    summary = []
    for group in groups:
        name = str(group.get("name"))
        rotated = not viewer._inspection_group_is_reachable_now(group)
        viewer._inspection_target_groups = [group]
        viewer._handle_request_check_ef_pose_ik({"ik_solver": args.ik_solver, "ik_normalize": True,
                                                 "_identity": None})
        # colliding links/obstacles are the red preview actors drawn by the handler
        red = [a for a in viewer._inspection_goal_robot_actors if a.color() is not None
               and np.allclose(a.color(), vedo.get_color("red"), atol=1e-3)]
        centre = np.mean([np.asarray(group[k], float)[:3, 3] for k in ("dda_pose", "rt_pose") if group.get(k) is not None],
                         axis=0)
        label = (f"{name}  [{'phase 2: positioner rotated' if rotated else 'phase 1'}]  "
                 f"{'COLLISION (red)' if red else 'collision free'}")
        text = vedo.Text2D(label, pos="top-left", s=0.8, c="black")
        plotter.add(text)
        base = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
        views = {"iso": (centre + np.array([2.2, -2.6, 1.8]), (0, 0, 1)),
                 "side": (centre + np.array([3.5, 0.0, 0.3]), (0, 0, 1)),
                 "top": (centre + np.array([0.0, 0.0, 4.5]), (0, 1, 0))}
        for view, (cam, up) in views.items():
            plotter.camera.SetPosition(*cam)
            plotter.camera.SetFocalPoint(*centre)
            plotter.camera.SetViewUp(*up)
            plotter.render()
            plotter.screenshot(str(out_dir / f"{base}_{view}.png"))
        plotter.remove(text)
        summary.append(f"{name}: {'phase2' if rotated else 'phase1'}, red actors={len(red)}")
    print("\n".join(summary))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
