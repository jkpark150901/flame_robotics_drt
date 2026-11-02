"""Render the EF target groups of a snapshot the way the Viewer shows them (RT green = reachable without positioner
rotation, orange-red = needs rotation, DDA gold) together with both robots, the rails and a label per robot, from the top
and from an iso view. Used to judge which side of the pipe each robot reaches.

    python python/render_target_groups.py --snapshot debug/snapshots/planning_snapshot_new_urdf.pkl --out-dir reports/2026-10-29
"""
import argparse
import pathlib
import pickle
import sys

import numpy as np
import vedo

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_PATH / "python"))
_Plotter = vedo.Plotter


class _Offscreen(_Plotter):
    def __init__(self, *a, **k):
        k["offscreen"] = True
        super().__init__(*a, **k)


vedo.Plotter = _Offscreen
from common.config_loader import load_config  # noqa: E402
from viewervedo.visualizer import Visualizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT_PATH / "python" / "viewervedo.cfg"))
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg["root_path"] = ROOT_PATH
    cfg["app_path"] = pathlib.Path(a.config).resolve().parent / "viewervedo"
    v = Visualizer(config=cfg)
    snap = pickle.load(open(a.snapshot, "rb"))
    verts, faces = np.asarray(snap["spool_vertices"]), np.asarray(snap["spool_triangles"], dtype=int)
    pipe = vedo.Mesh([verts, faces]).c("gray").alpha(0.35)
    pipe.pickable(False)
    v.plotter.add(pipe)
    groups = [g for g in snap["target_groups"]]
    v._show_ef_target_groups(groups)
    for m in v._robot_models:
        if "positioner" in str(m.name):
            continue
        base = np.asarray(m._urdf and m.get_link_world_T(next(iter(m._link_actors))))[:3, 3]
        label = vedo.Text3D(f"{m.name}\n(base y={base[1]:.1f})", pos=base + np.array([0, 0, 1.6]), s=0.18, c="black", justify="center")
        label.follow_camera()
        v.plotter.add(label)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    centre = np.array([4.8, 2.5, 1.4])
    for name, cam, up in (("top", centre + np.array([0, 0, 9.0]), (0, 1, 0)),
                          ("iso_front", centre + np.array([0.5, -6.5, 4.0]), (0, 0, 1)),
                          ("iso_back", centre + np.array([0.5, 6.5, 4.0]), (0, 0, 1))):
        v.plotter.camera.SetPosition(*cam)
        v.plotter.camera.SetFocalPoint(*centre)
        v.plotter.camera.SetViewUp(*up)
        v.plotter.render()
        v.plotter.screenshot(str(out / f"target_groups_{name}.png"))
    print("saved", out)


if __name__ == "__main__":
    main()
