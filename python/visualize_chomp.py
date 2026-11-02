"""Before/after views of a CHOMP optimisation and the APF potential in task space.

For each selected target of a saved run (runs/<run>/), the planner path is the seed of plugins/optimizer/chomp.py (run
here with the same collision scene as planning: pipe, positioner, virtual side wall), then:
- <target>_goal_compare_{iso,side}.png      seed goal posture (orange) vs CHOMP goal posture (green), same TCP pose;
                                            per-link obstacle distance before -> after in the caption
- <target>_mid_compare_{iso,side}.png       the seed waypoint closest to an obstacle (sensor-pipe approach excluded)
                                            vs the CHOMP posture at the same path fraction
- <target>_apf_slice_3d.png                 APF repulsive potential U = 0.5*eta*(1/d - 1/d0)^2 (d = distance to pipe /
                                            positioner) on the plane through the goal TCP perpendicular to the pipe
                                            axis, with the EE paths (seed orange, CHOMP green)
- <target>_apf_slice.png                    the same slice as a 2D map with the EE paths and the goal arm skeletons
                                            (link origins) projected onto it

    python python/visualize_chomp.py --run runs/RRTConnect_none_p1 --targets "Point 1 - Inspection pose 1:DDA" \\
        --out-dir reports/2026-10-30/chomp_viz
"""
import argparse
import copy
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
from plugins.optimizer.chomp import Chomp  # noqa: E402
from plugins.optimizer.sdf_field import StaticSDF  # noqa: E402
from plugins.robotics.path_metrics import PIPE_OBSTACLE  # noqa: E402
from simtool.playback_loader import load_playback_plan_sequence  # noqa: E402
from viewervedo.visualizer import Visualizer  # noqa: E402


class _Planner:
    """What the optimizer needs from a planner: the backend/robot and a segment collision check."""

    def __init__(self, backend, robot):
        self.backend, self.robot = backend, robot

    def _robotics_collision_backend(self):
        return self.backend, self.robot

    def _check_collision(self, a, b):
        return bool(self.backend.check_edge_collision(self.robot, a, b).collision)


def _ghost(viewer, robot, q, color, alpha):
    """Clones of the robot's link actors at q (the live robot is restored afterwards)."""
    model = viewer._find_robot(robot)
    names = list(viewer._robotics_backend.joint_names(robot))
    saved = {n: float(model._joint_cfg.get(n, 0.0)) for n in names}
    for n, v in zip(names, q):
        model.set_joint(n, float(v))
    model.update_fk()
    actors = []
    for a in model.actors:
        c = a.clone(deep=True)
        c.c(color).alpha(alpha).pickable(False)
        actors.append(c)
    for n, v in saved.items():
        model.set_joint(n, v)
    model.update_fk()
    return actors


def _hide_robot(viewer, robot, hidden=True):
    for a in viewer._find_robot(robot).actors:
        a.alpha(0.0 if hidden else 1.0)


def _per_link(backend, robot, q, skip_sensor_pipe=True):
    out = {}
    for e in backend.link_obstacle_distances(robot, q, max_distance=0.3):
        if skip_sensor_pipe and "link_end" in e["link"] and e["obstacle"] == PIPE_OBSTACLE:
            continue
        k = re.sub(r"_0$", "", e["link"]).split("_", 1)[-1]
        out[k] = min(out.get(k, 1.0), float(e["distance"]))
    return out


def _link_origins(backend, robot, q):
    import pinocchio as pin
    h = backend._handle(robot)
    m, d = h.model, h.model.createData()
    pin.forwardKinematics(m, d, np.asarray(q, float))
    pin.updateFramePlacements(m, d)
    base = np.asarray(h.description.base_T, float)
    pts = {}
    for f in m.frames:
        n = str(f.name)
        if re.search(r"link[0-6]$|link_end$", n):
            p = np.asarray(d.oMf[m.getFrameId(n)].translation)
            pts[n.split("_", 1)[-1]] = base[:3, :3] @ p + base[:3, 3]
    return pts


def _apf_plane(sdfs, base_T, x_plane, center, d0, eta, half=0.9, n=181):
    """APF U on the world plane x = x_plane (perpendicular to the pipe axis) around center; sdfs are base-frame."""
    ys = np.linspace(center[1] - half, center[1] + half, n)
    zs = np.linspace(center[2] - half, center[2] + half, n)
    Y, Z = np.meshgrid(ys, zs)
    pts_w = np.stack([np.full(Y.size, x_plane), Y.ravel(), Z.ravel()], axis=1)
    inv = np.linalg.inv(base_T)
    pts_b = pts_w @ inv[:3, :3].T + inv[:3, 3]
    dist = np.min([s.distance(pts_b) for s in sdfs], axis=0).reshape(Y.shape)
    dd = np.maximum(dist, 1e-3)
    U = np.where(dd < d0, 0.5 * eta * (1.0 / dd - 1.0 / d0) ** 2, 0.0)
    return Y, Z, dist, U, pts_w


def _plane_mesh(pts_w, U, n):
    faces = [[j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i]
             for j in range(n - 1) for i in range(n - 1)]
    mesh = vedo.Mesh([pts_w, faces])
    val = np.log10(np.maximum(U.ravel(), 1e-2))
    mesh.cmap("inferno_r", val, vmin=-2, vmax=4).alpha(0.8)
    return mesh


def _shoot(plotter, center, out_base, views):
    for name, offset in views.items():
        plotter.camera.SetPosition(*(center + np.asarray(offset)))
        plotter.camera.SetFocalPoint(*center)
        plotter.camera.SetViewUp(0, 0, 1)
        plotter.render()
        plotter.screenshot(f"{out_base}_{name}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--targets", required=True, help="comma list '<group name>:<DDA|RT>'")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=str(ROOT_PATH / "python" / "viewervedo.cfg"))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config.update(load_config(pathlib.Path(args.config).resolve().parent / "path_planning.cfg"))
    config["root_path"] = ROOT_PATH
    config["app_path"] = pathlib.Path(args.config).resolve().parent / "viewervedo"
    viewer = Visualizer(config=config)
    run = pathlib.Path(args.run)
    viewer._handle_request_load_planning_snapshot({"path": str((run / "scene_snapshot.pkl").resolve())})
    viewer._clear_inspection_goal_pose_visuals(render=False)
    viewer._clear_ef_pose_visuals()
    backend = viewer._robotics_backend
    plotter = viewer.plotter
    seq = load_playback_plan_sequence(run)["plan_sequence"]
    r0 = float(viewer._positioner_r_deg)

    for spec in [s.strip() for s in args.targets.split(",") if s.strip()]:
        group_name, _, kind = spec.rpartition(":")
        robot = "dda_rb10_1300e" if kind.upper() == "DDA" else "rb20_1900es"
        group = next(g for g in seq if g["name"] == group_name)
        r = float(group.get("positioner_r_deg") or r0)
        if abs(r - float(viewer._positioner_r_deg)) > 1e-6:
            viewer._move_positioner_r_to(r, visualize_verification=False)
        seed = [np.asarray(q, float) for q in group["plans"][robot]["q_path"]]

        # collision scene as in planning (pipe, positioner, virtual side wall) in the robot base frame
        model = viewer._find_robot(robot)
        base_T = np.asarray(model._base_T, float)
        pipe_w = viewer._current_spool_collision_mesh()
        pos_w = viewer._build_positioner_collision_mesh()
        pipe_b = viewer._base_frame_collision_mesh(robot, "obstacle", pipe_w, base_T)
        pos_b = viewer._base_frame_collision_mesh(robot, "positioner", pos_w, base_T)
        wall = viewer._robot_side_wall_mesh(robot, pipe_b)
        backend.configure_collision(robot, static_meshes=[m for m in (pipe_b, pos_b, wall) if m is not None])

        opt = Chomp()
        opt.save_convergence_plot = opt.save_task_space_plot = opt.save_playback_trajectory = False
        chomp_path = [np.asarray(q, float) for q in opt.optimize(seed, _Planner(backend, robot))]
        tag = re.sub(r"[^0-9A-Za-z]+", "_", f"{group_name}_{kind}").strip("_")
        print(f"[{tag}] CHOMP {opt.last_optimization_status} {opt.last_stats}")

        frame = viewer._robot_target_link_name(robot)
        tcp_goal = np.asarray(backend.frame_world_T(robot, seed[-1], frame))[:3, 3]
        _hide_robot(viewer, robot, True)

        # 1. goal posture: seed vs CHOMP
        a0, a1 = _per_link(backend, robot, seed[-1]), _per_link(backend, robot, chomp_path[-1])
        changed = [k for k in sorted(a0) if abs(a0[k] - a1[k]) > 1e-3 or a0[k] < 0.299]
        caption = (f"{group_name} {kind} - goal posture: seed (orange) vs CHOMP (green), same TCP pose\n"
                   "link-obstacle distance [mm] seed -> CHOMP (sensor-pipe excluded): "
                   + ", ".join(f"{k} {a0[k] * 1000:.0f}->{a1[k] * 1000:.0f}" for k in changed))
        actors = _ghost(viewer, robot, seed[-1], "orange", 0.45) + _ghost(viewer, robot, chomp_path[-1], "green", 0.75)
        text = vedo.Text2D(caption, pos="top-left", s=0.75, c="black", bg="white", alpha=0.85)
        plotter.add(*actors, text)
        _shoot(plotter, tcp_goal, str(out_dir / f"{tag}_goal_compare"),
               {"iso": (2.2, -2.8, 1.8), "side": (3.6, 0.0, 0.6)})
        plotter.remove(*actors, text)

        # 2. mid-path: the seed waypoint closest to an obstacle (sensor-pipe excluded) vs CHOMP at the same fraction
        def clearance(q):
            return min(_per_link(backend, robot, q).values())
        inner = list(range(1, len(seed) - 1)) or [0]
        i_seed = min(inner, key=lambda i: clearance(seed[i]))
        frac = i_seed / max(1, len(seed) - 1)
        i_chomp = int(round(frac * (len(chomp_path) - 1)))
        c0, c1 = clearance(seed[i_seed]), clearance(chomp_path[i_chomp])
        mid_center = np.asarray(backend.frame_world_T(robot, seed[i_seed], frame))[:3, 3]
        caption = (f"{group_name} {kind} - mid path at {frac * 100:.0f}% of the path: seed waypoint {i_seed} (orange) vs "
                   f"CHOMP point {i_chomp} (green)\nclosest link-obstacle distance (sensor-pipe excluded) {c0 * 1000:.1f} mm -> "
                   f"{c1 * 1000:.1f} mm")
        actors = _ghost(viewer, robot, seed[i_seed], "orange", 0.45) + _ghost(viewer, robot, chomp_path[i_chomp],
                                                                              "green", 0.75)
        text = vedo.Text2D(caption, pos="top-left", s=0.75, c="black", bg="white", alpha=0.85)
        plotter.add(*actors, text)
        _shoot(plotter, mid_center, str(out_dir / f"{tag}_mid_compare"),
               {"iso": (2.2, -2.8, 1.8), "side": (3.6, 0.0, 0.6)})
        plotter.remove(*actors, text)

        # 3. APF in task space (same scene as CHOMP: pipe, positioner, side wall) on two planes perpendicular to the
        #    pipe axis (x): through the goal TCP, and through the seed's worst link at its closest waypoint
        d0, eta = opt.d0, opt.w_obs
        sdfs = [StaticSDF(m, resolution=0.02, cap=0.35) for m in (pipe_b, pos_b, wall) if m is not None]
        worst = min(_per_link(backend, robot, seed[i_seed]).items(), key=lambda kv: kv[1])[0]
        lo_seed_mid = _link_origins(backend, robot, seed[i_seed])
        lo_chomp_mid = _link_origins(backend, robot, chomp_path[i_chomp])
        worst_p = lo_seed_mid.get(worst, mid_center)
        ee0 = np.asarray([np.asarray(backend.frame_world_T(robot, q, frame))[:3, 3] for q in seed])
        ee1 = np.asarray([np.asarray(backend.frame_world_T(robot, q, frame))[:3, 3] for q in chomp_path])
        planes = [("goal TCP", tcp_goal, _link_origins(backend, robot, seed[-1]),
                   _link_origins(backend, robot, chomp_path[-1]), "goal posture"),
                  (f"seed {worst} @ waypoint {i_seed}", worst_p, lo_seed_mid, lo_chomp_mid,
                   f"{frac * 100:.0f}% posture")]

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
        slice_actors = []
        for ax, (label, center, sk0, sk1, pose_label) in zip(axes, planes):
            Y, Z, dist, U, pts_w = _apf_plane(sdfs, base_T, center[0], center, d0, eta)
            slice_actors.append(_plane_mesh(pts_w, U, Y.shape[0]))
            im = ax.pcolormesh(Y, Z, np.where(U > 0, U, np.nan), shading="auto", cmap="inferno_r",
                               norm=LogNorm(vmin=1e-2, vmax=1e4))
            ax.contour(Y, Z, dist, levels=[0.005], colors="black", linewidths=1.5)
            ax.contour(Y, Z, dist, levels=[d0], colors="gray", linestyles="--", linewidths=1.0)
            ax.plot(ee0[:, 1], ee0[:, 2], "-o", color="orange", ms=3, lw=1.5, label="EE path seed (RRT)")
            ax.plot(ee1[:, 1], ee1[:, 2], "-", color="green", lw=2, label="EE path CHOMP")
            for sk, c, who in ((sk0, "darkorange", "seed"), (sk1, "darkgreen", "CHOMP")):
                arr = np.asarray(list(sk.values()))
                ax.plot(arr[:, 1], arr[:, 2], "-s", color=c, ms=5, lw=1.2, alpha=0.9,
                        label=f"{pose_label} link origins, {who}")
            ax.plot([center[1]], [center[2]], "*", color="blue", ms=14, label=label)
            ax.set_xlim(Y.min(), Y.max())
            ax.set_ylim(Z.min(), Z.max())
            ax.set_xlabel("world y [m]")
            ax.set_ylabel("world z [m]")
            ax.set_title(f"plane x = {center[0]:.2f} m through {label}", fontsize=11)
            ax.set_aspect("equal")
            ax.legend(loc="lower left", fontsize="x-small")
        fig.colorbar(im, ax=axes, shrink=0.8, label=f"APF U = 0.5*eta*(1/d - 1/d0)^2   (eta={eta}, d0={d0} m)")
        fig.suptitle(f"{group_name} {kind}: APF of pipe + positioner + side wall on planes perpendicular to the pipe\n"
                     "black = obstacle surface, gray dashed = influence boundary d0; paths and arms projected onto "
                     "the plane", fontsize=12)
        fig.savefig(out_dir / f"{tag}_apf_slice.png", dpi=130)
        plt.close(fig)

        actors = slice_actors + [vedo.Line(ee0, c="orange", lw=4), vedo.Line(ee1, c="green", lw=4)] \
            + _ghost(viewer, robot, chomp_path[-1], "green", 0.3)
        text = vedo.Text2D(f"{group_name} {kind}: APF slices (log10 U, -2..4) through the goal TCP and the seed's "
                           f"worst link ({worst})\nEE path: seed orange / CHOMP green", pos="top-left", s=0.75,
                           c="black", bg="white", alpha=0.85)
        plotter.add(*actors, text)
        _shoot(plotter, 0.5 * (tcp_goal + worst_p), str(out_dir / f"{tag}_apf_slice_3d"),
               {"iso": (2.8, -2.8, 1.8)})
        plotter.remove(*actors, text)
        _hide_robot(viewer, robot, False)
        print(f"[{tag}] images -> {out_dir}")


if __name__ == "__main__":
    main()
