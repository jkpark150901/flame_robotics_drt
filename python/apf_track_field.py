"""APF (artificial potential field) visualization over the linear-track x carriage plane only.

For scaling the optimizer potential (plugins/optimizer/apf.py: d0, eta, k_att, attractive scale) it is enough to look at
the two external axes of a robot: the arm is frozen in one posture and only the `*linear_track` and `*carriage`
joints are swept over their full URDF ranges, in the collision scene of a planning snapshot (pipe + positioner,
optionally the positioner-rotated pipe). Distances come from the same backend call the optimizers use
(link_obstacle_distances), so what you see is what STOMP/TrajOpt/GPMP2 would be penalized with.

Per robot and per (d0, eta) pair it writes into --output-dir:
  <robot>_track_apf_d0<..>_eta<..>.png   repulsive cost over (track, carriage) + force arrows + d0 contour of the min distance
  <robot>_track_clearance.png            min link-obstacle distance, which link is the closest one, cost-vs-distance curves
  <robot>_track_scale.png                1-D slices over the linear track: repulsive vs attractive potential (scale review)
  <robot>_track_apf.npz / .csv           raw grids (re-plot without recomputing)
and prints cost statistics (share of the plane inside d0, max, percentiles, repulsive/attractive ratio).

Usage:
    python python/apf_track_field.py --snapshot debug/snapshots/planning_snapshot_new_urdf.pkl \
        --d0 0.10 0.15 0.25 --eta 1.0 --steps 45 [--rotate] [--arm-pose zero|<joint_states.csv>:<waypoint>]
"""

from __future__ import annotations

import argparse
import copy
import csv
import pathlib
import pickle
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm, ListedColormap

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from common.config_loader import load_config
from util.logger.console import ConsoleLogger
from plugins.optimizer import apf

USE_SDF = True   # set by --exact
DEFAULT_ROBOTS = ("dda_rb10_1300e", "rb20_1900es")


def _tag(x):
    return f"{x:g}".replace(".", "p")


def _configure_scene(engine, robot_name, snapshot, rotate):
    backend = engine._robotics_backend
    obstacle = engine._current_spool_collision_mesh()
    if rotate and snapshot.get("second_group_rotation_T") is not None:
        obstacle = copy.deepcopy(obstacle)
        obstacle.transform(np.asarray(snapshot["second_group_rotation_T"], dtype=float))
    positioner = engine._build_positioner_collision_mesh()
    handle = backend._handle(robot_name)
    base_T = np.asarray(handle.description.base_T, dtype=float)
    if not np.allclose(base_T, np.eye(4)):
        inv = np.linalg.inv(base_T)
        obstacle = copy.deepcopy(obstacle)
        obstacle.transform(inv)
        if positioner is not None:
            positioner = copy.deepcopy(positioner)
            positioner.transform(inv)
    backend.configure_collision(
        robot_name, static_meshes=[m for m in (obstacle, positioner) if m is not None], sample_resolution=0.02)
    return backend


def _track_joints(backend, robot_name):
    names = backend.joint_names(robot_name)
    lin = next(i for i, n in enumerate(names) if "linear_track" in n)
    car = next(i for i, n in enumerate(names) if "carriage" in n)
    model = backend.robot_model(robot_name)
    lo, hi = np.asarray(model.lowerPositionLimit, float), np.asarray(model.upperPositionLimit, float)
    return names, lin, car, (lo[lin], hi[lin]), (lo[car], hi[car])


def _arm_q(spec, backend, robot_name):
    dof = backend.dof(robot_name)
    if spec in (None, "zero"):
        return np.zeros(dof)
    path, _, wp = spec.partition(":")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return np.array([float(v) for v in rows[1 + int(wp or 0)][1:]], dtype=float)


def _sweep(backend, robot_name, q_arm, lin, car, xs, ys, exclude, include):
    """Per cell: sorted (distance, link) entries are reduced to what the plots need."""
    n_y, n_x = len(ys), len(xs)
    min_d = np.zeros((n_y, n_x))
    closest = np.empty((n_y, n_x), dtype=object)
    entries_grid = [[None] * n_x for _ in range(n_y)]
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            q = q_arm.copy()
            q[lin], q[car] = x, y
            entries = apf.filter_excluded(
                apf.obstacle_distances(backend, robot_name, q) if USE_SDF else backend.link_obstacle_distances(robot_name, q),
                exclude, include)
            entries_grid[iy][ix] = [(e["distance"], e["link"]) for e in entries]
            best = min(entries, key=lambda e: e["distance"]) if entries else None
            min_d[iy, ix] = best["distance"] if best else np.inf
            closest[iy, ix] = best["link"] if best else ""
    return min_d, closest, entries_grid


def _drop_fixed_links(entries_grid, closest, min_d):
    """Links whose distance does not change over the whole track plane (e.g. the rail beam itself) only add a constant
    to the cost and hide the closest-link map - drop them. Returns (fixed link names, min_d, closest)."""
    per_link = {}
    for iy, row in enumerate(entries_grid):
        for ix, entries in enumerate(row):
            for d, link in entries:
                per_link.setdefault(link, {})
                key = (iy, ix)
                per_link[link][key] = min(d, per_link[link].get(key, np.inf))
    fixed = sorted(l for l, v in per_link.items() if np.ptp(list(v.values())) < 1e-6)
    if not fixed:
        return [], min_d, closest
    fx = set(fixed)
    for iy, row in enumerate(entries_grid):
        for ix, entries in enumerate(row):
            row[ix] = [(d, l) for d, l in entries if l not in fx]
            best = min(row[ix], default=None)
            min_d[iy, ix] = best[0] if best else np.inf
            closest[iy, ix] = best[1] if best else ""
    return fixed, min_d, closest


def _cost(entries_grid, d0, eta):
    n_y, n_x = len(entries_grid), len(entries_grid[0])
    out = np.zeros((n_y, n_x))
    for iy in range(n_y):
        for ix in range(n_x):
            out[iy, ix] = sum(apf.repulsive_potential(d, d0, eta) for d, _ in entries_grid[iy][ix])
    return out


def _norm(grid, scale, gamma):
    vmax = float(grid.max())
    if vmax <= 0 or scale == "linear":
        return None
    if scale == "log":
        pos = grid[grid > 0]
        return LogNorm(vmin=max(float(pos.min()), 1e-9), vmax=vmax)
    return PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)


def _plot_cost(robot, names, lin, car, xs, ys, cost, min_d, d0, eta, out, scale, gamma):
    fy, fx = np.gradient(cost, ys, xs)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(cost, origin="lower", aspect="auto", cmap="inferno", norm=_norm(cost, scale, gamma),
                   extent=[xs[0], xs[-1], ys[0], ys[-1]])
    fig.colorbar(im, ax=ax, label=f"repulsive cost ({scale})")
    stride = max(1, len(xs) // 14)
    ax.quiver(xs[::stride], ys[::stride], -fx[::stride, ::stride], -fy[::stride, ::stride], color="cyan",
              angles="xy", scale_units="xy", pivot="mid")
    try:
        ax.contour(xs, ys, min_d, levels=[0.0], colors="white", linewidths=1.5)
        ax.contour(xs, ys, min_d, levels=[d0], colors="lime", linewidths=1.0, linestyles="--")
    except Exception:
        pass
    inside = float((cost > 0).mean()) * 100
    ax.set_xlabel(f"{names[lin]} (m)")
    ax.set_ylabel(f"{names[car]} (m)")
    ax.set_title(f"{robot}  d0={d0} m  eta={eta}   cost>0 on {inside:.0f}% of the plane, max={cost.max():.3g}\n"
                 "white: d=0 (contact)   green dashed: d=d0   cyan: -grad (repulsive force)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_clearance(robot, names, lin, car, xs, ys, min_d, closest, d0s, eta, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    finite = np.where(np.isfinite(min_d), min_d, np.nan)
    im = axes[0].imshow(finite, origin="lower", aspect="auto", cmap="viridis_r", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                        vmin=min(0.0, np.nanmin(finite)), vmax=min(float(np.nanmax(finite)), 1.0))
    fig.colorbar(im, ax=axes[0], label="min link-obstacle distance (m, clipped at 1 m)")
    axes[0].set_title("min distance")
    links = sorted({str(l) for l in closest.ravel() if l})
    idx = np.vectorize(lambda l: links.index(str(l)) if l else -1)(closest)
    cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, max(len(links), 2))))
    axes[1].imshow(idx, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=max(len(links) - 1, 1),
                   extent=[xs[0], xs[-1], ys[0], ys[-1]])
    axes[1].set_title("closest link: " + ", ".join(f"{i}={l}" for i, l in enumerate(links)), fontsize=7)
    for ax in axes[:2]:
        ax.set_xlabel(f"{names[lin]} (m)")
        ax.set_ylabel(f"{names[car]} (m)")
    d = np.linspace(0.002, max(d0s) * 1.5, 300)
    for d0 in d0s:
        axes[2].plot(d, [apf.repulsive_potential(x, d0, eta) for x in d], label=f"d0={d0}")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("distance (m)")
    axes[2].set_ylabel(f"0.5*eta*(1/d-1/d0)^2, eta={eta}")
    axes[2].set_title("potential vs distance")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(robot)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_scale(robot, names, lin, xs, ys, entries_grid, d0s, eta, k_att, att_scale, goal_x, out):
    """Repulsive vs attractive potential along the linear track for the min / mid / max carriage position."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    att = np.array([apf.attractive_potential(np.array([x]), np.array([goal_x]), k_att, att_scale) for x in xs])
    for ax, iy in zip(axes, (0, len(ys) // 2, len(ys) - 1)):
        for d0 in d0s:
            rep = [sum(apf.repulsive_potential(d, d0, eta) for d, _ in entries_grid[iy][ix]) for ix in range(len(xs))]
            ax.plot(xs, np.maximum(rep, 1e-6), label=f"repulsive d0={d0}")
        ax.plot(xs, np.maximum(att, 1e-6), "k--", label=f"attractive k={k_att}, scale={att_scale} (goal {goal_x:.2f})")
        ax.set_yscale("log")
        ax.set_title(f"carriage = {ys[iy]:.2f} m")
        ax.set_xlabel(f"{names[lin]} (m)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("potential (log)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{robot}: repulsive vs attractive scale along the linear track (arm frozen)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(pathlib.Path(__file__).with_name("viewervedo.cfg")))
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--robot-name", nargs="+", default=list(DEFAULT_ROBOTS))
    ap.add_argument("--rotate", action="store_true", help="use the positioner-rotated pipe (second_group_rotation_T)")
    ap.add_argument("--arm-pose", default="zero", help="'zero' or '<joint_states.csv>:<waypoint>' (arm joints kept, track swept)")
    ap.add_argument("--d0", type=float, nargs="+", default=[apf.DEFAULT_D0])
    ap.add_argument("--eta", type=float, default=apf.DEFAULT_ETA)
    ap.add_argument("--k-att", type=float, default=apf.DEFAULT_K_ATT)
    ap.add_argument("--att-scale", type=float, default=apf.DEFAULT_ATTRACTIVE_SCALE)
    ap.add_argument("--goal-track", type=float, default=None, help="linear-track goal for the attractive curve (default mid range)")
    ap.add_argument("--steps", type=int, default=45, help="grid points per axis")
    ap.add_argument("--color-scale", choices=["power", "log", "linear"], default="power")
    ap.add_argument("--color-gamma", type=float, default=0.35)
    ap.add_argument("--exclude-links", nargs="+", default=None)
    ap.add_argument("--only-links", nargs="+", default=None)
    ap.add_argument("--keep-fixed-links", action="store_true",
                    help="keep links whose distance is constant over the track plane (dropped by default)")
    ap.add_argument("--exact", action="store_true", help="exact PyBullet distances instead of the precomputed scene SDF")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--verbose-level", default="WARNING")
    args = ap.parse_args()
    global USE_SDF
    USE_SDF = not args.exact

    config = load_config(args.config)
    extra = pathlib.Path(args.config).resolve().parent / "path_planning.cfg"
    if extra.exists():
        config.update(load_config(extra))
    config["root_path"] = ROOT_PATH
    config["verbose_level"] = args.verbose_level.upper()
    ConsoleLogger.configure(config.get("logging", {}) or {}, force=True)
    with open(args.snapshot, "rb") as f:
        snapshot = pickle.load(f)
    out_dir = pathlib.Path(args.output_dir or ROOT_PATH / "debug" / "apf_track_field")
    out_dir.mkdir(parents=True, exist_ok=True)

    from robot_core.worker import RobotCoreEngine
    engine = RobotCoreEngine(config, snapshot)

    for robot in args.robot_name:
        backend = _configure_scene(engine, robot, snapshot, args.rotate)
        names, lin, car, (lx0, lx1), (cy0, cy1) = _track_joints(backend, robot)
        xs, ys = np.linspace(lx0, lx1, args.steps), np.linspace(cy0, cy1, args.steps)
        q_arm = _arm_q(args.arm_pose, backend, robot)
        print(f"\n== {robot}: {names[lin]} [{lx0:.2f},{lx1:.2f}] x {names[car]} [{cy0:.2f},{cy1:.2f}], "
              f"arm pose '{args.arm_pose}', rotate={args.rotate}")
        min_d, closest, grid = _sweep(backend, robot, q_arm, lin, car, xs, ys, args.exclude_links, args.only_links)
        if not args.keep_fixed_links:
            fixed, min_d, closest = _drop_fixed_links(grid, closest, min_d)
            if fixed:
                print(f"   dropped links with constant distance over the plane: {fixed}")
        finite = min_d[np.isfinite(min_d)]
        print(f"   min distance over the plane: min={finite.min():.3f} m, median={np.median(finite):.3f} m, "
              f"in contact (d<=0): {100 * (min_d <= 0).mean():.0f}% of cells")

        for d0 in args.d0:
            cost = _cost(grid, d0, args.eta)
            nz = cost[cost > 0]
            print(f"   d0={d0:<5} eta={args.eta}: cost>0 on {100 * (cost > 0).mean():4.0f}% of cells, max={cost.max():.4g}"
                  + (f", p50/p95 of nonzero={np.percentile(nz, 50):.3g}/{np.percentile(nz, 95):.3g}" if nz.size else ""))
            _plot_cost(robot, names, lin, car, xs, ys, cost, min_d, d0, args.eta,
                       out_dir / f"{robot}_track_apf_d0{_tag(d0)}_eta{_tag(args.eta)}.png", args.color_scale, args.color_gamma)
        goal_x = args.goal_track if args.goal_track is not None else 0.5 * (lx0 + lx1)
        att_max = apf.attractive_potential(np.array([lx0]), np.array([goal_x]), args.k_att, args.att_scale)
        att_max = max(att_max, apf.attractive_potential(np.array([lx1]), np.array([goal_x]), args.k_att, args.att_scale))
        rep_max = max(_cost(grid, d0, args.eta).max() for d0 in args.d0)
        print(f"   attractive max over the track (goal {goal_x:.2f}): {att_max:.3g}; repulsive max {rep_max:.3g} "
              f"-> repulsive/attractive = {rep_max / max(att_max, 1e-12):.3g}x")
        _plot_clearance(robot, names, lin, car, xs, ys, min_d, closest, args.d0, args.eta, out_dir / f"{robot}_track_clearance.png")
        _plot_scale(robot, names, lin, xs, ys, grid, args.d0, args.eta, args.k_att, args.att_scale, goal_x,
                    out_dir / f"{robot}_track_scale.png")
        np.savez(out_dir / f"{robot}_track_apf.npz", track=xs, carriage=ys, min_distance=min_d, q_arm=q_arm,
                 d0=np.array(args.d0), eta=args.eta,
                 cost=np.stack([_cost(grid, d0, args.eta) for d0 in args.d0]))
        with open(out_dir / f"{robot}_track_apf.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([names[lin], names[car], "min_distance", "closest_link"] + [f"cost_d0_{d}" for d in args.d0])
            costs = [_cost(grid, d0, args.eta) for d0 in args.d0]
            for iy, y in enumerate(ys):
                for ix, x in enumerate(xs):
                    w.writerow([x, y, min_d[iy, ix], closest[iy, ix]] + [c[iy, ix] for c in costs])
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
