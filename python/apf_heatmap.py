"""Render an APF (artificial potential field) heatmap + repulsive-force
vector field around one waypoint of a saved q-space path.

Full path playback (3D animation, per-waypoint clearance) now lives in
SimTool's "Load Playback Result" button - this script is only for the APF
field itself, since that's expensive (steps*steps distance queries) and only
useful when zoomed into one waypoint.

Input: a joint_states.csv as saved by test_ompl_planning.py's --target all/
single-target runs (debug/<method>_<timestamp>/<...>/joint_states.csv) -
columns: waypoint, <joint_name_0>, <joint_name_1>, ...

Usage:
    python python/apf_heatmap.py \
        --config python/viewervedo.cfg \
        --snapshot sample/my_snapshot.pkl \
        --joint-states-csv debug/RRTConnect_20260811_090000/05_rb20_1900es_RT/joint_states.csv \
        --robot-name rb20_1900es \
        --waypoint 12
        # add --rotate if this target was planned against a positioner-rotated pipe
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from common.config_loader import load_config
from util.logger.console import ConsoleLogger
from plugins.optimizer import apf


def _load_joint_states_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        joint_names = header[1:]
        q_path = [[float(v) for v in row[1:]] for row in reader]
    return joint_names, q_path


def _resolve_joint_index(spec, joint_names):
    """--apf-field-joints accepts either a bare index ("2") or a joint name
    ("j_linear_track") - names are easier to get right than guessing which
    index a link corresponds to, especially with mixed revolute/prismatic
    DOF where index order isn't obvious from the plot alone."""
    try:
        return int(spec)
    except ValueError:
        pass
    if spec in joint_names:
        return joint_names.index(spec)
    raise ValueError(f"joint {spec!r} not found - available: {joint_names}")


def _plot_apf_field(backend, robot_name, q_center, joint_names, joint_indices, out_path, *,
                     d0, eta, half_range, steps, title, color_scale="power", color_gamma=0.35,
                     exclude_links=None, include_links=None, q_target=None, k_att=0.0):
    """q_target/k_att (if k_att>0) add the attractive-to-straight-line-target
    term (apf.attractive_potential) on top of the repulsive field, so this
    shows the same combined potential gpmp2.py/trajopt.py/stomp.py now
    optimize against instead of repulsive alone - only the two swept joints
    (joint_indices) move, so q_target's other dims come from q_center as-is."""
    jx, jy = joint_indices
    range_x, range_y = half_range
    axis_x = np.linspace(q_center[jx] - range_x, q_center[jx] + range_x, steps)
    axis_y = np.linspace(q_center[jy] - range_y, q_center[jy] + range_y, steps)
    repulsive_grid = np.zeros((steps, steps))
    attractive_grid = np.zeros((steps, steps))
    for iy, vy in enumerate(axis_y):
        for ix, vx in enumerate(axis_x):
            q = np.array(q_center, dtype=float)
            q[jx] = vx
            q[jy] = vy
            repulsive_grid[iy, ix] = apf.repulsive_cost_at(backend, robot_name, q, d0, eta, exclude_links, include_links)
            if k_att and q_target is not None:
                attractive_grid[iy, ix] = apf.attractive_potential(q, q_target, k_att)
    cost_grid = repulsive_grid + attractive_grid

    grad_y, grad_x = np.gradient(cost_grid, axis_y, axis_x)
    force_x, force_y = -grad_x, -grad_y  # net force = -gradient of combined potential

    # The potential blows up as d->0 (0.5*eta*(1/d - 1/d0)^2), so a handful of
    # near-obstacle cells dwarf everything else under a linear colormap and
    # the rest of the grid reads as flat/monochrome. Compress the dynamic
    # range instead of showing raw cost.
    vmax = float(cost_grid.max())
    from matplotlib.colors import PowerNorm, LogNorm
    if vmax <= 0:
        norm = None  # entire swept region is outside d0 - genuinely flat, not a display issue
    elif color_scale == "log":
        vmin = max(float(cost_grid[cost_grid > 0].min()), 1e-9) if np.any(cost_grid > 0) else 1e-9
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif color_scale == "linear":
        norm = None
    else:
        norm = PowerNorm(gamma=color_gamma, vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        cost_grid, origin="lower", aspect="auto", cmap="inferno", norm=norm,
        extent=[axis_x[0], axis_x[-1], axis_y[0], axis_y[-1]])
    label = f"APF cost ({color_scale})" if k_att and q_target is not None else f"APF repulsive cost ({color_scale})"
    fig.colorbar(im, ax=ax, label=label)
    stride = max(1, steps // 12)
    ax.quiver(
        axis_x[::stride], axis_y[::stride],
        force_x[::stride, ::stride], force_y[::stride, ::stride],
        color="cyan", scale_units="xy")
    ax.plot(q_center[jx], q_center[jy], "w*", markersize=16, markeredgecolor="black", label="waypoint")
    if k_att and q_target is not None:
        ax.plot(q_target[jx], q_target[jy], "g^", markersize=14, markeredgecolor="black",
                label="attractive target (straight-line)")
    ax.set_xlabel(joint_names[jx] if jx < len(joint_names) else f"joint[{jx}]")
    ax.set_ylabel(joint_names[jy] if jy < len(joint_names) else f"joint[{jy}]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "axis_x": axis_x, "axis_y": axis_y, "cost": cost_grid,
        "repulsive": repulsive_grid, "attractive": attractive_grid,
        "force_x": force_x, "force_y": force_y,
    }


def _plot_single_grid(grid, axis_x, axis_y, joint_names, joint_indices, q_center, out_path, *,
                       title, colorbar_label, color_scale="power", color_gamma=0.35, q_target=None):
    """Render one already-computed grid (repulsive-only, attractive-only, or
    combined) on its own - same colormap/scaling logic as _plot_apf_field,
    factored out so repulsive and attractive can be plotted at their own
    natural scale instead of attractive being invisible next to a repulsive
    peak that's often 10^4-10^6x larger."""
    jx, jy = joint_indices
    from matplotlib.colors import PowerNorm, LogNorm
    vmax = float(grid.max())
    if vmax <= 0:
        norm = None
    elif color_scale == "log":
        vmin = max(float(grid[grid > 0].min()), 1e-9) if np.any(grid > 0) else 1e-9
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif color_scale == "linear":
        norm = None
    else:
        norm = PowerNorm(gamma=color_gamma, vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        grid, origin="lower", aspect="auto", cmap="inferno", norm=norm,
        extent=[axis_x[0], axis_x[-1], axis_y[0], axis_y[-1]])
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.plot(q_center[jx], q_center[jy], "w*", markersize=16, markeredgecolor="black", label="waypoint")
    if q_target is not None:
        ax.plot(q_target[jx], q_target[jy], "g^", markersize=14, markeredgecolor="black",
                label="attractive target (straight-line)")
    ax.set_xlabel(joint_names[jx] if jx < len(joint_names) else f"joint[{jx}]")
    ax.set_ylabel(joint_names[jy] if jy < len(joint_names) else f"joint[{jy}]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return vmax


def _save_apf_field_raw(field, out_base, *, joint_names, joint_indices, q_center, d0, eta, k_att=0.0,
                         q_target=None):
    """Raw sweep data - .npz (full grids, for re-plotting/rescaling without
    recomputing the steps*steps distance queries) + a flat .csv (x, y, cost,
    repulsive, attractive, force_x, force_y - one row per grid cell, for
    spreadsheet/quick inspection)."""
    jx, jy = joint_indices
    npz_path = out_base.with_suffix(".npz")
    np.savez(
        npz_path,
        axis_x=field["axis_x"], axis_y=field["axis_y"], cost=field["cost"],
        repulsive=field["repulsive"], attractive=field["attractive"],
        force_x=field["force_x"], force_y=field["force_y"],
        joint_x=joint_names[jx], joint_y=joint_names[jy],
        joint_x_index=jx, joint_y_index=jy,
        q_center=np.asarray(q_center, dtype=float), d0=d0, eta=eta, k_att=k_att,
        q_target=np.asarray(q_target, dtype=float) if q_target is not None else np.zeros(0),
    )
    csv_path = out_base.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([joint_names[jx], joint_names[jy], "cost", "repulsive", "attractive", "force_x", "force_y"])
        for iy, vy in enumerate(field["axis_y"]):
            for ix, vx in enumerate(field["axis_x"]):
                writer.writerow([
                    vx, vy, field["cost"][iy, ix], field["repulsive"][iy, ix], field["attractive"][iy, ix],
                    field["force_x"][iy, ix], field["force_y"][iy, ix]])
    return npz_path, csv_path


def _independent_ee_pipe_check(backend, robot_name, q, link_name, snapshot):
    """Cross-check link_obstacle_distances()'s hppfcl mesh-to-mesh distance
    against a completely independent computation: the queried link's TRUE
    WORLD position vs. the nearest raw pipe mesh vertex from the snapshot -
    no hppfcl/BVH involved at all.

    pin.forwardKinematics(model, data, q) (what link_obstacle_distances()
    uses) gives placements in the robot's LOCAL model-root frame, not
    multiplied by this robot's base_T (its world mount pose - see
    plugins/robotics/backend.py's RobotDescription.base_T and
    visualizer.py:3780-3797's _base_frame_collision_mesh, which real planning
    already corrects for). This function applies base_T explicitly so the
    returned position is comparable to the snapshot's world-frame pipe
    vertices - confirmed via verify_ee_pipe_distance.py that skipping this
    reproduces the exact mount-offset error.

    Reports the nearest-vertex distance BOTH with and without
    second_group_rotation_T applied, regardless of --rotate - since
    _current_spool_collision_mesh() (robot_core/worker.py) returns the raw,
    UNrotated snapshot vertices and configure_collision() gets whichever one
    --rotate picked, applying the rotation to the wrong target (or skipping
    it when it was actually needed) reproduces the exact same offset in both
    this check and the real hppfcl distance - so agreement between the two
    doesn't rule out "both wrong the same way". Comparing rotated vs
    unrotated here tells you which one matches what you saw in playback.
    """
    handle = backend._handle(robot_name)
    geom_id = None
    for i, geom in enumerate(handle.geom_model.geometryObjects):
        if geom.name == link_name:
            geom_id = i
            break
    if geom_id is None:
        return None
    backend.link_obstacle_distances(robot_name, q)  # updates handle.geom_data.oMg for this q
    local_pos = np.array(handle.geom_data.oMg[geom_id].translation, dtype=float)
    base_T = np.asarray(handle.description.base_T, dtype=float)
    link_world_pos = (base_T @ np.append(local_pos, 1.0))[:3]

    pipe_vertices = snapshot.get("spool_vertices")
    if pipe_vertices is None:
        return link_world_pos, None, None
    pipe_vertices = np.asarray(pipe_vertices, dtype=float)
    unrotated_dist = float(np.min(np.linalg.norm(pipe_vertices - link_world_pos, axis=1)))

    rotated_dist = None
    rotation_T = snapshot.get("second_group_rotation_T")
    if rotation_T is not None:
        T = np.asarray(rotation_T, dtype=float)
        homog = np.hstack([pipe_vertices, np.ones((pipe_vertices.shape[0], 1))])
        rotated_vertices = (homog @ T.T)[:, :3]
        rotated_dist = float(np.min(np.linalg.norm(rotated_vertices - link_world_pos, axis=1)))

    return link_world_pos, unrotated_dist, rotated_dist


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(pathlib.Path(__file__).with_name("viewervedo.cfg")))
    parser.add_argument("--snapshot", required=True, help="Planning snapshot .pkl saved from SimTool")
    parser.add_argument("--joint-states-csv", required=True, help="joint_states.csv from test_ompl_planning.py")
    parser.add_argument("--robot-name", required=True)
    parser.add_argument("--waypoint", type=int, required=True, help="Waypoint index to center the field on")
    parser.add_argument(
        "--rotate", action="store_true",
        help="Rotate the pipe by the snapshot's second_group_rotation_T before querying distances - "
             "set this if the run that produced --joint-states-csv logged "
             "'needed a positioner rotation' for this target.")
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to save apf_field_wpN.png. Default: next to --joint-states-csv")
    parser.add_argument("--apf-d0", type=float, default=apf.DEFAULT_D0, help="APF influence distance (m)")
    parser.add_argument("--apf-eta", type=float, default=apf.DEFAULT_ETA, help="APF repulsive gain")
    parser.add_argument(
        "--apf-k-att", type=float, default=0.0,
        help="Attractive gain (goal pull, see plugins/optimizer/apf.py:attractive_potential). "
             "0 (default) shows repulsive-only, matching the field before this was added. "
             "Set >0 to see the combined field gpmp2.py/trajopt.py/stomp.py now actually optimize "
             "against - the attractive target is this waypoint's straight-line point between the "
             "path's first and last waypoint (apf.straight_line_targets), not the final goal_conf.")
    parser.add_argument(
        "--apf-field-joints", type=str, nargs=2, default=["0", "1"], metavar=("J_X", "J_Y"),
        help="Which 2 joints to sweep - index (e.g. 2) or joint name (e.g. j_linear_track). "
             "Default: 0 1. See the console log for this csv's joint name list.")
    parser.add_argument(
        "--apf-field-range", type=float, nargs="+", default=[0.5], metavar="RANGE",
        help="+/- sweep range around the waypoint, in that joint's own unit (rad for revolute, "
             "m for prismatic/linear track). One value applies to both axes, or give two "
             "(RANGE_X RANGE_Y) - a shared range washes out a small-range axis (e.g. a linear "
             "track) next to a large-range one, which reads as 'unresolved'/flat.")
    parser.add_argument("--apf-field-steps", type=int, default=40, help="Grid resolution per axis")
    parser.add_argument(
        "--apf-color-scale", choices=["power", "log", "linear"], default="power",
        help="Colormap scaling - 'power' (default) and 'log' compress the dynamic range so the "
             "near-obstacle spike doesn't wash out the rest of the grid; 'linear' shows raw cost.")
    parser.add_argument(
        "--apf-color-gamma", type=float, default=0.35,
        help="Gamma for --apf-color-scale=power (0<gamma<1 compresses high values; lower = more compression)")
    parser.add_argument(
        "--apf-exclude-links", type=str, nargs="+", default=None, metavar="LINK",
        help="Link name(s) to drop before computing cost/min-distance (e.g. a fixed rail base link whose "
             "distance doesn't depend on the joints being swept and would dominate/mask everything else "
             "since it's the tightest clearance in the robot regardless of what's swept).")
    parser.add_argument(
        "--apf-only-links", type=str, nargs="+", default=None, metavar="LINK",
        help="Restrict cost/min-distance to ONLY these link(s) (e.g. just the end-effector link, to see "
             "how close the EE specifically is to obstacles instead of whichever link on the whole robot "
             "happens to be closest). Applied before --apf-exclude-links.")
    args = parser.parse_args()

    config = load_config(args.config)
    extra_config_path = pathlib.Path(args.config).resolve().parent / "path_planning.cfg"
    if extra_config_path.exists():
        config.update(load_config(extra_config_path))
    config["root_path"] = ROOT_PATH
    ConsoleLogger.configure(config.get("logging", {}) or {}, force=True)
    console = ConsoleLogger.get_logger()

    with open(args.snapshot, "rb") as f:
        import pickle
        snapshot = pickle.load(f)

    joint_names, q_path = _load_joint_states_csv(args.joint_states_csv)
    if not q_path:
        raise ValueError(f"no waypoints in {args.joint_states_csv}")
    if not (0 <= args.waypoint < len(q_path)):
        raise ValueError(f"--waypoint {args.waypoint} out of range [0, {len(q_path) - 1}]")
    console.info(f"Loaded {len(q_path)} waypoint(s), {len(joint_names)} joint(s): {joint_names}")

    from robot_core.worker import RobotCoreEngine
    engine = RobotCoreEngine(config, snapshot)

    model = engine._find_robot(args.robot_name)
    if model is None:
        available = [str(getattr(m, "name", "")) for m in getattr(engine, "_robot_models", [])]
        raise RuntimeError(
            f"robot not found in snapshot: {args.robot_name!r} - available: {available} "
            "(matches the folder name test_ompl_planning.py saved this csv under, e.g. "
            "'17_rb20_1900es_RT' -> robot name is 'rb20_1900es', 'RT' is just the pose label)")
    backend = engine._robotics_backend

    import copy
    obstacle_mesh = engine._current_spool_collision_mesh()
    if args.rotate and snapshot.get("second_group_rotation_T") is not None:
        obstacle_mesh = copy.deepcopy(obstacle_mesh)
        obstacle_mesh.transform(np.asarray(snapshot["second_group_rotation_T"], dtype=float))
    positioner_mesh = engine._build_positioner_collision_mesh()

    # _current_spool_collision_mesh()/_build_positioner_collision_mesh() are
    # in true WORLD frame (visualizer.py:3275-3282's docstring confirms:
    # world = T @ local, matching the visible pipe actor). But
    # link_obstacle_distances() computes geometry placements via plain
    # pin.forwardKinematics(model, data, q) - the robot's LOCAL model-root
    # frame, NOT multiplied by this robot's base_T (its world mount pose,
    # from viewervedo.cfg's "base" entry). Real planning
    # (visualizer.py:3780-3797, _base_frame_collision_mesh) already corrects
    # for this by transforming obstacle meshes into the robot's base frame
    # before configure_collision - do the same here, or every distance is
    # off by exactly this robot's mount transform (confirmed via
    # verify_ee_pipe_distance.py: a robot mounted 3.75m away read as ~3.5m
    # from a pipe it was actually touching).
    handle = backend._handle(args.robot_name)
    base_T = np.asarray(handle.description.base_T, dtype=float)
    if not np.allclose(base_T, np.eye(4)):
        base_T_inv = np.linalg.inv(base_T)
        console.info(f"Correcting obstacle mesh into robot base frame (base_T translation={base_T[:3, 3]})")
        obstacle_mesh = copy.deepcopy(obstacle_mesh)
        obstacle_mesh.transform(base_T_inv)
        if positioner_mesh is not None:
            positioner_mesh = copy.deepcopy(positioner_mesh)
            positioner_mesh.transform(base_T_inv)

    backend.configure_collision(
        args.robot_name,
        static_meshes=[m for m in (obstacle_mesh, positioner_mesh) if m is not None],
        sample_resolution=0.02,
    )

    joint_indices = [_resolve_joint_index(spec, joint_names) for spec in args.apf_field_joints]
    field_range = args.apf_field_range * 2 if len(args.apf_field_range) == 1 else args.apf_field_range
    if len(field_range) != 2:
        raise ValueError("--apf-field-range takes either 1 value (both axes) or 2 (RANGE_X RANGE_Y)")
    console.info(
        f"Sweeping {joint_names[joint_indices[0]]} (+/-{field_range[0]}) x "
        f"{joint_names[joint_indices[1]]} (+/-{field_range[1]})")

    # Sanity check independent of d0/eta/cost math: print the actual minimum
    # link-obstacle distance at the waypoint itself and at the sweep's 4
    # corners. If these aren't all identical, q really is affecting the
    # collision scene (the bug fixed in pinocchio_backend.py was every one of
    # these coming back bit-identical regardless of q).
    jx, jy = joint_indices
    rx, ry = field_range
    q_center = np.array(q_path[args.waypoint], dtype=float)
    probe_points = {"center": q_center}
    for label, dx, dy in [("-x-y", -rx, -ry), ("+x-y", rx, -ry), ("-x+y", -rx, ry), ("+x+y", rx, ry)]:
        q = q_center.copy()
        q[jx] += dx
        q[jy] += dy
        probe_points[label] = q
    if args.apf_only_links:
        console.info(f"Restricting cost/min-distance to link(s): {args.apf_only_links}")
    if args.apf_exclude_links:
        console.info(f"Excluding link(s) from cost/min-distance: {args.apf_exclude_links}")

    check_link = (args.apf_only_links or [None])[0]
    if check_link:
        result = _independent_ee_pipe_check(backend, args.robot_name, q_center, check_link, snapshot)
        if result is not None:
            link_world_pos, unrotated_dist, rotated_dist = result
            console.info(
                f"Independent check at waypoint {args.waypoint}: {check_link} world pos={link_world_pos}")
            console.info(
                f"  nearest raw pipe vertex distance WITHOUT rotation: {unrotated_dist}"
                f"{'  <-- matches --rotate=False (currently ' + ('ON' if args.rotate else 'OFF') + ')' if not args.rotate else ''}")
            console.info(
                f"  nearest raw pipe vertex distance WITH rotation:    {rotated_dist}"
                f"{'  <-- matches --rotate=True (currently ' + ('ON' if args.rotate else 'OFF') + ')' if args.rotate else ''}")
            console.info(
                "  Compare whichever of the two matches what you saw in playback against the hppfcl "
                "distance for collision_object_0 below - if playback showed the EE close to the pipe but "
                "neither of these two numbers is small, --rotate is not the issue.")

    for label, q in probe_points.items():
        entries = apf.filter_excluded(
            backend.link_obstacle_distances(args.robot_name, q), args.apf_exclude_links, args.apf_only_links)
        console.info(f"  {label}: {entries}")

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path(args.joint_states_csv).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    q_target = None
    if args.apf_k_att:
        path_arr = np.asarray(q_path, dtype=float)
        q_target = apf.straight_line_targets(path_arr)[args.waypoint]
        console.info(
            f"Attractive term on (k_att={args.apf_k_att}): waypoint {args.waypoint}'s straight-line "
            f"target = {q_target}")

    field_base = out_dir / f"apf_field_wp{args.waypoint}"
    field_png_path = field_base.with_suffix(".png")
    field = _plot_apf_field(
        backend, args.robot_name, q_path[args.waypoint], joint_names, joint_indices, field_png_path,
        d0=args.apf_d0, eta=args.apf_eta, half_range=field_range, steps=args.apf_field_steps,
        title=f"{args.robot_name} - APF field around waypoint {args.waypoint}",
        color_scale=args.apf_color_scale, color_gamma=args.apf_color_gamma,
        exclude_links=args.apf_exclude_links, include_links=args.apf_only_links,
        q_target=q_target, k_att=args.apf_k_att)
    npz_path, csv_path = _save_apf_field_raw(
        field, field_base, joint_names=joint_names, joint_indices=joint_indices,
        q_center=q_path[args.waypoint], d0=args.apf_d0, eta=args.apf_eta,
        k_att=args.apf_k_att, q_target=q_target)
    written = [field_png_path, npz_path, csv_path]

    # Repulsive and attractive are saved combined above (field["cost"]), but
    # combined-on-one-colorscale is exactly what buries attractive under
    # repulsive when their scales differ by orders of magnitude. Plot each
    # alone, at its own natural scale, so the shapes are actually visible,
    # and log both peaks side by side so the scale gap itself is a number
    # instead of a guess from squinting at a heatmap.
    if args.apf_k_att:
        rep_vmax = _plot_single_grid(
            field["repulsive"], field["axis_x"], field["axis_y"], joint_names, joint_indices,
            q_path[args.waypoint], field_base.with_name(field_base.name + "_repulsive").with_suffix(".png"),
            title=f"{args.robot_name} - repulsive-only around waypoint {args.waypoint}",
            colorbar_label=f"APF repulsive cost ({args.apf_color_scale})",
            color_scale=args.apf_color_scale, color_gamma=args.apf_color_gamma)
        att_vmax = _plot_single_grid(
            field["attractive"], field["axis_x"], field["axis_y"], joint_names, joint_indices,
            q_path[args.waypoint], field_base.with_name(field_base.name + "_attractive").with_suffix(".png"),
            title=f"{args.robot_name} - attractive-only around waypoint {args.waypoint}",
            colorbar_label=f"APF attractive cost ({args.apf_color_scale})",
            color_scale=args.apf_color_scale, color_gamma=args.apf_color_gamma, q_target=q_target)
        written.append(field_base.with_name(field_base.name + "_repulsive.png"))
        written.append(field_base.with_name(field_base.name + "_attractive.png"))
        ratio = (rep_vmax / att_vmax) if att_vmax > 0 else float("inf")
        console.info(
            f"Scale comparison over this sweep: repulsive max={rep_vmax:.3f}, attractive max={att_vmax:.3f}, "
            f"repulsive/attractive ratio={ratio:.1f}x "
            f"({'repulsive dominates - raise --apf-k-att to balance' if ratio > 10 else 'attractive dominates - raise --apf-eta or lower --apf-k-att to balance' if ratio < 0.1 else 'comparable scale'})")

    console.info(f"Wrote {', '.join(str(p) for p in written)}")


if __name__ == "__main__":
    main()
