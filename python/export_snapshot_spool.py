"""Rebuild a loadable spool file (+ alignment sidecar) from a benchmark planning
snapshot (.pkl saved by SimTool's "Save Planning Snapshot" button - see
viewervedo/visualizer.py's _inspection_robot_core_snapshot).

Why this exists: a snapshot's spool_vertices/spool_triangles are already-baked
WORLD-frame collision geometry - enough on their own for benchmark_path_
planners.py to replan against, but not something SimTool's "Load Spool" combo
box can open directly (it treats a loaded .ply's vertices as spool LOCAL,
chuck-relative frame and re-applies spool_world_T = T_chuck @ T_offset itself -
see _handle_request_load_spool/_apply_spool_world_T). This script writes the
embedded geometry back out as a plain .ply in that LOCAL frame, plus an
alignment sidecar (.json, same basename) _load_spool_alignment_state() reads
to restore the exact positioner/chuck pose the snapshot was made against.
Point SimTool's "Load Spool" combo box at the output .ply and the scene is
visually the same one every optimizer was compared against - no need to
remember which original spool file was loaded.

Getting the LOCAL frame right needs T_chuck (a forward-kinematics result, not
stored directly): snapshots saved after spool_world_T was added carry it
already; older ones don't, so this script recomputes it via FK on the
snapshot's own robot_joint_states (present in every snapshot, old or new) -
pass --config so it can build the RobotCoreEngine needed for that. Without
--config (or if FK fails), it falls back to writing already-WORLD-frame
vertices unmodified, which "Load Spool" will double-transform and
mis-position - only acceptable for a quick geometry-only look, not for an
accurate reload.

Usage:
    python export_snapshot_spool.py --snapshot sample/planning3.pkl \\
        --out sample/planning3_spool_snapshot.ply --config viewervedo.cfg
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import pickle

import numpy as np
import open3d as o3d


def _positioner_pose_from_joint_states(joint_states: dict) -> dict:
    """Inverse of visualizer.py's _apply_positioner_pose_values mapping
    (base_to_m_column=-x, base_to_f_column_z=z, f_column_z_to_f_column_r=
    radians(r), f_column_r_to_f_column_passive_clamp=-clamp) - lets the
    sidecar drive the positioner back to the exact joint state the snapshot
    was captured at, from robot_joint_states alone (present in every
    snapshot, unlike spool_alignment)."""
    js = joint_states or {}
    return {
        "x": -float(js.get("base_to_m_column", 0.0)),
        "z": float(js.get("base_to_f_column_z", 0.0)),
        "r": float(np.degrees(float(js.get("f_column_z_to_f_column_r", 0.0)))),
        "clamp": -float(js.get("f_column_r_to_f_column_passive_clamp", 0.0)),
    }


def _compute_spool_world_T_via_fk(snapshot: dict, engine):
    """Recompute spool_world_T = T_chuck @ T_offset for a snapshot that
    predates saving it directly - T_chuck via FK on the positioner's own
    saved joint state (always present), T_offset from spool_alignment.spool
    if the snapshot has it, else assumed identity (and forced to identity in
    the sidecar this produces, so the assumption holds at load time too).

    Returns (T, alignment_dict) or raises if FK isn't possible (e.g. no
    "positioner" robot in this scene).
    """
    from viewervedo import geometry_utils

    backend = engine._robotics_backend
    robot_backend_model = backend.robot_model("positioner")
    joint_names = engine._robot_joint_names("positioner", robot_backend_model)
    joint_states = (snapshot.get("robot_joint_states") or {}).get("positioner") or {}
    q = np.array([float(joint_states.get(str(n), 0.0)) for n in joint_names], dtype=float)
    t_chuck = backend.frame_world_T("positioner", q, "m_column_passive_r")

    existing_alignment = snapshot.get("spool_alignment") or {}
    spool_offset = existing_alignment.get("spool") or {
        "x": 0.0, "y": 0.0, "z": 0.0, "x_rotation": 0.0, "z_rotation": 0.0,
    }
    t_offset = (
        geometry_utils.transl([spool_offset["x"], spool_offset["y"], spool_offset["z"]])
        @ geometry_utils.rotz(spool_offset["z_rotation"])
        @ geometry_utils.rotx(spool_offset["x_rotation"])
    )
    t = t_chuck @ t_offset

    fix_r = bool(snapshot.get("spool_fix_r", False))
    alignment = {
        "version": 2,
        "geometry_file": None,
        "spool_file": None,
        "positioner": _positioner_pose_from_joint_states(joint_states),
        "spool": spool_offset,
        "fix_f_column_r": fix_r,
        "fix_m_column_z": bool(existing_alignment.get("fix_m_column_z", False)),
        "fixation": {
            "fixed": fix_r or bool(existing_alignment.get("fixation", {}).get("fixed", False)),
            "fix_f_column_r": fix_r,
            "fix_m_column_z": bool(existing_alignment.get("fix_m_column_z", False)),
        },
        "chuck_mount_points": existing_alignment.get("chuck_mount_points", {}),
    }
    return t, alignment


def export_spool_from_snapshot(snapshot: dict, out_path: str, engine=None) -> dict:
    """Core logic, taking an already-unpickled snapshot dict - shared with
    benchmark_path_planners.py's --save-paths, which writes a spool.ply next
    to each run's summary.csv this same way so "Load Playback Result" can
    auto-load the matching spool (see simtool/window.py) without the caller
    needing to still have the original snapshot .pkl around.

    engine: an already-built RobotCoreEngine for this same snapshot, used to
    recompute spool_world_T via FK when the snapshot doesn't already have it
    (see _compute_spool_world_T_via_fk). Callers that already build one for
    planning (benchmark_path_planners.py's _run_once) should pass it through
    instead of leaving this to silently fall back to the mispositioned path.
    """
    verts = np.asarray(snapshot["spool_vertices"], dtype=float)
    tris = np.asarray(snapshot["spool_triangles"], dtype=np.int32)

    spool_world_T = snapshot.get("spool_world_T")
    computed_alignment = None
    if spool_world_T is None and engine is not None:
        try:
            spool_world_T, computed_alignment = _compute_spool_world_T_via_fk(snapshot, engine)
        except Exception:
            spool_world_T = None

    if spool_world_T is not None:
        t = np.asarray(spool_world_T, dtype=float)
        r, tr = t[:3, :3], t[:3, 3]
        verts = (np.linalg.inv(r) @ (verts - tr).T).T
        frame_note = "local (spool_world_T undone - will round-trip correctly through Load Spool)"
    else:
        frame_note = (
            "WORLD frame, unmodified (no spool_world_T and no engine to recompute it via FK) - "
            "Load Spool will re-apply the CURRENT chuck/offset transform on top of these "
            "already-world coordinates, so the reloaded pipe will likely be mispositioned. "
            "Pass --config (CLI) / engine= (API) so this can be computed via FK instead."
        )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), mesh)

    result = {
        "out_path": str(out_path),
        "n_vertices": verts.shape[0],
        "n_triangles": tris.shape[0],
        "alignment_written": False,
        "world_frame_uncorrected": spool_world_T is None,
        "source_path": snapshot.get("spool_source_path") or None,
        "frame_note": frame_note,
    }

    alignment = snapshot.get("spool_alignment") or computed_alignment
    if alignment:
        # Point the sidecar at the file we just wrote, not wherever the
        # original spool file used to live - _load_spool_alignment_state only
        # warns on a name mismatch, but there is no reason to leave a stale
        # name in a freshly-written sidecar.
        alignment = copy.deepcopy(alignment)
        alignment["geometry_file"] = out_path.name
        alignment["spool_file"] = out_path.name
        sidecar_path = out_path.with_suffix(".json")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(alignment, f, indent=4, ensure_ascii=False)
        result["alignment_written"] = True
        result["sidecar_path"] = str(sidecar_path)

    # positioner_r_deg is the base/first-phase angle the snapshot's spool
    # geometry and robot_joint_states were captured at (see
    # _inspection_robot_core_snapshot) - surfaced here since it's the one
    # alignment fact every snapshot has, even pre-spool_alignment ones, and
    # the thing worth setting by hand if there is no sidecar to restore it.
    result["positioner_r_deg"] = snapshot.get("positioner_r_deg")
    result["positioner_joint_states"] = (snapshot.get("robot_joint_states") or {}).get("positioner")
    return result


def _build_engine(config_path: str, snapshot: dict):
    import sys

    root = pathlib.Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from common.config_loader import load_config
    from robot_core.worker import RobotCoreEngine
    from util.logger.console import ConsoleLogger

    config = load_config(config_path)
    extra = pathlib.Path(config_path).resolve().parent / "path_planning.cfg"
    if extra.exists():
        config.update(load_config(extra))
    config["root_path"] = str(root.parent)
    config["verbose_level"] = "ERROR"
    ConsoleLogger.configure(config.get("logging", {}) or {}, force=True)
    console = ConsoleLogger.get_logger()
    return RobotCoreEngine(config, snapshot), console


def export_snapshot_spool(snapshot_path: str, out_path: str, config_path: str = None) -> dict:
    with open(snapshot_path, "rb") as f:
        snapshot = pickle.load(f)
    engine = None
    if config_path and snapshot.get("spool_world_T") is None:
        try:
            engine, _ = _build_engine(config_path, snapshot)
        except Exception as exc:
            print(f"warning: could not build engine for FK ({exc}) - falling back to unmodified world frame")
    return export_spool_from_snapshot(snapshot, out_path, engine=engine)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", required=True, help="Planning snapshot .pkl (SimTool 'Save Planning Snapshot')")
    parser.add_argument("--out", required=True, help="Output .ply path (put it under sample/ so SimTool's spool combo box finds it)")
    parser.add_argument(
        "--config", default=None,
        help="viewervedo.cfg path - needed to recompute spool_world_T via FK for snapshots saved "
             "before that field existed. Omit only for a quick geometry-only look (mispositioned "
             "on Load Spool).")
    args = parser.parse_args()

    result = export_snapshot_spool(args.snapshot, args.out, config_path=args.config)
    print(f"wrote {result['out_path']} ({result['n_vertices']} vertices, {result['n_triangles']} triangles)")
    print(f"vertex frame: {result['frame_note']}")
    if result["source_path"]:
        print(f"original spool file (as recorded in the snapshot): {result['source_path']}")
    if result["alignment_written"]:
        print(f"wrote alignment sidecar: {result['sidecar_path']} "
              "(SimTool will restore positioner/chuck pose automatically on Load Spool)")
    else:
        print(
            "no spool_alignment in this snapshot (saved before this field existed) - "
            f"after Load Spool, set the positioner rotation to {result['positioner_r_deg']}° "
            "by hand to match visually.")
        if result["positioner_joint_states"]:
            print(f"positioner joint states at snapshot time: {result['positioner_joint_states']}")


if __name__ == "__main__":
    main()
