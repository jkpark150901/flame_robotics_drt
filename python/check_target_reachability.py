"""Is every target pose of a snapshot reachable, ignoring and respecting obstacles?

Per target (the resolved poses, i.e. after the positioner rotation for rotated groups): PyBullet IK from many random seeds
(linear track near the target x, carriage random), counting
  - converged         seeds whose solution reaches the pose (6-D error below tol),
  - collision_free    converged solutions that do not collide with the pipe/positioner/arm itself,
and the best pose error found. converged == 0 means the pose itself is out of reach of that robot (kinematically, joint
limits included); converged > 0 but collision_free == 0 means reachable only through collisions.

    python python/check_target_reachability.py --snapshot debug/snapshots/planning_snapshot_new_urdf.pkl --seeds 60
"""
import argparse
import copy
import pathlib
import pickle
import sys

import numpy as np

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from common.config_loader import load_config
from util.logger.console import ConsoleLogger
from plugins.robotics.backend import IKOptions
from plugins.robotics.inspection_workflow import inspection_group_pose_items, partition_and_sort_target_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(pathlib.Path(__file__).with_name("viewervedo.cfg")))
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--pos-tol", type=float, default=1e-3)
    ap.add_argument("--rot-tol-deg", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=2000)
    a = ap.parse_args()
    config = load_config(a.config)
    config.update(load_config(pathlib.Path(a.config).resolve().parent / "path_planning.cfg"))
    config["root_path"] = ROOT_PATH
    config["verbose_level"] = "WARNING"
    ConsoleLogger.configure(config.get("logging", {}) or {}, force=True)
    snapshot = pickle.load(open(a.snapshot, "rb"))

    from robot_core.worker import RobotCoreEngine
    engine = RobotCoreEngine(config, snapshot)
    backend = engine._robotics_backend
    rotation_T = snapshot.get("second_group_rotation_T")
    phases = partition_and_sort_target_groups(
        snapshot["target_groups"], rt_pipe_facing_axis=snapshot.get("rt_pipe_facing_axis", (0.0, -1.0, 0.0)))
    rng = np.random.default_rng(0)
    options = IKOptions(solver="pybullet", max_iter=a.max_iter, tol=1e-6, damping=1e-3)
    rows = []

    for phase in phases:
        rotated = bool(phase.get("requires_positioner_rotation"))
        pipe = engine._current_spool_collision_mesh()
        if rotated and rotation_T is not None:
            pipe = copy.deepcopy(pipe)
            pipe.transform(np.asarray(rotation_T, float))
        positioner = engine._build_positioner_collision_mesh()
        configured = set()
        for group in phase["groups"]:
            for robot, pose_name, T in inspection_group_pose_items(group):
                if robot not in configured:
                    inv = np.linalg.inv(np.asarray(backend._handle(robot).description.base_T, float))
                    meshes = []
                    for m in (pipe, positioner):
                        m = copy.deepcopy(m)
                        m.transform(inv)
                        meshes.append(m)
                    backend.configure_collision(robot, static_meshes=meshes)
                    configured.add(robot)
                frame = engine._robot_target_link_name(robot)
                model = backend.robot_model(robot)
                lo = np.where(np.isfinite(model.lowerPositionLimit), model.lowerPositionLimit, -np.pi)
                hi = np.where(np.isfinite(model.upperPositionLimit), model.upperPositionLimit, np.pi)
                names = backend.joint_names(robot)
                lin = [i for i, n in enumerate(names) if "linear_track" in n]
                car = [i for i, n in enumerate(names) if "carriage" in n]
                conv = free = 0
                best = (np.inf, np.inf)
                for _ in range(a.seeds):
                    q0 = rng.uniform(lo, hi)
                    for i in lin:
                        q0[i] = rng.uniform(max(lo[i], 0.0), hi[i])
                    for i in car:
                        q0[i] = rng.uniform(lo[i], hi[i])
                    res = backend.solve_ik(robot, T, q0, options=options, frame_name=frame)
                    if res.q is None:
                        continue
                    pe, oe = res.position_error, np.degrees(res.orientation_error)
                    if pe + 0.0 < best[0] or (pe == best[0] and oe < best[1]):
                        best = (pe, oe)
                    if pe < a.pos_tol and oe < a.rot_tol_deg:
                        conv += 1
                        if not backend.check_collision(robot, res.q).collision:
                            free += 1
                rows.append((robot, pose_name, rotated, group.get("name"), conv, free, best, T[:3, 3]))
                print(f"{robot[:3]} {pose_name:3s} {'rot' if rotated else '   '} {str(group.get('name'))[:30]:30s} "
                      f"converged {conv:3d}/{a.seeds}  collision_free {free:3d}  best err {best[0]*1000:7.2f} mm "
                      f"{best[1]:6.2f} deg  target={np.round(T[:3, 3], 2).tolist()}", flush=True)
    unreachable = [r for r in rows if r[4] == 0]
    blocked = [r for r in rows if r[4] > 0 and r[5] == 0]
    print(f"\n{len(rows)} targets: unreachable (no converged IK) {len(unreachable)}, reachable only in collision {len(blocked)}, "
          f"reachable collision-free {len(rows) - len(unreachable) - len(blocked)}")


if __name__ == "__main__":
    main()
