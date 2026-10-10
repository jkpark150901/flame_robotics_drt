"""Time a single link_obstacle_distances() call (the Pinocchio SDF/distance
query APF's repulsive_cost is built on) against a real snapshot, to find out
whether gpmp2/trajopt/stomp's slowness comes from this call itself being
slow, or just from calling it too many times (finite-diff gradient).

Usage (run inside the same conda env / from python/ dir, same as
benchmark_path_planners.py):
    python time_apf_distance.py --config viewervedo.cfg --snapshot ../sample/planning3.pkl
"""
import argparse
import pathlib
import pickle
import sys
import time

ROOT_PATH = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT_PATH))

from common.config_loader import load_config
from robot_core.worker import RobotCoreEngine
from plugins.robotics.inspection_workflow import inspection_group_pose_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--n-calls", type=int, default=50, help="repeat count to average over")
    args = parser.parse_args()

    config = load_config(args.config)
    with open(args.snapshot, "rb") as f:
        snapshot = pickle.load(f)

    engine = RobotCoreEngine(config, snapshot)
    target_groups = snapshot.get("target_groups") or []

    # Grab the first (robot_name, q) we can find: current robot joint state.
    robot_name = None
    for group in target_groups:
        for rn, pose_name, target_pose in inspection_group_pose_items(group):
            robot_name = rn
            break
        if robot_name:
            break
    if robot_name is None:
        print("No robot found in snapshot's target_groups.")
        return

    model = engine._find_robot(robot_name)
    backend = getattr(engine, "_robotics_backend", None)
    robot_backend_model = backend.robot_model(robot_name) if backend is not None else None
    q = engine._current_robot_q(model, robot_backend_model, robot_name=robot_name).tolist()
    print(f"robot_name={robot_name} dof={len(q)}")

    # This first call may include one-time setup (geom_model deepcopy etc,
    # see pinocchio_backend.py register_robot/plan_single_target flow) -
    # time it separately from the steady-state repeated calls.
    t0 = time.perf_counter()
    entries = backend.link_obstacle_distances(robot_name, q)
    t1 = time.perf_counter()
    print(f"first call: {(t1 - t0) * 1000:.3f} ms, {len(entries)} (link, obstacle) pairs")

    times = []
    for _ in range(args.n_calls):
        t0 = time.perf_counter()
        backend.link_obstacle_distances(robot_name, q)
        times.append(time.perf_counter() - t0)

    times.sort()
    mean_ms = sum(times) / len(times) * 1000
    median_ms = times[len(times) // 2] * 1000
    print(f"steady-state over {args.n_calls} calls: mean={mean_ms:.3f} ms, median={median_ms:.3f} ms, "
          f"min={times[0]*1000:.3f} ms, max={times[-1]*1000:.3f} ms")


if __name__ == "__main__":
    main()
