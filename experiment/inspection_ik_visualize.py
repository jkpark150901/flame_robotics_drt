import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import vedo


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from viewervedo.robot import RobotModel  # noqa: E402


def _resolve_relative_path(path, meta_dir=None):
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [
        (REPO_ROOT / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    if meta_dir is not None:
        candidates.append((Path(meta_dir) / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _paired_meta_path(csv_path):
    candidate = Path(csv_path).with_suffix(".json")
    return candidate if candidate.exists() else None


def _load_meta(input_path, args):
    input_path = Path(input_path).resolve()
    if input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["_meta_dir"] = str(input_path.parent)
        return meta

    if input_path.suffix.lower() == ".csv":
        paired = _paired_meta_path(input_path)
        if paired is not None:
            with paired.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["_meta_dir"] = str(paired.parent)
            meta["csv_path"] = str(input_path)
            return meta
        if not args.urdf:
            raise RuntimeError("CSV만 입력할 때 같은 이름의 JSON이 없으면 --urdf가 필요합니다.")
        return {
            "robot_name": args.robot_name or input_path.stem,
            "urdf_path": args.urdf,
            "base_pose": args.base_pose,
            "csv_path": str(input_path),
            "joint_names": [],
            "target_link_name": args.target_link,
            "_meta_dir": str(input_path.parent),
        }

    raise ValueError(f"unsupported trace input: {input_path}")


def _load_trace(meta):
    csv_path = Path(meta["csv_path"])
    if not csv_path.is_absolute():
        csv_path = _resolve_relative_path(csv_path, meta.get("_meta_dir"))
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        q_fields = [name for name in fieldnames if name.startswith("q")]
        q_fields.sort(key=lambda name: int(name.split("_", 1)[0][1:]))
        joint_names = list(meta.get("joint_names", [])) or [name.split("_", 1)[1] for name in q_fields]
        for row in reader:
            q = []
            for field in q_fields:
                q.append(float(row[field]))
            rows.append({
                "iteration": int(float(row["iteration"])),
                "err_norm": float(row["err_norm"]),
                "position_error": float(row["position_error"]),
                "orientation_error": float(row["orientation_error"]),
                "tcp": np.array([
                    float(row["tcp_x"]),
                    float(row["tcp_y"]),
                    float(row["tcp_z"]),
                ], dtype=float),
                "q": np.array(q, dtype=float),
            })
    meta["joint_names"] = joint_names
    return rows


def _apply_q(robot, joint_names, q):
    for name, value in zip(joint_names, q):
        robot.set_joint(name, float(value))
    robot.update_fk()


def _make_trace_actor(rows):
    points = np.array([row["tcp"] for row in rows], dtype=float)
    if len(points) < 2:
        return None
    return vedo.Line(points, c="magenta", lw=3).alpha(0.75)


def _make_target_actor(meta):
    target_T = meta.get("target_T")
    if target_T is None:
        return None
    target_T = np.asarray(target_T, dtype=float)
    return vedo.Sphere(target_T[:3, 3], r=0.035, c="red").alpha(0.85)


def _make_frame_actors(T, scale=0.22, prefix=""):
    T = np.asarray(T, dtype=float)
    origin = T[:3, 3]
    actors = []
    for axis_idx, color in ((0, "red"), (1, "green"), (2, "blue")):
        arrow = vedo.Arrow(origin, origin + T[:3, axis_idx] * scale, s=0.0007, c=color)
        arrow.name = f"{prefix}axis_{axis_idx}"
        actors.append(arrow)
    return actors


def main():
    parser = argparse.ArgumentParser(description="Replay an inspection IK q-space convergence trace.")
    parser.add_argument("trace", help="Path to inspection_ik_*.json or inspection_ik_*.csv")
    parser.add_argument("--step", type=int, default=1, help="Replay every Nth IK iteration.")
    parser.add_argument("--delay", type=float, default=0.03, help="Delay between frames in seconds.")
    parser.add_argument("--start-index", type=int, default=0, help="Initial trace row index.")
    parser.add_argument("--end-index", type=int, default=None, help="Final trace row index. Defaults to the last row.")
    parser.add_argument("--static", action="store_true", help="Show only the selected q frame.")
    parser.add_argument("--urdf", help="URDF path when input is CSV without paired JSON.")
    parser.add_argument("--robot-name", help="Robot name when input is CSV without paired JSON.")
    parser.add_argument("--target-link", help="Target/TCP link name for axis visualization.")
    parser.add_argument("--base-pose", nargs=6, type=float, default=[0, 0, 0, 0, 0, 0])
    args = parser.parse_args()

    meta = _load_meta(args.trace, args)

    urdf_path = Path(meta["urdf_path"])
    if not urdf_path.is_absolute():
        urdf_path = (REPO_ROOT / urdf_path).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    rows = _load_trace(meta)
    if not rows:
        raise RuntimeError("IK trace is empty")

    joint_names = list(meta.get("joint_names", []))
    target_link_name = meta.get("target_link_name") or args.target_link
    robot = RobotModel(
        name=meta.get("robot_name", "robot"),
        urdf_path=str(urdf_path),
        base_pose=meta.get("base_pose", [0, 0, 0, 0, 0, 0]),
    )
    robot.load()

    start_idx = min(max(int(args.start_index), 0), len(rows) - 1)
    end_idx = len(rows) - 1 if args.end_index is None else min(max(int(args.end_index), start_idx), len(rows) - 1)
    _apply_q(robot, joint_names, rows[start_idx]["q"])

    plotter = vedo.Plotter(
        title=f"Inspection IK Trace - {meta.get('robot_name', '-')}",
        bg="white",
        axes=1,
    )
    plotter.add(*robot.actors)
    trace_actor = _make_trace_actor(rows)
    target_actor = _make_target_actor(meta)
    if trace_actor is not None:
        plotter.add(trace_actor)
    if target_actor is not None:
        plotter.add(target_actor)
    if meta.get("target_T") is not None:
        plotter.add(*_make_frame_actors(np.asarray(meta["target_T"], dtype=float), scale=0.28, prefix="target_"))

    text = vedo.Text2D("", pos="top-left", s=0.9, c="black")
    plotter.add(text)
    plotter.show(interactive=False)
    link_axis_actors = []

    def update_text(row):
        text.text(
            f"iter={row['iteration']}  "
            f"err={row['err_norm']:.6g}  "
            f"pos={row['position_error']:.6g}m  "
            f"ori={row['orientation_error']:.6g}rad"
        )

    def update_frame(row):
        nonlocal link_axis_actors
        _apply_q(robot, joint_names, row["q"])
        if link_axis_actors:
            plotter.remove(link_axis_actors)
            link_axis_actors = []
        if target_link_name:
            T = robot.get_link_world_T(target_link_name)
            if T is not None:
                link_axis_actors = _make_frame_actors(T, scale=0.24, prefix="current_")
                plotter.add(*link_axis_actors)

    if args.static:
        update_frame(rows[start_idx])
        update_text(rows[start_idx])
        plotter.render()
        plotter.interactive()
        return

    step = max(1, int(args.step))
    try:
        while True:
            for row in rows[start_idx:end_idx + 1:step]:
                update_frame(row)
                update_text(row)
                plotter.render()
                time.sleep(max(0.0, float(args.delay)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
