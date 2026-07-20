import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _load_meta(trace_path):
    trace_path = Path(trace_path).resolve()
    if trace_path.suffix.lower() == ".json":
        with trace_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["_meta_dir"] = str(trace_path.parent)
        return meta

    if trace_path.suffix.lower() == ".csv":
        paired = trace_path.with_suffix(".json")
        if paired.exists():
            with paired.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["_meta_dir"] = str(paired.parent)
            meta["csv_path"] = str(trace_path)
            return meta
        raise RuntimeError("CSV만 입력할 때 같은 stem의 JSON이 필요합니다.")

    raise ValueError(f"unsupported trace path: {trace_path}")


def _load_trace(meta):
    csv_path = _resolve_relative_path(meta["csv_path"], meta.get("_meta_dir"))
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        q_fields = [name for name in fieldnames if name.startswith("q")]
        q_fields.sort(key=lambda name: int(name.split("_", 1)[0][1:]))
        joint_names = list(meta.get("joint_names", [])) or [name.split("_", 1)[1] for name in q_fields]
        for row in reader:
            rows.append({
                "iteration": int(float(row["iteration"])),
                "err_norm": float(row["err_norm"]),
                "position_error": float(row["position_error"]),
                "orientation_error": float(row["orientation_error"]),
                "q": np.array([float(row[name]) for name in q_fields], dtype=float),
            })
    if not rows:
        raise RuntimeError(f"trace is empty: {csv_path}")
    return csv_path, joint_names, rows


def _base_transform(base_pose):
    x, y, z, r, p, yaw = [float(v) for v in base_pose]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    T = np.eye(4)
    T[:3, :3] = rz @ ry @ rx
    T[:3, 3] = [x, y, z]
    return T


def _pin_pose_error(pin, model, data, q, frame_id, target_se3):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    current = data.oMf[frame_id]
    err = pin.log6(current.inverse() * target_se3).vector
    position_error = float(np.linalg.norm(current.translation - target_se3.translation))
    rot_delta = current.rotation.T @ target_se3.rotation
    cos_angle = (float(np.trace(rot_delta)) - 1.0) * 0.5
    orientation_error = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return float(np.linalg.norm(err)), position_error, orientation_error


def _run_pink(meta, q_init, max_iter, dt, tol, solver):
    try:
        import pinocchio as pin
        import pink
        from pink import solve_ik
        from pink.tasks import FrameTask
    except ImportError as exc:
        wrong_pink_hint = ""
        try:
            import pink as installed_pink
            pink_file = getattr(installed_pink, "__file__", "")
            if pink_file and pink_file.endswith("pink.py"):
                wrong_pink_hint = (
                    "\n현재 import된 pink는 로봇 IK용 Pink가 아니라 다른 패키지입니다: "
                    f"{pink_file}\n"
                    "아래처럼 잘못 설치된 pink를 제거한 뒤 로봇 IK용 패키지를 설치하세요:\n"
                    "  pip uninstall -y pink\n"
                    "  pip install pin-pink qpsolvers quadprog\n"
                )
        except Exception:
            pass
        raise RuntimeError(
            "pink 기반 비교를 실행하려면 pink와 pinocchio가 필요합니다. "
            "로봇 IK용 Pink는 보통 pin-pink 패키지로 설치합니다."
            + wrong_pink_hint
        ) from exc

    urdf_path = _resolve_relative_path(meta["urdf_path"], meta.get("_meta_dir"))
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    frame_name = meta.get("target_link_name")
    if not frame_name:
        raise RuntimeError("meta JSON에 target_link_name이 없습니다.")
    frame_id = model.getFrameId(frame_name)
    if frame_id >= len(model.frames):
        raise RuntimeError(f"target frame not found in URDF: {frame_name}")

    target_world_T = np.asarray(meta["target_T"], dtype=float)
    base_T = _base_transform(meta.get("base_pose", [0, 0, 0, 0, 0, 0]))
    target_local_T = np.linalg.inv(base_T) @ target_world_T
    target_se3 = pin.SE3(target_local_T[:3, :3], target_local_T[:3, 3])

    configuration = pink.Configuration(model, data, np.asarray(q_init, dtype=float).copy())
    task = FrameTask(frame_name, position_cost=1.0, orientation_cost=1.0)
    task.set_target(target_se3)

    rows = []
    for iteration in range(int(max_iter) + 1):
        err_norm, position_error, orientation_error = _pin_pose_error(
            pin, model, data, configuration.q, frame_id, target_se3
        )
        rows.append({
            "iteration": iteration,
            "err_norm": err_norm,
            "position_error": position_error,
            "orientation_error": orientation_error,
            "q": np.asarray(configuration.q, dtype=float).copy(),
        })
        if err_norm < tol or iteration >= int(max_iter):
            break
        velocity = solve_ik(configuration, [task], dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)
    return rows


def _save_pink_csv(path, joint_names, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "iteration",
            "err_norm",
            "position_error",
            "orientation_error",
        ] + [f"q{i}_{name}" for i, name in enumerate(joint_names)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = {
                "iteration": int(row["iteration"]),
                "err_norm": float(row["err_norm"]),
                "position_error": float(row["position_error"]),
                "orientation_error": float(row["orientation_error"]),
            }
            q = np.asarray(row["q"], dtype=float)
            for i, name in enumerate(joint_names):
                data[f"q{i}_{name}"] = float(q[i]) if i < q.size else np.nan
            writer.writerow(data)


def _plot_compare(output, original_rows, pink_rows, title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    for ax, key, label in (
        (axes[0], "err_norm", "SE3 error norm"),
        (axes[1], "position_error", "position error [m]"),
        (axes[2], "orientation_error", "orientation error [rad]"),
    ):
        ax.plot(
            [row["iteration"] for row in original_rows],
            [row[key] for row in original_rows],
            label="current DLS",
            linewidth=1.8,
        )
        ax.plot(
            [row["iteration"] for row in pink_rows],
            [row[key] for row in pink_rows],
            label="pink",
            linewidth=1.8,
        )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("iteration")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare saved inspection IK trace with Pink IK.")
    parser.add_argument("trace", help="Path to inspection_ik_*.json or paired CSV.")
    parser.add_argument("--max-iter", type=int, default=None, help="Pink max iteration. Defaults to saved max_iter.")
    parser.add_argument("--dt", type=float, default=0.35)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--solver", default="quadprog", help="qpsolvers backend used by pink.solve_ik.")
    parser.add_argument("-o", "--output-dir", help="Output directory. Defaults to trace folder/pink_compare.")
    args = parser.parse_args()

    meta = _load_meta(args.trace)
    csv_path, joint_names, original_rows = _load_trace(meta)
    if not original_rows:
        raise RuntimeError("original trace is empty")
    q_init = np.asarray(original_rows[0]["q"], dtype=float)
    max_iter = args.max_iter
    if max_iter is None:
        max_iter = int((meta.get("ik_result", {}) or {}).get("max_iter", len(original_rows) - 1))

    pink_rows = _run_pink(meta, q_init, max_iter=max_iter, dt=args.dt, tol=args.tol, solver=args.solver)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else csv_path.parent / "pink_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    pink_csv = out_dir / f"{csv_path.stem}_pink.csv"
    plot_path = out_dir / f"{csv_path.stem}_pink_compare.png"
    _save_pink_csv(pink_csv, joint_names, pink_rows)
    _plot_compare(plot_path, original_rows, pink_rows, title=csv_path.name)

    print(f"saved pink trace: {pink_csv}")
    print(f"saved compare plot: {plot_path}")
    print(
        "final errors | current DLS: "
        f"pos={original_rows[-1]['position_error']:.6g}, ori={original_rows[-1]['orientation_error']:.6g}, "
        "pink: "
        f"pos={pink_rows[-1]['position_error']:.6g}, ori={pink_rows[-1]['orientation_error']:.6g}"
    )


if __name__ == "__main__":
    main()
