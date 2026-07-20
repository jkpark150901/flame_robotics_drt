import argparse
import csv
import json
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


def _find_trace_inputs(path):
    path = Path(path).resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)

    json_paths = sorted(path.rglob("inspection_ik_*.json"))
    json_csv_stems = {p.with_suffix("").name for p in json_paths}
    csv_paths = [
        p for p in sorted(path.rglob("inspection_ik_*.csv"))
        if p.with_suffix("").name not in json_csv_stems
    ]
    return json_paths + csv_paths


def _resolve_csv_path(path):
    path = Path(path).resolve()
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        csv_path = Path(meta["csv_path"])
        if not csv_path.is_absolute():
            csv_path = _resolve_relative_path(csv_path, path.parent)
        return csv_path, meta
    return path, {}


def _load_trace(path):
    csv_path, meta = _resolve_csv_path(path)
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        q_fields = [name for name in fieldnames if name.startswith("q")]
        q_fields.sort(key=lambda name: int(name.split("_", 1)[0][1:]))
        joint_names = [name.split("_", 1)[1] for name in q_fields]

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
    return csv_path, meta, joint_names, rows


def _normalize_q_series(q_values):
    q_values = np.asarray(q_values, dtype=float)
    q_min = np.nanmin(q_values, axis=0)
    q_max = np.nanmax(q_values, axis=0)
    span = q_max - q_min
    span[span < 1e-12] = 1.0
    return (q_values - q_min) / span


def plot_trace(path, output=None, normalized_view=False):
    csv_path, meta, joint_names, rows = _load_trace(path)
    iterations = np.array([row["iteration"] for row in rows], dtype=int)
    q_values = np.vstack([row["q"] for row in rows])
    plot_q_values = _normalize_q_series(q_values) if normalized_view else q_values

    err_norm = np.array([row["err_norm"] for row in rows], dtype=float)
    pos_err = np.array([row["position_error"] for row in rows], dtype=float)
    ori_err = np.array([row["orientation_error"] for row in rows], dtype=float)

    ik_result = meta.get("ik_result", {}) if isinstance(meta, dict) else {}
    solver = ik_result.get("solver", "-")
    normalize = ik_result.get("normalize", "-")
    robot = meta.get("robot_name", csv_path.stem) if isinstance(meta, dict) else csv_path.stem

    fig_height = max(7.0, 0.45 * len(joint_names) + 4.5)
    fig, (ax_q, ax_err) = plt.subplots(
        2,
        1,
        figsize=(13, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": [max(3, len(joint_names) * 0.28), 1.6]},
    )

    for idx, name in enumerate(joint_names):
        ax_q.plot(iterations, plot_q_values[:, idx], linewidth=1.5, label=f"q{idx}: {name}")
    ax_q.set_ylabel("joint value" + (" (min-max normalized)" if normalized_view else ""))
    ax_q.set_title(f"Inspection IK joint convergence | robot={robot}, solver={solver}, normalize={normalize}")
    ax_q.grid(True, alpha=0.25)
    ax_q.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    ax_err.plot(iterations, err_norm, label="SE3 err norm", linewidth=1.8)
    ax_err.plot(iterations, pos_err, label="position error [m]", linewidth=1.8)
    ax_err.plot(iterations, ori_err, label="orientation error [rad]", linewidth=1.8)
    ax_err.set_xlabel("iteration")
    ax_err.set_ylabel("error")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(loc="best")

    fig.tight_layout()
    if output is None:
        suffix = "_joint_normalized_plot.png" if normalized_view else "_joint_plot.png"
        output = csv_path.with_name(csv_path.stem + suffix)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"saved plot: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot joint convergence from an inspection IK q-space trace.")
    parser.add_argument("trace", help="Path to inspection_ik_*.json/csv or a session folder.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output PNG path for a single trace, or output folder for a trace folder.",
    )
    parser.add_argument(
        "--normalized-view",
        action="store_true",
        help="Plot each joint after min-max normalization for shape comparison.",
    )
    args = parser.parse_args()
    inputs = _find_trace_inputs(args.trace)
    if not inputs:
        raise RuntimeError(f"no inspection IK traces found: {args.trace}")

    output_arg = Path(args.output).resolve() if args.output else None
    if len(inputs) == 1 and (output_arg is None or output_arg.suffix):
        plot_trace(inputs[0], output=output_arg, normalized_view=args.normalized_view)
        return

    output_dir = output_arg if output_arg is not None else Path(args.trace).resolve() / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    for trace_path in inputs:
        csv_path, _ = _resolve_csv_path(trace_path)
        suffix = "_joint_normalized_plot.png" if args.normalized_view else "_joint_plot.png"
        output = output_dir / f"{csv_path.stem}{suffix}"
        plot_trace(trace_path, output=output, normalized_view=args.normalized_view)


if __name__ == "__main__":
    main()
