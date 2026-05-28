"""
icp_trajectory_eval.py
======================
저장된 robot TCP 궤적과 mocap 궤적을 오프라인으로 정합한 뒤 오차를 계산합니다.

입력 CSV는 verify_calibration_trajectory.py 또는 motive_robot_calibration.py가 저장한
결과 형식을 지원합니다.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_points(path: Path, mocap_source: str):
    robot_cols = ("tcp_x_m", "tcp_y_m", "tcp_z_m")
    mocap_cols_by_source = {
        "aligned": ("rb_aligned_x_m", "rb_aligned_y_m", "rb_aligned_z_m"),
        "corrected": ("rb_corrected_x_m", "rb_corrected_y_m", "rb_corrected_z_m"),
        "raw": ("rb_raw_x_m", "rb_raw_y_m", "rb_raw_z_m"),
    }
    mocap_cols = mocap_cols_by_source[mocap_source]

    rows = []
    robot = []
    mocap = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                p_robot = np.array([float(row[c]) for c in robot_cols], dtype=float)
                p_mocap = np.array([float(row[c]) for c in mocap_cols], dtype=float)
            except (KeyError, ValueError):
                continue
            if np.any(np.isnan(p_robot)) or np.any(np.isnan(p_mocap)):
                continue
            rows.append(row)
            robot.append(p_robot)
            mocap.append(p_mocap)

    if len(robot) < 3:
        raise ValueError(f"유효한 점이 너무 적습니다: {len(robot)}")
    return rows, np.asarray(robot), np.asarray(mocap)


def rigid_transform(source: np.ndarray, target: np.ndarray):
    src_centroid = np.mean(source, axis=0)
    tgt_centroid = np.mean(target, axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid

    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = tgt_centroid - R @ src_centroid
    return R, t


def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray):
    return (R @ points.T).T + t


def nearest_neighbors(source: np.ndarray, target: np.ndarray):
    indices = []
    distances = []
    for p in source:
        diff = target - p
        d2 = np.sum(diff * diff, axis=1)
        idx = int(np.argmin(d2))
        indices.append(idx)
        distances.append(float(np.sqrt(d2[idx])))
    return np.asarray(indices, dtype=int), np.asarray(distances)


def icp_nearest(source: np.ndarray, target: np.ndarray, max_iter: int, tol: float):
    R_total = np.eye(3)
    t_total = np.zeros(3)
    transformed = source.copy()
    prev_rmse = np.inf

    for iteration in range(1, max_iter + 1):
        indices, _ = nearest_neighbors(transformed, target)
        R_delta, t_delta = rigid_transform(transformed, target[indices])
        transformed = apply_transform(transformed, R_delta, t_delta)
        R_total = R_delta @ R_total
        t_total = R_delta @ t_total + t_delta

        rmse = rmse_mm(transformed, target[indices])
        if abs(prev_rmse - rmse) < tol:
            return R_total, t_total, transformed, indices, iteration
        prev_rmse = rmse

    indices, _ = nearest_neighbors(transformed, target)
    return R_total, t_total, transformed, indices, max_iter


def rmse_mm(a: np.ndarray, b: np.ndarray):
    err = np.linalg.norm(a - b, axis=1)
    return float(np.sqrt(np.mean(err * err)) * 1000.0)


def summarize_errors(aligned: np.ndarray, target: np.ndarray):
    errors_mm = np.linalg.norm(aligned - target, axis=1) * 1000.0
    return {
        "count": int(len(errors_mm)),
        "rmse_mm": float(np.sqrt(np.mean(errors_mm * errors_mm))),
        "mean_mm": float(np.mean(errors_mm)),
        "median_mm": float(np.median(errors_mm)),
        "p95_mm": float(np.percentile(errors_mm, 95)),
        "max_mm": float(np.max(errors_mm)),
    }, errors_mm


def save_aligned_csv(rows, aligned: np.ndarray, targets: np.ndarray, errors_mm: np.ndarray, path: Path):
    fieldnames = list(rows[0].keys()) + [
        "icp_mocap_x_m",
        "icp_mocap_y_m",
        "icp_mocap_z_m",
        "icp_target_x_m",
        "icp_target_y_m",
        "icp_target_z_m",
        "icp_error_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, p, q, e in zip(rows, aligned, targets, errors_mm):
            out = dict(row)
            out.update({
                "icp_mocap_x_m": round(float(p[0]), 6),
                "icp_mocap_y_m": round(float(p[1]), 6),
                "icp_mocap_z_m": round(float(p[2]), 6),
                "icp_target_x_m": round(float(q[0]), 6),
                "icp_target_y_m": round(float(q[1]), 6),
                "icp_target_z_m": round(float(q[2]), 6),
                "icp_error_mm": round(float(e), 4),
            })
            writer.writerow(out)


def main():
    p = argparse.ArgumentParser(description="저장된 robot/mocap 궤적을 ICP로 정합하고 오차를 계산")
    p.add_argument("csv_path", help="검증/캘리브레이션 결과 CSV")
    p.add_argument("--mocap_source", choices=["aligned", "corrected", "raw"], default="aligned")
    p.add_argument("--mode", choices=["paired", "nearest"], default="paired",
                   help="paired: 같은 row끼리 Kabsch 정합, nearest: 최근접점 ICP")
    p.add_argument("--max_iter", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-6, help="ICP RMSE 수렴 기준 mm")
    p.add_argument("--output_csv", default=None)
    p.add_argument("--output_json", default=None)
    args = p.parse_args()

    csv_path = Path(args.csv_path)
    rows, robot, mocap = load_points(csv_path, args.mocap_source)

    before_summary, before_errors = summarize_errors(mocap, robot)
    if args.mode == "paired":
        R, t = rigid_transform(mocap, robot)
        aligned = apply_transform(mocap, R, t)
        targets = robot
        iterations = 1
    else:
        R, t, aligned, nn_indices, iterations = icp_nearest(mocap, robot, args.max_iter, args.tol)
        targets = robot[nn_indices]
        rows = [rows[i] for i in range(len(rows))]

    after_summary, after_errors = summarize_errors(aligned, targets)
    result = {
        "input_csv": str(csv_path),
        "mocap_source": args.mocap_source,
        "mode": args.mode,
        "iterations": iterations,
        "before": before_summary,
        "after": after_summary,
        "T_icp_mocap_to_robot": [
            [float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(t[0])],
            [float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(t[1])],
            [float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(t[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }

    print(json.dumps(result, indent=2))

    output_csv = Path(args.output_csv) if args.output_csv else csv_path.with_name(f"{csv_path.stem}_icp.csv")
    output_json = Path(args.output_json) if args.output_json else csv_path.with_name(f"{csv_path.stem}_icp.json")
    save_aligned_csv(rows, aligned, targets, after_errors, output_csv)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"saved_csv: {output_csv}")
    print(f"saved_json: {output_json}")


if __name__ == "__main__":
    main()
