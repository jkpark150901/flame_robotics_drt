"""joint_states.csv 궤적을 시간 파라미터화 + 3차 스플라인 내삽해서 고주기 CSV로 저장.

원본에는 시간 열이 없고 waypoint만 있으므로, 구간별 최대 관절 변위 / vmax 로 시간을 부여한다.
RT 궤적의 앞 2개 열(linear_track, carriage)은 로봇 팔이 아니라 외부축이라 기본으로 제외하고
마지막 6개 관절(팔)만 내삽한다.

사용 예:
    python python/traj_interp.py <joint_states.csv> --vmax 20 --rate 50 -o interp.csv
"""
import argparse
import csv

import numpy as np
from scipy.interpolate import CubicSpline


def load(path, n_arm=6):
    with open(path) as f:
        r = list(csv.reader(f))
    names = r[0][1:]
    q = np.array([[float(x) for x in row[1:]] for row in r[1:]])
    return names[-n_arm:], q[:, -n_arm:]


def time_parametrize(q, vmax, tmin=0.2):
    """구간 시간 = max(|dq|)/vmax (rad/s). 너무 짧은 구간은 tmin 으로 하한."""
    dt = np.maximum(np.abs(np.diff(q, axis=0)).max(axis=1) / vmax, tmin)
    return np.concatenate([[0.0], np.cumsum(dt)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="interp_traj.csv")
    ap.add_argument("--vmax", type=float, default=20.0, help="구간 시간 산정용 최대 관절속도 (deg/s)")
    ap.add_argument("--rate", type=float, default=50.0, help="출력 주파수 (Hz)")
    args = ap.parse_args()

    names, q = load(args.csv)
    # 중복 waypoint 제거 (스플라인은 시간이 엄격히 증가해야 함)
    keep = np.r_[True, np.abs(np.diff(q, axis=0)).max(axis=1) > 1e-9]
    q = q[keep]

    t = time_parametrize(q, np.deg2rad(args.vmax))
    # 시작/끝 속도 0 (clamped) -> 정지 상태에서 출발/정지
    cs = CubicSpline(t, q, bc_type="clamped")

    ts = np.arange(0.0, t[-1], 1.0 / args.rate)
    ts = np.append(ts, t[-1])
    qs = cs(ts)
    vs = cs(ts, 1)
    as_ = cs(ts, 2)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + names)
        for ti, row in zip(ts, qs):
            w.writerow([f"{ti:.4f}"] + [f"{v:.6f}" for v in row])

    ov = np.maximum(qs.max(0) - q.max(0), q.min(0) - qs.min(0)).max()
    print(f"waypoints {len(q)} -> samples {len(ts)}  duration {t[-1]:.2f}s @ {args.rate:g}Hz")
    print(f"max |vel| {np.rad2deg(np.abs(vs).max()):.1f} deg/s, "
          f"max |acc| {np.rad2deg(np.abs(as_).max()):.1f} deg/s^2, "
          f"overshoot beyond waypoint range {np.rad2deg(max(ov, 0)):.2f} deg")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
