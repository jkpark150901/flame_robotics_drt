"""원본 waypoint 와 내삽 결과의 관절각/속도를 그려서 PNG 로 저장.

사용 예:
    python rb_servo_test/plot_traj.py --raw rb_servo_test/data/traj_raw.csv \
        --interp rb_servo_test/out/traj_interp_50hz.csv -o rb_servo_test/out/traj_plot.png
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_interp import load, time_parametrize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--interp", required=True)
    ap.add_argument("--vmax", type=float, default=20.0, help="traj_interp 때 쓴 값과 동일해야 함")
    ap.add_argument("-o", "--out", default="traj_plot.png")
    args = ap.parse_args()

    names, q = load(args.raw)
    keep = np.r_[True, np.abs(np.diff(q, axis=0)).max(axis=1) > 1e-9]
    q = q[keep]
    tw = time_parametrize(q, np.deg2rad(args.vmax))

    d = np.genfromtxt(args.interp, delimiter=",", skip_header=1)
    ts, qs = d[:, 0], d[:, 1:]
    vs = np.gradient(qs, ts, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = plt.cm.tab10(np.arange(len(names)))
    for i, n in enumerate(names):
        ax1.plot(ts, np.rad2deg(qs[:, i]), color=colors[i], lw=1.5, label=n)
        ax1.plot(tw, np.rad2deg(q[:, i]), "o", color=colors[i], ms=4)
        ax2.plot(ts, np.rad2deg(vs[:, i]), color=colors[i], lw=1.2)
    ax1.set_ylabel("joint angle [deg]")
    ax1.set_title("joint trajectory (line: interpolated, dot: waypoint)")
    ax1.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    ax2.set_ylabel("joint velocity [deg/s]")
    ax2.set_xlabel("time [s]")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
