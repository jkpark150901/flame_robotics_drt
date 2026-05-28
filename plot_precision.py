#!/usr/bin/env python3
"""
로봇 TCP 궤적 vs 모션캡처 RB 궤적 시각화

사용법:
  python plot_precision.py                        # 최신 svd 파일
  python plot_precision.py --verify               # 최신 calibration_verify 파일
  python plot_precision.py --file <csv>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def latest(pattern: str) -> pd.DataFrame:
    files = sorted(Path(".").glob(pattern))
    if not files:
        raise FileNotFoundError(f"{pattern} 파일을 찾을 수 없습니다.")
    df = pd.read_csv(files[-1])
    df["source"] = files[-1].stem
    return df


def plot_trajectories(df: pd.DataFrame, title: str = ""):
    tcp_x, tcp_y, tcp_z = df["tcp_x_m"], df["tcp_y_m"], df["tcp_z_m"]
    rb_x,  rb_y,  rb_z  = df["rb_aligned_x_m"], df["rb_aligned_y_m"], df["rb_aligned_z_m"]

    mean_err = df["error_mm"].mean()
    std_err  = df["error_mm"].std()
    max_err  = df["error_mm"].max()

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(
        f"{title or df['source'].iloc[0]}\n"
        f"mean = {mean_err:.1f} mm  |  std = {std_err:.1f} mm  |  max = {max_err:.1f} mm",
        fontsize=11, fontweight="bold",
    )

    # ── YZ ───────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(tcp_y, tcp_z, color="steelblue", lw=1.5, label="TCP (robot)")
    ax1.plot(rb_y,  rb_z,  color="tomato",    lw=1.5, label="RB (mocap)", linestyle="--")
    ax1.set_aspect("equal")
    ax1.set_xlabel("Y (m)"); ax1.set_ylabel("Z (m)")
    ax1.set_title("YZ")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── XY ───────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(tcp_x, tcp_y, color="steelblue", lw=1.5, label="TCP (robot)")
    ax2.plot(rb_x,  rb_y,  color="tomato",    lw=1.5, label="RB (mocap)", linestyle="--")
    ax2.set_aspect("equal")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.set_title("XY")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── XZ ───────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(tcp_x, tcp_z, color="steelblue", lw=1.5, label="TCP (robot)")
    ax3.plot(rb_x,  rb_z,  color="tomato",    lw=1.5, label="RB (mocap)", linestyle="--")
    ax3.set_aspect("equal")
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Z (m)")
    ax3.set_title("XZ")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",   help="CSV 파일 경로")
    parser.add_argument("--verify", action="store_true",
                        help="calibration_verify 모드 (기본: precision_eval_svd)")
    parser.add_argument("--save",   action="store_true", help="PNG 저장")
    args = parser.parse_args()

    if args.file:
        df = pd.read_csv(args.file)
        df["source"] = Path(args.file).stem
    elif args.verify:
        df = latest("calibration_verify_*.csv")
    else:
        df = latest("precision_eval_svd_*.csv")
        df = df[df["error_mm"] < 500].reset_index(drop=True)

    fig = plot_trajectories(df)

    if args.save:
        out = df["source"].iloc[0] + ".png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"저장: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
