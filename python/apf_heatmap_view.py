"""Re-plot a saved APF field (.npz from apf_heatmap.py) with a different
color scale, without touching the robot_core/pinocchio/numpy-pickled-snapshot
stack apf_heatmap.py needs.

apf_heatmap.py must run in the same conda env the snapshot .pkl was saved
from (numpy/pinocchio version-sensitive - see the numpy._core.numeric error
if it doesn't). This script only reads the .npz apf_heatmap.py already wrote
(plain numpy arrays, not the pickled snapshot) and re-draws it - it needs
nothing but numpy + matplotlib, so it runs in any env without matching
versions to whatever produced the snapshot.

Usage:
    python python/apf_heatmap_view.py debug/.../apf_field_wp3.npz
    python python/apf_heatmap_view.py debug/.../apf_field_wp3.npz --color-scale log
    python python/apf_heatmap_view.py debug/.../apf_field_wp3.npz --color-scale power --color-gamma 0.2
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_path", help="apf_field_wpN.npz written by apf_heatmap.py")
    parser.add_argument("--output", default=None, help="Output PNG path (default: same name as npz, .png)")
    parser.add_argument(
        "--color-scale", choices=["power", "log", "linear"], default="power",
        help="'power'/'log' compress the near-obstacle spike so the rest of the grid stays visible; "
             "'linear' shows raw cost.")
    parser.add_argument("--color-gamma", type=float, default=0.35, help="Gamma for --color-scale=power")
    parser.add_argument("--quiver-density", type=int, default=12, help="~Number of arrows per axis")
    args = parser.parse_args()

    data = np.load(args.npz_path)
    axis_x, axis_y, cost = data["axis_x"], data["axis_y"], data["cost"]
    force_x, force_y = data["force_x"], data["force_y"]
    joint_x, joint_y = str(data["joint_x"]), str(data["joint_y"])
    q_x, q_y = float(data["q_center"][int(data["joint_x_index"])]), float(data["q_center"][int(data["joint_y_index"])])
    d0, eta = float(data["d0"]), float(data["eta"])
    steps = len(axis_x)

    vmax = float(cost.max())
    if vmax <= 0:
        norm = None
        print("warning: entire saved grid is 0 (outside d0 everywhere) - nothing to compress/rescale")
    elif args.color_scale == "log":
        vmin = max(float(cost[cost > 0].min()), 1e-9) if np.any(cost > 0) else 1e-9
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif args.color_scale == "linear":
        norm = None
    else:
        norm = PowerNorm(gamma=args.color_gamma, vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        cost, origin="lower", aspect="auto", cmap="inferno", norm=norm,
        extent=[axis_x[0], axis_x[-1], axis_y[0], axis_y[-1]])
    fig.colorbar(im, ax=ax, label=f"APF repulsive cost ({args.color_scale})")
    if vmax > 0:
        stride = max(1, steps // max(1, args.quiver_density))
        ax.quiver(
            axis_x[::stride], axis_y[::stride],
            force_x[::stride, ::stride], force_y[::stride, ::stride],
            color="cyan", scale_units="xy")
    ax.plot(q_x, q_y, "w*", markersize=16, markeredgecolor="black", label="waypoint")
    ax.set_xlabel(joint_x)
    ax.set_ylabel(joint_y)
    ax.set_title(f"d0={d0}, eta={eta} ({args.color_scale} scale)")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()

    out_path = pathlib.Path(args.output) if args.output else pathlib.Path(args.npz_path).with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
