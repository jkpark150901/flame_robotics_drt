#!/usr/bin/env python3
"""Straighten a CAD-exported URDF's arms: find joint zero offsets so the links are parallel at q = 0.

A CAD export bakes the assembly pose into the joint origins, so q = 0 is a bent pose instead of the robot's
zero pose. This tool solves for per-joint offsets (theta) that bring each 6-axis arm to the same straight zero
pose as the reference URDFs (axes parallel / perpendicular as in the reference, all joints in one plane, upper
arm and forearm pointing up, tool frame axis-aligned) and bakes them into the joint ``<origin rpy>``:

    T_origin' = T_origin * Rot(axis, theta)      =>   FK'(q = 0) == FK(q = theta)

so a joint value q' of the new URDF corresponds to q' + theta of the original. Nothing else changes: link
geometry, joint axes, limits and every other origin stay as exported. The input file is never modified.

Usage:
    python tools/urdf_zero_offsets.py urdf/model.cad_original.urdf --out urdf/model_zeroed.urdf
    python tools/urdf_zero_offsets.py urdf/model.cad_original.urdf --robots dda --out urdf/model_zeroed.urdf
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

X, Y, Z = np.eye(3)


def rpy_to_R(rpy):
    return R.from_euler("xyz", rpy).as_matrix()


def make_T(xyz, rot):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot, xyz
    return T


def rot_axis(axis, ang):
    a = np.asarray(axis, float)
    return R.from_rotvec(a / np.linalg.norm(a) * ang).as_matrix()


class Urdf:
    """Minimal kinematics read straight from the XML (no dependency on a robotics library's axis conventions)."""

    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.joints, self.parent_of = {}, {}
        for j in self.root.findall("joint"):
            o, ax = j.find("origin"), j.find("axis")
            parse = lambda el, key: [float(v) for v in (el.get(key, "0 0 0") if el is not None else "0 0 0").split()]
            d = dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                     child=j.find("child").get("link"), xyz=np.array(parse(o, "xyz")), rpy=np.array(parse(o, "rpy")),
                     axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else X.copy())
            self.joints[d["name"]] = d
            self.parent_of[d["child"]] = d["name"]

    def origin_T(self, j):
        d = self.joints[j]
        return make_T(d["xyz"], rpy_to_R(d["rpy"]))

    def joint_T(self, j, q=0.0):
        d, T = self.joints[j], self.origin_T(j)
        if d["type"] in ("revolute", "continuous"):
            return T @ make_T(np.zeros(3), rot_axis(d["axis"], q))
        if d["type"] == "prismatic":
            return T @ make_T(d["axis"] / np.linalg.norm(d["axis"]) * q, np.eye(3))
        return T

    def link_T(self, link, q=None, root=None):
        q, chain, cur = q or {}, [], link
        while cur in self.parent_of and cur != root:
            j = self.parent_of[cur]
            chain.append(j)
            cur = self.joints[j]["parent"]
        T = np.eye(4)
        for j in reversed(chain):
            T = T @ self.joint_T(j, q.get(j, 0.0))
        return T

    def joint_world(self, j, q=None, root=None):
        d = self.joints[j]
        T = self.link_T(d["parent"], q, root) @ self.origin_T(j)
        return T[:3, 3].copy(), T[:3, :3] @ (d["axis"] / np.linalg.norm(d["axis"]))


# Each arm: rail link (frame the pose is judged in), arm joints in kinematic order, tool link, and the reference
# zero pose (joint axes in the rail frame from the original per-robot URDFs: base, shoulder, elbow, wrist1..3).
# signed=True: the axis must point exactly that way (fixes the yaw); otherwise the sign is free.
ARMS = {
    "dda": dict(
        rail="link_21_rail_DDA", tool="link_0_DDA",
        joints=["joint_5_arm_dda_link1", "joint_6_arm_dda_link2", "joint_7_arm_dda_link3",
                "joint_8_arm_dda_link4", "joint_9_arm_dda_link5", "joint_11_arm_dda_link6"],
        axes=[Z, -X, -X, -X, Z, -X], signed=[False] * 6,
        up_segments=[(1, 2), (2, 3)],   # shoulder->elbow, elbow->wrist1 point up (+z) as in the reference
    ),
    "rt": dict(
        rail="link_20_rail_RT", tool="link_2_RT",
        joints=["joint_12_arm_rt_link1", "joint_13_arm_rt_link2", "joint_14_arm_rt_link3",
                "joint_17_arm_rt_link4", "joint_15_arm_rt_link5", "joint_16_arm_rt_link6"],
        axes=[Z, -X, -X, Z, -X, Z], signed=[False] * 6,
        up_segments=[(1, 2), (3, 4)],   # shoulder->elbow, wrist1->wrist2 point up
    ),
}


def pose_residuals(urdf, arm, theta, penalty_weight=20.0):
    q = {j: t for j, t in zip(arm["joints"], theta)}
    pos, axes = zip(*(urdf.joint_world(j, q, arm["rail"]) for j in arm["joints"]))
    res = []
    for k in range(1, 6):                       # axis directions (base axis is vertical by construction)
        target = arm["axes"][k]
        res.extend((axes[k] - target) if arm["signed"][k] else np.cross(axes[k], target))
    res.extend(np.cross(axes[0], Z))             # base axis vertical
    for k in range(1, 6):                       # all joints in the plane through the base axis, spanned by z and the shoulder axis
        res.append((pos[k] - pos[0]) @ Y)
    for a, b in arm["up_segments"]:             # branch: segment must point up like in the reference
        dz = (pos[b] - pos[a]) @ Z
        res.append(penalty_weight * max(0.0, 0.05 - dz))
    return np.array(res)


def solve_arm(urdf, arm, seeds=None):
    best = None
    grid = np.deg2rad([-180, -90, 0, 90, 180])
    rng = np.random.default_rng(0)
    starts = [np.zeros(5)] + [np.array(s) for s in itertools.islice(itertools.product(grid, repeat=5), 0, None, 13)]
    starts += [rng.uniform(-np.pi, np.pi, 5) for _ in range(40)]
    for s in starts:
        sol = least_squares(lambda t: pose_residuals(urdf, arm, np.append(t, 0.0)), s, xtol=1e-12, ftol=1e-12)
        cost = sol.cost
        theta = np.arctan2(np.sin(sol.x), np.cos(sol.x))
        score = (round(cost, 10), float(np.abs(theta).sum()))
        if best is None or score < best[0]:
            best = (score, theta)
    theta = np.append(best[1], 0.0)
    # tool roll (joint 6): make the tool frame axis-aligned (nearest of the 24 cube rotations)
    q = {j: t for j, t in zip(arm["joints"], theta)}
    _, a6 = urdf.joint_world(arm["joints"][5], q, arm["rail"])
    R_tool = urdf.link_T(arm["tool"], q, arm["rail"])[:3, :3]
    cube = [np.array(p) * np.array(s) for p in itertools.permutations(np.eye(3)) for s in itertools.product([1, -1], repeat=3)]
    cube = [np.array(m).reshape(3, 3) for m in cube if np.linalg.det(np.array(m).reshape(3, 3)) > 0.5]
    best6 = None
    for t6 in np.deg2rad(np.arange(-180, 180.01, 0.1)):
        Rt = rot_axis(a6, t6) @ R_tool
        dist = min(np.arccos(np.clip((np.trace(Rt.T @ g) - 1) / 2, -1, 1)) for g in cube)
        score = (round(float(dist), 6), abs(t6))
        if best6 is None or score < best6[0]:
            best6 = (score, t6)
    theta[5] = best6[1]
    return theta, best6[0][0]


def evaluate(urdf, arm, theta):
    q = {j: t for j, t in zip(arm["joints"], theta)}
    pos, axes = zip(*(urdf.joint_world(j, q, arm["rail"]) for j in arm["joints"]))
    out = {"axis_error_deg": [], "plane_offset_mm": [], "segments": []}
    for k in range(6):
        t = arm["axes"][k]
        out["axis_error_deg"].append(float(np.degrees(np.arccos(np.clip(abs(axes[k] @ t), -1, 1)))))
    out["plane_offset_mm"] = [float(((pos[k] - pos[0]) @ Y) * 1000) for k in range(6)]
    names = ["base->shoulder", "shoulder->elbow", "elbow->wrist1", "wrist1->wrist2", "wrist2->wrist3"]
    for k in range(5):
        s = pos[k + 1] - pos[k]
        out["segments"].append((names[k], np.round(s, 3).tolist()))
    Rt = urdf.link_T(arm["tool"], q, arm["rail"])[:3, :3]
    out["tool_rotation"] = np.round(Rt, 3).tolist()
    out["axes"] = [np.round(a, 3).tolist() for a in axes]
    return out


def bake(text, urdf, arm, theta):
    """Rewrite each arm joint's <origin rpy=...> with Rot(axis, theta) appended (only those attributes change)."""
    for j, t in zip(arm["joints"], theta):
        if abs(t) < 1e-12:
            continue
        d = urdf.joints[j]
        new_rot = rpy_to_R(d["rpy"]) @ rot_axis(d["axis"], t)
        rpy = R.from_matrix(new_rot).as_euler("xyz")
        pattern = re.compile(r'(<joint\b[^>]*\bname="%s"[^>]*>.*?<origin\b[^>]*?\brpy=")[^"]*(")' % re.escape(j), re.S)
        text, n = pattern.subn(lambda m: m.group(1) + " ".join(f"{v:.10g}" for v in rpy) + m.group(2), text, count=1)
        if n != 1:
            raise RuntimeError(f"could not rewrite <origin rpy> of joint {j}")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urdf")
    ap.add_argument("--out", required=True, help="corrected URDF to write (the input is never modified)")
    ap.add_argument("--robots", default="dda,rt", help="comma list of arms to straighten: dda,rt")
    ap.add_argument("--report", help="JSON report path (default: <out>.offsets.json)")
    args = ap.parse_args()
    src = Path(args.urdf)
    urdf = Urdf(src)
    text = src.read_text(encoding="utf-8")
    report = {"source": str(src), "convention": "q_original = q_new + offset_rad   (new URDF zero = original pose at q = offset)",
              "robots": {}}
    for name in [r.strip().lower() for r in args.robots.split(",") if r.strip()]:
        arm = ARMS[name]
        before = evaluate(urdf, arm, np.zeros(6))
        theta, tool_dist = solve_arm(urdf, arm)
        after = evaluate(urdf, arm, theta)
        text = bake(text, urdf, arm, theta)
        report["robots"][name] = {
            "joints": {j: {"offset_deg": float(np.degrees(t)), "offset_rad": float(t)} for j, t in zip(arm["joints"], theta)},
            "before": before, "after": after, "tool_axis_alignment_residual_deg": float(np.degrees(tool_dist)),
        }
        print(f"== {name.upper()}")
        for j, t in zip(arm["joints"], theta):
            print(f"  {j:30s} offset = {np.degrees(t):9.3f} deg")
        print("  axis error vs reference zero pose (deg): before", np.round(before["axis_error_deg"], 1).tolist(),
              "| after", np.round(after["axis_error_deg"], 2).tolist())
        print("  joints out of the arm plane (mm):        before", np.round(before["plane_offset_mm"], 0).tolist(),
              "| after", np.round(after["plane_offset_mm"], 1).tolist())
        print("  tool frame axis-alignment residual:", f"{np.degrees(tool_dist):.3f} deg")
    out = Path(args.out)
    out.write_text(text, encoding="utf-8")
    rep = Path(args.report) if args.report else out.with_suffix(".offsets.json")
    rep.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out} and {rep}")


if __name__ == "__main__":
    sys.exit(main())
