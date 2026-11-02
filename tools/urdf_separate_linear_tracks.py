"""Make the two linear tracks of the combined CAD model (urdf/model.urdf) independent and zero them at the -x end.

The CAD export drives both scan bodies with one joint: joint_1_scan_body_DDA is a <mimic> of joint_3_scan_body_RT, and
both have the CAD assembly position as zero (limits -3.601 .. 1.769, +q moving towards world -x). After this script:
- no <mimic>: DDA and RT tracks move independently;
- q = 0 is the end of the track with the smallest world x, +q moves towards world +x, limits [0, travel];
- the geometry is unchanged: the joint origin is moved to the old -x end (origin @ trans(axis * q_end)) and the axis is
  flipped when +q pointed to -x, so every reachable pose is still reachable (q_new = |q_old - q_end|).
Only the two <joint> blocks are rewritten (text edit, the rest of the file keeps its formatting). The per-robot URDFs
(tools/urdf_export_app_models.py) already used this convention; the exporter now reads the direction from the model, so
it gives the same result for the old and the converted model.

    python tools/urdf_separate_linear_tracks.py [--model urdf/model.urdf] [--dry-run]
    python tools/urdf_separate_linear_tracks.py --shift 0.5   # on a converted model: zero moved 0.5 m towards +x

--shift d (converted model): moves q = 0 by d along +q (origin @ trans(axis * d), upper -= d), e.g. so the folded arm at
q = 0 stays clear of the positioner; the track then cannot go below that point (lower stays 0). q_new = q_old - d.
"""
import argparse
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urdf_zero_offsets import Urdf, make_T, rpy_to_R  # noqa: E402
from urdf_export_app_models import clean_axis, rpy_of, trans, vec_str  # noqa: E402

TRACK_JOINTS = ("joint_1_scan_body_DDA", "joint_3_scan_body_RT")


def _block(text, name):
    m = re.search(r'(\s*)<joint name="%s" type="prismatic">.*?</joint>' % re.escape(name), text, flags=re.S)
    if m is None:
        raise SystemExit(f"joint {name} not found")
    return m


def convert(text, kin):
    report = {}
    for name in TRACK_JOINTS:
        m = _block(text, name)
        block = m.group(0)
        d = kin.joints[name]
        lo = float(re.search(r'lower="([^"]+)"', block).group(1))
        hi = float(re.search(r'upper="([^"]+)"', block).group(1))
        ax = clean_axis(d["axis"])
        _, ax_world = kin.joint_world(name)
        T = make_T(d["xyz"], rpy_to_R(d["rpy"]))
        if ax_world[0] >= 0.0:      # +q already towards +x: zero at the lower limit
            T_new, ax_new, q_end = T @ trans(ax * lo), ax, lo
        else:                       # +q towards -x: zero at the upper limit, axis flipped
            T_new, ax_new, q_end = T @ trans(ax * hi), -ax, hi
        travel = hi - lo
        new = block
        new = re.sub(r'<origin xyz="[^"]*" rpy="[^"]*"\s*/>',
                     f'<origin xyz="{vec_str(T_new[:3, 3])}" rpy="{vec_str(rpy_of(T_new[:3, :3]))}" />', new, count=1)
        new = re.sub(r'<axis xyz="[^"]*"\s*/>', f'<axis xyz="{vec_str(ax_new)}" />', new, count=1)
        new = re.sub(r'lower="[^"]*"', 'lower="0"', new, count=1)
        new = re.sub(r'upper="[^"]*"', f'upper="{travel:.6g}"', new, count=1)
        new = re.sub(r'\s*<mimic [^>]*/>', "", new)
        text = text[:m.start()] + new + text[m.end():]
        report[name] = {"old_limits": [lo, hi], "zero_was_at_old_q": q_end, "new_limits": [0.0, travel],
                        "axis_world_old": np.round(ax_world, 6).tolist(), "mimic_removed": "<mimic" in block}
    return text, report


def shift_zero(text, kin, d):
    report = {}
    for name in TRACK_JOINTS:
        m = _block(text, name)
        block = m.group(0)
        info = kin.joints[name]
        lo = float(re.search(r'lower="([^"]+)"', block).group(1))
        hi = float(re.search(r'upper="([^"]+)"', block).group(1))
        if abs(lo) > 1e-9 or "<mimic" in block:
            raise SystemExit(f"{name}: run the conversion first (lower must be 0, no mimic)")
        if not 0.0 < d < hi:
            raise SystemExit(f"{name}: shift {d} outside (0, {hi})")
        ax = clean_axis(info["axis"])
        T_new = make_T(info["xyz"], rpy_to_R(info["rpy"])) @ trans(ax * d)
        new = re.sub(r'<origin xyz="[^"]*" rpy="[^"]*"\s*/>',
                     f'<origin xyz="{vec_str(T_new[:3, 3])}" rpy="{vec_str(rpy_of(T_new[:3, :3]))}" />', block, count=1)
        new = re.sub(r'upper="[^"]*"', f'upper="{hi - d:.6g}"', new, count=1)
        text = text[:m.start()] + new + text[m.end():]
        report[name] = {"shift": d, "old_limits": [lo, hi], "new_limits": [0.0, hi - d]}
    return text, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="urdf/model.urdf")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shift", type=float, default=None, help="move the zero of the converted tracks by this (m)")
    a = ap.parse_args()
    text = open(a.model, encoding="utf-8").read()
    if a.shift is not None:
        new_text, report = shift_zero(text, Urdf(a.model), float(a.shift))
        for name, r in report.items():
            print(name, r)
        if not a.dry_run:
            backup = a.model + f".before_shift_{a.shift:g}"
            if not os.path.exists(backup):
                shutil.copyfile(a.model, backup)
            with open(a.model, "w", encoding="utf-8") as f:
                f.write(new_text)
            print(f"written {a.model} (backup {backup})")
        return
    if all("<mimic" not in _block(text, n).group(0) and 'lower="0"' in _block(text, n).group(0) for n in TRACK_JOINTS):
        print("already converted (no mimic, lower=0) - nothing to do")
        return
    new_text, report = convert(text, Urdf(a.model))
    for name, r in report.items():
        print(name, r)
    if a.dry_run:
        return
    backup = a.model + ".before_track_split"
    if not os.path.exists(backup):
        shutil.copyfile(a.model, backup)
    with open(a.model, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"written {a.model} (backup {backup})")


if __name__ == "__main__":
    main()
