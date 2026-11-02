"""debug 폴더의 joint_states.csv(플래너 출력, rad, 외부축 2 + 팔 6)를 waypoint_test 형식으로 변환한다.

출력 형식 (experimental/waypoint_test/robot1_*.csv 와 동일): `Order,J1..J6,Speed,Acceleration`, 단위 deg, 외부축 없음.
Speed/Acceleration 은 robot1 파일과 같은 자리 표시값(80/100)이며 이 프로그램은 읽지 않는다.

포함 기준: 대상 로봇(--robot, 기본 dda) / summary.csv 상태가 success / failure_info.json·collisions.json 없음 /
          웨이포인트 3개 이상, 팔 이동량 1deg 이상 / 같은 팔 경로는 한 번만(우선순위: 검증된 폴더 먼저).
제외 기준과 개수는 실행 끝에 출력하고 index.csv 와 README.md 에도 남긴다.

사용: python convert_debug_trajectories.py --debug-dir ~/flame_robotics_drt/debug/debug --out-dir ~/flame_robotics_drt/waypoint_converted
"""
import argparse
import collections
import csv
import hashlib
import os

import numpy as np

# 변환 우선순위(중복 시 앞의 것을 남김)와 폴더별 검증 수준
PRIORITY = [
    ("2026-09-29/rrt_guarantee_test_v2", "독립 재검증(0.02 rad) 무충돌 확인"),
    ("2026-09-29/planner_compare", "플래너 성공 (독립 재검증 없음)"),
    ("2026-09-29/algo_compare", "플래너 성공 (독립 재검증 없음)"),
    ("2026-09-29/single_check", "플래너 성공, collisions.json 없음 (독립 재검증 없음)"),
    ("archive", "플래너 성공 (독립 재검증 없음)"),
]
SUPERSEDED = "2026-09-29/rrt_guarantee_test/"  # README: 구버전, 마이크로 충돌 1건 확인됨 -> 폴더 전체 제외


def rel_top(path):
    for top, note in PRIORITY:
        if path.startswith(top + "/"):
            return top, note
    return None, None


def status_of(jpath, debug_dir):
    """summary.csv 의 해당 인덱스 상태. 폴더명이 'NN_...' 이고 상위에 summary.csv 가 있어야 판정 가능."""
    d = os.path.dirname(jpath)
    sib = set(os.listdir(d))
    if "failure_info.json" in sib:
        return "failed(failure_info.json)"
    if "collisions.json" in sib:
        return "collision(collisions.json)"
    summ = os.path.join(os.path.dirname(d), "summary.csv")
    name = os.path.basename(d)
    if not os.path.exists(summ) or not name[:2].isdigit():
        return "unknown"
    idx = int(name.split("_")[0])
    with open(summ, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                if int(r.get("index", -1)) == idx:
                    return r.get("status", "unknown")
            except ValueError:
                continue
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--robot", default="dda", help="접두 일치(dda / rt). 기본 dda")
    ap.add_argument("--unit", choices=["deg", "rad"], default="deg", help="출력 각도 단위 (rad 이면 열 이름이 J1_rad..J6_rad)")
    args = ap.parse_args()
    dbg, out = os.path.abspath(os.path.expanduser(args.debug_dir)), os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out, exist_ok=True)

    found = []
    for root, _, files in os.walk(dbg):
        if "joint_states.csv" in files:
            found.append(os.path.join(root, "joint_states.csv"))
    found.sort(key=lambda p: ([i for i, (t, _) in enumerate(PRIORITY) if os.path.relpath(p, dbg).startswith(t + "/")] or [99])[0:1] + [p])

    excl = collections.Counter()
    seen = {}
    rows = []
    for jp in found:
        rel = os.path.relpath(jp, dbg)
        with open(jp, encoding="utf-8-sig") as f:
            r = list(csv.reader(f))
        hdr = r[0]
        if len(hdr) < 7:
            excl["형식 불명(열 부족)"] += 1
            continue
        robot = hdr[1].split("_")[0]
        if robot != args.robot:
            excl[f"대상 아닌 로봇({robot})"] += 1
            continue
        if (rel + "/").startswith(SUPERSEDED) or rel.startswith(SUPERSEDED):
            excl["구버전 폴더(rrt_guarantee_test) 제외"] += 1
            continue
        top, note = rel_top(rel)
        if top is None:
            excl["우선순위 목록에 없는 폴더"] += 1
            continue
        st = status_of(jp, dbg)
        if st != "success":
            excl[f"상태 {st}" if not st.startswith("failed") and not st.startswith("collision") else st.split("(")[0] + " 기록 있음"] += 1
            continue
        q = np.array([[float(x) for x in row[1:]] for row in r[1:]])
        arm_rad, ext = q[:, -6:], q[:, :-6]
        arm = np.rad2deg(arm_rad)
        seg = np.abs(np.diff(arm, axis=0)).max(1)
        if len(arm) < 3 or seg.sum() < 1.0:
            excl["너무 짧음(웨이포인트<3 또는 이동<1°)"] += 1
            continue
        key = hashlib.md5(np.round(arm, 2).tobytes()).hexdigest()
        if key in seen:
            rows[seen[key]]["duplicates"] += 1
            excl["중복 경로"] += 1
            continue
        n = len(rows) + 1
        name = f"{args.robot}_{n:04d}_{len(arm)}wp_{seg.sum():.0f}deg.csv"
        with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            suf = "_rad" if args.unit == "rad" else ""
            w.writerow(["Order", *[f"J{k}{suf}" for k in range(1, 7)], "Speed", "Acceleration"])
            for i, a in enumerate(arm, 1):
                vals = np.deg2rad(a) if args.unit == "rad" else a
                w.writerow([i, *[(f"{v:.7f}" if args.unit == "rad" else f"{v:.3f}") for v in vals], 80, 100])
        ext_travel = np.abs(np.diff(ext, axis=0)).sum(0) if ext.shape[1] else np.zeros(2)
        seen[key] = len(rows)
        rows.append({"file": name, "robot": robot, "n_wp": len(arm), "path_deg": round(float(seg.sum()), 1), "max_seg_deg": round(float(seg.max()), 1),
                     **{f"start_J{i+1}": round(float(arm[0, i]), 1) for i in range(6)}, **{f"end_J{i+1}": round(float(arm[-1, i]), 1) for i in range(6)},
                     "ext_linear_track_travel_m": round(float(ext_travel[0]), 3), "ext_carriage_travel": round(float(ext_travel[1]), 3),
                     "status": st, "verification": note, "source": rel, "duplicates": 0})

    with open(os.path.join(out, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"스캔 {len(found)}개 -> 변환 {len(rows)}개 ({out})")
    for k, v in sorted(excl.items(), key=lambda x: -x[1]):
        print(f"  제외: {k}: {v}")
    return rows, excl, len(found)


if __name__ == "__main__":
    main()
