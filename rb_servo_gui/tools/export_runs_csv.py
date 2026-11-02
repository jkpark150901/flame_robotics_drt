"""실행 폴더들을 지정한 시각 이후 것만 골라 한 개의 CSV(실행당 한 줄)로 합친다. 실행 폴더는 수정하지 않는다.

표면 지표(surf_*)가 분석 결과에 없으면 폴더의 compare_*_pairs.csv (측정 자세 / 매칭된 원본 자세, deg) 로 URDF 메시에서 계산해 채운다.

예)  python tools/export_runs_csv.py --since 20261008_160000 --urdf ../urdf/rb10_1300e.urdf
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rb_servo_gui.analysis import SURFACE_KEYS, surface_errors  # noqa: E402
from rb_servo_gui.settings import Settings  # noqa: E402


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        f.readline()
        rows = list(csv.reader(f))
    hdr, data = rows[0], rows[1:]
    ix = {h: i for i, h in enumerate(hdr)}
    a = np.array([[float(v) for v in r] for r in data])
    cols = lambda pre: a[:, [ix[f"{pre}{i}"] for i in range(6)]]  # noqa: E731
    return a[:, ix["t_track"]], cols("meas_q"), cols("ref_q")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="~/rb_servo_gui_runs")
    ap.add_argument("--since", required=True, help="폴더명 시각 YYYYMMDD_HHMMSS (이 시각 이후)")
    ap.add_argument("--urdf", default="", help="표면 지표가 없는 실행에 쓸 URDF (비우면 계산 생략)")
    ap.add_argument("--out", default="")
    ap.add_argument("--samples", type=int, default=0, help="표면 변위 계산에 쓸 매칭쌍 수 (0 = 전체; 일부만 쓰면 순간 최댓값을 놓칠 수 있음)")
    a = ap.parse_args()
    root = os.path.expanduser(a.runs)
    s = Settings(urdf_path=os.path.expanduser(a.urdf), surface_samples=a.samples if a.samples > 0 else 10**9)
    recs = []
    for d in sorted(glob.glob(os.path.join(root, "*_step*"))):
        name = os.path.basename(d)
        if not os.path.isdir(d) or name[:15] < a.since:
            continue
        for j in sorted(glob.glob(os.path.join(d, "compare_*.json"))):
            r = json.load(open(j, encoding="utf-8"))
            M = dict(r["metrics"])
            src = "recorded" if M.get("surf_disp_max") is not None else ""
            if not src and s.urdf_path:
                pj = j.replace(".json", "_pairs.csv")
                if os.path.exists(pj):
                    t, qm, qr = load_pairs(pj)
                    surf = surface_errors(s, qm, qr, t - t[0], log=lambda *_: None)
                    M.update(surf["summary"])
                    src = os.path.basename(s.urdf_path)
            recs.append({"run": name, "real": "REAL" in name, "mode": r["mode"], "step": r["step"],
                         "params": json.dumps(r["params"], ensure_ascii=False), **M,
                         "surface_urdf": src, "surface_samples": ("all" if a.samples <= 0 else a.samples) if src and src != "recorded" else "", "all_pass": r.get("all_pass")})
            print(name, M.get("surf_disp_max"))
    if not recs:
        sys.exit("해당하는 실행이 없습니다")
    keys = list(dict.fromkeys(k for r in recs for k in r))
    out = os.path.expanduser(a.out) or os.path.join(root, f"combined_since_{a.since}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀에서 한글/기호 깨짐 방지
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in recs:
            w.writerow({k: ("" if r.get(k) is None else f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k)) for k in keys})
    print(f"\n{len(recs)}개 실행 → {out}")


if __name__ == "__main__":
    main()
