"""보고서(report.md)에 쓰인 실험 데이터만 모아 데이터 묶음을 만든다. 원본 실행 폴더는 수정하지 않는다.

구성: runs/(비교 대상 24회), aux_runs/(설명 그림용 2회), inputs/(입력 궤적), urdf/(rb20 URDF + 충돌 메시), tables/(요약 표), DATA_SPEC.md(명세서)

사용: cd rb_servo_gui && PYTHONPATH=src python docs/report/make_report_dataset.py [--runs ~/rb_servo_gui_runs] [--out <출력 폴더>]
"""
import argparse
import csv
import glob
import json
import os
import shutil

import numpy as np

from rb_servo_gui import analysis as A
from rb_servo_gui.settings import Settings
from rb_servo_gui.surface import RobotSurface, compare_poses

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
URDF = os.path.abspath(os.path.join(REPO, "..", "urdf", "rb20_1900es_RT.urdf"))
SINCE = "20261008_160000"
TRAJ = {"waypoints_sample.csv": "A", "waypoints_robot1.csv": "B", "waypoints_robot2.csv": "C"}
AUX = {"20261008_132802_servo_step10": "종료 오버슈트 사례 (마지막 점 반복 없음): 3.2절 그림 fig_start_end",
       "20261008_162955_servo_step10": "지표 설명 그림 원본 (fig_dtw, fig_error, fig_ripple, fig_wobble_real), 비교에는 미사용"}
DATA_FILES = ("config.json", "traj_ref.csv", "traj_sub10.csv", "follow_*.csv", "compare_*.json", "compare_*_pairs.csv",
              "velocity_*.csv", "smooth_*.csv")


def method(r):
    p = r["params"]
    if r["mode"] == "jb2":
        return f"jb2 blend {p['blend']:g}"
    return "servo"


def in_set(r):
    p = r["params"]
    if r["mode"] == "jb2":
        return p.get("accel") == 120.0
    return p.get("gain") == 1.0 and p.get("alpha") == 1.0


def load_run(d):
    j = glob.glob(os.path.join(d, "compare_*.json"))
    if not j:
        return None
    r = json.load(open(j[0], encoding="utf-8"))
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))["effective"]
    r["traj"] = TRAJ[os.path.basename(cfg["traj_path"])]
    r["traj_file"] = os.path.basename(cfg["traj_path"])
    r["real"] = "REAL" in os.path.basename(d)
    r["name"] = os.path.basename(d)
    r["dir"] = d
    return r


def copy_data(src, dst):
    os.makedirs(dst, exist_ok=True)
    for pat in DATA_FILES:
        for f in glob.glob(os.path.join(src, pat)):
            shutil.copy2(f, dst)


def load_pairs(d):
    pj = glob.glob(os.path.join(d, "compare_*_pairs.csv"))[0]
    with open(pj, encoding="utf-8") as f:
        f.readline()
        rows = list(csv.reader(f))
    a = np.array([[float(v) for v in x] for x in rows[1:]])
    return a[:, 0], a[:, 4:10], a[:, 10:16]


def fmt(v, n=4):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return v


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("저장", os.path.relpath(path))


LINK_LABEL = {"rt_link1_0": "link1", "rt_link2_0": "link2", "rt_link3_0": "link3", "rt_link4_0": "link4",
              "rt_link5_0": "link5", "rt_link6_0": "link6", "rt_link_end_0": "link_end"}


def link_table(runs, robot, out):
    """궤적·방식·모드별 링크 변위: 링크마다 '그 순간 링크의 최악 점 변위' 시계열(매칭쌍 전체)의 최댓값, P95, 평균."""
    rows = []
    for r in sorted(runs, key=lambda r: (r["traj"], ["jb2 blend 0", "jb2 blend 50", "servo", "jb2 blend 100"].index(method(r)), r["real"])):
        _, qm, qr = load_pairs(r["dir"])
        L = {n: [] for n in LINK_LABEL}
        for i in range(len(qm)):
            lm = compare_poses(robot, np.deg2rad(qm[i]), np.deg2rad(qr[i]))["link_max"]
            for n in LINK_LABEL:
                L[n].append(lm[n])
        for n, lab in LINK_LABEL.items():
            x = np.array(L[n])
            rows.append([r["traj"], method(r), "실기" if r["real"] else "시뮬레이션", r["name"], lab,
                         f"{x.max():.3f}", f"{np.percentile(x, 95):.3f}", f"{x.mean():.3f}"])
    write_csv(os.path.join(out, "tables", "link_displacement_rb20.csv"),
              ["trajectory", "method_group", "mode", "run", "link", "disp_max_mm", "disp_p95_mm", "disp_mean_mm"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="~/rb_servo_gui_runs")
    ap.add_argument("--out", default="~/rb_servo_gui_runs/report_dataset_20261008")
    ap.add_argument("--update-link-table", action="store_true", help="기존 묶음에 링크 변위 표만 다시 쓴다")
    a = ap.parse_args()
    root, out = os.path.expanduser(a.runs), os.path.expanduser(a.out)
    if a.update_link_table:
        runs = [r for r in (load_run(d) for d in sorted(glob.glob(os.path.join(root, "*_step*")))
                            if os.path.basename(d)[:15] >= SINCE) if r and in_set(r)]
        link_table(runs, RobotSurface(URDF, n_points=600), out)
        return
    if os.path.exists(out):
        raise SystemExit(f"출력 폴더가 이미 있습니다: {out} (덮어쓰지 않음)")
    for sub in ("runs", "aux_runs", "inputs", "urdf", "tables"):
        os.makedirs(os.path.join(out, sub))

    runs = [r for r in (load_run(d) for d in sorted(glob.glob(os.path.join(root, "*_step*")))
                        if os.path.basename(d)[:15] >= SINCE) if r and in_set(r)]
    assert len(runs) == 24, len(runs)
    robot = RobotSurface(URDF, n_points=600)
    s = Settings(urdf_path=URDF, surface_samples=10 ** 9)

    # ---- 실행 데이터 + 표면 편차 시계열
    for r in runs:
        dst = os.path.join(out, "runs", r["name"])
        copy_data(r["dir"], dst)
        t, qm, qr = load_pairs(r["dir"])
        surf = A.surface_errors(s, qm, qr, t - t[0], log=lambda *_: None)
        r["surf"] = surf["summary"]
        r["surf_links"] = surf["links"]
        stem = os.path.basename(glob.glob(os.path.join(r["dir"], "compare_*_pairs.csv"))[0])[len("compare_"):-len("_pairs.csv")]
        sr = surf["series"]
        write_csv(os.path.join(dst, f"surface_rb20_{stem}.csv"),
                  ["t_track", "disp_max_mm", "disp_p95_mm", "disp_mean_mm", "chamfer_mm", "hausdorff_mm"],
                  [[f"{surf['t'][k]:.4f}"] + [f"{sr[c][k]:.3f}" for c in ("disp_max", "disp_p95", "disp_mean", "chamfer", "hausdorff")]
                   for k in range(len(surf["idx"]))])
    for name in AUX:
        copy_data(os.path.join(root, name), os.path.join(out, "aux_runs", name))

    # ---- 입력, URDF
    for f in TRAJ:
        shutil.copy2(os.path.join(REPO, "examples", f), os.path.join(out, "inputs", f))
    shutil.copy2(URDF, os.path.join(out, "urdf"))
    udir = os.path.dirname(URDF)
    for rel in ("meshes/rb20_1900es/collision", "meshes/link_rt.stl"):
        src = os.path.join(udir, rel)
        dst = os.path.join(out, "urdf", rel)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    # ---- 표 1: 실행 목록
    rows = []
    for r in sorted(runs, key=lambda r: (method(r) != "servo", method(r), r["traj"], r["real"])):
        p = r["params"]
        rows.append([r["name"], "비교 대상", method(r), r["traj"], r["traj_file"], "실기" if r["real"] else "시뮬레이션",
                     r["params"]["method"], fmt(p.get("speed")), fmt(p.get("accel")), fmt(p.get("blend")), fmt(p.get("t1")), fmt(p.get("t2")),
                     fmt(p.get("gain")), fmt(p.get("alpha")), fmt(p.get("final_hold")), r["step"]])
    for name, role in AUX.items():
        r = load_run(os.path.join(root, name))
        p = r["params"]
        rows.append([name, "보조: " + role, "servo", r["traj"], r["traj_file"], "실기" if r["real"] else "시뮬레이션", p["method"], "", "", "",
                     fmt(p.get("t1")), fmt(p.get("t2")), fmt(p.get("gain")), fmt(p.get("alpha")), fmt(p.get("final_hold")), r["step"]])
    write_csv(os.path.join(out, "tables", "runs_index.csv"),
              ["run", "role", "method_group", "trajectory", "trajectory_file", "mode", "command", "speed", "accel", "blend", "t1", "t2",
               "gain", "alpha", "final_hold", "subsample_step"], rows)

    # ---- 표 2: 실행별 지표
    keys = ["t_track", "path_dev_mean", "path_dev_p95", "path_dev_max", "cover_max", "progress_max", "max_step", "start_err", "end_err",
            "smooth_ripple", "smooth_wobble", "smooth_stall", "t_latency"]
    sk = ["surf_disp_mean", "surf_disp_p95", "surf_disp_max", "surf_chamfer_max", "surf_hausdorff_max", "surf_worst_link"]
    rows = []
    for r in sorted(runs, key=lambda r: (method(r) != "servo", method(r), r["traj"], r["real"])):
        v = {k: x for k, x in r["verdict"].items() if "리플" not in k}
        fails = [k for k, x in v.items() if not x]
        rows.append([r["name"], method(r), r["traj"], "실기" if r["real"] else "시뮬레이션"] + [fmt(r["metrics"].get(k)) for k in keys]
                    + [fmt(r["surf"][k], 3) for k in sk] + ["PASS" if not fails else "FAIL", "; ".join(fails)])
    write_csv(os.path.join(out, "tables", "run_metrics.csv"), ["run", "method_group", "trajectory", "mode"] + keys + sk + ["verdict", "failed_items"], rows)

    # ---- 표 3: 방식별 요약
    groups = ["jb2 blend 0", "jb2 blend 50", "servo", "jb2 blend 100"]
    mk = ["path_dev_p95", "path_dev_max", "smooth_ripple", "smooth_wobble"]
    rows = []
    for g in groups:
        rr = [r for r in runs if method(r) == g]
        row = [g, len(rr)]
        for k in mk:
            x = [r["metrics"][k] for r in rr]
            row += [fmt(float(np.median(x))), fmt(min(x)), fmt(max(x))]
        for k in ("surf_disp_mean", "surf_disp_p95", "surf_disp_max"):
            x = [r["surf"][k] for r in rr]
            row += [fmt(float(np.median(x)), 3), fmt(min(x), 3), fmt(max(x), 3)]
        rows.append(row)
    hdr = ["method_group", "n_runs"]
    for k in mk + ["surf_disp_mean", "surf_disp_p95", "surf_disp_max"]:
        hdr += [f"{k}_median", f"{k}_min", f"{k}_max"]
    write_csv(os.path.join(out, "tables", "method_summary.csv"), hdr, rows)

    # ---- 표 4: 시뮬레이션 대 실기 경로 거리
    def motion(d):
        f = max(glob.glob(os.path.join(d, "follow_*.csv")), key=os.path.getmtime)
        t, ph, q, st, meta = A.load_follow(f)
        m = "servo" if "servo" in meta["marks"] else "jb2"
        sel = np.array([x.startswith(m) and not x.endswith("move_to_start") for x in ph])
        q = q[sel]
        return q[int(np.argmax(np.abs(q - q[0]).max(1) > 0.2)):]

    def poly_dist(P, Q):
        a_, ab = Q[:-1], np.diff(Q, axis=0)
        den = np.maximum((ab * ab).sum(1), 1e-12)
        return np.array([np.linalg.norm(a_ + np.clip(((p - a_) * ab).sum(1) / den, 0, 1)[:, None] * ab - p, axis=1).min() for p in P])

    rows = []
    for r in [r for r in runs if r["real"]]:
        sim = [x for x in runs if not x["real"] and method(x) == method(r) and x["traj"] == r["traj"]][0]
        d = poly_dist(motion(r["dir"])[::2], motion(sim["dir"]))
        rows.append([method(r), r["traj"], sim["name"], r["name"], fmt(float(np.percentile(d, 95))), fmt(float(d.max())),
                     fmt(sim["metrics"]["path_dev_p95"]), fmt(r["metrics"]["path_dev_p95"]), fmt(sim["surf"]["surf_disp_p95"], 3), fmt(r["surf"]["surf_disp_p95"], 3)])
    write_csv(os.path.join(out, "tables", "sim_real_pairs.csv"),
              ["method_group", "trajectory", "sim_run", "real_run", "path_dist_p95_deg", "path_dist_max_deg", "sim_path_dev_p95", "real_path_dev_p95",
               "sim_surf_disp_p95", "real_surf_disp_p95"], rows)

    # ---- 표 5: 지령 꺾은선의 현 오차 (궤적별)
    rows = []
    for f, tj in TRAJ.items():
        d = [r for r in runs if r["traj"] == tj][0]["dir"]
        ra = np.loadtxt(os.path.join(d, "traj_ref.csv"), delimiter=",", skiprows=1)
        sa = np.loadtxt(os.path.join(d, "traj_sub10.csv"), delimiter=",", skiprows=1)
        qr, qs = np.rad2deg(ra[:, 1:7]), np.rad2deg(sa[:, 1:7])
        dl = poly_dist(qr, qs)
        rows.append([tj, f, fmt(float(ra[-1, 0]), 3), len(ra), len(sa), fmt(float(np.linalg.norm(np.diff(qr, axis=0), axis=1).sum()), 1),
                     fmt(float(np.percentile(dl, 95))), fmt(float(dl.max()))])
    write_csv(os.path.join(out, "tables", "trajectory_chord_error.csv"),
              ["trajectory", "file", "duration_s", "n_ref_points_50Hz", "n_command_points", "path_length_deg", "chord_err_p95_deg", "chord_err_max_deg"], rows)

    # ---- 표 6: 궤적별 링크 변위
    link_table(runs, robot, out)

    # ---- 표 7: 관절별 1° 감도
    d = [r for r in runs if r["traj"] == "A"][0]["dir"]
    q0 = np.deg2rad(load_pairs(d)[2][0])
    rows = []
    for k in range(6):
        e = np.zeros(6)
        e[k] = np.deg2rad(1.0)
        rows.append([f"J{k + 1}", fmt(compare_poses(robot, q0 + e, q0)["disp_max"], 2)])
    write_csv(os.path.join(out, "tables", "joint_sensitivity_rb20.csv"), ["joint", "surface_disp_max_mm_per_1deg"], rows)

    shutil.copy2(os.path.join(HERE, "DATA_SPEC.md"), os.path.join(out, "DATA_SPEC.md"))
    print("완료:", out)


if __name__ == "__main__":
    main()
