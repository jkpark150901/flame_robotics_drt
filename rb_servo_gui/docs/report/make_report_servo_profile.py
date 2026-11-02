"""보고서 3.8절(servo 프로파일 타이밍, 2026-10-14 실험)의 그림과 표를 만든다. 실행 폴더는 복사해서 현재 분석 코드로 다시 계산한다(원본 불변).

사용: cd rb_servo_gui && PYTHONPATH=src python docs/report/make_report_servo_profile.py [--runs ~/rb_servo_gui_runs] [--out <tables 폴더>]
"""
import argparse
import csv
import glob
import json
import os
import shutil
import tempfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.figure import Figure

from rb_servo_gui import analysis as A
from rb_servo_gui.settings import Settings

have = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic"):
    if _f in have:
        matplotlib.rcParams["font.sans-serif"] = [_f] + list(matplotlib.rcParams["font.sans-serif"])
        break
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "img")
TJ = {"sample": "A", "robot1": "B", "robot2": "C"}
# 보고서에 쓰는 실행: 10/14 16:12 이후 profile 타이밍(주기 0.02 s), 분석 가능한 것 전부
SINCE = "20261014_163047"


def poly_dist(P, Q):
    a, ab = Q[:-1], np.diff(Q, axis=0)
    den = np.maximum((ab * ab).sum(1), 1e-12)
    return np.array([np.linalg.norm(a + np.clip(((p - a) * ab).sum(1) / den, 0, 1)[:, None] * ab - p, axis=1).min() for p in P])


def analyze_all(root, tmp):
    rows = []
    for src in sorted(glob.glob(os.path.join(root, "20261014_*servo*"))):
        run = os.path.basename(src)
        if run[:15] < SINCE:
            continue
        d = os.path.join(tmp, run)
        shutil.copytree(src, d)
        for f in glob.glob(d + "/compare_*") + glob.glob(d + "/smooth_*") + glob.glob(d + "/velocity_*"):
            os.remove(f)
        c = json.load(open(src + "/config.json", encoding="utf-8"))["effective"]
        base = Settings(**{k: v for k, v in c.items() if k in Settings.__dataclass_fields__})
        res = A.analyze(d, base, summary_dir=tmp, log=lambda *_: None)
        j = json.load(open(glob.glob(d + "/compare_*_servo.json")[0], encoding="utf-8"))
        p, M = j["params"], j["metrics"]
        dd = A.profile_data(d, "servo")
        t, q = dd["t_rel"], dd["qw"]
        v = A.velocity(t, q, 0.1)
        a = A.velocity(t, v, 0.1)
        pa = np.loadtxt(glob.glob(d + "/compare_*_pairs.csv")[0], delimiter=",", skiprows=2)
        ref_deg = np.rad2deg(np.loadtxt(d + "/traj_ref.csv", delimiter=",", skiprows=1)[:, 1:7])
        near = poly_dist(pa[:, 4:10], ref_deg)  # 시간 매칭 없이 원본 경로(꺾은선)까지의 최단 거리
        rows.append(dict(near_p95=float(np.percentile(near, 95)), near_max=float(near.max()), near_mean=float(near.mean()),
                         n_spike=int((pa[:, 22] - near > 0.05).sum()), n=len(pa), run=run, real="REAL" in run, traj=TJ[os.path.basename(c["traj_path"]).replace("waypoints_", "").replace(".csv", "")],
                         speed=p["speed"], accel=p["accel"], gain=p["gain"], alpha=p["alpha"], period=p.get("period"),
                         T_ideal=dd["ideal"]["T"], M=M, vmax=float(np.abs(v).max()), amax=float(np.abs(a).max()),
                         cmd_v=p.get("cmd_speed_max"), cmd_a=p.get("cmd_accel_max"), dir=d))
    return rows


def qw_of(r):
    return A.profile_data(r["dir"], "servo")["qw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="~/rb_servo_gui_runs")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    root = os.path.expanduser(a.runs)
    tmp = tempfile.mkdtemp(prefix="servo_profile_")
    rows = analyze_all(root, tmp)
    print(len(rows), "개 실행")
    real = [r for r in rows if r["real"]]
    # ---- 그림 1: 실기 vs 시뮬(궤적 C, 30/50) — 속도·가속도 한계와 이상 프로파일
    rc = [r for r in real if r["traj"] == "C"][0]
    sc = [r for r in rows if not r["real"] and r["traj"] == "C" and (r["speed"], r["accel"]) == (rc["speed"], rc["accel"])][-1]
    fig = Figure(figsize=(12, 7.5))
    ax = fig.subplots(3, 1, sharex=True)
    dd = A.profile_data(rc["dir"], "servo")
    I = dd["ideal"]
    for lab, r, col, ls in (("실기", rc, "C3", "-"), ("시뮬레이션", sc, "C0", "-")):
        d2 = A.profile_data(r["dir"], "servo")
        v = A.velocity(d2["t_rel"], d2["qw"], 0.1)
        ac = A.velocity(d2["t_rel"], v, 0.1)
        ax[0].plot(d2["t_rel"], np.abs(v).max(1), color=col, lw=1.2, label=f"{lab}  최대 {np.abs(v).max():.1f} deg/s")
        ax[1].plot(d2["t_rel"], np.abs(ac).max(1), color=col, lw=1.0, alpha=0.8, label=f"{lab}  최대 {np.abs(ac).max():.0f} deg/s²")
        q = d2["qw"]
        qi = np.column_stack([np.interp(d2["t_rel"], I["t"], I["q"][:, j]) for j in range(6)])
        ax[2].plot(d2["t_rel"], np.linalg.norm(q - qi, axis=1), color=col, lw=1.0, label=lab)
    ax[0].plot(I["t"], np.abs(I["v"]).max(1), ":", color="k", lw=1.6, label="이상 프로파일")
    ax[1].plot(I["t"], np.abs(I["a"]).max(1), ":", color="k", lw=1.2, label="이상 프로파일")
    ax[0].axhline(rc["speed"], color="r", ls="--", lw=0.8, label=f"한계 {rc['speed']:g}")
    ax[1].axhline(rc["accel"], color="r", ls="--", lw=0.8, label=f"한계 {rc['accel']:g}")
    ax[0].set_ylabel("관절 최대 |속도| [deg/s]")
    ax[1].set_ylabel("관절 최대 |가속도| [deg/s²]\n(±0.05 s 차분)")
    ax[2].set_ylabel("이상 프로파일과의 위치 차이\n(6관절 L2) [deg]")
    ax[2].set_xlabel("모션 시작 후 시간 [s]")
    for x in ax:
        x.grid(alpha=0.3)
        x.legend(fontsize=8, loc="upper right", ncol=2)
    ax[0].set_title(f"궤적 C, speed {rc['speed']:g} / accel {rc['accel']:g}, 주기 20 ms: 실기 대 시뮬레이션", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "r_servo_profile_real.png"), dpi=110)
    print("저장 r_servo_profile_real.png")
    # ---- 그림 2: 시뮬레이션, 궤적 B — 한계(speed/accel)에 따른 추종 오차·리플, alpha 영향
    sb = [r for r in rows if not r["real"] and r["traj"] == "B" and r["cmd_v"]]  # 지령 한계 수정 이후 실행만
    fig = Figure(figsize=(12, 4.2))
    ax = fig.subplots(1, 3)
    for r in sb:
        mk = "s" if r["alpha"] < 1 else "o"
        x = r["accel"]
        col = {50.0: "C3", 40.0: "C1", 30.0: "C2", 20.0: "C0"}.get(r["speed"], "0.5")
        for a_, val in ((ax[0], r["near_p95"]), (ax[1], r["M"]["smooth_ripple"] * 100), (ax[2], r["M"]["smooth_wobble"])):
            a_.plot(x, val, mk, color=col, mfc="white" if r["alpha"] < 1 else col, ms=6)
    from matplotlib.lines import Line2D
    for a_, t in zip(ax, ("경로까지 거리 p95 [deg]", "속도 리플 R [%]", "횡방향 흔들림 W [deg]")):
        a_.set_title(t, fontsize=10)
        a_.set_xlabel("accel [deg/s²]")
        a_.grid(alpha=0.3)
    h = [Line2D([], [], color=c, marker="o", ls="", label=f"speed {s:g}") for s, c in ((50, "C3"), (40, "C1"), (30, "C2"), (20, "C0"))]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("move_servo_j 프로파일 타이밍, 궤적 B 시뮬레이션: 속도·가속도 설정에 따른 결과 (alpha 1)", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(os.path.join(OUT, "r_servo_profile_sweep.png"), dpi=110)
    print("저장 r_servo_profile_sweep.png")
    # ---- 표: 실행별 CSV
    out = os.path.expanduser(a.out) if a.out else os.path.join(root, "report_dataset_20261008", "tables")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "servo_profile_runs_1014.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run", "mode", "trajectory", "speed", "accel", "gain", "alpha", "period_s", "ideal_T_s", "track_T_s", "path_dist_p95_deg", "path_dist_max_deg", "path_dist_mean_deg", "pair_dev_p95_deg", "pair_dev_max_deg", "n_spike_samples", "n_samples",
                    "end_err_deg", "ripple", "wobble_deg", "meas_vmax_deg_s", "meas_amax_deg_s2", "cmd_vmax_deg_s", "cmd_amax_deg_s2"])
        for r in rows:
            M = r["M"]
            w.writerow([r["run"], "실기" if r["real"] else "시뮬레이션", r["traj"], r["speed"], r["accel"], r["gain"], r["alpha"], r["period"], f"{r['T_ideal']:.3f}",
                        f"{M['t_track']:.3f}", f"{r['near_p95']:.4f}", f"{r['near_max']:.4f}", f"{r['near_mean']:.4f}", f"{M['path_dev_p95']:.4f}", f"{M['path_dev_max']:.4f}", r["n_spike"], r["n"], f"{M['end_err']:.5f}", f"{M['smooth_ripple']:.4f}",
                        f"{M['smooth_wobble']:.4f}", f"{r['vmax']:.2f}", f"{r['amax']:.1f}", r["cmd_v"], r["cmd_a"]])
    print("저장", path)
    # ---- 실기 대 시뮬 경로 거리 (궤적 C)
    d = poly_dist(qw_of(rc)[::2], qw_of(sc))
    print("실기 vs 시뮬 C 경로 거리 p95 %.3f max %.3f" % (np.percentile(d, 95), d.max()))
    json.dump(dict(sim_real_C=dict(p95=float(np.percentile(d, 95)), max=float(d.max()), sim=sc["run"], real=rc["run"]),
                   rows=[{k: v for k, v in r.items() if k not in ("dir",)} for r in rows]),
              open(os.path.join(OUT, "numbers_servo_profile.json"), "w"), default=float, ensure_ascii=False, indent=1)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
