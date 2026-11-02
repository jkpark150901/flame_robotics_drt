"""report.md 에 쓰는 그림(img/r_*.png)과 수식 이미지(img/f_*.png)를 만든다.

- 수식은 matplotlib mathtext 로 조판한 PNG 이다 (마크다운 뷰어의 수식 지원과 무관하게 깨지지 않는다).
- 지표 해설 그림은 docs/img 에서 복사하고, 실험 비교 그림은 실행 폴더의 결과로 새로 그린다.

사용: cd rb_servo_gui && PYTHONPATH=src python docs/report/make_report_figures.py [--runs ~/rb_servo_gui_runs] [--urdf ../urdf/rb10_1300e.urdf]
"""
import argparse
import csv
import glob
import json
import os
import shutil

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

from rb_servo_gui import analysis as A
from rb_servo_gui import trajectory as TJ

have = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic"):
    if _f in have:
        matplotlib.rcParams["font.sans-serif"] = [_f] + list(matplotlib.rcParams["font.sans-serif"])
        break
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "img")
DOCS_IMG = os.path.join(HERE, "..", "img")
EXAMPLES = os.path.join(HERE, "..", "..", "examples")
TRAJ = {"sample": "waypoints_sample.csv", "robot1": "waypoints_robot1.csv", "robot2": "waypoints_robot2.csv"}
TRAJ_LABEL = {"sample": "궤적 A (sample, 14.7 s)", "robot1": "궤적 B (robot1, 8.2 s)", "robot2": "궤적 C (robot2, 5.8 s)"}
TCOL = {"sample": "C0", "robot1": "C1", "robot2": "C2"}

FORMULAS = {
    "r_time_param": r"\Delta t_k=\max\left(\frac{\max_j\left|w_{k+1,j}-w_{k,j}\right|}{v_{max}},\ 0.2\ s\right),\qquad t_k=\sum_{m<k}\Delta t_m",
    "r_spline": r"\mathbf{r}(t)=\mathrm{CubicSpline}\left(\{t_k,\mathbf{w}_k\}\right),\quad \dot{\mathbf{r}}(0)=\dot{\mathbf{r}}(T)=\mathbf{0},\qquad \mathbf{r}_n=\mathbf{r}(n/f_s),\ f_s=50\ \mathrm{Hz}",
    "r_subsample": r"\mathbf{c}_m=\mathbf{r}_{m\cdot s}\ \ (m=0,1,\ldots),\quad \Delta t_{cmd}=\frac{s}{f_s}=\frac{10}{50}=0.2\ s\quad (\mathrm{last\ point\ always\ kept})",
    "r_floor": r"\delta_n=\min_m\ \mathrm{dist}\left(\mathbf{r}_n,\ [\mathbf{c}_m,\mathbf{c}_{m+1}]\right)\qquad (\mathrm{chord\ error\ of\ the\ command\ polyline})",
    "r_dtw": r"D(i,k)=C(i,k)+\min_{k-K\leq k'\leq k}D(i-1,k'),\qquad C(i,k)=\left\Vert\mathbf{q}_i-\mathbf{r}_k\right\Vert_2",
    "r_dtw_bound": r"k(1)\leq k(2)\leq\cdots\leq k(N),\qquad k(1)<\frac{M}{20},\quad k(N)>M-\frac{M}{20},\qquad K=\max\left(30,\ \frac{M}{5}\right)",
    "r_servo_over": r"\Delta q_{over}\approx\dot{q}_{last}\cdot t_2\qquad (\mathrm{observed;\ not\ in\ the\ manual})",
}

COPY = ["f_pair_error.png", "f_dev_stats.png", "f_end_err.png", "f_coverage.png", "f_progress.png",
        "f_sm_speed.png", "f_sm_window.png", "f_sm_hp.png", "f_sm_ripple.png", "f_sm_wobble.png", "f_sm_stall.png",
        "f_surf_disp.png", "f_surf_chamfer.png", "f_surf_hausdorff.png",
        "fig_dtw.png", "fig_error.png", "fig_ripple.png", "fig_wobble_real.png", "fig_start_end.png", "fig_surface.png"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110)
    print("저장", name)


def formula_images():
    for name, tex in FORMULAS.items():
        fig = Figure(figsize=(0.2, 0.2), facecolor="white")
        fig.text(0, 0, "$" + tex + "$", fontsize=15)
        fig.savefig(os.path.join(OUT, f"f_{name}.png"), dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    for f in COPY:
        shutil.copy(os.path.join(DOCS_IMG, f), os.path.join(OUT, f))
    shutil.copy(os.path.join(DOCS_IMG, "..", "smoothness_wobble.png"), os.path.join(OUT, "smoothness_wobble.png"))
    print("수식", len(FORMULAS), "개 + 복사", len(COPY) + 1, "개")


# ---------------------------------------------------------------- 실행 목록
def load_runs(root, since):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "*_step*"))):
        name = os.path.basename(d)
        js = glob.glob(os.path.join(d, "compare_*.json"))
        if name[:15] < since or not js:
            continue
        r = json.load(open(js[0], encoding="utf-8"))
        cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))["effective"]
        traj = [k for k, v in TRAJ.items() if os.path.basename(cfg["traj_path"]) == v][0]
        runs.append(dict(dir=d, name=name, real="REAL" in name, mode=r["mode"], traj=traj, p=r["params"], M=r["metrics"]))
    return runs


def add_surface(runs, combined):
    """표면 지표는 export_runs_csv.py 로 만든 통합 CSV 에서 가져온다 (실행 폴더에는 기록되지 않음)."""
    rows = {r["run"]: r for r in csv.DictReader(open(combined, encoding="utf-8-sig"))}
    for r in runs:
        x = rows.get(r["name"], {})
        for k in ("surf_disp_max", "surf_disp_p95", "surf_disp_mean", "surf_chamfer_max", "surf_worst_link"):
            v = x.get(k, "")
            r["M"][k] = (float(v) if k != "surf_worst_link" else v) if v else None


def in_set(r):
    """비교 집합: 궤적 A·B·C 모두 시뮬레이션과 실기로 수행한 조건만 (jb2 accel 120, servo gain 1 / alpha 1)."""
    if r["mode"] == "jb2":
        return r["p"].get("accel") == 120.0
    return r["p"].get("gain") == 1.0 and r["p"].get("alpha") == 1.0


def follow(d):
    f = max(glob.glob(os.path.join(d, "follow_*.csv")), key=os.path.getmtime)
    t, ph, q, st, meta = A.load_follow(f)
    m = "servo" if "servo" in meta["marks"] else "jb2"
    return t, ph, q, meta, m


def motion(d):
    t, ph, q, meta, m = follow(d)
    sel = np.array([p.startswith(m) and not p.endswith("move_to_start") for p in ph])
    t, q = t[sel], q[sel]
    i = int(np.argmax(np.abs(q - q[0]).max(1) > 0.2))
    return t[i:] - t[i], q[i:]


def ref(d):
    a = np.loadtxt(os.path.join(d, "traj_ref.csv"), delimiter=",", skiprows=1)
    s = np.loadtxt(os.path.join(d, "traj_sub10.csv"), delimiter=",", skiprows=1)
    return a[:, 0], np.rad2deg(a[:, 1:7]), s[:, 0], np.rad2deg(s[:, 1:7])


def poly_dist(P, Q):
    """P 의 각 점에서 꺾은선 Q 까지의 최소 거리."""
    a, ab = Q[:-1], np.diff(Q, axis=0)
    den = np.maximum((ab * ab).sum(1), 1e-12)
    out = np.empty(len(P))
    for i, p in enumerate(P):
        u = np.clip(((p - a) * ab).sum(1) / den, 0, 1)
        out[i] = np.linalg.norm(a + u[:, None] * ab - p, axis=1).min()
    return out


# ---------------------------------------------------------------- 1장: 궤적 준비
def fig_interp():
    names, w = TJ.load_waypoints(os.path.join(EXAMPLES, TRAJ["robot1"]))
    keep = np.r_[True, np.abs(np.diff(w, axis=0)).max(axis=1) > 1e-9]
    w = w[keep]
    tk = TJ.time_parametrize(w, np.deg2rad(20.0))
    t, q = TJ.interpolate(w, 20.0, 50.0)
    W, Q = np.rad2deg(w), np.rad2deg(q)
    V = np.gradient(Q, t, axis=0)
    fig = Figure(figsize=(12, 6.2))
    ax = fig.subplots(2, 1, sharex=True)
    for j in range(6):
        ax[0].plot(t, Q[:, j], color=f"C{j}", lw=1.4, label=f"J{j + 1}")
        ax[0].plot(tk, W[:, j], "o", color=f"C{j}", ms=4, mfc="white")
        ax[1].plot(t, V[:, j], color=f"C{j}", lw=1.2)
    ax[0].set_ylabel("관절각 [deg]"); ax[1].set_ylabel("관절 속도 [deg/s]"); ax[1].set_xlabel("시간 [s]")
    ax[0].set_title(f"웨이포인트 {len(w)}개(○) → 시간 부여(v_max = 20 deg/s) → 3차 스플라인 50 Hz 내삽(선), {len(t)}점, {t[-1]:.2f} s  (궤적 B)", fontsize=10)
    ax[1].set_title("내삽 궤적의 관절 속도: 시작/끝 0, 중간은 연속", fontsize=10)
    ax[0].legend(ncol=6, fontsize=8, loc="upper right")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "r_interp.png")


def fig_subsample(d):
    tr, qr, ts, qs = ref(d)
    dl = poly_dist(qr, qs)
    j = int(np.argmax(np.ptp(qr, axis=0)))
    fig = Figure(figsize=(12, 6.4))
    ax = fig.subplots(2, 1)
    lo, hi = 2.0, 4.0
    m, ms = (tr >= lo) & (tr <= hi), (ts >= lo - 0.2) & (ts <= hi + 0.2)
    ax[0].plot(tr[m], qr[m, j], ".", color="0.45", ms=5, label="원본 내삽 궤적 (50 Hz, 0.02 s 간격)")
    ax[0].plot(ts[ms], qs[ms, j], "o-", color="C3", ms=6, lw=1.2, label="로봇 지령점 (1/10 샘플링, 0.2 s 간격)과 그 직선 연결")
    ax[0].set_xlim(lo, hi); ax[0].set_xlabel("원본 시간 [s]"); ax[0].set_ylabel(f"J{j + 1} [deg]")
    ax[0].set_title("지령점 사이를 직선으로 이으면 곡선 구간에서 원본과 차이(현 오차)가 생긴다 (확대)", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(tr, dl, color="C3", lw=1.2)
    ax[1].axhline(np.percentile(dl, 95), color="k", ls="--", lw=0.8, label=f"p95 {np.percentile(dl, 95):.2f}°")
    ax[1].axhline(dl.max(), color="k", ls=":", lw=0.8, label=f"최대 {dl.max():.2f}°")
    ax[1].set_xlabel("원본 시간 [s]"); ax[1].set_ylabel("현 오차 δ [deg] (6관절 L2)")
    ax[1].set_title("원본의 각 점에서 지령 꺾은선까지 거리 — 로봇이 지령점을 정확히 지나도 남는 오차의 하한에 가까운 값", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "r_subsample.png")


def fig_record(d):
    t, ph, q, meta, m = follow(d)
    t0 = meta["marks"]["servo"]
    dt = meta["dt"]
    n_cmd = meta["n_points"] - 1 + meta["methods"]["servo"]["final_hold"]
    cmd_t = t0 + dt * np.arange(n_cmd) - t0
    tt = t - t0
    tr, qr, ts, qs = ref(d)
    j = int(np.argmax(np.ptp(qr, axis=0)))
    sel = (tt > -0.5) & (tt < 4.0)
    fig = Figure(figsize=(12, 4.6))
    ax = fig.subplots()
    ax.plot(tt[sel], q[sel, j], ".", color="C0", ms=4, label=f"측정 (reqdata 50 Hz, jnt_ang) J{j + 1}")
    ms_ = ts <= 4.0
    ax.plot(ts[ms_], qs[ms_, j], "s", color="C3", ms=6, mfc="white", label="지령점 c_m: (m−1)·0.2 s 에 전송, t1 = 0.2 s 뒤인 m·0.2 s 에 도달 목표 (그 시각에 표시)")
    for c in cmd_t[cmd_t < 4.0]:
        ax.axvline(c, color="C3", lw=0.5, alpha=0.35)
    ax.set_xlim(-0.5, 4.0)
    ax.set_xlabel("첫 move_servo_j 전송 후 시간 [s]   (세로선: move_servo_j 전송 시각, 0.2 s 간격)")
    ax.set_ylabel("관절각 [deg]")
    ax.set_title("명령 전송 스레드(0.2 s 주기)와 측정 스레드(50 Hz)가 동시에 돈다 — 실기 servo, 궤적 C", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "r_record.png")


# ---------------------------------------------------------------- DTW 예시
def toy_dtw():
    """예시: 원본은 사인파, 측정은 늦게 출발하고 중간에 잠시 느려진 신호."""
    tr = np.linspace(0, 1, 60)
    r = np.sin(2 * np.pi * tr) + 0.5 * np.sin(6 * np.pi * tr)
    tm = np.linspace(0, 1.25, 75)
    warp = np.clip((tm - 0.1) / 1.0, 0, 1) ** 1.15          # 늦게 출발 + 진행 속도 변화
    rng = np.random.default_rng(1)
    qm = np.interp(warp, tr, r) + 0.04 * rng.standard_normal(len(tm))
    return tr, r, tm, qm


def fig_dtw_toy():
    tr, r, tm, qm = toy_dtw()
    idx = A.ordered_match(qm[:, None], r[:, None], kmax=8)
    C = np.abs(qm[:, None] - r[None, :])
    fig = Figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1])
    a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    a0.plot(tr, r, "o-", color="0.4", ms=3, lw=1, label="원본 r_k")
    a1.plot(tm, qm, "o-", color="C0", ms=3, lw=1, label="측정 q_i (늦게 출발, 속도 변화)")
    for i in range(0, len(tm), 3):
        fig.add_artist(ConnectionPatch(xyA=(tm[i], qm[i]), coordsA=a1.transData, xyB=(tr[idx[i]], r[idx[i]]), coordsB=a0.transData,
                                       color="C0", lw=0.7, alpha=0.6))
    a0.set_title("DTW 매칭쌍: 측정 샘플 ↔ 원본 점 (시간 순서 보존)", fontsize=10)
    for a in (a0, a1):
        a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper right"); a.set_ylim(-1.8, 1.8)
    a0.set_xlabel("원본 시간"); a1.set_xlabel("측정 시간")
    a2.imshow(C.T, origin="lower", aspect="auto", cmap="Greys", extent=[-0.5, len(tm) - 0.5, -0.5, len(tr) - 0.5])
    a2.plot(np.arange(len(tm)), idx, color="C3", lw=2, label="최소 누적 비용 경로 k(i)")
    a2.set_xlabel("측정 샘플 i"); a2.set_ylabel("원본 점 k")
    a2.set_title("비용 행렬 C(i,k)=|q_i − r_k| (어두울수록 큼)", fontsize=10)
    a2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "r_dtw_toy.png")


# ---------------------------------------------------------------- 3장: 비교
def pick(runs, **kw):
    out = []
    for r in runs:
        ok = True
        for k, v in kw.items():
            if k in ("real", "mode", "traj"):
                ok &= r[k] == v
            else:
                ok &= r["p"].get(k) == v
        if ok:
            out.append(r)
    return out


def fig_jb2_cases(runs):
    metrics = (("path_dev_p95", "궤적 추종 오차 p95 [deg]", None), ("smooth_ripple", "속도 리플 R (참고, 판정 미사용)", None),
               ("smooth_wobble", "횡방향 흔들림 W [deg]", 0.3), ("surf_disp_p95", "최악 점 변위 D 의 시간 p95 [mm]", None))
    fig = Figure(figsize=(14, 4.2))
    ax = fig.subplots(1, 4)
    for k, (key, lab, th) in enumerate(metrics):
        a = ax[k]
        for tj in TRAJ:
            for real, mk, ls in ((False, "o", "-"), (True, "s", "--")):
                pts = sorted((r["p"]["blend"], r["M"][key]) for r in pick(runs, mode="jb2", traj=tj, real=real, accel=120.0) if r["M"].get(key) is not None)
                if not pts:
                    continue
                x, y = zip(*pts)
                a.plot(x, y, ls, marker=mk, color=TCOL[tj], mfc="white" if real else TCOL[tj], ms=6 if real else 4, lw=1)
        if th:
            a.axhline(th, color="r", lw=0.8, ls=":")
        a.set_title(lab, fontsize=10); a.set_xlabel("blend [%]"); a.set_xticks([0, 50, 100]); a.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=TCOL[t], lw=2, label=TRAJ_LABEL[t]) for t in TRAJ] + [
        Line2D([], [], color="k", ls="-", marker="o", label="시뮬레이션"), Line2D([], [], color="k", ls="--", marker="s", mfc="white", label="실기")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=8)
    fig.suptitle("move_jb2: blend 에 따른 결과 (speed 50 deg/s, accel 120 deg/s² 고정)", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save(fig, "r_jb2_cases.png")


def fig_speed_profiles(runs, tj="robot1"):
    cases = [("jb2", dict(accel=120.0, blend=0.0), "jb2 accel 120, blend 0"), ("jb2", dict(accel=120.0, blend=50.0), "jb2 accel 120, blend 50"),
             ("jb2", dict(accel=120.0, blend=100.0), "jb2 accel 120, blend 100"), ("servo", dict(gain=1.0, alpha=1.0), "servo gain 1, alpha 1")]
    fig = Figure(figsize=(12, 7.5))
    ax = fig.subplots(len(cases), 1, sharex=True)
    d0 = pick(runs, mode="servo", traj=tj)[0]["dir"]
    tr, qr, _, _ = ref(d0)
    vr = np.linalg.norm(np.gradient(qr, tr, axis=0), axis=1)
    for a, (m, kw, lab) in zip(ax, cases):
        rr = pick(runs, mode=m, traj=tj, real=False, **kw)
        if not rr:
            continue
        r = rr[-1]
        v = np.loadtxt(glob.glob(os.path.join(r["dir"], "velocity_*.csv"))[0], delimiter=",", skiprows=1)
        a.plot(tr - tr[0], vr, color="0.6", lw=1.2, label="원본 궤적의 속력 (참고)")
        a.plot(v[:, 0], v[:, 7], color="C0", lw=1.2, label="측정 속력 (시뮬)")
        a.set_title(f"{lab}:  리플 R {r['M']['smooth_ripple']:.1%},  추종 오차 p95 {r['M']['path_dev_p95']:.2f}°", fontsize=9.5)
        a.set_ylabel("deg/s"); a.grid(alpha=0.3); a.legend(fontsize=7.5, loc="upper right")
    ax[-1].set_xlabel("모션 시작 후 시간 [s]")
    fig.suptitle("관절공간 속력 ‖v‖ 의 시간 변화 — 궤적 B", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "r_speed_profiles.png")


def fig_servo_cases(runs):
    cases = [(dict(gain=0.2, alpha=1.0), "gain 0.2\nalpha 1"), (dict(gain=1.0, alpha=1.0), "gain 1\nalpha 1"), (dict(gain=1.0, alpha=0.5), "gain 1\nalpha 0.5")]
    metrics = (("path_dev_p95", "궤적 추종 오차 p95 [deg]", None), ("smooth_ripple", "속도 리플 R", 0.15),
               ("smooth_wobble", "횡방향 흔들림 W [deg]", 0.3), ("surf_disp_max", "최악 점 변위 최대 [mm]", None))
    fig = Figure(figsize=(14, 3.9))
    ax = fig.subplots(1, 4)
    for k, (key, lab, th) in enumerate(metrics):
        a = ax[k]
        for ci, (kw, cl) in enumerate(cases):
            for real, off, hatch in ((False, -0.18, ""), (True, 0.18, "//")):
                rr = [r for r in pick(runs, mode="servo", traj="robot1", real=real, **kw) if r["M"].get(key) is not None]
                if rr:
                    a.bar(ci + off, np.mean([r["M"][key] for r in rr]), 0.34, color="C1", alpha=0.55 if not real else 0.95, hatch=hatch, edgecolor="k", lw=0.5)
        if th:
            a.axhline(th, color="r", lw=0.8, ls=":")
        a.set_xticks(range(len(cases))); a.set_xticklabels([c[1] for c in cases], fontsize=8)
        a.set_title(lab, fontsize=10); a.grid(alpha=0.3, axis="y")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="C1", alpha=0.55, label="시뮬레이션"), Patch(facecolor="C1", hatch="//", edgecolor="k", label="실기")],
               loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("move_servo_j: gain, alpha 에 따른 결과 (궤적 B, t1 = t2 = 0.2 s, 마지막 점 30회 반복; 실기는 gain 0.2 / 1, alpha 1 만 있음)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save(fig, "r_servo_cases.png")


GROUPS = [("jb2 blend 0", lambda r: r["mode"] == "jb2" and r["p"]["blend"] == 0.0, "C3"),
          ("jb2 blend 50", lambda r: r["mode"] == "jb2" and r["p"]["blend"] == 50.0, "C1"),
          ("jb2 blend 100", lambda r: r["mode"] == "jb2" and r["p"]["blend"] == 100.0, "C2"),
          ("servo", lambda r: r["mode"] == "servo", "C0")]
TMARK = {"sample": "o", "robot1": "s", "robot2": "^"}


def fig_tradeoff(runs):
    """명령 방식별 추종 오차 대 속도 리플, 표면 변위 대 속도 리플 (실행마다 한 점)."""
    from matplotlib.lines import Line2D
    fig = Figure(figsize=(13, 5.0))
    ax = fig.subplots(1, 2)
    for a, key, lab in ((ax[0], "path_dev_p95", "궤적 추종 오차 p95 [deg]"), (ax[1], "surf_disp_p95", "최악 점 변위 D 의 시간 p95 [mm]")):
        for gl, f, col in GROUPS:
            for r in runs:
                if in_set(r) and f(r) and r["M"].get(key) is not None:
                    a.plot(r["M"]["smooth_ripple"] * 100, r["M"][key], TMARK[r["traj"]], color=col, ms=6,
                           mfc="white" if r["real"] else col, alpha=0.85)
        a.set_xlabel("속도 리플 R [%]  (오른쪽일수록 속도 불안정)"); a.set_ylabel(lab); a.grid(alpha=0.3)
    ax[0].set_title("추종 오차 대 속도 안정성", fontsize=10); ax[1].set_title("링크 표면 변위 대 속도 안정성", fontsize=10)
    h = [Line2D([], [], color=c, marker="o", ls="", label=g) for g, _, c in GROUPS] + \
        [Line2D([], [], color="k", marker=m, ls="", mfc="white", label=TRAJ_LABEL[t]) for t, m in TMARK.items()] + \
        [Line2D([], [], color="0.5", marker="o", ls="", label="채움 = 시뮬, 빈 점 = 실기", mfc="white")]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=8)
    fig.suptitle("명령 방식별 트레이드오프: 리플을 줄이면(blend↑) 추종 오차와 표면 변위가 커짐 — 실행 24회 (jb2 accel 120, servo gain 1)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save(fig, "r_tradeoff.png")


def sim_real_pairs(runs):
    out = []
    for r in runs:
        if not r["real"] or not in_set(r):
            continue
        keys = ("accel", "blend") if r["mode"] == "jb2" else ("gain", "alpha")
        cands = [s for s in runs if not s["real"] and s["mode"] == r["mode"] and s["traj"] == r["traj"] and all(s["p"].get(k) == r["p"].get(k) for k in keys)]
        if cands:
            out.append((cands[-1], r))
    return out


def fig_sim_real(runs, pairs):
    metrics = (("t_track", "추종 시간 [s]"), ("path_dev_p95", "궤적 추종 오차 p95 [deg]"), ("smooth_ripple", "속도 리플 R"), ("surf_disp_max", "최악 점 변위 최대 [mm]"))
    fig = Figure(figsize=(14, 4.0))
    ax = fig.subplots(1, 4)
    for a, (key, lab) in zip(ax, metrics):
        xs, ys = [], []
        for s, r in pairs:
            x, y = s["M"].get(key), r["M"].get(key)
            if x is None or y is None:
                continue
            a.plot(x, y, "o" if r["mode"] == "jb2" else "^", color=TCOL[r["traj"]], ms=6, mfc="white" if r["mode"] == "jb2" else TCOL[r["traj"]])
            xs.append(x); ys.append(y)
        lo, hi = min(xs + ys), max(xs + ys)
        pad = 0.05 * (hi - lo)
        a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.7)
        a.set_xlabel("시뮬레이션"); a.set_ylabel("실기"); a.set_title(lab, fontsize=10); a.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=TCOL[t], marker="s", ls="", label=TRAJ_LABEL[t]) for t in TRAJ] + [
        Line2D([], [], color="k", marker="o", mfc="white", ls="", label="jb2"), Line2D([], [], color="k", marker="^", ls="", label="servo")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(f"같은 궤적·같은 파라미터의 시뮬레이션 대 실기 ({len(pairs)}쌍) — 점선 위에 있으면 같은 값", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    save(fig, "r_sim_real.png")


def sim_real_path_dist(pairs):
    """실기 측정 경로의 각 점에서 시뮬레이션 측정 경로까지 거리(시간 무관)."""
    res = []
    for s, r in pairs:
        _, qs = motion(s["dir"])
        _, qr_ = motion(r["dir"])
        d = poly_dist(qr_[::2], qs)
        res.append((s, r, float(np.percentile(d, 95)), float(d.max())))
    return res


def fig_sim_real_overlay(s, r):
    ts, qs = motion(s["dir"])
    tr_, qr_ = motion(r["dir"])
    fig = Figure(figsize=(12, 4.2))
    ax = fig.subplots(1, 2)
    for j in range(6):
        ax[0].plot(ts, qs[:, j], color=f"C{j}", lw=2.2, alpha=0.35)
        ax[0].plot(tr_, qr_[:, j], color=f"C{j}", lw=0.9, ls="--", label=f"J{j + 1}")
    ax[0].set_xlabel("모션 시작 후 시간 [s]"); ax[0].set_ylabel("관절각 [deg]"); ax[0].grid(alpha=0.3); ax[0].legend(ncol=3, fontsize=7)
    ax[0].set_title("굵은 선 = 시뮬레이션 (jnt_ref), 점선 = 실기 (jnt_ang)", fontsize=9.5)
    d = poly_dist(qr_, qs)
    ax[1].plot(tr_, d, color="C3", lw=1)
    ax[1].set_xlabel("모션 시작 후 시간 [s]"); ax[1].set_ylabel("실기 → 시뮬 경로 거리 [deg]"); ax[1].grid(alpha=0.3)
    ax[1].set_title(f"경로 거리 (시간 무관): p95 {np.percentile(d, 95):.3f}°, 최대 {d.max():.3f}°", fontsize=9.5)
    fig.suptitle(f"{s['name']}  대  {r['name']}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "r_sim_real_overlay.png")


def fig_surface_cases(runs):
    groups = [("jb2 blend 0", dict(mode="jb2", blend=0.0)), ("jb2 blend 50", dict(mode="jb2", blend=50.0)),
              ("jb2 blend 100", dict(mode="jb2", blend=100.0)), ("servo gain 1", dict(mode="servo"))]
    fig = Figure(figsize=(13, 4.0))
    ax = fig.subplots(1, 2)
    for a, key, lab in ((ax[0], "surf_disp_max", "최악 점 변위 D 의 시간 최대 [mm]"), (ax[1], "surf_disp_p95", "D 의 시간 p95 [mm]")):
        for ti, tj in enumerate(TRAJ):
            for gi, (gl, kw) in enumerate(groups):
                v = [r["M"][key] for r in pick(runs, traj=tj, **kw) if in_set(r) and r["M"].get(key) is not None]
                if v:
                    x = gi + (ti - 1) * 0.26
                    a.bar(x, np.mean(v), 0.24, color=TCOL[tj], alpha=0.8, label=TRAJ_LABEL[tj] if gi == 0 else None)
                    a.plot([x, x], [min(v), max(v)], color="k", lw=1)
        a.set_xticks(range(len(groups))); a.set_xticklabels([g[0] for g in groups]); a.set_title(lab, fontsize=10); a.grid(alpha=0.3, axis="y")
    ax[0].legend(fontsize=8)
    fig.suptitle("링크 표면 변위 (rb20_1900es_RT URDF, jb2 accel 120 · servo gain 1, 시뮬·실기; 막대 = 평균, 세로선 = 최소~최대)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "r_surface_cases.png")


def load_pairs(d):
    pj = glob.glob(os.path.join(d, "compare_*_pairs.csv"))[0]
    with open(pj, encoding="utf-8") as f:
        f.readline()
        rows = list(csv.reader(f))
    a = np.array([[float(v) for v in x] for x in rows[1:]])
    return a[:, 0], a[:, 4:10], a[:, 10:16], a[:, 22]


def fig_mesh(r, urdf):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from rb_servo_gui import meshview
    from rb_servo_gui.settings import Settings
    t, qm, qr, dev = load_pairs(r["dir"])
    surf = A.surface_errors(Settings(urdf_path=urdf, surface_samples=10**9), qm, qr, t - t[0], log=lambda *_: None)  # 매칭쌍 전체
    i_w = surf["idx"][int(np.argmax(surf["series"]["disp_max"]))]
    scene = meshview.MeshScene(urdf)
    fig = Figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2.3], width_ratios=[1, 1, 0.04])
    at = fig.add_subplot(gs[0, :2])
    at.plot(surf["t"], surf["series"]["disp_max"], color="C3", lw=1.3, label="최악 점 변위 D")
    at.plot(surf["t"], surf["series"]["chamfer"], color="C0", lw=1.0, label="Chamfer CD")
    at2 = at.twinx(); at2.plot(t - t[0], dev, color="0.6", lw=0.8); at2.set_ylabel("관절 추종 오차 d [deg]", color="0.4")
    at.axvline(t[i_w] - t[0], color="k", ls="--", lw=0.8)
    at.set_xlabel("추종 시간 [s]"); at.set_ylabel("mm"); at.grid(alpha=0.3); at.legend(fontsize=8, loc="upper left")
    at.set_title(f"{r['name']}: 시간에 따른 표면 변위 (점선 = 아래 메시 시점)", fontsize=10)
    infos = []
    for k, (az, el, ttl) in enumerate(((40, 22, "등각"), (90, 0, "측면"))):
        a = fig.add_subplot(gs[1, k])
        info = meshview.render(a, scene, np.deg2rad(qm[i_w]), np.deg2rad(qr[i_w]), azim=az, elev=el)
        a.set_title(f"{ttl} 보기", fontsize=9)
        infos.append(info)
    info = infos[0]
    w_link, w_mm = info["worst"]
    fig.text(0.02, 0.60, f"t = {t[i_w] - t[0]:.2f} s   최악 점 변위 {w_mm:.1f} mm ({w_link})   관절 오차 {dev[i_w]:.2f}°\n"
             f"색 = 같은 메시 점의 실제 변위, 형상은 실제 크기(×1), 회색 반투명 = 기준 자세", fontsize=8.5, va="top")
    fig.text(0.02, 0.05, "링크별 최대 변위 [mm]\n" + "\n".join(f"{n}: {v:.1f}" for n, v in info["per_link"].items()), fontsize=7.5, va="bottom")
    cax = fig.add_subplot(gs[1, 2])
    fig.colorbar(ScalarMappable(norm=Normalize(0, info["vmax"]), cmap=info["cmap"]), cax=cax).set_label("변위 [mm]")
    fig.tight_layout()
    save(fig, "r_mesh.png")
    return dict(t=float(t[i_w] - t[0]), worst=w_link, mm=float(w_mm), dev=float(dev[i_w]), per_link=info["per_link"], links_max=surf["links"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="~/rb_servo_gui_runs")
    ap.add_argument("--since", default="20261008_160000")
    ap.add_argument("--urdf", default=os.path.join(HERE, "..", "..", "..", "urdf", "rb20_1900es_RT.urdf"))
    a = ap.parse_args()
    root = os.path.expanduser(a.runs)
    urdf = os.path.abspath(os.path.expanduser(a.urdf))
    os.makedirs(OUT, exist_ok=True)
    formula_images()
    runs = load_runs(root, a.since)
    add_surface(runs, os.path.join(root, f"combined_since_{a.since}.csv"))
    print(len(runs), "개 실행")
    by = {r["name"][9:15]: r for r in runs}
    # 프로그램이 실행마다 저장하는 결과 그림 예시 (실기 servo, 궤적 C)
    for pat, dst in (("compare_*_after.png", "r_run_compare_after.png"), ("smooth_map_*.png", "r_run_smooth_map.png"), ("sim_follow_*.png", "r_run_sim_follow.png")):
        shutil.copy(glob.glob(os.path.join(by["165140"]["dir"], pat))[0], os.path.join(OUT, dst))
    fig_interp()
    fig_subsample(by["165406"]["dir"])
    fig_record(by["165140"]["dir"])
    fig_dtw_toy()
    fig_jb2_cases(runs)
    fig_speed_profiles(runs)
    fig_tradeoff(runs)
    pairs = sim_real_pairs(runs)
    fig_sim_real(runs, pairs)
    fig_sim_real_overlay(*[p for p in pairs if p[1]["name"][9:15] == "164156"][0])
    fig_surface_cases(runs)
    worst = max((r for r in runs if in_set(r) and r["real"] and r["M"].get("surf_disp_max") is not None), key=lambda r: r["M"]["surf_disp_max"])  # 실기 중 최대
    mesh = fig_mesh(worst, urdf)
    mesh["run"] = worst["name"]
    # 보고서 표에 쓰는 수치
    floor = {}
    for tj in TRAJ:
        d = pick(runs, traj=tj)[0]["dir"]
        tr, qr, ts, qs = ref(d)
        dl = poly_dist(qr, qs)
        floor[tj] = dict(n_ref=len(tr), n_cmd=len(ts), T=float(tr[-1]), L=float(np.linalg.norm(np.diff(qr, axis=0), axis=1).sum()),
                         p95=float(np.percentile(dl, 95)), max=float(dl.max()))
    pd_ = [dict(sim=s["name"], real=r["name"], mode=r["mode"], traj=r["traj"], p=r["p"], p95=p95, max=mx) for s, r, p95, mx in sim_real_path_dist(pairs)]
    import collections
    stats = collections.OrderedDict()
    for gl, f, _ in GROUPS:
        rr = [r for r in runs if in_set(r) and f(r)]
        stats[gl] = {k: dict(n=len(rr), med=float(np.median([r["M"][k] for r in rr])), lo=float(min(r["M"][k] for r in rr)), hi=float(max(r["M"][k] for r in rr)))
                     for k in ("path_dev_p95", "path_dev_max", "smooth_ripple", "smooth_wobble", "surf_disp_mean", "surf_disp_p95", "surf_disp_max", "surf_chamfer_max")}
    json.dump(dict(floor=floor, set_stats=stats, sim_real=pd_, mesh=mesh, runs=[dict(name=r["name"], real=r["real"], mode=r["mode"], traj=r["traj"], p=r["p"], M=r["M"]) for r in runs]),
              open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("수치 → img/numbers.json")


if __name__ == "__main__":
    main()
