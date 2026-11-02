"""docs/metrics.md 에 쓰는 수식 이미지(f_*.png)와 설명 그래프(fig_*.png)를 만든다.

- 수식은 matplotlib mathtext 로 조판해 PNG 로 저장한다 (마크다운 뷰어의 수식 지원 여부와 무관하게 깨지지 않는다).
- 설명 그래프는 실제 실행 기록에서 현재 분석 함수(analysis.py)로 다시 계산해 그린다.

사용: PYTHONPATH=src python docs/make_metric_figures.py --servo <servo 실행 폴더> --jb2 <jb2 실행 폴더> [--out docs/img]
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

from rb_servo_gui import analysis as A
from rb_servo_gui.settings import Settings

have = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic"):  # KR 면은 .ttc 에서 인식되지 않을 수 있어 JP 면(한글 포함)도 허용
    if _f in have:
        matplotlib.rcParams["font.sans-serif"] = [_f] + list(matplotlib.rcParams["font.sans-serif"])
        break
matplotlib.rcParams["axes.unicode_minus"] = False
C = matplotlib.colormaps["tab10"](np.arange(10))
TH = Settings()


# ---------------------------------------------------------------- 계산 (analysis.analyze 와 같은 순서)
def compute(run_dir):
    follow = max(glob.glob(os.path.join(run_dir, "follow_*.csv")), key=os.path.getmtime)
    t, phase, meas, state, meta = A.load_follow(follow)
    marks = meta["marks"]
    m = "servo" if "servo" in marks else "jb2"
    ref = np.loadtxt(os.path.join(run_dir, "traj_ref.csv"), delimiter=",", skiprows=1)
    tr, qr = ref[:, 0], np.rad2deg(ref[:, 1:7])
    sel = np.array([p.startswith(m) and not p.endswith("move_to_start") for p in phase])
    ts, q = t[sel], meas[sel]
    tail = ts >= ts[-1] - max(TH.reach_hold, 0.3)
    end_pose = np.median(q[tail], axis=0)
    i_s, i_e, _ = A.motion_window(ts, q, end_pose, TH.reach_tol, TH.move_thr)
    t_s, t_e = ts[i_s], ts[i_e]
    T = t_e - t_s
    qw, t_rel = q[i_s:i_e + 1], ts[i_s:i_e + 1] - t_s
    tw = t_rel / T
    s_ref, cum, seg = A.ref_progress(qr)
    tau_ref = (tr - tr[0]) / (tr[-1] - tr[0])
    idx = A.ordered_match(qw, qr)
    pair, s_m, t_match, dev = A.refine_pairs(qw, qr, idx, cum, seg, tr)
    e = qw - pair
    cov = A.coverage(qr, qw, idx)
    lag = np.interp(s_ref, s_m, tw) - tau_ref
    vel = A.velocity(t_rel, qw)
    sm = A.smoothness(t_rel, s_m * cum[-1], s_m, e, T, TH.th_ripple, TH.th_wobble)
    pre = np.array([p == f"{m}_move_to_start" for p in phase])
    start_pose = None
    if pre.any():
        tp, qp = t[pre], meas[pre]
        start_pose = np.median(qp[tp >= tp[-1] - max(TH.reach_hold, 0.3)], axis=0)
    tau = np.linspace(0, 1, 500)
    qr_n = np.column_stack([np.interp(tau, tau_ref, qr[:, j]) for j in range(6)])
    qm_n = np.column_stack([np.interp(tau, tw, qw[:, j]) for j in range(6)])
    return dict(run=os.path.basename(run_dir), mode=m, t=t, meas=meas, phase=phase, marks=marks, tr=tr, qr=qr, ts=ts, q=q, i_s=i_s, i_e=i_e,
                t_s=t_s, t_e=t_e, T=T, qw=qw, t_rel=t_rel, tw=tw, s_ref=s_ref, cum=cum, tau_ref=tau_ref, idx=idx, pair=pair, s_m=s_m,
                t_match=t_match, dev=dev, e=e, cov=cov, lag=lag, vel=vel, sm=sm, end_pose=end_pose, start_pose=start_pose,
                rmse_before=float(np.sqrt(((qm_n - qr_n) ** 2).mean())), rmse_after=float(np.sqrt((e ** 2).mean())), sel=sel)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=115)
    print("저장", name)


# ---------------------------------------------------------------- 수식 이미지
FORMULAS = {
    # 구간과 시간
    "t_window_start": r"i_s=\min\{\,i:\ \max_j\left|q_{ij}-q_{0j}\right|>\theta_{move}\}-1",
    "t_window_end": r"i_e=\min\{\,i:\ \max_j\left|q_{kj}-q^{end}_{j}\right|<\epsilon_{reach}\ \ \mathrm{for\ all}\ k\geq i\,\},\quad q^{end}=\mathrm{median}\ (\mathrm{last}\ \max(t_{hold},\,0.3\ s))",
    "t_track": r"T_{track}=t_e-t_s",
    "t_latency": r"t_{latency}=\max\left(t_s-t_{cmd},\ 0\right)",
    "t_total": r"t_{total}=t_{done}-t_{cmd}\qquad (\mathrm{if\ no}\ t_{done}:\ t_e\ \mathrm{is\ used})",
    "t_build": r"t_{build}=t_{cmd}-t_{list\ start}\quad (\mathrm{jb2})",
    "t_ratio": r"\mathrm{ratio}=\frac{T_{track}}{T_{ref}}",
    # DTW 매칭과 추종 오차
    "pair_error": r"\mathbf{e}_i=\mathbf{q}_i-\mathbf{p}_i,\qquad d_i=\left\Vert\mathbf{e}_i\right\Vert_2=\sqrt{\sum_{j=1}^{6}e_{ij}^{2}}",
    "dev_stats": r"d_{mean}=\frac{1}{N}\sum_i d_i,\quad d_{rms}=\sqrt{\frac{1}{N}\sum_i d_i^{2}},\quad d_{p95}=P_{95}(d_i),\quad d_{max}=\max_i d_i",
    "rmse_after": r"\mathrm{RMSE}_{after}=\sqrt{\frac{1}{6N}\sum_{i=1}^{N}\sum_{j=1}^{6}e_{ij}^{2}}",
    "rmse_before": r"\mathrm{RMSE}_{before}=\sqrt{\frac{1}{6M}\sum_{k=1}^{M}\sum_{j=1}^{6}\left(q^{\,norm}_{kj}-r^{\,norm}_{kj}\right)^{2}}",
    # 커버리지, 진행도, 샘플간 이동
    "coverage": r"c_k=\min_{\ell\in W(k)}\ \mathrm{dist}\left(\mathbf{r}_k,\ [\mathbf{q}_\ell,\mathbf{q}_{\ell+1}]\right),\quad c_{mean},\ c_{p95},\ c_{max}",
    "progress": r"\mathrm{progress}=\max\{\,p_i:\ d_i\leq d_{th}\,\},\qquad p_i=\frac{\mathrm{arc\ length\ up\ to\ matched\ point}}{L}",
    "step": r"\Delta_{max}=\max_i\left\Vert\mathbf{q}_{i+1}-\mathbf{q}_i\right\Vert_2",
    # 시작/종료 각 오차
    "end_err": r"e_{end}=\max_j\left|\ \mathrm{median}_{\,t\geq t_{last}-\max(t_{hold},0.3)}\ q_j(t)-r_{N,j}\right|",
    "start_err": r"e_{start}=\max_j\left|\ \mathrm{median}\ q^{\,pre}_j-r_{0,j}\right|\quad (\mathrm{pre}=\mathrm{move\_to\_start}\ \mathrm{phase})",
    # 타이밍 지연
    "lag": r"\Delta\tau(s)=\tau_{meas}(s)-\tau_{ref}(s)\qquad (+:\ \mathrm{late},\ -:\ \mathrm{early})",
    "lag_stats": r"\Delta t_{max}=\Delta\tau_{max}\cdot T_{track},\qquad \mathrm{late}=\frac{\#\{\Delta\tau>\theta_{lag}\}}{M},\quad \mathrm{early}=\frac{\#\{\Delta\tau<-\theta_{lag}\}}{M}",
    # 속도
    "velocity": r"v_{ij}=\frac{q_{hj}-q_{lj}}{t_h-t_l},\quad l=\mathrm{first}\ (t\geq t_i-0.1),\ \ h=\mathrm{last}\ (t\leq t_i+0.1)",
    "vel_stats": r"v_{max,joint}=\max_{i,j}\left|v_{ij}\right|,\quad \mathbf{v}_i\ \mathrm{norm}=\left\Vert\mathbf{v}_i\right\Vert_2,\quad v_{max,norm}=\max_i\left\Vert\mathbf{v}_i\right\Vert_2,\quad v_{mean,norm}=\frac{1}{N}\sum_i\left\Vert\mathbf{v}_i\right\Vert_2",
    # 매끄러움
    "sm_speed": r"s_i=p_i\cdot L,\qquad v^{s}_i=\frac{s_h-s_l}{t_h-t_l}\quad (\mathrm{same\ \pm 0.1\ s\ window})",
    "sm_window": r"w=\mathrm{clip}\left(\frac{T_{track}}{10},\ 0.3,\ 2.0\right)\ [s],\qquad n=\max\left(\mathrm{round}\left(\frac{w}{\Delta t}\right),\ 3\right)",
    "sm_hp": r"hp_i=v^{s}_i-\mathrm{MA}_n\left(v^{s}\right)_i,\qquad \mathbf{e}^{hp}_i=\mathbf{e}_i-\mathrm{MA}_n\left(\mathbf{e}\right)_i",
    "sm_ripple": r"R=\frac{\sqrt{\mathrm{mean}_{i\in core}\ hp_i^{\,2}}}{\mathrm{mean}_{i\in core}\ v^{s}_i}",
    "sm_wobble": r"W=\sqrt{\mathrm{mean}_{i\in core}\left\Vert\mathbf{e}^{hp}_i\right\Vert_2^{2}}\ \ [\mathrm{deg}]",
    "sm_stall": r"S=\frac{\#\{\,i\in\mathcal{M}:\ v^{s}_i<0.2\cdot \mathrm{median}(v^{s}_{\mathcal{M}})\}}{\left|\mathcal{M}\right|},\qquad \mathcal{M}=\{\,i:\ 0.05<p_i<0.95\,\}",
    "sm_core": r"core=\{\,i:\ n/2\leq i<N-n/2\,\}\quad (N\leq 2n:\ \mathrm{all})",
    "sm_score": r"\mathrm{score}_i=\max\left(\frac{R_i}{R_{th}},\ \frac{W_i}{W_{th}}\right),\quad R_i=\frac{\sqrt{\mathrm{MA}_n(hp^{2})_i}}{\mathrm{mean}\ v^{s}},\quad W_i=\sqrt{\mathrm{MA}_n\left(\left\Vert\mathbf{e}^{hp}\right\Vert^{2}\right)_i}",
    "surf_disp": r"D_i=\max_{v}\left\Vert\,\mathbf{x}_v(\mathbf{q}_i)-\mathbf{x}_v(\mathbf{p}_i)\right\Vert_2,\qquad D_{max}=\max_i D_i,\quad D_{p95}=P_{95}(D_i),\quad D_{mean}=\frac{1}{N}\sum_i D_i",
    "surf_chamfer": r"\mathrm{CD}_i=\frac{1}{2}\left(\frac{1}{|A|}\sum_{a\in A}\min_{b\in B}\left\Vert a-b\right\Vert+\frac{1}{|B|}\sum_{b\in B}\min_{a\in A}\left\Vert a-b\right\Vert\right)",
    "surf_hausdorff": r"\mathrm{HD}_i=\max\left(\max_{a\in A}\min_{b\in B}\left\Vert a-b\right\Vert,\ \max_{b\in B}\min_{a\in A}\left\Vert a-b\right\Vert\right)",
    "sm_rough": r"\mathrm{rough\_ratio}=\frac{\#\{\,i:\ \mathrm{score}_i>1\,\}}{N},\qquad \mathrm{rough\_score\_max}=\max_i\ \mathrm{score}_i",
}


def formula_images(out):
    for name, tex in FORMULAS.items():
        fig = Figure(figsize=(0.2, 0.2), facecolor="white")
        fig.text(0, 0, "$" + tex + "$", fontsize=15)
        try:
            fig.savefig(os.path.join(out, f"f_{name}.png"), dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
        except Exception as ex:  # noqa: BLE001
            print("수식 실패:", name, str(ex)[:100])
    print("수식 이미지", len(FORMULAS), "개")


# ---------------------------------------------------------------- 설명 그래프
def fig_timeline(out, S, J):
    fig = Figure(figsize=(12, 7.2))
    ax = fig.subplots(2, 1)
    for a, D, title in ((ax[0], S, "servo (robot2): 지령 → 모션 시작 → 종료 → 완료"), (ax[1], J, "jb2: 리스트 구성 → 지령(run) → 모션 시작 → 종료 → 완료")):
        t, m = D["t"], D["mode"]
        v = np.linalg.norm(A.velocity(t, D["meas"]), axis=1)
        a.plot(t, v, color="0.55", lw=1.0, label="측정 속력 |v| (6관절 L2)")
        mk = D["marks"]
        t_cmd = mk[m]
        t_done = mk.get(m + "_done")
        ymax = max(v.max(), 1) * 1.05
        marks = [("t_cmd (추종 지령)", t_cmd, "C3"), ("t_s (모션 시작)", D["t_s"] + 0, "C2"), ("t_e (모션 종료)", D["t_e"], "C1")]
        if t_done is not None:
            marks.append(("t_done (완료 확인)", t_done, "C4"))
        if (m + "_build") in mk:
            marks.insert(0, ("리스트 구성 시작", mk[m + "_build"], "C5"))
        for k_, (lab, x, c) in enumerate(marks):
            a.axvline(x, color=c, ls="--", lw=1.4)
            a.text(x, ymax * (0.98 - 0.30 * (k_ % 2)), " " + lab, rotation=0, va="top", ha="left", fontsize=8, color=c,
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        y0 = -0.10 * ymax
        a.annotate("", xy=(D["t_s"], y0), xytext=(t_cmd, y0), arrowprops=dict(arrowstyle="<->", color="C3"))
        a.text((t_cmd + D["t_s"]) / 2, y0 - 0.05 * ymax, "t_latency", fontsize=8, color="C3", ha="center", va="top")
        a.annotate("", xy=(D["t_e"], y0 * 1.9), xytext=(D["t_s"], y0 * 1.9), arrowprops=dict(arrowstyle="<->", color="C2"))
        a.text((D["t_s"] + D["t_e"]) / 2, y0 * 1.9 - 0.05 * ymax, f"T_track = {D['T']:.1f} s", fontsize=9, color="C2", ha="center", va="top")
        end = t_done if t_done is not None else D["t_e"]
        a.annotate("", xy=(end, y0 * 3.0), xytext=(t_cmd, y0 * 3.0), arrowprops=dict(arrowstyle="<->", color="C4"))
        a.text((t_cmd + end) / 2, y0 * 3.0 - 0.05 * ymax, f"t_total = {end - t_cmd:.1f} s", fontsize=9, color="C4", ha="center", va="top")
        if (m + "_build") in mk:
            a.annotate("", xy=(t_cmd, y0 * 4.1), xytext=(mk[m + "_build"], y0 * 4.1), arrowprops=dict(arrowstyle="<->", color="C5"))
            a.text((mk[m + "_build"] + t_cmd) / 2, y0 * 4.1 - 0.05 * ymax, f"t_build = {t_cmd - mk[m + '_build']:.1f} s", fontsize=9, color="C5", ha="center", va="top")
        a.set_title(f"{title}   [{D['run']}]", fontsize=10)
        a.set_ylabel("속력 [deg/s]")
        a.set_xlabel("기록 시간 [s]")
        a.set_ylim(y0 * 5.3 if (m + "_build") in mk else y0 * 4.2, ymax * 1.02)
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    save(fig, out, "fig_timeline.png")


def fig_dtw(out, D):
    j = int(np.argmax(np.ptp(D["qr"], axis=0)))
    fig = Figure(figsize=(12, 7))
    ax = fig.subplots(2, 1)
    t0 = D["tr"] - D["tr"][0]
    ax[0].plot(t0, D["qr"][:, j], color="0.4", lw=2.0, label=f"원본(기준) 관절 {j + 1}")
    ax[0].set_title("위: 원본 궤적(원본 시간)   /   아래: 측정(추종 시간)   —   파란 선 = DTW 로 짝지은 측정 샘플 ↔ 원본 위치", fontsize=10)
    ax[1].plot(D["t_rel"], D["qw"][:, j], color="C0", lw=1.6, label=f"측정 관절 {j + 1}")
    for a, lab, xl in ((ax[0], "원본 각도 [deg]", "원본 시간 [s]"), (ax[1], "측정 각도 [deg]", "추종 시간 [s] (0 = 모션 시작)")):
        a.set_ylabel(lab); a.set_xlabel(xl); a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper right")
    ymin = min(D["qr"][:, j].min(), D["qw"][:, j].min()); ymax_ = max(D["qr"][:, j].max(), D["qw"][:, j].max()); pad = 0.06 * (ymax_ - ymin)
    for a in ax:
        a.set_ylim(ymin - pad, ymax_ + pad)
    step = max(len(D["t_rel"]) // 28, 1)
    for i in range(0, len(D["t_rel"]), step):
        con = ConnectionPatch(xyA=(D["t_rel"][i], D["qw"][i, j]), coordsA=ax[1].transData, xyB=(D["t_match"][i], D["pair"][i, j]),
                              coordsB=ax[0].transData, color="C0", lw=0.7, alpha=0.7)
        fig.add_artist(con)
        ax[0].plot(D["t_match"][i], D["pair"][i, j], "o", color="C0", ms=3)
        ax[1].plot(D["t_rel"][i], D["qw"][i, j], "o", color="C0", ms=3)
    fig.tight_layout()
    save(fig, out, "fig_dtw.png")


def fig_error(out, D):
    d = D["dev"]
    fig = Figure(figsize=(13, 4.8))
    ax = fig.subplots(1, 3, gridspec_kw={"width_ratios": [3, 1.3, 1.1]})
    a = ax[0]
    a.plot(D["t_rel"], d, color="C0", lw=1.2, label="d_i  (6관절 L2 오차)")
    stats = [("mean", d.mean(), "C2"), ("rms", np.sqrt((d ** 2).mean()), "C1"), ("p95", np.percentile(d, 95), "C4"), ("max", d.max(), "C3")]
    for lab, val, c in stats:
        a.axhline(val, color=c, ls="--", lw=1.2, label=f"d_{lab} = {val:.2f}°")
    a.set_title("궤적 추종 오차 d_i (DTW 매칭쌍 기준)", fontsize=10); a.set_xlabel("추종 시간 [s]"); a.set_ylabel("[deg]"); a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper right")
    h = ax[1]
    h.hist(d, bins=30, orientation="horizontal", color="C0", alpha=0.7)
    for lab, val, c in stats:
        h.axhline(val, color=c, ls="--", lw=1.2)
    h.set_title("분포", fontsize=10); h.set_xlabel("샘플 수"); h.grid(alpha=0.3)
    b = ax[2]
    b.bar(["매칭 전\n(같은 정규화\n시간끼리)", "매칭 후\n(DTW 쌍)"], [D["rmse_before"], D["rmse_after"]], color=["0.6", "C0"])
    for k, v in enumerate([D["rmse_before"], D["rmse_after"]]):
        b.text(k, v, f"{v:.2f}°", ha="center", va="bottom", fontsize=9)
    b.set_title("각도 오차 RMSE", fontsize=10); b.set_ylabel("[deg]"); b.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save(fig, out, "fig_error.png")


def _pca(D):
    mu = D["qr"].mean(0)
    _, _, vt = np.linalg.svd(D["qr"] - mu, full_matrices=False)
    return (lambda X: (X - mu) @ vt[:2].T)


def fig_cover(out, D):
    proj = _pca(D)
    R, P = proj(D["qr"]), proj(D["qw"])
    k = int(np.argmax(D["cov"]))
    fig = Figure(figsize=(13, 5))
    ax = fig.subplots(1, 2)
    a = ax[0]
    a.plot(R[:, 0], R[:, 1], color="0.65", lw=5, label="원본 경로")
    a.plot(P[:, 0], P[:, 1], color="C0", lw=1.4, label="측정 경로")
    a.plot(R[k, 0], R[k, 1], "o", color="C3", ms=8, label=f"c_max 인 원본 점 (c = {D['cov'][k]:.2f}°)")
    i = int(np.argmin(np.linalg.norm(D["qw"] - D["qr"][k], axis=1)))
    a.plot([R[k, 0], P[i, 0]], [R[k, 1], P[i, 1]], color="C3", lw=1.5)
    a.set_aspect("equal", adjustable="datalim"); a.set_title("커버리지: 원본의 각 점 r_k 에서 측정 경로까지의 거리 c_k\n(6관절 경로를 주성분 평면에 투영한 그림, 거리는 6관절 공간 값)", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    b = ax[1]
    b.plot(D["s_ref"], D["cov"], color="C0", lw=1.3)
    for lab, val, c in (("mean", D["cov"].mean(), "C2"), ("p95", np.percentile(D["cov"], 95), "C4"), ("max", D["cov"].max(), "C3")):
        b.axhline(val, color=c, ls="--", lw=1.2, label=f"c_{lab} = {val:.2f}°")
    b.axhline(TH.th_cover, color="k", ls=":", lw=1.2, label=f"판정 기준 {TH.th_cover:g}°")
    b.set_title("원본 경로 위치별 c_k (건너뛴 구간이 있으면 크게 솟음)", fontsize=10); b.set_xlabel("원본 경로 진행도 s"); b.set_ylabel("c_k [deg]"); b.legend(fontsize=8); b.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out, "fig_cover.png")


def fig_progress_step(out, D):
    fig = Figure(figsize=(13, 4.8))
    ax = fig.subplots(1, 2)
    a = ax[0]
    ok = D["dev"] <= TH.th_path_max
    a.plot(D["t_rel"], D["s_m"], color="C0", lw=1.2)
    a.plot(D["t_rel"][~ok], D["s_m"][~ok], "x", color="C3", ms=5, label=f"d_i > {TH.th_path_max:g}° (진행도 계산에서 제외)")
    prog = D["s_m"][ok].max() if ok.any() else 0
    a.axhline(prog, color="C2", ls="--", label=f"progress = {prog:.3f}")
    a.axhline(TH.th_progress, color="k", ls=":", label=f"판정 기준 {TH.th_progress:g}")
    a.set_title("경로 진행도 p_i (원본 경로를 몇 % 지났나)", fontsize=10); a.set_xlabel("추종 시간 [s]"); a.set_ylabel("p_i"); a.legend(fontsize=8, loc="lower right"); a.grid(alpha=0.3)
    st = np.linalg.norm(np.diff(D["qw"], axis=0), axis=1)
    b = ax[1]
    b.plot(D["t_rel"][1:], st, color="C0", lw=1.2)
    b.axhline(st.max(), color="C3", ls="--", label=f"Δ_max = {st.max():.2f}°")
    b.axhline(TH.th_step, color="k", ls=":", label=f"판정 기준 {TH.th_step:g}° (순간이동 검출)")
    b.set_title("연속한 측정 샘플 사이 이동량 ‖q_(i+1) − q_i‖", fontsize=10); b.set_xlabel("추종 시간 [s]"); b.set_ylabel("[deg]"); b.legend(fontsize=8); b.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out, "fig_progress_step.png")


def fig_start_end(out, D):
    fig = Figure(figsize=(13, 5))
    ax = fig.subplots(1, 2)
    m = D["mode"]
    pre = np.array([p == f"{m}_move_to_start" for p in D["phase"]])
    a = ax[0]
    if pre.any() and D["start_pose"] is not None:
        tp, qp = D["t"][pre], D["meas"][pre]
        tt = tp - tp[-1]
        keep = tt > -1.5
        for j in range(6):
            a.plot(tt[keep], qp[keep, j], color=C[j], lw=1.3)
            a.plot(0.12, D["qr"][0, j], ">", color=C[j], ms=7)
        e = np.abs(D["start_pose"] - D["qr"][0]).max()
        a.set_title(f"시작 각 오차 e_start = {e:.2f}°\n(시작 자세 이동의 마지막 안정 구간 중앙값 vs 원본 시작점 ▶)", fontsize=10)
        a.axvspan(-max(TH.reach_hold, 0.3), 0, color="0.9", label="중앙값을 구하는 구간")
        a.set_xlabel("시작 자세 이동 종료 기준 시간 [s]"); a.legend(fontsize=8)
    a.set_ylabel("각도 [deg]"); a.grid(alpha=0.3)
    b = ax[1]
    ts, q = D["ts"], D["q"]
    tt = ts - ts[-1]
    keep = tt > -2.5
    for j in range(6):
        b.plot(tt[keep], q[keep, j], color=C[j], lw=1.3, label=f"J{j + 1}")
        b.plot(0.12, D["qr"][-1, j], "<", color=C[j], ms=7)
    e = np.abs(D["end_pose"] - D["qr"][-1]).max()
    b.axvspan(-max(TH.reach_hold, 0.3), 0, color="0.9", label="중앙값을 구하는 구간")
    b.set_title(f"종료 각 오차 e_end = {e:.2f}°\n(마지막 안정 구간 중앙값 vs 원본 마지막 점 ◀, reach_tol 과 무관)", fontsize=10)
    b.set_xlabel("추종 종료 기준 시간 [s]"); b.legend(fontsize=7, ncol=2); b.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out, "fig_start_end.png")


def fig_lag(out, D):
    fig = Figure(figsize=(13, 6.5))
    ax = fig.subplots(2, 1, sharex=True)
    ax[0].plot(D["tau_ref"], D["s_ref"], color="C0", lw=2, label="원본: 진행도 s_ref (정규화 시간 τ_ref)")
    ax[0].plot(D["tw"], D["s_m"], "--", color="C3", lw=1.6, label="측정: 진행도 p_i (정규화 시간 τ_meas)")
    ax[0].set_ylabel("경로 진행도"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    ax[0].set_title("같은 경로 진행도에 도달한 (정규화) 시간의 차이가 타이밍 지연 Δτ", fontsize=10)
    lag = D["lag"]
    ax[1].plot(D["tau_ref"], lag, color="k", lw=1.3)
    ax[1].axhline(TH.th_lag, color="C1", ls=":", label=f"+θ_lag = {TH.th_lag:g} (늦음 기준)")
    ax[1].axhline(-TH.th_lag, color="C0", ls=":", label=f"−θ_lag (빠름 기준)")
    ax[1].fill_between(D["tau_ref"], 0, lag, where=lag > TH.th_lag, color="C1", alpha=0.4)
    ax[1].fill_between(D["tau_ref"], 0, lag, where=lag < -TH.th_lag, color="C0", alpha=0.4)
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set_ylabel("Δτ  (+ 늦음 / − 빠름)"); ax[1].set_xlabel("정규화 시간 τ (0 = 모션 시작, 1 = 모션 종료)")
    ax[1].text(0.01, 0.92, f"Δτ_max = {lag.max():+.3f} ({lag.max() * D['T']:+.2f} s),  Δτ_min = {lag.min():+.3f},  평균 {lag.mean():+.3f}", transform=ax[1].transAxes, fontsize=9, va="top")
    ax[1].legend(fontsize=8, loc="lower right"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out, "fig_lag.png")


def fig_velocity(out, D):
    v = D["vel"]
    spd = np.linalg.norm(v, axis=1)
    fig = Figure(figsize=(12, 5.2))
    a = fig.subplots(1, 1)
    for j in range(6):
        a.plot(D["t_rel"], np.abs(v[:, j]), color=C[j], lw=0.9, alpha=0.7, label=f"|v_J{j + 1}|")
    a.plot(D["t_rel"], spd, color="k", lw=2.0, label="‖v_i‖ (6관절 L2 속력)")
    ij = np.unravel_index(np.argmax(np.abs(v)), v.shape)
    a.plot(D["t_rel"][ij[0]], abs(v[ij]), "o", color="C3", ms=8, label=f"v_max,joint = {abs(v[ij]):.1f} (J{ij[1] + 1})")
    a.axhline(spd.max(), color="C3", ls="--", lw=1.1, label=f"v_max,norm = {spd.max():.1f}")
    a.axhline(spd.mean(), color="C2", ls="--", lw=1.1, label=f"v_mean,norm = {spd.mean():.1f}")
    a.set_title("추종 속도: 중심 차분(±0.1 s)으로 구한 관절별 |v| 와 6관절 속력 [deg/s]", fontsize=10)
    a.set_xlabel("추종 시간 [s]"); a.set_ylabel("[deg/s]"); a.legend(fontsize=8, ncol=2); a.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out, "fig_velocity.png")


def fig_ripple(out, D):
    sm = D["sm"]; t = D["t_rel"]; n = sm["n"]; N = len(t)
    core = slice(n // 2, N - n // 2) if N > 2 * n else slice(0, N)
    v, tr_ = sm["v_s"], sm["trend"]
    v_hp = v - tr_
    fig = Figure(figsize=(12, 9.5))
    ax = fig.subplots(3, 1, sharex=True)
    ax[0].plot(t, v, color="C0", lw=1.0, label="경로 따라 속력 v^s_i")
    ax[0].plot(t, tr_, color="C3", lw=2.0, label=f"추세 MA_n (창 w = {sm['window']:.2f} s, n = {n})")
    ax[0].axvspan(t[0], t[core.start], color="0.88"); ax[0].axvspan(t[core.stop - 1], t[-1], color="0.88", label="양 끝 n/2 샘플 (요약에서 제외)")
    ax[0].set_ylabel("[deg/s]"); ax[0].legend(fontsize=8, loc="upper right"); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"① 속도 리플 R: 속력에서 추세를 뺀 고주파 hp_i 의 RMS ÷ 평균 속력      R = {sm['ripple']:.1%}", fontsize=10)
    ax[1].plot(t, v_hp, color="C3", lw=1.0)
    rms = np.sqrt(np.mean(v_hp[core] ** 2))
    ax[1].axhline(rms, color="k", ls="--", lw=1); ax[1].axhline(-rms, color="k", ls="--", lw=1, label=f"±RMS(hp) = {rms:.2f} deg/s")
    ax[1].set_ylabel("hp_i [deg/s]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    mid = (D["s_m"] > 0.05) & (D["s_m"] < 0.95)
    med = np.median(v[mid])
    ax[2].plot(t, v, color="C0", lw=1.0)
    ax[2].axhline(0.2 * med, color="C3", ls="--", label=f"0.2 × 중앙값 = {0.2 * med:.2f} deg/s")
    ax[2].axvspan(t[mid][0], t[mid][-1], color="C2", alpha=0.08, label="M: 진행도 5~95 %")
    st = mid & (v < 0.2 * med)
    ax[2].plot(t[st], v[st], "o", color="C3", ms=4, label=f"정체 샘플 (S = {sm['stall']:.1%})")
    ax[2].set_ylabel("v^s_i [deg/s]"); ax[2].set_xlabel("추종 시간 [s]"); ax[2].legend(fontsize=8, loc="upper right"); ax[2].grid(alpha=0.3)
    ax[2].set_title("② 정체 비율 S: 경로 중간에서 속력이 중앙값의 20 % 미만인 샘플 비율", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig_ripple.png")


def fig_wobble_real(out, D):
    sm = D["sm"]; t = D["t_rel"]; n = sm["n"]; N = len(t)
    core = slice(n // 2, N - n // 2) if N > 2 * n else slice(0, N)
    e_hp = sm["e_hp"]; nrm = np.linalg.norm(e_hp, axis=1)
    fig = Figure(figsize=(12, 5.6))
    ax = fig.subplots(2, 1, sharex=True)
    for j in range(6):
        ax[0].plot(t, e_hp[:, j], color=C[j], lw=0.9, label=f"J{j + 1}")
    ax[0].set_ylabel("e^hp_ij [deg]"); ax[0].legend(ncol=6, fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_title("궤적 추종 오차에서 추세를 뺀 고주파 성분 (관절별)", fontsize=10)
    ax[1].plot(t, nrm, color="k", lw=1.2, label="‖e^hp_i‖ (6관절 L2)")
    ax[1].axhline(sm["wobble"], color="C3", ls="--", label=f"W = RMS(‖e^hp‖) = {sm['wobble']:.3f}°")
    ax[1].axhline(TH.th_wobble, color="0.4", ls=":", label=f"판정 기준 {TH.th_wobble:g}°")
    ax[1].axvspan(t[0], t[core.start], color="0.88"); ax[1].axvspan(t[core.stop - 1], t[-1], color="0.88")
    ax[1].set_ylabel("[deg]"); ax[1].set_xlabel("추종 시간 [s]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out, "fig_wobble_real.png")


def fig_score(out, D):
    sm = D["sm"]; t = D["t_rel"]
    r_i = sm["rip_loc"] / TH.th_ripple
    w_i = sm["wob_loc"] / TH.th_wobble
    score = sm["score"]
    fig = Figure(figsize=(12, 4.8))
    a = fig.subplots(1, 1)
    a.plot(t, r_i, color="C0", lw=1.0, label=f"R_i / R_th  (R_th = {TH.th_ripple:g})")
    a.plot(t, w_i, color="C3", lw=1.0, label=f"W_i / W_th  (W_th = {TH.th_wobble:g}°)")
    a.plot(t, score, color="k", lw=1.8, label="score_i = max(두 값)")
    a.fill_between(t, 1, np.maximum(score, 1), where=score > 1, color="C3", alpha=0.25, label=f"기준 초과 구간 (rough_ratio = {np.mean(score > 1):.0%})")
    a.axhline(1, color="C3", ls="--", lw=1.2)
    a.set_title(f"국소 거칠기 점수 (1 = 기준선).  rough_score_max = {score.max():.2f}", fontsize=10)
    a.set_xlabel("추종 시간 [s]"); a.set_ylabel("점수"); a.legend(fontsize=8, ncol=2); a.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out, "fig_score.png")


def fig_surface(out, S, O, urdf):
    from rb_servo_gui.surface import RobotSurface, compare_poses

    R = RobotSurface(urdf, n_points=600)
    q0 = np.deg2rad(S["qr"][0])
    names = R.joint_names
    sens = []
    for j in range(6):
        dq = np.zeros(6); dq[j] = np.deg2rad(1.0)
        r = compare_poses(R, q0 + dq, q0)
        sens.append((r["disp_max"], r["hausdorff"], r["chamfer"]))
    sens = np.array(sens)
    fig = Figure(figsize=(14, 4.8))
    ax = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.1, 1.3, 1.3]})
    a = ax[0]
    x = np.arange(6); w = 0.27
    for k, (lab, c) in enumerate((("대응점 변위 최대 D", "C3"), ("Hausdorff HD", "C1"), ("Chamfer CD", "C0"))):
        a.bar(x + (k - 1) * w, sens[:, k], w, label=lab, color=c)
    a.set_xticks(x); a.set_xticklabels([f"J{j + 1}\n{names[j]}" for j in range(6)], fontsize=8)
    a.set_ylabel("표면 차이 [mm]"); a.set_title("관절 하나가 1° 틀렸을 때 (rb10_1300e)", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")
    for ak, D, ttl in ((ax[1], S, "마지막 점 반복 적용 실행"), (ax[2], O, "종료 오버슈트 3.6° 실행 (반복 없음)")):
        n = len(D["qw"]); idx = np.unique(np.linspace(0, n - 1, min(n, 160)).astype(int))
        rows = [compare_poses(R, np.deg2rad(D["qw"][i]), np.deg2rad(D["pair"][i])) for i in idx]
        t = D["t_rel"][idx]
        ak.plot(t, [r["disp_max"] for r in rows], color="C3", lw=1.6, label="최악 점 변위 D_i")
        ak.plot(t, [r["hausdorff"] for r in rows], color="C1", lw=1.0, ls="--", label="Hausdorff HD_i")
        ak.plot(t, [r["chamfer"] for r in rows], color="C0", lw=1.6, label="Chamfer CD_i")
        ak.set_title(ttl, fontsize=10); ak.set_xlabel("추종 시간 [s]"); ak.set_ylabel("[mm]"); ak.legend(fontsize=8); ak.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out, "fig_surface.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servo", required=True)
    ap.add_argument("--jb2", required=True)
    ap.add_argument("--urdf", default=os.path.expanduser("~/flame_robotics_drt/urdf/rb10_1300e.urdf"))
    ap.add_argument("--overshoot", help="마지막 점 반복 없이 종료 오버슈트가 있던 servo 실행 (종료 각 오차 그림용, 없으면 --servo 사용)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "img"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    formula_images(args.out)
    S, J = compute(args.servo), compute(args.jb2)
    print("servo 예제:", S["run"], f"| T_track {S['T']:.1f}s, d_p95 {np.percentile(S['dev'], 95):.2f}, R {S['sm']['ripple']:.1%}, W {S['sm']['wobble']:.3f}")
    fig_timeline(args.out, S, J)
    fig_dtw(args.out, S)
    fig_error(args.out, S)
    fig_cover(args.out, S)
    fig_progress_step(args.out, S)
    fig_start_end(args.out, compute(args.overshoot) if args.overshoot else S)
    fig_lag(args.out, S)
    fig_velocity(args.out, S)
    fig_ripple(args.out, S)
    fig_wobble_real(args.out, S)
    fig_score(args.out, S)
    if args.overshoot and os.path.exists(args.urdf):
        fig_surface(args.out, S, compute(args.overshoot), args.urdf)


if __name__ == "__main__":
    main()
