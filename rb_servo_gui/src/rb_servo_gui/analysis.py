"""기록(follow_*.csv)을 원본 궤적과 비교해 추종 성능을 평가한다.

- DTW(순서 보존) 매칭으로 측정 샘플 <-> 원본 위치 짝을 정하고, 오차는 매칭쌍 기준으로 계산한다.
- 저장하는 그래프는 3종: sim_*.png(측정 기록), smooth_map_*.png(국소 거칠기 지도), compare_*_after.png(DTW 매칭 후 비교).
  (매칭 전 비교, 프로파일, 속도, 매끄러움 그래프는 저장하지 않는다. 관절 선택형 속도/프로파일은 GUI 의 ★ 항목에서 데이터로 직접 그린다.)
- 추종 시간(지령->모션 시작, 모션 시작->종료, 지령->완료)을 함께 기록한다.
pyplot 을 쓰지 않고 Figure 객체만 사용하므로 GUI 스레드와 충돌하지 않는다.
"""
import csv
import glob
import json
import os
import time
import warnings

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from .settings import Settings
from .matching import extrema_dtw


# ------------------------------------------------------------------ 입출력
def load_follow(path):
    """첫 줄 '# {json}' 메타 + CSV. 측정값만 있는 형식(meas0..5)과 jref/jang 이전 형식 모두 지원."""
    with open(path) as f:
        first = f.readline()
        meta = json.loads(first[1:]) if first.startswith("#") else {}
        if not first.startswith("#"):
            f.seek(0)
        r = list(csv.reader(f))
    hdr, rows = r[0], r[1:]
    t = np.array([float(x[0]) for x in rows])
    phase = np.array([x[1] for x in rows])
    state = np.array([int(x[2]) for x in rows])
    if "meas0" in hdr:
        meas = np.array([[float(v) for v in x[3:9]] for x in rows])
    else:
        meas = np.array([[float(v) for v in x[9:15]] for x in rows])
        if np.abs(meas).max() == 0.0:
            meas = np.array([[float(v) for v in x[3:9]] for x in rows])
    return t, phase, meas, state, meta


# ------------------------------------------------------------------ 기하/매칭
def motion_window(t, q, final, tol, movethr, settle_tol=0.05):
    """모션 시작/종료 인덱스. 시작: 시작 자세에서 movethr 이상 벗어나기 직전, 종료: 이후 끝까지 final 의 tol 이내로 유지되기 시작한 점,
    다시 그 뒤 final 의 settle_tol(0.05°) 이내에 처음 닿는 샘플까지 연장 (마지막 샘플 = 원본 끝점 자세)."""
    moved = np.abs(q - q[0]).max(1) > movethr
    i_s = max(int(np.argmax(moved)) - 1, 0) if moved.any() else 0
    ok = np.abs(q - final).max(1) < tol
    suffix = np.flip(np.logical_and.accumulate(np.flip(ok)))
    reached = bool(suffix.any())
    i_e = int(np.argmax(suffix)) if reached else len(t) - 1
    if reached:  # 마지막 샘플을 원본 끝점(멈춘 자세)에 붙인다: tol 안쪽에서 끊기면 아직 움직이는 중이라, 실제로 끝점에 닿은 첫 샘플까지 구간을 늘린다
        near = suffix & (np.abs(q - final).max(1) < min(tol, settle_tol))
        if near.any():
            i_e = int(np.argmax(near))
    return i_s, max(i_e, i_s + 1), reached


def ref_progress(qr):
    """원본 경로의 호길이 기반 진행도 s(0~1)."""
    seg = np.linalg.norm(np.diff(qr, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    return (cum / cum[-1] if cum[-1] > 0 else np.zeros_like(cum)), cum, seg


def ordered_match(meas, qr, kmax=None):
    """측정 샘플 -> 원본 정점 인덱스의 순서를 지키는 매칭(비감소 정렬, DTW 계열 동적계획법).

    다른 시각에 우연히 같은 각도를 지나가도 거기에 매칭되지 않고, 시간 지연/속도 차이는 허용한다.
    시작/종료는 각각 원본 시작/끝 근처로 제한한다.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.spatial.distance import cdist

    n, m = len(meas), len(qr)
    kmax = max(30, m // 5) if kmax is None else kmax
    C = cdist(meas, qr).astype(np.float32)
    D = np.full((n, m), np.inf, dtype=np.float32)
    head = max(5, m // 20)
    D[0, :head] = C[0, :head]
    for i in range(1, n):
        prev = np.pad(D[i - 1], (kmax, 0), constant_values=np.inf)
        D[i] = C[i] + sliding_window_view(prev, kmax + 1).min(1)
    tail = max(5, m // 20)
    last = D[-1, m - tail:]
    j = m - tail + int(np.argmin(last)) if np.isfinite(last).any() else int(np.argmin(D[-1]))
    idx = np.zeros(n, dtype=int)
    idx[-1] = j
    for i in range(n - 1, 0, -1):
        lo = max(j - kmax, 0)
        j = lo + int(np.argmin(D[i - 1, lo:j + 1]))
        idx[i - 1] = j
    return idx


def refine_pairs(meas, qr, idx, cum, seg, tr, anchors=None):
    """Project between fixed extrema anchors, preserving one monotone mapping.

    Anchor pairs remain EXACT even when moving the reference toward the measured
    pose would reduce the error. Position, time, and progress share one parameter.
    """
    a, ab = qr[:-1], np.diff(qr, axis=0)
    denom = np.maximum((ab * ab).sum(1), 1e-12)
    n = len(meas)
    proj, s, t, dist = np.zeros_like(meas), np.zeros(n), np.zeros(n), np.zeros(n)
    fixed = dict(anchors or [])
    upcoming = sorted(fixed)
    previous = 0.0
    for i, p in enumerate(meas):
        if i in fixed:
            parameter = float(fixed[i])
        else:
            next_anchor = next((k for k in upcoming if k > i), None)
            upper = float(fixed[next_anchor]) if next_anchor is not None else float(len(qr)-1)
            cand = np.unique(np.clip([idx[i]-1, idx[i], int(previous)], 0, len(a)-1))
            low_u = np.maximum(0.0, previous-cand)
            high_u = np.minimum(1.0, upper-cand)
            valid = low_u <= high_u
            cand, low_u, high_u = cand[valid], low_u[valid], high_u[valid]
            if not len(cand):
                raise ValueError('No monotone projection within extrema anchors')
            u = np.clip(((p-a[cand])*ab[cand]).sum(1)/denom[cand],low_u,high_u)
            d = np.linalg.norm(a[cand]+u[:,None]*ab[cand]-p,axis=1)
            k = int(np.argmin(d))
            parameter = float(cand[k]+u[k])
        c = min(int(parameter),len(a)-1)
        u = parameter-c
        proj[i] = a[c]+u*ab[c]
        s[i] = (cum[c]+u*seg[c])/cum[-1] if cum[-1] > 0 else 0.0
        t[i] = tr[c]-tr[0]+u*(tr[c+1]-tr[c])
        dist[i] = np.linalg.norm(proj[i]-p)
        previous = parameter
    return proj,s,t,dist


def coverage(qr, meas, idx=None):
    """원본 각 점에서 측정 경로까지의 거리(매칭된 구간 근처만). 건너뛴 구간이 있으면 커진다."""
    a, ab = meas[:-1], np.diff(meas, axis=0)
    denom = np.maximum((ab * ab).sum(1), 1e-12)
    out = np.zeros(len(qr))
    for k, p in enumerate(qr):
        if idx is None:
            sl = slice(0, len(a))
        else:
            i0 = int(np.searchsorted(idx, k, "left"))
            i1 = int(np.searchsorted(idx, k, "right"))
            lo = max(i0 - 2, 0)
            hi = min(max(i1, i0 + 1) + 1, len(a))
            sl = slice(lo, max(hi, lo + 1))
        u = np.clip(((p - a[sl]) * ab[sl]).sum(1) / denom[sl], 0, 1)
        out[k] = np.linalg.norm(a[sl] + u[:, None] * ab[sl] - p, axis=1).min()
    return out


def velocity(t, q, win=0.2):
    """관절 속도 [deg/s]. t±win/2 초 구간의 중심 차분이라 측정 샘플 시각 지터에 강하다."""
    n = len(t)
    v = np.zeros_like(q)
    for i in range(n):
        lo = int(np.searchsorted(t, t[i] - win / 2, "left"))
        hi = min(int(np.searchsorted(t, t[i] + win / 2, "right")) - 1, n - 1)
        if hi <= lo:
            hi = min(lo + 1, n - 1)
        dt = t[hi] - t[lo]
        v[i] = (q[hi] - q[lo]) / dt if dt > 0 else 0.0
    return v


def moving_average(x, n):
    """축 0 방향 이동평균(가장자리는 끝값으로 연장). n 은 샘플 수."""
    n = max(int(n), 1)
    if n == 1:
        return x.copy()
    pad = n // 2
    xp = np.concatenate([np.repeat(x[:1], pad, axis=0), x, np.repeat(x[-1:], n - 1 - pad, axis=0)])
    ker = np.ones(n) / n
    if x.ndim == 1:
        return np.convolve(xp, ker, mode="valid")
    return np.column_stack([np.convolve(xp[:, j], ker, mode="valid") for j in range(x.shape[1])])


def smoothness(t_rel, s_arc, s_prog, e_pair, T_meas, th_ripple=0.15, th_wobble=0.3):
    """추종의 매끄러움. 경로 자체의 비매끄러움(꺾임 등)은 반영하지 않도록 경로 모양과 무관한 양만 본다.

    - 속도 리플: 경로를 따라가는 속력(호길이 변화율, 경로 모양과 무관)의 고주파 성분 RMS / 평균 속력
    - 횡방향 흔들림: 궤적 추종 오차(원본 경로 모양은 상쇄됨)의 고주파 성분 RMS [deg, 6관절 L2]
    - 정체 비율: 경로 중간(진행도 5~95%)에서 속력이 중앙값의 20% 미만인 시간 비율
    고주파 = 이동평균 창(추종 시간/10, 0.3~2 s)을 뺀 나머지. 가장자리 창 절반은 제외한다.
    """
    dts = float(np.median(np.diff(t_rel)))
    win = float(np.clip(T_meas / 10, 0.3, 2.0))
    n = max(int(round(win / dts)), 3)
    v_s = velocity(t_rel, s_arc[:, None])[:, 0]
    trend = moving_average(v_s, n)
    hp = v_s - trend
    core = slice(n // 2, len(v_s) - n // 2) if len(v_s) > 2 * n else slice(0, len(v_s))
    mean_v = float(np.mean(v_s[core])) if np.mean(v_s[core]) > 1e-9 else 1e-9
    ripple = float(np.sqrt(np.mean(hp[core] ** 2)) / mean_v)
    e_hp = e_pair - moving_average(e_pair, n)
    wobble = float(np.sqrt(np.mean(np.sum(e_hp[core] ** 2, axis=1))))
    mid = (s_prog > 0.05) & (s_prog < 0.95)
    med = float(np.median(v_s[mid])) if mid.any() else mean_v
    stall = float(np.mean(v_s[mid] < 0.2 * med)) if mid.any() else 0.0
    # 국소 거칠기(이동 창 RMS): 경로 위치별로 어디서 거칠어지는지. 점수 = 각 지표 / 합격선 중 큰 값 (1 = 기준선)
    rip_loc = np.sqrt(np.maximum(moving_average(hp ** 2, n), 0)) / mean_v
    wob_loc = np.sqrt(np.maximum(moving_average(np.sum(e_hp ** 2, axis=1), n), 0))
    score = np.maximum(rip_loc / th_ripple, wob_loc / th_wobble)
    return {"ripple": ripple, "wobble": wobble, "stall": stall, "window": win,
            "v_s": v_s, "trend": trend, "e_hp": e_hp,
            "rip_loc": rip_loc, "wob_loc": wob_loc, "score": score, "n": n}


_ROBOTS = {}


def surface_errors(s, qw, pair_ref, t_rel, log=print):
    """URDF 의 링크 메시로 각도 오차를 표면 위치 차이(mm)로 환산한다. urdf_path 가 없으면 None.

    측정 샘플 i 의 자세 qw[i] 와 DTW 로 짝지은 원본 자세 pair_ref[i] 를 각각 FK 로 배치해 비교한다 (surface.py).
    샘플은 surface_samples 개를 균등하게 뽑는다. 반환: {"idx", "t", "series": {...}, "links": {링크: 최대 변위}, "summary": {...}}.
    """
    path = os.path.expanduser(getattr(s, "urdf_path", "") or "")
    if not path:
        return None
    if not os.path.exists(path):
        log(f"[표면 오차] URDF 를 찾을 수 없음: {path}")
        return None
    from .surface import RobotSurface, compare_poses

    key = (os.path.abspath(path), int(s.surface_points))
    if key not in _ROBOTS:
        _ROBOTS[key] = RobotSurface(path, n_points=int(s.surface_points))
    robot = _ROBOTS[key]
    n = len(qw)
    idx = np.unique(np.linspace(0, n - 1, min(n, int(s.surface_samples))).astype(int))
    rows = [compare_poses(robot, np.deg2rad(qw[i]), np.deg2rad(pair_ref[i])) for i in idx]
    series = {k: np.array([r[k] for r in rows]) for k in ("disp_max", "disp_p95", "disp_mean", "chamfer", "hausdorff")}
    links = {name: float(max(r["link_max"][name] for r in rows)) for name in rows[0]["link_max"]}
    worst = max(links, key=links.get)
    summary = {
        "surf_disp_max": float(series["disp_max"].max()), "surf_disp_p95": float(np.percentile(series["disp_max"], 95)),
        "surf_disp_mean": float(series["disp_max"].mean()),
        "surf_chamfer_max": float(series["chamfer"].max()), "surf_chamfer_mean": float(series["chamfer"].mean()),
        "surf_hausdorff_max": float(series["hausdorff"].max()), "surf_worst_link": worst,
    }
    return {"idx": idx, "t": t_rel[idx], "series": series, "links": links, "summary": summary}


SURFACE_KEYS = ("surf_disp_max", "surf_disp_p95", "surf_disp_mean", "surf_chamfer_max", "surf_chamfer_mean", "surf_hausdorff_max", "surf_worst_link")


def runs(mask):
    """True 가 연속된 구간의 (시작, 끝) 인덱스 목록."""
    out, st = [], None
    for i, v in enumerate(mask):
        if v and st is None:
            st = i
        if not v and st is not None:
            out.append((st, i - 1))
            st = None
    if st is not None:
        out.append((st, len(mask) - 1))
    return out


# ------------------------------------------------------------------ 그래프 (Figure 반환)
def _tab10():
    return colormaps["tab10"](np.arange(6))


def ideal_profile(run_dir, mp, rate=100.0):
    """servo 프로파일 타이밍(params timing=profile)일 때 이상 위치/속도 프로파일과 전송 시간표를 읽는다. 아니면 None.

    실행 때와 같은 계산으로 다시 만든다: 원본 궤적(traj_ref.csv) 위의 관절 속도·가속도 한계 프로파일 (params 에 period 가 있는 실행).
    period 가 없는 이전 실행(2026-10-14 16시대, 지령점 꺾은선 사다리꼴)은 그때 방식(traj_sub*.csv)으로 다시 만든다.
    반환: {"t": (n,), "q": (n,6), "v": (n,6), "T", "vp", "label", "sched": (전송 시각, 도달 시각, 목표 (k,6)) 또는 None}
    """
    if not mp or mp.get("timing") != "profile":
        return None
    from .trajectory import PathProfile, SpeedProfile
    basis = mp.get("basis", "max_joint")
    if "period" in mp:
        qr = np.rad2deg(np.loadtxt(os.path.join(run_dir, "traj_ref.csv"), delimiter=",", skiprows=1, ndmin=2)[:, 1:7])
        prof = PathProfile.fit(qr, mp["speed"], mp["accel"], basis, mp["period"]) if "accel_plan" in mp else PathProfile(qr, mp["speed"], mp["accel"], basis)
    else:
        sf = sorted(glob.glob(os.path.join(run_dir, "traj_sub*.csv")))
        if not sf:
            return None
        prof = SpeedProfile(np.rad2deg(np.loadtxt(sf[0], delimiter=",", skiprows=1, ndmin=2)[:, 1:7]), mp["speed"], mp["accel"], basis)
    tt = np.linspace(0.0, prof.T, max(int(prof.T * rate), 2) + 1)
    q, v = prof.joints(tt)
    sched = None
    sp = os.path.join(run_dir, "servo_schedule.csv")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            hdr = f.readline().strip().split(",")
        a = np.loadtxt(sp, delimiter=",", skiprows=1, ndmin=2)[: int(mp.get("n_sent") or 10 ** 9)]
        c1, cq = hdr.index("t1"), hdr.index("q0")
        sched = (a[:, 0], a[:, 0] + a[:, c1], a[:, cq:cq + 6])
    label = f"ideal: speed {mp['speed']:g}, accel {mp['accel']:g} ({basis})"
    acc = np.gradient(v, tt, axis=0)
    return {"t": tt, "q": q, "v": v, "a": acc, "T": prof.T, "vp": prof.vp, "label": label, "sched": sched, "basis": basis}


def fig_sim(stem, t, phase, meas, state, meta, fig=None, ideal=None):
    """시뮬레이션 기록: 측정 관절각/각속도/robot_state 전체 타임라인."""
    ker = np.ones(5) / 5
    v = np.gradient(meas, t, axis=0)
    v = np.column_stack([np.convolve(v[:, j], ker, mode="same") for j in range(6)])
    if fig is None:
        fig = Figure(figsize=(12, 9))
    else:
        fig.clear()
    ax = fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 3, 1]})
    c = _tab10()
    for j in range(6):
        ax[0].plot(t, meas[:, j], color=c[j], lw=1.2, label=f"j{j}")
        ax[1].plot(t, v[:, j], color=c[j], lw=0.9)
    if ideal is not None and "servo" in meta.get("marks", {}):
        t0i = meta["marks"]["servo"]  # 이상 프로파일 t=0 = 첫 servo 지령 시각
        for j in range(6):
            ax[0].plot(ideal["t"] + t0i, ideal["q"][:, j], ":", color=c[j], lw=1.4)
            ax[1].plot(ideal["t"] + t0i, ideal["v"][:, j], ":", color=c[j], lw=1.4)
        ax[1].text(0.01, 0.97, f"dotted = {ideal['label']} (t=0 at first servo command)", transform=ax[1].transAxes, fontsize=8, va="top")
    ax[2].step(t, state, where="post", color="k", lw=1.0)
    ax[2].set_yticks([1, 3])
    ax[2].set_yticklabels(["idle", "run"])
    ax[2].set_ylabel("robot_state")
    edges = [0] + [i for i in range(1, len(phase)) if phase[i] != phase[i - 1]] + [len(phase)]
    shade = colormaps["Pastel1"](np.arange(9))
    for n, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        for a_ in ax:
            a_.axvspan(t[a], t[min(b, len(t) - 1)], color=shade[n % 9], alpha=0.35, lw=0)
        ax[0].text(t[a], 1.0, phase[a], transform=ax[0].get_xaxis_transform(), rotation=35, fontsize=7, va="bottom", ha="left")
    ms = ", ".join(f"{m}: " + ", ".join(f"{k}={v_:g}" if isinstance(v_, (int, float)) else f"{k}={v_}"
                                       for k, v_ in mp.items()) for m, mp in meta.get("methods", {}).items())
    ax[0].set_title(f"simulation record | {ms}\n{stem}", fontsize=9, pad=28)
    ax[0].set_ylabel("measured angle [deg]")
    ax[0].legend(ncol=6, fontsize=8, frameon=False, loc="upper right")
    ax[1].set_ylabel("angular velocity [deg/s]\n(5-sample avg)")
    ax[2].set_xlabel("recording time [s]")
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_smooth_map(head, qw, qr, t_rel, s_prog, sm, th_ripple, th_wobble, fig=None):
    """매끄러움의 정도 시각화: 경로 위에 국소 거칠기 점수를 색으로 표시.

    점수 = max(국소 속도 리플 / 기준, 국소 횡방향 흔들림 / 기준). 1 이상이면 기준 초과(붉은색). 경로 자체의 꺾임은 점수에 반영되지 않는다.
    """
    from matplotlib.gridspec import GridSpec

    score = sm["score"]
    if fig is None:
        fig = Figure(figsize=(13, 9))
    else:
        fig.clear()
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 2], width_ratios=[1, 1])
    # --- (1) 6D 경로를 주성분 평면에 투영, 측정 경로를 점수로 색칠
    mu = qr.mean(0)
    _, _, vt = np.linalg.svd(qr - mu, full_matrices=False)
    P = (qw - mu) @ vt[:2].T
    R = (qr - mu) @ vt[:2].T
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(R[:, 0], R[:, 1], color="0.75", lw=3, label="reference path", zorder=1)
    sc = ax.scatter(P[:, 0], P[:, 1], c=np.clip(score, 0, 2), cmap="RdYlGn_r", vmin=0, vmax=2, s=10, zorder=2)
    ax.plot(P[0, 0], P[0, 1], "ko", ms=7, zorder=3)
    ax.plot(P[-1, 0], P[-1, 1], "ks", ms=7, zorder=3)
    ax.text(P[0, 0], P[0, 1], " start", fontsize=8)
    ax.text(P[-1, 0], P[-1, 1], " end", fontsize=8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("roughness score (1 = limit)")
    ax.set_xlabel("PC1 [deg]")
    ax.set_ylabel("PC2 [deg]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("path colored by local roughness (PCA projection of 6-joint path)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="best")
    # --- (2) 경로 진행도별 국소 지표
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(s_prog, sm["rip_loc"] * 100, color="C0", lw=1.4, label="local speed ripple [%]")
    ax2.axhline(th_ripple * 100, color="C0", ls=":", lw=1)
    ax2.set_xlabel("path progress")
    ax2.set_ylabel("speed ripple [%]", color="C0")
    ax3 = ax2.twinx()
    ax3.plot(s_prog, sm["wob_loc"], color="C3", lw=1.4, label="local lateral wobble [deg]")
    ax3.axhline(th_wobble, color="C3", ls=":", lw=1)
    ax3.set_ylabel("lateral wobble [deg]", color="C3")
    bad = score > 1
    for a_, b_ in runs(bad):
        ax2.axvspan(s_prog[a_], s_prog[b_], color="red", alpha=0.15, lw=0)
    ax2.set_title("local smoothness along the path (red = over limit, dotted = limit)", fontsize=9)
    ax2.grid(alpha=0.3)
    # --- (3) 시간에 따른 점수
    ax4 = fig.add_subplot(gs[1, :])
    ax4.fill_between(t_rel, 0, np.clip(score, 0, None), where=score <= 1, color="tab:green", alpha=0.5, step="mid")
    ax4.fill_between(t_rel, 0, np.clip(score, 0, None), where=score > 1, color="tab:red", alpha=0.6, step="mid")
    ax4.plot(t_rel, score, color="k", lw=0.8)
    ax4.axhline(1, color="r", ls="--", lw=1)
    ax4.set_xlabel("tracking time [s] (0 = motion start)")
    ax4.set_ylabel("roughness score")
    ax4.set_xlim(t_rel[0], t_rel[-1])
    ax4.grid(alpha=0.3)
    ax4.set_title(f"roughness over time | over limit {np.mean(bad):.0%} of the time, max score {score.max():.2f}", fontsize=9)
    fig.suptitle(head, fontsize=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # 창이 작을 때의 tight_layout 경고
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def profile_data(run_dir, mode):
    """실행 폴더에 저장된 파일(traj_ref/traj_sub/compare_*_pairs/compare_*.json)만으로 프로파일 그래프에 필요한 데이터를 읽는다.
    GUI 에서 관절을 골라 다시 그릴 때 쓴다. 반환: dict(head, tr, qr, sub, t_rel, qw, vel, offset, speed_cmd)."""
    cj = sorted(glob.glob(os.path.join(run_dir, f"compare_*_{mode}.json")))[-1]
    with open(cj, encoding="utf-8") as f:
        r = json.load(f)
    M, mp = r["metrics"], r["params"]
    ref = np.loadtxt(os.path.join(run_dir, "traj_ref.csv"), delimiter=",", skiprows=1)
    sf = sorted(glob.glob(os.path.join(run_dir, "traj_sub*.csv")))
    sub = None
    if sf:
        sa = np.loadtxt(sf[0], delimiter=",", skiprows=1)
        sub = (sa[:, 0], np.rad2deg(sa[:, 1:7]))
    with open(cj.replace(".json", "_pairs.csv"), encoding="utf-8") as f:
        f.readline()  # '# {json}' 메타
        rows = list(csv.reader(f))
    t_rel = np.array([float(x[0]) for x in rows[1:]])
    qw = np.array([[float(v) for v in x[4:10]] for x in rows[1:]])
    desc = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in mp.items() if v is not None)
    fm = lambda v: "-" if v is None else f"{v:.1f}s"  # noqa: E731
    head = f"{desc} | step=1/{r.get('step')} | tracking {fm(M.get('t_track'))} (ref {fm(M.get('t_ref'))}) | {os.path.basename(run_dir)}"
    return {"head": head, "tr": ref[:, 0], "qr": np.rad2deg(ref[:, 1:7]), "sub": sub, "t_rel": t_rel, "qw": qw,
            "vel": velocity(t_rel, qw), "offset": -(M.get("t_latency") or 0.0), "speed_cmd": mp.get("speed"), "accel_cmd": mp.get("accel"),
            "basis": mp.get("basis", "max_joint"),
            "ideal": ideal_profile(run_dir, mp) if mode == "servo" else None}


def fig_profile(head, mode, tr, qr, sub, t_rel, qw, vel, offset=None, joints=None, fig=None, absolute=True, ideal=None):
    """위치/속도 프로파일 (2x2). 위: 원본 궤적과 실제로 보낸 지령점(위치, 속도) / 아래: 실제 추종(위치, 속도).

    servo 는 지령이 실시간으로 나가므로 아래 패널에 원본을 같은 시간축으로 겹쳐 그린다(offset = 지령 시각 - 모션 시작).
    jb2 는 로봇이 자체 시간으로 실행하므로 시간축이 달라 겹치지 않는다.
    ideal(servo 프로파일 타이밍)이 있으면 위 패널은 이상 프로파일(점선)과 실제 보낸 지령(도달 시각 기준),
    아래 패널은 실측과 이상 프로파일(점선, 모션 시작 = t 0)을 겹친다.
    """
    c = _tab10()
    joints = list(range(6)) if joints is None else list(joints)
    fig_given = fig is not None
    if fig is None:
        fig = Figure(figsize=(14, 9))
    else:
        fig.clear()
    ax = fig.subplots(2, 2)
    t0 = tr - tr[0]
    vref = np.gradient(qr, tr, axis=0)
    sg = np.abs if absolute else (lambda x: x)  # absolute=True 면 속도는 크기(|v|)로 그린다
    if ideal is not None:
        for j in joints:
            ax[0, 0].plot(ideal["t"], ideal["q"][:, j], ":", color=c[j], lw=1.6, label=f"j{j}")
            ax[0, 1].plot(ideal["t"], sg(ideal["v"][:, j]), ":", color=c[j], lw=1.6)
        if ideal["sched"] is not None:
            t_send, t_arr, qk = ideal["sched"]
            for j in joints:
                ax[0, 0].plot(t_arr, qk[:, j], ".", color=c[j], ms=4)
                prev = np.vstack([ideal["q"][0], qk[:-1]])
                vk = (qk[:, j] - prev[:, j]) / np.maximum(t_arr - t_send, 1e-9)
                ax[0, 1].step(t_send, sg(vk), where="post", color=c[j], lw=0.8, alpha=0.6)
        sub = None  # 원본 시간 기준 지령점은 그리지 않음
    else:
        for j in joints:
            ax[0, 0].plot(t0, qr[:, j], color=c[j], lw=1.2, label=f"j{j}")
            ax[0, 1].plot(t0, sg(vref[:, j]), color=c[j], lw=1.0)
    if sub is not None:
        ts, qs = sub[0] - sub[0][0], sub[1]
        vs = np.diff(qs, axis=0) / np.diff(ts)[:, None]
        for j in joints:
            ax[0, 0].plot(ts, qs[:, j], ".", color=c[j], ms=4)
            ax[0, 1].step(ts[:-1], sg(vs[:, j]), where="post", color=c[j], lw=0.8, alpha=0.6)
    gui = fig_given  # 화면(GUI)용은 영역이 좁아 제목을 짧게
    if ideal is not None:
        ax[0, 0].set_title("ideal (dotted) + commands: position" if gui else f"{ideal['label']} (dotted) and sent points at planned arrival (dots) — position", fontsize=8 if gui else 9)
        ax[0, 1].set_title("ideal (dotted) + commands: velocity" if gui else "ideal velocity (dotted) and sent segments Δq/t1 (steps) — velocity", fontsize=8 if gui else 9)
    else:
        ax[0, 0].set_title("reference + commands: position" if gui else "reference (line) and commanded points (dots) — position", fontsize=8 if gui else 9)
        ax[0, 1].set_title("reference + commands: velocity" if gui else "reference (line) and commanded segments (steps) — velocity", fontsize=8 if gui else 9)
    ax[0, 0].set_ylabel("angle [deg]")
    ax[0, 1].set_ylabel("|velocity| [deg/s]" if absolute else "velocity [deg/s]")
    for a in ax[0]:
        a.set_xlabel("profile time [s] (0 = first command)" if ideal is not None else "reference time [s]", fontsize=8 if gui else 10)
    for j in joints:
        ax[1, 0].plot(t_rel, qw[:, j], color=c[j], lw=1.3)
        ax[1, 1].plot(t_rel, sg(vel[:, j]), color=c[j], lw=1.1)
        if ideal is not None:  # 이상 프로파일: 모션 시작을 t=0 으로 맞춤
            ax[1, 0].plot(ideal["t"], ideal["q"][:, j], ":", color=c[j], lw=1.6)
            ax[1, 1].plot(ideal["t"], sg(ideal["v"][:, j]), ":", color=c[j], lw=1.6)
        elif mode == "servo" and offset is not None:  # 실시간 지령: 원본을 같은 시간축에 겹침
            ax[1, 0].plot(t0 + offset, qr[:, j], "--", color=c[j], lw=0.8, alpha=0.7)
            ax[1, 1].plot(t0 + offset, sg(vref[:, j]), "--", color=c[j], lw=0.8, alpha=0.7)
    if ideal is not None:
        ov = "  (dotted = ideal profile, start aligned to motion start)"
    else:
        ov = "  (dashed = reference on the same clock, offset by command latency)" if mode == "servo" and offset is not None else ""
    if gui:
        tag = " (dotted=ideal)" if ideal is not None else (" (dashed=ref)" if ov else "")
        ax[1, 0].set_title("measured: position" + tag, fontsize=8)
        ax[1, 1].set_title("measured: velocity" + tag, fontsize=8)
    else:
        ax[1, 0].set_title("measured — position" + ov, fontsize=9)
        ax[1, 1].set_title("measured — velocity (±0.1 s central difference)", fontsize=9)
    ax[1, 0].set_ylabel("angle [deg]")
    ax[1, 1].set_ylabel("|velocity| [deg/s]" if absolute else "velocity [deg/s]")
    for a in ax[1]:
        a.set_xlabel("tracking time [s]" if gui else "tracking time [s] (0 = motion start)", fontsize=8 if gui else 10)
    ax[0, 0].legend(ncol=min(max(len(joints), 1), 6), fontsize=8, frameon=False, loc="lower right")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(f"POSITION / VELOCITY PROFILES | joints {[j + 1 for j in joints]}" + ("" if gui else f" | {head}"), fontsize=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # 창이 작을 때의 tight_layout 경고
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def fig_velocity(head, t_rel, v, speed_cmd=None, joints=None, fig=None, absolute=True, ideal=None, accel_cmd=None, basis="max_joint"):
    """실제 추종 시 속도·가속도. 시간축은 추종 시작 기준 실제 시간.

    위: 관절별 각속도. 가운데: 속도 한계 대상(basis=max_joint 면 선택 관절 중 최대 |v|, l2 면 ‖v‖₂)과 설정 속도.
    아래: 가속도 한계 대상(같은 기준, ±0.05 s 중심 차분)과 설정 가속도.
    ideal(servo 프로파일 타이밍)이 있으면 이상 프로파일을 점선으로 겹친다 (모션 시작 = t 0)."""
    joints = list(range(6)) if joints is None else list(joints)
    if fig is None:
        fig = Figure(figsize=(11, 10))
    else:
        fig.clear()
    ax = fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]})
    c = _tab10()
    for j in joints:
        ax[0].plot(t_rel, (np.abs(v[:, j]) if absolute else v[:, j]), color=c[j], lw=1.2, label=f"j{j}")
        if ideal is not None:
            ax[0].plot(ideal["t"], (np.abs(ideal["v"][:, j]) if absolute else ideal["v"][:, j]), ":", color=c[j], lw=1.6)
    if not absolute:
        ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_ylabel("joint speed |v| [deg/s]" if absolute else "joint velocity [deg/s]")
    ax[0].set_title(f"MEASURED TRACKING VELOCITY | joints {[j + 1 for j in joints]}" + ("  (dotted = ideal profile)" if ideal is not None else "") + f" | {head}", fontsize=8)
    ax[0].legend(ncol=min(max(len(joints), 1), 6), fontsize=8, frameon=False, loc="lower right")

    l2 = basis == "l2"
    nrm = (lambda x: np.linalg.norm(x[:, joints], axis=1)) if l2 else (lambda x: np.abs(x[:, joints]).max(axis=1))
    what = "‖v‖₂" if l2 else "max joint |v|"
    acc = velocity(t_rel, v, win=0.1)
    sp, ap = nrm(v), nrm(acc)
    ax[1].plot(t_rel, sp, color="k", lw=1.4, label=f"measured {what}  max {sp.max():.1f} deg/s")
    ax[2].plot(t_rel, ap, color="k", lw=1.2, label=f"measured {what.replace('v', 'a')} (±0.05 s)  max {ap.max():.0f} deg/s²")
    if ideal is not None:
        ax[1].plot(ideal["t"], nrm(ideal["v"]), ":", color="C3", lw=1.8, label=f"ideal  max {nrm(ideal['v']).max():.1f}")
        ax[2].plot(ideal["t"], nrm(ideal["a"]), ":", color="C3", lw=1.8, label=f"ideal  max {nrm(ideal['a']).max():.0f}")
    if speed_cmd:
        ax[1].axhline(speed_cmd, color="r", ls="--", lw=1, label=f"speed limit {speed_cmd:g} deg/s")
    if accel_cmd:
        ax[2].axhline(accel_cmd, color="r", ls="--", lw=1, label=f"accel limit {accel_cmd:g} deg/s²")
    ax[1].set_ylabel(f"{what} [deg/s]")
    ax[2].set_ylabel(f"{what.replace('v', 'a')} [deg/s²]")
    ax[2].set_xlabel("tracking time [s] (0 = motion start)")
    for a2 in ax[1:]:
        a2.legend(fontsize=8, frameon=False, loc="upper right")
    for a2 in ax:
        a2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_after(head, tr, qr, t_match, qw, e_pair, dev, t_meas_s, T_ref, T_meas, rmse, M, end_err, fig=None, matching=None):
    """매칭 후: DTW 매칭쌍 기준. 측정 시각 -> 매칭된 원본 시각(6관절 공통 하나의 변환)."""
    if fig is None:
        fig = Figure(figsize=(11, 12))
    else:
        fig.clear()
    ax = fig.subplots(3, 1, gridspec_kw={"height_ratios": [3, 2, 2]})
    c = _tab10()
    for j in range(6):
        ax[0].plot(tr - tr[0], qr[:, j], color=c[j], lw=1.4, label=f"j{j}")
        ax[0].plot(t_match, qw[:, j], "--", color=c[j], lw=1.2)
        ax[1].plot(t_match, e_pair[:, j], color=c[j], lw=1.0)
    ax[1].plot(t_match, dev, color="k", lw=1.4, label="L2 tracking error")
    if matching:
        for event in matching.get('extrema', []):
            if event['status'] != 'anchor':
                continue
            j = event['joint']-1
            ax[0].scatter([event['ref_time_s']], [event['meas_deg']],
                          marker='x', s=35, color=c[j], zorder=5)
            ax[1].scatter([event['ref_time_s']], [event['error_deg']],
                          marker='x', s=35, color=c[j], zorder=5)

    ax[0].set_ylabel("angle [deg]")
    method_title = 'extrema anchors + segment DTW; x = extrema anchor' if matching else 'ordered matching pairs'
    ax[0].set_title(f"AFTER DTW ({method_title}) | {head}\n"
                    f"trajectory tracking error (DTW) RMSE {rmse:.2f} deg, L2 p95 {M['path_dev_p95']:.2f} / max {M['path_dev_max']:.2f} deg, "
                    f"coverage max {M['cover_max']:.2f} deg, progress {M['progress_max']:.1%}, end err {end_err:.2f} deg\n"
                    "(x = matched reference time; solid = reference, dashed = measured re-timed by DTW pairs)", fontsize=8)
    ax[0].legend(ncol=6, fontsize=8, frameon=False, loc="lower right")
    ax[1].set_ylabel("trajectory tracking error [deg]\n(measured - DTW-matched ref)")
    ax[1].legend(fontsize=8, frameon=False, loc="upper right")
    for a2 in ax[:2]:
        a2.set_xlim(0, T_ref)
        a2.set_xlabel("matched reference time [s]")
    ax[2].plot(t_meas_s, t_match, color="C3", lw=1.6, label="DTW matched ref time")
    ax[2].plot([0, T_meas], [0, T_ref], color="gray", ls=":", lw=1.2, label="uniform scaling (start/end aligned)")
    ax[2].set_ylabel("matched reference time [s]")
    ax[2].set_xlabel(f"tracking time [s]  (0 - {T_meas:.1f})")
    ax[2].legend(fontsize=8, frameon=False)
    for a2 in ax:
        a2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ 분석 본체
def analyze(run_dir, s=None, ref_path=None, follow_path=None, summary_dir=None, log=print, npts=500):
    """실행 폴더 하나를 분석한다. 반환: {mode: {"metrics":…, "verdict":…, "files":[…]}}.

    run_dir 안의 follow_*.csv(가장 최근)와 traj_ref.csv(기준 궤적)를 사용한다. s 는 판정 기준/허용오차(Settings).
    결과: sim_*.png, compare_*_before.png/_after.png, compare_*.json, compare_*_pairs.csv, summary.csv 한 줄.
    """
    s = s or Settings()
    path = follow_path or max(glob.glob(os.path.join(run_dir, "follow_*.csv")), key=os.path.getmtime)
    ref_path = ref_path or os.path.join(run_dir, "traj_ref.csv")
    ref = np.genfromtxt(ref_path, delimiter=",", skip_header=1)
    tr, qr = ref[:, 0], np.rad2deg(ref[:, 1:7])
    T_ref = tr[-1] - tr[0]
    tau_ref = (tr - tr[0]) / T_ref
    s_ref, cum, seg = ref_progress(qr)

    stem = os.path.splitext(os.path.basename(path))[0]
    t, phase, meas, state, meta = load_follow(path)
    marks = meta.get("marks", {})
    results = {}
    files = []

    f = fig_sim(stem, t, phase, meas, state, meta, ideal=ideal_profile(run_dir, meta.get("methods", {}).get("servo")))
    p = os.path.join(run_dir, f"sim_{stem}.png")
    f.savefig(p, dpi=120)
    files.append(p)

    tau = np.linspace(0.0, 1.0, npts)
    qr_n = np.column_stack([np.interp(tau, tau_ref, qr[:, j]) for j in range(6)])

    for m in [k for k in ("jb2", "servo") if k in marks]:
        sel = np.array([p_.startswith(m) and not p_.endswith("move_to_start") for p_ in phase])
        ts, q = t[sel], meas[sel]
        # 모션 종료는 '목표'가 아니라 로봇이 실제로 멈춘 자세(마지막 안정 구간) 기준으로 찾는다. 종료 오차가 reach_tol 보다 커도
        # (예: 서보 오버슈트) 추종 시간이 기록 끝까지 늘어나지 않는다. 목표 도달 여부(reached)는 멈춘 자세와 목표의 차이로 따로 본다.
        tail = ts >= ts[-1] - max(s.reach_hold, 0.3)
        end_pose = np.median(q[tail], axis=0)
        i_s, i_e, _ = motion_window(ts, q, end_pose, s.reach_tol, s.move_thr)
        reached = bool(np.abs(end_pose - qr[-1]).max() < s.reach_tol)
        t_s, t_e = ts[i_s], ts[i_e]
        T_meas = t_e - t_s
        qw, tw = q[i_s:i_e + 1], (ts[i_s:i_e + 1] - t_s) / T_meas
        qm_n = np.column_stack([np.interp(tau, tw, qw[:, j]) for j in range(6)])
        err = qm_n - qr_n

        matching = extrema_dtw(qw, qr, tw * T_meas, tr,
                               prominence=s.extrema_prominence, max_shift=s.extrema_max_shift)
        idx_m = matching['indices']
        pair_ref, s_m, t_match, dev = refine_pairs(
            qw, qr, idx_m, cum, seg, tr, anchors=matching['anchors'])
        e_pair = qw - pair_ref
        t_meas_s = tw * T_meas
        cov = coverage(qr, qw, idx_m)
        lag = np.interp(s_ref, s_m, tw) - tau_ref
        late, early = runs(lag > s.th_lag), runs(lag < -s.th_lag)

        # ---- 실제 추종 속도 (추종 구간) ----
        t_rel = ts[i_s:i_e + 1] - t_s
        vel = velocity(t_rel, qw)
        spd = np.linalg.norm(vel, axis=1)

        sm = smoothness(t_rel, s_m * cum[-1], s_m, e_pair, T_meas, s.th_ripple, s.th_wobble)
        rough = runs(sm["score"] > 1)

        # 종료 각 오차: 추종 구간의 마지막 안정 구간(마지막 reach_hold~0.3초 이상)의 중앙값 자세 vs 원본 마지막 각.
        # 모션 종료 시점 판정(reach_tol)과 무관하게 따로 잰다.
        end_err = float(np.abs(end_pose - qr[-1]).max())
        # 시작 각 오차: 추종 직전 단계(시작 자세 이동)의 마지막 안정 구간. 추종 구간 첫 샘플은 이미 움직이기 시작했을 수 있다.
        pre = np.array([p_ == f"{m}_move_to_start" for p_ in phase])
        if pre.any():
            tp, qp = t[pre], meas[pre]
            start_pose = np.median(qp[tp >= tp[-1] - max(s.reach_hold, 0.3)], axis=0)
        else:
            start_pose = qw[0]
        start_err = float(np.abs(start_pose - qr[0]).max())
        rmse_before = float(np.sqrt((err**2).mean()))
        rmse_after = float(np.sqrt((e_pair**2).mean()))

        # ---- 추종 시간 (기록 시각 기준, 초) ----
        t_cmd = marks.get(m)
        t_done = marks.get(m + "_done")
        t_build = (t_cmd - marks[m + "_build"]) if (m + "_build") in marks and t_cmd is not None else None
        t_latency = max(t_s - t_cmd, 0.0) if t_cmd is not None else None  # 시작 판정이 샘플 하나 앞서므로 0 미만은 0 으로
        t_total = ((t_done if t_done is not None else t_e) - t_cmd) if t_cmd is not None else None

        M = {
            "t_track": float(T_meas),                       # 추종 시간: 모션 시작 -> 종료
            "t_ref": float(T_ref),                          # 원본 궤적 시간
            "track_ratio": float(T_meas / T_ref),
            "t_latency": None if t_latency is None else float(t_latency),   # 지령 -> 모션 시작
            "t_total": None if t_total is None else float(t_total),         # 지령 -> 완료
            "t_build": None if t_build is None else float(t_build),         # (jb2) 포인트 리스트 구성 시간
            "v_max_joint": float(np.abs(vel).max()),        # 추종 중 관절 최대 속도 [deg/s]
            "v_max_norm": float(spd.max()), "v_mean_norm": float(spd.mean()),  # 관절공간 속력 최대/평균 [deg/s]
            "rough_ratio": float(np.mean(sm["score"] > 1)), "rough_score_max": float(sm["score"].max()),
            "smooth_ripple": sm["ripple"], "smooth_wobble": sm["wobble"], "smooth_stall": sm["stall"],
            "path_dev_mean": float(dev.mean()), "path_dev_rms": float(np.sqrt((dev**2).mean())),
            "path_dev_p95": float(np.percentile(dev, 95)), "path_dev_max": float(dev.max()),
            "pair_rmse_after": rmse_after, "time_rmse_before": rmse_before,
            "cover_mean": float(cov.mean()), "cover_p95": float(np.percentile(cov, 95)), "cover_max": float(cov.max()),
            "progress_max": float(s_m[dev <= s.th_path_max].max()) if (dev <= s.th_path_max).any() else 0.0,
            "max_step": float(np.linalg.norm(np.diff(qw, axis=0), axis=1).max()),
            "start_err": start_err, "end_err": end_err,
            "lag_max": float(lag.max()), "lag_min": float(lag.min()), "lag_mean": float(lag.mean()),
            "lag_abs_max": float(np.abs(lag).max()), "lag_max_s": float(lag.max() * T_meas),
            "late_ratio": float((lag > s.th_lag).mean()), "early_ratio": float((lag < -s.th_lag).mean()),
            "reached": reached,
        }
        matched_extrema = [e for e in matching['extrema'] if e['error_deg'] is not None]
        M.update({
            'extrema_abs_max': max((abs(e['error_deg']) for e in matched_extrema), default=None),
            'extrema_matched': len(matched_extrema),
            'extrema_missing': sum(e['status'] == 'reference_missing' for e in matching['extrema']),
            'extrema_order_conflicts': sum(e['status'] == 'order_conflict' for e in matching['extrema']),
            'extrema_anchor_pairs': len(matching['anchors'])-2,
        })
        verdict = {
            f"궤적 추종 오차 p95 <= {s.th_path_p95:g}deg": M["path_dev_p95"] <= s.th_path_p95,
            f"궤적 추종 오차 max <= {s.th_path_max:g}deg": M["path_dev_max"] <= s.th_path_max,
            f"커버리지 max <= {s.th_cover:g}deg": M["cover_max"] <= s.th_cover,
            f"진행도 >= {s.th_progress:.0%}": M["progress_max"] >= s.th_progress,
            f"샘플간 이동 <= {s.th_step:g}deg": M["max_step"] <= s.th_step,
            f"속도 리플 <= {s.th_ripple:.0%}": M["smooth_ripple"] <= s.th_ripple,
            f"횡방향 흔들림 <= {s.th_wobble:g}deg": M["smooth_wobble"] <= s.th_wobble,
            f"정체 비율 <= {s.th_stall:.0%}": M["smooth_stall"] <= s.th_stall,
            f"시작/종료 각 오차 <= {s.th_end:g}deg": max(start_err, end_err) <= s.th_end,
        }

        # ---- 링크 표면 오차 (URDF 지정 시) ----
        surf = surface_errors(s, qw, pair_ref, t_rel, log)
        M.update(surf["summary"] if surf else {k: None for k in SURFACE_KEYS})

        mp = meta["methods"][m]
        desc = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in mp.items())
        desc += f" | step=1/{meta['step']} ({meta['n_points']} pts, dt={meta['dt']*1000:.0f}ms)"
        fm = lambda v: "-" if v is None else f"{v:.1f}s"  # noqa: E731
        timing = (f"reference {T_ref:.1f}s / tracking {T_meas:.1f}s (x{T_meas / T_ref:.1f}) | "
                  f"cmd->start {fm(t_latency)}, cmd->done {fm(t_total)}")
        head = f"extrema-anchored DTW | {desc}\n{timing} | {stem}"

        log(f"\n=== {m} === {desc}" + ("" if reached else "  [!] 측정이 최종 목표에 도달하지 않음"))
        log(f"[극값 매칭] 기준점 {M['extrema_anchor_pairs']}쌍, 극값 대응 {M['extrema_matched']}개, "
            f"누락 {M['extrema_missing']}개, 공통 시간순서 충돌 {M['extrema_order_conflicts']}개")
        log(f"[추종 시간] 지령→모션 시작 {fm(t_latency)} / 추종(시작→종료) {T_meas:.1f}s (원본 {T_ref:.1f}s, x{T_meas / T_ref:.2f}) "
            f"/ 지령→완료 {fm(t_total)}" + (f" / 리스트 구성 {fm(t_build)}" if t_build is not None else ""))
        log(f"[추종 속도] 관절 최대 {M['v_max_joint']:.1f} deg/s, 관절공간 속력 최대 {M['v_max_norm']:.1f} / 평균 {M['v_mean_norm']:.1f} deg/s"
            + (f" (지령 속도 {mp['speed']:g} deg/s)" if "speed" in mp else ""))
        log(f"[매끄러움] 속도 리플 {sm['ripple']:.1%} / 횡방향 흔들림 {sm['wobble']:.3f} deg / 정체 비율 {sm['stall']:.1%}"
            f"  (고주파 기준 창 {sm['window']:.2f}s, 경로 자체의 비매끄러움은 반영 안 함)")
        log(f"[거친 구간] 기준 초과 시간 {M['rough_ratio']:.0%}, 최대 점수 {M['rough_score_max']:.2f} (1 = 기준)")
        for a_, b_ in rough[:6]:
            log(f"   거친 구간: 경로 진행도 {s_m[a_]:.2f}~{s_m[b_]:.2f}, 추종 {t_rel[a_]:.1f}~{t_rel[b_]:.1f}s, 최대 점수 {sm['score'][a_:b_+1].max():.2f}")
        log(f"[매칭 전] 같은 정규화 시간끼리 짝: 각도 오차 RMSE {rmse_before:.2f} deg, 최대 {np.abs(err).max():.2f}")
        log(f"[매칭 후] 궤적 추종 오차(DTW 매칭쌍): 각도 오차 RMSE {rmse_after:.2f} deg, 관절별 {np.round(np.sqrt((e_pair**2).mean(0)), 2)}, 최대 {np.abs(e_pair).max():.2f}")
        log(f"[궤적 추종 오차 L2] mean {M['path_dev_mean']:.2f} / p95 {M['path_dev_p95']:.2f} / max {M['path_dev_max']:.2f} deg")
        log(f"[커버리지]   mean {M['cover_mean']:.2f} / p95 {M['cover_p95']:.2f} / max {M['cover_max']:.2f} deg, 진행도 {M['progress_max']:.1%}, 샘플간 최대 이동 {M['max_step']:.2f} deg")
        log(f"[도달 정확도] 시작 {start_err:.2f} deg, 종료 {end_err:.2f} deg")
        log(f"[타이밍 참고] 지연 max {M['lag_max']:+.3f} ({M['lag_max_s']:.1f}s) / min {M['lag_min']:+.3f} (+ 늦음 / - 빠름, 정규화 시간 비율)")
        for a_, b_ in late[:6]:
            log(f"   늦은 구간(>{s.th_lag:.0%}): 원본 {tr[a_]-tr[0]:.1f}~{tr[b_]-tr[0]:.1f}s 지점, 최대 {lag[a_:b_+1].max():+.3f}")
        for a_, b_ in early[:6]:
            log(f"   빠른 구간(>{s.th_lag:.0%}): 원본 {tr[a_]-tr[0]:.1f}~{tr[b_]-tr[0]:.1f}s 지점, 최소 {lag[a_:b_+1].min():+.3f}")
        if surf:
            S_ = surf["summary"]
            log(f"[링크 표면 오차] 최악 점 변위 최대 {S_['surf_disp_max']:.1f} mm / p95 {S_['surf_disp_p95']:.1f} / 평균 {S_['surf_disp_mean']:.1f} mm"
                f" | Chamfer 최대 {S_['surf_chamfer_max']:.2f} / 평균 {S_['surf_chamfer_mean']:.2f} mm | Hausdorff 최대 {S_['surf_hausdorff_max']:.1f} mm")
            log("   링크별 최대 변위 [mm]: " + ", ".join(f"{k} {v:.1f}" for k, v in surf["links"].items()) + f"  (최악: {S_['surf_worst_link']})")
        log("[판정] " + " | ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in verdict.items()))

        base = os.path.join(run_dir, f"compare_{stem}_{m}")
        with open(base + ".json", "w", encoding="utf-8") as fj:
            json.dump({"source": os.path.basename(path), "mode": m, "params": mp, "step": meta["step"],
                       "metrics": M, "verdict": verdict, "all_pass": all(verdict.values()),
                       "matching": {k: matching[k] for k in ('method','anchors','extrema','prominence_deg','max_shift')},
                       "surface_links_max_mm": surf["links"] if surf else None,
                       "surface_urdf": os.path.abspath(os.path.expanduser(s.urdf_path)) if surf else None}, fj, ensure_ascii=False, indent=1)
        files.append(base + ".json")
        extrema_fields = ('joint','kind','ref_index','meas_index','ref_time_s','meas_time_s',
                          'ref_deg','meas_deg','error_deg','deficit_deg','status')
        with open(base + '_extrema.csv', 'w', newline='', encoding='utf-8') as ef:
            writer = csv.DictWriter(ef, fieldnames=extrema_fields)
            writer.writeheader()
            writer.writerows(matching['extrema'])
        files.append(base + '_extrema.csv')
        with open(base + '_dtw.csv', 'w', newline='', encoding='utf-8') as df:
            writer = csv.writer(df)
            writer.writerow(['meas_index','ref_index'])
            writer.writerows(matching['dtw_pairs'])
        files.append(base + '_dtw.csv')

        summ = os.path.join(summary_dir or os.path.dirname(os.path.abspath(run_dir)), "summary.csv")
        keys = ["run", "mode", "step", "params"] + list(M.keys()) + ["all_pass"]
        row = [os.path.basename(os.path.abspath(run_dir)), m, meta["step"], json.dumps(mp, ensure_ascii=False)] \
            + [("" if v is None else f"{v:.4f}" if isinstance(v, float) else v) for v in M.values()] + [all(verdict.values())]
        old = list(csv.reader(open(summ, encoding="utf-8"))) if os.path.exists(summ) else []
        if old and old[0] != keys:
            os.replace(summ, summ.replace(".csv", f"_old_{int(time.time())}.csv"))
            old = []
        with open(summ, "a", newline="", encoding="utf-8") as fs:
            w = csv.writer(fs)
            if not old:
                w.writerow(keys)
            w.writerow(row)

        if surf:
            with open(os.path.join(run_dir, f"surface_{stem}_{m}.csv"), "w", newline="", encoding="utf-8") as fsf:
                w = csv.writer(fsf)
                w.writerow(["t_track", "disp_max_mm", "disp_p95_mm", "disp_mean_mm", "chamfer_mm", "hausdorff_mm"])
                for k_, i_ in enumerate(surf["idx"]):
                    sr = surf["series"]
                    w.writerow([f"{t_rel[i_]:.4f}", *(f"{sr[c][k_]:.3f}" for c in ("disp_max", "disp_p95", "disp_mean", "chamfer", "hausdorff"))])
            files.append(os.path.join(run_dir, f"surface_{stem}_{m}.csv"))
        with open(base + "_pairs.csv", "w", newline="", encoding="utf-8") as fp:
            fp.write("# " + json.dumps({"source": os.path.basename(path), **mp, "step": meta["step"], "T_ref": T_ref,
                                      "T_track": T_meas, "reached": reached, "matching": matching["method"]}, ensure_ascii=False) + "\n")
            w = csv.writer(fp)
            w.writerow(["t_track", "t_ref_matched", "ref_idx", "progress"] + [f"meas_q{i}" for i in range(6)]
                       + [f"ref_q{i}" for i in range(6)] + [f"err_q{i}" for i in range(6)] + ["pair_dist"])
            for k in range(len(qw)):
                w.writerow([f"{t_meas_s[k]:.4f}", f"{t_match[k]:.4f}", int(idx_m[k]), f"{s_m[k]:.5f}", *map("{:.4f}".format, qw[k]),
                            *map("{:.4f}".format, pair_ref[k]), *map("{:.4f}".format, e_pair[k]), f"{dev[k]:.4f}"])
        files.append(base + "_pairs.csv")

        with open(os.path.join(run_dir, f"velocity_{stem}_{m}.csv"), "w", newline="", encoding="utf-8") as fv:
            w = csv.writer(fv)
            w.writerow(["t_track"] + [f"v{i}" for i in range(6)] + ["speed_norm"])
            for k in range(len(t_rel)):
                w.writerow([f"{t_rel[k]:.4f}", *map("{:.3f}".format, vel[k]), f"{spd[k]:.3f}"])
        files.append(os.path.join(run_dir, f"velocity_{stem}_{m}.csv"))
        f = fig_smooth_map(head, qw, qr, t_rel, s_m, sm, s.th_ripple, s.th_wobble)
        f.savefig(os.path.join(run_dir, f"smooth_map_{stem}_{m}.png"), dpi=120)
        files.append(os.path.join(run_dir, f"smooth_map_{stem}_{m}.png"))
        with open(os.path.join(run_dir, f"smooth_{stem}_{m}.csv"), "w", newline="", encoding="utf-8") as fs2:
            w = csv.writer(fs2)
            w.writerow(["t_track", "progress", "speed_along_path", "trend", "local_ripple", "local_wobble_deg", "score"])
            for k in range(len(t_rel)):
                w.writerow([f"{t_rel[k]:.4f}", f"{s_m[k]:.5f}", f"{sm['v_s'][k]:.3f}", f"{sm['trend'][k]:.3f}",
                            f"{sm['rip_loc'][k]:.4f}", f"{sm['wob_loc'][k]:.4f}", f"{sm['score'][k]:.3f}"])
        files.append(os.path.join(run_dir, f"smooth_{stem}_{m}.csv"))
        f = fig_after(head, tr, qr, t_match, qw, e_pair, dev, t_meas_s, T_ref, T_meas, rmse_after, M, end_err, matching=matching)
        f.savefig(base + "_after.png", dpi=120)
        files.append(base + "_after.png")
        results[m] = {"metrics": M, "verdict": verdict, "all_pass": all(verdict.values()), "files": files}
    return results


def redraw_saved_graph(run_dir, filename, fig):
    """Rebuild a saved plot from its exact CSV companions, without re-analysis."""
    if filename.startswith('sim_') and filename.endswith('.png'):
        stem = filename[4:-4]
        t, phase, meas, state, meta = load_follow(os.path.join(run_dir, stem + '.csv'))
        fig_sim(stem, t, phase, meas, state, meta, fig=fig, ideal=ideal_profile(run_dir, meta.get('methods', {}).get('servo')))
        return True
    if filename.startswith('compare_') and filename.endswith('_after.png'):
        base = filename[:-len('_after.png')]
    elif filename.startswith('smooth_map_') and filename.endswith('.png'):
        base = 'compare_' + filename[len('smooth_map_'):-4]
    else:
        return False
    with open(os.path.join(run_dir, base + '.json'), encoding='utf-8') as f:
        result = json.load(f)
    ref = np.loadtxt(os.path.join(run_dir, 'traj_ref.csv'), delimiter=',', skiprows=1, ndmin=2)
    pairs = np.loadtxt(os.path.join(run_dir, base + '_pairs.csv'), delimiter=',', skiprows=2, ndmin=2)
    tr, qr = ref[:, 0], np.rad2deg(ref[:, 1:7])
    M = result['metrics']
    params = ', '.join(f'{k}={v}' for k, v in result.get('params', {}).items() if v is not None)
    head = f'{params} | {base}'
    if filename.startswith('compare_'):
        fig_after(head, tr, qr, pairs[:, 1], pairs[:, 4:10], pairs[:, 16:22],
                  pairs[:, 22], pairs[:, 0], M['t_ref'], M['t_track'],
                  M['pair_rmse_after'], M, M['end_err'], fig=fig, matching=result.get('matching'))
    else:
        smooth = np.loadtxt(os.path.join(run_dir, 'smooth_' + base[len('compare_'):] + '.csv'),
                            delimiter=',', skiprows=1, ndmin=2)
        if len(smooth) != len(pairs):
            raise ValueError('Smoothness and matched-pair sample counts differ')
        sm = {'score': smooth[:, 6], 'rip_loc': smooth[:, 4], 'wob_loc': smooth[:, 5]}
        # Recover the thresholds used during analysis from saved verdict labels.
        verdict = result.get('verdict', {})
        ripple, wobble = .15, .30
        for label in verdict:
            if label.startswith('속도 리플 <= '):
                ripple = float(label.split('<= ')[1].rstrip('%')) / 100
            if label.startswith('횡방향 흔들림 <= '):
                wobble = float(label.split('<= ')[1].rstrip('deg'))
        fig_smooth_map(head, pairs[:, 4:10], qr, smooth[:, 0], smooth[:, 1],
                       sm, ripple, wobble, fig=fig)
    return True
