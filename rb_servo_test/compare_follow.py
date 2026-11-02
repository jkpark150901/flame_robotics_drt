"""servo_follow.py 기록(follow_<mode>.csv)을 원본 궤적과 같은 시간 간격으로 재샘플링해 비교.

- 모션 시작(시작 자세를 벗어나는 시점)과 종료(최종 목표에 수렴한 시점)를 맞춰 시간축을 0~1 로 일반화하고,
  원본/측정을 같은 정규화 시간에 재샘플링해 각도만 비교한다. 속도는 비교하지 않는다(실행 시간이 다르므로 무의미).
- 시간 무관 경로 편차(path deviation)와 원본/측정 소요시간도 함께 출력한다.

사용 예:
    python rb_servo_test/compare_follow.py                       # out/ 의 가장 최근 follow_*.csv
    python rb_servo_test/compare_follow.py --file out/follow_servo_20260101_120000.csv

결과 파일(compare_<기록파일명>_<mode>.csv/png)에는 method 와 매개변수가 제목/헤더로 들어간다.
"""
import argparse
import csv
import glob
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================ 추종 성능 판정 기준 (초기값, 필요 시 조정) ============================
# 경로 정확도 (시간 무관): 측정점이 원본 경로(폴리라인)에서 벗어난 거리, 6관절 L2 [deg]
TH_PATH_P95 = 1.0     # 95% 지점이 이 값 이하
TH_PATH_MAX = 2.0     # 최대값이 이 값 이하
# 경로 커버리지: 원본의 모든 점이 측정 경로에서 이 거리 이내에 있어야 함(구간을 건너뛰지 않았는지) [deg]
TH_COVER = 2.0
TH_PROGRESS = 0.99    # 경로 위에서 실제로 도달한 최대 진행도(0~1, 경로 편차가 TH_PATH_MAX 이내인 샘플만 인정)
TH_STEP = 5.0         # 연속한 측정 샘플 사이 최대 이동량 [deg] (순간이동/건너뜀 검출)
# 도달 정확도: 시작/종료 자세 오차 (관절별 최대) [deg]
TH_END = 0.5
# 타이밍: 같은 경로 진행도에서 측정이 원본보다 늦거나 빠른 정도 |Δτ| (정규화 시간 비율, 0.03 = 전체 시간의 3%)
TH_LAG = 0.03
# ===================================================================================================


def load_follow(path):
    """첫 줄 '# {json}' 메타를 읽고 나머지는 CSV 로 파싱. 측정값만 있는 새 형식과 jref/jang 이전 형식 모두 지원."""
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
    else:  # 이전 형식: jref(3:9), jang(9:15). 시뮬레이션에서는 jnt_ang 가 0 고정이라 jnt_ref 사용
        meas = np.array([[float(v) for v in x[9:15]] for x in rows])
        if np.abs(meas).max() == 0.0:
            print("note: jnt_ang 가 전부 0 -> jnt_ref 를 측정각으로 사용")
            meas = np.array([[float(v) for v in x[3:9]] for x in rows])
    return t, phase, meas, state, meta


def path_deviation(meas, ref):
    """측정 점마다 기준 폴리라인까지의 최소 거리(deg, 6관절 L2)."""
    a, b = ref[:-1], ref[1:]
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-12)
    out = []
    for p in meas:
        u = np.clip(((p - a) * ab).sum(1) / denom, 0, 1)
        out.append(np.linalg.norm(a + u[:, None] * ab - p, axis=1).min())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="follow_*.csv (기본: outdir 의 가장 최근 파일)")
    ap.add_argument("--ref", default=os.path.join(HERE, "out/traj_interp_50hz.csv"))
    ap.add_argument("--outdir", help="결과 저장 폴더 (기본: --file 이 있는 폴더, 없으면 out/)")
    ap.add_argument("--summary-dir", default=os.path.join(HERE, "out"), help="실행 간 비교용 summary.csv 를 쌓는 폴더")
    ap.add_argument("--tol", type=float, default=0.5, help="모션 종료(목표 도달) 판정 허용 오차 [deg]")
    ap.add_argument("--movethr", type=float, default=0.2, help="모션 시작 판정 변화량 [deg]")
    args = ap.parse_args()
    outdir = args.outdir or (os.path.dirname(os.path.abspath(args.file)) if args.file else os.path.join(HERE, "out"))
    compare(args.file, args.ref, outdir, args.tol, args.movethr, summary_dir=args.summary_dir)


def plot_sim(stem, t, phase, jang, state, meta, outdir):
    """시뮬레이션(기록) 그래프: 원본 궤적과 무관하게 측정된 관절각/각속도/robot_state 전체 타임라인."""
    k = 5
    ker = np.ones(k) / k
    v = np.gradient(jang, t, axis=0)
    v = np.column_stack([np.convolve(v[:, j], ker, mode="same") for j in range(6)])
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1]})
    c = plt.cm.tab10(np.arange(6))
    for j in range(6):
        ax[0].plot(t, jang[:, j], color=c[j], lw=1.2, label=f"j{j}")
        ax[1].plot(t, v[:, j], color=c[j], lw=0.9)
    ax[2].step(t, state, where="post", color="k", lw=1.0)
    ax[2].set_yticks([1, 3])
    ax[2].set_yticklabels(["idle", "run"])
    ax[2].set_ylabel("robot_state")
    # 단계(phase) 구간 표시
    edges = [0] + [i for i in range(1, len(phase)) if phase[i] != phase[i - 1]] + [len(phase)]
    shade = plt.cm.Pastel1(np.arange(9))
    for n, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        for a_ in ax:
            a_.axvspan(t[a], t[min(b, len(t) - 1)], color=shade[n % 9], alpha=0.35, lw=0)
        ax[0].text(t[a], 1.0, phase[a], transform=ax[0].get_xaxis_transform(), rotation=35,
                   fontsize=7, va="bottom", ha="left")
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
    png = os.path.join(outdir, f"sim_{stem}.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print("saved", png)


def motion_window(t, q, final, tol, movethr):
    """측정 구간 q 에서 모션 시작/종료 인덱스. 시작: 시작 자세에서 movethr 이상 벗어나기 직전,
    종료: 이후 끝까지 final 의 tol 이내로 유지되기 시작한 첫 점."""
    moved = np.abs(q - q[0]).max(1) > movethr
    i_s = max(int(np.argmax(moved)) - 1, 0) if moved.any() else 0
    ok = np.abs(q - final).max(1) < tol
    suffix = np.flip(np.logical_and.accumulate(np.flip(ok)))
    reached = bool(suffix.any())
    i_e = int(np.argmax(suffix)) if reached else len(t) - 1
    return i_s, max(i_e, i_s + 1), reached


def ref_progress(qr):
    """원본 경로의 호길이 기반 진행도 s(0~1). 관절공간 L2 거리 누적."""
    seg = np.linalg.norm(np.diff(qr, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    return cum / cum[-1], cum, seg


def ordered_match(meas, qr, kmax=None):
    """측정 샘플 -> 원본 정점 인덱스의 '순서를 지키는' 매칭 (비감소 정렬, DTW 계열 동적계획법).

    측정이 시간 순서대로 진행하면 짝지은 원본 위치도 같은 순서로 진행해야 하므로, 다른 시각에 우연히 같은 각도를
    지나가더라도 거기에 매칭되지 않는다. 시간 지연/속도 차이는 허용한다(한 샘플에 원본 인덱스가 0~kmax 만큼 전진 가능).
    전체 경로의 거리 합이 최소가 되도록 한 번에 정하므로 지역 최근접 오류가 없다.
    시작/종료는 각각 원본 시작/끝 근처로 제한한다.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.spatial.distance import cdist

    n, m = len(meas), len(qr)
    kmax = max(30, m // 5) if kmax is None else kmax
    C = cdist(meas, qr).astype(np.float32)  # (n, m)
    D = np.full((n, m), np.inf, dtype=np.float32)
    head = max(5, m // 20)
    D[0, :head] = C[0, :head]  # 시작은 원본 시작 근처
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


def refine_pairs(meas, qr, idx, cum, seg, tr):
    """DTW 가 정한 원본 정점 idx[i] 양옆 선분 위에서 측정점의 연속 위치(짝)를 구한다. 정점 단위 양자화 오차를 없앤다.
    반환: 짝지은 원본 점 proj, 진행도 s, 원본 시각 t(0 기준), 거리 dist."""
    a, ab = qr[:-1], np.diff(qr, axis=0)
    denom = np.maximum((ab * ab).sum(1), 1e-12)
    n = len(meas)
    proj, s, t, dist = np.zeros_like(meas), np.zeros(n), np.zeros(n), np.zeros(n)
    for i, p in enumerate(meas):
        cand = np.unique(np.clip([idx[i] - 1, idx[i]], 0, len(a) - 1))
        u = np.clip(((p - a[cand]) * ab[cand]).sum(1) / denom[cand], 0, 1)
        d = np.linalg.norm(a[cand] + u[:, None] * ab[cand] - p, axis=1)
        k = int(np.argmin(d))
        c, uu = cand[k], u[k]
        proj[i] = a[c] + uu * ab[c]
        s[i] = (cum[c] + uu * seg[c]) / cum[-1]
        t[i] = tr[c] - tr[0] + uu * (tr[c + 1] - tr[c])
        dist[i] = d[k]
    return proj, np.maximum.accumulate(s), np.maximum.accumulate(t), dist


def coverage(qr, meas, idx=None):
    """원본 각 점에서 측정 경로(폴리라인)까지의 거리. idx(순서 매칭)가 있으면 그 원본 점에 매칭된 측정 구간 근처에서만 재서
    다른 시각에 지나간 우연한 일치를 인정하지 않는다. 건너뛴 구간이 있으면 커진다."""
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


def runs(mask):
    """True 가 연속된 구간의 (시작 인덱스, 끝 인덱스) 목록."""
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


def compare(file, ref_path, outdir, tol=0.5, movethr=0.2, npts=500, summary_dir=None):
    """기록 파일 하나를 원본과 비교해 추종 성능을 평가한다.

    평가 기준 (시간이 어긋나도 공정하게 보도록 분리):
      1) 경로 정확도: 시간 무관 경로 편차(mean/rms/p95/max) -> TH_PATH_P95, TH_PATH_MAX
      2) 도달 정확도: 시작/종료 자세 오차 -> TH_END
      3) 타이밍: 같은 경로 진행도에서의 지연 Δτ(정규화 시간) -> TH_LAG, 늦은 구간 목록
      (참고) 정규화 시간에서 각도 오차 RMSE
    file=None 이면 outdir 의 최근 follow_*.csv. 결과: compare_*.csv/png/json, summary_dir/summary.csv(기본 outdir)에 한 줄 추가.
    """
    ref = np.genfromtxt(ref_path, delimiter=",", skip_header=1)
    tr, qr = ref[:, 0], np.rad2deg(ref[:, 1:])
    T_ref = tr[-1] - tr[0]
    tau_ref = (tr - tr[0]) / T_ref
    s_ref, cum, seg = ref_progress(qr)

    path = file or max(glob.glob(os.path.join(outdir, "follow_*.csv")), key=os.path.getmtime)
    stem = os.path.splitext(os.path.basename(path))[0]
    t, phase, meas, state, meta = load_follow(path)
    print("file:", path)
    plot_sim(stem, t, phase, meas, state, meta, outdir)

    tau = np.linspace(0.0, 1.0, npts)
    qr_n = np.column_stack([np.interp(tau, tau_ref, qr[:, j]) for j in range(6)])
    names = [f"j{i}" for i in range(6)]

    for m in [k for k in ("jb2", "servo") if k in meta.get("marks", {})]:
        sel = np.array([p.startswith(m) and not p.endswith("move_to_start") for p in phase])
        ts, q = t[sel], meas[sel]
        i_s, i_e, reached = motion_window(ts, q, qr[-1], tol, movethr)
        t_s, t_e = ts[i_s], ts[i_e]
        T_meas = t_e - t_s
        qw, tw = q[i_s:i_e + 1], (ts[i_s:i_e + 1] - t_s) / T_meas
        qm_n = np.column_stack([np.interp(tau, tw, qw[:, j]) for j in range(6)])
        err = qm_n - qr_n

        # ===== DTW 매칭 (순서 보존): 측정 샘플 i <-> 원본 정점 idx_m[i] =====
        idx_m = ordered_match(qw, qr)
        # DTW 쌍을 원본 정점 양옆 선분 위의 연속 위치로 다듬음 (정점 단위 양자화 오차 제거)
        pair_ref, s_m, t_match, dev = refine_pairs(qw, qr, idx_m, cum, seg, tr)
        e_pair = qw - pair_ref                     # DTW 매칭쌍 기준 오차(관절별) = 측정 - 짝지은 원본 점
        # t_match: 매칭된 원본 시각 (6관절 공통 하나의 시간 변환)
        t_meas_s = tw * T_meas
        cov = coverage(qr, qw, idx_m)              # 원본 점 -> 측정 경로 (건너뜀 검출)
        # 타이밍(참고): 원본의 각 진행도 s_ref 에서 측정이 도달한 정규화 시간 - 원본 정규화 시간 (+ 늦음 / - 빠름)
        lag = np.interp(s_ref, s_m, tw) - tau_ref
        late = runs(lag > TH_LAG)
        early = runs(lag < -TH_LAG)

        end_err = float(np.abs(qm_n[-1] - qr[-1]).max())
        start_err = float(np.abs(qm_n[0] - qr[0]).max())
        rmse_before = float(np.sqrt((err**2).mean()))
        rmse_after = float(np.sqrt((e_pair**2).mean()))
        M = {
            # DTW 매칭쌍 오차 (측정 샘플 -> 짝지은 원본 점, 6관절 L2)
            "path_dev_mean": float(dev.mean()), "path_dev_rms": float(np.sqrt((dev**2).mean())),
            "path_dev_p95": float(np.percentile(dev, 95)), "path_dev_max": float(dev.max()),
            "pair_rmse_after": rmse_after, "time_rmse_before": rmse_before,
            # 커버리지: 원본 점 -> 측정 경로
            "cover_mean": float(cov.mean()), "cover_p95": float(np.percentile(cov, 95)), "cover_max": float(cov.max()),
            "progress_max": float(s_m[dev <= TH_PATH_MAX].max()) if (dev <= TH_PATH_MAX).any() else 0.0,
            "max_step": float(np.linalg.norm(np.diff(qw, axis=0), axis=1).max()),
            "start_err": start_err, "end_err": end_err,
            "lag_max": float(lag.max()), "lag_min": float(lag.min()), "lag_mean": float(lag.mean()),
            "lag_abs_max": float(np.abs(lag).max()), "lag_max_s": float(lag.max() * T_meas),
            "late_ratio": float((lag > TH_LAG).mean()), "early_ratio": float((lag < -TH_LAG).mean()),
            "T_ref": float(T_ref), "T_meas": float(T_meas), "reached": reached,
        }
        verdict = {
            f"경로 p95 <= {TH_PATH_P95:g}deg": M["path_dev_p95"] <= TH_PATH_P95,
            f"경로 max <= {TH_PATH_MAX:g}deg": M["path_dev_max"] <= TH_PATH_MAX,
            f"커버리지 max <= {TH_COVER:g}deg": M["cover_max"] <= TH_COVER,
            f"진행도 >= {TH_PROGRESS:.0%}": M["progress_max"] >= TH_PROGRESS,
            f"샘플간 이동 <= {TH_STEP:g}deg": M["max_step"] <= TH_STEP,
            f"시작/종료 오차 <= {TH_END:g}deg": max(start_err, end_err) <= TH_END and reached,
        }

        mp = meta["methods"][m]
        desc = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in mp.items())
        desc += f" | step=1/{meta['step']} ({meta['n_points']} pts, dt={meta['dt']*1000:.0f}ms)"
        timing = f"duration ref {T_ref:.1f}s / measured {T_meas:.1f}s (x{T_meas / T_ref:.1f})"

        print(f"\n=== {m} === {desc}\n     {timing}" + ("" if reached else "  [!] 측정이 최종 목표에 도달하지 않음"))
        print(f"[매칭 전] 같은 정규화 시간끼리 짝: 각도 오차 RMSE {rmse_before:.2f} deg, 관절별 {np.round(np.sqrt((err**2).mean(0)), 2)}, max {np.abs(err).max():.2f}")
        print(f"[매칭 후] DTW 매칭쌍 오차:        각도 오차 RMSE {rmse_after:.2f} deg, 관절별 {np.round(np.sqrt((e_pair**2).mean(0)), 2)}, max {np.abs(e_pair).max():.2f}")
        print(f"[DTW 쌍 L2] mean {M['path_dev_mean']:.2f} / rms {M['path_dev_rms']:.2f} / p95 {M['path_dev_p95']:.2f} / max {M['path_dev_max']:.2f} deg")
        print(f"[커버리지]   원본점->측정 경로 거리 mean {M['cover_mean']:.2f} / p95 {M['cover_p95']:.2f} / max {M['cover_max']:.2f} deg, 도달 진행도 {M['progress_max']:.1%}, 샘플간 최대 이동 {M['max_step']:.2f} deg")
        print(f"[도달 정확도] 시작 {start_err:.2f} deg, 종료 {end_err:.2f} deg")
        print(f"[타이밍 참고] 지연 max {M['lag_max']:+.3f} (= {M['lag_max_s']:.1f}s) / min {M['lag_min']:+.3f} (정규화 시간 비율, + 늦음 / - 빠름)")
        for a_, b_ in late[:6]:
            print(f"   늦은 구간(>{TH_LAG:.0%}): 원본 {tr[a_]-tr[0]:.1f}~{tr[b_]-tr[0]:.1f}s 지점, 최대 {lag[a_:b_+1].max():+.3f}")
        for a_, b_ in early[:6]:
            print(f"   빠른 구간(>{TH_LAG:.0%}): 원본 {tr[a_]-tr[0]:.1f}~{tr[b_]-tr[0]:.1f}s 지점, 최소 {lag[a_:b_+1].min():+.3f}")
        print("[판정]", " | ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in verdict.items()))

        base = os.path.join(outdir, f"compare_{stem}_{m}")
        with open(base + ".json", "w") as f:
            json.dump({"source": os.path.basename(path), "mode": m, "params": mp, "step": meta["step"],
                       "metrics": M, "verdict": verdict}, f, ensure_ascii=False, indent=1)
        summ = os.path.join(summary_dir or outdir, "summary.csv")
        keys = ["source", "mode", "step", "params"] + list(M.keys()) + ["all_pass"]
        row = [os.path.join(os.path.basename(os.path.dirname(os.path.abspath(path))), os.path.basename(path)), m, meta["step"], json.dumps(mp, ensure_ascii=False)] \
            + [f"{v:.4f}" if isinstance(v, float) else v for v in M.values()] + [all(verdict.values())]
        old = list(csv.reader(open(summ))) if os.path.exists(summ) else []
        if old and old[0] != keys:  # 컬럼이 바뀌었으면 이전 파일을 보존하고 새로 시작
            os.replace(summ, summ.replace(".csv", f"_old_{int(time.time())}.csv"))
            old = []
        with open(summ, "a", newline="") as f:
            w = csv.writer(f)
            if not old:
                w.writerow(keys)
            w.writerow(row)

        # DTW 매칭쌍 CSV: 측정 샘플 하나당 짝지은 원본 정점과 오차
        with open(base + "_pairs.csv", "w", newline="") as f:
            f.write("# " + json.dumps({"source": os.path.basename(path), **mp, "step": meta["step"], "T_ref": T_ref,
                                     "T_meas": T_meas, "reached": reached}, ensure_ascii=False) + "\n")
            w = csv.writer(f)
            w.writerow(["t_meas", "t_ref_matched", "ref_idx", "progress"] + [f"meas_q{i}" for i in range(6)]
                       + [f"ref_q{i}" for i in range(6)] + [f"err_q{i}" for i in range(6)] + ["pair_dist"])
            for k in range(len(qw)):
                w.writerow([f"{t_meas_s[k]:.4f}", f"{t_match[k]:.4f}", int(idx_m[k]), f"{s_m[k]:.5f}", *map("{:.4f}".format, qw[k]),
                            *map("{:.4f}".format, pair_ref[k]), *map("{:.4f}".format, e_pair[k]), f"{dev[k]:.4f}"])

        c = plt.cm.tab10(np.arange(6))
        # ---------- 매칭 전: 시작/종료만 맞춘 정규화 시간에서 같은 시각끼리 비교 ----------
        fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        for j in range(6):
            ax[0].plot(tau, qr_n[:, j], color=c[j], lw=1.3, label=names[j])
            ax[0].plot(tau, qm_n[:, j], "--", color=c[j], lw=1.3)
            ax[1].plot(tau, err[:, j], color=c[j], lw=1.0)
        ax[0].set_ylabel("angle [deg]")
        ax[0].set_title(f"BEFORE DTW (same normalized time pairs) | {desc}\n{timing} | {stem}\n"
                        f"angle error RMSE {rmse_before:.2f} deg, max {np.abs(err).max():.2f} deg\n"
                        "(solid = reference, dashed = measured, only start/end aligned)", fontsize=8, pad=30)
        ax[0].legend(ncol=6, fontsize=8, frameon=False, loc="lower right")
        ax[1].set_ylabel("angle error [deg]\n(measured - ref, same normalized time)")
        ax[1].set_xlabel("normalized time (0 = motion start, 1 = motion end)")
        sx_ref = ax[0].secondary_xaxis("top", functions=(lambda x: x * T_ref, lambda x: x / T_ref))
        sx_ref.set_xlabel(f"reference time [s]  (0 - {T_ref:.1f})", fontsize=8)
        sx_meas = ax[1].secondary_xaxis(-0.32, functions=(lambda x: x * T_meas, lambda x: x / T_meas))
        sx_meas.set_xlabel(f"measured time [s]  (0 - {T_meas:.1f})", fontsize=8)
        for sx in (sx_ref, sx_meas):
            sx.tick_params(labelsize=8)
        for a2 in ax:
            a2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(base + "_before.png", dpi=130)
        plt.close(fig)
        print("saved", base + "_before.png")

        # ---------- 매칭 후: DTW 매칭쌍 기준 (측정 시각 -> 매칭된 원본 시각, 6관절 공통) ----------
        fig, ax = plt.subplots(3, 1, figsize=(11, 12), gridspec_kw={"height_ratios": [3, 2, 2]})
        for j in range(6):
            ax[0].plot(tr - tr[0], qr[:, j], color=c[j], lw=1.4, label=names[j])
            ax[0].plot(t_match, qw[:, j], "--", color=c[j], lw=1.2)
            ax[1].plot(t_match, e_pair[:, j], color=c[j], lw=1.0)
        ax[1].plot(t_match, dev, color="k", lw=1.4, label="L2 pair error")
        ax[0].set_ylabel("angle [deg]")
        ax[0].set_title(f"AFTER DTW (ordered matching pairs) | {desc}\n{timing} | {stem}\n"
                        f"pair error RMSE {rmse_after:.2f} deg, L2 p95 {M['path_dev_p95']:.2f} / max {M['path_dev_max']:.2f} deg, "
                        f"coverage max {M['cover_max']:.2f} deg, progress {M['progress_max']:.1%}, end err {end_err:.2f} deg\n"
                        "(x = matched reference time; solid = reference, dashed = measured re-timed by DTW pairs, same for all joints)", fontsize=8)
        ax[0].legend(ncol=6, fontsize=8, frameon=False, loc="lower right")
        ax[1].set_ylabel("DTW pair error [deg]\n(measured - matched ref)")
        ax[1].legend(fontsize=8, frameon=False, loc="upper right")
        for a2 in ax[:2]:
            a2.set_xlim(0, T_ref)
            a2.set_xlabel("matched reference time [s]")
        ax[2].plot(t_meas_s, t_match, color="C3", lw=1.6, label="DTW matched ref time")
        ax[2].plot([0, T_meas], [0, T_ref], color="gray", ls=":", lw=1.2, label="uniform scaling (start/end aligned)")
        ax[2].set_ylabel("matched reference time [s]")
        ax[2].set_xlabel(f"measured time [s]  (0 - {T_meas:.1f})")
        ax[2].legend(fontsize=8, frameon=False)
        for a2 in ax:
            a2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(base + "_after.png", dpi=130)
        plt.close(fig)
        print("saved", base + "_after.png")


if __name__ == "__main__":
    main()
