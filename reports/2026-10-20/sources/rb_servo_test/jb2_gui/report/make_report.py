"""웨이포인트 분할에 따른 속도·경유점 도달 오차 보고서(report.md)와 그림·CSV 를 만든다. 실행 폴더는 읽기만 한다.

사용: cd flame_robotics_drt && venv/bin/python rb_servo_test/jb2_gui/report/make_report.py
이후: python3 rb_servo_gui/docs/report/md_to_html.py rb_servo_test/jb2_gui/report/report.md rb_servo_test/jb2_gui/report/report.html --pdf
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.figure import Figure

from collect import collect

have = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic"):
    if _f in have:
        matplotlib.rcParams["font.sans-serif"] = [_f] + list(matplotlib.rcParams["font.sans-serif"])
        break
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
BCOL = {0.0: "C0", 50.0: "C1", 100.0: "C3"}


def fmt(x, n=2):
    return f"{x:.{n}f}"


def main():
    runs = [r for r in collect() if r.get("dist")]
    excluded = [r for r in collect() if not r.get("dist")]
    jb = [r for r in runs if r["method"] == "jb2"]
    sv = sorted([r for r in runs if r["method"] == "servo"], key=lambda r: r["period_ms"])
    Ns = sorted({r["segments"] for r in jb})
    Ns4 = [N for N in Ns if N != 48]   # blend 0·50 은 N=4~24 만 실험

    def get(mode, N, blend):
        x = [r for r in jb if r["mode"] == mode and r["segments"] == N and r["blend"] == blend]
        return x[0] if x else None

    # ------------------------------------------------------------------ CSV
    path = os.path.join(HERE, "runs_summary.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run", "mode", "command", "segments", "blend", "servo_period_ms", "gain", "speed", "accel", "leg_segments", "segment_len_deg",
                    "via1_dist_deg", "via2_dist_deg", "via3_dist_deg", "end_dist_deg", "via_max_dist_deg", "cmd_points_mean_dist_deg", "cmd_points_max_dist_deg",
                    "peak_speed_deg_s", "peak_speed_L2_deg_s", "motion_time_s", "ideal_time_s"])
        for r in runs:
            d = r["dist"]
            w.writerow([r["run"], r["mode"], r["method"], r["segments"] if r["method"] == "jb2" else "", r["blend"] if r["method"] == "jb2" else "",
                        r["period_ms"] if r["method"] == "servo" else "", r["gain"] if r["method"] == "servo" else "", r["speed"], r["accel"],
                        r.get("leg_counts", ""), [round(x, 2) for x in r["seg_len"]] if "seg_len" in r else "",
                        f"{d[1]:.4f}", f"{d[2]:.4f}", f"{d[3]:.4f}", f"{d[4]:.4f}", f"{max(d[1:4]):.4f}",
                        f"{r['cmd_mean']:.4f}" if "cmd_mean" in r else "", f"{r['cmd_max']:.4f}" if "cmd_max" in r else "",
                        f"{r['linf']:.2f}", f"{r['l2']:.2f}", f"{r['T_meas']:.2f}", f"{r['T_ideal']:.2f}"])

    # ------------------------------------------------------------------ 구간 길이 대비 오차 비율 (blend 50, 100)
    pts = []  # (blend, 구간 길이, L2 거리, 비율)
    for r in jb:
        if r["blend"] == 0:
            continue
        for k in (1, 2, 3):
            ell = (r["seg_len"][k - 1] + r["seg_len"][k]) / 2
            pts.append((r["blend"], ell, r["dist"][k], r["dist"][k] / (ell * np.sqrt(6))))
    pts = np.array(pts)

    def ratio_stats(blend, lo, hi):
        m = (pts[:, 0] == blend) & (pts[:, 1] > lo) & (pts[:, 1] <= hi)
        return (pts[m, 3].mean(), pts[m, 3].min(), pts[m, 3].max(), int(m.sum())) if m.any() else None

    bins = [(0, 5), (5, 12), (12, 40)]

    # ------------------------------------------------------------------ 그림 1: 경유점 도달 오차
    fig = Figure(figsize=(12, 4.6))
    ax = fig.subplots(1, 2)
    for blend in (50.0, 100.0):
        for mode, mk, filled in (("시뮬레이션", "o", False), ("실기", "s", True)):
            xs = [N for N in Ns if get(mode, N, blend)]
            ys = [max(get(mode, N, blend)["dist"][1:4]) for N in xs]
            ax[0].plot(xs, ys, "-" if filled else "--", marker=mk, color=BCOL[blend], mfc=BCOL[blend] if filled else "white", ms=7, lw=1.2,
                       label=f"blend {blend:g} {mode}")
    ax[0].axhline(0.01, color=BCOL[0.0], ls=":", lw=1.2)
    ax[0].text(Ns[-1], 0.0115, "blend 0: 0.000(시뮬레이션) ~ 0.010°(실기)", color=BCOL[0.0], ha="right", fontsize=8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xticks(Ns); ax[0].set_xticklabels(Ns)
    ax[0].set_xlabel("jb2 전체 분할 수 N"); ax[0].set_ylabel("경유점 최대 도달 오차 [deg] (6관절 L2)")
    ax[0].set_title("분할 수가 많을수록 경유점에 가까워진다", fontsize=10); ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8)
    for blend, col in ((50.0, "C1"), (100.0, "C3")):
        m = pts[:, 0] == blend
        ax[1].plot(pts[m, 1], pts[m, 2], "o", color=col, label=f"blend {blend:g} (경유점 3개 × 분할 수별)")
        k = np.median(pts[m & (pts[:, 1] <= 12), 3])
        xx = np.array([1.5, 30]); ax[1].plot(xx, k * np.sqrt(6) * xx, "-", color=col, lw=0.8, alpha=0.6, label=f"오차 ≈ {k:.3f} × √6 × 구간 길이")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("경유점 양쪽 jb2 구간 길이의 평균 [deg] (선두 관절)"); ax[1].set_ylabel("경유점 도달 오차 [deg] (6관절 L2)")
    ax[1].set_title("오차는 구간 길이에 거의 비례한다", fontsize=10); ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "r_via_error.png"), dpi=110)

    # ------------------------------------------------------------------ 그림 2: 속도, 소요 시간
    fig = Figure(figsize=(12, 4.6))
    ax = fig.subplots(1, 2)
    for blend in (0.0, 50.0, 100.0):
        for mode, mk, filled in (("시뮬레이션", "o", False), ("실기", "s", True)):
            xs = [N for N in Ns if get(mode, N, blend)]
            kw = dict(color=BCOL[blend], marker=mk, mfc=BCOL[blend] if filled else "white", ms=7, lw=1.2, ls="-" if filled else "--")
            ax[0].plot(xs, [get(mode, N, blend)["linf"] for N in xs], label=f"blend {blend:g} {mode}", **kw)
            ax[1].plot(xs, [get(mode, N, blend)["T_meas"] for N in xs], **kw)
    ax[0].axhline(jb[0]["speed"], color="k", ls=":", lw=1); ax[0].text(Ns[0], jb[0]["speed"] + 0.15, f"설정 속도 {jb[0]['speed']:g} deg/s", fontsize=8)
    ax[1].axhline(jb[0]["T_ideal"], color="k", ls=":", lw=1); ax[1].text(Ns[0], jb[0]["T_ideal"] + 0.4, f"이상 시간 {jb[0]['T_ideal']:.0f} s (경유점마다 정지 가정)", fontsize=8)
    for a, t, yl in ((ax[0], "측정 최고 속도", "선두 관절 최대 |속도| [deg/s]"), (ax[1], "모션 소요 시간", "모션 시작 → 종료 [s]")):
        a.set_xscale("log"); a.set_xticks(Ns); a.set_xticklabels(Ns); a.set_xlabel("jb2 전체 분할 수 N"); a.set_ylabel(yl); a.set_title(t, fontsize=10); a.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "r_speed_time.png"), dpi=110)

    # ------------------------------------------------------------------ 그림 3: 오차-시간 트레이드오프 (실기)
    fig = Figure(figsize=(7.5, 5))
    ax = fig.subplots()
    for blend in (0.0, 50.0, 100.0):
        rr = [get("실기", N, blend) for N in Ns if get("실기", N, blend)]
        ax.plot([r["T_meas"] for r in rr], [max(max(r["dist"][1:4]), 0.005) for r in rr], "-s", color=BCOL[blend], label=f"jb2 blend {blend:g}", ms=7)
        for r in rr:
            ax.annotate(f"N={r['segments']}", (r["T_meas"], max(max(r["dist"][1:4]), 0.005)), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.plot([r["T_meas"] for r in sv], [max(r["dist"][1:4]) for r in sv], "^", color="C2", ms=10, label="servo (주기 20/100/200 ms)")
    for r in sv:
        ax.annotate(f"{r['period_ms']:g} ms", (r["T_meas"], max(r["dist"][1:4])), textcoords="offset points", xytext=(5, -10), fontsize=7)
    ax.axvline(runs[0]["T_ideal"], color="k", ls=":", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("모션 소요 시간 [s]"); ax.set_ylabel("경유점 최대 도달 오차 [deg]"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("실기: 경유점 정확도와 소요 시간의 트레이드오프", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "r_tradeoff.png"), dpi=110)

    # ------------------------------------------------------------------ 그림 4: 예시 궤적 (J6, 실기)
    ex = [("jb2 blend 100, N=4", get("실기", 4, 100.0)), ("jb2 blend 100, N=12", get("실기", 12, 100.0)), ("jb2 blend 100, N=48", get("실기", 48, 100.0)),
          ("jb2 blend 0, N=12", get("실기", 12, 0.0)), ("servo 20 ms", sv[0])]
    fig = Figure(figsize=(12, 9.5))
    axs = fig.subplots(len(ex), 1, sharex=True)
    for a, (lab, r) in zip(axs, ex):
        v = np.loadtxt(r["dir"] + "velocity_0.csv", delimiter=",", skiprows=1, ndmin=2)
        a.plot(v[:, 0], v[:, 6], color="C3", lw=1.4)
        for tgt in (20, 0, 30):
            a.axhline(tgt, color="gray", ls="--", lw=0.6, alpha=0.6)
        a.set_ylabel("J6 [deg]")
        vias = ", ".join("via%d %.2f" % (k, r["dist"][k]) for k in (1, 2, 3))
        a.set_title("%s: 경유점 도달 오차 %s° / 소요 %.1f s" % (lab, vias, r["T_meas"]), fontsize=9)
        a.grid(alpha=0.25)
    axs[-1].set_xlabel("모션 시작 후 시간 [s]   (점선 = 경유 목표 20°, 0°, 30°; 6관절이 같은 값으로 움직이므로 J6 만 표시)")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "r_examples.png"), dpi=100)

    # ------------------------------------------------------------------ 수치 모음
    sim_real = max(abs(get("시뮬레이션", N, b)["dist"][k] - get("실기", N, b)["dist"][k]) for N in Ns for b in (50.0, 100.0) if get("시뮬레이션", N, b) and get("실기", N, b) for k in (1, 2, 3))
    via_tab = lambda b: "| " + " | ".join("-") + " |"
    L = []
    A = L.append
    A("# 웨이포인트 분할에 따른 속도·경유점 도달 오차 (jb2 / servo)\n")
    A("- 목적: 같은 경로를 jb2 의 분할 수·blend 와 servo 의 전송 주기를 바꿔 실행하고, **속도**와 **경유점에 얼마나 가깝게 도달했는지** 비교")
    A("- 실험 시각: 2026-10-16 15:22 ~ 15:37, 도구: `rb_servo_test/jb2_gui`, 로봇 IP 10.0.2.7")
    A(f"- 대상: 분석 {len(runs)}건 (jb2 시뮬레이션 {sum(r['method']=='jb2' and r['mode']=='시뮬레이션' for r in runs)}, jb2 실기 {sum(r['method']=='jb2' and r['mode']=='실기' for r in runs)}, servo 실기 {len(sv)}), 제외 {len(excluded)}건")
    A("- 수치는 모두 `runs_summary.csv` 에 있음 (실행 폴더별 원시 값은 `out/<폴더>/waypoint_proximity_0.csv`)\n")
    A("---\n")
    A("## 요약\n")
    t4 = {b: [max(get("실기", N, b)["dist"][1:4]) for N in Ns if get("실기", N, b)] for b in (0.0, 50.0, 100.0)}
    A("| 명령 | 경유점 도달 오차 (최대, 실기) | 모션 시간 (이상 12 s) | 측정 최고 속도 (설정 10) |")
    A("|---|---|---|---|")
    A(f"| jb2 blend 0 | {min(t4[0.0]):.3f} ~ {max(t4[0.0]):.3f}° (분할 수와 무관) | {min(get('실기', N, 0.0)['T_meas'] for N in Ns4):.1f} ~ {max(get('실기', N, 0.0)['T_meas'] for N in Ns4):.1f} s (분할 수↑ → 느림) | {min(get('실기', N, 0.0)['linf'] for N in Ns4):.1f} ~ {max(get('실기', N, 0.0)['linf'] for N in Ns4):.1f} |")
    A(f"| jb2 blend 50 | {t4[50.0][0]:.2f}° (N=4) → {t4[50.0][-1]:.2f}° (N=24) | {get('실기', 4, 50.0)['T_meas']:.1f} → {get('실기', 24, 50.0)['T_meas']:.1f} s | {min(get('실기', N, 50.0)['linf'] for N in Ns if get('실기', N, 50.0)):.1f} ~ {max(get('실기', N, 50.0)['linf'] for N in Ns if get('실기', N, 50.0)):.1f} |")
    A(f"| jb2 blend 100 | {t4[100.0][0]:.1f}° (N=4) → {get('실기', 48, 100.0)['dist'][3]:.2f}° (N=48) | {get('실기', 4, 100.0)['T_meas']:.1f} → {get('실기', 48, 100.0)['T_meas']:.1f} s | {min(r['linf'] for r in jb if r['blend']==100.0):.1f} ~ **{max(r['linf'] for r in jb if r['blend']==100.0):.1f}** |")
    A(f"| servo (gain 1) | {min(max(r['dist'][1:4]) for r in sv):.3f} ~ {max(max(r['dist'][1:4]) for r in sv):.3f}° (주기 20 → 200 ms) | {min(r['T_meas'] for r in sv):.1f} ~ {max(r['T_meas'] for r in sv):.1f} s | {min(r['linf'] for r in sv):.1f} ~ {max(r['linf'] for r in sv):.1f} |")
    A("")
    A("- **blend 0**: 모든 경유점에 정확히 도달 (시뮬레이션 0.000°, 실기 0.01° 이하). 대신 점마다 정지하므로 분할 수가 늘수록 느려짐")
    A("- **blend 50·100**: 코너를 깎아 경유점에 못 미침. **오차는 구간 길이에 거의 비례** (blend 100 은 구간 길이의 약 0.25배, blend 50 은 약 0.06배)")
    A("- 분할 수를 늘리면 오차는 줄지만 **모션 시간은 늘고 속도는 설정에 못 미침** (blend 100: N=4 12.4 s → N=48 22.1 s)")
    A("- **servo**: 경유점에 0.06° 이내로 도달하면서 시간은 이상값(12 s)과 거의 같음. 전송 주기가 길수록 오차 증가 (20 → 200 ms 에서 5배)")
    A(f"- 시뮬레이션과 실기의 도달 오차 차이는 최대 {sim_real:.3f}° → 경유점 도달은 제어기 계획이 결정함\n")
    A("---\n")
    A("## 1. 실험 조건\n")
    A("- 경로: 6관절이 같은 값으로 움직이는 왕복 `0 → 20 → 0 → 30 → 0` (deg), 시작 자세 `0 0 0 0 0 0`")
    A("  - 웨이포인트 5개: start, via1 (20°), via2 (0°, 반전), via3 (30°), end (0°)")
    A("  - 구간(leg) 4개: 최대 이동 20°, 20°, 30°, 30°")
    A("- 공통 설정: speed 10 deg/s, accel 20 deg/s², speed bar 1.0, 기록 100 Hz, 속도 차분 창 0.1 s")
    A("- 이상 모션 시간 12 s: 경유점마다 정지하는 사다리꼴 프로파일 (`vp = min(speed, √(accel × 구간 길이))`)의 합")
    A("")
    A("| 명령 | 변경한 항목 | 값 |")
    A("|---|---|---|")
    A("| jb2 (`move_jb2_add` → `move_jb2_run`) | 전체 분할 수 N | 4, 8, 12, 24 (실기 blend 100 은 48 추가) |")
    A("| | blend | 0, 50, 100 |")
    A("| servo (`move_servo_j`) | 전송 주기 | 20, 100, 200 ms (gain 1, alpha 1, t2 0.1 s) |")
    A("")
    A("- 분할 수 N 은 **경로 전체**의 jb2 구간 수이며 leg 길이에 비례해 나눔 (각 leg 최소 1개)")
    A("")
    A("| N | leg 별 분할 수 (20°, 20°, 30°, 30°) | jb2 구간 길이 [deg] (선두 관절) |")
    A("|---|---|---|")
    for N in Ns:
        r = get("실기", N, 100.0) or get("시뮬레이션", N, 0.0)
        A(f"| {N} | {r['leg_counts']} | {', '.join(f'{x:.2f}' for x in r['seg_len'])} |")
    A("")
    A("- N=4 는 jb2 에 보낸 점이 경유점·목표 4개뿐임 (중간 점 없음)")
    A("- 시뮬레이션 jb2: 12건 (N 4/8/12/24 × blend 0/50/100), 실기 jb2: 13건 (같은 조합 + N=48 blend 100), 실기 servo: 3건")
    A(f"- 제외: servo gain 0.1 (사용자 중지로 미완료) {len(excluded)}건\n")
    A("---\n")
    A("## 2. 평가 지표\n")
    A("- **경유점 도달 오차**: 실측 경로(100 Hz 샘플 사이를 선분으로 이은 꺾은선)와 웨이포인트의 **6관절 공간 최단 거리** (deg, L2)")
    A("  - 시간은 보지 않음 (timing 이 달라도 경로가 그 점을 지났는지만 평가)")
    A("  - 웨이포인트를 지정한 순서대로 지나는 것으로 배정 (시작=목표인 왕복도 구분)")
    A("  - 본 보고서의 값은 \"경유점 3개 중 최댓값\"이며, 개별 값은 CSV 의 via1/via2/via3")
    A("- **측정 최고 속도**: 선두 관절의 최대 |속도| (0.1 s 중심 차분)")
    A("- **모션 소요 시간**: 시작 자세에서 벗어난 시각 ~ 최종 자세의 0.05° 이내에 계속 머무는 시각")
    A("- 같은 조건 1회씩 실행 (반복 없음)\n")
    A("---\n")
    A("## 3. 결과\n")
    A("### 3.1 jb2: 분할 수와 경유점 도달 오차\n")
    A("![경유점 도달 오차](img/r_via_error.png)\n")
    A("경유점 최대 도달 오차 [deg] (시뮬레이션 / 실기)\n")
    A("| 분할 수 N | blend 0 | blend 50 | blend 100 |")
    A("|---|---|---|---|")
    for N in Ns:
        cells = []
        for b in (0.0, 50.0, 100.0):
            s_, r_ = get("시뮬레이션", N, b), get("실기", N, b)
            fs = fmt(max(s_["dist"][1:4]), 3) if s_ else "-"
            fr = fmt(max(r_["dist"][1:4]), 3) if r_ else "-"
            cells.append(f"{fs} / {fr}")
        A(f"| {N} | " + " | ".join(cells) + " |")
    A("")
    A("경유점별 오차 (실기, blend 100) [deg]\n")
    A("| N | via1 (20°) | via2 (0°, 반전) | via3 (30°) | 보낸 분할점 평균 / 최대 |")
    A("|---|---|---|---|---|")
    for N in Ns + [48]:
        r = get("실기", N, 100.0)
        if r:
            A(f"| {N} | {r['dist'][1]:.2f} | {r['dist'][2]:.2f} | {r['dist'][3]:.2f} | {r['cmd_mean']:.2f} / {r['cmd_max']:.2f} |")
    A("")
    A("- **blend 0**: 분할 수와 무관하게 모든 경유점에 정확히 도달 (시뮬레이션 0.000°, 실기 0.008 ~ 0.010° = 엔코더 측정 수준)")
    A(f"- **blend 50**: N=4 {max(get('실기', 4, 50.0)['dist'][1:4]):.2f}° → N=24 {max(get('실기', 24, 50.0)['dist'][1:4]):.2f}°")
    A(f"- **blend 100**: N=4 {max(get('실기', 4, 100.0)['dist'][1:4]):.1f}° → N=24 {max(get('실기', 24, 100.0)['dist'][1:4]):.2f}° → N=48 {max(get('실기', 48, 100.0)['dist'][1:4]):.2f}°")
    A("- 오차가 가장 큰 경유점은 via3 (30° 정점). 이동량이 큰 구간(30°)에서 코너 절삭이 큼")
    A("- **오차와 구간 길이의 관계**: 경유점 양쪽 jb2 구간 길이의 평균 ℓ 에 대해 오차 / (ℓ × √6) 비율 (√6 = 6관절이 같은 값으로 움직일 때 L2 환산)\n")
    A("| 구간 길이 ℓ | blend 50 비율 (평균, 범위, n) | blend 100 비율 (평균, 범위, n) |")
    A("|---|---|---|")
    for lo, hi in bins:
        cells = []
        for b in (50.0, 100.0):
            s_ = ratio_stats(b, lo, hi)
            cells.append(f"{s_[0]:.3f} ({s_[1]:.3f} ~ {s_[2]:.3f}, {s_[3]})" if s_ else "-")
        A(f"| {lo} ~ {hi}° | " + " | ".join(cells) + " |")
    A("")
    A("- 구간이 12° 이하이면 비율이 거의 일정: blend 50 은 약 0.05 ~ 0.07, blend 100 은 약 0.18 ~ 0.26")
    A("  - 관절당 환산: 경유점에서 **각 관절이 목표에서 벗어나는 양 ≈ ℓ × (blend 50: 0.06, blend 100: 0.23)** (6관절 같은 이동량, speed 10 기준)")
    A("- 구간이 12° 를 넘으면 비율이 더 큼 (평균 blend 50 0.079, blend 100 0.301; 표 참고)")
    A("- 같은 경로에서 얻은 경험식이므로 속도·자세·이동량 비율이 다른 경로에는 그대로 적용하지 말고 확인해야 함\n")
    A("### 3.2 jb2: 속도와 소요 시간\n")
    A("![속도와 소요 시간](img/r_speed_time.png)\n")
    A("측정 최고 속도 [deg/s] (시뮬레이션 / 실기, 설정 10)\n")
    A("| 분할 수 N | blend 0 | blend 50 | blend 100 |")
    A("|---|---|---|---|")
    for N in Ns:
        cells = []
        for b in (0.0, 50.0, 100.0):
            s_, r_ = get("시뮬레이션", N, b), get("실기", N, b)
            cells.append(f"{fmt(s_['linf'], 1) if s_ else '-'} / {fmt(r_['linf'], 1) if r_ else '-'}")
        A(f"| {N} | " + " | ".join(cells) + " |")
    A(f"| 48 | - | - | - / {get('실기', 48, 100.0)['linf']:.1f} |")
    A("")
    A("모션 소요 시간 [s] (시뮬레이션 / 실기, 이상 12.0 s)\n")
    A("| 분할 수 N | blend 0 | blend 50 | blend 100 |")
    A("|---|---|---|---|")
    for N in Ns:
        cells = []
        for b in (0.0, 50.0, 100.0):
            s_, r_ = get("시뮬레이션", N, b), get("실기", N, b)
            cells.append(f"{fmt(s_['T_meas'], 1) if s_ else '-'} / {fmt(r_['T_meas'], 1) if r_ else '-'}")
        A(f"| {N} | " + " | ".join(cells) + " |")
    A(f"| 48 | - | - | - / {get('실기', 48, 100.0)['T_meas']:.1f} |")
    A("")
    A("- 대부분 **설정 속도 10 에 못 미침** (5.0 ~ 7.1). 구간이 짧으면 속도를 올릴 거리가 부족하기 때문 (`vp ≈ √(accel × 구간 길이)`)")
    A(f"- **blend 100, N=8 에서만 설정 속도 초과**: 시뮬레이션 {get('시뮬레이션', 8, 100.0)['linf']:.1f}, 실기 {get('실기', 8, 100.0)['linf']:.1f} (설정의 {get('실기', 8, 100.0)['linf']/10:.2f}배), N=12 는 {get('실기', 12, 100.0)['linf']:.1f}. 코너를 가로지르며 속도가 올라간 것으로 보이며 원인은 확인하지 못함")
    A("- **blend 0**: 점마다 정지해 시간이 길고 분할 수가 늘수록 더 길어짐 (N=4 18.9 s → N=24 29.7 s, N이 6배일 때 시간은 약 1.6배)")
    A("- **blend 100**: 이상 시간(12 s)에 가장 가까움 (N=4 12.4 s) 하지만 분할 수를 늘릴수록 길어짐 (N=48 22.1 s)\n")
    A("### 3.3 servo: 전송 주기와 경유점 도달 오차\n")
    A("| 설정 주기 | 실제 전송 간격 평균 ± 표준편차 [ms] (최소 ~ 최대) | 계획 대비 늦음 평균 / 최대 [ms] | 경유점 최대 오차 [deg] | via1 / via2 / via3 [deg] | 측정 최고 속도 | 모션 시간 [s] |")
    A("|---|---|---|---|---|---|---|")
    for r in sv:
        A(f"| {r['period_ms']:g} ms | {r['iv_mean']:.2f} ± {r['iv_std']:.2f} ({r['iv_min']:.1f} ~ {r['iv_max']:.1f}) | {r['lag_mean']:.2f} / {r['lag_max']:.2f} | {max(r['dist'][1:4]):.3f} | {r['dist'][1]:.3f} / {r['dist'][2]:.3f} / {r['dist'][3]:.3f} | {r['linf']:.1f} | {r['T_meas']:.2f} |")
    A("")
    A("- **전송 주기의 정확도**: PC 에서 `move_servo_j` 를 보낸 시각 기준으로 계획 간격과 거의 같음 (20 ms, 100 ms 는 표준편차 0.03 ~ 0.04 ms, 계획 대비 늦음 최대 0.7 ms)")
    A(f"- **설정 200 ms 는 실제로 평균 {sv[-1]['iv_mean']:.1f} ms**: 모션 시간 12 s 를 설정 주기 이하의 같은 간격으로 나누므로 (`ceil(12 / 0.2) = 62`개) 설정값은 주기의 상한임. 이 표의 \"200 ms\"는 실제 193.5 ms")
    A("- 측정한 것은 **PC 가 보낸 시각**임. 제어기가 명령을 받아 처리한 시각(네트워크·버퍼 지연)은 측정하지 못했음")
    A("- 지터 최대 0.7 ms 는 20 ms 주기의 3.5 %, t2 (0.1 s) 에 비해 작음. 다만 일반 Linux 라 드물게 큰 지연이 생길 수 있고, 본 실험 600회에서는 관측되지 않음")
    A("")
    A("- 모두 0.06° 이내로 경유점에 도달하고 모션 시간은 이상값 12 s 와 거의 같음 (11.9 ~ 12.0 s)")
    A("- 전송 주기가 길수록 오차가 커짐 (20 ms 0.012° → 설정 200 ms(실제 193.5 ms) 0.060°, 5배)")
    A("- 측정 최고 속도는 10.5 ~ 10.7 로 설정의 1.05 ~ 1.07배")
    A("- servo 는 시뮬레이션 실행이 없고, gain 0.1 실행은 중단되어 gain 의 영향은 확인하지 못함\n")
    A("- servo 는 경로 전체를 하나의 시간 프로파일로 보내므로 jb2 와 달리 분할 수·blend 개념이 없음\n")
    A("### 3.4 트레이드오프\n")
    A("![정확도와 시간의 트레이드오프](img/r_tradeoff.png)\n")
    A("- jb2 는 **blend 0 (정확하지만 느림) ↔ blend 100 (빠르지만 코너를 깎음)** 사이에서 선택하고, 분할 수로 정확도를 사 오면 시간이 늘어남")
    A("- servo 는 0.06° 이하의 도달 오차와 이상 시간에 가까운 시간을 동시에 만족")
    A("- 각 실행의 궤적 예 (J6):\n")
    A("![예시 궤적](img/r_examples.png)\n")
    A("---\n")
    A("## 4. 결론과 권장\n")
    A("1. **경유점을 정확히 지나야 하면 jb2 blend 0 또는 servo**: 0.01 ~ 0.06° 이내 (jb2 blend 0 은 느림, servo 는 주기 20 ms 권장)")
    A("2. **jb2 blend 50·100 은 코너 절삭이 있음**: 오차 ≈ 구간 길이 × 계수 (blend 50 약 0.06, blend 100 약 0.23, 관절당)")
    A("   - 허용 오차 e [deg/관절] 를 정하면 구간 길이 ≈ e / 계수 이하로 분할 (예: blend 100 에서 관절당 0.5° 이내이면 구간 약 2° 이하)")
    A("   - 분할 수가 늘수록 시간이 늘고 속도가 낮아지므로, 정확도와 시간 사이에서 정함")
    A("3. **jb2 blend 100 은 일부 분할 수에서 설정 속도를 초과**할 수 있음 (N=8 에서 1.2배). 속도 제한이 중요하면 실측으로 확인")
    A("4. **시뮬레이션으로 사전 선정 가능**: 경유점 도달 오차 차이 최대 " + f"{sim_real:.3f}° (jb2)\n")
    A("---\n")
    A("## 5. 한계\n")
    A("- 경로 1종 (6관절 같은 값으로 움직이는 왕복, 이동량 20°·30°), 속도 10 deg/s, 가속도 20 deg/s² 한 조건. 속도·가속도·관절 구성이 다르면 계수와 속도 거동이 달라질 수 있음")
    A("- 조건마다 1회 실행 (반복 오차 미확인). 단, 시뮬레이션과 실기가 같은 값을 줘 재현성은 높아 보임")
    A("- servo 는 실기 3건만 (시뮬레이션 없음, gain·alpha 변화 없음, 분할 개념 없음)")
    A("- 오차 계수(blend 별 비율)는 이 경로에서 얻은 경험값이며 제어기 내부 blending 식은 확인하지 못함")
    A("- 이상 시간 12 s 는 경유점마다 정지하는 프로파일의 합으로, blend 가 있는 제어기 동작의 예측값이 아님")
    A("- 6관절이 같은 값으로 움직이는 경로라 관절마다 속도·가속도 한계가 다른 효과(이전 단일 관절 실험)는 이 경로에서 분리되지 않음\n")
    A("---\n")
    A("## 부록. 파일\n")
    A("| 파일 | 내용 |")
    A("|---|---|")
    A("| `runs_summary.csv` | 실행별 조건, 경유점 오차, 속도, 시간 (분석 29건 중 완료 28건) |")
    A("| `out/20261016_152212_* ~ 153709_*/` | 원본 실행 폴더 (`waypoint_proximity_0.csv`, `command_points_proximity_0.csv`, `velocity_0.csv`, `summary.csv`) |")
    A("| `report/make_report.py`, `collect.py` | 이 보고서의 표·그림 생성 |")
    open(os.path.join(HERE, "report.md"), "w", encoding="utf-8").write("\n".join(L))
    print("report.md", len(L), "줄;", "sim-real max diff", round(sim_real, 4))


if __name__ == "__main__":
    main()
