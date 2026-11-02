"""jb2_gui 실행 폴더(out/20261016_152000 이후)에서 보고서용 표를 만든다. 실행 폴더는 읽기만 한다."""
import csv, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from model import ideal_arrival_times, path_plan  # noqa: E402

OUT = os.path.join(HERE, "..", "out")
SINCE = "20261016_152000"


def motion_end(t, q):
    """모션 끝: 이후 계속 최종 자세의 0.05° 이내에 머무는 첫 시각 (모션 시작 = 0)."""
    ok = np.abs(q - q[-1]).max(1) < 0.05
    suf = np.flip(np.logical_and.accumulate(np.flip(ok)))
    return float(t[int(np.argmax(suf))]) if suf.any() else float(t[-1])


def collect():
    runs = []
    for d in sorted(glob.glob(os.path.join(OUT, "20261016_*/"))):
        name = os.path.basename(d.rstrip("/"))
        if name < SINCE or not os.path.exists(d + "config.json"):
            continue
        c = json.load(open(d + "config.json"))
        m = json.load(open(d + "manifest.json"))[0] if os.path.exists(d + "manifest.json") else {}
        r = dict(run=name, mode="실기" if c["mode"] == "real" else "시뮬레이션", method=c["method"], blend=c.get("blend"), gain=c.get("gain"),
                 period_ms=c.get("servo_period_ms"), segments=c.get("segments"), speed=c["speed"], accel=c["accel"], completed=bool(m.get("completed")), error=m.get("error"), dir=d)
        if r["completed"] and os.path.exists(d + "waypoint_proximity_0.csv"):
            wp = list(csv.DictReader(open(d + "waypoint_proximity_0.csv")))
            r["dist"] = [float(x["dist_L2_deg"]) for x in wp]              # start, via1, via2, via3, end
            r["joint_err"] = [float(x["max_joint_error_deg"]) for x in wp]
            s = list(csv.DictReader(open(d + "summary.csv")))[0]
            r["linf"], r["l2"] = float(s["linf_peak"]), float(s["l2_peak"])
            v = np.loadtxt(d + "velocity_0.csv", delimiter=",", skiprows=1, ndmin=2)
            r["T_meas"] = motion_end(v[:, 0], v[:, 1:7])
            r["T_ideal"] = float(ideal_arrival_times(c)[-1])
            if c["method"] == "servo" and os.path.exists(d + "servo_schedule.csv"):
                sc = np.loadtxt(d + "servo_schedule.csv", delimiter=",", skiprows=1, ndmin=2)
                iv = np.diff(sc[:, 1]) * 1000
                r["iv_mean"], r["iv_std"], r["iv_min"], r["iv_max"] = float(iv.mean()), float(iv.std()), float(iv.min()), float(iv.max())
                r["lag_mean"], r["lag_max"] = float((sc[:, 1] - sc[:, 0]).mean() * 1000), float((sc[:, 1] - sc[:, 0]).max() * 1000)
                r["n_send"] = int(len(sc))
            if c["method"] == "jb2":
                pts, times, anchors = path_plan(c)
                cnt = np.diff(np.r_[0, anchors])
                verts = np.array([c["base"]] + c["via"] + [c["end"]])
                L = np.abs(np.diff(verts, axis=0)).max(1)
                r["leg_counts"] = cnt.tolist()
                r["seg_len"] = (L / cnt).tolist()                            # leg 별 jb2 구간 길이 (선두 관절 °)
                cp = list(csv.DictReader(open(d + "command_points_proximity_0.csv")))
                dd = np.array([float(x["dist_L2_deg"]) for x in cp])
                r["cmd_mean"], r["cmd_max"] = float(dd.mean()), float(dd.max())
        runs.append(r)
    return runs


if __name__ == "__main__":
    runs = collect()
    print(len(runs), "건")
    for r in runs:
        if r.get("dist"):
            print(r["run"][9:15], r["mode"][:2], r["method"], r["segments"] if r["method"] == "jb2" else r["period_ms"], r["blend"] if r["method"] == "jb2" else r["gain"],
                  "T_meas %.2f / ideal %.2f" % (r["T_meas"], r["T_ideal"]), "linf %.1f" % r["linf"], "via %.3f" % max(r["dist"][1:4]),
                  "cmd mean/max %.3f/%.3f" % (r["cmd_mean"], r["cmd_max"]) if "cmd_mean" in r else "")
