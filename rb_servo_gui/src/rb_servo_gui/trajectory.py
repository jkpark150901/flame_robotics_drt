"""입력 궤적 읽기/보간/서브샘플링. 내부 단위: 보간 궤적은 rad, 로봇 명령은 deg."""
import csv
import re

import numpy as np
from scipy.interpolate import CubicSpline


_JCOL = re.compile(r"^j([1-6])(?:[\s_\[(]*(deg|rad)[\])]*)?$")


def load_waypoints(path, n_arm=6):
    """웨이포인트 CSV 를 읽어 (열 이름, 팔 6관절 [rad]) 를 반환한다. 두 가지 형식을 자동 인식한다.

    1) J1..J6 열 형식 (예: `Order,J1..J6,Speed,Acceleration`): 외부축 없음. 열 이름의 단위 표시로 구분한다.
       `J1_rad`, `J1[rad]`, `J1(rad)` 는 rad, `J1_deg` 등 deg 표시 또는 **단위 표시가 없으면 deg** (기존 robot1 형식). Speed/Acceleration 열은 무시.
    2) 플래너 출력 형식 (첫 열 waypoint 번호 + 관절열, 단위 **rad**): 마지막 n_arm 개 열(팔)만 사용 (앞의 외부축 열은 무시).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = [row for row in csv.reader(f) if row]
    header = [h.strip() for h in r[0]]
    cols, units = {}, set()
    for i, h in enumerate(header):
        m = _JCOL.match(h.lower())
        if m:
            cols[int(m.group(1))] = i
            units.add(m.group(2) or "deg")
    if set(cols) == set(range(1, n_arm + 1)):
        if len(units) > 1:
            raise ValueError(f"관절 열의 단위가 섞여 있음: {sorted(units)}")
        q = np.array([[float(row[cols[j]]) for j in range(1, n_arm + 1)] for row in r[1:]])
        if units == {"deg"}:
            q = np.deg2rad(q)
        return [header[cols[j]] for j in range(1, n_arm + 1)], q
    q = np.array([[float(x) for x in row[1:]] for row in r[1:]])
    return header[1:][-n_arm:], q[:, -n_arm:]


def time_parametrize(q, vmax_rad, tmin=0.2):
    """구간 시간 = max(|dq|)/vmax. 너무 짧은 구간은 tmin 으로 하한."""
    dt = np.maximum(np.abs(np.diff(q, axis=0)).max(axis=1) / vmax_rad, tmin)
    return np.concatenate([[0.0], np.cumsum(dt)])


def interpolate(q, vmax_deg, rate):
    """웨이포인트(rad) -> 시간 부여 + 3차 스플라인(시작/끝 속도 0) 보간. 반환 t[s], qs[rad]."""
    keep = np.r_[True, np.abs(np.diff(q, axis=0)).max(axis=1) > 1e-9]  # 중복 웨이포인트 제거
    q = q[keep]
    t = time_parametrize(q, np.deg2rad(vmax_deg))
    cs = CubicSpline(t, q, bc_type="clamped")
    ts = np.append(np.arange(0.0, t[-1], 1.0 / rate), t[-1])
    return ts, cs(ts)


def load_interp(path):
    """보간된 CSV (t, j0..j5 [rad]) 읽기."""
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    return d[:, 0], d[:, 1:7]


def save_interp(path, t, q):
    np.savetxt(path, np.c_[t, q], delimiter=",", header="t,j0,j1,j2,j3,j4,j5", comments="", fmt="%.7f")


def prepare(s):
    """Settings 에 따라 기준(원본) 궤적 t[s], q[rad] 를 만든다."""
    if not s.traj_path:
        raise ValueError("궤적 파일이 지정되지 않았습니다")
    if s.traj_kind == "interp":
        return load_interp(s.traj_path)
    _, q = load_waypoints(s.traj_path)
    return interpolate(q, s.vmax, s.interp_rate)


def segment_lengths(q_deg, basis="max_joint"):
    """지령점 꺾은선의 구간 길이 [deg]. max_joint: 가장 많이 움직이는 관절의 이동량, l2: 6관절 벡터 크기."""
    dq = np.diff(np.asarray(q_deg, dtype=float), axis=0)
    if basis == "max_joint":
        return np.abs(dq).max(axis=1)
    if basis == "l2":
        return np.linalg.norm(dq, axis=1)
    raise ValueError("basis 는 max_joint | l2")


class SpeedProfile:
    """지령점 꺾은선 전체를 시작·끝에서만 정지하는 사다리꼴(가속-등속-감속) 속도로 지나가는 이상 프로파일.

    경로 길이 L 은 구간 길이(basis 기준)의 합, 속력 v 는 speed 이하, 가감속은 accel 이하.
      vp = min(speed, sqrt(accel * L)),  ta = vp / accel,  T = L / vp + ta
    관절 속도는 구간마다 v · Δq / D (D = 구간 길이). 꺾은선의 각 점에서 방향만 바뀌고 속력은 연속이다.
    """

    def __init__(self, q_deg, speed, accel, basis="max_joint"):
        if not (speed > 0 and accel > 0):
            raise ValueError("speed, accel 은 0 보다 커야 합니다")
        self.q = np.asarray(q_deg, dtype=float)
        self.basis = basis
        self.D = segment_lengths(self.q, basis)
        self.S = np.concatenate([[0.0], np.cumsum(self.D)])  # 각 점까지의 누적 거리
        self.L = float(self.S[-1])
        if self.L <= 0:
            raise ValueError("경로 길이가 0 입니다")
        self.A = float(accel)
        self.vp = min(float(speed), float(np.sqrt(self.A * self.L)))
        self.ta = self.vp / self.A
        self.da = 0.5 * self.A * self.ta ** 2
        self.T = self.L / self.vp + self.ta
        self.times = self.t_of_s(self.S)  # 각 지령점의 이상 도달 시각

    def s_of_t(self, t):
        t = np.clip(np.asarray(t, dtype=float), 0.0, self.T)
        A, ta, vp, T = self.A, self.ta, self.vp, self.T
        return np.where(t < ta, 0.5 * A * t ** 2,
                        np.where(t <= T - ta, self.da + vp * (t - ta), self.L - 0.5 * A * (T - t) ** 2))

    def v_of_t(self, t):
        t = np.asarray(t, dtype=float)
        v = np.minimum(np.minimum(self.A * t, self.vp), self.A * (self.T - t))
        return np.where((t < 0) | (t > self.T), 0.0, np.maximum(v, 0.0))

    def t_of_s(self, s):
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.L)
        A, ta, vp, T, da = self.A, self.ta, self.vp, self.T, self.da
        return np.where(s < da, np.sqrt(2 * s / A),
                        np.where(s <= self.L - da, ta + (s - da) / vp, T - np.sqrt(np.maximum(2 * (self.L - s) / A, 0.0))))

    def joints(self, t):
        """시각 t [s] 의 이상 관절각 [deg] 과 관절 속도 [deg/s] (n, 6)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        s = self.s_of_t(t)
        v = self.v_of_t(t)
        k = np.clip(np.searchsorted(self.S, s, side="right") - 1, 0, len(self.D) - 1)
        D = self.D[k]
        dq = self.q[k + 1] - self.q[k]
        frac = np.divide(s - self.S[k], D, out=np.zeros_like(s), where=D > 0)
        dirn = np.divide(dq, D[:, None], out=np.zeros_like(dq), where=D[:, None] > 0)
        return self.q[k] + frac[:, None] * dq, v[:, None] * dirn

    def schedule(self, min_period):
        """servo 전송 시간표. 이상 도달 시각 간격이 min_period 보다 짧은 점은 건너뛴다(마지막 점은 항상 보냄).
        반환: (점 번호 배열 k_1..k_n, 전송 시각 = 직전 보낸 점의 도달 시각, t1 = 도달 시각 간격)."""
        keep = [0]
        for m in range(1, len(self.q)):
            if self.D[m - 1] == 0 and self.times[m] == self.times[keep[-1]]:
                continue  # 같은 자세의 반복 점
            if m == len(self.q) - 1:
                if len(keep) > 1 and self.times[m] - self.times[keep[-1]] < min_period:
                    keep.pop()  # 마지막 점과 너무 가까운 직전 점 대신 마지막 점을 보낸다
                keep.append(m)
            elif self.times[m] - self.times[keep[-1]] >= min_period:
                keep.append(m)
        k = np.array(keep)
        tk = self.times[k]
        return k[1:], tk[:-1], np.diff(tk)


class PathProfile:
    """조밀한 원본 궤적(경로) 위를 관절 속도·가속도 한계 안에서 가장 빨리 지나가는 속도 프로파일 (시작·끝 정지).

    경로를 길이 s(basis 기준: max_joint 는 가장 많이 움직이는 관절의 이동량, l2 는 6관절 벡터 크기)로 매개화하고,
    관절 속도 dq/dt = q'(s)·v, 관절 가속도 = q''(s)·v² + q'(s)·a 가 다음을 만족하도록 v(s) 를 정한다.
      max_joint: |관절 속도| ≤ speed, |관절 가속도| ≤ accel (관절마다)
      l2       : ‖관절 속도‖₂ ≤ speed, ‖관절 가속도‖₂ ≤ accel
    전진(최대 가속)·후진(최대 감속) 적분으로 구한다. 직선 경로에서는 사다리꼴(가속-등속-감속)과 같고,
    굽은 곳·방향이 바뀌는 곳에서는 관절 가속도 한계 때문에 감속한다. 격자 이산화로 한계를 수 % 넘을 수 있다.
    """

    def __init__(self, q_deg, speed, accel, basis="max_joint"):
        if not (speed > 0 and accel > 0):
            raise ValueError("speed, accel 은 0 보다 커야 합니다")
        if basis not in ("max_joint", "l2"):
            raise ValueError("basis 는 max_joint | l2")
        q = np.asarray(q_deg, dtype=float)
        ds = segment_lengths(q, basis)
        q = q[np.r_[True, ds > 1e-9]]  # 같은 자세의 반복 샘플 제거
        if len(q) < 2:
            raise ValueError("경로 길이가 0 입니다")
        ds = segment_lengths(q, basis)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        # 경로 q(s): 원본 샘플을 지나는 3차 스플라인 (지령을 뽑을 때 직선 보간의 꺾임이 가속도 튐이 되지 않도록)
        self._cs = CubicSpline(s, q, axis=0)
        qp = self._cs(s, 1)     # dq/ds
        qpp = self._cs(s, 2)    # d²q/ds²
        V, A = float(speed), float(accel)
        if basis == "max_joint":
            qp_n = np.abs(qp).max(axis=1)
            curv = np.abs(qpp).max(axis=1)
        else:
            qp_n = np.linalg.norm(qp, axis=1)
            curv = np.linalg.norm(qpp, axis=1)
        # 속도 한계 곡선: 관절 속도 ≤ V, 그리고 a = 0 으로도 관절 가속도 ≤ A 가 되는 속도
        xmax = np.minimum((V / np.maximum(qp_n, 1e-12)) ** 2, np.where(curv > 1e-12, A / np.maximum(curv, 1e-12), np.inf))

        def a_range(i, x):
            """속도² = x 일 때 허용되는 경로 가속도 a 범위."""
            c = qpp[i] * x
            if basis == "max_joint":
                p = qp[i]
                m = np.abs(p) > 1e-9
                if not m.any():  # 반전점 등 q' ≈ 0: 가속도는 a 와 무관 (속도는 곡률 한계 xmax 가 제한)
                    return -np.inf, np.inf
                b1, b2 = (-A - c[m]) / p[m], (A - c[m]) / p[m]
                return float(np.minimum(b1, b2).max()), float(np.maximum(b1, b2).min())
            # l2: ‖c + p·a‖ ≤ A 인 a 구간 (2차 부등식)
            p = qp[i]
            pp, pc, cc = p @ p, p @ c, c @ c
            disc = pc * pc - pp * (cc - A * A)
            if pp < 1e-12 or disc < 0:
                return 0.0, 0.0
            r = np.sqrt(disc)
            return (-pc - r) / pp, (-pc + r) / pp

        n = len(s)
        x = np.zeros(n)
        for i in range(n - 1):  # 전진: 최대 가속
            _, hi = a_range(i, x[i])
            x[i + 1] = min(xmax[i + 1], x[i] + 2 * max(hi, 0.0) * ds[i])
        x[-1] = 0.0
        for i in range(n - 2, -1, -1):  # 후진: 최대 감속
            lo, _ = a_range(i + 1, x[i + 1])
            x[i] = min(x[i], x[i + 1] - 2 * min(lo, 0.0) * ds[i])
        v = np.sqrt(np.maximum(x, 0.0))
        dt = 2 * ds / np.maximum(v[:-1] + v[1:], 1e-12)
        self.basis, self.q, self.s, self.v = basis, q, s, v
        self.t = np.concatenate([[0.0], np.cumsum(dt)])
        self.qd = qp * v[:, None]
        self.T = float(self.t[-1])
        self.L = float(s[-1])
        self.vp = float(v.max())

    def s_v(self, t):
        """시각 t 의 경로 위치 s 와 속력 v. 격자 구간마다 등가속 운동 (격자 시간 dt = 2ds/(v_i+v_(i+1)) 과 일치)."""
        t = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), 0.0, self.T)
        i = np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2)
        tau = t - self.t[i]
        a = (self.v[i + 1] - self.v[i]) / np.maximum(self.t[i + 1] - self.t[i], 1e-12)
        return np.minimum(self.s[i] + self.v[i] * tau + 0.5 * a * tau ** 2, self.L), np.maximum(self.v[i] + a * tau, 0.0)

    def joints(self, t):
        """시각 t [s] 의 이상 관절각 [deg] 과 관절 속도 [deg/s] (n, 6)."""
        sp, v = self.s_v(t)
        return self._cs(sp), self._cs(sp, 1) * v[:, None]

    @classmethod
    def fit(cls, q_deg, speed, accel, basis="max_joint", period=0.02, iters=8):
        """실제로 보낼 지령(period 주기 샘플)의 관절 속도·가속도(차분)가 speed·accel 이하가 되도록 맞춘 프로파일.

        격자 이산화와 주기 샘플링 때문에 지령 차분이 한계를 조금 넘을 수 있어, 넘으면 계획용 한계를 그 비율만큼 줄여 다시 만든다.
        결과 프로파일의 cmd_speed_max / cmd_accel_max 는 지령 차분으로 잰 값, speed_plan / accel_plan 은 실제 계획에 쓴 한계.
        """
        V, A = float(speed), float(accel)
        sv, sa = V, A
        for _ in range(iters):
            p = cls(q_deg, sv, sa, basis)
            vmax, amax = p.command_limits(period)
            if vmax <= V * (1 + 1e-6) and amax <= A * (1 + 1e-6):
                break
            sv = sv * min(1.0, V / vmax) * (0.995 if vmax > V else 1.0)
            sa = sa * min(1.0, A / amax) * (0.995 if amax > A else 1.0)
        p.speed_plan, p.accel_plan = sv, sa
        p.cmd_speed_max, p.cmd_accel_max = p.command_limits(period)
        return p

    def command_limits(self, period):
        """period 주기로 샘플링한 지령의 차분 속도·가속도 최대 (basis 기준: max_joint 는 관절별 최대, l2 는 벡터 크기).
        시작 전과 끝난 뒤는 정지(속도 0)로 본다."""
        q, _, dt = self.schedule(period)
        d = float(dt[0])
        pts = np.vstack([self.q[0], q])
        v = np.vstack([np.zeros(6), np.diff(pts, axis=0) / d, np.zeros(6)])
        a = np.diff(v, axis=0) / d
        nrm = (lambda x: np.abs(x).max(axis=1)) if self.basis == "max_joint" else (lambda x: np.linalg.norm(x, axis=1))
        return float(nrm(v).max()), float(nrm(a).max())

    def schedule(self, period):
        """servo 전송 시간표: 프로파일을 고정 주기(≤ period, 끝을 정확히 포함)로 샘플링.
        반환: (목표 관절각 (n,6), 전송 시각 (n,), t1 (n,)). 목표 k 는 시각 k·dt 의 프로파일 위치이며 (k−1)·dt 에 전송한다."""
        if not 0.002 <= period <= 0.2:
            raise ValueError("servo 전송 주기는 0.002 ~ 0.2 s")
        n = max(int(np.ceil(self.T / period - 1e-9)), 1)
        times = np.linspace(0.0, self.T, n + 1)
        q, _ = self.joints(times[1:])
        return q, times[:-1], np.diff(times)


def subsample(t, q_deg, step):
    """Uniform stride plus every joint's sampled extrema and plateau boundaries.

    Preserve all six coordinates at each selected time. Exact continuous-time
    extrema between input samples cannot be recovered from a sampled CSV.
    """
    t, q_deg = np.asarray(t), np.asarray(q_deg)
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError("step must be a positive integer")
    if (t.ndim != 1 or len(t) < 2 or q_deg.ndim != 2 or len(q_deg) != len(t)
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(q_deg))
            or np.any(np.diff(t) <= 0)):
        raise ValueError("Trajectory must contain finite positions and increasing times")
    # + -> - / - -> + preserves strict extrema. +/- -> 0 -> +/-
    # preserves both ends of every stationary plateau, including reversals.
    signs = np.sign(np.diff(q_deg, axis=0))
    features = np.flatnonzero(np.any(signs[:-1] != signs[1:], axis=1)) + 1
    idx = np.unique(np.r_[np.arange(0, len(t), step), features, len(t) - 1])
    return t[idx], q_deg[idx]
