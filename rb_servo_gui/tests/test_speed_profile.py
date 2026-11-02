import unittest

import numpy as np

from rb_servo_gui.trajectory import PathProfile, SpeedProfile


def line(n, delta):
    """시작 0 에서 delta 까지 n 구간으로 균등 분할한 직선 경로 (deg)."""
    return np.outer(np.linspace(0.0, 1.0, n + 1), np.asarray(delta, dtype=float))


class SpeedProfileTests(unittest.TestCase):
    def test_trapezoid_matches_closed_form(self):
        # 120 deg, 50 deg/s, 100 deg/s²: ta = 0.5 s, T = 120/50 + 0.5 = 2.9 s
        p = SpeedProfile(line(12, [120, 60, 0, 0, 0, 0]), 50, 100)
        self.assertAlmostEqual(p.L, 120.0)
        self.assertAlmostEqual(p.vp, 50.0)
        self.assertAlmostEqual(p.ta, 0.5)
        self.assertAlmostEqual(p.T, 2.9)
        self.assertAlmostEqual(p.times[-1], 2.9)

    def test_triangle_when_too_short_for_full_speed(self):
        p = SpeedProfile(line(4, [10, 0, 0, 0, 0, 0]), 50, 100)  # sqrt(100*10) = 31.6 < 50
        self.assertAlmostEqual(p.vp, np.sqrt(1000.0))
        self.assertAlmostEqual(p.T, 2 * p.vp / 100)

    def test_speed_and_acceleration_limits_and_endpoints(self):
        rng = np.random.default_rng(0)
        q = np.cumsum(rng.normal(0, 2, (60, 6)), axis=0)
        p = SpeedProfile(q, 30, 60)
        t = np.linspace(0, p.T, 4001)
        qt, vt = p.joints(t)
        np.testing.assert_allclose(qt[0], q[0], atol=1e-9)
        np.testing.assert_allclose(qt[-1], q[-1], atol=1e-9)
        self.assertLessEqual(np.abs(vt).max(axis=1).max(), 30 + 1e-9)  # max_joint 기준: 최대 관절 속도 ≤ speed
        v = p.v_of_t(t)
        self.assertLessEqual(np.abs(np.diff(v) / np.diff(t)).max(), 60 + 1e-6)
        self.assertTrue(np.all(np.diff(p.times) >= 0))
        # 지령점 m 은 이상 도달 시각에 정확히 그 자세
        qm, _ = p.joints(p.times)
        np.testing.assert_allclose(qm, q, atol=1e-7)

    def test_l2_basis(self):
        p = SpeedProfile(line(10, [30, 40, 0, 0, 0, 0]), 10, 1000, basis="l2")
        self.assertAlmostEqual(p.L, 50.0)
        _, v = p.joints(np.array([p.T / 2]))
        self.assertAlmostEqual(float(np.linalg.norm(v[0])), 10.0, places=6)

    def test_schedule_respects_min_period_and_keeps_last_point(self):
        q = line(200, [100, 0, 0, 0, 0, 0])  # 0.5 deg 간격
        p = SpeedProfile(q, 50, 200)
        idx, send_t, t1 = p.schedule(0.05)
        self.assertEqual(idx[-1], len(q) - 1)
        self.assertTrue(np.all(t1[:-1] >= 0.05 - 1e-12))
        self.assertTrue(np.all(t1 > 0))
        np.testing.assert_allclose(send_t[1:], send_t[:-1] + t1[:-1])
        self.assertAlmostEqual(send_t[-1] + t1[-1], p.T)

    def test_repeated_points_are_not_sent(self):
        q = np.vstack([line(3, [9, 0, 0, 0, 0, 0]), [[9, 0, 0, 0, 0, 0]], line(3, [9, 0, 0, 0, 0, 0])[::-1][1:]])
        p = SpeedProfile(q, 10, 100)
        idx, _, t1 = p.schedule(0.0)
        self.assertNotIn(4, idx)
        self.assertTrue(np.all(t1 > 0))

    def test_zero_length_path_rejected(self):
        with self.assertRaises(ValueError):
            SpeedProfile(np.zeros((5, 6)), 10, 10)



class PathProfileTests(unittest.TestCase):
    """원본 궤적 위 관절 속도·가속도 한계 프로파일 (servo profile 타이밍)."""

    def test_straight_line_is_trapezoid(self):
        p = PathProfile(line(600, [120, 60, 0, 0, 0, 0]), 50, 100)  # 직선: 사다리꼴 T = 120/50 + 0.5
        self.assertAlmostEqual(p.T, 2.9, places=3)
        self.assertAlmostEqual(p.vp, 50.0, places=6)

    def test_joint_speed_and_acceleration_limits_on_curved_path(self):
        th = np.linspace(0, 2 * np.pi, 800)
        q = np.column_stack([40 * np.sin(th), 30 * (1 - np.cos(2 * th)), 10 * th, np.zeros_like(th), 20 * np.sin(3 * th), np.zeros_like(th)])
        p = PathProfile(q, 30, 60)
        t = np.linspace(0, p.T, 20001)
        qt, vt = p.joints(t)
        np.testing.assert_allclose(qt[0], q[0], atol=1e-9)
        np.testing.assert_allclose(qt[-1], q[-1], atol=1e-9)
        self.assertLessEqual(np.abs(vt).max(), 30 * 1.001)
        acc = np.gradient(p.qd, p.t, axis=0)
        self.assertLessEqual(np.abs(acc).max(), 60 * 1.1)   # 격자 이산화 오차 허용

    def test_reversal_slows_to_zero(self):
        # 시간 매개 스플라인처럼 반전점에 가까울수록 샘플이 촘촘한 왕복 (관절 1 이 갔다 돌아옴, 관절 2 는 조금 움직임)
        u = np.linspace(0, 1, 401)
        q = np.zeros((len(u), 6))
        q[:, 0] = 30 * np.sin(np.pi * u)
        q[:, 1] = 2 * u
        p = PathProfile(q, 50, 100)
        k = int(np.argmax(np.abs(p.q[:, 0])))
        self.assertLess(p.v[k], 0.15 * p.vp)   # 반전점에서 크게 감속 (격자 간격 안의 반전만 허용)
        self.assertGreater(p.vp, 30.0)
        acc = np.gradient(p.qd, p.t, axis=0)
        self.assertLessEqual(np.abs(acc).max(), 100 * 1.1)

    def test_schedule_uniform_period_ending_at_T(self):
        p = PathProfile(line(300, [60, 20, 0, 0, 0, 0]), 30, 60)
        q, send_t, t1 = p.schedule(0.02)
        self.assertTrue(np.all(t1 <= 0.02 + 1e-12))
        self.assertTrue(np.allclose(t1, t1[0]))
        self.assertAlmostEqual(send_t[-1] + t1[-1], p.T)
        np.testing.assert_allclose(q[-1], [60, 20, 0, 0, 0, 0], atol=1e-9)
        with self.assertRaises(ValueError):
            p.schedule(0.5)

    def test_l2_limits(self):
        p = PathProfile(line(400, [30, 40, 0, 0, 0, 0]), 10, 20, basis="l2")
        self.assertAlmostEqual(p.L, 50.0)
        self.assertAlmostEqual(p.T, 50 / 10 + 10 / 20, places=3)


class CommandLimitTests(unittest.TestCase):
    """실제로 보낼 지령(주기 샘플)의 차분 속도·가속도가 설정 이하인지."""

    def _curved(self):
        th = np.linspace(0, 2 * np.pi, 400)
        return np.column_stack([40 * np.sin(th), 30 * (1 - np.cos(2 * th)), 10 * th, np.zeros_like(th), 20 * np.sin(3 * th), np.zeros_like(th)])

    def test_fit_keeps_commands_within_limits(self):
        for basis in ("max_joint", "l2"):
            for V, A, P in ((30, 60, .02), (50, 100, .02), (20, 200, .05)):
                p = PathProfile.fit(self._curved(), V, A, basis, P)
                self.assertLessEqual(p.cmd_speed_max, V * (1 + 1e-6), (basis, V, A, P))
                self.assertLessEqual(p.cmd_accel_max, A * (1 + 1e-6), (basis, V, A, P))
                vmax, amax = p.command_limits(P)
                self.assertAlmostEqual(vmax, p.cmd_speed_max)

    def test_fit_does_not_slow_down_straight_line_much(self):
        p = PathProfile.fit(line(600, [120, 60, 0, 0, 0, 0]), 50, 100, "max_joint", .02)
        self.assertLess(p.T, 2.9 * 1.05)


if __name__ == "__main__":
    unittest.main()
