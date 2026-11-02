import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from rb_servo_gui.trajectory import subsample, prepare
from rb_servo_gui.settings import Settings
from rb_servo_gui.runner import Runner


class SamplingTests(unittest.TestCase):
    def test_each_joint_turn_and_plateau_boundaries(self):
        q = np.zeros((21, 6))
        q[:, 0] = -(np.arange(21)-5)**2
        q[:, 1] = (np.arange(21)-7)**2
        q[:, 2] = [0, 1, 2, 3, 4, 4, 4, 3] + list(range(2, -11, -1))
        t, out = subsample(np.arange(21), q, 10)
        self.assertTrue({0, 4, 5, 6, 7, 10, 20}.issubset(set(t)))
        np.testing.assert_array_equal(out, q[t])
        self.assertTrue(np.all(np.diff(t) > 0))

    def test_monotonic_constant_and_step_one(self):
        q = np.column_stack([np.arange(23), np.zeros(23)])
        t, _ = subsample(np.arange(23), q, 10)
        np.testing.assert_array_equal(t, [0, 10, 20, 22])
        t, out = subsample(np.arange(23), q, 1)
        np.testing.assert_array_equal(out, q)

    def test_flat_maximum_and_minimum(self):
        q = np.array([0, 1, 2, 2, 2, 1, 0, -1, -1, 0])[:, None]
        t, _ = subsample(np.arange(len(q)), q, 100)
        np.testing.assert_array_equal(t, [0, 2, 4, 7, 8, 9])

    def test_invalid_input(self):
        for step in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                subsample([0, 1], [[0], [1]], step)
        with self.assertRaises(ValueError):
            subsample([0, 0], [[0], [1]], 1)

    def test_sample_file_all_joint_extrema_retained(self):
        path = Path(__file__).resolve().parents[1] / 'examples/waypoints_robot1.csv'
        t, q = prepare(Settings(traj_path=str(path)))
        selected, _ = subsample(t, q, 10)
        d = np.diff(q, axis=0)
        for j in range(q.shape[1]):
            turning = np.flatnonzero(d[:-1,j]*d[1:,j] < 0)+1
            self.assertTrue(set(t[turning]).issubset(set(selected)))

    def test_servo_uses_nonuniform_intervals_and_explicit_parameters(self):
        for explicit in (False, True):
            s = Settings(servo_timing='original', final_hold=1, t1=.05 if explicit else 0., t2=.07 if explicit else 0., final_t2=.03)
            runner = Runner(s)
            runner._send = Mock()
            runner._finish = Mock(return_value=True)
            cmd, rec, dp = Mock(denied=False), Mock(), Mock()
            now = [0.]
            sent = []
            cmd.send.side_effect = lambda line: sent.append((now[0], line))
            def sleep(dt):
                now[0] += dt
            with patch('rb_servo_gui.runner.time.perf_counter', side_effect=lambda: now[0]), patch('rb_servo_gui.runner.time.sleep', side_effect=sleep):
                runner._run_servo(cmd, rec, dp, np.zeros((4,6)), np.array([0., .1, .14, .3]))
            np.testing.assert_allclose([t for t,_ in sent], [0., .1, .14, .3])
            self.assertAlmostEqual(now[0], .46)
            expected_t1 = [.05]*4 if explicit else [.1, .04, .16, .16]
            expected_t2 = [.07, .07, .03, .03] if explicit else [.1, .04, .03, .03]
            for (_, line), t1, t2 in zip(sent, expected_t1, expected_t2):
                self.assertIn(f'], {t1:g}, {t2:g},', line)


    def test_servo_profile_timing_samples_profile_at_fixed_period(self):
        from rb_servo_gui.trajectory import PathProfile
        q = np.outer(np.linspace(0, 1, 41), [40, 20, 0, 0, 0, 0])  # 직선
        s = Settings(servo_timing='profile', speed=20., accel=40., servo_period=.02, final_hold=2)
        runner = Runner(s)
        runner._send = Mock()
        runner._finish = Mock(return_value=True)
        cmd, rec, dp = Mock(denied=False), Mock(), Mock()
        now = [0.]
        sent = []
        cmd.send.side_effect = lambda line: sent.append((now[0], line))
        with patch('rb_servo_gui.runner.time.perf_counter', side_effect=lambda: now[0]), \
                patch('rb_servo_gui.runner.time.sleep', side_effect=lambda dt: now.__setitem__(0, now[0] + dt)):
            runner._run_servo(cmd, rec, dp, q, np.arange(41) * .1)
        prof = PathProfile.fit(q, 20., 40., 'max_joint', .02)
        targets, send_t, t1 = prof.schedule(.02)
        self.assertEqual(len(sent), len(targets) + 2)
        np.testing.assert_allclose([t for t, _ in sent[:len(targets)]], send_t)
        d = float(t1[0])
        for _, line in sent:  # 균일 주기, t2 = 주기 × 2
            self.assertIn(f'], {d:g}, {min(max(2 * d, .02), .2):g},', line)
        self.assertAlmostEqual(send_t[-1] + t1[-1], prof.T)
        self.assertEqual(len(runner.servo_log), len(sent))


if __name__ == '__main__':
    unittest.main()
