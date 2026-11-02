import csv, json, tempfile, unittest
from pathlib import Path
import numpy as np
import timing_plot as tp


def make_run(d, method='servo', wall=True, period=0.02, n=100, lag=0.003, jitter=0.0003, seed=0):
    """합성 실행 폴더. 송신은 period 간격(+지터), 응답은 송신 + lag, 모션은 첫 송신 + 0.1 s 뒤 시작."""
    rng = np.random.default_rng(seed)
    base = 1_800_000_000.0
    d = Path(d)
    json.dump(dict(method=method, mode='real'), (d / 'config.json').open('w'))
    json.dump([dict(file='follow.csv', label='x', completed=True)], (d / 'manifest.json').open('w'))
    send = base + 5 + np.arange(n) * period + rng.normal(0, jitter, n)
    resp = [dict(time=base + 1, stage='initial', response='{ "type": ACK }')]
    if method == 'servo':
        for k in range(n):
            resp.append(dict(time=float(send[k] + lag), stage=f'servo[{k}]', response='{ "type": ACK, "category": "", "msg": "" }'))
        resp.append(dict(time=float(send[0] + 0.1), stage='servo[5]', response='{ "type": Info, "category": "motion_changed", "msg": "7100" }'))
        sched = np.c_[np.arange(n) * period, send - send[0], np.full(n, period), np.zeros((n, 6)), send, send + lag]
        np.savetxt(d / 'servo_schedule.csv', sched if wall else sched[:, :9], delimiter=',', header='planned_send_s,actual_send_s,t1,J1,J2,J3,J4,J5,J6' + (',wall_call,wall_return' if wall else ''), comments='')
    json.dump(resp, (d / 'responses.json').open('w'))
    # 모션: 첫 송신 0.1 s 뒤부터 J1 이 10 deg/s 로 움직임, 100 Hz 기록
    t = np.arange(0, 4, 0.01)
    t_start = send[0] + 0.1
    wall_t = send[0] - 0.5 + t              # 기록 시작이 첫 송신 0.5 s 전
    q = np.zeros((len(t), 6)); q[:, 0] = np.clip(wall_t - t_start, 0, None) * 10
    state = np.where(wall_t >= t_start, 3, 1)
    hdr = ['host_time', 'controller_time', 'state', 'sim', 'speed_bar'] + [f'ref{j}' for j in range(6)] + [f'ang{j}' for j in range(6)] + (['wall_time'] if wall else [])
    with (d / 'follow.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(hdr)
        for i in range(len(t)):
            w.writerow([t[i], 0, state[i], 0, 1, *[0] * 6, *q[i], *([wall_t[i]] if wall else [])])
    return send, t_start


class TimingTests(unittest.TestCase):
    def test_servo_exact_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            send, t_start = make_run(tmp, wall=True, lag=0.003)
            d = tp.build_data(tmp)
            s = tp.summary(d)
            self.assertEqual(d['aligned'], 'exact')
            self.assertAlmostEqual(s['send_interval_ms'][0], 20.0, places=1)
            self.assertAlmostEqual(s['ack_latency_ms'][0], 3.0, places=3)                 # 응답 − 송신 = lag
            self.assertAlmostEqual(s['motion_start_after_first_send_ms'], 100.0, delta=30)  # 모션 시작 ≈ 첫 송신 + 0.1 s
            self.assertAlmostEqual(d['t0'], send[0])

    def test_servo_old_format_is_estimated_and_hides_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_run(tmp, wall=False, lag=0.003)
            d = tp.build_data(tmp); s = tp.summary(d)
            self.assertEqual(d['aligned'], 'estimated')
            self.assertNotIn('ack_latency_ms', s); self.assertNotIn('motion_start_after_first_send_ms', s)
            self.assertAlmostEqual(s['ack_interval_ms'][0], 20.0, places=1)

    def test_outputs_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_run(tmp, wall=True)
            s = tp.save(tmp)
            for name in ('timing_compare.png', 'timing_zoom.png', 'timing_compare.csv'):
                self.assertTrue((Path(tmp) / name).stat().st_size > 0, name)
            rows = list(csv.DictReader((Path(tmp) / 'timing_compare.csv').open()))
            self.assertEqual(len(rows), 100)
            self.assertAlmostEqual(float(rows[10]['send_interval_ms']), 20.0, delta=2.0)
            self.assertAlmostEqual(float(rows[10]['ack_latency_ms']), 3.0, places=2)
            self.assertTrue(tp.describe(s))

    def test_jb2_last_info_at_end_is_end_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            json.dump(dict(method='jb2', mode='real'), (d / 'config.json').open('w'))
            json.dump([dict(file='follow.csv', label='x', completed=True)], (d / 'manifest.json').open('w'))
            base = 1_800_000_000.0
            info = lambda t, m, st='wait_final': dict(time=base + t, stage=st, response='{ "type": Info, "category": "motion_changed", "msg": "%s" }' % m)
            json.dump([info(0.3, '1200', 'move_to_start'), info(1.0, '900'), info(4.0, '901'), info(7.0, '902'), info(9.0, '903'), info(9.0, '0')], (d / 'responses.json').open('w'))
            t = np.arange(0, 10, 0.01); q = np.zeros((len(t), 6)); q[:, 0] = np.clip(t - 1.0, 0, 8) * 5
            with (d / 'follow.csv').open('w', newline='') as f:
                w = csv.writer(f); w.writerow(['host_time', 'controller_time', 'state', 'sim', 'speed_bar'] + [f'ref{j}' for j in range(6)] + [f'ang{j}' for j in range(6)])
                for i in range(len(t)):
                    w.writerow([t[i], 0, 3 if t[i] >= 1.0 else 1, 0, 1, *[0] * 6, *q[i]])
            dd = tp.build_data(tmp); s = tp.summary(dd)
            self.assertEqual(len(dd['seg_starts']), 3)                     # 900, 901, 902 만 구간 시작, 903 은 종료와 같은 시각이라 종료 표시
            np.testing.assert_allclose(s['segment_durations_s'], [3, 3, 2], atol=1e-6)
            tp.save(tmp)


if __name__ == '__main__':
    unittest.main()
