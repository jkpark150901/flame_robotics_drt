"""명령 전송 시각(TCP) ↔ 제어기 응답 ↔ 실제 모션을 한 시간축에 겹쳐 그린다.

    python timing_plot.py out/<실행 폴더> [--zoom 시작[s] 길이[s]]

만드는 파일(실행 폴더 안): timing_compare.png (전체), timing_zoom.png (모션 시작 부근 확대), timing_compare.csv

시간축은 벽시계(time.time) 기준이고 0 = 모션 명령의 첫 송신.
- 새 형식 폴더(follow.csv 에 wall_time 열, servo_schedule.csv 에 wall_call 열): 세 기록이 같은 시계라 그대로 겹친다.
- 이전 형식 폴더: 정렬을 추정한다. 송신 시각 = 응답(ACK) 도착 시각 − 송신 때 잰 상대 시각의 중앙값(응답 지연이 포함됨),
  모션 기록 = robot_state 가 처음 3 이 된 시각을 제어기 Info(motion_changed) 응답 시각에 맞춤. 오차는 수 ms ~ 20 ms 이며 그래프 제목에 "추정 정렬"로 표시한다.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _setup_font():
    import matplotlib
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Noto Sans CJK KR', 'Noto Sans CJK JP', 'NanumGothic', 'Malgun Gothic', 'Apple SD Gothic Neo'):
        if name in have:
            matplotlib.rcParams['font.sans-serif'] = [name] + list(matplotlib.rcParams['font.sans-serif'])
            break
    matplotlib.rcParams['axes.unicode_minus'] = False


_setup_font()


def _info(resp):
    """Info/motion_changed 응답의 msg 값 (없으면 None)."""
    if 'motion_changed' not in resp:
        return None
    return resp.split('"msg":')[-1].strip(' "}')


def load(out):
    out = Path(out)
    cfg = json.loads((out / 'config.json').read_text())
    with (out / 'follow.csv').open() as f:
        header = f.readline().strip().split(',')
    a = np.genfromtxt(out / 'follow.csv', delimiter=',', skip_header=1, ndmin=2)
    follow = dict(t=a[:, 0], state=a[:, 2], q=a[:, 5:11] if cfg['mode'] == 'simulation' else a[:, 11:17],
                  wall=a[:, header.index('wall_time')] if 'wall_time' in header else None)
    resp = json.loads((out / 'responses.json').read_text()) if (out / 'responses.json').exists() else []
    sched = None
    if (out / 'servo_schedule.csv').exists():
        s = np.loadtxt(out / 'servo_schedule.csv', delimiter=',', skiprows=1, ndmin=2)
        sched = dict(planned=s[:, 0], rel=s[:, 1], t1=s[:, 2], wall_call=s[:, 9] if s.shape[1] >= 11 else None,
                     wall_return=s[:, 10] if s.shape[1] >= 11 else None)
    jb2 = None
    if (out / 'jb2_schedule.csv').exists():
        with (out / 'jb2_schedule.csv').open() as f:
            jb2 = [dict(kind=r['kind'], index=int(r['index']), call=float(r['wall_call']), ret=float(r['wall_return'])) for r in csv.DictReader(f)]
    return cfg, follow, resp, sched, jb2


def build_data(out):
    """분석에 쓸 값을 모아 dict 로 돌려준다 (그림과 CSV 의 공통 입력). 시간은 벽시계 [s]."""
    cfg, follow, resp, sched, jb2 = load(out)
    d = dict(method=cfg['method'], mode=cfg['mode'], aligned='exact', notes=[])
    acks = {}
    for r in resp:
        if 'ACK' in r['response']:
            acks.setdefault(r['stage'], []).append(r['time'])
    infos = [dict(time=r['time'], stage=r['stage'], msg=_info(r['response'])) for r in resp if _info(r['response']) is not None]
    d['infos'] = infos
    if cfg['method'] == 'servo':
        if sched is None:
            raise ValueError('servo_schedule.csv 가 없습니다')
        n = len(sched['rel'])
        ack = np.array([acks.get(f'servo[{k}]', [np.nan])[0] for k in range(n)])
        if sched['wall_call'] is not None:
            send = sched['wall_call']
            d['ret'] = sched['wall_return']
        else:
            ok = ~np.isnan(ack)
            send = sched['rel'] + np.median(ack[ok] - sched['rel'][ok])    # 응답 지연이 포함된 추정
            d['aligned'] = 'estimated'
            d['notes'].append('송신 시각은 응답 도착 시각에서 역산한 추정값 (응답 지연 포함)')
            d['ret'] = None
        d.update(send=send, ack=ack, planned=sched['planned'])
        t_first = float(send[0])
    else:
        if jb2 is None:
            # 이전 형식: 송신 기록 없음. 모션 명령은 마지막 'run' 이전이라 응답/Info 시각만 있다
            run_call = None
        else:
            run_call = next((e['call'] for e in jb2 if e['kind'] == 'run'), None)
            d['adds'] = [e for e in jb2 if e['kind'] == 'add']
        d['run_call'] = run_call
        # 구간 시작 Info (msg 900, 901, ...) 만 골라 지령점별 모션 시작 시각으로 본다
        d['seg_starts'] = [(i['time'], i['msg']) for i in infos if i['msg'] not in (None, '0') and i['stage'] not in ('move_to_start', 'initial')]
        d['seg_end'] = next((i['time'] for i in reversed(infos) if i['msg'] == '0' and i['stage'] not in ('move_to_start', 'initial')), None)
        if d['seg_end'] is not None and d['seg_starts'] and abs(d['seg_starts'][-1][0] - d['seg_end']) < 0.05:
            d['seg_starts'] = d['seg_starts'][:-1]   # 마지막 번호(예: 904)는 종료와 같은 시각에 오는 종료 표시
        t_first = run_call if run_call is not None else (d['seg_starts'][0][0] if d['seg_starts'] else float('nan'))
        if run_call is None:
            d['aligned'] = 'estimated'
            d['notes'].append('jb2 송신 기록이 없는 이전 형식: 0 = 첫 구간 시작 Info 응답')
    # 모션(측정) 시계
    if follow['wall'] is not None:
        wall = follow['wall']
    else:
        i3 = int(np.argmax(follow['state'] == 3)) if (follow['state'] == 3).any() else 0
        start_info = (d['seg_starts'][0][0] if cfg['method'] == 'jb2' and d['seg_starts'] else
                      next((i['time'] for i in infos if i['msg'] not in (None, '0') and i['stage'].startswith('servo')), None))
        if start_info is None:
            raise ValueError('모션 기록을 시계에 맞출 기준(Info 응답)이 없습니다')
        wall = follow['t'] - follow['t'][i3] + start_info
        d['aligned'] = 'estimated'
        d['notes'].append('모션 기록은 robot_state 가 3 이 된 시각을 제어기 Info 응답 시각에 맞춘 추정 정렬 (오차 약 ±20 ms)')
    d['wall'] = wall
    d['q'] = follow['q']
    dt = np.gradient(wall)
    v = np.gradient(follow['q'], wall, axis=0)
    k = 5   # 약 ±0.05 s 이동 평균 (100 Hz 기준)
    ker = np.ones(k) / k
    d['v'] = np.column_stack([np.convolve(v[:, j], ker, mode='same') for j in range(6)])
    d['t0'] = t_first
    d['cfg'] = cfg
    return d


def summary(d):
    """텍스트 요약 (로그/제목용)과 수치 dict."""
    s = dict(aligned=d['aligned'])
    if d['method'] == 'servo':
        si = np.diff(d['send']) * 1e3
        s['send_interval_ms'] = (float(si.mean()), float(si.std()), float(si.min()), float(si.max()))
        ok = ~np.isnan(d['ack'])
        ai = np.diff(d['ack'][ok]) * 1e3
        s['ack_interval_ms'] = (float(ai.mean()), float(ai.std()), float(ai.min()), float(ai.max()))
        if d['aligned'] == 'exact':   # 추정 정렬에서는 두 값이 정렬 방법에 의해 정해지므로 의미가 없다
            lat = (d['ack'] - d['send']) * 1e3
            s['ack_latency_ms'] = (float(np.nanmean(lat)), float(np.nanstd(lat)), float(np.nanmin(lat)), float(np.nanmax(lat)))
            mv = np.abs(d['q'] - d['q'][0]).max(1) > 0.05
            i0 = int(np.argmax(mv & (d['wall'] >= d['send'][0]))) if (mv & (d['wall'] >= d['send'][0])).any() else None
            if i0 is not None:
                s['motion_start_after_first_send_ms'] = float((d['wall'][i0] - d['send'][0]) * 1e3)
    else:
        if d['seg_starts']:
            t = np.array([x[0] for x in d['seg_starts']] + ([d['seg_end']] if d['seg_end'] else []))
            s['segment_starts_s'] = [float(x - t[0]) for x in t]
            s['segment_durations_s'] = [float(x) for x in np.diff(t)]
        if d.get('adds'):
            call = np.array([e['call'] for e in d['adds']]); ret = np.array([e['ret'] for e in d['adds']])
            s['add_call_ms'] = (float(np.mean((ret - call) * 1e3)), float(np.max((ret - call) * 1e3)))
            s['add_total_s'] = float(ret[-1] - call[0])
    return s


def figure(d, zoom=None):
    """zoom=(시작, 길이) [s, t0 기준] 이면 그 구간만 확대."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(13, 8.5))
    ax = fig.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1.3, 2.2]})
    t0 = d['t0']
    tw = d['wall'] - t0
    speed = np.abs(d['v']).max(1)
    ax[0].plot(tw, speed, color='k', lw=1.3, label='측정 최대 관절 속도 |v|')
    for j in range(6):
        ax[0].plot(tw, np.abs(d['v'][:, j]), lw=0.7, alpha=0.5, color=f'C{j}')
    ax[0].set_ylabel('관절 속도 [deg/s]')
    summ = summary(d)
    title = f"{d['method']} / {d['mode']} — 전송·응답·모션 타이밍" + ('  (추정 정렬)' if d['aligned'] == 'estimated' else '')
    ax[0].set_title(title, fontsize=11)
    strip_names, strip_y = [], []
    if d['method'] == 'servo':
        send, ack = d['send'] - t0, d['ack'] - t0
        ax[1].plot(send, np.full(len(send), 2), '|', color='C0', ms=11, mew=1.2)
        ax[1].plot(ack[~np.isnan(ack)], np.full((~np.isnan(ack)).sum(), 1), '|', color='C1', ms=11, mew=1.2)
        strip_names, strip_y = ['제어기 응답(ACK) 도착', 'PC 송신(move_servo_j)'], [1, 2]
        if len(send) > 200 and zoom is None:
            ax[1].text(0.5, 0.5, '송신이 너무 빽빽해 줄무늬로 보임 → timing_zoom.png (모션 시작 1 s 확대)', transform=ax[1].transAxes, ha='center', va='center', fontsize=9,
                       bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))
        si = np.diff(send) * 1e3
        ai = np.diff(ack[~np.isnan(ack)]) * 1e3
        ax[2].plot(send[1:], si, '.', ms=3, color='C0', label=f"송신 간격 (평균 {si.mean():.2f}, 표준편차 {si.std():.2f} ms)")
        ax[2].plot(ack[~np.isnan(ack)][1:], ai, '.', ms=3, color='C1', label=f"응답 도착 간격 (평균 {ai.mean():.2f}, 표준편차 {ai.std():.2f} ms)")
        ax[2].set_ylabel('간격 [ms]')
        if d['aligned'] == 'exact':
            lat = (d['ack'] - d['send']) * 1e3
            ax2 = ax[2].twinx()
            ax2.plot(send, lat, '-', lw=0.8, color='C2', alpha=0.8, label=f"응답 지연 = 응답 도착 − 송신 (평균 {np.nanmean(lat):.1f}, 최대 {np.nanmax(lat):.1f} ms)")
            ax2.set_ylabel('응답 지연 [ms]', color='C2')
            h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax[2].legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
        else:
            ax[2].legend(fontsize=8, loc='upper right')
            ax[2].text(0.01, 0.04, '※ 이전 형식 기록: 송신 시각을 응답에서 역산했으므로 응답 지연은 표시하지 않음 (새 실행부터 기록)', transform=ax[2].transAxes, fontsize=8, color='gray')
        lo = float(np.nanmin(np.r_[si, ai])); hi = float(np.nanmax(np.r_[si, ai]))
        ax[2].set_ylim(max(lo - 2, 0), hi + 2)
    else:
        if d.get('adds'):
            call = np.array([e['call'] for e in d['adds']]) - t0
            ax[1].plot(call, np.full(len(call), 2), '|', color='C0', ms=11, mew=1.2)
            strip_names.append('PC 송신(move_jb2_add)'); strip_y.append(2)
        if d['run_call'] is not None:
            ax[1].axvline(0, color='C0', lw=1.5)
            ax[1].text(0, 2.55, 'move_jb2_run 송신', color='C0', fontsize=8)
        st = [(x - t0, m) for x, m in d['seg_starts']]
        ax[1].plot([x for x, _ in st], np.full(len(st), 1), 'D', color='C3', ms=7)
        for x, m in st:
            ax[1].text(x, 1.25, m, fontsize=7, ha='center')
        if d['seg_end']:
            ax[1].plot([d['seg_end'] - t0], [1], 's', color='k', ms=7)
        strip_names.append('제어기 Info(구간 시작 ◆ / 종료 ■)'); strip_y.append(1)
        for x, _ in st:
            for a in (ax[0], ax[2]):
                a.axvline(x, color='C3', lw=0.8, ls='--', alpha=0.6)
        if len(st) > 1:
            xs = np.array([x for x, _ in st] + ([d['seg_end'] - t0] if d['seg_end'] else []))
            ax[2].bar(xs[:-1] + np.diff(xs) / 2, np.diff(xs), width=np.diff(xs) * 0.9, color='C3', alpha=0.5)
            for x0, w in zip(xs[:-1] + np.diff(xs) / 2, np.diff(xs)):
                ax[2].text(x0, w, f'{w:.2f} s', ha='center', va='bottom', fontsize=8)
            ax[2].set_ylabel('구간 지속 시간 [s]\n(Info 응답 간격)')
        else:
            ax[2].text(0.5, 0.5, '구간 시작 Info 응답 없음', transform=ax[2].transAxes, ha='center')
    ax[1].set_yticks(strip_y); ax[1].set_yticklabels(strip_names, fontsize=8); ax[1].set_ylim(0.4, 2.9)
    ax[2].set_xlabel('시간 [s] (0 = 모션 명령의 첫 송신)')
    for a in ax:
        a.grid(alpha=0.25)
    if zoom is not None:
        ax[2].set_xlim(zoom[0], zoom[0] + zoom[1])
    ax[0].legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    return fig


def table(d):
    """timing_compare.csv 행 (servo: 송신마다 / jb2: 이벤트마다)."""
    t0 = d['t0']
    if d['method'] == 'servo':
        hdr = ['index', 'planned_s', 'send_s', 'ack_s', 'send_interval_ms', 'ack_interval_ms', 'ack_latency_ms', 'speed_at_send_deg_s']
        rows = []
        sp = np.abs(d['v']).max(1)
        for k in range(len(d['send'])):
            si = (d['send'][k] - d['send'][k - 1]) * 1e3 if k else ''
            ai = (d['ack'][k] - d['ack'][k - 1]) * 1e3 if k and not np.isnan(d['ack'][k]) and not np.isnan(d['ack'][k - 1]) else ''
            lat = (d['ack'][k] - d['send'][k]) * 1e3 if not np.isnan(d['ack'][k]) else ''
            sp_k = float(np.interp(d['send'][k], d['wall'], sp))
            rows.append([k, f"{d['planned'][k]:.4f}", f"{d['send'][k] - t0:.5f}", '' if np.isnan(d['ack'][k]) else f"{d['ack'][k] - t0:.5f}",
                         f'{si:.3f}' if si != '' else '', f'{ai:.3f}' if ai != '' else '', f'{lat:.3f}' if lat != '' else '', f'{sp_k:.2f}'])
        return hdr, rows
    hdr = ['event', 'index', 'time_s', 'detail']
    rows = []
    for e in d.get('adds', []):
        rows.append(['jb2_add_call', e['index'], f"{e['call'] - t0:.5f}", f"return_after_ms={(e['ret'] - e['call']) * 1e3:.2f}"])
    if d['run_call'] is not None:
        rows.append(['jb2_run_call', '', '0.00000', ''])
    for k, (x, m) in enumerate(d['seg_starts']):
        rows.append(['segment_start_info', k, f'{x - t0:.5f}', f'msg={m}'])
    if d['seg_end']:
        rows.append(['motion_end_info', '', f"{d['seg_end'] - t0:.5f}", 'msg=0'])
    return hdr, rows


def save(out, zoom=None):
    """PNG 2장 + CSV 를 실행 폴더에 저장하고 요약 dict 를 돌려준다."""
    out = Path(out)
    d = build_data(out)
    figure(d).savefig(out / 'timing_compare.png', dpi=120)
    if d['method'] == 'servo':
        z = zoom or (0.0, 1.0)
        figure(d, zoom=z).savefig(out / 'timing_zoom.png', dpi=120)
    hdr, rows = table(d)
    with (out / 'timing_compare.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    return summary(d)


def describe(s):
    """summary() 를 한 줄씩 읽기 쉬운 문장으로."""
    lines = []
    if 'send_interval_ms' in s:
        m, sd, lo, hi = s['send_interval_ms']
        lines.append(f'PC 송신 간격 {m:.2f}±{sd:.2f} ms ({lo:.1f}~{hi:.1f})')
        m, sd, lo, hi = s['ack_interval_ms']
        lines.append(f'제어기 응답 도착 간격 {m:.2f}±{sd:.2f} ms ({lo:.1f}~{hi:.1f})')
        if 'ack_latency_ms' in s:
            m, sd, lo, hi = s['ack_latency_ms']
            lines.append(f'응답 지연(응답−송신) 평균 {m:.1f} ms, 최대 {hi:.1f} ms')
        if 'motion_start_after_first_send_ms' in s:
            lines.append(f"첫 송신 후 모션 시작까지 {s['motion_start_after_first_send_ms']:.0f} ms")
    if 'segment_durations_s' in s:
        lines.append('jb2 구간 지속 시간 [s]: ' + ', '.join(f'{x:.2f}' for x in s['segment_durations_s']))
    if 'add_call_ms' in s:
        lines.append(f"move_jb2_add 호출 평균 {s['add_call_ms'][0]:.1f} ms (최대 {s['add_call_ms'][1]:.1f}), 전체 전송 {s['add_total_s']:.2f} s")
    if s['aligned'] == 'estimated':
        lines.append('※ 이전 형식 기록: 시계 정렬은 추정')
    return lines


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    zoom = None
    if '--zoom' in args:
        i = args.index('--zoom')
        zoom = (float(args[i + 1]), float(args[i + 2]))
        del args[i:i + 3]
    s = save(args[0], zoom)
    print('\n'.join(describe(s)))
    print('저장:', args[0], '→ timing_compare.png / timing_zoom.png / timing_compare.csv')
