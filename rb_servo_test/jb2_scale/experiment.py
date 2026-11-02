#!/usr/bin/env python3
"""Standalone jb2 joint-speed scaling experiment; default is offline plan only."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rb_servo_gui/src'))
import numpy as np
from rb_servo_gui.runner import Runner
from rb_servo_gui.settings import Settings
from rb_servo_gui.trajectory import save_interp
from rb_servo_gui.analysis import load_follow, velocity

CASES = {'single': [1,0,0,0,0,0], 'equal': [1,1,0,0,0,0],
         'unequal': [1,.5,0,0,0,0], 'swapped': [.5,1,0,0,0,0]}


def predictions(delta, speed):
    d = np.abs(np.asarray(delta, dtype=float))
    return {'max_joint': (speed*d/d.max()).tolist(),
            'joint_l2': (speed*d/np.linalg.norm(d)).tolist()}


def summarize(path, delta, speed, window):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t, phase, q, state, meta = load_follow(str(path))
    start = meta['marks']['jb2']
    end = meta['marks'].get('jb2_done')
    if end is None:
        raise RuntimeError('jb2 완료 기록이 없어 유효한 실험으로 처리할 수 없습니다')
    keep = (t >= start) & (t <= end) & np.char.startswith(phase.astype(str), 'jb2')
    t, q = t[keep]-start, q[keep]
    if len(t) < 5 or np.any(np.diff(t) <= 0):
        raise ValueError('속도 분석에 필요한 유효 샘플 부족')
    v = velocity(t, q, win=window)
    raw = np.gradient(q, t, axis=0)
    linf = np.max(np.abs(v),axis=1)
    peak = int(np.argmax(linf))
    near = linf >= .95*linf[peak]
    plateau = float(np.sum(np.diff(t)* (near[:-1] & near[1:])))
    result = {'speed_command_deg_s':speed, 'delta_deg':list(delta),
              'source':meta['measured_source'], 'velocity_window_s':window,
              'joint_peak_deg_s':np.max(np.abs(v),axis=0).tolist(),
              'raw_joint_peak_deg_s':np.max(np.abs(raw),axis=0).tolist(),
              'joint_velocity_at_linf_peak':v[peak].tolist(),
              'linf_peak_deg_s':float(linf[peak]),
              'l2_peak_deg_s':float(np.linalg.norm(v,axis=1).max()),
              'near_peak_duration_s':plateau,
              'plateau_resolved':bool(plateau >= max(.3, 2*window)),
              'predictions':predictions(delta,speed)}
    (path.parent/'speed_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savetxt(path.parent/'speed_trace.csv', np.c_[t,q,v,raw],delimiter=',',
               header=','.join(['time_s']+[f'q{j+1}_deg' for j in range(6)]+[f'v{j+1}_deg_s' for j in range(6)]+[f'raw_v{j+1}_deg_s' for j in range(6)]),comments='')
    fig, ax = plt.subplots(2,1,figsize=(11,7),sharex=True)
    for j in range(6):
        ax[0].plot(t,q[:,j]-q[0,j],label=f'J{j+1}')
        ax[1].plot(t,np.abs(v[:,j]),label=f'J{j+1}')
    ax[1].plot(t,np.linalg.norm(v,axis=1),'k--',label='L2 speed')
    ax[1].axhline(speed,color='red',ls=':',label='Command speed')
    ax[0].set_ylabel('Displacement [deg]');ax[1].set_ylabel('Speed magnitude [deg/s]');ax[1].set_xlabel('Time since jb2_run [s]')
    for a in ax:a.grid(alpha=.3);a.legend(ncol=4)
    fig.tight_layout();fig.savefig(path.parent/'speed.png',dpi=140);plt.close(fig)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=float,nargs=6,required=True,metavar='DEG',help='검토한 시작 자세 J1..J6 [deg]')
    p.add_argument('--amplitude',type=float,default=30)
    p.add_argument('--speed',type=float,default=10)
    p.add_argument('--accel',type=float,default=20)
    p.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    p.add_argument('--scales',nargs=6,type=float,help='기본 4개 대신 사용할 사용자 지정 비율')
    p.add_argument('--ip',default='10.0.2.7')
    p.add_argument('--execute',action='store_true',help='없으면 계획만 저장, 연결하지 않음')
    p.add_argument('--real',action='store_true',help='실기 모드로 실행')
    p.add_argument('--real-ack',action='store_true',help='검토한 모션의 실기 실행 확인')
    p.add_argument('--mock',action='store_true',help='통신 검증용 목업, 속도 규칙 검증에는 사용 불가')
    p.add_argument('--record-hz',type=float,default=100)
    p.add_argument('--velocity-window',type=float,default=.1)
    p.add_argument('--out',type=Path,default=Path('rb_servo_test/jb2_scale/out'))
    a=p.parse_args()
    numeric=[*a.base,a.amplitude,a.speed,a.accel,a.record_hz,a.velocity_window]
    if not np.all(np.isfinite(numeric)) or min(a.amplitude,a.speed,a.accel,a.record_hz,a.velocity_window)<=0:p.error('유한한 양수 설정 필요')
    if a.real and not a.real_ack:p.error('--real은 --real-ack 필요')
    if a.real and a.mock:p.error('--real과 --mock은 함께 사용할 수 없음')
    if a.real and (a.speed>30 or a.accel>60):p.error('실기 기본 상한: speed <=30, accel <=60 (자동 보정으로 인한 실험 혼동 방지)')
    cases={'custom':a.scales} if a.scales is not None else {k:CASES[k] for k in a.cases}
    for scales in cases.values():
        if not np.all(np.isfinite(scales)) or not np.any(scales):p.error('scales는 유한한 비영 벡터 필요')
    root=a.out/datetime.now().strftime('%Y%m%d_%H%M%S_%f');root.mkdir(parents=True)
    plan=[]
    for name,scales in cases.items():
        delta=a.amplitude*np.asarray(scales); target=np.asarray(a.base)+delta
        plan.append(dict(name=name,base=a.base,target=target.tolist(),delta=delta.tolist(),predictions=predictions(delta,a.speed)))
    (root/'plan.json').write_text(json.dumps({'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'cases':plan},indent=2))
    for case in plan:print(case['name'],'delta=',case['delta'],'target=',case['target'],'expected=',case['predictions'])
    print('계획:',root/'plan.json')
    if not a.execute:return
    mock=None
    try:
        if a.mock:
            from rb_servo_gui.mock_robot import MockRobot
            mock=MockRobot();mock.start()
            print('MOCK: 결과는 실행/분석 파이프라인 검증용이며 제어기 속도 규칙의 증거가 아닙니다')
        rows=[]
        for case in plan:
            folder=root/case['name'];folder.mkdir()
            trajectory=folder/'targets.csv'
            save_interp(trajectory,[0,1],np.deg2rad([case['base'],case['target']]))
            s=Settings(traj_path=str(trajectory),traj_kind='interp',mode='jb2',step=1,
                       speed=a.speed,accel=a.accel,blend=0,ip=a.ip,use_mock=a.mock,
                       robot_mode='real' if a.real else 'simulation',real_ack=a.real_ack,
                       record_hz=a.record_hz,out_dir=str(folder))
            run=Path(Runner(s).run(analyze_results=False))
            result=summarize(next(run.glob('follow_*.csv')),case['delta'],a.speed,a.velocity_window)
            row={'case':case['name'],'linf_peak':result['linf_peak_deg_s'],'l2_peak':result['l2_peak_deg_s'],
                 'near_peak_s':result['near_peak_duration_s'],'plateau_resolved':result['plateau_resolved']}
            row.update({f'J{j+1}_peak':v for j,v in enumerate(result['joint_peak_deg_s'])});rows.append(row)
            with (root/'summary.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
            print(case['name'],row)
    finally:
        if mock:mock.stop()
    print('결과:',root/'summary.csv')

if __name__=='__main__':main()
