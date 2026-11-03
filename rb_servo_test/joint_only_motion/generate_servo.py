"""Generate offline Servo J TXT and timing CSV. Standard library only; no network IO."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'external_script_gui'))
from motion import ServoProfile


def command(kind, joints, parameters):
    return f"{kind}(jnt[{','.join(f'{v:.12g}' for v in joints)}], {', '.join(f'{v:.12g}' for v in parameters)})"


def generate(name, spec, args, output):
    period = args.period_ms / 1000
    speed, accel = math.degrees(args.speed), math.degrees(args.accel)
    begin = tuple(float(v) for v in spec['begin'])
    points = [tuple(float(v) for v in q) for q in spec['points']]
    waits = [float(v) for v in spec['waits']]
    if not points or len(points) != len(waits):
        raise ValueError(f'{name}: points and waits must have the same nonzero length')
    if any(len(q) != 6 or not all(math.isfinite(v) for v in q) for q in [begin, *points]):
        raise ValueError(f'{name}: each pose requires six finite joint angles in degrees')
    if any(not math.isfinite(w) or w < 0 for w in waits):
        raise ValueError(f'{name}: waits must be finite and nonnegative')
    if begin != points[-1]:
        raise ValueError(f'{name}: final point must equal begin for repeatable cycle')
    rows, lines = [], [f'# period={period:.12g} s; t1=period; one loop body; see schedule.csv',
                       '# First execute setup.txt (Move J + move_finish_wait); await completion before starting the periodic schedule.']
    time_s = 0.
    trajectory, dwell_info = [], []
    current = begin
    vmax = amax = 0.
    for index, (point, wait) in enumerate(zip(points, waits), 1):
        targets = []
        if current != point:
            profile = ServoProfile([current, point], speed, accel, period)
            targets = [q for _, q in profile.samples()]
        # Quantize dwell upward to whole periods, keeping a uniform command grid.
        move_count = len(targets)
        dwell_count = max(0, math.ceil(wait / period - 1e-9))
        dwell_info.append(dict(waypoint=index, requested_wait_s=wait,
                               dwell_samples=dwell_count, actual_wait_s=dwell_count*period))
        # Two settling samples remain separate from the full requested dwell.
        targets.extend([point] * (2 + dwell_count))
        last_q, last_v = current, (0.,)*6
        for sample_index, q in enumerate(targets):
            phase = 'motion' if sample_index < move_count else ('settle' if sample_index < move_count+2 else 'dwell')
            velocity = tuple((b-a)/period for a,b in zip(last_q,q))
            acceleration = tuple((b-a)/period for a,b in zip(last_v,velocity))
            vmax = max(vmax, *map(abs, velocity))
            amax = max(amax, *map(abs, acceleration))
            cmd = command('move_servo_j', q, [period,args.t2,args.gain,args.alpha])
            rows.append([f'{time_s:.9f}', 'command', index, '', cmd, phase])
            trajectory.append([f'{time_s:.9f}', f'{time_s+period:.9f}', index, phase, *q])
            lines.append(cmd)
            time_s = len(rows) * period
            last_q, last_v = q, velocity
        current = point
    if vmax > speed + 1e-7 or amax > accel + 1e-6:
        raise ValueError(f'{name}: sampled velocity/acceleration exceeds configured limits')
    lines.append('# END LOOP: repeat this body from time zero after cycle_duration_s.')
    (output/f'{name}_servo_j.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (output/f'{name}_setup.txt').write_text(command('move_j',begin,[speed,accel,1])+'\nmove_finish_wait()\n', encoding='utf-8')
    with (output/f'{name}_schedule.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s','action','waypoint','wait_s','command','phase'])
        writer.writerows(rows)
    with (output/f'{name}_trajectory.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s','target_time_s','waypoint','phase',*[f'J{i}_deg' for i in range(1,7)]])
        writer.writerows(trajectory)
    with (output/f'{name}_waypoints.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['point',*[f'J{i}_deg' for i in range(1,7)],'wait_s'])
        writer.writerow(['begin',*begin,0])
        writer.writerows([i,*p,w] for i,(p,w) in enumerate(zip(points,waits),1))
    return dict(urdf=spec.get('urdf'), begin_deg=begin, waypoint_count=len(points),
                period_ms=args.period_ms, t1_s=period, t2_s=args.t2, gain=args.gain, alpha=args.alpha,
                speed_limit_rad_s=args.speed, accel_limit_rad_s2=args.accel,
                sampled_max_speed_rad_s=math.radians(vmax), sampled_max_accel_rad_s2=math.radians(amax),
                command_count=sum(r[1]=='command' for r in rows), cycle_duration_s=round(time_s,9),
                dwell_segments=dwell_info, settling_samples_per_waypoint=2,
                repeat_count=-1, setup='Execute setup once and wait for completion; initial approach is not timed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--period-ms',type=float,default=20,help='Servo interval in milliseconds (2..100); t1 follows it')
    parser.add_argument('--speed',type=float,default=.4,help='Joint speed limit [rad/s]')
    parser.add_argument('--accel',type=float,default=.1,help='Joint acceleration limit [rad/s²]')
    parser.add_argument('--t2',type=float,default=None,help='Servo t2 [s]; default=max(0.04, period)')
    parser.add_argument('--gain',type=float,default=.1)
    parser.add_argument('--alpha',type=float,default=.5)
    parser.add_argument('--robot',choices=['all','r1_RT','r2_DDA'],default='all')
    parser.add_argument('--motions',type=Path,default=HERE/'motions.json',help='Motion JSON; angles in degrees, waits in seconds')
    parser.add_argument('--output-dir',type=Path,default=None)
    args=parser.parse_args()
    if not math.isfinite(args.period_ms) or not 2 <= args.period_ms <= 100:
        parser.error('--period-ms must be finite and between 2 and 100')
    if any(not math.isfinite(v) or v<=0 for v in [args.speed,args.accel]):
        parser.error('--speed and --accel must be finite and positive')
    args.t2 = max(.04,args.period_ms/1000) if args.t2 is None else args.t2
    if not math.isfinite(args.t2) or not .02 <= args.t2 <= .2 or args.t2 < args.period_ms/1000:
        parser.error('--t2 must be in [0.02, 0.2] seconds and at least the period')
    if not math.isfinite(args.gain) or not 0 < args.gain <= 1 or not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error('require 0 < gain <= 1 and 0 <= alpha <= 1')
    output=args.output_dir or HERE/f'servo_{args.period_ms:g}ms'
    try:
        specs=json.loads(args.motions.read_text(encoding='utf-8'))
        output.mkdir(parents=True,exist_ok=True)
        names=['r1_RT','r2_DDA'] if args.robot=='all' else [args.robot]
        summaries={name:generate(name,specs[name],args,output) for name in names}
        (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n',encoding='utf-8')
    except (ValueError,KeyError,OSError) as exc:
        parser.error(str(exc))
    print(f'Output: {output.resolve()}')
    for name,s in summaries.items():
        print(f"{name}: {s['waypoint_count']} waypoints, {s['command_count']} commands, {s['cycle_duration_s']:.3f} s/cycle")


if __name__=='__main__':
    main()
