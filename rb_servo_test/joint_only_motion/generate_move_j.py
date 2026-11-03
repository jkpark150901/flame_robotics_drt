"""Export offline Move J commands and a sampled estimated trajectory including dwell."""
import argparse
import csv
import json
import math
from pathlib import Path

from generate_servo import HERE, ServoProfile, command


def generate(name, spec, args, output):
    dt = args.period_ms / 1000
    speed, accel = math.degrees(args.speed), math.degrees(args.accel)
    begin = tuple(map(float, spec['begin']))
    points = [tuple(map(float, q)) for q in spec['points']]
    waits = list(map(float, spec['waits']))
    if not points or len(points) != len(waits):
        raise ValueError(f'{name}: points/waits must have equal nonzero lengths')
    if any(len(q) != 6 or not all(math.isfinite(v) for v in q) for q in [begin,*points]):
        raise ValueError(f'{name}: six finite joint angles required for each pose')
    if any(not math.isfinite(w) or w < 0 for w in waits) or begin != points[-1]:
        raise ValueError(f'{name}: waits must be nonnegative and last point must equal begin')
    schedule = []
    trajectory = [[0.,0,'begin',*begin]]
    lines = ['# Offline Move J recipe; comments are execution instructions.',
             '# Execute setup once and wait for completion, then repeat this body forever.',
             '# Each Move J waits for completion. Times in CSV are estimates only.']
    current, clock = begin, 0.
    for index,(point,wait) in enumerate(zip(points,waits),1):
        profile = ServoProfile([current,point],speed,accel,dt) if current != point else None
        duration = profile.duration if profile else 0.
        cmd = command('move_j',point,[speed,accel,1])
        schedule.append([clock,'move_j',index,duration,cmd])
        lines += [cmd,'# WAIT until this Move J completes.']
        if profile:
            for step,(_,q) in enumerate(profile.samples(),1):
                trajectory.append([clock+step*dt,index,'motion',*q])
        clock += duration
        if wait:
            schedule.append([clock,'wait',index,wait,''])
            lines.append(f'# WAIT {wait:.12g} s after completion (not a robot command).')
            # Preserve exact wait duration. Include the dwell endpoint even if
            # a custom wait is not divisible by the preview sampling interval.
            steps = max(1,math.ceil(wait/dt-1e-9))
            for step in range(1,steps+1):
                trajectory.append([clock+min(step*dt,wait),index,'dwell',*point])
            clock += wait
        current = point
    lines.append('# END LOOP; repeat the body, not setup.')
    (output/f'{name}_move_j.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/f'{name}_setup.txt').write_text(command('move_j',begin,[speed,accel,1])+'\n',encoding='utf-8')
    with (output/f'{name}_move_j_schedule.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['estimated_time_s','action','waypoint','duration_s','command'])
        writer.writerows(schedule)
    with (output/f'{name}_move_j_trajectory.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['estimated_time_s','waypoint','phase',*[f'J{i}_deg' for i in range(1,7)]])
        writer.writerows(trajectory)
    return dict(urdf=spec['urdf'],begin_deg=begin,waypoint_count=len(points),
                command_count=len(points),preview_sample_period_ms=args.period_ms,
                speed_limit_rad_s=args.speed,accel_limit_rad_s2=args.accel,
                total_wait_s=sum(waits),estimated_cycle_duration_s=round(clock,9),
                trajectory_sample_count=len(trajectory),repeat_count=-1,
                timing='Estimated synchronized trapezoidal profile; controller timing may differ. Execute each move to completion, then wait the exact requested duration. The trajectory CSV is for preview, not a stream of Move J commands.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--period-ms',type=float,default=20,help='Estimated trajectory sampling interval, NOT Move J transmission interval (2..100 ms)')
    parser.add_argument('--speed',type=float,default=.4,help='Joint speed limit [rad/s]')
    parser.add_argument('--accel',type=float,default=.1,help='Joint acceleration limit [rad/s²]')
    parser.add_argument('--robot',choices=['all','r1_RT','r2_DDA'],default='all')
    parser.add_argument('--motions',type=Path,default=HERE/'motions.json')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if not math.isfinite(args.period_ms) or not 2<=args.period_ms<=100:
        parser.error('--period-ms must be finite and between 2 and 100')
    if any(not math.isfinite(v) or v<=0 for v in (args.speed,args.accel)):
        parser.error('--speed and --accel must be finite and positive')
    output=args.output_dir or HERE/f'move_j_{args.period_ms:g}ms'
    try:
        specs=json.loads(args.motions.read_text(encoding='utf-8'))
        names=list(specs) if args.robot=='all' else [args.robot]
        output.mkdir(parents=True,exist_ok=True)
        summary={name:generate(name,specs[name],args,output) for name in names}
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    except (ValueError,KeyError,OSError) as exc:
        parser.error(str(exc))
    print(f'Output: {output.resolve()}')
    for name,s in summary.items():
        print(f"{name}: {s['command_count']} Move J commands, {s['trajectory_sample_count']} preview samples, estimated {s['estimated_cycle_duration_s']:.3f} s/cycle")


if __name__=='__main__':
    main()
