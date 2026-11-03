"""Sample each motion to 20 joint waypoints and export Servo J / JB2 recipes offline."""
import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

from generate_servo import HERE, command, generate

OUT = HERE / 'waypoints_20'
ROOT = HERE.parents[1]


def sample(spec):
    anchors, waits = [spec['begin']], [0.]
    for q,w in zip(spec['points'],spec['waits']):
        if q == anchors[-1]:
            waits[-1] += w
        else:
            anchors.append(q)
            waits.append(w)
    lengths=[math.dist(a,b) for a,b in zip(anchors,anchors[1:])]
    extras=19-len(lengths)
    if extras < 0 or not sum(lengths):
        raise ValueError('Cannot preserve anchors within 20 points')
    shares=[extras*d/sum(lengths) for d in lengths]
    counts=[1+math.floor(s) for s in shares]
    for i in sorted(range(len(lengths)),key=lambda i:shares[i]%1,reverse=True)[:19-sum(counts)]:
        counts[i]+=1
    points, dwells = [anchors[0]], [waits[0]]
    for a,b,w,n in zip(anchors,anchors[1:],waits[1:],counts):
        for k in range(1,n+1):
            points.append(b if k==n else [x+(y-x)*k/n for x,y in zip(a,b)])
            dwells.append(w if k==n else 0.)
    if len(points)!=20:
        raise ValueError('Expected exactly 20 waypoints')
    return dict(spec,points=points,waits=dwells)


def check_limits(spec, poses):
    joints=[j for j in ET.parse(ROOT/'urdf'/spec['urdf']).getroot().findall('joint')
            if j.get('type')=='revolute']
    if len(joints)!=6:
        raise ValueError('Expected six URDF arm joints')
    report=[]
    for i,j in enumerate(joints):
        limit=j.find('limit')
        lower,upper=math.degrees(float(limit.get('lower'))),math.degrees(float(limit.get('upper')))
        low,high=min(q[i] for q in poses),max(q[i] for q in poses)
        passed=all(math.isfinite(q[i]) and lower<=q[i]<=upper for q in poses)
        report.append(dict(joint=j.get('name'),lower_deg=lower,upper_deg=upper,
                           commanded_min_deg=low,commanded_max_deg=high,passed=passed))
    if not all(r['passed'] for r in report):
        raise ValueError('URDF joint position limit exceeded')
    return report


def jb2(name,spec,blend):
    speed,accel=math.degrees(.4),math.degrees(.1)
    lines=['# Offline recipe: comments are execution instructions, not robot commands.',
           '# SETUP ONCE; wait until begin pose is reached.',
           command('move_j',spec['begin'],[speed,accel,1]),
           'move_finish_wait()',
           '# LOOP FOREVER FROM HERE. WAIT comments must be applied by the execution side.',
           'move_jb2_clear()']
    for i,(q,wait) in enumerate(zip(spec['points'],spec['waits'])):
        lines.append(command('move_jb2_add',q,[speed,accel,0,blend]))
        if wait or i==19:
            lines += ['move_jb2_run(1)','move_finish_wait()']
            if wait:
                lines.append(f'# WAIT {wait:g} s after completion.')
            if i!=19:
                lines.append('move_jb2_clear()')
    lines.append('# END LOOP. Blending applies within batches; waits split batches.')
    (OUT/f'{name}_jb2_blend_{blend}.txt').write_text('\n'.join(lines)+'\n')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--period-ms',type=float,default=20,help='Servo J interval (2..100 ms)')
    options=parser.parse_args()
    if not math.isfinite(options.period_ms) or not 2<=options.period_ms<=100:
        parser.error('--period-ms must be finite and between 2 and 100')
    if options.period_ms!=20:
        OUT=HERE/f'waypoints_20_{options.period_ms:g}ms'
    OUT.mkdir(exist_ok=True)
    specs={name:sample(spec) for name,spec in json.loads((HERE/'motions.json').read_text()).items()}
    # Validate all sampled targets before producing motion files.
    reports={name:dict(waypoint_count=20,joints=check_limits(spec,spec['points'])) for name,spec in specs.items()}
    args=SimpleNamespace(period_ms=options.period_ms,speed=.4,accel=.1,
                         t2=max(.04,options.period_ms/1000),gain=.1,alpha=.5)
    summaries={}
    for name,spec in specs.items():
        summaries[name]=generate(name,spec,args,OUT)
        # Include setup in the Servo recipe so each of the four TXT files is self-contained.
        path=OUT/f'{name}_servo_j.txt'
        path.write_text('# SETUP ONCE; await completion.\n'+
            command('move_j',spec['begin'],[math.degrees(.4),math.degrees(.1),1])+ '\nmove_finish_wait()\n'+
            '# LOOP BODY STARTS BELOW.\n'+path.read_text().replace(
                '# First execute setup.txt (Move J + move_finish_wait); await completion before starting the periodic schedule.',
                '# Setup is included above; do not repeat setup each cycle.').replace(
                '# END LOOP:', 'move_finish_wait()\n# END LOOP:'))
        (OUT/f'{name}_setup.txt').unlink()
        with (OUT/f'{name}_trajectory.csv').open() as f:
            poses=[[float(r[f'J{i}_deg']) for i in range(1,7)] for r in csv.DictReader(f)]
        reports[name]['servo_sample_limits']=check_limits(spec,poses)
        for blend in [100,50,0]:
            jb2(name,spec,blend)
    (OUT/'sampled_motions.json').write_text(json.dumps(specs,indent=2)+'\n')
    (OUT/'joint_limits.json').write_text(json.dumps(reports,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (OUT/'README.md').write_text('''# 20-waypoint motion exports

R1 = RT (begin 90,45,-130,0,-90,-90); R2 = DDA (begin 89.81,10.06,-133.84,65.50,-91.36,-0.01).
Each cycle has exactly 20 geometric waypoints including begin and return. Original anchors and waits are retained.
Extra points are distributed approximately by joint-space arc length; sampling uses straight joint interpolation.
Former linear endpoints are treated as joint targets; TCP straight motion is not preserved.

Each robot has four TXT files: servo_j and jb2_blend_100/50/0. Speed=0.40 rad/s, acceleration=0.10 rad/s²
(converted to deg/s and deg/s² in command strings).
Servo J interpolates the 20 geometric points at 20ms, stops at each waypoint, adds two settling samples,
and includes waits as repeated endpoint commands. It therefore has more than 20 command lines.
Use schedule.csv for transmission timing; do not add extra waits for Servo J.
JB2 has exactly 20 add commands. Waits split batches: await run(1) completion, then apply the WAIT comment.
Explicit move_finish_wait() is included after setup and every JB2 run. Servo J includes it only
after setup and at the end of the cycle, never between its 20ms streaming targets.
TXT comments for LOOP/WAIT are execution instructions, not controller commands. No robot connection is made.

joint_limits.json validates the 20 joint targets and dense Servo command positions against each URDF's six
revolute-joint lower/upper bounds (radians converted to degrees). Straight interpolation between in-range
joint targets stays in the bounds. This does not validate the controller's blend=50/100 intermediate path,
collision clearance, dynamics, or real robot tracking. Only commanded joint position limits are checked here.
'''.replace('at 20ms',f'at {options.period_ms:g}ms').replace('its 20ms',f'its {options.period_ms:g}ms')+
        '\nDwell durations round upward to a whole number of command periods. See summary.json for requested and actual waits.\n')
    archive=HERE/f'{OUT.name}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():
                z.write(p,arcname=OUT.name+'/'+p.name)
    print(f'Generated {OUT}: 20 waypoints per robot, 8 TXT files; commanded joint limits PASS')
    print(archive)


if __name__=='__main__':
    main()
