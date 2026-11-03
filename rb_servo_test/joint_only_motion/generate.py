"""Offline command export and URDF preview; never connects to a robot."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rb_servo_test/external_script_gui'))
from app import build_command
from motion import ServoProfile

OUT = Path(__file__).resolve().parent / 'output'
SPEED, ACCEL, PERIOD = math.degrees(.4), math.degrees(.1), .02
# Only move-linear blocks are removed. Their following waits remain.
ROBOTS = {
    'r1_RT': dict(urdf='rb20_1900es_RT.urdf', prefix='rt',
        begin=[89.81,10.06,-133.84,65.50,-91.36,-.01],
        points=[[89.81,10.06,-133.84,65.50,-91.36,-.01],
                [89.88,-17.82,-103.57,63.12,-91.41,-.06],
                [89.88,-10.36,-88.05,40.14,-91.36,-.03],
                [89.81,10.06,-133.84,65.50,-91.36,-.01]],
        waits=[1,0,3,0]),
    'r2_DDA': dict(urdf='rb10_1300e_DDA.urdf', prefix='dda',
        begin=[90,45,-130,0,-90,-90],
        points=[[89.99,27.75,-118.95,-.01,-83.79,-90],
                [89.99,10.21,-97.49,-.02,-87.72,-90],
                [90,45,-130,0,-90,-90]], waits=[1,3,1]),
}


def origin(element):
    T = np.eye(4)
    if element is not None:
        T[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        T[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return T


class Model:
    def __init__(self, spec):
        self.path = ROOT / 'urdf' / spec['urdf']
        self.root = ET.parse(self.path).getroot()
        self.joints = self.root.findall('joint')
        self.arm = [j for j in self.joints if j.get('type') == 'revolute']
        assert len(self.arm) == 6
        self.base = spec['prefix'] + '_link0'
        self.tip = spec['prefix'] + '_link_end'

    def fk(self, degrees):
        values = dict(zip([j.get('name') for j in self.arm], np.deg2rad(degrees)))
        child_names = {j.find('child').get('link') for j in self.joints}
        world = {l.get('name'): np.eye(4) for l in self.root.findall('link') if l.get('name') not in child_names}
        pending = list(self.joints)
        while pending:
            progress = False
            for joint in pending[:]:
                parent = joint.find('parent').get('link')
                if parent not in world:
                    continue
                T = origin(joint.find('origin'))
                if joint.get('name') in values:
                    axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                    R = np.eye(4)
                    R[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * values[joint.get('name')]).as_matrix()
                    T = T @ R
                world[joint.find('child').get('link')] = world[parent] @ T
                pending.remove(joint)
                progress = True
            if not progress:
                raise ValueError('URDF joint graph cannot be resolved')
        # Display each robot in its own mounting-base coordinates.
        inverse = np.linalg.inv(world[self.base])
        return {name: inverse @ T for name, T in world.items()}


def export(name, spec):
    rows, samples = [], [(0., spec['begin'])]
    clock = 0.
    current = spec['begin']
    for index, (point, wait) in enumerate(zip(spec['points'], spec['waits']), 1):
        rows.append(dict(time_s=clock, action='move_j', waypoint=index,
                         command=build_command('move_j', point, [SPEED, ACCEL])))
        if current != point:
            profile = ServoProfile([current, point], SPEED, ACCEL, PERIOD)
            for offset, q in profile.samples():
                samples.append((clock + offset + PERIOD, q))
            clock += profile.duration
        else:
            profile = None
        # Separate servo schedule: a static target also gets an explicit command.
        spec.setdefault('servo_rows', [])
        start = rows[-1]['time_s']
        if profile:
            for offset, q in profile.samples():
                spec['servo_rows'].append(dict(time_s=start + offset, action='move_servo_j', waypoint=index,
                    command=build_command('move_servo_j', q, [PERIOD,.04,.1,.5])))
        # Zero target velocity at every stop, including reversals and loop boundary.
        for hold in range(2):
            spec['servo_rows'].append(dict(time_s=clock + hold * PERIOD, action='move_servo_j', waypoint=index,
                command=build_command('move_servo_j', point, [PERIOD,.04,.1,.5])))
        # Holds extend the schedule; waits stay their original full duration.
        clock += 2 * PERIOD
        samples.append((clock, point))
        if wait:
            for target_rows in (rows, spec['servo_rows']):
                target_rows.append(dict(time_s=clock, action='wait', waypoint=index, command=f'{wait:.1f}'))
            clock += wait
            samples.append((clock, point))
        current = point
    for kind, schedule in [('move_j', rows), ('servo_j', spec.pop('servo_rows'))]:
        with (OUT / f'{name}_{kind}_schedule.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['time_s','action','waypoint','command'])
            writer.writeheader()
            writer.writerows(schedule)
        with (OUT / f'{name}_{kind}.txt').open('w') as f:
            f.write('# Offline recipe: comments are execution instructions, not robot commands.\n')
            f.write('# SETUP ONCE: reach begin pose; wait for completion.\n')
            f.write(build_command('move_j', spec['begin'], [SPEED, ACCEL]) + '\n')
            f.write('# LOOP FOREVER FROM HERE (one body is listed below).\n')
            for row in schedule:
                if row['action'] == 'wait':
                    f.write(f"# WAIT {row['command']} s after preceding motion reaches endpoint.\n")
                else:
                    f.write(row['command'] + '\n')
                    if kind == 'move_j':
                        f.write('# WAIT for motion completion.\n')
            f.write('# END LOOP. Servo command timing: use companion schedule.csv, period=0.02 s.\n')
    with (OUT / f'{name}_waypoints.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['point','J1_deg','J2_deg','J3_deg','J4_deg','J5_deg','J6_deg','wait_s'])
        writer.writerow(['begin', *spec['begin'], 0])
        for i, (q, wait) in enumerate(zip(spec['points'], spec['waits']), 1):
            writer.writerow([i,*q,wait])
    times, qs = np.array([t for t,q in samples]), np.array([q for t,q in samples])
    # Check the generated sampled profile, including every stop and reversal.
    grid = np.arange(0, clock + PERIOD / 2, PERIOD)
    dense = np.column_stack([np.interp(grid, times, qs[:, k]) for k in range(6)])
    vel = np.diff(dense, axis=0) / PERIOD
    acc = np.diff(vel, axis=0) / PERIOD
    assert np.abs(vel).max() <= SPEED + 1e-7
    assert np.abs(acc).max() <= ACCEL + 1e-6
    return times, qs, dict(duration_s=clock, max_speed_rad_s=float(np.deg2rad(np.abs(vel).max())),
                           max_accel_rad_s2=float(np.deg2rad(np.abs(acc).max())))


def preview(data):
    import trimesh
    from plotly.offline import get_plotlyjs
    sys.path.insert(0, str(ROOT / 'rb_servo_gui/src'))
    from rb_servo_gui.meshview import decimate
    payload = []
    for name, spec in ROBOTS.items():
        model = Model(spec)
        meshes = []
        for link in model.root.findall('link'):
            if not (link.get('name').startswith(spec['prefix'] + '_link') and
                    link.get('name').split('_link')[-1] in ['0','1','2','3','4','5','6','_end']):
                continue
            for visual in link.findall('visual'):
                geometry = visual.find('geometry')
                mesh = geometry.find('mesh')
                if mesh is not None:
                    obj = trimesh.load(model.path.parent / mesh.get('filename'), force='mesh')
                    obj.apply_scale(np.fromstring(mesh.get('scale', '1 1 1'), sep=' '))
                elif geometry.find('box') is not None:
                    obj = trimesh.creation.box(np.fromstring(geometry.find('box').get('size'), sep=' '))
                else:
                    raise ValueError('Unsupported visual geometry')
                V,F = decimate(obj.vertices, obj.faces, 450)
                T = origin(visual.find('origin'))
                V = V @ T[:3,:3].T + T[:3,3]
                meshes.append(dict(link=link.get('name'), vertices=V.round(7).tolist(), faces=F.tolist()))
        times, qs, info = data[name]
        frame_times = np.unique(np.r_[np.arange(0, times[-1], .1), times])
        # Bound HTML size while keeping true waypoint/dwell boundaries.
        frame_times = np.unique(np.r_[np.arange(0, times[-1], .1),
            [t for t,q in zip(times,qs) if any(np.allclose(q,p) for p in spec['points'])], times[-1]])
        frame_q = np.column_stack([np.interp(frame_times,times,qs[:,k]) for k in range(6)])
        frames = []
        for t,q in zip(frame_times,frame_q):
            fk = model.fk(q)
            frames.append(dict(t=float(t),q=q.round(4).tolist(),
                transforms={m['link']:fk[m['link']].round(8).tolist() for m in meshes},
                tcp=fk[model.tip][:3,3].round(7).tolist()))
        payload.append(dict(name=name,meshes=meshes,frames=frames,info=info))
    html = '''<!doctype html><meta charset="utf-8"><title>RT / DDA joint-only URDF motion</title>
<style>body{font:16px system-ui;margin:24px;background:#111827;color:#e5e7eb}button,select{padding:8px;margin:5px}#view{height:72vh}input{width:60%}</style>
<h2>RT / DDA · Joint-only URDF motion</h2>
<p>0.40 rad/s · 0.10 rad/s² · linear blocks removed · original waits retained · base-local coordinates · no robot connection</p>
<select id="robot"></select><button id="play">Play / Pause</button><input id="seek" type="range" min="0" value="0"><span id="time"></span>
<div id="view"></div><p id="angles"></p><p>Preview uses synchronized trapezoidal joint profiles. Controller-specific Move J timing may differ. External axes = 0; collision checking is not performed. Tool endpoint follows the repository URDF.</p>
<script>__PLOTLY__</script><script>
const robots=__DATA__;let selected=0,index=0,playing=false,last=0;
const selector=document.getElementById('robot'),seek=document.getElementById('seek');
robots.forEach((r,i)=>selector.add(new Option(r.name,i)));
function render(){const r=robots[selected],f=r.frames[index];seek.max=r.frames.length-1;seek.value=index;
const traces=r.meshes.map((m,k)=>{const T=f.transforms[m.link],p=m.vertices.map(v=>[0,1,2].map(a=>T[a][3]+v.reduce((s,x,b)=>s+T[a][b]*x,0)));
return {type:'mesh3d',x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),i:m.faces.map(v=>v[0]),j:m.faces.map(v=>v[1]),k:m.faces.map(v=>v[2]),color:['#64748b','#f59e0b','#d1d5db','#d1d5db','#f59e0b','#d1d5db','#f59e0b','#38bdf8'][k%8],name:m.link,flatshading:true};});
const path=r.frames.map(v=>v.tcp);traces.push({type:'scatter3d',mode:'lines',x:path.map(p=>p[0]),y:path.map(p=>p[1]),z:path.map(p=>p[2]),line:{color:'#22d3ee',width:5},name:'URDF tool path'});
Plotly.react('view',traces,{uirevision:r.name,margin:{t:0,b:0,l:0,r:0},paper_bgcolor:'#111827',font:{color:'#e5e7eb'},showlegend:false,scene:{aspectmode:'data',xaxis:{title:'X [m]'},yaxis:{title:'Y [m]'},zaxis:{title:'Z [m]'},camera:{eye:{x:1.5,y:1.5,z:.8}}}},{responsive:true});
document.getElementById('time').textContent=f.t.toFixed(2)+' / '+r.info.duration_s.toFixed(2)+' s';document.getElementById('angles').textContent='J1–J6 [deg]: '+f.q.join(', ');}
selector.onchange=()=>{selected=+selector.value;index=0;last=0;render()};seek.oninput=()=>{index=+seek.value;last=0;render()};document.getElementById('play').onclick=()=>{playing=!playing;last=0};
function tick(now){if(playing){const fs=robots[selected].frames;if(!last)last=now-fs[index].t*1000;const t=(now-last)/1000;if(t>fs[fs.length-1].t){index=0;last=now;render()}else{let next=index;while(next+1<fs.length&&fs[next+1].t<=t)next++;if(next!==index){index=next;render()}}}requestAnimationFrame(tick)}render();requestAnimationFrame(tick);
</script>'''
    (OUT/'preview.html').write_text(html.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(payload,separators=(',',':'))))
    # Static preview showing every retained waypoint with actual URDF meshes.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    columns = max(len(spec['points']) for spec in ROBOTS.values())
    fig = plt.figure(figsize=(4*columns,8))
    for row, robot in enumerate(payload):
        spec = ROBOTS[robot['name']]
        model = Model(spec)
        for column, point in enumerate(spec['points']):
            ax=fig.add_subplot(2,columns,row*columns+column+1,projection='3d')
            fk=model.fk(point)
            all_vertices=[]
            for m in robot['meshes']:
                V=np.array(m['vertices']);T=fk[m['link']];V=V@T[:3,:3].T+T[:3,3]
                all_vertices.append(V)
                ax.add_collection3d(Poly3DCollection(V[np.array(m['faces'])],facecolor='#b5bec9',edgecolor='none',alpha=1))
            V=np.vstack(all_vertices);center=(V.min(0)+V.max(0))/2;radius=np.ptp(V,axis=0).max()*.6
            ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
            ax.set_box_aspect([1,1,1]);ax.view_init(22,45);ax.set_title(f"{robot['name']} P{column+1} / wait {spec['waits'][column]} s")
            ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.set_zlabel('Z [m]')
    fig.suptitle('URDF joint-only waypoints | v=0.40 rad/s, a=0.10 rad/s² | begin pose = final waypoint')
    fig.tight_layout();fig.savefig(OUT/'waypoints.png',dpi=140);plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-linear', action='store_true',
                        help='Convert linear endpoint joint states to joint moves (not TCP straight lines).')
    parser.add_argument('--swap-models', action='store_true',
                        help='Render R1 motion on DDA and R2 motion on RT, retaining joint angles.')
    args = parser.parse_args()
    # Canonical mapping confirmed by the user: R1=RT (begin 90,45,...),
    # R2=DDA (begin 89.81,10.06,...). Supersedes earlier preview labels.
    canonical = json.loads((Path(__file__).resolve().parent / 'motions.json').read_text())
    ROBOTS = {name: dict(spec, prefix='rt' if name == 'r1_RT' else 'dda')
              for name, spec in canonical.items()}
    if args.include_linear:
        OUT = OUT.parent / 'with_linear_endpoints'
    else:
        del ROBOTS['r1_RT']['points'][2:4]
        ROBOTS['r1_RT']['waits'] = [1,3,1]
        del ROBOTS['r2_DDA']['points'][3:5]
        ROBOTS['r2_DDA']['waits'] = [1,0,3,0]
    if args.swap_models:
        OUT = OUT.parent / (OUT.name + '_swapped')
        r1, r2 = ROBOTS['r1_RT'], ROBOTS['r2_DDA']
        r1['urdf'], r2['urdf'] = r2['urdf'], r1['urdf']
        r1['prefix'], r2['prefix'] = r2['prefix'], r1['prefix']
        ROBOTS = {'r1_DDA': r1, 'r2_RT': r2}
    OUT.mkdir(exist_ok=True)
    data={name:export(name,spec) for name,spec in ROBOTS.items()}
    summary={name:info for name,(_,_,info) in data.items()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    preview(data)
    if args.include_linear:
        preview_path = OUT / 'preview.html'
        preview_path.write_text(preview_path.read_text().replace(
            'linear blocks removed', 'linear endpoints converted to joint moves; TCP straight paths are not retained'))
    print(json.dumps(summary,indent=2))
    print(OUT/'preview.html')
