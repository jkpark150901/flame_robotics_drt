import json,hashlib
from pathlib import Path
from collections import defaultdict
roots=[Path('/home/dev/rb_servo_gui_runs'),Path('rb_servo_test/jb2_gui/out')]
allgroups=[]
for root in roots:
 groups=defaultdict(list)
 for p in sorted(root.iterdir()):
  if not (p/'config.json').exists():continue
  c=json.loads((p/'config.json').read_text());c=c.get('effective',c)
  if c.get('use_mock'):continue
  if root==roots[0]:
   files=sorted(p.glob('traj_sub*.csv'))+sorted(p.glob('servo_profile*.csv'))
   if not files:continue
   ignore={'out_dir','traj_path','urdf_path','surface_points','surface_samples','real_ack'}
   cfg={k:v for k,v in c.items() if k not in ignore and not k.startswith('th_')}
   mode=c.get('mode');follow=list(p.glob('follow_*.csv'))
   if not follow:continue
   with follow[0].open() as f:meta=json.loads(f.readline()[1:])
   marks=meta.get('marks',{});complete=mode+'_done' in marks
  else:
   files=[p/'waypoints.csv'];ignore={'window','timeout','via_sample_indices'}
   if not files[0].exists() or not (p/'manifest.json').exists():continue
   cfg={k:v for k,v in c.items() if k not in ignore}
   complete=all(e.get('completed') for e in json.loads((p/'manifest.json').read_text()))
  if not complete:continue
  digest=hashlib.sha256(b''.join(f.read_bytes() for f in files)).hexdigest()
  key=json.dumps([cfg,digest],sort_keys=True)
  groups[key].append(str(p))
 print('\nROOT',root,'completed groups',len(groups))
 for key,paths in groups.items():
  if len(paths)>1:
   cfg,digest=json.loads(key)
   print('REPEAT',len(paths),'mode',cfg.get('robot_mode',cfg.get('mode')),'method',cfg.get('method',cfg.get('mode')),'traj',Path(paths[0]).name,'speed',cfg.get('speed'),'accel',cfg.get('accel'))
   print('\n'.join(paths))
   allgroups.append(dict(config=cfg,paths=paths,hash=digest))
Path('rb_servo_gui/docs/report/repeatability/matching_runs.json').write_text(json.dumps(allgroups,indent=2))
