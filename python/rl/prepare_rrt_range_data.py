"""Point-level split then current-URDF IK validation at the snapshot's fixed positioner angle."""
import argparse,hashlib,json,pickle,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rl.path_env import load_planning_config
from robot_core.worker import RobotCoreEngine


def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);snapshot=pickle.loads(a.snapshot.read_bytes());cfg=load_planning_config()
 robot='dda_rb10_1300e';groups=snapshot['target_groups'];points=sorted({g['point_index'] for g in groups})
 rng=np.random.default_rng(20261102);rng.shuffle(points);cut=int(len(points)*.7);train=set(points[:cut])
 split={'train_points':points[:cut],'eval_points':points[cut:],'seed':20261102};(a.out/'split.json').write_text(json.dumps(split,indent=2))
 engine=RobotCoreEngine(cfg,snapshot);backend=engine._robotics_backend
 model=engine._find_robot(robot);bm=backend.robot_model(robot);start=engine._current_robot_q(model,bm,robot_name=robot)
 obstacle=engine._current_spool_collision_mesh();rows=[]
 stored=a.out/'snapshot.pkl'
 dataset={'robot':robot,'snapshot':str(stored.resolve()),'targets':rows,'split':split,'fixed_positioner':True,
          'urdf_sha256':{x['path']:hashlib.sha256(Path(x['path']).read_bytes()).hexdigest() for x in cfg['urdf']}}
 def save(state):
  stored.write_bytes(pickle.dumps(snapshot));(a.out/'dataset.json').write_text(json.dumps(dataset,indent=2))
  (a.out/'status.json').write_text(json.dumps({'state':state,'processed':len(rows),'total':len(groups),'valid':sum(x['valid'] for x in rows),'updated':time.time()},indent=2))
 save('running')
 for g in groups:
  t0=time.time();row={'group':g['name'],'point_index':g['point_index'],'split':'train' if g['point_index'] in train else 'eval','valid':False,'target_pose':g['dda_pose']}
  try:
   res=engine._plan_inspection_path_for_robot({'robot_name':robot,'planner':'rrt_connect','step_size':.1,'max_iter':3000,'ik_solver':'collision_free','ik_only':True,'_start_q_override_by_robot':{robot:start.tolist()},'context_label':g['name']},robot,np.asarray(g['dda_pose']),obstacle,context_label=g['name'])
   q0=np.asarray(start).copy();q1=np.asarray(res['goal_q'])
   lo,hi,_=backend.joint_limits_for_metric(robot)
   bounds_ok=bool(np.all(q0>=lo) and np.all(q0<=hi) and np.all(q1>=lo) and np.all(q1<=hi))
   valid=bounds_ok and not res.get('ik_fallback') and bool(res.get('collision_free')) and not backend.check_collision(robot,q0).collision and not backend.check_collision(robot,q1).collision
   row.update(valid=valid,start=q0.tolist(),goal=q1.tolist(),reason=None if valid else 'IK or endpoint collision failed')
   g['ik_goal_info']={robot:{'success':valid,'start_q':q0.tolist(),'phase':1}};g['ik_goal_q']={robot:q1.tolist()} if valid else {}
  except Exception as exc:row['reason']=str(exc)
  row['seconds']=time.time()-t0;rows.append(row);save('running');print(len(rows),len(groups),row['group'],row['valid'],flush=True)
 save('complete')
if __name__=='__main__':main()
