"""Sample snapshot pipe surfaces and retain points with >=2 collision-free EF pose pairs."""
import argparse,copy,hashlib,json,pickle,sys,time
from pathlib import Path
import numpy as np
import open3d as o3d
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rl.path_env import load_planning_config
from robot_core.worker import RobotCoreEngine
from robot_core.pose_service import PoseDeterminationService
from plugins.poseDeterminator.PipeProfileFitting import PipeProfileParams


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=20);p.add_argument('--attempts',type=int,default=150);p.add_argument('--seed',type=int,default=20261102);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False)
 s=pickle.loads(a.snapshot.read_bytes());cfg=load_planning_config();worker=RobotCoreEngine(cfg,s)
 mesh=worker._current_spool_collision_mesh();obstacles=[mesh,worker._build_positioner_collision_mesh()]
 service=PoseDeterminationService(cfg,worker._robotics_backend,obstacle_meshes=obstacles)
 o3d.utility.random.seed(a.seed);cloud=mesh.sample_points_uniformly(200000);cloud=service._point_cloud(np.asarray(cloud.points))
 opt,params,facing=service._optimizer(cloud)
 # PoseDeterminationService installs the same URDF-link PyBullet checker as the GUI.
 profile=PipeProfileParams.from_config(params)
 candidates=np.asarray(mesh.sample_points_uniformly(a.attempts).points)
 accepted=[];groups=[];records=[]
 def save(state):
  result=copy.deepcopy(s);result['target_groups']=groups;result['inspection_points']=[x.tolist() for x in accepted]
  result.pop('ik_goals_meta',None)
  result['augmentation']={'source':str(a.snapshot.resolve()),'seed':a.seed,'minimum_pose_groups':2,'collision_engine':'pybullet','validation':'EF vs pipe/positioner only; full-arm IK/path not validated'}
  tmp=a.out/'augmented.tmp';tmp.write_bytes(pickle.dumps(result));tmp.replace(a.out/'augmented.pkl')
  (a.out/'status.json').write_text(json.dumps({'state':state,'attempted':len(records),'accepted_points':len(accepted),'pose_groups':len(groups),'updated':time.time()},indent=2))
  (a.out/'attempts.json').write_text(json.dumps(records,indent=2))
 save('running')
 for point in candidates:
  if any(np.linalg.norm(point-x)<.05 for x in accepted):continue
  record={'point':point.tolist()}
  try:
   opt.debuging_info={};opt.calculate_pipe_profile(point,profile)
   found=opt.calculate_DDA_RT_pose_for_taking_xray(point,num_candidates=int(params.get('num_candidates',9)),distance_from_dda_to_surface=float(params.get('distance_from_dda_to_surface',.01)),distance_from_dda_to_rt=float(params.get('distance_from_dda_to_rt',.3)),angle_of_rt=float(params.get('angle_of_rt',10)),rt_pipe_facing_axis=facing,force_90_fallback=False)
   record['pose_groups']=len(found)
   if len(found)>=2 and all(g.get('dda_pose') is not None and g.get('rt_pose') is not None for g in found):
    idx=len(accepted);accepted.append(point)
    for j,g in enumerate(found):g.update(point_index=idx,index=len(groups),name=f'Point {idx+1} - Inspection pose {j+1}');groups.append(g)
    record['accepted']=True
  except Exception as exc:record['error']=str(exc)
  records.append(record);save('running');print(len(records),len(accepted),record,flush=True)
  if len(accepted)>=a.count:break
 save('complete' if len(accepted)>=a.count else 'candidate_limit')
 if accepted:
  pc=o3d.geometry.PointCloud();pc.points=o3d.utility.Vector3dVector(accepted);pc.paint_uniform_color([1,0,0]);o3d.io.write_point_cloud(str(a.out/'inspection_points.pcd'),pc)
 (a.out/'README.md').write_text(f'# Inspection point augmentation\n\nSource: {a.snapshot}\n\nAccepted points: {len(accepted)}; pose groups: {len(groups)}.\n\nSampled from saved pipe collision mesh surface. Minimum separation 50 mm. Each point has at least two DDA/RT groups checked with PyBullet (1 mm margin). Current project URDFs are used. Full-arm IK, robot self-collision and connecting path feasibility have NOT been validated.\n\nLoad augmented.pkl using Load Snapshot.\n')
if __name__=='__main__':main()
