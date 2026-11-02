"""Prepare 40 augmented points, point-level 70/30 split, IK and real-geometry smoke with current URDFs.
Does not start training. Existing output directories are refused.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 source=a.snapshot.resolve();archive=out/'sources';archive.mkdir()
 paths=[ROOT/'urdf'/f for f in ['model.urdf','rb10_1300e_DDA.urdf','rb20_1900es_RT.urdf','positioner.urdf']]
 paths += [ROOT/'python/rl'/f for f in ['augment_inspection_points.py','prepare_rrt_range_data.py','rrt_range_env.py','run_rrt_range_experiment.py']]
 for path in paths:shutil.copy2(path,archive/path.name)
 (out/'manifest.json').write_text(json.dumps({'source':str(source),'sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [source,*paths]}},indent=2))
 def status(**kw):(out/'status.json').write_text(json.dumps(dict(updated=time.time(),**kw),indent=2))
 stages=[('augment',['augment_inspection_points.py','--snapshot',str(source),'--out',str(out/'data'),'--count','40','--attempts','300']),
         ('ik',['prepare_rrt_range_data.py','--snapshot',str(out/'data/augmented.pkl'),'--out',str(out/'prepared')]),
         ('smoke',['run_rrt_range_experiment.py','--dataset',str(out/'prepared/dataset.json'),'--out',str(out/'smoke'),'--smoke'])]
 try:
  for stage,args in stages:
   status(state=stage)
   with (out/f'{stage}.log').open('w') as f:subprocess.run([sys.executable,'-u',str(ROOT/'python/rl'/args[0]),*args[1:]],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  status(state='ready_for_training')
 except BaseException as exc:status(state='failed',error=str(exc));raise
if __name__=='__main__':main()
