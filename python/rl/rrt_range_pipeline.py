"""Continue the already-started data preparation with real-scene smoke, training and evaluation."""
import json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'debug/rrt_range_20261102'

def main():
 sources=OUT/'sources';sources.mkdir(exist_ok=True)
 for name in ['rrt_range_env.py','run_rrt_range_experiment.py','prepare_rrt_range_data.py','augment_inspection_points.py']:
  shutil.copy2(ROOT/'python/rl'/name,sources/name)
 def status(**kw):(OUT/'status.json').write_text(json.dumps(dict(updated=time.time(),**kw),indent=2))
 try:
  status(state='preparing_ik')
  while True:
   file=OUT/'prepared/status.json'
   try:ready=json.loads(file.read_text()).get('state')=='complete'
   except (FileNotFoundError,json.JSONDecodeError):ready=False
   if ready:break
   # Do not silently wait forever if the preparation process exits.
   try:
    state=Path('/proc/968624/stat').read_text().split(') ',1)[1].split()[0]
    if state=='Z':raise RuntimeError('IK preparation exited before completion; inspect prepare.log')
   except FileNotFoundError:raise RuntimeError('IK preparation process missing; inspect prepare.log')
   time.sleep(5)
  dataset=OUT/'prepared/dataset.json'
  for label,extra in [('smoke',['--smoke']),('train_eval',[])]:
   status(state=label)
   with (OUT/f'{label}.log').open('w') as log:
    subprocess.run([sys.executable,'-u',str(ROOT/'python/rl/run_rrt_range_experiment.py'),'--dataset',str(dataset),'--out',str(OUT/label),*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  status(state='complete')
 except BaseException as exc:status(state='failed',error=str(exc));raise
if __name__=='__main__':main()
