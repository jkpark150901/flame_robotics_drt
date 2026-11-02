"""Fresh SAC training of joint sampling ranges, followed by held-out fixed-width comparison."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rl.rrt_range_env import RRTRangeEnv


def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=10000);p.add_argument('--smoke',action='store_true');p.add_argument('--config',type=Path,help='JSON with search/reward weights and max_steps');a=p.parse_args()
 overrides=json.loads(a.config.read_text()) if a.config else {}
 if not isinstance(overrides,dict):p.error('--config must contain a JSON object')
 max_steps=int(overrides.pop('max_steps',200))
 if max_steps<=0:p.error('max_steps must be positive')
 from stable_baselines3 import SAC
 from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,CallbackList
 from rl.convergence import Plateau
 import torch
 torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False)
 env=RRTRangeEnv(a.dataset,max_steps=max_steps,config=overrides);obs,_=env.reset(seed=0)
 assert env.observation_space.contains(obs)
 # A short real-geometry smoke is required before allocating the training run.
 smoke=[]
 for _ in range(5):
  obs,r,done,trunc,info=env.step(np.zeros(env.n,dtype=np.float32));assert np.isfinite(r) and env.observation_space.contains(obs)
  smoke.append(dict(reward=r,**info))
  if done or trunc:break
 (a.out/'smoke.json').write_text(json.dumps(smoke,indent=2))
 if a.smoke:return
 cfg={'steps':a.steps,'max_steps':max_steps,'reward_and_search':env.cfg,'dataset':str(a.dataset.resolve()),'network':[128,128],
      'learning_rate':.0003,'batch_size':256,'learning_starts':1000,'gradient_steps':1,'seed':0,'device':'cpu'}
 (a.out/'config.json').write_text(json.dumps(cfg,indent=2))
 class Stats(BaseCallback):
  def __init__(self):super().__init__();self.episodes=[];self.total=0.;self.start=0;self.losses=[];self.stop_reason='step_limit';self.plateau=Plateau(minimum=4000,tolerance=.1,patience=3)
  def _on_step(self):
   self.total+=float(self.locals['rewards'][0]);info=self.locals['infos'][0]
   if self.locals['dones'][0]:self.episodes.append((self.total,info.get('result')=='success'));self.total=0.
   val=self.logger.name_to_value
   if 'train/actor_loss' in val:self.losses.append((float(val['train/actor_loss']),float(val['train/critic_loss'])))
   if self.num_timesteps%100==0:(a.out/'progress.json').write_text(json.dumps({'steps':self.num_timesteps,'max_steps':a.steps,'updated':time.time()}))
   if self.num_timesteps-self.start>=1000:
    if self.episodes and self.losses:
     metrics={'reward':float(np.mean([x[0] for x in self.episodes])),'success':float(np.mean([x[1] for x in self.episodes])),'actor_loss':float(np.median([x[0] for x in self.losses])),'critic_loss':float(np.median([x[1] for x in self.losses]))}
     stop,changes=self.plateau.update(self.num_timesteps,metrics)
     with (a.out/'convergence.jsonl').open('a') as f:f.write(json.dumps(dict(step=self.num_timesteps,**metrics,changes=changes,stop=stop))+'\n')
     if stop:self.stop_reason='plateau';return False
    self.start=self.num_timesteps;self.episodes=[];self.losses=[]
   return True
 stats=Stats();model=SAC('MlpPolicy',env,policy_kwargs={'net_arch':[128,128]},learning_rate=.0003,batch_size=256,buffer_size=100000,learning_starts=1000,gradient_steps=1,seed=0,device='cpu',tensorboard_log=str(a.out/'tb'),verbose=1)
 (a.out/'status.json').write_text(json.dumps({'state':'training'}))
 model.learn(a.steps,callback=CallbackList([stats,CheckpointCallback(save_freq=1000,save_path=str(a.out/'checkpoints'),name_prefix='range')]))
 model.save(a.out/'final_model');(a.out/'training_result.json').write_text(json.dumps({'steps':model.num_timesteps,'stop_reason':stats.stop_reason}))
 (a.out/'status.json').write_text(json.dumps({'state':'evaluating'}))
 env.close();ev=RRTRangeEnv(a.dataset,split='eval',max_steps=max_steps,config=overrides);results=[]
 # This fixed-range baseline uses precisely the same tree code and seeds; not OMPL itself.
 for target in ev.targets:
  for seed in [810000,820000,830000]:
   for method in ['fixed_range','sac_range']:
    obs,_=ev.reset(seed=seed,options={'target':target});start=time.time();reward=0.
    while True:
     action=np.zeros(ev.n) if method=='fixed_range' else model.predict(obs,deterministic=True)[0]
     obs,r,done,trunc,info=ev.step(action);reward+=r
     if done or trunc:break
    results.append(dict(method=method,seed=seed,elapsed=time.time()-start,reward=reward,**info));(a.out/'evaluation.json').write_text(json.dumps(results,indent=2))
 data=json.loads(a.dataset.read_text());lines=['# Learned RRT sampling range evaluation','','Fixed range is the same custom RRT-Connect search, not OMPL. Invalid IK targets are reported separately.','','| method | success / valid attempts | mean path length | mean smoothness | mean obstacle cost |','|---|---:|---:|---:|---:|']
 for method in ['fixed_range','sac_range']:
  rows=[x for x in results if x['method']==method];ok=[x for x in rows if x['result']=='success']
  means=[f"{np.mean([x['costs'][k] for x in ok]):.4f}" if ok else '-' for k in ['length','smooth','obstacle']]
  lines.append(f"| {method} | {len(ok)}/{len(rows)} | {' | '.join(means)} |")
 invalid=[t for t in data['targets'] if t['split']=='eval' and not t['valid']];lines+=['',f'Excluded invalid IK/endpoints in held-out split: {len(invalid)} targets.']
 (a.out/'summary.md').write_text('\n'.join(lines));(a.out/'status.json').write_text(json.dumps({'state':'complete','updated':time.time()}));ev.close()
if __name__=='__main__':main()
