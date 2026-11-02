"""SAC controls per-joint sampling ranges of a bidirectional RRT-Connect search.
Not OMPL: explicit trees expose each expansion as a Gym step. Physics checks
are delegated to the project's PyBullet backend, including interpolated edges.
"""
import json
import hashlib
from pathlib import Path
import gymnasium as gym
import numpy as np
from rl.path_env import _Scene, load_planning_config


def path_costs(path, lo, span, clearance, influence=.3, samples=32):
    """Normalized arc length, resampled geometric curvature, clearance integral.
    Independent of input waypoint count; clearance callback uses scene geometry.
    """
    q=np.asarray(path); z=(q-lo)/span
    lengths=np.linalg.norm(np.diff(z,axis=0),axis=1); s=np.r_[0,np.cumsum(lengths)]
    length=float(s[-1])
    if length < 1e-12:return dict(length=0.,smooth=0.,obstacle=0.,min_clearance=float(clearance(q[0])))
    keep=np.r_[True,np.diff(s)>1e-12]; s=s[keep]; z=z[keep]
    u=np.linspace(0,length,samples)
    zz=np.stack([np.interp(u,s,z[:,j]) for j in range(z.shape[1])],axis=1)
    ds=length/(samples-1)
    smooth=float(np.sum(np.diff(zz,n=2,axis=0)**2)/max(ds**3,1e-12))
    d=np.asarray([clearance(lo+x*span) for x in zz])
    penalty=np.maximum(0,1-d/influence)**2
    obstacle=float(np.sum((penalty[:-1]+penalty[1:])*.5)*ds)
    return dict(length=length,smooth=smooth,obstacle=obstacle,min_clearance=float(d.min()))


class RRTRangeEnv(gym.Env):
    def __init__(self, dataset, split='train', max_steps=200, config=None):
        super().__init__();self.dataset=Path(dataset).resolve();self.specs=json.loads(self.dataset.read_text())
        for path, digest in self.specs.get('urdf_sha256', {}).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'URDF changed since IK preparation: {path}')
        self.robot=self.specs['robot'];self.targets=[t for t in self.specs['targets'] if t['split']==split and t['valid']]
        if not self.targets:raise ValueError(f'No valid targets in {split}')
        self.cfg=dict(influence=.3,range_min=.02,range_max=.4,global_prob=.15,goal_prob=.1,
                      extend_step=.05,connect_limit=20,obstacle_weight=5.,length_weight=1.,smooth_weight=.01,
                      success_reward=100.,collision_penalty=2.,time_penalty=.05)
        unknown=set(config or {})-set(self.cfg)
        if unknown:raise ValueError(f'Unknown search/reward setting(s): {sorted(unknown)}')
        self.cfg.update(config or {});self.max_steps=max_steps
        c=self.cfg
        if not (0<c['range_min']<=c['range_max']<=1 and c['influence']>0 and c['extend_step']>0
                and 0<=c['global_prob']<=1 and 0<=c['goal_prob']<=1
                and c['global_prob']+c['goal_prob']<=1 and int(c['connect_limit'])>0):
            raise ValueError('Invalid RRT search range, influence, or sampling probabilities')
        if any(c[k]<0 for k in ('obstacle_weight','length_weight','smooth_weight','success_reward',
                                'collision_penalty','time_penalty')):
            raise ValueError('Reward weights must be nonnegative')
        self.scene=_Scene(load_planning_config(), self.specs['snapshot']);self.backend=self.scene.backend
        if not self.backend._use_pybullet():raise RuntimeError('PyBullet is required')
        self.lo,self.hi,self.span=self.backend.joint_limits_for_metric(self.robot)
        self.n=len(self.lo);self.action_space=gym.spaces.Box(-1,1,(self.n,),dtype=np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(4*self.n+5,),dtype=np.float32)

    def _activate(self,t):
        self.scene.activate(dict(snapshot=self.specs['snapshot'],group=t['group'],phase=1,
                                target_pose=np.array(t['target_pose']),goal_q=np.array(t['goal']),start_q=np.array(t['start'])),self.robot)

    def clearance(self,q):
        values=self.backend.link_obstacle_distances(self.robot,q,max_distance=self.cfg['influence'])
        return min((float(x['distance']) for x in values),default=self.cfg['influence'])

    def reset(self,seed=None,options=None):
        super().reset(seed=seed);options=options or {}
        t=options.get('target',self.targets[int(self.np_random.integers(len(self.targets)))])
        self._activate(t);self.target=t;self.start=np.asarray(t['start']);self.goal=np.asarray(t['goal'])
        for q in (self.start,self.goal):
            if self.backend.check_collision(self.robot,q).collision:raise RuntimeError('Stored start/goal invalid under current URDF')
        self.trees=[[self.start.copy()],[self.goal.copy()]];self.parents=[[-1],[-1]];self.active=0;self.steps=0
        self.anchor=self.start.copy();self.prev=np.zeros(self.n);self.last_clearance=self.clearance(self.anchor)
        self.collisions=0;self.checks=0
        return self._obs(),{}

    def _nearest(self,tree,q):return int(np.argmin(np.linalg.norm((np.asarray(tree)-q)/self.span,axis=1)))

    def _obs(self):
        other=self.trees[1-self.active];near=other[self._nearest(other,self.anchor)]
        return np.r_[2*(self.anchor-self.lo)/self.span-1,(self.goal-self.anchor)/self.span,
                     (near-self.anchor)/self.span,self.prev,self.last_clearance/self.cfg['influence'],
                     self.steps/self.max_steps,len(self.trees[0])/(1+self.steps*21),len(self.trees[1])/(1+self.steps*21),self.active].astype(np.float32)

    def _extend(self,which,target):
        tree=self.trees[which];idx=self._nearest(tree,target);q=tree[idx];delta=target-q
        d=float(np.linalg.norm(delta/self.span))
        if d<1e-9:return idx,True,0.
        nxt=q+delta*min(1.,self.cfg['extend_step']/d);self.checks+=1
        if self.backend.check_edge_collision(self.robot,q,nxt).collision:
            self.collisions+=1;return None,False,0.
        # Three samples per accepted edge; the backend separately checks the entire edge.
        distances=[self.clearance(q+(nxt-q)*u) for u in (.0,.5,1.)]
        length=float(np.linalg.norm((nxt-q)/self.span));obs=length*float(np.mean(np.maximum(0,1-np.array(distances)/self.cfg['influence'])**2))
        tree.append(nxt);self.parents[which].append(idx)
        return len(tree)-1,d<=self.cfg['extend_step'],self.cfg['length_weight']*length+self.cfg['obstacle_weight']*obs

    def _branch(self,which,idx):
        branch=[]
        while idx>=0:branch.append(self.trees[which][idx]);idx=self.parents[which][idx]
        return branch[::-1]

    def step(self,action):
        a=np.clip(np.asarray(action),-1,1);c=self.cfg;self.steps+=1
        width=c['range_min']+(a+1)*.5*(c['range_max']-c['range_min'])
        r=self.np_random.random()
        if r<c['global_prob']:sample=self.np_random.uniform(self.lo,self.hi)
        elif r<c['global_prob']+c['goal_prob']:sample=self.trees[1-self.active][0]
        else:sample=np.clip(self.anchor+self.np_random.uniform(-1,1,self.n)*width*self.span,self.lo,self.hi)
        before=self.collisions;idx,_,cost=self._extend(self.active,sample);reward=-c['time_penalty']-cost;info={};done=False
        if idx is not None:
            q=self.trees[self.active][idx]
            for _ in range(c['connect_limit']):
                j,reached,extra=self._extend(1-self.active,q);reward-=extra
                if j is None:break
                if reached:
                    first=self._branch(self.active,idx);second=self._branch(1-self.active,j)
                    path=first+second[-2::-1]
                    if self.active==1:path=path[::-1]
                    # Recheck assembled path; no success for unverified connections.
                    valid=all(not self.backend.check_edge_collision(self.robot,x,y).collision for x,y in zip(path[:-1],path[1:]))
                    if not valid:raise RuntimeError('RRT tree assembled an invalid path')
                    costs=path_costs(path,self.lo,self.span,self.clearance,c['influence'])
                    reward+=c['success_reward']-c['length_weight']*costs['length']-c['obstacle_weight']*costs['obstacle']-c['smooth_weight']*costs['smooth']
                    info.update(result='success',q_path=np.asarray(path).tolist(),costs=costs);done=True;break
        reward-=c['collision_penalty']*(self.collisions-before)
        self.prev=a;self.active=1-self.active;self.anchor=self.trees[self.active][-1];self.last_clearance=self.clearance(self.anchor)
        truncated=not done and self.steps>=self.max_steps
        if truncated:info['result']='timeout'
        if done or truncated:info.update(steps=self.steps,collision_rejections=self.collisions,edge_checks=self.checks,group=self.target['group'],point_index=self.target['point_index'])
        return self._obs(),float(reward),done,truncated,info

    def close(self):
        # Isolated worker processes own the shared PyBullet client.
        pass
