import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'rb_servo_gui/src')
from rb_servo_gui.analysis import load_follow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('rb_servo_gui/docs/report/repeatability');out.mkdir(exist_ok=True)
groups=json.load(open('rb_servo_gui/docs/report/repeatability/matching_runs.json'))
(out/'matching_runs.json').write_text(json.dumps(groups,indent=2))
results=[]
for g in groups:
 c=g['config']
 if c.get('robot_mode',c.get('mode'))!='real':continue
 bins=[]
 for folder in g['paths']:
  p=Path(folder)
  if (p/'traj_ref.csv').exists():
   ref=np.rad2deg(np.loadtxt(p/'traj_ref.csv',delimiter=',',skiprows=1)[:,1:7])
   t,phase,q,_,meta=load_follow(str(next(p.glob('follow_*.csv'))))
   keep=(t>=meta['marks']['servo'])&(t<=meta['marks']['servo_done'])
   q=q[keep]
  else:
   ref=np.vstack([c['base'],np.loadtxt(p/'waypoints.csv',delimiter=',',skiprows=1,ndmin=2)])
   a=np.loadtxt(p/'follow.csv',delimiter=',',skiprows=1,ndmin=2);q=a[:,11:17]
  d=np.diff(ref,axis=0);l=np.linalg.norm(d,axis=1);keep=l>1e-9
  starts=ref[:-1][keep];d=d[keep];l=l[keep];cum=np.r_[0,np.cumsum(l)]
  u=np.clip(np.einsum('nsi,si->ns',q[:,None]-starts,d)/l**2,0,1)
  proj=starts[None]+u[:,:,None]*d[None]
  idx=np.argmin(np.sum((q[:,None]-proj)**2,axis=2),axis=1)
  errors=q-proj[np.arange(len(q)),idx]
  prog=(cum[idx]+u[np.arange(len(q)),idx]*l[idx])/cum[-1]
  edges=np.linspace(.05,.95,91);b=[]
  for lo,hi in zip(edges[:-1],edges[1:]):
   mask=(prog>=lo)&(prog<hi)
   b.append(np.mean(errors[mask],axis=0) if mask.sum()>=2 else np.full(6,np.nan))
  bins.append(np.array(b))
 a,b=bins;valid=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
 strong=valid&(np.linalg.norm(a,axis=1)>.02)&(np.linalg.norm(b,axis=1)>.02)
 cosine=np.sum(a[strong]*b[strong],axis=1)/(np.linalg.norm(a[strong],axis=1)*np.linalg.norm(b[strong],axis=1))
 info=dict(paths=g['paths'],common_bins=int(valid.sum()),direction_bins=int(strong.sum()),median_cosine=float(np.median(cosine)) if len(cosine) else None,positive_direction_ratio=float(np.mean(cosine>0)) if len(cosine) else None)
 info['joints']=[]
 for j in range(6):
  mask=valid&(abs(a[:,j])>.02)&(abs(b[:,j])>.02)
  info['joints'].append(dict(joint=j+1,bins=int(mask.sum()),same_sign=float(np.mean(a[mask,j]*b[mask,j]>0)) if mask.any() else None))
 print(json.dumps(info,ensure_ascii=False));results.append(info)
 if 'rb_servo_gui_runs' in g['paths'][0]:
  fig,axes=plt.subplots(3,2,figsize=(11,8),sharex=True)
  x=(edges[:-1]+edges[1:])/2*100
  for j,ax in enumerate(axes.flat):
   for k,arr in enumerate(bins):ax.plot(x,arr[:,j],label=Path(g['paths'][k]).name[9:15])
   ax.axhline(0,color='gray',lw=.6);ax.set_title(f'J{j+1}');ax.set_ylabel('Signed path error [deg]');ax.grid(alpha=.2);ax.legend()
  for ax in axes[-1]:ax.set_xlabel('Reference arc length [%]')
  fig.suptitle('Two REAL servo runs: nearest-polyline residual, averaged in 1% bins')
  fig.tight_layout();fig.savefig(out/'real_servo_signed_errors.png',dpi=150)
  np.savetxt(out/'real_servo_signed_errors.csv',np.c_[x,a,b],delimiter=',',header='progress_pct,'+','.join(f'run{k}_J{j}' for k in [1,2] for j in range(1,7)),comments='')
(out/'direction_metrics.json').write_text(json.dumps(results,indent=2))
