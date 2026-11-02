"""Audit saved matching pairs without changing original experimental data."""
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'rb_servo_gui/src'))
from rb_servo_gui.surface import RobotSurface
OUT=Path(__file__).resolve().parent
RUNS=Path('/home/dev/rb_servo_gui_runs')
robot=RobotSurface(str(ROOT/'urdf/rb20_1900es_RT.urdf'),n_points=600,seed=0)
names=sorted(robot.moving-robot.struct)

def displacement(A,q):
    B=robot.points(np.deg2rad(q))
    values=np.array([np.linalg.norm(A[n]-B[n],axis=1).max()*1000 for n in names])
    return values

def nearest(q,ref):
    d=np.diff(ref,axis=0);l2=np.sum(d*d,axis=1);l2=np.maximum(l2,1e-20)
    u=np.clip(np.einsum('nsi,si->ns',q[:,None]-ref[:-1],d)/l2,0,1)
    projected=ref[:-1]+u[:,:,None]*d
    dist=np.sum((q[:,None]-projected)**2,axis=2)
    idx=np.argmin(dist,axis=1)
    return projected[np.arange(len(q)),idx],idx,u[np.arange(len(q)),idx]

paths=sorted((RUNS/'report_dataset_20261008/runs').iterdir())
latest=json.loads((ROOT/'rb_servo_gui/docs/report/img/numbers_servo_profile.json').read_text())
paths += [RUNS/r['run'] for r in latest['rows'] if r['real']]
summary=[]
for p in paths:
    f=next(p.glob('*_pairs.csv'));a=np.loadtxt(f,delimiter=',',skiprows=2,ndmin=2)
    meta=json.loads(f.open().readline()[1:]);q=a[:,4:10];paired=a[:,10:16]
    refarr=np.loadtxt(p/'traj_ref.csv',delimiter=',',skiprows=1);rt=refarr[:,0];ref=np.rad2deg(refarr[:,1:7])
    close,idx,u=nearest(q,ref);near_t=rt[idx]+u*np.diff(rt)[idx]
    old=[];new=[]
    for measured,pm,pn in zip(q,paired,close):
        A=robot.points(np.deg2rad(measured));old.append(displacement(A,pm));new.append(displacement(A,pn))
    old=np.array(old);new=np.array(new);om=old.max(axis=1);nm=new.max(axis=1)
    k=int(om.argmax());j=int(old[k].argmax());l=int(nm.argmax())
    json_path=next(p.glob('compare_*.json'));saved=json.loads(json_path.read_text())
    matching=saved.get('matching') or {};anchors=matching.get('anchors',[])
    row=dict(run=p.name,source=str(p),method=meta.get('method'),matching=meta.get('matching','legacy/unspecified'),
        samples=len(q),dtw_max_mm=float(om[k]),nearest_at_dtw_peak_mm=float(nm[k]),nearest_global_max_mm=float(nm[l]),
        reduction_at_peak_pct=float((1-nm[k]/om[k])*100),dtw_peak_time_s=float(a[k,0]),nearest_peak_time_s=float(a[l,0]),
        peak_link=names[j],same_link_nearest_mm=float(new[k,j]),dtw_angle_deg=float(np.linalg.norm(q[k]-paired[k])),
        nearest_angle_deg=float(np.linalg.norm(q[k]-close[k])),dtw_ref_time_s=float(a[k,1]),nearest_ref_time_s=float(near_t[k]),
        dtw_peak_is_anchor=any(int(m)==k for m,r in anchors),dtw_p95_mm=float(np.percentile(om,95)),nearest_p95_mm=float(np.percentile(nm,95)),
        pairs_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    summary.append(row)
    cols=np.c_[a[:,0],om,nm,a[:,1],near_t,np.linalg.norm(q-paired,axis=1),np.linalg.norm(q-close,axis=1),old,new]
    np.savetxt(OUT/(p.name+'.csv'),cols,delimiter=',',header=','.join(['t_track','dtw_max_mm','nearest_max_mm','dtw_ref_time','nearest_ref_time','dtw_angle_deg','nearest_angle_deg']+['dtw_'+n for n in names]+['nearest_'+n for n in names]),comments='')
    (OUT/(p.name+'_peak.json')).write_text(json.dumps(dict(**row,measured_deg=q[k].tolist(),dtw_ref_deg=paired[k].tolist(),nearest_ref_deg=close[k].tolist()),indent=2))
    print(p.name,round(om[k],3),'->',round(nm[k],3),'nearest max',round(nm.max(),3),flush=True)
with (OUT/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
