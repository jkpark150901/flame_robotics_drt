"""Recompute matching with current motion-window logic; originals unchanged."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'rb_servo_gui/src'))
from rb_servo_gui.analysis import load_follow,motion_window,ref_progress,refine_pairs
from rb_servo_gui.matching import extrema_dtw
from rb_servo_gui.surface import RobotSurface
out=Path(__file__).resolve().parent
robot=RobotSurface(str(ROOT/'urdf/rb20_1900es_RT.urdf'),n_points=600)
result=[]
for old in json.loads((out/'summary.json').read_text())[-3:]:
 p=Path(old['source']);cfg=json.load(open(p/'config.json'));cfg=cfg.get('effective',cfg)
 t,phase,q,_,meta=load_follow(str(next(p.glob('follow_*.csv'))))
 sel=np.array([x.startswith('servo') and not x.endswith('move_to_start') for x in phase]);t=t[sel];q=q[sel]
 end=np.median(q[t>=t[-1]-max(cfg['reach_hold'],.3)],axis=0)
 start,stop,_=motion_window(t,q,end,cfg['reach_tol'],cfg['move_thr'])
 q=q[start:stop+1];tm=t[start:stop+1]-t[start]
 ref=np.loadtxt(p/'traj_ref.csv',delimiter=',',skiprows=1);rt=ref[:,0];qr=np.rad2deg(ref[:,1:7])
 m=extrema_dtw(q,qr,tm,rt,prominence=cfg.get('extrema_prominence',.05),max_shift=cfg.get('extrema_max_shift',.35))
 _,cum,seg=ref_progress(qr);pair,s,tr,dev=refine_pairs(q,qr,m['indices'],cum,seg,rt,anchors=m['anchors'])
 disp=[]
 for a,b in zip(q,pair):
  A=robot.points(np.deg2rad(a));B=robot.points(np.deg2rad(b))
  disp.append(max(np.linalg.norm(A[n]-B[n],axis=1).max()*1000 for n in robot.moving-robot.struct))
 k=int(np.argmax(disp))
 item=dict(run=p.name,old_track_end_s=old['dtw_peak_time_s'],corrected_track_end_s=float(tm[-1]),corrected_end_angle_error_deg=float(np.linalg.norm(q[-1]-qr[-1])),old_max_mm=old['dtw_max_mm'],corrected_max_mm=float(disp[k]),corrected_peak_time_s=float(tm[k]),corrected_peak_is_anchor=any(i==k for i,j in m['anchors']),corrected_peak_is_endpoint=k in (0,len(q)-1))
 delta=np.diff(qr,axis=0);den=np.maximum(np.sum(delta*delta,axis=1),1e-20)
 u=np.clip(np.sum((q[k]-qr[:-1])*delta,axis=1)/den,0,1)
 projections=qr[:-1]+u[:,None]*delta;nearest=projections[np.argmin(np.sum((projections-q[k])**2,axis=1))]
 A=robot.points(np.deg2rad(q[k]));B=robot.points(np.deg2rad(nearest))
 item['nearest_at_corrected_peak_mm']=float(max(np.linalg.norm(A[n]-B[n],axis=1).max()*1000 for n in robot.moving-robot.struct))
 result.append(item);print(item,flush=True)
 np.savetxt(out/(p.name+'_current_window.csv'),np.c_[tm,disp,dev,tr],delimiter=',',header='t_track,disp_max_mm,pair_dist_deg,ref_time',comments='')
(out/'endpoint_check.json').write_text(json.dumps(result,indent=2))
