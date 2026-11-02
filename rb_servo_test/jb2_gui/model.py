"""Offline trajectory, ideal trapezoidal/triangular velocity, measured velocity."""
import numpy as np


def motion(base, scales, amplitude, segments, kind):
    base, scales = np.asarray(base, float), np.asarray(scales, float)
    if base.shape != (6,) or scales.shape != (6,) or not np.isfinite([base, scales]).all() or not np.any(scales):
        raise ValueError('시작 자세/비율은 유한한 6개 값, 비율은 비영 벡터여야 합니다')
    if not np.isfinite(amplitude) or amplitude <= 0 or segments < 2:
        raise ValueError('진폭 > 0, 분할 수 >= 2 필요')
    peak = base + amplitude*scales
    if kind == 'reversal':
        end = base.copy()
    elif kind == 'corner':
        end = base+2*amplitude*scales
        end[np.flatnonzero(scales)[0]] = base[np.flatnonzero(scales)[0]]
    else:
        raise ValueError('경로는 reversal 또는 corner')
    n = segments//2
    points = np.vstack([np.linspace(base, peak, n+1)[1:], np.linspace(peak, end, segments-n+1)[1:]])
    return points, peak, end


def profile(base, points, speed, accel, basis='max_joint', stops='turn', split=None, dt=.01):
    """Synchronized scalar path. Rest-to-rest each leg or each waypoint.

    No controller blending model, speed override, jerk or joint-specific limits.
    """
    if min(speed, accel, dt) <= 0 or not np.isfinite([speed, accel, dt]).all():
        raise ValueError('속도/가속도/시간 간격은 유한한 양수 필요')
    if basis not in ('max_joint', 'l2') or stops not in ('turn', 'waypoint', 'end'):
        raise ValueError('프로파일 기준 설정 오류')
    points=np.asarray(points,float)
    if stops == 'end':
        vertices=np.vstack([base,points[-1]])
    elif stops == 'turn':
        if split is None or not 0 < split < len(points): raise ValueError('반전점 인덱스 필요')
        vertices=np.vstack([base,points[split-1],points[-1]])
    else:
        vertices=np.vstack([base,points])
    times=[]; positions=[]; velocities=[]; offset=0.
    for start,end in zip(vertices[:-1],vertices[1:]):
        delta=end-start
        length=float(np.max(np.abs(delta)) if basis=='max_joint' else np.linalg.norm(delta))
        if length<=1e-12:continue
        direction=delta/length
        vp=min(speed,np.sqrt(accel*length))
        ta=vp/accel; cruise=max(0.,length/vp-ta); duration=2*ta+cruise
        t=np.linspace(0,duration,max(3,int(np.ceil(duration/dt-1e-10))+1))
        v=np.where(t<ta,accel*t,np.where(t<ta+cruise,vp,accel*(duration-t)))
        s=np.where(t<ta,.5*accel*t*t,np.where(t<ta+cruise,.5*vp*ta+vp*(t-ta),length-.5*accel*(duration-t)**2))
        sl=slice(None) if not times else slice(1,None)
        times.extend((offset+t[sl]).tolist());positions.extend((start+s[sl,None]*direction).tolist());velocities.extend((v[sl,None]*direction).tolist())
        offset+=duration
    return np.array(times),np.array(positions),np.array(velocities)


def velocity(t,q,window=.1):
    t,q=np.asarray(t),np.asarray(q)
    if len(t)<3 or np.any(np.diff(t)<=0):raise ValueError('증가하는 시간의 3개 이상 샘플 필요')
    v=np.zeros_like(q)
    for i,ti in enumerate(t):
        lo=np.searchsorted(t,ti-window/2)
        hi=min(len(t)-1,np.searchsorted(t,ti+window/2,side='right')-1)
        if hi>lo:v[i]=(q[hi]-q[lo])/(t[hi]-t[lo])
    return v


def analyze(rows, config):
    a=np.asarray(rows,float)
    q=a[:,5:11] if config['mode']=='simulation' else a[:,11:17]
    t=a[:,0];v=velocity(t,q,config['window'])
    # Start alignment only, no DTW or time scaling of the theoretical curve.
    active=np.max(np.abs(q-q[0]),axis=1)>.05
    onset=max(0,int(np.flatnonzero(active)[0])-1) if active.any() else 0
    t=t-t[onset]
    base=np.asarray(config['base']);peak=np.asarray(config.get('peak',config['end']));end=np.asarray(config['end'])
    events=[]
    for j in np.flatnonzero((peak-base)*(end-peak)<0):
        sign=np.sign(peak[j]-base[j]);i=int(np.argmax(sign*q[:,j]))
        found=0<i<len(t)-1 and sign*(q[i,j]-q[0,j])>.05 and sign*(q[i,j]-q[-1,j])>.05
        events.append(dict(joint=int(j+1),observed=bool(found),target=float(peak[j]),measured=float(q[i,j]),
                           deficit=float(sign*(peak[j]-q[i,j])) if found else None,time=float(t[i]),speed=float(v[i,j])))
    return t,q,v,events


def linear_motion(base, end, segments):
    base,end=np.asarray(base,float),np.asarray(end,float)
    if base.shape!=(6,) or end.shape!=(6,) or not np.isfinite([base,end]).all():
        raise ValueError('시작/목표 자세는 유한한 6개 각도 필요')
    if not isinstance(segments,int) or isinstance(segments,bool) or segments<1 or np.max(abs(end-base))<1e-6:
        raise ValueError('분할 수 >=1, 시작과 다른 목표 자세 필요')
    return np.linspace(base,end,segments+1)[1:]


def waypoint_times(base,end,segments,speed,accel,basis='max_joint'):
    delta=np.asarray(end)-np.asarray(base)
    length=float(np.max(abs(delta)) if basis=='max_joint' else np.linalg.norm(delta))
    if min(length,speed,accel)<=0:raise ValueError('거리/속도/가속도 >0 필요')
    vp=min(speed,np.sqrt(accel*length));ta=vp/accel;da=.5*vp*ta
    duration=length/vp+ta
    s=np.linspace(0,length,segments+1)
    times=np.empty_like(s)
    first=s<=da;last=s>=length-da;middle=~(first|last)
    times[first]=np.sqrt(2*s[first]/accel)
    times[last]=duration-np.sqrt(2*(length-s[last])/accel)
    times[middle]=ta+(s[middle]-da)/vp
    return times


def path_plan(config):
    """Total N intervals allocated by leg length; exact via poses retained."""
    vertices=np.asarray([config['base'],*config.get('via',[]),config['end']],float)
    if vertices.ndim!=2 or vertices.shape[1]!=6 or not np.isfinite(vertices).all():
        raise ValueError('시작/경유/목표 자세는 유한한 6개 각도 필요')
    delta=np.diff(vertices,axis=0)
    lengths=np.max(abs(delta),axis=1) if config['basis']=='max_joint' else np.linalg.norm(delta,axis=1)
    if np.any(lengths<1e-6):raise ValueError('연속한 두 자세가 같습니다. 중복 경유점을 삭제하세요.')
    n=config['segments'];legs=len(lengths)
    if not isinstance(n,int) or isinstance(n,bool) or n<legs:
        raise ValueError(f'전체 분할 수는 경로 구간 수({legs}) 이상이어야 합니다')
    share=(n-legs)*lengths/lengths.sum();counts=np.floor(share).astype(int)+1
    order=np.argsort(-(share-np.floor(share)),kind='stable')
    counts[order[:n-int(counts.sum())]]+=1
    points=[];times=[0.]
    for start,end,count in zip(vertices[:-1],vertices[1:],counts):
        points.extend(linear_motion(start,end,int(count)))
        local=waypoint_times(start,end,int(count),config['speed'],config['accel'],config['basis'])
        times.extend(times[-1]+local[1:])
    return np.asarray(points),np.asarray(times),np.cumsum(counts)


def path_profile(config, dt=.01):
    # Rest-to-rest at each user via pose, NOT at each interpolated sample.
    controls=np.asarray([*config.get('via',[]),config['end']],float)
    return profile(config['base'],controls,config['speed'],config['accel'],config['basis'],'waypoint',dt=dt)


def servo_plan(config):
    period=float(config.get('servo_period_ms',10.))/1000
    if not np.isfinite(period) or not .002 <= period <= .2:
        raise ValueError('servo 주기는 2~200ms로 설정하세요')
    t,q,_=path_profile(config,dt=period)
    if len(t)>200001:raise ValueError('servo 샘플 수가 너무 많습니다')
    return q[1:],t


def ideal_arrival_times(config):
    """이상 프로파일(경유점마다 정지)에서 [시작, 경유..., 목표] 각 자세의 도달 시각 [s]. profile(stops='waypoint')과 같은 계산."""
    vertices=np.asarray([config['base'],*config.get('via',[]),config['end']],float)
    if config['basis'] not in ('max_joint','l2'):raise ValueError('프로파일 기준 설정 오류')
    times=[0.]
    for start,end in zip(vertices[:-1],vertices[1:]):
        delta=end-start
        length=float(np.max(np.abs(delta)) if config['basis']=='max_joint' else np.linalg.norm(delta))
        if length<=1e-12:times.append(times[-1]);continue
        vp=min(config['speed'],np.sqrt(config['accel']*length))
        times.append(times[-1]+length/vp+vp/config['accel'])
    return np.asarray(times)


def waypoint_proximity(t, q, poses):
    """실측 경로가 각 웨이포인트에 얼마나 가까이 갔는지. 시간은 보지 않고 6관절 공간의 거리만 본다.

    측정 샘플 사이를 선분으로 이어(샘플 간격보다 정확하게) 각 웨이포인트까지의 최단 거리를 구한다.
    웨이포인트는 주어진 순서대로 경로를 따라 지나가야 한다(선분 번호가 줄어들지 않음). 순서 제약 아래 거리 합이 최소가 되도록
    배정해서, 시작=목표인 왕복 경로에서도 시작/목표가 같은 지점에 배정되지 않는다.
    반환: dict(dist (W,), time (W,), pose (W,6), error (W,6)=측정-목표, segment (W,)).
    """
    t,q,poses=np.asarray(t,float),np.asarray(q,float),np.asarray(poses,float)
    if q.ndim!=2 or len(q)<2 or len(t)!=len(q) or poses.ndim!=2 or poses.shape[1]!=q.shape[1] or len(poses)<1:
        raise ValueError('측정 경로(2샘플 이상)와 웨이포인트 자세가 필요합니다')
    a,ab=q[:-1],np.diff(q,axis=0);den=np.maximum((ab*ab).sum(1),1e-18)
    W,S=len(poses),len(ab)
    pa=poses@ab.T-(a*ab).sum(1)[None]                                   # (P-a)·ab, (W,S)
    u=np.clip(pa/den[None],0,1)
    d2=(poses*poses).sum(1)[:,None]-2*poses@a.T+(a*a).sum(1)[None]-2*u*pa+u*u*den[None]
    dist=np.sqrt(np.maximum(d2,0))                                      # 선분 위 최단 거리, (W,S) — (W,S,6) 배열은 만들지 않음
    cost=np.empty((W,S));back=np.zeros((W,S),int);cost[0]=dist[0];cols=np.arange(S)
    for k in range(1,W):
        running=np.minimum.accumulate(cost[k-1])
        back[k]=np.maximum.accumulate(np.where(cost[k-1]<=running,cols,0))   # 선분 i 이하에서 이전 웨이포인트의 최적 위치
        cost[k]=dist[k]+running
    seg=np.zeros(W,int);seg[-1]=int(np.argmin(cost[-1]))
    for k in range(W-1,0,-1):seg[k-1]=back[k,seg[k]]
    idx=np.arange(W);uu=u[idx,seg];pose=a[seg]+uu[:,None]*ab[seg]
    return dict(dist=dist[idx,seg],time=t[seg]+uu*(t[seg+1]-t[seg]),pose=pose,error=pose-poses,segment=seg)
