"""Direct rbpodo joint-vector commands; recording on its own data channel."""
import csv
import json
import time
import threading
import numpy as np
from model import servo_plan


def snapshot(channel):
    packet=channel.request_data(1.)
    if packet is None:raise TimeoutError('상태 응답 없음')
    s=packet.sdata
    return dict(time=float(s.time),state=int(s.robot_state),sim=int(s.real_vs_simulation_mode),bar=float(s.default_speed),
                ref=np.array(s.jnt_ref,float,copy=True),ang=np.array(s.jnt_ang,float,copy=True))


def read_current(ip):
    import rbpodo as rb
    return snapshot(rb.CobotData(ip))


def execute(config, points, out, stop, emit):
    import rbpodo as rb
    robot=rb.Cobot(config['ip']);channel=rb.CobotData(config['ip']);rc=rb.ResponseCollector()
    expected=1 if config['mode']=='simulation' else 0
    stage='initial';last_servo=None;response_log=[]
    def on_response(response):
        response_log.append(dict(time=time.time(),stage=stage,response=str(response)))
    rc.set_callback(on_response)
    servo_t2=float(config.get('servo_t2',.1))
    if not np.isfinite(servo_t2) or not .02<=servo_t2<=.2:raise ValueError('servo t2는 0.02~0.2초로 설정하세요')
    def drain_errors():
        robot.flush(rc)
        rc.error().throw_if_not_empty()
    def send_servo(point,dt,phase):
        nonlocal stage,last_servo
        stage=phase
        last_servo=dict(target=np.asarray(point).tolist(),t1=float(dt),t2=servo_t2,gain=config['gain'],alpha=config['alpha'])
        check(robot.move_servo_j(rc,np.asarray(point),float(dt),servo_t2,config['gain'],config['alpha'],timeout=3.))
    rows=[];errors=[];finish=threading.Event();thread=None;schedule=[];jb2_log=[]
    def check(result):
        rc.error().throw_if_not_empty()
        if not result.is_success():raise RuntimeError('rbpodo 명령 실패/시간 초과')
    def guard():
        if stop.is_set():raise InterruptedError('사용자 중지')
        if errors:raise errors[0]
    def read():
        s=snapshot(channel)
        if s['sim']!=expected or abs(s['bar']-1)>.01:raise RuntimeError('모드/speed bar 변경')
        return s
    def reached(target, idle=True, require_motion=False):
        begin=time.perf_counter();stable=None;moved=False
        while time.perf_counter()-begin<config['timeout']:
            guard();drain_errors();s=read();q=s['ref'] if expected else s['ang'];now=time.perf_counter()
            moved=moved or s['state']!=1 or np.max(abs(q-np.asarray(config['base'])))>.05
            if require_motion and rows:
                moved=moved or any(r[2]!=1 or np.max(abs(np.asarray(r[5:11] if expected else r[11:17])-config['base']))>.05 for r in rows)
            ok=(moved or not require_motion) and np.max(abs(q-target))<.05 and (s['state']==1 or not idle)
            stable=(stable if stable is not None else now) if ok else None
            if stable is not None and now-stable>=.3:return
            stop.wait(.02)
        raise TimeoutError('목표 도달 대기 시간 초과')
    def record():
        try:
            dc=rb.CobotData(config['ip']);begin=time.perf_counter()
            while not finish.is_set():
                now=time.perf_counter();s=snapshot(dc);wall=time.time()
                if s['sim']!=expected or abs(s['bar']-1)>.01:raise RuntimeError('기록 중 모드/speed bar 변경')
                rows.append([now-begin,s['time'],s['state'],s['sim'],s['bar'],*s['ref'],*s['ang'],wall])   # 마지막 열 wall_time: 송신·응답 기록과 같은 시계(time.time)
                finish.wait(max(0,1/config['hz']-(time.perf_counter()-now)))
        except Exception as e:errors.append(e)
    before=snapshot(channel)
    if before['state']!=1:raise RuntimeError('로봇이 정지 상태가 아닙니다')
    method=config['method'];label=f"{method} gain={config['gain']:g}" if method=='servo' else f"jb2 blend={config['blend']:g}"
    entry=dict(file='follow.csv',label=label,completed=False)
    try:
        guard();check(robot.set_operation_mode(rc,rb.OperationMode.Simulation if expected else rb.OperationMode.Real,timeout=3.))
        check(robot.set_speed_bar(rc,1.,timeout=3.));stop.wait(.3)
        stage='move_to_start'
        guard();emit('log','시작 자세 이동')
        check(robot.move_j(rc,np.array(config['base']),config['speed'],config['accel'],timeout=3.))
        reached(np.array(config['base']))
        if method=='jb2':
            stage='jb2_clear';t_call=time.time();check(robot.move_jb2_clear(rc,timeout=3.));jb2_log.append(['clear',-1,t_call,time.time()])
            for i,point in enumerate(points):
                guard();stage=f'jb2_add[{i}]';t_call=time.time()
                check(robot.move_jb2_add(rc,point,config['speed'],config['accel'],config['blend'],timeout=3.));jb2_log.append(['add',i,t_call,time.time()])
        thread=threading.Thread(target=record,daemon=True);thread.start()
        emit('log',f'{label}: {len(points)}개 관절 벡터 전송')
        guard()
        if method=='jb2':
            stage='jb2_run';t_call=time.time();row=['run',-1,t_call,float('nan')];jb2_log.append(row)
            check(robot.move_jb2_run(rc,timeout=3.));row[3]=time.time()   # 반환 시각: 응답(ACK)이 모션 종료 뒤에 올 수 있음
        else:
            planned_points,times=servo_plan(config)
            if not np.allclose(planned_points,points):raise ValueError('전송 궤적과 시간표 불일치')
            intervals=np.diff(times);begin=time.perf_counter()
            for i,(point,dt) in enumerate(zip(points,intervals)):
                guard()
                delay=begin+times[i]-time.perf_counter()
                if delay>0:stop.wait(delay)
                guard();sent=time.perf_counter()-begin
                if sent-times[i] > max(.02,float(dt)):
                    raise RuntimeError('servo 전송 지연: servo 주기(ms)를 늘려 주세요')
                row=[times[i],sent,dt,*point,time.time(),float('nan')]   # 끝 두 열: 송신 호출 시작/반환 시각 (time.time)
                schedule.append(row)
                send_servo(point,dt,f'servo[{i}]');row[-1]=time.time()
            stop.wait(max(0,begin+times[-1]-time.perf_counter()));guard()
            # Repeat the final pose once to remove residual commanded velocity.
            send_servo(config['end'],intervals[-1],'servo_final_hold')

        stage='wait_final'
        reached(np.array(config['end']),idle=method=='jb2',require_motion=method=='jb2');guard();drain_errors();entry['completed']=True
    except BaseException as e:
        entry['error']=str(e);entry['error_stage']=stage
        (out/'error.json').write_text(json.dumps(dict(error=str(e),stage=stage,last_servo=last_servo,responses=response_log),ensure_ascii=False,indent=2))
        emit('log',f'오류 단계={stage}: {e} / 마지막 servo={last_servo}')
        try:robot.task_stop(rc,timeout=3.)
        except Exception:pass
        raise
    finally:
        if schedule:
            np.savetxt(out/'servo_schedule.csv',schedule,delimiter=',',header='planned_send_s,actual_send_s,t1,J1,J2,J3,J4,J5,J6,wall_call,wall_return',comments='')
        if jb2_log:
            with (out/'jb2_schedule.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['kind','index','wall_call','wall_return']);w.writerows(jb2_log)
        finish.set()
        if thread:thread.join(timeout=2.)
        with (out/entry['file']).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['host_time','controller_time','state','sim','speed_bar']+[f'ref{j}' for j in range(6)]+[f'ang{j}' for j in range(6)]+['wall_time']);w.writerows(rows)
        (out/'manifest.json').write_text(json.dumps([entry],indent=2))
        try:
            if snapshot(channel)['state']==1:
                stage='restore_speed_bar'
                check(robot.set_speed_bar(rc,before['bar'],timeout=3.));drain_errors()
                stage='restore_mode'
                check(robot.set_operation_mode(rc,rb.OperationMode.Simulation if before['sim'] else rb.OperationMode.Real,timeout=3.));drain_errors()
        except Exception as e:
            entry['restore_error']=str(e);emit('log',f'기존 모드/speed bar 복원 실패 ({stage}): {e}')
        (out/'manifest.json').write_text(json.dumps([entry],ensure_ascii=False,indent=2))
        (out/'responses.json').write_text(json.dumps(response_log,ensure_ascii=False,indent=2))
    emit('result',(label,rows))
