import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
import numpy as np
import controller
from model import linear_motion,servo_plan

class ControllerTests(unittest.TestCase):
    def test_direct_jb2_calls_and_saved_data(self):
        state=dict(q=np.zeros(6),target=np.zeros(6),ticks=10,bar=.5,sim=1)
        calls=[];servo_params=[]
        class Robot:
            def __getattr__(self,name):
                def call(*args,**kwargs):
                    calls.append(name)
                    if name=='flush' and state.get('inject_late') and calls.count('move_servo_j')>=len(points)+1:
                        state['pending_error']=True;state['inject_late']=False
                    if name=='move_servo_j':servo_params.append(args[2:6])
                    if name=='set_speed_bar':state['bar']=args[1]
                    if name=='set_operation_mode':state['sim']=args[1]
                    if name in ('move_j','move_jb2_add','move_servo_j'):state['target']=np.array(args[1])
                    if name in ('move_j','move_jb2_run','move_servo_j'):state['ticks']=0
                    return types.SimpleNamespace(is_success=lambda:True)
                return call
        class RC:
            def set_callback(self,fn):pass
            def error(self):return self
            def throw_if_not_empty(self):
                if state.pop('pending_error',False):raise RuntimeError('568 Servo J parameters outside available range')
        def snapshot(channel):
            state['ticks']+=1
            moving=state['ticks']<5
            if not moving:state['q']=state['target'].copy()
            return dict(time=float(state['ticks']),state=3 if moving else 1,sim=state['sim'],bar=state['bar'],ref=state['q'],ang=state['q'])
        rb=types.SimpleNamespace(Cobot=lambda ip:Robot(),CobotData=lambda ip:None,ResponseCollector=RC,
                                OperationMode=types.SimpleNamespace(Simulation=1,Real=0))
        end=np.array([.3,.15,-.1,.2,.05,-.15]);points=linear_motion(np.zeros(6),end,4)
        config=dict(ip='unused',mode='simulation',base=[0]*6,end=end.tolist(),speed=1,accel=2,hz=100,timeout=4,method='jb2',gain=.1,alpha=.5,blend=50,basis='max_joint',segments=4,via=[])
        with tempfile.TemporaryDirectory() as directory,patch.dict('sys.modules',rbpodo=rb),patch.object(controller,'snapshot',snapshot):
            out=Path(directory);events=[]
            controller.execute(config,points,out,threading.Event(),lambda kind,data:events.append(kind))
            self.assertEqual(calls.count('move_jb2_add'),4)
            self.assertEqual(calls.count('move_jb2_run'),1)
            self.assertEqual(events.count('result'),1)
            self.assertTrue(all(e['completed'] for e in json.loads((out/'manifest.json').read_text())))
            self.assertEqual(sorted(x.name for x in out.glob('*.csv')),['follow.csv','jb2_schedule.csv'])
            with (out/'jb2_schedule.csv').open() as f:rows=list(__import__('csv').DictReader(f))
            self.assertEqual([r['kind'] for r in rows],['clear']+['add']*4+['run'])        # 명령별 송신 호출/반환 시각 (time.time)
            self.assertEqual([int(r['index']) for r in rows if r['kind']=='add'],[0,1,2,3])
            calls_t=[float(r['wall_call']) for r in rows];self.assertEqual(calls_t,sorted(calls_t))
            self.assertTrue(all(float(r['wall_return'])>=float(r['wall_call']) for r in rows))
            head=(out/'follow.csv').read_text().splitlines()[0].split(',')
            self.assertEqual(head[-1],'wall_time')                                           # 기존 열 위치는 그대로, 끝에만 추가
            follow=np.genfromtxt(out/'follow.csv',delimiter=',',skip_header=1)
            self.assertTrue(np.all(np.diff(follow[:,-1])>=0));self.assertTrue(follow[0,-1]>1e9)
            self.assertEqual(state['bar'],.5)
            calls.clear();config['method']='servo';config['servo_period_ms']=10
            points,times=servo_plan(config)
            controller.execute(config,points,out,threading.Event(),lambda kind,data:None)
            self.assertEqual(calls.count('move_servo_j'),len(points)+1)
            self.assertNotIn('move_jb2_add',calls)
            self.assertTrue(all(params[1]==.1 for params in servo_params))
            self.assertTrue((out/'servo_schedule.csv').exists())
            schedule=np.genfromtxt(out/'servo_schedule.csv',delimiter=',',skip_header=1)
            np.testing.assert_allclose(schedule[:,3:9],points)
            self.assertEqual(schedule.shape[1],11)                                           # ... J6, wall_call, wall_return
            self.assertTrue(np.all(schedule[:,10]>=schedule[:,9]));self.assertTrue(np.all(np.diff(schedule[:,9])>=0))
            self.assertTrue((out/'servo_schedule.csv').read_text().splitlines()[0].endswith('wall_call,wall_return'))
            self.assertTrue(json.loads((out/'manifest.json').read_text())[0]['completed'])
            calls.clear();state['inject_late']=True
            with self.assertRaisesRegex(RuntimeError,'568'):
                controller.execute(config,points,out,threading.Event(),lambda kind,data:None)
            self.assertFalse(json.loads((out/'manifest.json').read_text())[0]['completed'])
            error=json.loads((out/'error.json').read_text())
            self.assertEqual(error['stage'],'wait_final')
            self.assertEqual(error['last_servo']['t2'],.1)
            self.assertIn('task_stop',calls)

if __name__=='__main__':unittest.main()
