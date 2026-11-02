import unittest
import numpy as np
from model import motion,profile,analyze,linear_motion,waypoint_times,path_plan,path_profile,servo_plan,ideal_arrival_times,waypoint_proximity

class ModelTests(unittest.TestCase):
    def test_dense_servo_120_degrees(self):
        c=dict(base=[0]*6,end=[120,60,-30,0,0,0],via=[],speed=50,accel=100,basis='max_joint',servo_period_ms=10)
        q,t=servo_plan(c)
        self.assertEqual(len(q),290)
        np.testing.assert_allclose(np.diff(t),.01,atol=1e-12)
        self.assertAlmostEqual(q[0,0],.005)
        self.assertAlmostEqual(np.max(np.diff(q[:,0])),.5)
        np.testing.assert_allclose(q[-1],c['end'])
        np.testing.assert_allclose(q[:,1],q[:,0]/2)
        c['via']=[c['end']];c['end']=[0]*6
        q,t=servo_plan(c)
        np.testing.assert_allclose(q[289],c['via'][0]);np.testing.assert_allclose(q[-1],0)
        self.assertEqual(len(q),580)

    def test_multi_via_turn_preserved_and_timed(self):
        c=dict(base=[0]*6,end=[0]*6,via=[[30,15,-10,20,5,-15],[-10,-5,4,-8,2,5]],segments=100,speed=10,accel=20,basis='max_joint')
        points,times,anchors=path_plan(c)
        self.assertEqual(points.shape,(100,6));self.assertEqual(len(times),101)
        self.assertTrue(np.all(np.diff(times)>0))
        np.testing.assert_allclose(points[anchors[:-1]-1],c['via'])
        np.testing.assert_allclose(points[-1],c['end'])
        t,q,v=path_profile(c)
        for anchor in anchors[:-1]:
            i=np.argmin(abs(t-times[anchor]))
            self.assertAlmostEqual(t[i],times[anchor])
            np.testing.assert_allclose(v[i],0,atol=1e-10)
        self.assertTrue(np.any(v[:,0]<0))
        c['segments']=2
        with self.assertRaises(ValueError):path_plan(c)
        c['segments']=100;c['via']=[[0]*6]
        with self.assertRaises(ValueError):path_plan(c)

    def test_linear_all_joint_progress_and_schedule(self):
        base=np.array([10.,20.,30.,40.,50.,60.]);end=base+[30,15,-10,20,5,-15]
        points=linear_motion(base,end,100)
        np.testing.assert_allclose((points-base)/(end-base),np.repeat(np.linspace(.01,1,100)[:,None],6,axis=1))
        times=waypoint_times(base,end,100,10,20)
        self.assertAlmostEqual(times[-1],3.5)
        self.assertTrue(np.all(np.diff(times)>0))
        t,q,v=profile(base,points,10,20,stops='end',dt=.001)
        for j in range(6):np.testing.assert_allclose(np.interp(times[1:],t,q[:,j]),points[:,j],atol=3e-6)

    def test_trapezoid_time_and_acceleration(self):
        base=np.zeros(6);points,_,_=motion(base,[1,.5,0,0,0,0],30,100,'reversal')
        t,q,v=profile(base,points,10,20,split=50)
        self.assertAlmostEqual(t[-1],7.) # each leg 2*.5 + 2.5 seconds
        self.assertAlmostEqual(np.max(v[:,0]),10)
        np.testing.assert_allclose(v[:,1],v[:,0]/2)
        self.assertLessEqual(np.max(np.abs(np.diff(v[:,0])/np.diff(t))),20+1e-8)
        np.testing.assert_allclose(q[-1],base,atol=1e-10)
        self.assertAlmostEqual(v[np.argmin(abs(t-3.5)),0],0)

    def test_triangle_and_waypoint_stops(self):
        base=np.zeros(6);points,_,_=motion(base,[1,0,0,0,0,0],1,2,'reversal')
        t,q,v=profile(base,points,10,4,split=1)
        self.assertAlmostEqual(t[-1],2.)
        self.assertAlmostEqual(v[:,0].max(),2.)
        points,_,_=motion(base,[1,0,0,0,0,0],30,100,'reversal')
        t,q,v=profile(base,points,10,20,stops='waypoint')
        self.assertAlmostEqual(t[-1],100*2*np.sqrt(.6/20))
        self.assertGreater(np.sum(np.max(abs(v),axis=1)<1e-9),99)

    def test_l2_and_corner(self):
        base=np.zeros(6);points,peak,end=motion(base,[1,.5,0,0,0,0],30,101,'corner')
        self.assertEqual(len(points),101)
        np.testing.assert_allclose(points[49],peak)
        np.testing.assert_allclose(end,[0,30,0,0,0,0])
        t,q,v=profile(base,points,10,20,basis='l2',split=50)
        self.assertAlmostEqual(np.linalg.norm(v,axis=1).max(),10)
        self.assertTrue(np.any(v[:,0]<0));self.assertTrue(np.all(v[:,1]>=0))

    def test_extremum_deficit_not_time_warped(self):
        t=np.linspace(0,3,301);q=np.zeros((301,6));q[:,0]=np.interp(t,[0,1.8,3],[0,8,0])
        a=np.c_[t,t,np.ones((len(t),3)),q,q]
        c=dict(mode='simulation',window=.1,base=[0]*6,peak=[10,0,0,0,0,0],end=[0]*6)
        time,q,v,events=analyze(a,c)
        self.assertEqual(events[0]['deficit'],2)
        self.assertTrue(events[0]['observed'])
        self.assertAlmostEqual(time[-1]-time[0],3.)


class WaypointProximityTests(unittest.TestCase):
    def test_ideal_arrival_times_match_profile(self):
        c=dict(base=[0]*6,end=[0]*6,via=[[30,15,-10,20,5,-15],[-10,-5,4,-8,2,5]],speed=10,accel=20,basis='max_joint')
        t,q,_=path_profile(c,dt=.01)
        arrival=ideal_arrival_times(c)
        vertices=np.vstack([c['base'],c['via'],c['end']])
        self.assertEqual(len(arrival),4);self.assertAlmostEqual(arrival[0],0)
        for time,pose in zip(arrival[1:],vertices[1:]):
            i=int(np.argmin(np.abs(t-time)));np.testing.assert_allclose(q[i],pose,atol=.3)
        self.assertAlmostEqual(arrival[-1],t[-1],delta=.011)

    def test_exact_pass_and_corner_cut(self):
        t=np.linspace(0,4,401)  # 0.01 s 샘플, 속도 10 deg/s
        x=np.where(t<2,10*t,20-10*(t-2))   # 0 -> 20 -> 0 왕복 (J1)
        q=np.zeros((len(t),6));q[:,0]=x
        poses=np.array([[0]*6,[20,0,0,0,0,0],[0]*6],float)
        r=waypoint_proximity(t,q,poses)
        np.testing.assert_allclose(r['dist'],0,atol=1e-9)
        self.assertAlmostEqual(r['time'][1],2.0,places=2)
        self.assertLess(r['time'][0],.05);self.assertGreater(r['time'][2],3.95)   # 시작/목표가 서로 다른 끝에 배정됨
        q[:,0]=np.minimum(x,18.0)           # 코너를 깎아 18 에서 반전
        r=waypoint_proximity(t,q,poses)
        self.assertAlmostEqual(r['dist'][1],2.0,places=6)
        np.testing.assert_allclose(r['error'][1],[-2,0,0,0,0,0],atol=1e-9)    # 측정 - 목표

    def test_sub_sample_accuracy(self):
        t=np.array([0,1.,2.]);q=np.zeros((3,6));q[:,0]=[0,10,20]   # 샘플이 성긴 직선
        r=waypoint_proximity(t,q,np.array([[7.5,3,0,0,0,0]]))       # 샘플 사이 + 옆으로 3 deg
        self.assertAlmostEqual(r['dist'][0],3.0);self.assertAlmostEqual(r['time'][0],.75);self.assertAlmostEqual(r['pose'][0][0],7.5)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):waypoint_proximity([0.],np.zeros((1,6)),np.zeros((1,6)))
        with self.assertRaises(ValueError):waypoint_proximity([0,1.],np.zeros((2,6)),np.zeros((1,5)))


if __name__=='__main__':unittest.main()
