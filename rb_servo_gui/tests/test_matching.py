import unittest
import numpy as np
from rb_servo_gui.matching import extrema_dtw, _dtw_path, _anchor_chain
from rb_servo_gui.analysis import ref_progress, refine_pairs


class ExtremaMatchingTests(unittest.TestCase):
    def test_undershoot_delayed_peak_stays_fixed_after_projection(self):
        rt=np.linspace(0,2,201); mt=np.linspace(0,3,301)
        r=np.interp(rt,[0,1,2],[0,10,0])[:,None]
        m=np.interp(mt,[0,1.8,3],[0,8,0])[:,None]
        result=extrema_dtw(m,r,mt,rt)
        self.assertIn((180,100),result['anchors'])
        _,cum,seg=ref_progress(r)
        pair,s,t,d=refine_pairs(m,r,result['indices'],cum,seg,rt,result['anchors'])
        self.assertEqual(pair[180,0],10)
        self.assertEqual(d[180],2)
        self.assertEqual(t[180],1)
        self.assertTrue(np.all(np.diff(t)>=0))
        np.testing.assert_allclose(pair[:,0],np.interp(t,rt,r[:,0]))
        np.testing.assert_allclose(d,np.linalg.norm(pair-m,axis=1))
        self.assertAlmostEqual(result['extrema'][0]['deficit_deg'],2)

    def test_repeated_peaks_plateaus_and_minima(self):
        r=np.array([0,4,4,4,0,-3,-3,0,5,0.])[:,None]
        m=r*.8
        result=extrema_dtw(m,r)
        rows=[e for e in result['extrema'] if e['status']=='anchor']
        self.assertEqual([e['kind'] for e in rows],['max','min','max'])
        np.testing.assert_allclose([e['deficit_deg'] for e in rows],[.8,.6,1.])

    def test_missing_extremum_is_reported_not_fabricated(self):
        r=np.array([0,1,2,3,2,1,0.])[:,None]
        m=np.linspace(0,1,7)[:,None]
        result=extrema_dtw(m,r)
        self.assertEqual(result['extrema'][0]['status'],'reference_missing')
        self.assertIsNone(result['extrema'][0]['error_deg'])
        self.assertEqual(result['anchors'],[(0,0),(6,6)])

    def test_conflicting_joints_never_reverse_common_clock(self):
        rows=[dict(meas_index=4,ref_index=6,status='candidate'),
              dict(meas_index=6,ref_index=4,status='candidate'),
              dict(meas_index=8,ref_index=8,status='candidate')]
        chain=_anchor_chain(rows)
        self.assertEqual(len(chain),2)
        self.assertEqual(sum(r['status']=='order_conflict' for r in rows),1)
        self.assertTrue(all(a[0]<b[0] and a[1]<b[1] for a,b in zip(chain,chain[1:])))

    def test_no_extrema_dtw_covers_both_sequences(self):
        m=np.linspace(0,1,5)[:,None];r=np.linspace(0,1,21)[:,None]
        result=extrema_dtw(m,r)
        self.assertEqual(set(i for i,j in result['dtw_pairs']),set(range(len(m))))
        self.assertEqual(set(j for i,j in result['dtw_pairs']),set(range(len(r))))
        self.assertTrue(np.all(np.diff(result['indices'])>=0))
        self.assertEqual(result['indices'][0],0)
        self.assertEqual(result['indices'][-1],20)

    def test_vectorized_dtw_matches_standard_optimal_cost(self):
        rng=np.random.default_rng(4)
        for n,m in [(3,7),(8,4),(6,6)]:
            a,b=rng.normal(size=(n,2)),rng.normal(size=(m,2))
            c=np.linalg.norm(a[:,None]-b[None,:],axis=2)
            dp=np.full((n+1,m+1),np.inf);dp[0,0]=0
            for i in range(n):
                for j in range(m):
                    dp[i+1,j+1]=c[i,j]+min(dp[i,j],dp[i,j+1],dp[i+1,j])
            path=_dtw_path(a,b)
            self.assertAlmostEqual(sum(c[i,j] for i,j in path),dp[-1,-1])

    def test_noise_threshold_and_constant_reference(self):
        q=np.array([0,.01,0,.01,0.])[:,None]
        result=extrema_dtw(q,q,prominence=.05)
        self.assertFalse(result['extrema'])
        r=np.zeros((5,1)); result=extrema_dtw(r,r)
        _,cum,seg=ref_progress(r)
        _,s,t,d=refine_pairs(r,r,result['indices'],cum,seg,np.arange(5.),result['anchors'])
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertTrue(np.all(np.diff(t)>=0))
        self.assertTrue(np.all(d==0))

    def test_invalid_thresholds_and_times(self):
        q=np.zeros((3,1))
        for kw in ({'prominence':-1},{'max_shift':0},{'meas_times':[0,0,1]}):
            with self.assertRaises(ValueError): extrema_dtw(q,q,**kw)


if __name__=='__main__': unittest.main()
