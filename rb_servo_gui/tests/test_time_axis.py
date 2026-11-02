import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib.figure import Figure
from rb_servo_gui.time_axis import time_groups, set_time_limits, TimeAxisControls
from rb_servo_gui.analysis import redraw_saved_graph


class TimeAxesTests(unittest.TestCase):
    def test_clocks_are_separate_and_y_unchanged(self):
        fig = Figure()
        a, b, c, spatial = fig.subplots(4)
        for ax in (a,b,c,spatial):
            ax.plot([0,10],[-2,4])
        a.set_xlabel('matched reference time [s]')
        b.set_xlabel('matched reference time [s]')
        c.set_xlabel('tracking time [s]')
        spatial.set_xlabel('PC1 [deg]')
        groups = time_groups(fig)
        y = a.get_ylim()
        before = c.get_xlim()
        set_time_limits(groups['matched reference time'],2,3)
        self.assertEqual(a.get_xlim(), (2,3))
        self.assertEqual(b.get_xlim(), (2,3))
        self.assertEqual(c.get_xlim(), before)
        self.assertEqual(a.get_ylim(), y)
        self.assertNotIn(spatial, sum(groups.values(), []))
        for x,z in [(3,2),(1,1),(float('nan'),2),(0,float('inf'))]:
            with self.assertRaises(ValueError):
                set_time_limits([a],x,z)

    def test_shared_axes_and_twin_axis(self):
        fig = Figure()
        axes = fig.subplots(3,sharex=True)
        axes[-1].set_xlabel('recording time [s]')
        twin=axes[0].twinx()
        group=time_groups(fig)['recording time']
        self.assertEqual(len(group),4)
        set_time_limits(group,1,2)
        self.assertEqual(twin.get_xlim(),(1,2))

    def test_rebuild_saved_comparison_from_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); t=np.linspace(0,2,11)
            np.savetxt(p/'traj_ref.csv',np.c_[t,np.zeros((11,6))],delimiter=',',header='t,q0,q1,q2,q3,q4,q5',comments='')
            data=np.zeros((11,23));data[:,0]=t;data[:,1]=t*.5;data[:,22]=.2
            np.savetxt(p/'compare_follow_test_jb2_pairs.csv',data,delimiter=',',header='# {}\n'+','.join('c'+str(i) for i in range(23)),comments='')
            metrics=dict(t_ref=2,t_track=2,pair_rmse_after=.1,end_err=.1,path_dev_p95=.2,path_dev_max=.2,cover_max=.3,progress_max=1)
            (p/'compare_follow_test_jb2.json').write_text(json.dumps({'metrics':metrics}))
            fig=Figure()
            self.assertTrue(redraw_saved_graph(d,'compare_follow_test_jb2_after.png',fig))
            groups=time_groups(fig)
            self.assertEqual(len(groups['matched reference time']),2)
            self.assertEqual(len(groups['tracking time']),1)
            np.testing.assert_array_equal(fig.axes[1].lines[-1].get_ydata(),data[:,22])
            self.assertFalse(redraw_saved_graph(d,'unknown.png',fig))


class TkTimeControlsTests(unittest.TestCase):
    def test_zoom_redraw_reset_and_clock_selection(self):
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        try:
            root=tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        root.withdraw()
        try:
            fig=Figure(); ax=fig.subplots();ax.plot([0,10],[0,1]);ax.set_xlabel('time [s]')
            canvas=FigureCanvasTkAgg(fig,master=root)
            control=TimeAxisControls(root,canvas)
            control.refresh('test')
            full=ax.get_xlim()
            control._set(2,4)
            control._scroll(SimpleNamespace(xdata=3,inaxes=ax,button='up'))
            np.testing.assert_allclose(ax.get_xlim(),[2.2,3.8])
            fig.clear();ax=fig.subplots();ax.plot([0,10],[0,1]);ax.set_xlabel('time [s]')
            control.refresh('test')
            np.testing.assert_allclose(ax.get_xlim(),[2.2,3.8])
            control.reset()
            np.testing.assert_allclose(ax.get_xlim(),full)
            fig.clear();control.refresh('image')
            self.assertFalse(control.groups)
            self.assertEqual(str(control.widgets[0]['state']),'disabled')
        finally:
            root.destroy()


if __name__=='__main__':
    unittest.main()
