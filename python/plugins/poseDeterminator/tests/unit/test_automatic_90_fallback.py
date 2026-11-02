import numpy as np
import pytest
from python.plugins.poseDeterminator.EndEffectorPoseOptimizer import EndEffectorPoseOptimizer


@pytest.mark.parametrize('valid, force, expected', [
    ({0, 6, 8}, False, {'0', '90'}),
    ({0, 8, 16}, False, {'0', '120', '240'}),
    ({0, 6, 8, 16}, True, {'0', '90'}),
    ({0}, False, set()),
])
def test_triple_then_automatic_quarter_turn(monkeypatch, valid, force, expected):
    opt = EndEffectorPoseOptimizer(debug_mode=False)
    prefix = '_EndEffectorPoseOptimizer__'
    setattr(opt, prefix+'dda_mesh', object())
    setattr(opt, prefix+'dda_invers_transform_mat', np.eye(4))
    monkeypatch.setattr(opt, prefix+'dda_candidate_centerline_radius', lambda *a: .1)
    monkeypatch.setattr(opt, prefix+'calculate_dda_pose_candidate',
                        lambda p,r,n: np.array([[i,0,0,0,0,0] for i in range(n)],dtype=float))
    monkeypatch.setattr(opt, prefix+'adjust_dda_candidates_for_mesh_surface_distance', lambda p,*a:p)
    monkeypatch.setattr(opt, prefix+'check_collision', lambda m,p,t: int(p[0]) not in valid)
    monkeypatch.setattr(opt, prefix+'process_dda_rt_combination', lambda p,*a: {'DDA':p.tolist(),'RT1':p.tolist()})
    monkeypatch.setattr(opt, prefix+'rt_slot_facing_angles_deg', lambda slot,**kw:[45 if slot['DDA'][0]==0 else -45])
    _, groups = opt.calculate_DDA_RT_pose_for_taking_xray_3pair_120(
        (0,0,0), .01,.3,10,candidate_step_deg=15,gap_tolerance_deg=0,force_90_fallback=force)
    assert (set(groups[0]) if groups else set()) == expected
    if expected == {'0','90'}:
        assert groups[0]['0']['DDA'][0] == 0
        assert groups[0]['90']['DDA'][0] == 6
