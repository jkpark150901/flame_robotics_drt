from types import SimpleNamespace
import gymnasium as gym
import numpy as np
from rl.rrt_range_env import RRTRangeEnv,path_costs


def test_path_cost_is_resampling_invariant_and_rewards_clearance():
    kwargs=dict(lo=np.zeros(2),span=np.ones(2),clearance=lambda q:.2)
    x=path_costs([[0,0],[1,0]],**kwargs)
    y=path_costs([[0,0],[.2,0],[.7,0],[1,0]],**kwargs)
    assert np.isclose(x['length'],y['length']) and np.isclose(x['obstacle'],y['obstacle'])
    assert x['smooth']<1e-20
    near=path_costs([[0,0],[1,0]],np.zeros(2),np.ones(2),lambda q:.05)
    assert near['obstacle']>x['obstacle']
    turn=path_costs([[0,0],[.5,.5],[1,0]],**kwargs)
    assert turn['smooth']>x['smooth']


def make_env(block=False):
    env=RRTRangeEnv.__new__(RRTRangeEnv);gym.Env.__init__(env)
    env.n=2;env.lo=np.zeros(2);env.hi=np.ones(2);env.span=np.ones(2);env.robot='fake';env.max_steps=4
    env.cfg=dict(influence=.3,range_min=.02,range_max=.4,global_prob=.15,goal_prob=.1,extend_step=.1,connect_limit=20,obstacle_weight=5.,length_weight=1.,smooth_weight=.01,success_reward=100.,collision_penalty=2.,time_penalty=.05)
    env.targets=[dict(start=[.1,.1],goal=[.9,.9],group='test',point_index=0)]
    env._activate=lambda target:None;env.clearance=lambda q:.3
    env.backend=SimpleNamespace(check_collision=lambda *a:SimpleNamespace(collision=False),check_edge_collision=lambda *a:SimpleNamespace(collision=block))
    return env


def test_connected_path_preserves_endpoints_for_both_tree_orders():
    for active in [0,1]:
        env=make_env();env.reset(seed=42);env.active=active;env.anchor=env.trees[active][0]
        _,r,done,trunc,info=env.step(np.zeros(2))
        assert done and not trunc
        assert np.allclose(info['q_path'][0],[.1,.1]) and np.allclose(info['q_path'][-1],[.9,.9])


def test_blocked_extensions_never_insert_colliding_nodes_or_claim_success():
    env=make_env(True);env.reset(seed=42)
    for i in range(4):
        _,_,done,trunc,info=env.step(np.zeros(2));assert not done
    assert trunc and info['result']=='timeout'
    assert list(map(len,env.trees))==[1,1]
