import numpy as np
import pybullet as pb
from plugins.robotics.pybullet_scene import PyBulletRobotScene
from plugins.robotics.inspection_workflow import inspection_group_pose_items


def test_rt_is_first():
    items = inspection_group_pose_items({'dda_pose': np.eye(4), 'rt_pose': np.eye(4)})
    assert [item[1] for item in items] == ['RT', 'DDA']


def test_peer_collision_tracks_joints_base_and_inertial_offset(tmp_path):
    path = tmp_path / 'robot.urdf'
    path.write_text('''<robot name="probe"><link name="base">
      <inertial><origin xyz=".12 0 0"/><mass value="1"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <joint name="slide" type="prismatic"><parent link="base"/><child link="end"/>
      <axis xyz="1 0 0"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
      <link name="end"><collision><geometry><box size=".2 .2 .2"/></geometry></collision></link></robot>''')
    scene = PyBulletRobotScene(str(path), ['slide'], [])
    try:
        base = np.eye(4); base[0, 3] = 1
        scene.set_peer_robot('peer', path, ['slide'], [0], base)
        assert not scene.collides([0])
        assert scene.collides([1])
        assert scene.check([1]) == [('end_0', 'end_0')]
        scene.set_peer_robot('peer', path, ['slide'], [1], base)
        assert not scene.collides([1])
        assert scene.collides([2])
        base[:3, :3] = np.diag([-1, -1, 1])
        scene.set_peer_robot('peer', path, ['slide'], [1], base)
        assert scene.collides([0])
    finally:
        for peer, _ in scene.peer_robots.values():
            pb.removeBody(peer.body, physicsClientId=scene.client)
        pb.removeBody(scene.body, physicsClientId=scene.client)


def test_playback_queues_rt_before_dda_even_for_old_plan_order():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    tree = ast.parse(Path('python/viewervedo/visualizer.py').read_text(encoding='utf-8-sig'))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_start_group_plans')
    ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<test>', 'exec'), ns)
    calls = []
    viewer = SimpleNamespace(_start_multi_robot_path_playback=lambda plans, **kw: calls.append(plans) or True)
    state = {}
    assert ns[method.name](viewer, {'plans': {'dda_rb10_1300e': {}, 'rb20_1900es': {}}}, state)
    assert list(calls[0]) == ['rb20_1900es']
    assert list(state['robot_queue'][0]) == ['dda_rb10_1300e']


def test_pending_rotation_waits_for_second_retreat():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    tree = ast.parse(Path('python/viewervedo/visualizer.py').read_text(encoding='utf-8-sig'))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_start_next_inspection_sequence_group_once')
    ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<test>', 'exec'), ns)
    calls = []
    pending = {'group': {'positioner_r_deg': 90}, 'idx': 0}
    state = {'pending_group': pending, 'robot_queue': [{'dda_rb10_1300e': {}}]}
    viewer = SimpleNamespace(_inspection_sequence_playback=state,
        _start_multi_robot_path_playback=lambda plans, **kw: calls.append(plans) or True)
    assert ns[method.name](viewer)
    assert state['pending_group'] is pending
    assert list(calls[0]) == ['dda_rb10_1300e']
