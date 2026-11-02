import numpy as np
import open3d as o3d
import pybullet as pb
from scipy.spatial.transform import Rotation

from plugins.robotics.pybullet_scene import PyBulletRobotScene
from plugins.robotics.pybullet_collision import PyBulletURDFLinkCollisionChecker


def test_urdf_link_matches_planning_with_inertial_and_collision_offsets(tmp_path):
    path = tmp_path / 'probe.urdf'
    path.write_text('''<robot name="probe">
      <link name="base"><inertial><origin xyz=".03 .01 .02" rpy=".1 .2 .3"/>
        <mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <joint name="joint" type="revolute"><parent link="base"/><child link="end"/>
        <origin xyz=".2 .1 .3"/><axis xyz="0 0 1"/><limit lower="-3.2" upper="3.2" effort="1" velocity="1"/></joint>
      <link name="end"><inertial><origin xyz=".02 0 .01"/><mass value="1"/>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
        <collision><origin xyz=".12 0 0"/><geometry><box size=".1 .1 .1"/></geometry></collision>
        <collision><origin xyz="-.12 .1 .02"/><geometry><box size=".06 .06 .06"/></geometry></collision>
      </link></robot>''')
    obstacle = o3d.geometry.TriangleMesh.create_box(.08, .08, .08)
    obstacle.translate([.28, .06, .26])
    scene = PyBulletRobotScene(str(path), ['joint'], [])
    scene.set_static_meshes([obstacle])
    checker = PyBulletURDFLinkCollisionChecker([obstacle], ['pipe'])
    token = object()
    checker.register_link(token, path, 'end')
    offset = np.eye(4)
    offset[:3, 3] = [.03, -.02, .01]
    outcomes = []
    try:
        for angle in [0., 1.3, 3., -.5]:
            scene.set_q([angle])
            state = pb.getLinkState(scene.body, scene._link_index['end'],
                                    computeForwardKinematics=True, physicsClientId=scene.client)
            end = checker._transform(state[4], state[5])
            tcp = end @ np.linalg.inv(offset)
            pose = np.r_[tcp[:3, 3], Rotation.from_matrix(tcp[:3, :3]).as_euler('xyz')]
            expected = bool(scene.check([angle]))
            actual = checker(token, pose, offset)
            assert actual == expected
            outcomes.append(actual)
        assert any(outcomes) and not all(outcomes)
    finally:
        checker.close()
        from plugins.robotics.pybullet_collision import remove_static_body
        for body in scene.static_bodies:
            remove_static_body(body, scene.client)
        pb.removeBody(scene.body, physicsClientId=scene.client)


def test_pose_worker_snapshot_contains_planning_meshes():
    import ast
    import copy
    from pathlib import Path
    from types import SimpleNamespace
    source = Path(__file__).resolve().parents[3] / 'viewervedo' / 'visualizer.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == '_inspection_pose_process_snapshot')
    ns = {'np': np, 'copy': copy}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)
    pipe = o3d.geometry.TriangleMesh.create_box(.2, .3, .4)
    pipe.translate([4., 2., 1.])
    positioner = o3d.geometry.TriangleMesh.create_box(.4, .5, .6)
    viewer = SimpleNamespace(_get_spool_points=lambda: np.ones((10, 3)),
        _current_spool_collision_mesh=lambda: pipe,
        _build_positioner_collision_mesh=lambda: positioner,
        _inspection_points=[[4., 2., 1.]], _robot_models=[])
    snapshot = ns['_inspection_pose_process_snapshot'](viewer, {})
    np.testing.assert_array_equal(snapshot['spool_vertices'], np.asarray(pipe.vertices))
    np.testing.assert_array_equal(snapshot['spool_triangles'], np.asarray(pipe.triangles))
    np.testing.assert_array_equal(snapshot['positioner_vertices'], np.asarray(positioner.vertices))
    assert snapshot['inspection_points'] == viewer._inspection_points


def test_pose_service_registers_optimizer_owned_geometry(monkeypatch):
    from types import SimpleNamespace
    import robot_core.pose_service as module
    made = []
    class Optimizer:
        def __init__(self, **kwargs):
            made.append(self)
        def __getattr__(self, name):
            return lambda *args: None
        def set_DDA_geometry(self, mesh, offset):
            self.dda = object()  # optimizer owns a copy, not the input mesh
            return self.dda
        def set_RT_geometry(self, mesh, offset):
            self.rt = object()
            return self.rt
        def set_collision_checker(self, checker):
            self.checker = checker
    expected = object()
    def factory(meshes, names, links):
        assert links[0][2] is made[0].dda
        assert links[1][2] is made[0].rt
        return expected
    backend = SimpleNamespace(
        end_effector_collision_geometry=lambda *a, **kw: (object(), np.eye(4)),
        create_pose_collision_checker=factory)
    monkeypatch.setattr(module, 'EndEffectorPoseOptimizer', Optimizer)
    opt, _, _ = module.PoseDeterminationService({}, backend, [])._optimizer(None)
    assert opt.checker is expected
