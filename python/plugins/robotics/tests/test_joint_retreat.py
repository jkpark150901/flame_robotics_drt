from types import SimpleNamespace
import numpy as np
import pytest
from plugins.robotics.inspection_planning_base import InspectionPlanningBase


def setup_planner(path, collisions=0):
    handle = SimpleNamespace(sample_resolution=0.1)
    backend = SimpleNamespace(robot_handle=lambda name: handle)
    planner = SimpleNamespace(robotics_backend=backend, robotics_robot_name='robot',
                              pin_collision_sample_resolution=0.1, planning_deadline=None)
    def generate(start, goal):
        assert handle.sample_resolution == 0.02
        assert planner.pin_collision_sample_resolution == 0.02
        return path
    planner.generate = generate
    planner.verify_path = lambda path: {'colliding_edges': collisions, 'colliding_waypoints': 0}
    return InspectionPlanningBase(backend), planner, handle


def test_retreat_verified_and_resolution_restored():
    path = [[0, 0], [0, 0.5], [0, 1]]
    service, planner, handle = setup_planner(path)
    result = service.plan_joint_retreat(planner, path[0], path[-1])
    assert result['q_path'] == path
    assert result['status'] == 'success'
    assert handle.sample_resolution == planner.pin_collision_sample_resolution == 0.1
    assert planner.planning_deadline is None


@pytest.mark.parametrize('path,collisions', [([], 0), ([[0, 0], [0, .8]], 0),
                                            ([[0, 0], [0, 1]], 1),
                                            ([[0, 0], [0, float('nan')]], 0)])
def test_retreat_rejects_failed_approximate_or_colliding_path(path, collisions):
    service, planner, handle = setup_planner(path, collisions)
    with pytest.raises(RuntimeError):
        service.plan_joint_retreat(planner, [0, 0], [0, 1])
    assert handle.sample_resolution == planner.pin_collision_sample_resolution == 0.1
    assert planner.planning_deadline is None


def test_playback_stops_before_rotation_if_retreat_planning_fails():
    import ast
    import logging
    from pathlib import Path
    source = Path('python/viewervedo/visualizer.py').read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == '_start_next_inspection_sequence_group_once')
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<playback>', 'exec'), scope)
    class Viewer:
        _positioner_r_deg = -90
        _inspection_sequence_playback = {'sequence': [{'positioner_r_deg': 90}], 'index': 0}
        _Visualizer__console = logging.getLogger('test')
        def _build_retreat_animation_plans(self, *args, **kwargs):
            raise RuntimeError('no safe retreat')
        def _move_positioner_r_to(self, *args, **kwargs):
            pytest.fail('positioner must not rotate after failed retreat')
    # A method compiled outside the class has an unmangled private attribute.
    viewer = Viewer()
    setattr(viewer, '__console', logging.getLogger('test'))
    assert scope[method.name](viewer) is False
    assert viewer._inspection_sequence_playback is None
