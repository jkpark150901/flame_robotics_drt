import numpy as np
import pytest
from plugins.robotics.path_repair import audit_and_repair


def collision(q):
    return .45 < q[0] < .55 and abs(q[1]) < .1


def test_hidden_mid_edge_collision_is_repaired_and_rechecked():
    calls = []
    def replan(a, b, resolution):
        calls.append((a, b))
        return [a, [a[0], .3], [b[0], .3], b]
    path, audit = audit_and_repair([[0, 0], [1, 0]], collision, replan)
    assert len(calls) == 1
    assert len(audit['repairs']) == 1
    assert np.allclose(path[[0, -1]], [[0, 0], [1, 0]])
    assert audit['colliding_edges'] == 0


def test_retry_bound_rejects_still_colliding_replacement():
    with pytest.raises(RuntimeError, match='after 2 repairs'):
        audit_and_repair([[0, 0], [1, 0]], collision, lambda a, b, r: [a, b], max_repairs=2)


def test_colliding_goal_cannot_be_repaired_without_changing_task():
    with pytest.raises(RuntimeError, match='start/goal collision'):
        audit_and_repair([[0, 0], [.5, 0]], collision, lambda *a: pytest.fail('must not replan'))


def test_collision_query_error_propagates():
    def broken(q):
        raise RuntimeError('scene failure')
    with pytest.raises(RuntimeError, match='scene failure'):
        audit_and_repair([[0, 0], [1, 0]], broken, lambda *a: [])


def test_incomplete_repair_is_rejected():
    with pytest.raises(RuntimeError, match='safe anchors'):
        audit_and_repair([[0, 0], [1, 0]], collision, lambda a, b, r: [a, [.8, .3]])


@pytest.mark.parametrize('hit,raises', [(True, False), (False, True), (False, False)])
def test_frame_guard_stops_before_applying_collision_or_query_failure(hit, raises):
    import ast
    import logging
    from pathlib import Path
    from types import SimpleNamespace
    tree = ast.parse(Path('python/viewervedo/visualizer.py').read_text(encoding='utf-8-sig'))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_guard_playback_step')
    scope = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<guard>', 'exec'), scope)
    handle = SimpleNamespace(sample_resolution=.1)
    def check(*args):
        assert handle.sample_resolution == .005
        if raises:
            raise RuntimeError('query failure')
        return SimpleNamespace(collision=hit)
    viewer = SimpleNamespace(_config={}, _playback_collision_scene=lambda name: True,
                             _robotics_backend=SimpleNamespace(robot_handle=lambda n: handle, check_edge_collision=check),
                             _robot_path_playback={}, _inspection_sequence_playback={})
    setattr(viewer, '__console', logging.getLogger('test'))
    rb = {'robot_name': 'robot', 'q_points': [[0, 0], [1, 0]]}
    assert scope[method.name](viewer, rb, [1, 0]) == (not hit and not raises)
    assert handle.sample_resolution == .1
    if hit or raises:
        assert viewer._robot_path_playback is None
        assert viewer._inspection_sequence_playback is None
        assert 'last_safe_q' not in rb


def test_loaded_playback_frame_guard_skips_collision_queries():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    tree = ast.parse(Path('python/viewervedo/visualizer.py').read_text(encoding='utf-8-sig'))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_guard_playback_step')
    scope = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<guard>', 'exec'), scope)
    viewer = SimpleNamespace(_playback_collision_scene=lambda _: pytest.fail('collision scene configured'))
    rb = {'robot_name': 'robot', 'skip_collision_checks': True}
    assert scope[method.name](viewer, rb, [1, 0]) is True


def test_loaded_path_starts_without_simulation_audit():
    import ast
    import logging
    from pathlib import Path
    from types import SimpleNamespace
    tree = ast.parse(Path('python/viewervedo/visualizer.py').read_text(encoding='utf-8-sig'))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_start_multi_robot_path_playback')
    scope = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<playback>', 'exec'), scope)
    calls = []
    model = object()
    viewer = SimpleNamespace(
        _inspection_sequence_playback={'skip_collision_checks': True},
        _find_robot=lambda _: model,
        _robotics_backend=SimpleNamespace(robot_model=lambda _: object()),
        _current_robot_q=lambda *a, **k: np.array([0., 0.]),
        _audit_playback_path=lambda *a: pytest.fail('simulation audit called'),
        _warn_collision_preview_playback=lambda *a: None,
        _clear_collision_highlights=lambda: None,
        _clear_path_playback_marker=lambda: None,
        _build_pin_model_for_robot=lambda _: object(),
        _playback_timing=lambda *a: None,
        _apply_robot_q=lambda *a: None,
        _send_robot_joint_state_update=lambda *a, **k: None,
        _show_robot_tcp_axes=lambda **k: None,
        _update_playback_collision_highlight=lambda rb: calls.append(rb),
        plotter=SimpleNamespace(render=lambda: None),
    )
    setattr(viewer, '__console', logging.getLogger('test'))
    assert scope[method.name](viewer, {'robot': {'q_path': [[0., 0.], [1., 0.]]}})
    assert viewer._robot_path_playback['robot']['skip_collision_checks'] is True
    assert np.array_equal(viewer._robot_path_playback['robot']['q_points'], [[0., 0.], [1., 0.]])
