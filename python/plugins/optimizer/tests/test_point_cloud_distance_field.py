import numpy as np
import pytest

from plugins.optimizer.sdf_field import PointCloudDistanceField


def test_point_distance_and_outside_grid():
    resolution = .01
    field = PointCloudDistanceField([[0, 0, 0]], resolution, .1)
    queries = np.array([[0, 0, 0], [.04, 0, 0], [0, .06, 0]])
    distances = field.distance(queries)
    truth = np.linalg.norm(queries, axis=1)
    assert np.max(np.abs(distances - truth)) <= np.sqrt(3) * resolution
    assert distances[0] == 0
    assert field.distance([[100, 0, 0]])[0] == .1


def test_rigid_translation_of_source_and_queries():
    points = np.array([[0., 0, 0], [.1, .1, .1]])
    q = np.array([[.02, .03, 0], [.09, .08, .1]])
    offset = np.array([4., -2., 1.])
    a = PointCloudDistanceField(points, .01, .1)
    b = PointCloudDistanceField(points + offset, .01, .1)
    np.testing.assert_allclose(a.distance(q), b.distance(q + offset), atol=1e-7)


def test_invalid_points():
    with pytest.raises(ValueError):
        PointCloudDistanceField([])
    with pytest.raises(ValueError):
        PointCloudDistanceField([[np.nan, 0, 0]])
    with pytest.raises(ValueError):
        PointCloudDistanceField([[0, 0, 0]], resolution=0)


@pytest.mark.parametrize("use_scan", [False, True])
def test_pipe_source_and_cad_obstacle_separation(monkeypatch, use_scan):
    from types import SimpleNamespace
    import plugins.optimizer.sdf_field as module
    pipe = SimpleNamespace(vertices=np.array([[0., 0, 0]]))
    cad = object()
    cad_calls = []
    def cad_field(mesh, resolution, cap):
        cad_calls.append(mesh)
        return object()
    monkeypatch.setattr(module, "StaticSDF", cad_field)
    monkeypatch.setattr(module, "link_surface_points", lambda _: {"link": np.zeros((1, 3))})
    model = SimpleNamespace(createData=lambda: None, getFrameId=lambda _: 0)
    scan = np.array([[1., 0, 0]]) if use_scan else None
    field = module.SceneDistanceField(None, model, "unused", [pipe, cad],
                                      resolution=.01, cap=.1, pipe_points=scan)
    assert cad_calls == [cad]
    query = [[1., 0, 0]] if use_scan else [[0., 0, 0]]
    assert field.sdfs[0].distance(query)[0] < 1e-7
    assert field.pipe_source == ("scan_points" if use_scan else "snapshot_vertices")
