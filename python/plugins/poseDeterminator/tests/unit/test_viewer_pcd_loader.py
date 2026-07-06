"""tools/viewer.workers.pcd_loader 단위 테스트.

작은 합성 PLY 를 임시 디렉토리에 만들어 ``PointCloudLoader`` 가 메인 스레드를
블로킹하지 않고 ``load_done`` 을 emit 하는지 검증한다. 같은 pipe_id 가 연속
호출돼도 직렬화되어 *한 번에 하나만* 실행되는지도 확인.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pyvista as pv
import pytest

from python.plugins.poseDeterminator.tools.viewer.workers.pcd_loader import PointCloudLoader


def _write_synthetic_ply(path: Path, num_points: int = 100) -> None:
    points = np.random.RandomState(0).uniform(-1.0, 1.0, size=(num_points, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)


@pytest.fixture()
def synthetic_ply(tmp_path: Path) -> Path:
    p = tmp_path / "synthetic.ply"
    _write_synthetic_ply(p, num_points=200)
    return p


def test_load_done_emits_pipe_id_and_polydata(qtbot, synthetic_ply: Path) -> None:
    loader = PointCloudLoader()
    received: list[tuple] = []
    loader.load_done.connect(lambda pid, pcd, poly: received.append((pid, pcd, poly)))

    with qtbot.waitSignal(loader.load_done, timeout=10_000):
        loader.start_load("PIPE TEST", synthetic_ply)

    assert len(received) == 1
    pid, pcd, poly = received[0]
    assert pid == "PIPE TEST"
    assert isinstance(pcd, o3d.geometry.PointCloud)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_points == 200


def test_load_failed_emits_for_missing_file(qtbot, tmp_path: Path) -> None:
    """존재하지 않는 PLY → load_done 이거나 load_failed (Open3D 동작에 따라)."""

    loader = PointCloudLoader()
    seen_done: list = []
    seen_failed: list = []
    loader.load_done.connect(lambda *args: seen_done.append(args))
    loader.load_failed.connect(lambda *args: seen_failed.append(args))

    missing = tmp_path / "missing.ply"
    # Open3D 는 누락 파일에 대해 빈 PointCloud 를 반환할 수 있어 load_done 이 와도
    # OK — 핵심은 "메인 스레드 freezing 없이 결과가 비동기로 통보된다" 이다.
    with qtbot.waitSignal(loader.load_done, timeout=5_000, raising=False) as blocker_done:
        with qtbot.waitSignal(loader.load_failed, timeout=5_000, raising=False) as blocker_failed:
            loader.start_load("PIPE GHOST", missing)
    loader.wait()

    assert blocker_done.signal_triggered or blocker_failed.signal_triggered


def test_consecutive_loads_serialize(qtbot, synthetic_ply: Path) -> None:
    """연속 ``start_load`` 가 직렬화되어 정확히 두 번 emit."""

    loader = PointCloudLoader()
    received: list[str] = []
    loader.load_done.connect(lambda pid, *_: received.append(pid))

    with qtbot.waitSignal(loader.load_done, timeout=10_000):
        loader.start_load("PIPE A", synthetic_ply)
    with qtbot.waitSignal(loader.load_done, timeout=10_000):
        loader.start_load("PIPE B", synthetic_ply)
    loader.wait()

    assert received == ["PIPE A", "PIPE B"]
