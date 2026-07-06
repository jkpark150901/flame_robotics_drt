"""PlotView 4-view 의 subplot 경계선 시각 검증.

새 PlotView 는 4개의 독립 ``QtInteractor`` 를 ``QGridLayout`` 으로 묶은
컨테이너 ``QWidget`` 이므로 단일 plotter API (``subplot``, ``screenshot``)
가 없고 ``plotters()`` 를 통해 4개 plotter 에 직접 접근한다.

각 plotter 에 더미 점군을 추가한 뒤 ``plotter.screenshot()`` 으로 quadrant
별 PNG 를 ``tests/visual/_artifacts/`` 에 저장한다. 4 plotter 사이의 검정
분할선은 layout spacing 으로 형성되므로 단일 plotter screenshot 만으로는
보이지 않으며, 전체 화면 분할선 시각 검증은 ``_capture_viewer_window.ps1``
(Win32 PrintWindow) 와 멀티모달 ``Read`` 로 수행한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

# tools.viewer.widgets.plot_view 는 conftest.py 에서 sys.path 가 셋업된 후 import
from python.plugins.poseDeterminator.tools.viewer.widgets.plot_view import PlotView

ARTIFACT_DIR = Path(__file__).resolve().parent / "_artifacts"


def _ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _make_dummy_polydata() -> pv.PolyData:
    n = 8
    xs = np.linspace(-0.1, 0.1, n)
    ys = np.linspace(-0.1, 0.1, n)
    zs = np.linspace(-0.1, 0.1, n)
    pts = np.array(np.meshgrid(xs, ys, zs)).reshape(3, -1).T
    return pv.PolyData(pts)


def test_plotview_renders_four_quadrants(qtbot):
    """각 quadrant plotter 에 더미 점군을 추가하고 VTK off-screen 스크린샷 검증."""

    _ensure_artifact_dir()
    plot_view = PlotView()
    qtbot.addWidget(plot_view)
    plot_view.resize(1200, 800)
    plot_view.show()
    qtbot.waitExposed(plot_view)

    pcd = _make_dummy_polydata()
    for plotter, _view_fn in plot_view.plotters():
        plotter.add_mesh(pcd, color="skyblue", point_size=3)

    plot_view.render()
    qtbot.wait(500)

    for plotter, view_fn in plot_view.plotters():
        png_path = ARTIFACT_DIR / f"plotview_quadrant_{view_fn}.png"
        plotter.screenshot(filename=str(png_path))
        assert png_path.exists(), f"{png_path} 생성 실패"
        # 각 quadrant 는 PlotView 의 1/4 크기라 빈 화면 PNG 도 1KB 정도 됨.
        # 점군이 그려졌는지 sanity check 만 — 2KB 이상이면 의미 있는 픽셀 존재.
        assert png_path.stat().st_size > 2000, f"{png_path} 가 비정상적으로 작음"

    plot_view.close()


def test_plotview_plotters_signature(qtbot):
    """``PlotView.plotters()`` 가 정확히 4쌍을 반환하고 각 view_fn 이 plotter 에 존재한다."""

    plot_view = PlotView()
    qtbot.addWidget(plot_view)
    plot_view.resize(800, 600)
    plot_view.show()
    qtbot.waitExposed(plot_view)

    specs = plot_view.plotters()
    assert len(specs) == 4

    expected_view_fns = {"view_isometric", "view_xy", "view_xz", "view_yz"}
    actual_view_fns = {view_fn for _, view_fn in specs}
    assert actual_view_fns == expected_view_fns

    for plotter, view_fn in specs:
        assert hasattr(plotter, view_fn)
        # 평행 투영 활성 확인 (PlotView.__init__ 에서 모두 활성화)
        assert plotter.camera.parallel_projection is True

    plot_view.close()
