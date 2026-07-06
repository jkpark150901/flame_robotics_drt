"""4개의 독립 ``QtInteractor`` 를 ``QGridLayout`` 으로 묶은 4-view 메인 뷰.

기존에는 단일 ``QtInteractor(shape=(2,2))`` 를 사용했으나 PyVista 의 native
``border`` 옵션이 OpenGL 렌더링 경로에는 반영되지 않아 사용자 화면에서
4 subplot 사이 분할선이 보이지 않는 문제가 있었다 (off-screen
``screenshot()`` 에는 보임). 4개의 독립 plotter 를 ``QGridLayout`` spacing
+ 위젯 자체 검정 배경으로 묶으면 layout gap 이 자연스러운 분할선이 되어
렌더 경로와 무관하게 시각적 분리가 보장된다.

각 quadrant 는 독립 카메라/뷰포트를 가지며 ``view_isometric`` /
``view_xy`` / ``view_xz`` / ``view_yz`` 프리셋으로 초기화된다.
"""

from __future__ import annotations

from typing import Any, Iterable

from PyQt6.QtWidgets import QGridLayout, QWidget
from pyvistaqt import QtInteractor

from ..viz import scene_builder


class PlotView(QWidget):
    """4-view (iso/XY/XZ/YZ) 메인 3D 뷰. 4 plotter + 검정 분할선."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # 위젯 자체 배경을 검정으로 → grid spacing 이 검정 분할선처럼 보인다.
        self.setStyleSheet("background-color: black;")

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._iso = QtInteractor(self)
        self._xy = QtInteractor(self)
        self._xz = QtInteractor(self)
        self._yz = QtInteractor(self)

        layout.addWidget(self._iso, 0, 0)
        layout.addWidget(self._xy, 0, 1)
        layout.addWidget(self._xz, 1, 0)
        layout.addWidget(self._yz, 1, 1)

        for plotter, view_fn in self.plotters():
            plotter.set_background("white")
            getattr(plotter, view_fn)()
            plotter.add_axes()
            # CAD 스타일 평행 투영.
            plotter.camera.parallel_projection = True

    def plotters(self) -> list[tuple[Any, str]]:
        """``[(plotter, view_fn_name), ...]`` 4쌍을 반환."""

        return [
            (self._iso, "view_isometric"),
            (self._xy, "view_xy"),
            (self._xz, "view_xz"),
            (self._yz, "view_yz"),
        ]

    def clear_all(self) -> None:
        for plotter, _ in self.plotters():
            plotter.clear()

    def render(self) -> None:  # type: ignore[override]
        for plotter, _ in self.plotters():
            plotter.render()

    def set_pipe_only(
        self,
        scan_polydata: Any,
        target_point,
    ) -> None:
        scene_builder.render_pipe(self, scan_polydata, target_point)
        self.render()

    def set_full_result(
        self,
        scan_polydata: Any,
        target_point,
        pose_groups: list[dict],
        optimizer: Any,
    ) -> None:
        scene_builder.render_result(
            self,
            scan_polydata,
            target_point,
            pose_groups,
            optimizer,
        )
        self.render()

    def close(self) -> bool:  # type: ignore[override]
        for plotter, _ in self.plotters():
            try:
                plotter.close()
            except Exception:  # noqa: BLE001
                pass
        return super().close()
