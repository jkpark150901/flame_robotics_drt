"""tools/viewer 의 메인 윈도우.

좌측 사이드 패널 + 중앙 4-view PlotView + 하단 진행 바 를 조립하고,
``BatchWorker`` 와 ``PointCloudLoader`` 두 개의 ``QThread`` 라이프사이클을
관리한다. PIPE 변경 시 다음 순서를 직렬화한다:

  1. 진행 중인 ``BatchWorker`` 가 있으면 사용자 확인 후
     ``requestInterruption() + wait()`` 으로 안전 종료.
  2. ``PointCloudLoader.start_load(pipe_id, ply_path)`` 위임 (워커가
     PLY 를 비동기 로드).
  3. 워커의 ``load_done`` 슬롯에서만 ``optimizer._scan_data`` 갱신.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .. import app_paths
from ..data_loader import discover_pipes, load_inspection_points
from ..models import (
    InspectionPoint,
    InspectionResult,
    OptimizationMode,
    PipeData,
    PointStatus,
)
from ..workers.batch_worker import BatchWorker
from ..workers.pcd_loader import PointCloudLoader
from .plot_view import PlotView
from .side_panel import SidePanel


_logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PDM Viewer — 검사 포인트 자세 가시화")
        self.resize(1400, 900)

        # ---- 도메인 상태 ---------------------------------------------------
        self._pipes: list[PipeData] = []
        self._current_pipe: PipeData | None = None
        self._current_points: list[InspectionPoint] = []
        self._results: dict[int, InspectionResult] = {}
        self._current_polydata: Any = None
        self._scan_loaded_for_pipe: str | None = None
        # 자세 계산 모드 — SidePanel 콤보의 현재 선택을 거울 (단일 출처).
        self._current_mode: OptimizationMode = OptimizationMode.TWO_PAIR_90

        # ---- optimizer 1회 생성 + URDF 로드 (메인 스레드, 1회성 비용) ----
        from python.plugins.poseDeterminator.EndEffectorPoseOptimizer import EndEffectorPoseOptimizer  # type: ignore

        self._optimizer = EndEffectorPoseOptimizer(debug_mode=False)
        self._optimizer.load_DDA_from_urdf(str(app_paths.dda_urdf_path()))
        self._optimizer.load_RT_from_urdf(str(app_paths.rt_urdf_path()))

        # ---- 워커 ----------------------------------------------------------
        self._batch_worker: BatchWorker | None = None
        self._pcd_loader = PointCloudLoader(self)
        self._pcd_loader.load_done.connect(self._on_pcd_loaded)
        self._pcd_loader.load_failed.connect(self._on_pcd_failed)

        # ---- UI ------------------------------------------------------------
        self._side_panel = SidePanel(self)
        self._plot_view = PlotView(self)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._side_panel)
        splitter.addWidget(self._plot_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1100])

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(self._build_progress_row())
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar(self))

        # ---- 시그널 배선 ---------------------------------------------------
        self._side_panel.pipe_changed.connect(self._on_pipe_combo_changed)
        self._side_panel.optimize_clicked.connect(self._on_optimize_clicked)
        self._side_panel.point_selected.connect(self._on_point_selected)
        self._side_panel.mode_changed.connect(self._on_mode_changed)
        # SidePanel 콤보 기본값과 내부 상태 sync (초기 1회).
        self._current_mode = self._side_panel.current_mode()

        # ---- 데이터 디스커버리 + 콤보 채우기 ------------------------------
        self._pipes = discover_pipes()
        if self._pipes:
            self._side_panel.set_pipes([p.pipe_id for p in self._pipes])
            # 초기 PIPE 자동 로드
            QTimer.singleShot(0, lambda: self._on_pipe_combo_changed(0))
        else:
            self.statusBar().showMessage("data/ 디렉토리에서 *_fill.ply + *_testpoint.csv 페어를 찾지 못했습니다.")
            self._side_panel.set_optimize_enabled(False)

    # ----- UI 구성 ----------------------------------------------------------

    def _build_progress_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(8, 4, 8, 8)
        self._progress_label = QLabel("0 / 0")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        row.addWidget(QLabel("진행:"))
        row.addWidget(self._progress_bar, stretch=1)
        row.addWidget(self._progress_label)
        return row

    # ----- 슬롯 -------------------------------------------------------------

    def _on_pipe_combo_changed(self, combo_index: int) -> None:
        if combo_index < 0 or combo_index >= len(self._pipes):
            return
        new_pipe = self._pipes[combo_index]

        # ① BatchWorker 안전 종료 (사용자 확인)
        if self._batch_worker is not None and self._batch_worker.isRunning():
            answer = QMessageBox.question(
                self,
                "일괄 계산 중단",
                "진행 중인 일괄 자세 계산을 중단하고 다른 배관을 로드하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                # 콤보 원상 복구
                if self._current_pipe is not None:
                    for i, p in enumerate(self._pipes):
                        if p.pipe_id == self._current_pipe.pipe_id:
                            self._side_panel.set_current_pipe_index_silent(i)
                            break
                return
            self._batch_worker.requestInterruption()
            self._batch_worker.wait()
            self._batch_worker = None

        # ② 새 PIPE 컨텍스트 셋업
        self._current_pipe = new_pipe
        self._results.clear()
        self._current_polydata = None
        self._scan_loaded_for_pipe = None

        try:
            self._current_points = load_inspection_points(new_pipe.csv_path)
        except Exception as exc:  # noqa: BLE001
            self._current_points = []
            QMessageBox.warning(self, "CSV 로드 실패", f"{new_pipe.csv_path}\n{exc}")

        self._side_panel.set_points(self._current_points)
        self._side_panel.set_optimize_enabled(False)
        self._side_panel.set_optimize_label("배관 점군 로드 중…")
        self._side_panel.clear_overview()
        self._reset_progress()
        self._plot_view.clear_all()
        self._plot_view.render()
        self.statusBar().showMessage(
            f"{new_pipe.pipe_id} — 검사 포인트 {len(self._current_points)}개. 점군 로드 중…"
        )

        # ③ PointCloudLoader 위임 (메인 스레드 비-블록)
        self._pcd_loader.start_load(new_pipe.pipe_id, new_pipe.ply_path)

    def _on_pcd_loaded(self, pipe_id: str, o3d_pcd: Any, polydata: Any) -> None:
        # 콤보가 여전히 같은 PIPE 인지 검증 (race 보호)
        if self._current_pipe is None or pipe_id != self._current_pipe.pipe_id:
            return

        self._current_polydata = polydata
        # optimizer 의 점군은 워커가 이미 읽은 PointCloud 를 직접 주입한다.
        # ``load_scan_data(file_path)`` 가 메인 스레드 동기 I/O 를 강제하므로
        # principle #3 (signal 경계 통신) 충족을 위해 우회.
        # 알고리즘 측 PR 로 ``set_scan_data(pcd)`` public API 가 신설되면 그
        # 메서드로 전환할 것 (ADR Follow-up).
        self._optimizer._scan_data = o3d_pcd
        self._scan_loaded_for_pipe = pipe_id

        self._side_panel.set_optimize_enabled(True)
        self._side_panel.set_optimize_label("엔드이펙터 최적 위치 탐지")
        # 사이드 패널 오버뷰 위젯에 배관 전체 + 검사 포인트 라벨 렌더
        self._side_panel.update_overview(polydata, self._current_points)
        self.statusBar().showMessage(f"{pipe_id} — 점군 로드 완료. 검사 포인트를 클릭하거나 일괄 탐지를 실행하세요.")

        # 첫 항목 자동 선택 → 배관만 표시
        if self._current_points:
            self._render_for_index(self._current_points[0].index)

    def _on_pcd_failed(self, pipe_id: str, error: str) -> None:
        if self._current_pipe is None or pipe_id != self._current_pipe.pipe_id:
            return
        self._side_panel.set_optimize_enabled(False)
        self._side_panel.set_optimize_label("점군 로드 실패")
        self.statusBar().showMessage(f"{pipe_id} 점군 로드 실패: {error}")

    def _on_optimize_clicked(self) -> None:
        if self._current_pipe is None or not self._current_points:
            return
        if self._scan_loaded_for_pipe != self._current_pipe.pipe_id:
            return
        if self._batch_worker is not None and self._batch_worker.isRunning():
            return

        self._results.clear()
        for list_pos in range(len(self._current_points)):
            self._side_panel.update_point_status(list_pos, PointStatus.UNRUN)
        self._reset_progress(total=len(self._current_points))
        self._side_panel.set_optimize_enabled(False)
        # 실행 중에는 모드 변경 잠금 — race 회피 (worker 가 생성 시점의 mode 로 동작)
        self._side_panel.set_mode_combo_enabled(False)

        worker = BatchWorker(
            self._optimizer,
            self._current_points,
            parent=self,
            mode=self._current_mode,
        )
        worker.point_done.connect(self._on_batch_point_done)
        worker.progress.connect(self._on_batch_progress)
        worker.batch_finished.connect(self._on_batch_finished)
        worker.batch_aborted.connect(self._on_batch_aborted)
        self._batch_worker = worker
        worker.start()
        mode_label = (
            "3쌍 120°" if self._current_mode is OptimizationMode.THREE_PAIR_120 else "2쌍 90°"
        )
        self.statusBar().showMessage(
            f"{self._current_pipe.pipe_id} — 일괄 자세 탐지 시작 "
            f"({len(self._current_points)}개, {mode_label})…"
        )

    def _on_mode_changed(self, mode: OptimizationMode) -> None:
        """SidePanel 모드 콤보 변경 → 내부 상태 갱신 + 이전 결과 무효화.

        모드별로 pose_group 키 정책 (\"0\"/\"90\" vs \"0\"/\"120\"/\"240\")이 다르고
        viewer 의 색상도 달라지므로, 같은 검사 포인트라도 모드가 바뀌면 결과를
        새로 계산해야 한다. 기존 결과는 한 번에 무효화하고 포인트 라벨도 UNRUN
        으로 리셋한다.
        """

        if mode is self._current_mode:
            return
        self._current_mode = mode
        # 이전 결과 무효화 — 다른 알고리즘으로 다시 탐지해야 함
        self._results.clear()
        for list_pos in range(len(self._current_points)):
            self._side_panel.update_point_status(list_pos, PointStatus.UNRUN)
        self._reset_progress(total=len(self._current_points))
        # 현재 선택된 검사 포인트는 배관-only 로 다시 렌더 (이전 자세 메시 제거)
        current_index = self._side_panel_current_point_index()
        if current_index is not None:
            self._render_for_index(current_index)

    def _side_panel_current_point_index(self) -> int | None:
        """SidePanel 의 현재 선택 항목 index — 모드 전환 시 재렌더에 사용."""

        # SidePanel 내부 list 의 currentItem 을 통해 안전하게 조회
        try:
            item = self._side_panel._list.currentItem()  # noqa: SLF001
        except AttributeError:
            return None
        if item is None:
            return None
        from PyQt6.QtCore import Qt as _Qt
        value = item.data(_Qt.ItemDataRole.UserRole)
        return value if isinstance(value, int) else None

    def _on_batch_point_done(self, result: InspectionResult) -> None:
        self._results[result.point_index] = result
        list_pos = self._index_to_list_position(result.point_index)
        if list_pos is None:
            return
        if result.error_message:
            self._side_panel.update_point_status(list_pos, PointStatus.FAILURE, result.error_message)
        elif result.success:
            self._side_panel.update_point_status(list_pos, PointStatus.SUCCESS)
        else:
            self._side_panel.update_point_status(
                list_pos,
                PointStatus.FAILURE,
                "충돌 회피 가능한 자세 그룹이 없습니다.",
            )

    def _on_batch_progress(self, completed: int, total: int) -> None:
        self._progress_bar.setRange(0, max(total, 1))
        self._progress_bar.setValue(completed)
        self._progress_label.setText(f"{completed} / {total}")

    def _on_batch_finished(self) -> None:
        self._side_panel.set_optimize_enabled(True)
        self._side_panel.set_mode_combo_enabled(True)
        self.statusBar().showMessage(
            f"{self._current_pipe.pipe_id if self._current_pipe else ''} — 일괄 탐지 완료."
        )
        self._batch_worker = None

    def _on_batch_aborted(self, reason: str) -> None:
        self._side_panel.set_optimize_enabled(True)
        self._side_panel.set_mode_combo_enabled(True)
        self.statusBar().showMessage(f"일괄 탐지 중단: {reason}")
        self._batch_worker = None

    def _on_point_selected(self, point_index: int) -> None:
        self._render_for_index(point_index)

    # ----- 렌더링 헬퍼 ------------------------------------------------------

    def _render_for_index(self, point_index: int) -> None:
        point = self._find_point(point_index)
        if point is None or self._current_polydata is None:
            return
        result = self._results.get(point_index)
        if result is not None and result.success and result.pose_groups:
            self._plot_view.set_full_result(
                self._current_polydata,
                point.position_with_offset,
                result.pose_groups,
                self._optimizer,
            )
        else:
            self._plot_view.set_pipe_only(
                self._current_polydata,
                point.position_with_offset,
            )

    # ----- util -------------------------------------------------------------

    def _find_point(self, index: int) -> InspectionPoint | None:
        for point in self._current_points:
            if point.index == index:
                return point
        return None

    def _index_to_list_position(self, point_index: int) -> int | None:
        for pos, point in enumerate(self._current_points):
            if point.index == point_index:
                return pos
        return None

    def _reset_progress(self, total: int = 0) -> None:
        self._progress_bar.setRange(0, max(total, 1))
        self._progress_bar.setValue(0)
        self._progress_label.setText(f"0 / {total}")

    # ----- 종료 -------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self._batch_worker.requestInterruption()
            self._batch_worker.wait()
        if self._pcd_loader.isRunning():
            self._pcd_loader.wait()
        try:
            self._plot_view.close()
        except Exception:  # noqa: BLE001
            _logger.exception("plot_view.close 중 예외")
        try:
            self._side_panel.cleanup()
        except Exception:  # noqa: BLE001
            _logger.exception("side_panel.cleanup 중 예외")
        super().closeEvent(event)
