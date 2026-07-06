"""PIPE 콤보 변경 시 PLY → ``pyvista.PolyData`` 비동기 로드 워커.

메인 스레드는 콤보 변경 슬롯에서 ``start_load(...)`` 만 호출하고 즉시 반환.
로드가 끝나면 ``load_done(pipe_id, polydata)`` 로 메인 스레드에 통보. 같은
pipe_id 가 다시 요청되면 직전 작업이 끝날 때까지 직렬화한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pyvista as pv
from PyQt6.QtCore import QThread, pyqtSignal


class PointCloudLoader(QThread):
    load_done = pyqtSignal(str, object, object)  # (pipe_id, o3d_pcd, pv.PolyData)
    load_failed = pyqtSignal(str, str)           # (pipe_id, error_message)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pending_pipe_id: str = ""
        self._pending_ply_path: str = ""

    def start_load(self, pipe_id: str, ply_path: Path | str) -> None:
        """직렬화 보장: 이전 로드가 진행 중이면 끝날 때까지 대기 후 시작."""

        if self.isRunning():
            self.wait()
        self._pending_pipe_id = pipe_id
        self._pending_ply_path = str(ply_path)
        self.start()

    def run(self) -> None:
        pipe_id = self._pending_pipe_id
        ply_path = self._pending_ply_path
        try:
            pcd = o3d.io.read_point_cloud(ply_path)  # type: ignore[arg-type]
            polydata = _pcd_to_polydata(pcd)
            self.load_done.emit(pipe_id, pcd, polydata)
        except Exception as exc:  # noqa: BLE001
            self.load_failed.emit(pipe_id, str(exc))


def _pcd_to_polydata(pcd) -> pv.PolyData:
    points = np.asarray(pcd.points)
    polydata = pv.PolyData(points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if colors.dtype.kind == "f":
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        polydata["colors"] = colors
    return polydata
