"""tools/viewer 진입점.

호출:
    python -m plugins.plugin_pdm.tools.viewer        # python/ 디렉토리 기준
    python python/plugins/plugin_pdm/tools/viewer/__main__.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_plugin_pdm_on_sys_path() -> None:
    """plugin_pdm 디렉토리를 sys.path 에 추가.

    노트북/`controller.py:24` 가 사용하는 ``sys.path.append`` 패턴을 따르며
    중복 추가 방지를 위한 idempotent 가드를 둔다.
    """

    plugin_root = str(Path(__file__).resolve().parents[2])
    if plugin_root not in sys.path:
        sys.path.append(plugin_root)


def main() -> int:
    _ensure_plugin_pdm_on_sys_path()

    # Windows 한글 절대경로에서 Open3D 가 fopen 에 UTF-8 바이트열을 전달하며
    # 실패하는 문제를 우회하기 위해 8.3 단축경로 wrapper 로 교체.
    from . import _open3d_compat
    _open3d_compat.apply()

    # PyQt6 import 는 path 셋업 이후
    from PyQt6.QtWidgets import QApplication

    from .widgets.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
