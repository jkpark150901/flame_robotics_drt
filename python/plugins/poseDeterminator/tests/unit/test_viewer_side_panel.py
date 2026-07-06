"""SidePanel 모드 토글 단위 테스트.

배경:
    SidePanel 에 추가된 ``자세 계산 모드`` 콤보가 enum 값을 ``mode_changed``
    시그널로 emit 하는지, ``current_mode()`` 가 일관된 enum 을 반환하는지,
    ``set_mode_combo_enabled()`` 가 실제로 위젯 enable 상태를 토글하는지 확인.

데이터 의존:
    pytest-qt (qtbot, qapp_cls) 가 필요. 미설치 환경에서는 자동 skip.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytestqt", reason="pytest-qt 미설치")
pytest.importorskip("PyQt6", reason="PyQt6 미설치")

from python.plugins.poseDeterminator.tools.viewer.models import OptimizationMode  # noqa: E402
from python.plugins.poseDeterminator.tools.viewer.widgets.side_panel import SidePanel  # noqa: E402


@pytest.fixture
def panel(qtbot) -> SidePanel:
    sp = SidePanel()
    qtbot.addWidget(sp)
    return sp


def test_default_mode_is_two_pair_90(panel: SidePanel) -> None:
    """초기 콤보 선택이 첫 번째 항목 = TWO_PAIR_90 (회귀 보호)."""

    assert panel.current_mode() is OptimizationMode.TWO_PAIR_90


def test_mode_combo_emits_enum_on_change(panel: SidePanel, qtbot) -> None:
    """콤보 인덱스를 신규 항목으로 바꾸면 enum 값이 emit 된다."""

    received: list[OptimizationMode] = []
    panel.mode_changed.connect(received.append)

    # 첫 번째 (TWO_PAIR_90) → 두 번째 (THREE_PAIR_120)
    panel._mode_combo.setCurrentIndex(1)  # noqa: SLF001 — 테스트 한정 접근

    qtbot.waitUntil(lambda: len(received) >= 1, timeout=1000)
    assert received[-1] is OptimizationMode.THREE_PAIR_120
    assert panel.current_mode() is OptimizationMode.THREE_PAIR_120


def test_set_mode_combo_enabled_toggles_widget(panel: SidePanel) -> None:
    """``set_mode_combo_enabled`` 가 위젯의 isEnabled() 와 직접 연결."""

    panel.set_mode_combo_enabled(False)
    assert panel._mode_combo.isEnabled() is False  # noqa: SLF001
    panel.set_mode_combo_enabled(True)
    assert panel._mode_combo.isEnabled() is True  # noqa: SLF001


def test_mode_combo_has_exactly_two_options(panel: SidePanel) -> None:
    """콤보가 정확히 enum 멤버 수만큼 항목 보유 — 확장 시 명시적 갱신 강제."""

    assert panel._mode_combo.count() == len(OptimizationMode)  # noqa: SLF001
