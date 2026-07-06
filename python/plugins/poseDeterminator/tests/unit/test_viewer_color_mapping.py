"""scene_builder.render_result 색상 매핑·메타 처리·warning 회귀 테스트.

배경:
    deep-interview + ralplan consensus에서 viewer 키 매핑을 동적 추정에서
    정적 `_KEY_TO_COLOR` dict로 변경하고, 기존 90° 키 ("0"/"90")와 신규
    3-쌍 키 ("120"/"240")를 모두 정확히 매핑. 알 수 없는 키는 회색 fallback
    + logging.warning.

전략:
    - 색상 상수 동일성은 정적 dict 비교 (rendering 무관).
    - render_result 자체는 jv.add_mesh와 _iter_plotters를 monkeypatch해서
      pyvista/Qt 의존성 없이 호출 카운트와 색상 인자만 검증.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from python.plugins.poseDeterminator.tools.viewer.viz import scene_builder
from python.plugins.poseDeterminator.tools.viewer.viz.scene_builder import (
    COLOR_DDA_0DEG,
    COLOR_DDA_90DEG,
    COLOR_DDA_120DEG,
    COLOR_DDA_240DEG,
    COLOR_DDA_UNKNOWN,
    COLOR_RT_0DEG,
    COLOR_RT_90DEG,
    COLOR_RT_120DEG,
    COLOR_RT_240DEG,
    COLOR_RT_UNKNOWN,
    _KEY_TO_COLOR,
)


# ============================================================================
# 정적 dict 매핑 (rendering 무관, 즉시 검증)
# ============================================================================


def test_color_for_existing_90deg_keys():
    """기존 90° 함수의 키 "0"/"90"이 기존 색상 상수로 정확히 매핑 (회귀 보호)."""
    assert _KEY_TO_COLOR["0"] == (COLOR_DDA_0DEG, COLOR_RT_0DEG)
    assert _KEY_TO_COLOR["90"] == (COLOR_DDA_90DEG, COLOR_RT_90DEG)


def test_color_for_new_120_240_keys():
    """신규 3-쌍 함수의 키 "120"/"240"이 신규 색상 상수로 매핑."""
    assert _KEY_TO_COLOR["120"] == (COLOR_DDA_120DEG, COLOR_RT_120DEG)
    assert _KEY_TO_COLOR["240"] == (COLOR_DDA_240DEG, COLOR_RT_240DEG)


def test_key_to_color_has_exactly_expected_keys():
    """_KEY_TO_COLOR가 정확히 {"0","90","120","240"} 네 키만 가짐 (확장 시 명시적 갱신 강제)."""
    assert set(_KEY_TO_COLOR.keys()) == {"0", "90", "120", "240"}


# ============================================================================
# render_result mesh 호출 카운트 + 색상 인자 (jv mock)
# ============================================================================


class _FakePlotter:
    """pyvista plotter의 최소 stub."""

    def __init__(self):
        self.cleared = False
        self.background = None
        self.parallel_projection = False

    def clear(self):
        self.cleared = True

    def set_background(self, color):
        self.background = color

    def add_mesh(self, *args, **kwargs):
        pass

    @property
    def camera(self):
        return self

    def view_isometric(self):
        pass

    def view_xy(self):
        pass


class _FakeMesh:
    """add_mesh 호출 인자 검사용 sentinel."""

    def __init__(self, tag: str):
        self.tag = tag


@pytest.fixture
def render_setup(monkeypatch):
    """render_result를 호출할 수 있도록 환경을 mock한다.

    - jv.add_mesh / jv.add_sphere / jv.add_coordinate_frame를 카운터로 대체
    - _iter_plotters를 단일 fake plotter 1개 반환으로 대체
    - visualizer_adapter helper들을 stub 반환으로 대체
    """
    add_mesh_calls: list[dict] = []

    def fake_add_mesh(plotter, mesh, T, color=None, show_edges=False):
        add_mesh_calls.append({"mesh": mesh, "color": color})

    def fake_add_sphere(*args, **kwargs):
        pass

    def fake_add_coordinate_frame(*args, **kwargs):
        pass

    monkeypatch.setattr(scene_builder.jv, "add_mesh", fake_add_mesh)
    monkeypatch.setattr(scene_builder.jv, "add_sphere", fake_add_sphere)
    monkeypatch.setattr(scene_builder.jv, "add_coordinate_frame", fake_add_coordinate_frame)

    fake_plotter = _FakePlotter()
    monkeypatch.setattr(
        scene_builder,
        "_iter_plotters",
        lambda plot_view: [(fake_plotter, "view_isometric")],
    )

    dda_mesh = _FakeMesh("dda")
    rt_mesh = _FakeMesh("rt")
    monkeypatch.setattr(scene_builder, "get_dda_mesh", lambda opt: dda_mesh)
    monkeypatch.setattr(scene_builder, "get_rt_mesh", lambda opt: rt_mesh)
    monkeypatch.setattr(scene_builder, "get_dda_inv_transform", lambda opt: np.eye(4))
    monkeypatch.setattr(scene_builder, "get_rt_inv_transform", lambda opt: np.eye(4))
    monkeypatch.setattr(
        scene_builder,
        "link_transform_for_tcp_pose",
        lambda pose, inv: np.eye(4),
    )

    return {
        "calls": add_mesh_calls,
        "dda_mesh": dda_mesh,
        "rt_mesh": rt_mesh,
    }


def _make_slot(_actual_deg: int | None = None, _arc_deg: int | None = None) -> dict:
    """DDA + RT1 + RT2가 모두 있는 슬롯 (선택적 메타 포함)."""
    slot = {
        "DDA": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "RT1": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        "RT2": [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    if _actual_deg is not None:
        slot["_actual_deg"] = _actual_deg
    if _arc_deg is not None:
        slot["_arc_deg"] = _arc_deg
    return slot


def test_render_handles_3pair_groups_with_slot_meta(render_setup):
    """3-쌍 pose_groups (메타 포함) 렌더 시 메시 호출 카운트 = 3 슬롯 × 3 메시 = 9.

    슬롯 내부의 _actual_deg가 add_mesh를 추가로 호출하지 않음을 검증.
    """
    pose_groups = [
        {
            "0": _make_slot(_actual_deg=0),
            "120": _make_slot(_actual_deg=120),
            "240": _make_slot(_actual_deg=240),
        }
    ]
    scene_builder.render_result(
        plot_view=None,
        scan_polydata=None,
        target_point=(0.0, 0.0, 0.0),
        pose_groups=pose_groups,
        optimizer=None,
    )
    # 3 슬롯 × (DDA + RT1 + RT2) = 9
    assert len(render_setup["calls"]) == 9
    # 색상 분포: mesh 인자로 DDA/RT 분리 (COLOR_DDA_*DEG와 COLOR_RT_*DEG는
    # 같은 RGB 튜플이라 색상만으로는 구분 불가).
    dda_mesh = render_setup["dda_mesh"]
    rt_mesh = render_setup["rt_mesh"]
    dda_colors = [c["color"] for c in render_setup["calls"] if c["mesh"] is dda_mesh]
    rt_colors = [c["color"] for c in render_setup["calls"] if c["mesh"] is rt_mesh]
    # DDA: 슬롯당 1회, 3 슬롯 = 3 호출
    assert len(dda_colors) == 3
    assert dda_colors.count(COLOR_DDA_0DEG) == 1
    assert dda_colors.count(COLOR_DDA_120DEG) == 1
    assert dda_colors.count(COLOR_DDA_240DEG) == 1
    # RT: 슬롯당 2회 (RT1+RT2), 3 슬롯 = 6 호출
    assert len(rt_colors) == 6
    assert rt_colors.count(COLOR_RT_0DEG) == 2
    assert rt_colors.count(COLOR_RT_120DEG) == 2
    assert rt_colors.count(COLOR_RT_240DEG) == 2


def test_render_handles_fallback_2pair_with_arc_meta(render_setup):
    """폴백 2-쌍 (메타 포함) 렌더 시 메시 호출 카운트 = 2 슬롯 × 3 메시 = 6."""
    pose_groups = [
        {
            "0": _make_slot(_actual_deg=0, _arc_deg=120),
            "120": _make_slot(_actual_deg=120, _arc_deg=120),
        }
    ]
    scene_builder.render_result(
        plot_view=None,
        scan_polydata=None,
        target_point=(0.0, 0.0, 0.0),
        pose_groups=pose_groups,
        optimizer=None,
    )
    assert len(render_setup["calls"]) == 6


def test_render_existing_90deg_keys_no_warning(render_setup, caplog):
    """기존 90° 함수가 반환하는 키 "0"/"90"은 dict에 등록되어 warning 없음 (회귀 보호)."""
    pose_groups = [
        {
            "0": _make_slot(),
            "90": _make_slot(),
        }
    ]
    with caplog.at_level(logging.WARNING):
        scene_builder.render_result(
            plot_view=None,
            scan_polydata=None,
            target_point=(0.0, 0.0, 0.0),
            pose_groups=pose_groups,
            optimizer=None,
        )
    # 정확히 2 슬롯 × 3 메시 = 6 호출, warning 0건
    assert len(render_setup["calls"]) == 6
    assert not any("unknown angle key" in rec.getMessage() for rec in caplog.records)
    # 색상도 기존 그대로
    colors_used = [c["color"] for c in render_setup["calls"]]
    assert COLOR_DDA_0DEG in colors_used
    assert COLOR_DDA_90DEG in colors_used


def test_render_unknown_key_logs_warning_and_uses_fallback(render_setup, caplog):
    """알 수 없는 키 → logging.warning + COLOR_DDA_UNKNOWN/COLOR_RT_UNKNOWN fallback."""
    pose_groups = [{"999": _make_slot()}]
    with caplog.at_level(logging.WARNING):
        scene_builder.render_result(
            plot_view=None,
            scan_polydata=None,
            target_point=(0.0, 0.0, 0.0),
            pose_groups=pose_groups,
            optimizer=None,
        )
    # warning 발생
    assert any("unknown angle key" in rec.getMessage() for rec in caplog.records)
    # fallback 색상 사용. mesh 인자로 DDA/RT 분리 (UNKNOWN 색상이 둘 다 같은 회색).
    dda_mesh = render_setup["dda_mesh"]
    rt_mesh = render_setup["rt_mesh"]
    dda_colors = [c["color"] for c in render_setup["calls"] if c["mesh"] is dda_mesh]
    rt_colors = [c["color"] for c in render_setup["calls"] if c["mesh"] is rt_mesh]
    assert dda_colors == [COLOR_DDA_UNKNOWN]
    assert rt_colors == [COLOR_RT_UNKNOWN, COLOR_RT_UNKNOWN]
