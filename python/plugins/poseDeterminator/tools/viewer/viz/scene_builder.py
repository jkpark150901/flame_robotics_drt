"""4-view subplot 에 동일 actor 를 추가하는 빌더.

PyVista ``Plotter(shape=(2,2))`` 의 4 subplot (iso/XY/XZ/YZ) 에 동일한 객체
를 추가한다. 0도 / 90도 pose group 의 변별성은 노트북 demo (cell
``1ea8ce44`` line 626/640) 의 색상 정책을 그대로 재사용한다.

검사 포인트 부분 확대(줌)는 노트북 cell ``2d85e9ca`` 와 동일한 방식으로
``target_point ± CROP_HALF_EXTENT`` AABB 박스로 점군을 크롭해 자연스럽게
카메라 프레임이 좁아지도록 한다.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)

# JupyterVisualizer 는 plugin_pdm 루트에서 직접 import (sys.path 가드 후).
import python.plugins.poseDeterminator.JupyterVisualizer as jv  # type: ignore

from .visualizer_adapter import (
    get_dda_inv_transform,
    get_dda_mesh,
    get_rt_inv_transform,
    get_rt_mesh,
    link_transform_for_tcp_pose,
)

# 노트북 demo cell 1ea8ce44 의 색상 정책 + 3-쌍 120° 확장
COLOR_DDA_0DEG: tuple[int, int, int] = (255, 150, 150)
COLOR_DDA_90DEG: tuple[int, int, int] = (150, 150, 255)
COLOR_DDA_120DEG: tuple[int, int, int] = (150, 255, 150)   # 신규 — 녹색 계열
COLOR_DDA_240DEG: tuple[int, int, int] = (255, 220, 150)   # 신규 — 주황 계열
COLOR_RT_0DEG: tuple[int, int, int] = (255, 150, 150)
COLOR_RT_90DEG: tuple[int, int, int] = (150, 150, 255)
COLOR_RT_120DEG: tuple[int, int, int] = (150, 255, 150)
COLOR_RT_240DEG: tuple[int, int, int] = (255, 220, 150)
COLOR_DDA_UNKNOWN: tuple[int, int, int] = (180, 180, 180)  # fallback (회색)
COLOR_RT_UNKNOWN: tuple[int, int, int] = (180, 180, 180)

# 키 = 이상 라벨 문자열. EndEffectorPoseOptimizer의 두 함수 모두 이상 라벨 키를 사용함.
# module-level private (`_` prefix). 테스트에서 import 가능 (AC #11 viewer 매핑 회귀).
_KEY_TO_COLOR: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "0":   (COLOR_DDA_0DEG,   COLOR_RT_0DEG),
    "90":  (COLOR_DDA_90DEG,  COLOR_RT_90DEG),
    "120": (COLOR_DDA_120DEG, COLOR_RT_120DEG),
    "240": (COLOR_DDA_240DEG, COLOR_RT_240DEG),
}

COLOR_PIPE_DEFAULT: str = "skyblue"     # 점군 기본 색 (사용자 지정: 하늘색)
COLOR_TARGET_SPHERE: str = "red"
COLOR_BACKGROUND: str = "white"         # 단색 흰색 (그라데이션 사용 안 함)

# 오버뷰(전체 가시화) 색상 — 노트북 cell 67e03b7d 정책
COLOR_INSPECTION_NORMAL: str = "green"   # offset 없음 (position == position_with_offset)
COLOR_INSPECTION_OFFSET: str = "red"     # offset 있음 — 원본 검사 좌표
COLOR_DETECTION_OFFSET: str = "blue"     # offset 있음 — 표면으로 옮긴 좌표

# 4-view 줌: target ± 0.3m AABB 로 크롭 (notebook cell 2d85e9ca 과 동일)
CROP_HALF_EXTENT: float = 0.3

def _iter_plotters(plot_view: Any) -> Iterable[tuple[Any, str]]:
    """``PlotView.plotters()`` 호출 — 4쌍 ``(plotter, view_fn_name)``.

    PlotView 가 4개의 독립 ``QtInteractor`` 를 보유하는 컨테이너이므로 각
    plotter 에 대해 직접 작업한다. 단일 plotter shape=(2,2) 시절에 사용하던
    ``plot_view.subplot(r, c)`` 호출은 더 이상 필요하지 않다.
    """

    return plot_view.plotters()


def _crop_polydata_around(
    scan_polydata: Any,
    target_point: np.ndarray,
    half_extent: float = CROP_HALF_EXTENT,
) -> Any:
    """``target_point ± half_extent`` AABB 박스 안의 점만 남긴 PolyData 반환.

    notebook cell ``2d85e9ca`` 의 ``o3d.geometry.AxisAlignedBoundingBox`` +
    ``crop`` 와 동치. 박스 안에 점이 하나도 없으면 원본을 반환해 빈 화면을
    피한다 (검사 포인트가 점군 외곽에 찍힌 비정상 케이스 방어).
    """

    if scan_polydata is None:
        return None
    points = np.asarray(scan_polydata.points)
    if points.size == 0:
        return scan_polydata
    target = np.asarray(target_point, dtype=float)
    mask = np.all(np.abs(points - target) <= half_extent, axis=1)
    if not mask.any():
        return scan_polydata
    cropped = pv.PolyData(points[mask])
    if "colors" in scan_polydata.array_names:
        cropped["colors"] = np.asarray(scan_polydata["colors"])[mask]
    return cropped


def _draw_scan_and_target(
    plot_view: Any,
    scan_polydata: Any,
    target: np.ndarray,
    has_color: bool,
    sphere_radius: float,
) -> None:
    """현재 active subplot 에 배경·점군·타겟 sphere·좌표축을 일괄 추가."""

    plot_view.set_background(COLOR_BACKGROUND)
    if scan_polydata is not None:
        if has_color:
            plot_view.add_mesh(scan_polydata, scalars="colors", rgb=True, point_size=2)
        else:
            plot_view.add_mesh(scan_polydata, color=COLOR_PIPE_DEFAULT, point_size=2)
    jv.add_sphere(plot_view, target, radius=sphere_radius, color=COLOR_TARGET_SPHERE)
    jv.add_coordinate_frame(plot_view, origin=(0, 0, 0), length=0.05, size=0.001)


def render_pipe(
    plot_view: Any,
    scan_polydata: Any,
    target_point: tuple[float, float, float] | np.ndarray,
    sphere_radius: float = 0.01,
) -> None:
    """4 plotter 에 배관 점군 + 검사 포인트 sphere 만 표시 (포인트 주변 줌)."""

    target = np.asarray(target_point, dtype=float)
    cropped = _crop_polydata_around(scan_polydata, target)
    has_color = cropped is not None and "colors" in cropped.array_names

    for plotter, view_fn in _iter_plotters(plot_view):
        plotter.clear()
        _draw_scan_and_target(plotter, cropped, target, has_color, sphere_radius)
        getattr(plotter, view_fn)()
        plotter.camera.parallel_projection = True


def render_result(
    plot_view: Any,
    scan_polydata: Any,
    target_point: tuple[float, float, float] | np.ndarray,
    pose_groups: list[dict],
    optimizer: Any,
    sphere_radius: float = 0.01,
) -> None:
    """4 subplot 에 배관 + 포인트 + 채택된 pose_group 의 DDA/RT 메시 동시 렌더.

    ``pose_groups`` 는 ``calculate_DDA_RT_pose_for_taking_xray`` 의 두 번째
    반환값. 각 dict 는 ``"0"`` / ``"90"`` 키를 가지며 그 아래 ``DDA`` /
    ``RT1`` / ``RT2`` 가 ``[x,y,z,r,p,y]`` 6-벡터.

    검사 포인트 부분 확대를 위해 점군을 ``target_point ± CROP_HALF_EXTENT``
    AABB 로 크롭한 뒤 4 subplot 에 그린다 (notebook cell ``1ea8ce44`` 와 동일).
    """

    target = np.asarray(target_point, dtype=float)
    cropped = _crop_polydata_around(scan_polydata, target)
    has_color = cropped is not None and "colors" in cropped.array_names

    dda_mesh = get_dda_mesh(optimizer)
    rt_mesh = get_rt_mesh(optimizer)
    dda_inv = get_dda_inv_transform(optimizer)
    rt_inv = get_rt_inv_transform(optimizer)

    for plotter, view_fn in _iter_plotters(plot_view):
        plotter.clear()
        _draw_scan_and_target(plotter, cropped, target, has_color, sphere_radius)

        # 채택된 pose_group 모두 동시 표시.
        # 그룹 dict의 키 = 이상 라벨 문자열 ("0"/"90"/"120"/"240").
        # 메타 필드 (_actual_deg, _arc_deg)는 그룹이 아닌 *슬롯* 안에 들어가며,
        # 아래에서는 slot.get("DDA"/"RT1"/"RT2")로만 명시 접근하므로 메타가
        # 렌더 경로에 새는 일이 없음 → 그룹 키 skip 가드 불필요.
        for pg in pose_groups:
            for angle_key, slot in pg.items():
                if angle_key in _KEY_TO_COLOR:
                    dda_color, rt_color = _KEY_TO_COLOR[angle_key]
                else:
                    dda_color, rt_color = COLOR_DDA_UNKNOWN, COLOR_RT_UNKNOWN
                    logger.warning(
                        "scene_builder.render_result: unknown angle key %r in pose_group; "
                        "falling back to UNKNOWN color",
                        angle_key,
                    )
                dda_pose = slot.get("DDA")
                if dda_pose is not None:
                    T = link_transform_for_tcp_pose(dda_pose, dda_inv)
                    jv.add_mesh(plotter, dda_mesh, T, color=dda_color, show_edges=False)
                for rt_key in ("RT1", "RT2"):
                    rt_pose = slot.get(rt_key)
                    if rt_pose is None:
                        continue
                    T = link_transform_for_tcp_pose(rt_pose, rt_inv)
                    jv.add_mesh(plotter, rt_mesh, T, color=rt_color, show_edges=False)

        getattr(plotter, view_fn)()
        plotter.camera.parallel_projection = True


def render_overview(
    plot_view: Any,
    scan_polydata: Any,
    points: list,
    sphere_radius: float = 0.005,
    label_font_size: int = 12,
) -> None:
    """단일 plotter 에 배관 전체 + 검사 포인트 sphere + 인덱스 라벨 표시.

    notebook cell ``67e03b7d`` 의 가시화 정책을 그대로 따른다:
      - offset 이 없는 포인트(``position == position_with_offset``): green sphere
      - offset 이 있는 포인트: red(원본) + blue(표면 좌표) 두 sphere
      - 모든 포인트에 인덱스 라벨 (검은색, ``always_visible=True``)
    """

    plot_view.clear()
    plot_view.set_background(COLOR_BACKGROUND)

    if scan_polydata is not None:
        if "colors" in scan_polydata.array_names:
            plot_view.add_mesh(scan_polydata, scalars="colors", rgb=True, point_size=2)
        else:
            plot_view.add_mesh(scan_polydata, color=COLOR_PIPE_DEFAULT, point_size=2)

    label_positions: list = []
    label_texts: list[str] = []
    for point in points:
        is_offset = not _positions_equal(point.position, point.position_with_offset)
        if is_offset:
            jv.add_sphere(plot_view, point.position, radius=sphere_radius, color=COLOR_INSPECTION_OFFSET)
            jv.add_sphere(plot_view, point.position_with_offset, radius=sphere_radius, color=COLOR_DETECTION_OFFSET)
        else:
            jv.add_sphere(plot_view, point.position, radius=sphere_radius, color=COLOR_INSPECTION_NORMAL)
        label_positions.append(point.position)
        label_texts.append(str(point.index))

    if label_positions:
        # PyVista default: shape='rounded_rect', shape_color='lightgray' →
        # 회색 배경에 검은색 글씨 (notebook cell 67e03b7d 와 동일).
        plot_view.add_point_labels(
            label_positions,
            label_texts,
            font_size=label_font_size,
            point_color="black",
            text_color="black",
            always_visible=True,
        )

    plot_view.view_isometric()
    # CAD 스타일 평행 투영 (멀어져도 평행선이 평행하게 보이도록).
    plot_view.camera.parallel_projection = True


def _positions_equal(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    tol: float = 1e-9,
) -> bool:
    return all(abs(ai - bi) <= tol for ai, bi in zip(a, b))
