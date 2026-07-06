"""tools/viewer.data_loader 단위 테스트."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from python.plugins.poseDeterminator.tools.viewer.data_loader import discover_pipes, load_inspection_points


@pytest.fixture()
def fake_data_dir(tmp_path: Path) -> Path:
    """3 개의 PIPE NO.X 페어 + 1 개 _w_mesh.ply 변종 + 1 개 짝 없는 PLY."""

    # 정상 페어 3 개
    for i in (1, 2, 3):
        (tmp_path / f"PIPE NO.{i}_fill.ply").write_text("ply-stub")
        (tmp_path / f"PIPE NO.{i}_fill_testpoint.csv").write_text(
            ",x,y,z,dx,dy,dz\n1,0,0,0,,,\n"
        )

    # _w_mesh 변종 (csv 없음, 무시되어야 함)
    (tmp_path / "PIPE NO.2_fill_w_mesh.ply").write_text("ply-stub")
    (tmp_path / "PIPE NO.3_fill_w_mesh.ply").write_text("ply-stub")

    # CSV 짝 없는 PLY — 결과 list 에서 제외되어야 함
    (tmp_path / "PIPE ORPHAN_fill.ply").write_text("ply-stub")

    return tmp_path


def test_discover_pipes_finds_three_pairs(fake_data_dir: Path) -> None:
    pipes = discover_pipes(fake_data_dir)
    pipe_ids = {p.pipe_id for p in pipes}
    assert pipe_ids == {"PIPE NO.1", "PIPE NO.2", "PIPE NO.3"}


def test_discover_pipes_ignores_w_mesh_variant(fake_data_dir: Path) -> None:
    pipes = discover_pipes(fake_data_dir)
    for p in pipes:
        assert "_w_mesh" not in p.ply_path.name


def test_discover_pipes_skips_orphan_ply_without_csv(fake_data_dir: Path) -> None:
    pipes = discover_pipes(fake_data_dir)
    assert all(p.pipe_id != "PIPE ORPHAN" for p in pipes)


def test_discover_pipes_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert discover_pipes(missing) == []


def test_load_inspection_points_parses_full_row(tmp_path: Path) -> None:
    csv = tmp_path / "p.csv"
    csv.write_text(
        textwrap.dedent(
            """\
            ,x,y,z,dx,dy,dz
            1,1.0,2.0,3.0,0.1,-0.2,0.05
            """
        )
    )
    points = load_inspection_points(csv)
    assert len(points) == 1
    p = points[0]
    assert p.index == 1
    assert p.position == (1.0, 2.0, 3.0)
    assert p.offset == (0.1, -0.2, 0.05)
    assert p.position_with_offset == (1.1, 1.8, 3.05)


def test_load_inspection_points_treats_blank_offset_as_zero(tmp_path: Path) -> None:
    csv = tmp_path / "p.csv"
    csv.write_text(
        textwrap.dedent(
            """\
            ,x,y,z,dx,dy,dz
            1,1.0,2.0,3.0,,,
            2,4.0,5.0,6.0,0.01,,
            """
        )
    )
    points = load_inspection_points(csv)
    assert points[0].offset == (0.0, 0.0, 0.0)
    assert points[1].offset == (0.01, 0.0, 0.0)


def test_load_inspection_points_uses_first_column_as_index(tmp_path: Path) -> None:
    csv = tmp_path / "p.csv"
    csv.write_text(
        textwrap.dedent(
            """\
            ,x,y,z,dx,dy,dz
            42,0,0,0,,,
            """
        )
    )
    [point] = load_inspection_points(csv)
    assert point.index == 42
