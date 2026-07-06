"""data/ 디렉토리 스캔 + 검사 포인트 CSV 파서.

``*_fill.ply`` 와 같은 stem 의 ``*_fill_testpoint.csv`` 가 모두 있는 경우만
``PipeData`` 후보로 채택한다. ``*_fill_w_mesh.ply`` 변종은 무시 (spec
§Constraints "기존 PLY/CSV 만").
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .app_paths import DATA_DIR
from .models import InspectionPoint, PipeData


def discover_pipes(data_dir: Path | None = None) -> list[PipeData]:
    """``data/`` 안에서 ``*_fill.ply`` + ``*_fill_testpoint.csv`` 페어를 찾는다.

    ``*_fill_w_mesh.ply`` 등 변종은 stem 끝이 ``_fill`` 이 아니므로 자연 배제.
    """

    base = data_dir if data_dir is not None else DATA_DIR
    if not base.is_dir():
        return []

    pipes: list[PipeData] = []
    for ply in sorted(base.glob("*_fill.ply")):
        # stem 예: "PIPE NO.1_fill"
        if not ply.stem.endswith("_fill"):
            continue
        pipe_id = ply.stem[: -len("_fill")]
        csv_path = base / f"{ply.stem}_testpoint.csv"
        if not csv_path.is_file():
            continue
        pipes.append(PipeData(pipe_id=pipe_id, ply_path=ply, csv_path=csv_path))
    return pipes


def load_inspection_points(csv_path: Path) -> list[InspectionPoint]:
    """검사 포인트 CSV 를 읽어 ``InspectionPoint`` 리스트로 반환.

    CSV 컬럼: ``,x,y,z,dx,dy,dz`` (첫 열은 1-base 인덱스).
    ``dx/dy/dz`` 가 빈 셀이면 0.0 으로 보정한다.
    """

    df = pd.read_csv(csv_path, index_col=0)
    points: list[InspectionPoint] = []
    for raw_idx, row in df.iterrows():
        x = float(row["x"])
        y = float(row["y"])
        z = float(row["z"])
        dx = _coerce_float(row.get("dx"))
        dy = _coerce_float(row.get("dy"))
        dz = _coerce_float(row.get("dz"))
        try:
            idx_int = int(raw_idx)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            idx_int = len(points) + 1
        points.append(
            InspectionPoint(
                index=idx_int,
                position=(x, y, z),
                offset=(dx, dy, dz),
            )
        )
    return points


def _coerce_float(value) -> float:
    if value is None:
        return 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return 0.0
    return f
