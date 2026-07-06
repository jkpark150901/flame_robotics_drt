"""Windows 한글 경로용 Open3D 호환 박막.

``open3d.io.read_triangle_mesh`` 가 Windows 의 한글(또는 비-ASCII) 절대경로를
받으면 내부에서 UTF-8 디코딩 오류로 빈 mesh 를 반환한다 (예:
``UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb3 …``). 원인은
Open3D 가 std::string 으로 받은 경로를 fopen 에 그대로 전달하면서 Windows
system code page 와 UTF-8 사이의 불일치를 처리하지 않는 것.

본 모듈은 진입점 (`__main__.py`) 에서 한 번 ``apply()`` 를 호출해 STL 로더를
두 단계 wrapper 로 교체한다:
  1. ``GetShortPathNameW`` 로 8.3 단축경로 변환 시도. NTFS + ASCII leaf 면
     ASCII 경로가 반환되어 Open3D 가 정상 read.
  2. 1단계 결과가 빈 mesh 면 ``trimesh.load`` 로 한 번 더 read 한 뒤
     vertices/faces 를 ``o3d.geometry.TriangleMesh`` 로 옮긴다. trimesh 는
     Python 단에서 path 를 처리하므로 한글 경로/Google Drive 모두 안전.

실측: ``read_point_cloud`` 는 한글 경로/Google Drive 에서도 정상 동작하므로
패치 대상에서 제외했다 (Open3D 0.19.0 기준).

알고리즘 측 ``EndEffectorPoseOptimizer`` / ``JupyterVisualizer`` 코드는 한 줄도
수정하지 않으며, 이 박막은 viewer 진입점이 ``apply()`` 를 호출했을 때만
효과를 가진다.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)
_PATCHED = False


def _to_short_path(path: Any) -> Any:
    """Windows 에서 가능한 경우 경로를 8.3 단축형으로 변환. 비-Windows / 변환
    실패 시 원본 그대로 반환."""

    if os.name != "nt":
        return path
    if isinstance(path, Path):
        path_str = str(path)
    elif isinstance(path, (bytes, bytearray)):
        return path
    else:
        path_str = str(path)

    if not os.path.exists(path_str):
        # 존재하지 않는 경로는 단축이 불가능 — 원본 반환 (에러는 호출자가 처리)
        return path

    GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
    GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    GetShortPathNameW.restype = ctypes.c_uint

    needed = GetShortPathNameW(path_str, None, 0)
    if needed == 0:
        return path
    buf = ctypes.create_unicode_buffer(needed)
    written = GetShortPathNameW(path_str, buf, needed)
    if written == 0:
        return path
    return buf.value


def _trimesh_fallback_read(filename: Any, o3d_module) -> Any:
    """trimesh 로 STL 을 읽어 ``o3d.geometry.TriangleMesh`` 로 변환. 실패 시
    None 반환. trimesh 는 Python 단에서 path 를 처리하므로 한글 경로/Google
    Drive 모두 안전."""

    try:
        import numpy as np  # type: ignore
        import trimesh  # type: ignore
    except ImportError:
        _LOG.warning("trimesh fallback unavailable (numpy/trimesh import failed)")
        return None

    try:
        tm = trimesh.load(str(filename), force="mesh", process=False)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("trimesh.load failed for %s: %s", filename, exc)
        return None

    faces = getattr(tm, "faces", None)
    vertices = getattr(tm, "vertices", None)
    if faces is None or vertices is None or len(faces) == 0:
        return None

    mesh = o3d_module.geometry.TriangleMesh()
    mesh.vertices = o3d_module.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d_module.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    return mesh


def apply() -> None:
    """``open3d.io.read_triangle_mesh`` 를 한글 경로 호환 wrapper 로 교체. 멱등."""

    global _PATCHED
    if _PATCHED:
        return

    import open3d as _o3d  # type: ignore
    import open3d.io as _o3dio  # type: ignore

    # Open3D 가 비-UTF-8 경로에 대해 system code page 로 로그 출력하면 pybind11
    # 가 그 메시지를 UTF-8 로 디코딩하려다 실패한다. 로그를 끔으로써 회피.
    try:
        _o3d.utility.set_verbosity_level(_o3d.utility.VerbosityLevel.Error)
    except Exception:  # noqa: BLE001
        pass

    _orig_read_triangle_mesh = _o3dio.read_triangle_mesh

    def _patched_read_triangle_mesh(filename, *args, **kwargs):
        # 1단계: 8.3 단축경로 변환 후 Open3D 에 위임. NTFS + ASCII leaf 면 성공.
        mesh = _orig_read_triangle_mesh(_to_short_path(filename), *args, **kwargs)
        if len(mesh.triangles) > 0:
            return mesh

        # 2단계: 빈 mesh 인 경우 trimesh fallback. 한글 leaf / Google Drive 케이스.
        if isinstance(filename, (bytes, bytearray)) or not Path(str(filename)).is_file():
            return mesh

        fallback = _trimesh_fallback_read(filename, _o3d)
        if fallback is not None:
            _LOG.debug("trimesh fallback succeeded for: %s", filename)
            return fallback
        _LOG.error(
            "trimesh fallback failed for empty Open3D mesh — downstream will see 0 triangles: %s",
            filename,
        )
        return mesh

    _o3dio.read_triangle_mesh = _patched_read_triangle_mesh

    _PATCHED = True
