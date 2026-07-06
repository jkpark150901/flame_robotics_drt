"""pytest 공용 설정.

본 디렉토리의 테스트는 plugin_pdm 루트에서 직접 모듈을 import 한다
(예: ``from EndEffectorPoseOptimizer import EndEffectorPoseOptimizer``).
플러그인이 패키지로 배포되기 전이라 sys.path 에 루트를 추가해 둔다.

Windows + 한글 경로 환경에서 Open3D 의 verbose 로그가 CP949 로 인코딩한
경로 메시지를 stderr 에 쓰면 pybind11 이 UTF-8 로 디코딩하다 UnicodeDecodeError
를 발생시키므로, ``tools.viewer._open3d_compat.apply()`` 로 verbosity 를
Error 레벨로 낮춰 그 경로를 차단한다 (URDF/PLY 직접 로드 회귀 테스트에 필수).
"""

import sys
from pathlib import Path

PLUGIN_PDM_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_PDM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PDM_ROOT))

try:
    from python.plugins.poseDeterminator.tools.viewer import _open3d_compat

    _open3d_compat.apply()
except Exception:  # noqa: BLE001 — 환경/패키지 부재는 테스트 자체가 skip 처리
    pass
