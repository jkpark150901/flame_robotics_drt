"""tests/visual 공용 픽스처.

``plugin_pdm`` 디렉토리를 sys.path 에 추가해 ``tools.viewer`` 와
``EndEffectorPoseOptimizer`` 모듈을 임포트할 수 있게 한다 (controller.py:24
와 동일한 패턴).
"""

from __future__ import annotations

import sys
from pathlib import Path

# tests/visual/conftest.py → plugin_pdm/
PLUGIN_ROOT = Path(__file__).resolve().parents[2]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))
