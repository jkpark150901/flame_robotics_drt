"""tools/viewer 가 사용하는 경로 상수.

plugin_pdm 디렉토리 위치를 절대경로로 한 번만 계산해 두고, data 디렉토리
및 robot URDF 경로를 함수로 노출한다.
"""

from pathlib import Path

# tools/viewer/app_paths.py → parents[2] == plugin_pdm 디렉토리
PLUGIN_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PLUGIN_ROOT / "data"
ROBOT_MODELS_DIR: Path = DATA_DIR / "robot_models v1.5"


def dda_urdf_path() -> Path:
    return ROBOT_MODELS_DIR / "robots" / "rb10_1300e_DDA.urdf"


def rt_urdf_path() -> Path:
    return ROBOT_MODELS_DIR / "robots" / "rb10_1300e_RT.urdf"
