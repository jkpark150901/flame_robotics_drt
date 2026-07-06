'''
Calibration & Verification Tool
Entry point — mirrors simtool.py pattern.
'''

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFontDatabase, QFont
except ImportError:
    raise ImportError("PyQt6 is required.")

import sys
import pathlib
import json
import argparse
import logging

# python/ 디렉토리 (이 파일의 위치)
_PYTHON_PATH = pathlib.Path(__file__).parent
# 저장소 루트 (python/ 의 부모)
ROOT_PATH = _PYTHON_PATH.parent
APP_NAME = pathlib.Path(__file__).stem

sys.path.append(ROOT_PATH.as_posix())
sys.path.append(_PYTHON_PATH.as_posix())

from common.config_loader import load_config

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s  %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Calibration & Verification Tool")
    _default_cfg = _PYTHON_PATH / 'verifytool.cfg'
    parser.add_argument('--config', default=str(_default_cfg),
                        help='JSON config file (default: python/verifytool.cfg)')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        log.critical("Config file not found: %s", args.config)
        sys.exit(1)
    except json.JSONDecodeError as e:
        log.critical("Config parse error: %s", e)
        sys.exit(1)

    config['root_path'] = ROOT_PATH
    config['app_path'] = _PYTHON_PATH / APP_NAME

    app = QApplication(sys.argv)

    font_path = ROOT_PATH / config.get('font_path', 'python/resource/NanumSquareR.ttf')
    if font_path.is_file():
        fid = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(fid)
        if families:
            app.setFont(QFont(families[0], 11))
    else:
        log.warning("Font not found: %s", font_path)

    from verifytool.verifycobotwindow import VerifyCobotWindow
    window = VerifyCobotWindow(config=config)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
