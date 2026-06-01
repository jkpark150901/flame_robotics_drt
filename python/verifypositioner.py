"""
verifypositioner.py
===================
Positioner(모션 캡쳐 기반) 궤적 검증 툴 진입점.
로봇 SDK 없이 NatNet + 계획 CSV만으로 동작.

실행:
  python python/verifypositioner.py
  python python/verifypositioner.py --config python/verifypositioner.cfg
"""

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

_PYTHON_PATH = pathlib.Path(__file__).parent
ROOT_PATH    = _PYTHON_PATH.parent

sys.path.append(str(ROOT_PATH))
sys.path.append(str(_PYTHON_PATH))

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s  %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Verify Positioner")
    parser.add_argument('--config', default=str(_PYTHON_PATH / 'verifypositioner.cfg'),
                        help='JSON config file (default: python/verifypositioner.cfg)')
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            config = json.load(f)
    except FileNotFoundError:
        log.warning("Config not found: %s — using empty config.", args.config)
        config = {}
    except json.JSONDecodeError as e:
        log.critical("Config parse error: %s", e)
        sys.exit(1)

    config['root_path'] = ROOT_PATH

    app = QApplication(sys.argv)

    font_path = ROOT_PATH / config.get('font_path', 'python/resource/NanumSquareR.ttf')
    if font_path.is_file():
        fid = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(fid)
        if families:
            app.setFont(QFont(families[0], 11))

    from verifytool.verifypositioner import CsvMocapVerifyWindow
    window = CsvMocapVerifyWindow(config=config)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
