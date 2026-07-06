'''
ZPipe Proxy Application
@auhtor Byunghun Hwang<bh.hwnag@iae.re.kr>
'''

try:
    from PyQt6.QtGui import QImage, QPixmap, QCloseEvent, QFontDatabase, QFont
    from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QMessageBox
    from PyQt6.uic import loadUi
    from PyQt6.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal
except ImportError:
    print("PyQt6 is required to run this application.")

import sys
import os
import json
import time
import argparse
import pathlib

from common.config_loader import load_config
from common.zpipe import ZPipe, AsyncZSocket, zpipe_create_pipe, zpipe_destroy_pipe
from util.logger.console import ConsoleLogger
from common import zapi
from zproxy.window import AppWindow

ROOT_PATH = pathlib.Path(__file__).parent.parent
APP_NAME = pathlib.Path(__file__).stem
sys.path.append(ROOT_PATH.as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="python/zproxy.cfg", help="Path to config file")
    parser.add_argument('--verbose_level', nargs='?', required=False, help="Set Verbose Level", default="DEBUG")
    args = parser.parse_args()

    console = ConsoleLogger.get_logger(level="DEBUG")

    try:
        configure = load_config(args.config)

        # add path
        configure["root_path"] = ROOT_PATH
        configure["app_path"] = (pathlib.Path(__file__).parent / APP_NAME)
        configure["verbose_level"] = args.verbose_level.upper()

        # create zpipe context
        n_ctx_value = configure.get("n_io_context", 10)
        zpipe_instance = zpipe_create_pipe(io_threads=n_ctx_value)

        # run app
        app = QApplication(sys.argv)
        font_id = QFontDatabase.addApplicationFont((ROOT_PATH / configure['font_path']).as_posix())
        font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        app.setFont(QFont(font_family, 12))
        appwindow = AppWindow(config=configure, zpipe=zpipe_instance)
        appwindow.show()

        exit_cdoe = app.exec()

        # terminate pipeline
        zpipe_destroy_pipe()
        console.info(f"Successfully terminated")
        sys.exit(exit_cdoe)

    except json.JSONDecodeError as e:
        console.critical(f"Configuration File Parse Exception : {e}")
    except Exception as e:
        console.critical(f"General Exception : {e}")

