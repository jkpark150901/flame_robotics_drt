'''
DRT 3D Window Controller with Qt GUI
@auhtor Byunghun Hwang<bh.hwnag@iae.re.kr>
'''

try:
    from PyQt6.QtGui import QImage, QPixmap, QCloseEvent, QFontDatabase, QFont
    from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QMessageBox
    from PyQt6.uic import loadUi
    from PyQt6.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal
except ImportError:
    print("PyQt6 is required to run this application.")

import sys, os
import pathlib
import json
import atexit
from common.zpipe import zpipe_create_pipe, zpipe_destroy_pipe
from common.zpipe import ZPipe

# root directory registration on system environment
ROOT_PATH = pathlib.Path(__file__).parent.parent
APP_NAME = pathlib.Path(__file__).stem
sys.path.append(ROOT_PATH.as_posix())

import argparse
import multiprocessing
import zmq
from multiprocessing import Process
from common.config_loader import load_config
from controller.window import AppWindow as ControlWindow
from util.logger.console import ConsoleLogger


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', required=False, help="Configuration File(*.cfg)", default="controller.cfg")
    parser.add_argument('--verbose_level', nargs='?', required=False, help="Set Verbose Level", default="DEBUG")
    args = parser.parse_args()

    console = ConsoleLogger.get_logger(level="DEBUG")

    app = None
    try:
        configure = load_config(args.config)

        configure["root_path"] = ROOT_PATH
        configure["app_path"] = (pathlib.Path(__file__).parent / APP_NAME)
        configure["verbose_level"] = args.verbose_level.upper()

        if configure["verbose_level"] == "DEBUG":
            console.debug(f"Root Path : {configure['root_path']}")
            console.debug(f"Application Path : {configure['app_path']}")
            console.debug(f"Verbose Level : {configure['verbose_level']}")

        # zmq pipeline
        # create zpipe context
        n_ctx_value = configure.get("n_io_context", 10)
        zpipe_instance = zpipe_create_pipe(io_threads=n_ctx_value)

        app = QApplication(sys.argv)
        font_id = QFontDatabase.addApplicationFont((ROOT_PATH / configure['font_path']).as_posix())
        font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        app.setFont(QFont(font_family, 12))
        main_window = ControlWindow(config=configure, zpipe=zpipe_instance)
        main_window.show()

        exit_cdoe = app.exec()

        # terminate pipeline
        zpipe_destroy_pipe()
        console.info(f"Successfully terminated")
        sys.exit(exit_cdoe)

    except json.JSONDecodeError as e:
        console.critical(f"Configuration File Parse Exception : {e}")
    except Exception as e:
        console.critical(f"General Exception : {e}")
        
    
        
    
