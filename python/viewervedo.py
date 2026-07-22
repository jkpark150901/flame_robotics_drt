"""
DRT 3D Viewer 3D App with Vedo(https://vedo.embl.es/)
@author Byunghun Hwang <bh.hwang@iae.re.kr>
"""

import sys, os
import pathlib
import json
import argparse

from common.config_loader import load_config
from util.logger.console import ConsoleLogger
from viewervedo.visualizer import Visualizer
from viewervedo.zapi import ZAPI
from common.zpipe import zpipe_create_pipe, zpipe_destroy_pipe
from common.zpipe import ZPipe

# root directory registration on system environment
ROOT_PATH = pathlib.Path(__file__).parent.parent
APP_NAME = pathlib.Path(__file__).stem
sys.path.append(ROOT_PATH.as_posix())


if __name__ == "__main__":

    console = ConsoleLogger.get_logger(level="DEBUG")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', required=False, help="Configuration File(*.cfg)", default="viewervedo.cfg")
    parser.add_argument('--verbose_level', nargs='?', required=False, help="Set Verbose Level", default="DEBUG")
    args = parser.parse_args()

    try:
        configure = load_config(args.config)

        configure["root_path"] = ROOT_PATH
        configure["app_path"] = (pathlib.Path(__file__).parent / APP_NAME)
        configure["verbose_level"] = args.verbose_level.upper()
        log_config = configure.get("logging", {}) or {}
        log_config.setdefault("level", configure["verbose_level"])
        console = ConsoleLogger.configure(log_config, force=True)
        console.debug(f"Logger routing : {ConsoleLogger.describe()}")

        if configure["verbose_level"] == "DEBUG":
            console.debug(f"Root Path : {configure['root_path']}")
            console.debug(f"Application Path : {configure['app_path']}")
            console.debug(f"Verbose Level : {configure['verbose_level']}")

        # create zpipe context
        n_ctx_value = configure.get("n_io_context", 10)
        zpipe_instance = zpipe_create_pipe(io_threads=n_ctx_value)

        # create zapi (communication layer)
        zapi_config = configure.get("zapi", {})
        viewer = Visualizer(config=configure)
        zapi = ZAPI(config=zapi_config, zpipe=zpipe_instance, visualizer=viewer)
        viewer.set_zapi(zapi)
        zapi.run()

        # run render loop (blocks until close)
        viewer.run(60)

        # cleanup communication
        zapi.stop()

        # terminate pipeline
        zpipe_destroy_pipe()
        console.info(f"Successfully terminated")

    except json.JSONDecodeError as e:
        console.critical(f"Configuration File Parse Exception : {e}")
    except Exception as e:
        console.critical(f"General Exception : {e}")
