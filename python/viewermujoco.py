"""
DRT MuJoCo simulation backend.

This runs separately from viewervedo and listens on its own ZAPI channel.
"""

import argparse
import json
import pathlib
import sys

from common.zpipe import zpipe_create_pipe, zpipe_destroy_pipe
from util.logger.console import ConsoleLogger
from viewermujoco.simulator import MujocoSimulator
from viewermujoco.zapi import ZAPI


ROOT_PATH = pathlib.Path(__file__).parent.parent
APP_NAME = pathlib.Path(__file__).stem
sys.path.append(ROOT_PATH.as_posix())


if __name__ == "__main__":
    console = ConsoleLogger.get_logger(level="DEBUG")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="?", required=False, help="Configuration File(*.cfg)", default="viewermujoco.cfg")
    parser.add_argument("--verbose_level", nargs="?", required=False, help="Set Verbose Level", default="DEBUG")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as cfile:
            configure = json.load(cfile)

        configure["root_path"] = ROOT_PATH
        configure["app_path"] = pathlib.Path(__file__).parent / APP_NAME
        configure["verbose_level"] = args.verbose_level.upper()

        if configure["verbose_level"] == "DEBUG":
            console.debug(f"Root Path : {configure['root_path']}")
            console.debug(f"Application Path : {configure['app_path']}")
            console.debug(f"Verbose Level : {configure['verbose_level']}")

        n_ctx_value = configure.get("n_io_context", 10)
        zpipe_instance = zpipe_create_pipe(io_threads=n_ctx_value)

        simulator = MujocoSimulator(config=configure)
        zapi_config = configure.get("zapi", {}).copy()
        zapi_config["model"] = configure.get("model", "")
        zapi_config["models"] = configure.get("models", [])
        zapi_config["urdf"] = configure.get("urdf", [])
        zapi_config["operation_mode"] = configure.get("operation_mode", "simulation")
        zapi = ZAPI(config=zapi_config, zpipe=zpipe_instance, simulator=simulator)
        zapi.run()

        simulator.run(configure.get("fps", 60))

        zapi.stop()
        zpipe_destroy_pipe()
        console.info("Successfully terminated")

    except json.JSONDecodeError as exc:
        console.critical(f"Configuration File Parse Exception : {exc}")
    except Exception as exc:
        console.critical(f"General Exception : {exc}")
