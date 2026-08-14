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
from robot_core.service import EmbeddedRobotCoreClient, ExternalRobotCoreClient
from common.zpipe import zpipe_create_pipe, zpipe_destroy_pipe
from common.zpipe import ZPipe

# root directory registration on system environment
ROOT_PATH = pathlib.Path(__file__).parent.parent
APP_NAME = pathlib.Path(__file__).stem
sys.path.append(ROOT_PATH.as_posix())


if __name__ == "__main__":

    console = ConsoleLogger.get_logger(level="DEBUG")
    robot_core = None
    zapi = None
    zpipe_instance = None

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', required=False, help="Configuration File(*.cfg)", default="viewervedo.cfg")
    parser.add_argument('--verbose_level', nargs='?', required=False, help="Set Verbose Level", default="DEBUG")
    parser.add_argument(
        '--robot_core_mode', choices=('embedded', 'external'), default=None,
        help="Run an embedded Robot Core or connect to a standalone Robot Core service")
    parser.add_argument(
        '--robot_core_endpoint', default=None,
        help="Standalone Robot Core endpoint (default: robot_core_service.endpoint in config)")
    parser.add_argument('--planner_mode', choices=('embedded', 'external'), default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--planner_endpoint', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        configure = load_config(args.config)

        # path planning/IK solver 설정은 viewervedo.cfg와 같은 폴더의 별도 파일로 분리한다.
        # 있으면 병합해서 override하고, 없으면 코드 기본값을 그대로 쓴다.
        extra_config_path = pathlib.Path(args.config).resolve().parent / "path_planning.cfg"
        if extra_config_path.exists():
            configure.update(load_config(extra_config_path))

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
        robot_core_config = dict(
            configure.get("robot_core_service", {})
            or configure.get("planner_service", {})
            or {}
        )
        robot_core_mode = (
            args.robot_core_mode or args.planner_mode
            or robot_core_config.get("mode", "embedded")
        )
        robot_core_endpoint_override = args.robot_core_endpoint or args.planner_endpoint
        if robot_core_endpoint_override:
            robot_core_config["endpoint"] = robot_core_endpoint_override
        if robot_core_mode == "external":
            robot_core = ExternalRobotCoreClient(
                robot_core_config, zapi.robot_core_completed, zpipe=zpipe_instance)
        else:
            robot_core = EmbeddedRobotCoreClient(
                configure, zapi.robot_core_completed)
        robot_core.start()
        viewer.set_robot_core(robot_core)
        console.info(
            f"Robot Core ready: mode={robot_core_mode}, "
            f"pid={robot_core.pid}, "
            f"endpoint={robot_core.endpoint if robot_core_mode == 'external' else '-'}")
        viewer.set_zapi(zapi)
        zapi.run()

        # run render loop (blocks until close)
        viewer.run(60)

        console.info(f"Successfully terminated")

    except json.JSONDecodeError as e:
        console.critical(f"Configuration File Parse Exception : {e}")
    except Exception as e:
        console.critical(f"General Exception : {e}")
    finally:
        if robot_core is not None:
            robot_core.stop()
        if zapi is not None:
            zapi.stop()
        if zpipe_instance is not None:
            zpipe_destroy_pipe()
