"""Standalone DRT robot-core service for pose determination and path planning."""

import argparse
import pathlib
import sys

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from common.config_loader import load_config
from util.logger.console import ConsoleLogger
from robot_core.service import serve_robot_core


def main():
    parser = argparse.ArgumentParser(description="DRT robot-core service")
    parser.add_argument("--config", default=str(pathlib.Path(__file__).with_name("viewervedo.cfg")))
    parser.add_argument("--transport", default=None, help="zapi transport (tcp/ipc/inproc)")
    parser.add_argument("--channel", default=None, help="zapi channel (address for tcp, path for ipc)")
    parser.add_argument("--port", default=None, type=int, help="zapi port (tcp only)")
    parser.add_argument("--endpoint", default=None, help="Legacy tcp://host:port endpoint override")
    parser.add_argument("--verbose_level", default="INFO")
    args = parser.parse_args()

    config = load_config(args.config)
    extra_config_path = pathlib.Path(args.config).resolve().parent / "path_planning.cfg"
    if extra_config_path.exists():
        config.update(load_config(extra_config_path))
    config["root_path"] = ROOT_PATH
    config["verbose_level"] = args.verbose_level.upper()
    ConsoleLogger.configure(config.get("logging", {}) or {}, force=True)

    service_config = dict(
        config.get("robot_core_service", {}) or config.get("planner_service", {}) or {})
    if args.endpoint:
        service_config["endpoint"] = args.endpoint
    if args.transport:
        service_config["transport"] = args.transport
    if args.channel:
        service_config["channel"] = args.channel
    if args.port is not None:
        service_config["port"] = args.port

    try:
        serve_robot_core(config, service_config)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
