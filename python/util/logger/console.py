"""
Colorful logging on Console
@author Byunghun Hwang<bh.hwang@iae.re.kr>
"""

import logging
from pathlib import Path

import colorlog


class ConsoleLogger:
    """Project-wide logger factory.

    기본 동작은 기존과 동일하게 컬러 콘솔 로그를 출력한다. 필요한 경우
    console/file handler의 level과 파일 경로를 분리해 설정할 수 있다.
    """

    _logger = None
    _console_handler = None
    _file_handler = None

    @staticmethod
    def _level(level, default=logging.DEBUG):
        if isinstance(level, int):
            return level
        return logging._nameToLevel.get(str(level).upper(), default)

    @classmethod
    def get_logger(
        cls,
        level: str = "DEBUG",
        *,
        console_level=None,
        file_level=None,
        log_path=None,
        log_dir=None,
        name: str = "flame_robotics",
        force: bool = False,
    ):
        """Return the shared project logger.

        Args:
            level: logger 기본 level. 기존 호출 호환용.
            console_level: 터미널에 출력할 최소 level.
            file_level: 파일에 기록할 최소 level.
            log_path: 명시적인 로그 파일 경로.
            log_dir: 기본 파일명(`flame_robotics.log`)을 생성할 폴더.
            name: Python logger 이름.
            force: 기존 handler를 제거하고 재설정할지 여부.
        """
        base_level = cls._level(level)
        console_level = cls._level(console_level if console_level is not None else level)
        file_level = None if file_level is None else cls._level(file_level)

        if cls._logger is not None and not force:
            cls._logger.setLevel(min(
                [base_level, console_level] + ([file_level] if file_level is not None else [])
            ))
            return cls._logger

        if cls._logger is not None:
            for handler in list(cls._logger.handlers):
                cls._logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass

        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(min(
            [base_level, console_level] + ([file_level] if file_level is not None else [])
        ))
        logger.propagate = False

        console_formatter = colorlog.ColoredFormatter(
            "[%(asctime)s] %(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red,bg_white",
            },
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        file_handler = None
        if log_path is None and log_dir is not None:
            log_path = Path(log_dir) / "flame_robotics.log"
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setLevel(file_level if file_level is not None else base_level)
            file_handler.setFormatter(logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)d %(message)s"
            ))
            logger.addHandler(file_handler)

        cls._logger = logger
        cls._console_handler = console_handler
        cls._file_handler = file_handler

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logger.level)
        root_logger.addHandler(console_handler)
        if file_handler is not None:
            root_logger.addHandler(file_handler)
        return cls._logger

    @classmethod
    def configure(cls, config=None, *, force: bool = True):
        """Configure logger from a config dict.

        Supported keys:
            level, console_level, file_level, log_path, log_dir, enable_file, name
        """
        config = config or {}
        enable_file = bool(config.get("enable_file", bool(config.get("log_path") or config.get("log_dir"))))
        return cls.get_logger(
            level=config.get("level", "DEBUG"),
            console_level=config.get("console_level", config.get("level", "DEBUG")),
            file_level=config.get("file_level", config.get("level", "DEBUG")),
            log_path=config.get("log_path") if enable_file else None,
            log_dir=config.get("log_dir") if enable_file else None,
            name=config.get("name", "flame_robotics"),
            force=force,
        )

    @classmethod
    def describe(cls):
        """Return current logger routing for diagnostics."""
        if cls._logger is None:
            return {"configured": False}
        handlers = []
        for handler in cls._logger.handlers:
            item = {
                "type": handler.__class__.__name__,
                "level": logging.getLevelName(handler.level),
            }
            if hasattr(handler, "baseFilename"):
                item["path"] = handler.baseFilename
            handlers.append(item)
        return {
            "configured": True,
            "logger": cls._logger.name,
            "level": logging.getLevelName(cls._logger.level),
            "handlers": handlers,
        }
