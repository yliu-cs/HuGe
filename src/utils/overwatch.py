"""
overwatch.py

Utility class for creating a centralized/standardized logger (built on Rich) and accelerate handler.
"""

import logging
import logging.config
import os
from contextlib import nullcontext
from logging import LoggerAdapter
from typing import Any, Callable, ClassVar, Dict, MutableMapping, Tuple, Union

RICH_FORMATTER, DATEFMT = "| >> %(message)s", "[%H:%M:%S %m/%d]"
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {"simple-console": {"format": RICH_FORMATTER, "datefmt": DATEFMT}},
    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler",
            "formatter": "simple-console",
            "markup": True,
            "rich_tracebacks": True,
            "show_level": True,
            "show_path": True,
            "show_time": True,
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}
logging.config.dictConfig(LOG_CONFIG)

_warned_message = set()


class ContextAdapter(LoggerAdapter):
    CTX_PREFIXES: ClassVar[Dict[int, str]] = {
        **{0: "[*] "},
        **{idx: "|=> ".rjust(4 + (idx * 4)) for idx in [1, 2, 3]},
    }

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> Tuple[str, MutableMapping[str, Any]]:
        ctx_level = kwargs.pop("ctx_level", 0)
        return f"{self.CTX_PREFIXES[ctx_level]}{msg}", kwargs


class DistributedOverwatch:
    def __init__(self, name: str) -> None:
        self.logger = ContextAdapter(logging.getLogger(name), extra={})
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._rank = int(os.environ.get("RANK", 0))
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._is_main_process = self._rank == 0

        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        self.logger.setLevel(
            logging.INFO if self._is_main_process else logging.ERROR
        )

        self._file_handler = None

    def add_file_handler(self, log_dir: str, filename: str = "train.log") -> None:
        """Add a file handler to save logs to disk."""
        if self._file_handler is not None:
            return
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, filename)
        self._file_handler = logging.FileHandler(filepath, mode="a", encoding="utf-8")
        self._file_handler.setFormatter(logging.Formatter(RICH_FORMATTER, datefmt=DATEFMT))
        self.logger.logger.addHandler(self._file_handler)

    @staticmethod
    def _identity_decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        if self._is_main_process:
            return self._identity_decorator
        else:
            def skip_fn(fn: Callable[..., Any]) -> Callable[..., Any]:
                def wrapper(*args, **kwargs):
                    return None
                return wrapper
            return skip_fn

    def is_rank_zero(self) -> bool:
        return self._is_main_process

    def rank(self) -> int:
        return self._rank

    def local_rank(self) -> int:
        return self._local_rank

    def world_size(self) -> int:
        return self._world_size

    def warning_once(self, msg):
        if msg not in _warned_message:
            self.warning(msg)
            _warned_message.add(msg)


class PureOverwatch:
    def __init__(self, name: str) -> None:
        self.logger = ContextAdapter(logging.getLogger(name), extra={})
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical
        self.logger.setLevel(logging.INFO)

        self._file_handler = None

    def add_file_handler(self, log_dir: str, filename: str = "train.log") -> None:
        """Add a file handler to save logs to disk."""
        if self._file_handler is not None:
            return
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, filename)
        self._file_handler = logging.FileHandler(filepath, mode="a", encoding="utf-8")
        self._file_handler.setFormatter(logging.Formatter(RICH_FORMATTER, datefmt=DATEFMT))
        self.logger.logger.addHandler(self._file_handler)

    @staticmethod
    def get_identity_ctx() -> Callable[..., Any]:
        def identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn
        return identity

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        return self.get_identity_ctx()

    @staticmethod
    def is_rank_zero() -> bool:
        return True

    @staticmethod
    def rank() -> int:
        return 0

    def local_rank(self) -> int:
        return 0

    @staticmethod
    def world_size() -> int:
        return 1

    def warning_once(self, msg):
        if msg not in _warned_message:
            self.warning(msg)
            _warned_message.add(msg)


def initialize_overwatch(name: str) -> Union[DistributedOverwatch, PureOverwatch]:
    return (
        DistributedOverwatch(name)
        if int(os.environ.get("WORLD_SIZE", -1)) != -1
        else PureOverwatch(name)
    )
