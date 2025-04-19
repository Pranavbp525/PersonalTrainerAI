# elk_logging.py

import logging
import json
import socket
import time
from logging.handlers import SocketHandler
from typing import Optional, Dict, Any

from config import config  # your centralized config

APP_NAME = "fitness-chatbot"

class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record.
    """
    def __init__(self, **kwargs):
        # include environment and any other static fields
        self.default_fields = {"environment": config.ENVIRONMENT, **kwargs}
        super().__init__()

    def format(self, record):
        log_record: Dict[str, Any] = {
            "@timestamp": self.formatTime(record, self.datefmt),
            "message": record.getMessage(),
            "level": record.levelname,
            "application": APP_NAME,
            "host": {"name": socket.gethostname()},
            "function": record.funcName,
            "line_number": record.lineno,
            "pathname": record.pathname,
        }
        log_record.update(self.default_fields)

        # include any extra context
        extra = getattr(record, "extra", None)
        if extra:
            filtered = {k: v for k, v in extra.items() if k != "environment"}
            log_record.update(filtered)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record) + "\n"


class LogstashHandler(SocketHandler):
    """
    A socket handler that sends JSON‐formatted logs to Logstash,
    but silently drops any failures to avoid blocking your app.
    """
    def __init__(self, host: str, port: int):
        super().__init__(host, port)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self.sock is None:
                self.createSocket()
            msg = self.format(record)
            self.sock.sendall(msg.encode("utf-8"))
        except Exception:
            # silently drop on any failure
            return


class ContextLogger:
    """
    A thin wrapper that allows attaching context to every log call.
    """
    def __init__(self, logger: logging.Logger, context: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.context = context or {}

    def add_context(self, **kwargs: Any) -> "ContextLogger":
        self.context.update(kwargs)
        return self

    def _log(self, level: str, msg: str, *args: Any, **kwargs: Any) -> None:
        extra = kwargs.get("extra", {})
        kwargs["extra"] = {**self.context, **extra}
        if self.logger.isEnabledFor(getattr(logging, level.upper(), logging.INFO)):
            getattr(self.logger, level)(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log("debug", msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log("info", msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log("warning", msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log("error", msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log("critical", msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        kwargs["exc_info"] = True
        self._log("error", msg, *args, **kwargs)


# ─── Root logger configuration (only once) ────────────────────────────────────

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.WARNING)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
_root_logger.addHandler(_console_handler)


# ─── Factory to get a ContextLogger for any named logger ───────────────────────

def setup_elk_logging(
    logger_name: Optional[str] = None,
    console_level_str: str = "WARN",
    logstash_level_str: str = "ERROR"
) -> ContextLogger:
    logger = logging.getLogger(logger_name)
    # set the level for this specific logger
    console_level = getattr(logging, console_level_str.upper(), logging.WARN)
    logger.setLevel(console_level)

    # add Logstash handler once if configured
    if config.LOGSTASH_HOST and config.LOGSTASH_PORT:
        if not any(isinstance(h, LogstashHandler) for h in logger.handlers):
            try:
                lh = LogstashHandler(config.LOGSTASH_HOST, config.LOGSTASH_PORT)
                lh.setLevel(getattr(logging, logstash_level_str.upper(), logging.ERROR))
                lh.setFormatter(JsonFormatter())
                logger.addHandler(lh)
            except Exception:
                # if even that fails, emit to console only
                logger.error("Failed to initialize Logstash handler", exc_info=True)

    return ContextLogger(logger)


def get_agent_logger(agent_name: str, session_id: Optional[str] = None) -> ContextLogger:
    """
    Convenience for per‐agent loggers, with session_id context.
    """
    name = f"{APP_NAME}.agent.{agent_name}"
    ctx = setup_elk_logging(name, console_level_str="WARN", logstash_level_str="ERROR")
    context = {"agent": agent_name}
    if session_id:
        context["session_id"] = session_id
    return ctx.add_context(**context)
