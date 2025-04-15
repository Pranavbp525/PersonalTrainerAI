# elk_logging.py
import logging
import json
import socket
import time
import os
from logging.handlers import SocketHandler
from typing import Dict, Any, Optional
import platform

# --- Import the centralized config ---
from config import config, ConfigError

APP_NAME = "fitness-chatbot"

class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record.
    """
    def __init__(self, **kwargs):
        # Use the ENVIRONMENT from the config object
        self.default_fields = {"environment": config.ENVIRONMENT, **kwargs}
        super(JsonFormatter, self).__init__()

    def format(self, record):
        log_record = {
            "@timestamp": self.formatTime(record, self.datefmt),
            "message": record.getMessage(),
            "level": record.levelname,
            "application": APP_NAME,
            "host": {
                "name": socket.gethostname()
            },
            "function": record.funcName,
            "line_number": record.lineno,
            "pathname": record.pathname,
            # Environment is added via default_fields below
        }

        log_record.update(self.default_fields) # Adds environment

        if hasattr(record, 'extra'):
            extra_filtered = {k: v for k, v in record.extra.items() if k != 'environment'}
            log_record.update(extra_filtered)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record) + '\n'


class LogstashHandler(SocketHandler):
    # ... (no changes needed in LogstashHandler) ...
    def __init__(self, host, port):
        super(LogstashHandler, self).__init__(host, port)
        self.retries = 0
        self.max_retries = 5
        self.retry_interval = 2

    def emit(self, record):
        try:
            if self.sock is None:
                self.createSocket()
            msg = self.format(record)
            self.sock.sendall(msg.encode('utf-8'))
            self.retries = 0
        except socket.error:
            self.retries += 1
            if self.retries <= self.max_retries:
                time.sleep(self.retry_interval)
                self.createSocket()
                msg = self.format(record)
                self.sock.sendall(msg.encode('utf-8'))
            else:
                logging.Handler.handleError(self, record)
        except Exception:
             logging.Handler.handleError(self, record)


class ContextLogger:
    # ... (no changes needed in ContextLogger) ...
    def __init__(self, logger, context=None):
        self.logger = logger
        self.context = context or {}

    def add_context(self, **kwargs):
        self.context.update(kwargs)
        return self

    def _log(self, level, msg, *args, **kwargs):
        extra = kwargs.get('extra', {})
        kwargs['extra'] = {**self.context, **extra}
        # No need to handle environment specially here anymore
        if self.logger.isEnabledFor(getattr(logging, level.upper())):
           getattr(self.logger, level)(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        return self._log('debug', msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        return self._log('info', msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        return self._log('warning', msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        return self._log('error', msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        return self._log('critical', msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        kwargs['exc_info'] = True
        return self._log('error', msg, *args, **kwargs)


def setup_elk_logging(logger_name=None, console_level_str="INFO", logstash_level_str="INFO"):
    """
    Set up a logger with Logstash integration using settings from config object.
    """
    console_level = getattr(logging, console_level_str.upper(), logging.INFO)
    logstash_level = getattr(logging, logstash_level_str.upper(), logging.INFO)
    logger_level = getattr(logging, config.LOGGING_LEVEL.upper(), logging.DEBUG)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    # --- FIX 1: Remove %(environment)s from console format string ---
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)',
         datefmt='%Y-%m-%d %H:%M:%S'
    )
    # --- FIX 2: Remove the problematic setLogRecordFactory call ---
    # logging.setLogRecordFactory(...) # REMOVE THIS LINE
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Logstash handler
    if config.LOGSTASH_HOST and config.LOGSTASH_PORT:
        try:
            logstash_handler = LogstashHandler(config.LOGSTASH_HOST, config.LOGSTASH_PORT)
            logstash_handler.setLevel(logstash_level)
            # JsonFormatter correctly adds environment from config
            logstash_formatter = JsonFormatter()
            logstash_handler.setFormatter(logstash_formatter)
            logger.addHandler(logstash_handler)
            logger.info(f"ELK Logstash handler initialized for logger '{logger.name}' -> {config.LOGSTASH_HOST}:{config.LOGSTASH_PORT}")
        except Exception as e:
            logger.error(f"Failed to initialize Logstash handler for logger '{logger.name}': {e}. Logging to console only.", exc_info=True)
    else:
        # This log message itself might fail initially if environment isn't set yet, but removing it from the formatter prevents the crash
        logger.warning(f"Logstash host/port not configured. Skipping Logstash handler setup for '{logger.name}'.")

    return ContextLogger(logger)

def get_agent_logger(agent_name: str, session_id: Optional[str] = None):
    # ... (no changes needed in get_agent_logger) ...
    logger_name = f"{APP_NAME}.agent.{agent_name}"
    logger = setup_elk_logging(logger_name) # Uses default level strings from function signature

    context = {
        "agent": agent_name,
    }
    if session_id:
        context["session_id"] = session_id
    return logger.add_context(**context)