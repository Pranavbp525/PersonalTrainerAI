import logging
import json
import socket
import time
import os
from logging.handlers import SocketHandler
from typing import Dict, Any, Optional
import platform

# Configure constants
LOGSTASH_HOST = os.environ.get("LOGSTASH_HOST", "localhost")
LOGSTASH_PORT = int(os.environ.get("LOGSTASH_PORT", 5044))
ENV = os.environ.get("ENVIRONMENT", "development")
APP_NAME = "fitness-chatbot"

class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record.
    """
    def __init__(self, **kwargs):
        self.default_fields = kwargs
        super(JsonFormatter, self).__init__()
    
    def format(self, record):
        # Create the base log record
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
            "environment": ENV,
        }
        
        # Add default fields
        log_record.update(self.default_fields)
        
        # Add extra fields
        if hasattr(record, 'extra'):
            for key, value in record.extra.items():
                log_record[key] = value
        
        # Add exception info if available
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        # Make sure we return valid JSON with a newline at the end
        return json.dumps(log_record) + '\n'
    

class LogstashHandler(SocketHandler):
    """
    Socket handler that sends logs to Logstash.
    """
    def __init__(self, host, port):
        super(LogstashHandler, self).__init__(host, port)
        self.retries = 0
        self.max_retries = 5
        self.retry_interval = 2  # seconds

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
                self.handleError(record)

class ContextLogger:
    """
    A class that wraps a logger and adds context to all logs.
    """
    def __init__(self, logger, context=None):
        self.logger = logger
        self.context = context or {}
    
    def add_context(self, **kwargs):
        """Add additional context to the logger."""
        self.context.update(kwargs)
        return self
    
    def _log(self, level, msg, *args, **kwargs):
        # Merge extra with context
        extra = kwargs.get('extra', {})
        kwargs['extra'] = {**self.context, **extra}
        return getattr(self.logger, level)(msg, *args, **kwargs)
    
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
        return self._log('exception', msg, *args, **kwargs)

def setup_elk_logging(logger_name=None, console_level=logging.INFO, logstash_level=logging.INFO):
    """
    Set up a logger with Logstash integration.
    
    Args:
        logger_name: The name of the logger (default: root logger)
        console_level: Log level for console output
        logstash_level: Log level for Logstash output
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(min(console_level, logstash_level))
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Logstash handler
    try:
        logstash_handler = LogstashHandler(LOGSTASH_HOST, LOGSTASH_PORT)
        logstash_handler.setLevel(logstash_level)
        logstash_formatter = JsonFormatter()
        logstash_handler.setFormatter(logstash_formatter)
        logger.addHandler(logstash_handler)
        logger.info(f"ELK logging initialized for {logger_name} to {LOGSTASH_HOST}:{LOGSTASH_PORT}")
    except Exception as e:
        # If Logstash connection fails, log error but continue
        logger.warning(f"Failed to connect to Logstash: {e}. Will log to console only.")
    
    return ContextLogger(logger)

def get_agent_logger(agent_name: str, session_id: Optional[str] = None):
    """
    Get a logger preconfigured for a specific agent.
    
    Args:
        agent_name: Name of the agent (e.g., "coordinator", "planning_agent")
        session_id: Optional session ID for context
        
    Returns:
        A ContextLogger instance with agent context
    """
    logger_name = f"{APP_NAME}.agent.{agent_name}"
    logger = setup_elk_logging(logger_name)
    
    context = {
        "agent": agent_name,
    }
    
    if session_id:
        context["session_id"] = session_id
    
    return logger.add_context(**context)

