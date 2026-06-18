"""
Structured Logging Framework for OASIS Agentic Pipeline
Provides centralized logging configuration with multiple handlers and formatters
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import traceback


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Build colored message
        log_message = f"{color}[{timestamp}] {record.levelname:8s}{reset} {record.name:20s} | {record.getMessage()}"

        # Add exception info if present
        if record.exc_info:
            log_message += f"\n{self.formatException(record.exc_info)}"

        return log_message


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for adding context"""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message with extra context"""
        extra = kwargs.get("extra", {})

        # Add context from adapter
        if self.extra:
            extra.update(self.extra)

        kwargs["extra"] = {"extra_fields": extra}
        return msg, kwargs


class LoggingConfig:
    """Centralized logging configuration"""

    def __init__(
        self,
        app_name: str = "oasis-pipeline",
        log_dir: Optional[str] = None,
        log_level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ):
        self.app_name = app_name
        self.log_level = getattr(logging, log_level.upper())
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # Set up log directory
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        self._configure_logging()

    def _configure_logging(self):
        """Configure logging handlers and formatters"""
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Console handler with colored output
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(ColoredFormatter())
            root_logger.addHandler(console_handler)

        # File handler with standard format
        if self.enable_file:
            log_file = self.log_dir / f"{self.app_name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=self.max_bytes, backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        # JSON handler for structured logging
        if self.enable_json:
            json_file = self.log_dir / f"{self.app_name}.json"
            json_handler = logging.handlers.RotatingFileHandler(
                json_file, maxBytes=self.max_bytes, backupCount=self.backup_count
            )
            json_handler.setLevel(self.log_level)
            json_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(json_handler)

        # Error file handler (only errors and above)
        error_file = self.log_dir / f"{self.app_name}-errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file, maxBytes=self.max_bytes, backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s\n"
            "File: %(pathname)s:%(lineno)d\n"
            "Function: %(funcName)s\n",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        error_handler.setFormatter(error_formatter)
        root_logger.addHandler(error_handler)

    def get_logger(self, name: str, **context) -> LoggerAdapter:
        """Get a logger with optional context"""
        logger = logging.getLogger(name)
        if context:
            return LoggerAdapter(logger, context)
        return LoggerAdapter(logger, {})


# Global logging configuration instance
_logging_config: Optional[LoggingConfig] = None


def setup_logging(
    app_name: str = "oasis-pipeline",
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    **kwargs,
) -> LoggingConfig:
    """Setup global logging configuration"""
    global _logging_config

    _logging_config = LoggingConfig(
        app_name=app_name, log_dir=log_dir, log_level=log_level, **kwargs
    )

    return _logging_config


def get_logger(name: str, **context) -> LoggerAdapter:
    """Get a logger instance"""
    global _logging_config

    if _logging_config is None:
        _logging_config = setup_logging()

    return _logging_config.get_logger(name, **context)


# Convenience functions for common logging patterns
def log_api_request(logger: logging.Logger, method: str, path: str, **kwargs):
    """Log API request"""
    logger.info(
        f"API Request: {method} {path}",
        extra={"event_type": "api_request", "method": method, "path": path, **kwargs},
    )


def log_api_response(
    logger: logging.Logger, method: str, path: str, status_code: int, duration_ms: float, **kwargs
):
    """Log API response"""
    logger.info(
        f"API Response: {method} {path} - {status_code} ({duration_ms:.2f}ms)",
        extra={
            "event_type": "api_response",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            **kwargs,
        },
    )


def log_agent_execution(
    logger: logging.Logger, agent_name: str, patient_id: str, duration_ms: float, **kwargs
):
    """Log agent execution"""
    logger.info(
        f"Agent Execution: {agent_name} for patient {patient_id} ({duration_ms:.2f}ms)",
        extra={
            "event_type": "agent_execution",
            "agent_name": agent_name,
            "patient_id": patient_id,
            "duration_ms": duration_ms,
            **kwargs,
        },
    )


def log_diagnosis(
    logger: logging.Logger,
    patient_id: str,
    diagnosis: str,
    confidence: float,
    approved: bool,
    **kwargs,
):
    """Log diagnosis result"""
    logger.info(
        f"Diagnosis: {patient_id} - {diagnosis} (confidence: {confidence:.2f}%, approved: {approved})",
        extra={
            "event_type": "diagnosis",
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "approved": approved,
            **kwargs,
        },
    )


def log_error(logger: logging.Logger, error: Exception, context: str = "", **kwargs):
    """Log error with context"""
    logger.error(
        f"Error in {context}: {str(error)}",
        exc_info=True,
        extra={
            "event_type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            **kwargs,
        },
    )


def log_performance_metric(
    logger: logging.Logger, metric_name: str, value: float, unit: str = "", **kwargs
):
    """Log performance metric"""
    logger.info(
        f"Performance Metric: {metric_name} = {value}{unit}",
        extra={
            "event_type": "performance_metric",
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            **kwargs,
        },
    )


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(app_name="oasis-pipeline", log_level="DEBUG")

    # Get logger
    logger = get_logger(__name__, service="test")

    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test structured logging
    log_api_request(logger, "POST", "/diagnose", patient_id="OAS2_0001")
    log_api_response(logger, "POST", "/diagnose", 200, 1850.5, patient_id="OAS2_0001")
    log_agent_execution(logger, "VisionAgent", "OAS2_0001", 450.2, result="success")
    log_diagnosis(logger, "OAS2_0001", "Very Mild Dementia", 87.5, True)

    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_error(logger, e, context="test_function", patient_id="OAS2_0001")

    print(f"\nLogs written to: {_logging_config.log_dir}")
