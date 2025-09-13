"""
Logging configuration for OpenInvestments platform.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import structlog
from pythonjsonlogger import jsonlogger

from .config import config


def setup_logging(
    level: str = None,
    log_file: str = None,
    enable_json: bool = True,
    enable_console: bool = True
) -> None:
    """
    Configure structured logging for the platform.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        enable_json: Enable JSON logging format
        enable_console: Enable console logging
    """
    if level is None:
        level = config.log_level

    # Set up structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer() if enable_json else structlog.processors.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # JSON formatter for structured logging
    json_formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        if enable_json:
            console_handler.setFormatter(json_formatter)
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
            )
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)

    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("pymc").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLoggerBase:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def log_performance_metrics(
    operation: str,
    duration: float,
    metadata: Dict[str, Any] = None
) -> None:
    """
    Log performance metrics for operations.

    Args:
        operation: Name of the operation
        duration: Duration in seconds
        metadata: Additional metadata
    """
    logger = get_logger(__name__)
    log_data = {
        "operation": operation,
        "duration_seconds": duration,
        "timestamp": datetime.utcnow().isoformat()
    }

    if metadata:
        log_data.update(metadata)

    logger.info("Performance metric", **log_data)


def log_validation_event(
    model_id: str,
    validation_type: str,
    result: str,
    details: Dict[str, Any] = None
) -> None:
    """
    Log model validation events.

    Args:
        model_id: Unique model identifier
        validation_type: Type of validation performed
        result: Validation result
        details: Additional validation details
    """
    logger = get_logger(__name__)
    log_data = {
        "event_type": "model_validation",
        "model_id": model_id,
        "validation_type": validation_type,
        "result": result,
        "timestamp": datetime.utcnow().isoformat()
    }

    if details:
        log_data.update(details)

    logger.info("Model validation event", **log_data)


def log_risk_metric(
    portfolio_id: str,
    metric_type: str,
    value: float,
    confidence_level: float = None,
    details: Dict[str, Any] = None
) -> None:
    """
    Log risk metric calculations.

    Args:
        portfolio_id: Portfolio identifier
        metric_type: Type of risk metric (VaR, ES, etc.)
        value: Metric value
        confidence_level: Confidence level for the metric
        details: Additional metric details
    """
    logger = get_logger(__name__)
    log_data = {
        "event_type": "risk_metric",
        "portfolio_id": portfolio_id,
        "metric_type": metric_type,
        "value": value,
        "confidence_level": confidence_level or config.default_confidence_level,
        "timestamp": datetime.utcnow().isoformat()
    }

    if details:
        log_data.update(details)

    logger.info("Risk metric calculation", **log_data)
