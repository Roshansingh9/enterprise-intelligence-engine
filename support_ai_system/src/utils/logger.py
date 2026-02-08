"""
Logging Setup
=============
Configures logging for the entire system.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Any, Dict

# Try to import rich for fancy console output
try:
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def setup_logging(config: Dict[str, Any], log_level: str = "INFO") -> None:
    """
    Configure logging for the application.
    
    Args:
        config: System configuration
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory
    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get log settings
    log_config = config.get('logging', {})
    max_size = log_config.get('max_size_mb', 50) * 1024 * 1024
    backup_count = log_config.get('backup_count', 5)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Log format
    file_format = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Extract just the filename (remove any path prefix to avoid logs/logs issue)
    system_log_name = Path(log_config.get('system_log', 'system.log')).name
    error_log_name = Path(log_config.get('error_log', 'errors.log')).name
    learning_log_name = Path(log_config.get('learning_log', 'learning.log')).name
    
    # System log handler (all logs)
    system_handler = RotatingFileHandler(
        log_dir / system_log_name,
        maxBytes=max_size,
        backupCount=backup_count
    )
    system_handler.setLevel(logging.DEBUG)
    system_handler.setFormatter(file_format)
    root_logger.addHandler(system_handler)
    
    # Error log handler (errors only)
    error_handler = RotatingFileHandler(
        log_dir / error_log_name,
        maxBytes=max_size,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)
    root_logger.addHandler(error_handler)
    
    # Learning log handler
    learning_logger = logging.getLogger('learning')
    learning_handler = RotatingFileHandler(
        log_dir / learning_log_name,
        maxBytes=max_size,
        backupCount=backup_count
    )
    learning_handler.setLevel(logging.DEBUG)
    learning_handler.setFormatter(file_format)
    learning_logger.addHandler(learning_handler)
    
    # Console handler
    if log_config.get('console_enabled', True):
        if RICH_AVAILABLE and log_config.get('rich_console', True):
            console_handler = RichHandler(
                show_time=True,
                show_path=False,
                rich_tracebacks=True
            )
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(
                '%(levelname)s: %(message)s'
            ))
        
        console_handler.setLevel(getattr(logging, log_level))
        root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized at {log_level} level")


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)
