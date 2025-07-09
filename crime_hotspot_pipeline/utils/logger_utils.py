"""
Logging Utilities Module
Configures and manages logging for the pipeline
"""
import logging
import sys
from datetime import datetime
import os
from typing import Optional


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Optional custom log format
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Default format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_pipeline_logger(
    component: str,
    log_dir: str = "logs",
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Get a logger configured for pipeline components
    
    Args:
        component: Component name (e.g., 'data_loader', 'feature_engineer')
        log_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{component}_{timestamp}.log")
    
    # Custom format for pipeline
    log_format = f'%(asctime)s - [{component}] - %(levelname)s - %(funcName)s - %(message)s'
    
    return setup_logger(
        name=f"crime_hotspot.{component}",
        log_level=log_level,
        log_file=log_file,
        log_format=log_format
    )


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get or create logger for the class"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger