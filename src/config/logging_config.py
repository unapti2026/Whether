"""
Logging Configuration

Este módulo centraliza la configuración de logging para el proyecto.
"""

import logging
import logging.config
from pathlib import Path
from typing import Dict, Any, Optional
from .paths import LOGS_DIR

# Default logging configuration
DEFAULT_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(LOGS_DIR / "weather_prediction.log"),
            'mode': 'a',
            'encoding': 'utf-8'
        },
        'error_file': {
            'class': 'logging.FileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': str(LOGS_DIR / "errors.log"),
            'mode': 'a',
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'src': {  # Application logger
            'handlers': ['console', 'file', 'error_file'],
            'level': 'DEBUG',
            'propagate': False
        },
        'src.data': {  # Data processing logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'src.data.processors': {  # Processors logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'src.data.imputation': {  # Imputation logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'src.data.visualization': {  # Visualization logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        config: Custom logging configuration dictionary
    """
    logging_config = config or DEFAULT_LOGGING_CONFIG
    
    # Ensure logs directory exists
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info("Logging configuration initialized")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def set_log_level(level: str, logger_name: str = 'src') -> None:
    """
    Set the log level for a specific logger.
    
    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        logger_name: Name of the logger to configure
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))

def add_file_handler(filename: str, level: str = 'INFO', formatter: str = 'detailed') -> None:
    """
    Add a file handler to the root logger.
    
    Args:
        filename: Name of the log file
        level: Log level for the handler
        formatter: Formatter to use
    """
    # Ensure logs directory exists
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Create handler
    handler = logging.FileHandler(LOGS_DIR / filename, mode='a', encoding='utf-8')
    handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter_config = DEFAULT_LOGGING_CONFIG['formatters'][formatter]
    formatter_obj = logging.Formatter(
        formatter_config['format'],
        datefmt=formatter_config['datefmt']
    )
    handler.setFormatter(formatter_obj)
    
    # Add to root logger
    logging.getLogger('').addHandler(handler) 