"""
Logging Utility
Handles console and file logging with rotation
"""
import logging
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )
        
        return super().format(record)


def setup_logger(
    name: str = "driver_alertness",
    log_file: Optional[str] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        console_level: Console logging level
        file_level: File logging level
        max_bytes: Max file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger
    
    Example:
        >>> logger = setup_logger("my_module", "logs/app.log")
        >>> logger.info("Application started")
        >>> logger.warning("This is a warning")
        >>> logger.error("An error occurred")
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper()))
    
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation (if specified)
    if log_file:
        # Create log directory
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, file_level.upper()))
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger instance
logger = setup_logger(
    name="driver_alertness",
    log_file="outputs/logs/app.log",
    console_level="INFO",
    file_level="DEBUG"
)


if __name__ == "__main__":
    # Test logger
    print("="*70)
    print("LOGGER TEST")
    print("="*70)
    print()
    
    logger.debug("This is a debug message")
    logger.info("✓ Logger initialized successfully")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    print()
    print("="*70)
    print(f"Log file: outputs/logs/app.log")
    print("="*70)