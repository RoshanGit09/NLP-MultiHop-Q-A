"""
Logging configuration for FinancialKG
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> None:
    """
    Set up logging configuration for the entire package.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name (will be created in log_dir)
        log_dir: Directory for log files (default: 'logs')
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(levelname)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        file_path = log_path / log_file
    else:
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = log_path / f"financial_kg_{timestamp}.log"
    
    file_handler = logging.FileHandler(file_path, mode='a')
    file_handler.setLevel(logging.DEBUG)  # File gets all messages
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Log the setup
    root_logger.info(f"Logging initialized - Level: {level}, File: {file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Usually __name__ of the calling module
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize logging when module is imported
setup_logging()
