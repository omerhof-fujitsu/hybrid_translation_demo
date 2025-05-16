"""
Logger utility for the translation system.
"""

import logging
import os
import sys
from typing import Optional

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Initialize logger
logger = logging.getLogger("translation_system")
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Create file handler
file_handler = logging.FileHandler("logs/translation.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Set up colorized output
try:
    import colorama
    from colorama import Fore, Style
    
    colorama.init(autoreset=True)
    
    class ColoredFormatter(logging.Formatter):
        """Custom formatter to colorize log levels."""
        
        COLORS = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT
        }
        
        def format(self, record):
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
            return super().format(record)
    
    # Apply colored formatter to console handler
    console_handler.setFormatter(ColoredFormatter(LOG_FORMAT, DATE_FORMAT))
    
except ImportError:
    # colorama not installed, continue without colored output
    pass

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a named logger that inherits from the main logger.
    
    Args:
        name: Name of the logger (will be prefixed with 'translation_system.')
        
    Returns:
        logging.Logger: The named logger
    """
    if name:
        return logging.getLogger(f"translation_system.{name}")
    return logger