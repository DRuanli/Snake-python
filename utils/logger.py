"""
Logging utilities for Snake RL.
"""
import os
import logging
import time
from datetime import datetime

class Logger:
    """
    Logger for training and evaluation.
    Handles console and file logging.
    """
    
    def __init__(self, log_dir='logs', console_level=logging.INFO, file_level=logging.DEBUG):
        """
        Initialize the logger.
        
        Args:
            log_dir (str): Directory to save log files
            console_level (int): Logging level for console output
            file_level (int): Logging level for file output
        """
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create logger
        self.logger = logging.getLogger('snake_rl')
        self.logger.setLevel(logging.DEBUG)
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'snake_rl_{timestamp}.log')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        
        # Create formatters
        console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', 
                                             datefmt='%H:%M:%S')
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        
        # Set formatters
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Log initialization
        self.info(f"Logger initialized | Log file: {log_file}")
    
    def debug(self, message):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message."""
        self.logger.critical(message)


class Timer:
    """
    Timer utility for measuring performance.
    """
    
    def __init__(self, name=''):
        """
        Initialize timer.
        
        Args:
            name (str): Name for the timer
        """
        self.name = name
        self.start_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self):
        """
        Stop the timer and return elapsed time.
        
        Returns:
            float: Elapsed time in seconds
        """
        if self.start_time is None:
            raise ValueError("Timer must be started before it can be stopped")
        
        elapsed_time = time.time() - self.start_time
        self.start_time = None
        return elapsed_time
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        elapsed_time = self.stop()
        print(f"{self.name} took {elapsed_time:.4f} seconds")