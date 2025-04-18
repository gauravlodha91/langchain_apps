import os
import logging
from datetime import datetime

def setup_logging(log_level=None):
    """
    Configure application logging with customizable log level.
    
    Args:
        log_level: Optional log level from environment variable
    
    Returns:
        Logger instance
    """
    # Get log level from environment or use INFO as default
    if not log_level:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    level = getattr(logging, log_level, logging.INFO)
    
    # Create log directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log filename with timestamp
    log_filename = os.path.join(log_dir, f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure basic logging
    logging.basicConfig(
        filename=log_filename,
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create logger
    logger = logging.getLogger()
    
    # Add console handler to see logs in terminal
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    logger.info(f"Logging initialized at level {log_level}")
    return logger