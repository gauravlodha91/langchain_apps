import os
import logging
from datetime import datetime

def setup_logging():
    """Configure application logging."""
    log_filename = f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )
    logger = logging.getLogger()
    
    # Add console handler to see logs in terminal
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    logger.info("Logging initialized")
    return logger

def load_api_keys():
    """Load API keys from environment variables."""
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        logging.error("COHERE_API_KEY not found in environment variables")
        raise ValueError("COHERE_API_KEY is required. Please set it in your .env file.")
    
    return {"cohere_api_key": cohere_api_key}