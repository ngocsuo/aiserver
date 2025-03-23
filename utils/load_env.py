"""
Utility module to load environment variables
"""
import os
import logging
from dotenv import load_dotenv
import config

logger = logging.getLogger("env_loader")

def load_environment_variables():
    """
    Load environment variables from .env file and set them in config
    
    Returns:
        bool: True if environment variables were loaded successfully
    """
    try:
        # Try to load from .env file if it exists
        env_loaded = load_dotenv(override=True)
        
        if env_loaded:
            logger.info("Loaded environment variables from .env file")
        else:
            logger.info("No .env file found, using existing environment variables")
        
        # Get API keys from environment or use existing config values
        api_key = os.environ.get('BINANCE_API_KEY')
        api_secret = os.environ.get('BINANCE_API_SECRET')
        
        # Update config if environment variables are available
        if api_key:
            config.BINANCE_API_KEY = api_key
            logger.info("Updated BINANCE_API_KEY from environment")
            
        if api_secret:
            config.BINANCE_API_SECRET = api_secret
            logger.info("Updated BINANCE_API_SECRET from environment")
            
        # Log the status (without showing the actual keys for security)
        if config.BINANCE_API_KEY:
            logger.info(f"BINANCE_API_KEY is set: ***{config.BINANCE_API_KEY[-4:] if len(config.BINANCE_API_KEY) > 4 else ''}")
        else:
            logger.warning("BINANCE_API_KEY is not set")
            
        if config.BINANCE_API_SECRET:
            logger.info(f"BINANCE_API_SECRET is set: ***{config.BINANCE_API_SECRET[-4:] if len(config.BINANCE_API_SECRET) > 4 else ''}")
        else:
            logger.warning("BINANCE_API_SECRET is not set")
            
        return True
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        return False