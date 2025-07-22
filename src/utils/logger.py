from loguru import logger
import sys
"""
set up logging
"""
def set_logger():
    logger.remove()
    logger.add(sys.stdout, format="{level} {message}", level="INFO")
    logger.add("logs/app.log", rotation="1 MB", level="DEBUG")
    return logger

