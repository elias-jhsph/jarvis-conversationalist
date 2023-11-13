import logging
import os
import sys
from logging.handlers import RotatingFileHandler

level_path = "logs/level.txt"
logger_path = "logs/jarvis_process.log"
if getattr(sys, 'frozen', False):
    logger_path = os.path.join(sys._MEIPASS, "logs/jarvis_process.log")

# Set up a logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s',
                    handlers=[logging.FileHandler(logger_path), logging.StreamHandler()])

handler = RotatingFileHandler(logger_path, maxBytes=1024*10, backupCount=3)

logging.getLogger().addHandler(handler)

if os.path.exists(level_path):
    with open(level_path, 'r') as f:
        logging.getLogger().setLevel(f.read())


def get_logger():
    """
    Sets up the jarvis_process.log

    :return: The logger.
    :rtype: logging.Logger
    """
    return logging.getLogger(__name__)


def change_logging_level(level):
    """
    Change the logging level of the logger.

    :return: None
    :rtype: None
    """
    logging.getLogger().setLevel(level)
    with open(level_path, 'w') as f:
        f.write(level)
