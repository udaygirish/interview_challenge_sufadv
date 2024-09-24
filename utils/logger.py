import logging 
import sys 
from datetime import datetime as dt 
import os 

# Prevent Python from genrating a .pyc file
sys.dont_write_bytecode = True


def setup_logger(log_file="app.log", log_level = logging.DEBUG):
    def setup_logger(log_file="app.log", log_level=logging.DEBUG):
        """
        Sets up a logger with both file and console handlers.
        Args:
            log_file (str): The name of the log file. Default is "app.log".
            log_level (int): The logging level. Default is logging.DEBUG.
        Returns:
            logging.Logger: Configured logger instance.
        The logger will log messages to a file in the "logs" directory with a timestamped filename
        and also output to the console. The log format includes the timestamp, logger name, log level,
        and the log message.
        """
    # Create or Get Logger
    log_file = "logs/" + log_file.split(".")[0] + "_" + dt.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log"
    # Create logs folder if it does not exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    logger = logging.getLogger("body_transformer_logger")

    # Avoid adding the same handler multiple times
    if not logger.hasHandlers():
        # Set Logger Level
        logger.setLevel(log_level)

        # Create File Handler and set the log level
        filehandler = logging.FileHandler(log_file)
        filehandler.setLevel(log_level)

        # Create a console handler and set the log level
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Create a formatter and set it for the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Add formatter to the handlers 
        filehandler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(filehandler)
        logger.addHandler(console_handler)
    
    return logger

# Get the Logger
logger = setup_logger()


# Call the Logger
if __name__ == "__main__":
    # Example Usage
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")