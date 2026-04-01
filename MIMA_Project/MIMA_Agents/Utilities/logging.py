from __future__ import annotations

import logging

#Return a logging.Logger object with "name" as name
def get_logger(name:str) -> logging.Logger:
    # only show INFO and above
    logging.basicConfig(
        level=logging.INFO, 
        # timestamp, log level, logger name, your message
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    #Create a logger with the given name from parameter
    return logging.getLogger(name)