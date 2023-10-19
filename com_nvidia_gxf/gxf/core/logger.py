# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" Registry Logger
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
import tempfile
from typing import Optional

CORE_PACKAGE_NAME: str = 'Core'


class StdOutFormatter(logging.Formatter):

    grey = "\x1b[38;5;248m"
    white = "\x1b[38;97m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_ = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_ + reset,
        logging.INFO: white + format_ + reset,
        logging.WARNING: yellow + format_ + reset,
        logging.ERROR: red + format_ + reset,
        logging.CRITICAL: bold_red + format_ + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class CoreLogging:
    """ Class CoreLogger
    """
    _inited: bool = False

    @classmethod
    def is_inited(cls) -> bool:
        return cls._inited

    @classmethod
    def init_logger(
        cls) -> None:
        logger = cls.get_logger()
        logger.setLevel(logging.DEBUG)
        cls.clear_handlers(logger)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(StdOutFormatter())
        stream_handler.setLevel(logging.DEBUG) # change back
        logger.addHandler(stream_handler)
        fh = RotatingFileHandler(
            tempfile.gettempdir() + "//nvgraph_core.log", mode='a', maxBytes=1048576, backupCount=5)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        cls._inited = True

    @classmethod
    def clear_handlers(cls, logger):
        handlers = logger.handlers.copy()
        for h in handlers:
            try:
                h.acquire()
                h.flush()
                h.close()
            except (OSError, ValueError):
                pass
            finally:
                h.release()
                logger.removeHandler(h)
        logger.handlers.clear()

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """ Get a Core component logger

        Parameters:
        name: a module name

        Returns: A Logger Instance
        """
        propagate = True
        name = CORE_PACKAGE_NAME
        if not name:
            propagate = False


        logger = logging.getLogger(name)
        logger.propagate = propagate
        return logger


get_logger = CoreLogging.get_logger
init_logger = CoreLogging.init_logger
is_inited = CoreLogging.is_inited
