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

REGISTRY_PACKAGE_NAME: str = 'Registry'


class RegistryLogging:
    """ Class RegistryLogger
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
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        fh = RotatingFileHandler(
            tempfile.gettempdir() + "//nvgraph_registry.log", mode='a', maxBytes=1048576, backupCount=5)
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
        """ Get a Registry component logger

        Parameters:
        name: a module name

        Returns: A Logger Instance
        """
        propagate = True
        name = REGISTRY_PACKAGE_NAME
        if not name:
            propagate = False


        logger = logging.getLogger(name)
        logger.propagate = propagate
        return logger


get_logger = RegistryLogging.get_logger
init_logger = RegistryLogging.init_logger
is_inited = RegistryLogging.is_inited
