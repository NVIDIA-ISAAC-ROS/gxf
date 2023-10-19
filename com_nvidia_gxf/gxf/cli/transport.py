"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

from abc import ABC, abstractmethod
from typing import List
from result import Result

class Transport(ABC):
    """base class of data transportation between processes"""
    def __init__(self, server: str):
        self._server = server

    @abstractmethod
    def connect(self) -> Result:
        pass

    @abstractmethod
    def send_request(self, service:str, resource:str, data:str) -> Result:
        pass