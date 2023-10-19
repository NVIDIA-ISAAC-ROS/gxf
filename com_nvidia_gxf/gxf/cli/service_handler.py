"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import json
from abc import ABC, abstractmethod
from typing import List
from result import Ok, Err, Result
from gxf.cli.transport import Transport

class ServiceHandler(ABC):
    """base handler class to communicate with the server"""
    def __init__(self, transport: Transport):
        self._transport = transport
        self._connected = False

    @abstractmethod
    def request(self, params: List[str]) -> Result:
        pass

class StatServiceHandler(ServiceHandler):
    NAME = "stat"
    def request(self, params: List[str]) -> Result:
        """send a request with given parameters"""
        if not self._connected:
            res = self._transport.connect()
            if res.is_err():
                return res
        if not params:
            return Err("Resource being requested are missing for 'stat' service")

        resource = params[0]
        if len(params) == 2:
            resource += f"/{params[1]}"
        res = self._transport.send_request(self.NAME, resource, None)
        if res.is_err():
            return res
        data = json.loads(res.value)
        return Ok(data)

class ConfigServiceHandler(ServiceHandler):
    NAME = "config"
    def request(self, params) -> Result:
        if not self._connected:
            res = self._transport.connect()
            if res.is_err():
                return res
        return self._transport.send_request(
            service=self.NAME,
            resource=f"{params[0]}/{params[1]}",
            data=params[2]
        )

class DumpServiceHandler(ServiceHandler):
    NAME = "dump"
    def request(self, params) -> Result:
        if not self._connected:
            res = self._transport.connect()
            if res.is_err():
                return res
        return self._transport.send_request(
            service=self.NAME,
            resource=params[0],
            data=None
        )