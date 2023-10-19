"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import json
import requests
from gxf.cli.transport import Transport
from result import Ok, Err, Result

class HttpTransport(Transport):
    """a simple HTTP transport implementation with requests"""
    def connect(self) -> Result:
        return Ok('Ok')

    def send_request(self, service:str, resource:str, data:str) -> Result:
        """
        send a request to a specific remote service
        Args:
            service:  name of the service
            resource: resource requested from the service in format of 'a/b/c/...'
            data:     a json formatted string to update the resource
                      None for querying the resource
        """
        url = f"http://{self._server}/{service}/{resource}"
        try:
            response = requests.post(url, json=json.loads(data)) if data else requests.get(url)
        except requests.RequestException as e:
            message = "HTTP request failed:\n"
            return Err(message + str(e))

        return Ok(response.text) if response.status_code == 200 else Err(response.status_code)