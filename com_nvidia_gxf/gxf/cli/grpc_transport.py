"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

from result import Result, Ok, Err
import json
import grpc
import gxf.cli.grpc_service_pb2
import gxf.cli.grpc_service_pb2_grpc
from gxf.cli.transport import Transport

class GrpcTransport(Transport):
    """a simple GRPC transport implementation"""
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
        params = [resource]
        if data:
            params.append(data)
        with grpc.insecure_channel(self._server) as channel:
            stub = gxf.cli.grpc_service_pb2_grpc.ServiceHubStub(channel)
            try:
                response = stub.SendRequest(
                    gxf.cli.grpc_service_pb2.Request(service=service, params=params),
                    timeout=1
                )
            except grpc.RpcError as e:
                return Err(f"RPC error: code={e.code()}, {e.details()}")
            return Ok(response.result)
