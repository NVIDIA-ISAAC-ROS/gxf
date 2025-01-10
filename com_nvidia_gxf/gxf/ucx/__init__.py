'''
 SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

from gxf.serialization import Endpoint
from gxf.multimedia import VideoBuffer

try:
    from .Components import *
except:
    pass
from gxf.core import Entity
from gxf.std import Forward
from gxf.std import DoubleBufferTransmitter, DoubleBufferReceiver
from gxf.std import DownstreamReceptiveSchedulingTerm, CountSchedulingTerm
from gxf.std import UnboundedAllocator
from gxf.std import MessageAvailableSchedulingTerm

class UCXSource(Entity):
    """UCXSource Entity containing all the requied components to receive data on ucx address:port"""

    def __init__(self, name, address, port=13337, count=-1, capacity=1, min_message_reception=1,
                 allocator_type=None, **kwargs):
        super().__init__(name, True)
        self._address = address
        self._port = port
        self._count = count
        self._capacity = capacity
        self._min_message_reception = min_message_reception
        self._allocator_type = allocator_type
        self._kwargs = kwargs
        self.add(UnboundedAllocator(name="allocator"))
        self.add(UcxSerializationBuffer(
            name="serialization_buffer", allocator=self.allocator))
        self.add(UcxReceiver(name="input", port=self._port, address=self._address,
                 buffer=self.serialization_buffer))
        self.add(MessageAvailableSchedulingTerm(name='mast', receiver=self.input,
                                                min_size=min_message_reception))
        self.add(DoubleBufferTransmitter(name="output", capacity=capacity))
        # 'in' is a keyword in python. can't access as an attribute
        self.add(Forward(name="forward"))
        self.forward._params["in"] = self.input
        self.forward._params["out"] = self.output
        self.add(DownstreamReceptiveSchedulingTerm(name='drst', transmitter=self.output,
                                                   min_size=min_message_reception))
        if count >= 0:
            self.add(CountSchedulingTerm(name="cst", count=self.count))


class UCXSink(Entity):
    """UCXSink Entity containing all the required components to push data on a ucx address:port"""

    def __init__(self, name, address, port=13337, count=-1, capacity=1, min_message_available=1,
                 allocator_type=None, **kwargs):
        super().__init__(name, True)
        self._address = address
        self._port = port
        self._count = count
        self._capacity = capacity
        self._min_message_available = min_message_available
        self._allocator_type = allocator_type
        self._kwargs = kwargs
        self.add(UnboundedAllocator(name="allocator"))
        self.add(UcxSerializationBuffer(
            name="serialization_buffer", allocator=self.allocator))
        self.add(UcxTransmitter(name="output", port=self._port,
                 buffer=self.serialization_buffer, receiver_address=self._address))
        self.add(DoubleBufferReceiver(name="input", capacity=capacity))
        # in is a keyword in python. can't access as an attribute
        self.add(Forward(name="forward"))
        self.forward._params["in"] = self.input
        self.forward._params["out"] = self.output
        self.add(MessageAvailableSchedulingTerm(name='mast', receiver=self.input,
                                                min_size=min_message_available))
        if count >= 0:
            self.add(CountSchedulingTerm(name="cst", count=self._count))


class UCX(Entity):
    """UCX Entity requied to add UCXSource and UCXSink"""

    def __init__(self, name, allocator=None, reconnect=True, cpu_data_only=False):
        super().__init__(name, True)
        if not allocator:
            allocator = self.add(UnboundedAllocator(name="allocator"))
        self.add(UcxComponentSerializer(
            name="component_serializer", allocator=allocator))
        self.add(UcxEntitySerializer(name="entity_serializer",
                 component_serializers=[self.component_serializer]))
        self.add(UcxContext(name="ucx_context",
                            serializer=self.entity_serializer,
                            reconnect=reconnect,
                            cpu_data_only=cpu_data_only,
                            enable_async=True))