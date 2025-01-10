'''
 SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from gxf.core import Entity
from gxf.core import Component
from gxf.std import CountSchedulingTerm, PeriodicSchedulingTerm
from gxf.std import DoubleBufferReceiver, DoubleBufferTransmitter
from gxf.std import DownstreamReceptiveSchedulingTerm, MessageAvailableSchedulingTerm
from gxf.python_codelet import PyCodeletV0

try:
    from .Components import *
except:
    pass


class PyComputeEntity(Entity):
    """Compute Entity adds an entity in the graph with basic scheduling terms.
    A codelet can be added to insert custom user code
    """

    def __init__(self, name: str = "", count: int = 100,
                 recess_period: int = 0, *args, **kwargs):
        super().__init__(name, True)
        self._recess_period = recess_period
        self._count = count
        self._kwargs = kwargs
        if recess_period:
            self.add(PeriodicSchedulingTerm("pst",
                                            recess_period=recess_period))
        if count >= 0:
            self.add(CountSchedulingTerm("cst",
                                         count=count))

    def add_codelet(self, name, codelet, min_message_available=1, min_message_reception=1, rx_capacity: int = 1, tx_capacity: int = 1, **params) -> Component:
        params["_txs"] = []
        params["_rxs"] = []

        for tx in codelet.txs:
            self.add(DoubleBufferTransmitter(
                    name=tx, capacity=tx_capacity))
            self.add(DownstreamReceptiveSchedulingTerm(name=f'drst_{tx}',
                                                        transmitter=getattr(
                                                            self, tx),
                                                        min_size=min_message_reception))
            if tx in params:
                raise(ValueError(f"Cannot use name {tx} for transmitter and a param together"))
            params["_txs"].append(tx)

        for rx in codelet.rxs:
            self.add(DoubleBufferReceiver(
                    name=rx, capacity=rx_capacity))
            self.add(MessageAvailableSchedulingTerm(name=f'mast_{rx}',
                                                    receiver=getattr(
                                                        self, rx),
                                                    min_size=min_message_available))
            if rx in params:
                raise(ValueError(f"Cannot use name {rx} for transmitter and a param together"))
            params["_rxs"].append(rx)
        return self.add(PyCodeletV0(name=name, codelet=codelet.__class__, **params))
