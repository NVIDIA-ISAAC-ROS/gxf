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
from gxf.core import Graph, Entity, Component
from .compute_entity import ComputeEntity
from .standard_methods import set_clock, set_scheduler, _generate_name, connect

import random
import string
try:
    from .Components import *
except:
    pass


class StandardGraph(Graph):
    """A simple graph wrapper. On `add_codelet` an entity will get
    created automatically and the codelet will be added to it.
    All the transmitters and receivers will also be added depending on the
    parameters of the codelet.
    Defaults:
        clock: ManualClock
        scheduler: GreedyScheduler(max_duration_ms=1000000, clock=clock)
    """

    def __init__(self, name: str = '', clock: Component = None, scheduler: Component = None):
        super().__init__(name)
        if not clock:
            clock = ManualClock(name='clock')
        self._clock_component = set_clock(clock=clock, g=self)
        if not scheduler:
            scheduler = GreedyScheduler(max_duration_ms=1000000, clock=self._clock_component)
        self._scheduler_component = set_scheduler(g=self, sch=scheduler)

    def get_clock(self):
        return self._clock_component

    def get_scheduler(self):
        return self._scheduler_component

    def add_codelet(self, codelet: Component, count: int = -1,
                    recess_period: int = 0, min_message_available=1,
                    min_message_reception=1, rx_capacity: int = 1,
                    tx_capacity: int = 1, **kwargs) -> Component:
        e = self.add(ComputeEntity(name=_generate_name("_entity"), count=count,
                     recess_period=recess_period, kwargs=kwargs))
        return e.add_codelet(codelet, min_message_available,
                             min_message_reception, rx_capacity, tx_capacity)

    def connect(self, src: Component, target: Component):
        connect(src=src, target=target, graph=self)