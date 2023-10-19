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
from .allocator_pybind import *    # pylint: disable=no-name-in-module
from .clock_pybind import *    # pylint: disable=no-name-in-module
from .receiver_pybind import *    # pylint: disable=no-name-in-module
from .transmitter_pybind import *    # pylint: disable=no-name-in-module
from .tensor_pybind import *     # pylint: disable=no-name-in-module
from .vault_pybind import *     # pylint: disable=no-name-in-module
from .timestamp_pybind import *     # pylint: disable=no-name-in-module
from .scheduling_terms_pybind import *     # pylint: disable=no-name-in-module
from .scheduling_condition_pybind import *     # pylint: disable=no-name-in-module

from gxf.core import Entity
from gxf.core import Component
from gxf.core import Graph
import random
import string


try:
    from .Components import *
except:
    pass


class ComputeEntity(Entity):
    """Compute Entity adds an entity in the graph with basic scheduling terms.
    A codelet can be added to insert custom user code
    """

    def __init__(self, name: str = "", count: int = -1,
                 recess_period: int = 0, **kwargs):
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

    def add_codelet(self, codelet, min_message_available=1, min_message_reception=1, rx_capacity: int = 1, tx_capacity: int = 1) -> Component:
        for _, info in codelet._validation_info_parameters.items():
            if (info['handle_type'] == "nvidia::gxf::Transmitter"):
                self.add(DoubleBufferTransmitter(
                    name=info['key'], capacity=tx_capacity))
                self.add(DownstreamReceptiveSchedulingTerm(name='drst',
                                                           transmitter=getattr(
                                                               self, info['key']),
                                                           min_size=min_message_reception))
                # set the transmitter of the codelet since we are not passing the
                # params anymore
                codelet._params[info['key']] = getattr(self, info['key'])
            if (info['handle_type'] == "nvidia::gxf::Receiver"):
                self.add(DoubleBufferReceiver(
                    name=info['key'], capacity=rx_capacity))
                self.add(MessageAvailableSchedulingTerm(name='mast',
                                                        receiver=getattr(
                                                            self, info['key']),
                                                        min_size=min_message_available))
                # set the receiver of the codelet since we are not passing the
                # params anymore
                codelet._params[info['key']] = getattr(self, info['key'])
        self.add(codelet)
        return codelet


def connect(src: Component, target: Component, graph: Graph = None):
    if not src.added_to_graph():
        raise AssertionError(f"Component '{src.name}' not part of graph")
    if not target.added_to_graph():
        raise AssertionError(f"Component '{target.name}' not part of graph")

    if src.entity.graph != target.entity.graph and not graph:
        raise AssertionError("Different graph for source and target. "
                             "'graph' param required.")
    if not graph:
        graph = src.entity.graph

    generated_name = "_connection_"+''.join(random.choices(string.ascii_lowercase, k=7)) #magic number
    # TODO: if the entity exists (generate new one)
    connections = graph.add(Entity(generated_name))
    connections.add(Component('nvidia::gxf::Connection', name=None,
                              source=src,
                              target=target))
    return


# def connect(src, target, g=None):
#     if (isinstance(src, str) and isinstance(target, str)):
#         if g is None:
#             raise AssertionError("'g' parameter g cannot be None if both 'src' and 'target' are string")
#     else:
#         if isinstance(src, Component) and not src.added_to_graph():
#             raise AssertionError(f"Component '{src.name}' not part of graph")

#         if isinstance(target, Component) and not target.added_to_graph():
#             raise AssertionError(f"Component '{target.name}' not part of graph")
#         else:
#             g = target.entity.graph

#         if (isinstance(src, Component)):
#             src = ''.join([src.entity.name, '/', src.name])
#         if (isinstance(target, Component)):
#             target = ''.join([target.entity.name, '/', target.name])
#     connections = g.add(Entity())
#     connections.add(Component('nvidia::gxf::Connection', name='',
#                             source=src,
#                             target=target))
#     return

def enable_job_statistics(g: Graph,
                       name: str = '__job_stats',
                       **kwargs) -> Component:
    if not g._system:
        g.add(Entity('_system'))
    return g._system.add(JobStatistics(name, **kwargs))

def set_scheduler(g: Graph,
                  sch: Component) -> Component:
    if not g._scheduler:
        g.add(Entity('_scheduler'))
    return g._scheduler.add(sch)

# add Entity Monitor to the test extension
# def add_monitor(g: Graph, name: str = "__entity_monitor", **kwargs):
#     if not g._system:
#         g.add(Entity('_system'))
#     system_entity = g._system
#     return system_entity.add(EntityMonitor(name, **kwargs))

def set_clock(g: Graph,
              clock: Component) -> Component:
    if not g._system:
        g.add(Entity('_system'))
    return g._system.add(clock)
