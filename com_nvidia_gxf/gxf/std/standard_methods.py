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
from gxf.core import Graph

import random
import string

try:
    from .Components import *
except:
    pass

GXF_NAME_EXTRA_LENGTH = 7


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

    connections = graph.add(Entity(_generate_name('_connection')))
    connections.add(Connection(name=None,
                              source=src,
                              target=target))
    return


def enable_job_statistics(g: Graph,
                          name: str = '__job_stats',
                          **kwargs) -> Component:
    if not g._system:
        g.add(Entity('_system'))
    return g._system.add(JobStatistics(name, **kwargs))
    # return g._system.add(Component('nvidia::gxf::JobStatistics', name, **kwargs))



def set_scheduler(g: Graph,
                  sch: Component) -> Component:
    if not g._scheduler:
        g.add(Entity('_scheduler'))
    return g._scheduler.add(sch)


def set_clock(g: Graph,
              clock: Component) -> Component:
    if not g._system:
        g.add(Entity('_system'))
    return g._system.add(clock)


def _generate_name(base_name, extra_length=GXF_NAME_EXTRA_LENGTH):
    return base_name+'_' + \
        ''.join(random.choices(string.ascii_lowercase,
                k=extra_length))
