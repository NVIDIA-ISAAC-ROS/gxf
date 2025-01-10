'''
 SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gxf.benchmark as gxf_benchmark
import gxf.std as std
from gxf.core import Entity, Graph


class BenchmarkPublisher(Graph):
    """
    Benchmark publisher subgraph
    """

    def __init__(self, name, entity_buffer):
        super().__init__(name)
        # Benchmark publisher
        publisher_entity = self.add(Entity('publisher_entity'))

        # Benchmark publisher async scheduling term
        publisher_entity.add(std.AsynchronousSchedulingTerm('async_st'))

        # Benchmark publisher components
        publisher_entity.add(std.DoubleBufferTransmitter('transmitter', capacity=1))
        publisher_entity.add(gxf_benchmark.BenchmarkPublisher(
            'publisher',
            transmitter=publisher_entity.transmitter,
            entity_buffer=entity_buffer,
            benchmark_publisher_async_scheduling_term=publisher_entity.async_st))

        # Benchmark publisher scheduling terms
        publisher_entity.add(std.DownstreamReceptiveSchedulingTerm(
            name='downstream_receptive_st',
            transmitter=publisher_entity.transmitter,
            min_size=1))

        # Interface
        self.make_visible('tx', publisher_entity.transmitter)
        self.make_visible('async_st', publisher_entity.async_st)
        self.make_visible('publisher', publisher_entity.publisher)
