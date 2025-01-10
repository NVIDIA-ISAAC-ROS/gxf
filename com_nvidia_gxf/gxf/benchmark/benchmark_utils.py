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

from gxf.core import Graph
import gxf.std as std

from gxf.benchmark.benchmark_controller import BenchmarkController
from gxf.benchmark.benchmark_buffer import BenchmarkBuffer
from gxf.benchmark.benchmark_buffer_publisher import BenchmarkBufferPublisher
from gxf.benchmark.benchmark_publisher import BenchmarkPublisher
from gxf.benchmark.benchmark_sink import BenchmarkSink


def create_benchmark_controller(clock: std.Clock, **kwargs):
    return BenchmarkController('benchmark_controller',
                               clock=clock,
                               **kwargs)

def add_benchmark_sink(graph: Graph,
                       controller: BenchmarkController,
                       benchmark_publisher=None,
                       use_received_acqtime_as_published=None,
                       report_namespace=None,
                       subgraph_name='sink_subgraph'):
    sink_subgraph = graph.add(BenchmarkSink(
        subgraph_name,
        benchmark_publisher=benchmark_publisher,
        use_received_acqtime_as_published=use_received_acqtime_as_published,
        report_namespace=report_namespace))
    controller.attach_benchmark_sink(sink_subgraph.get('sink'))
    return sink_subgraph


def add_benchmark_buffer_publisher(graph: Graph,
                                   controller: BenchmarkController,
                                   subgraph_name='benchmark_buffer_publisher'):
    buffer_subgraph = graph.add(BenchmarkBufferPublisher(subgraph_name))
    controller.attach_benchmark_publisher(buffer_subgraph.get('publisher'))
    return buffer_subgraph


def add_benchmark_buffer(graph: Graph, subgraph_name='benchmark_buffer'):
    buffer_subgraph = graph.add(BenchmarkBuffer(subgraph_name))
    return buffer_subgraph


def add_benchmark_publisher(graph: Graph,
                            controller: BenchmarkController,
                            entity_buffer,
                            subgraph_name='benchmark_publisher'):
    publisher_subgraph = graph.add(BenchmarkPublisher(subgraph_name,
                                                      entity_buffer=entity_buffer))
    controller.attach_benchmark_publisher(publisher_subgraph.publisher)
    return publisher_subgraph
