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
from gxf.std import ComputeEntity
from gxf.core import Graph


class BenchmarkSink(Graph):
    """
    Benchmark sink subgraph
    """

    def __init__(self,
                 name,
                 benchmark_publisher=None,
                 use_received_acqtime_as_published=None,
                 report_namespace=None):
        super().__init__(name)
        sink_entity = self.add(ComputeEntity('sink_entity'))

        # Basic performance calculator
        basic_metrics_comp = sink_entity.add(gxf_benchmark.BasicMetricsCalculator('basic_metrics'))
        if report_namespace:
            basic_metrics_comp.set_param('namespace', report_namespace)

        # Benchmark sink
        sink_comp = sink_entity.add_codelet(gxf_benchmark.BenchmarkSink(
            name = 'sink',
            performance_calculators=[sink_entity.basic_metrics]), rx_capacity= 15)
        if benchmark_publisher:
            sink_comp.set_param(
                'benchmark_publisher', benchmark_publisher)
        if use_received_acqtime_as_published:
            sink_comp.set_param(
                'use_received_acqtime_as_published', use_received_acqtime_as_published)

        # Interface
        self.make_visible('rx', sink_entity.receiver)
        self.make_visible('sink', sink_entity.sink)
