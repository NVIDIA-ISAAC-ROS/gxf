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
from gxf.core import Entity, Graph
import gxf.serialization as serialization
import gxf.std as std


class BenchmarkController(Graph):
    """
    Benchmark controller subgraph
    """

    def __init__(self,
                 name,
                 clock,
                 title,
                 benchmark_publishers=None,
                 benchmark_sinks=None,
                 dependent_components=None,
                 data_source_async_scheduling_terms=None,
                 data_source_boolean_scheduling_terms=None,
                 data_replay_control_receiver=None,
                 report_file_path='/tmp/benchmark_report.json',
                 benchmark_duration_ms=5000,
                 benchmark_iterations=5,
                 entity_buffer_size=10):
        super().__init__(name)
        # Initialize instance variables
        self.benchmark_publishers = benchmark_publishers if benchmark_publishers else []
        self.benchmark_sinks = benchmark_sinks if benchmark_sinks else []
        self.dependent_components = dependent_components if dependent_components else []
        self.data_source_async_scheduling_terms = data_source_async_scheduling_terms \
            if data_source_async_scheduling_terms else []
        self.data_source_boolean_scheduling_terms = data_source_boolean_scheduling_terms \
            if data_source_boolean_scheduling_terms else []

        controller_entity = self.add(Entity('controller_entity'))

        # Benchmark report
        controller_entity.add(std.UnboundedAllocator("allocator"))
        controller_entity.add(serialization.File(
            'report',
            allocator=controller_entity.allocator,
            file_path=report_file_path,
            file_mode='wb'))

        # Scheduling terms
        controller_entity.add(std.BooleanSchedulingTerm('boolean_st', enable_tick=True))
        controller_entity.add(std.TargetTimeSchedulingTerm('target_time_st', clock=clock))

        # Optional data replay controller
        data_replay_control_transmitter = None
        if data_replay_control_receiver is not None:
            data_replay_control_transmitter = controller_entity.add(std.DoubleBufferTransmitter(
                'data_replay_control_transmitter',
                capacity=1))
            controller_entity.add(std.DownstreamReceptiveSchedulingTerm(
                'data_replay_control_downstream_st',
                transmitter=controller_entity.data_replay_control_transmitter,
                min_size=1))
            std.connect(src=controller_entity.data_replay_control_transmitter,
                        target=data_replay_control_receiver,
                        graph=self)

        # Benchmark controller
        self.benchmark_controller = controller_entity.add(gxf_benchmark.BenchmarkController(
            'controller',
            title=title,
            dependent_components=self.dependent_components,
            benchmark_controller_target_time_scheduling_term=controller_entity.target_time_st,
            benchmark_controller_boolean_scheduling_term=controller_entity.boolean_st,
            data_source_async_scheduling_terms=self.data_source_async_scheduling_terms,
            data_source_boolean_scheduling_terms=self.data_source_boolean_scheduling_terms,
            benchmark_sinks=self.benchmark_sinks,
            benchmark_publishers=self.benchmark_publishers,
            exported_report=controller_entity.report,
            benchmark_duration_ms=benchmark_duration_ms,
            benchmark_iterations=benchmark_iterations,
            entity_buffer_size=entity_buffer_size))

        if data_replay_control_transmitter:
            self.benchmark_controller.set_param('data_replay_control_transmitter', data_replay_control_transmitter)

        # Interface
        self.make_visible('report', controller_entity.report)

    def attach_dependent_component(self, dependent_component):
        """Add a component to the dependent list"""
        if dependent_component in self.dependent_components:
            return
        self.dependent_components.append(dependent_component)
        self.benchmark_controller.set_param(
            'dependent_components',
            self.dependent_components)

    def attach_data_source_async_scheduling_term(self, async_st):
        """Associate an async scheduling term with the benchmark controller"""
        if async_st in self.data_source_async_scheduling_terms:
            return
        self.data_source_async_scheduling_terms.append(async_st)
        self.benchmark_controller.set_param(
            'data_source_async_scheduling_terms',
            self.data_source_async_scheduling_terms)

    def attach_data_source_boolean_scheduling_term(self, boolean_st):
        """Associate a boolean scheduling term with the benchmark controller"""
        if boolean_st in self.data_source_boolean_scheduling_terms:
            return
        self.data_source_boolean_scheduling_terms.append(boolean_st)
        self.benchmark_controller.set_param(
            'data_source_boolean_scheduling_terms',
            self.data_source_boolean_scheduling_terms)

    def attach_benchmark_sink(self, sink):
        """Associate a benchmark sink with the benchmark controller"""
        if sink in self.benchmark_sinks:
            return
        self.benchmark_sinks.append(sink)
        self.benchmark_controller.set_param(
            'benchmark_sinks',
            self.benchmark_sinks)

    def attach_benchmark_publisher(self, publisher):
        """Associate a benchmark publisher with the benchmark controller"""
        if publisher in self.benchmark_publishers:
            return
        self.benchmark_publishers.append(publisher)
        self.benchmark_controller.set_param(
            'benchmark_publishers',
            self.benchmark_publishers)
