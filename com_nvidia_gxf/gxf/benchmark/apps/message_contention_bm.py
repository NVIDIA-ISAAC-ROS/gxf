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

import argparse
import sys

import gxf.std as std
from gxf.core import Graph, Entity
import gxf.benchmark as gxf_benchmark
import gxf.benchmark.benchmark_utils as benchmark_utils


class MessageContentionBm(Graph):
    """ A benchmark app for benchmarking framework with no graph-under-test. """

    def __init__(self):
        super().__init__('MessageContentionBm')

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--dummy_component_size',
            type=int,
            default=1,
            help="The number of components to be created in a dummy message",
        )
        parser.add_argument(
            '--pipeline_length',
            type=int,
            default=1,
            help="The number of codelets to form one pipeline to pass dummy messages",
        )
        parser.add_argument(
            '--parallel_pipeline_size',
            type=int,
            default=1,
            help="The number of identical pipelines to be created in a graph",
        )
        parser.add_argument(
            '--test_message_creation',
            action='store_true',
            help="If true, new messages are created in each codelet in the pipeline",
        )
        parser.add_argument(
            '--worker_thread_number',
            type=int,
            default=1,
            help="Worker thread number for the underlying scheduler",
        )
        args = parser.parse_args(sys.argv[1:])

        title = f'GXF Contention Benchmark, {args.parallel_pipeline_size} X ' \
            + f'({args.pipeline_length} hops with {args.dummy_component_size} msg comps), ' \
            + f'{args.worker_thread_number} threads'
        if args.test_message_creation:
            title += ' - Read/Write'
        else:
            title += ' - Read Only'

        # System components
        clock = std.set_clock(self, std.RealtimeClock(name='clock'))
        std.set_scheduler(
            self,
            std.EventBasedScheduler(name='scheduler',
                                    clock=clock,
                                    worker_thread_number=args.worker_thread_number))
        std.enable_job_statistics(self, clock=clock)

        benchmark_controller_subgraph = benchmark_utils.create_benchmark_controller(
            clock=clock,
            title=title,
            benchmark_duration_ms=5000,
            benchmark_iterations=5,
            entity_buffer_size=10)

        # Create parallel_pipeline_size number of benchmark buffer publishers
        for i in range(args.parallel_pipeline_size):
            # Dummy message data source
            data_source_subgraph = self.add(DummyMessageDataSource(
                f'dummy_message_data_source_{i}',
                dummy_component_size=args.dummy_component_size))
            benchmark_controller_subgraph.attach_data_source_boolean_scheduling_term(
                data_source_subgraph.get('boolean_st'))
            benchmark_controller_subgraph.attach_data_source_async_scheduling_term(
                data_source_subgraph.data_source_entity.async_st)

            # Add benchmark buffer/publisher and connect to the data source
            benchmark_buffer_publisher = benchmark_utils.add_benchmark_buffer_publisher(
                self,
                controller=benchmark_controller_subgraph,
                subgraph_name=f'benchmark_buffer_publisher_{i}')
            std.connect(src=data_source_subgraph.get('tx'),
                        target=benchmark_buffer_publisher.get('rx'),
                        graph=self)

            # Add benchmark sink
            benchmark_sink = benchmark_utils.add_benchmark_sink(
                self,
                controller=benchmark_controller_subgraph,
                benchmark_publisher=benchmark_buffer_publisher.get('publisher'),
                subgraph_name=f'benchmark_sink_{i}',
                report_namespace=f'Pipeline {i}')

            # Create a pipeline of pipeline_length number of codelets and connect to benchmark
            # publisher and sink
            message_forwarder_pipeline_subgraph = self.add(DummyMessageForwarderPipeline(
                f'dummy_message_forwarder_pipeline_{i}',
                length=args.pipeline_length,
                dummy_component_size=args.dummy_component_size,
                publish_new_message=args.test_message_creation))
            std.connect(src=benchmark_buffer_publisher.get('tx'),
                        target=message_forwarder_pipeline_subgraph.get('rx'),
                        graph=self)
            std.connect(src=message_forwarder_pipeline_subgraph.get('tx'),
                        target=benchmark_sink.get('rx'),
                        graph=self)

        self.add(benchmark_controller_subgraph)


class DummyMessageDataSource(Graph):
    """
    Dummy message data source subgraph
    """

    def __init__(self, name, dummy_component_size):
        super().__init__(name)
        data_source_entity = self.add(Entity('data_source_entity'))
        data_source_entity.add(std.DoubleBufferTransmitter(
            'transmitter', capacity=15))
        data_source_entity.add(gxf_benchmark.DummyMessageGenerator(
            'dummy_message_generator',
            dummy_component_size=dummy_component_size,
            transmitter=data_source_entity.transmitter))
        data_source_entity.add(std.BooleanSchedulingTerm(
            'boolean_st', enable_tick=True))
        data_source_entity.add(std.AsynchronousSchedulingTerm('async_st'))
        data_source_entity.add(std.PeriodicSchedulingTerm('periodic_st', recess_period='10Hz'))

        # Interface
        self.make_visible('tx', data_source_entity.transmitter)
        self.make_visible('boolean_st', data_source_entity.boolean_st)
        self.make_visible('async_st', data_source_entity.async_st)


class DummyMessageForwarder(Graph):
    """
    Dummy message forwarder subgraph
    """

    def __init__(self, name, dummy_component_size, publish_new_message):
        super().__init__(name)
        forwarder_entity = self.add(Entity('dummy_message_forwarder_entity'))

        forwarder_entity.add(std.DoubleBufferTransmitter('transmitter', capacity=15))
        forwarder_entity.add(std.DoubleBufferReceiver('receiver', capacity=2))

        forwarder_entity.add(gxf_benchmark.DummyMessageGenerator(
            'dummy_message_generator_forwarder',
            receiver=forwarder_entity.receiver,
            transmitter=forwarder_entity.transmitter,
            dummy_component_size=dummy_component_size,
            publish_new_message=publish_new_message))

        # Scheduling terms
        forwarder_entity.add(std.MessageAvailableSchedulingTerm(
            name='message_available_st',
            receiver=forwarder_entity.receiver,
            min_size=1))
        forwarder_entity.add(std.DownstreamReceptiveSchedulingTerm(
            name='downstream_receptive_st',
            transmitter=forwarder_entity.transmitter,
            min_size=1))

        # Interface
        self.make_visible('rx', forwarder_entity.receiver)
        self.make_visible('tx', forwarder_entity.transmitter)


class DummyMessageForwarderPipeline(Graph):
    """
    Dummy message forwarder pipeline subgraph
    """

    def __init__(self, name, length, dummy_component_size, publish_new_message):
        super().__init__(name)
        forwarder_subgraphs = []
        for i in range(length):
            forwarder_subgraph = self.add(
                DummyMessageForwarder(
                    f'dummy_message_forwarder_{i}',
                    dummy_component_size=dummy_component_size,
                    publish_new_message=publish_new_message))
            if i > 0:
                std.connect(src=forwarder_subgraphs[-1].get('tx'),
                            target=forwarder_subgraph.get('rx'),
                            graph=self)

            forwarder_subgraphs.append(forwarder_subgraph)

        # Interface
        self.make_visible('rx', forwarder_subgraphs[0].get('rx'))
        self.make_visible('tx', forwarder_subgraphs[-1].get('tx'))


if __name__ == '__main__':
    g = MessageContentionBm()
    g.load_extensions()
    g.run()
    g.destroy()
