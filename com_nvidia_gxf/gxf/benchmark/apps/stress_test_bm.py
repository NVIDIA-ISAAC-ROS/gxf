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

from gxf.core import Graph
from gxf.core import logger
import logging
from gxf.std import ComputeEntity
import gxf.std as std
import gxf.sample as sample
import gxf.benchmark.benchmark_utils as benchmark_utils


class StressTestBm(Graph):
    """ A benchmark app for benchmarking framework with no graph-under-test. """

    def __init__(self):
        super().__init__('StressTestBm')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--scheduler",
            type=str,
            default='MultiThreadScheduler',
            help='Select a scheduler to run this benchmark from either ' \
                 'MultiThreadScheduler, GreedyScheduler or EventBasedScheduler',
        )
        parser.add_argument(
            '--num_parallel_codelets',
            type=int,
            default=1,
            help="The max number of codelets ticking in parallel",
        )
        parser.add_argument(
            '--worker_thread_number',
            type=int,
            default=1,
            help="Worker thread number for the underlying scheduler",
        )
        args = parser.parse_args(sys.argv[1:])

        title_format_template = 'Blank (No Graph-under-Test) {} Benchmark'

        if args.scheduler not in ['MultiThreadScheduler', 'GreedyScheduler', 'EventBasedScheduler']:
            raise ValueError(f'Could not recognize the specified scheduler: "{args.scheduler}"')
        print(f'Specified to use {args.scheduler}')
        title = title_format_template.format(args.scheduler)

        # System components
        clock = std.set_clock(self, std.RealtimeClock(name='clock'))
        if args.scheduler == 'MultiThreadScheduler':
            std.set_scheduler(
                self,
                std.MultiThreadScheduler(name='scheduler',
                                         clock=clock,
                                         worker_thread_number=args.worker_thread_number,
                                         check_recession_period_ms=0))
        elif args.scheduler == 'EventBasedScheduler':
            std.set_scheduler(
                self,
                std.EventBasedScheduler(name='scheduler', clock=clock,
                 worker_thread_number=args.worker_thread_number
                ))
        else:
            std.set_scheduler(self, std.GreedyScheduler(name='scheduler', clock=clock))
       #std.enable_job_statistics(self, clock=clock)

        benchmark_controller_subgraph = benchmark_utils.create_benchmark_controller(
            clock=clock,
            title=title,
            benchmark_duration_ms=60000,
            benchmark_iterations=1,
            entity_buffer_size=1000)
        benchmark_controller_subgraph.benchmark_controller.set_param('benchmark_buffering_timeout_s',100)
        benchmark_controller_subgraph.benchmark_controller.set_param('trial_run',False)
        # Dummy data source
        data_source_subgraph = self.add(DummyDataSource("dummy_data_source", clock=clock))
        benchmark_controller_subgraph.attach_data_source_boolean_scheduling_term(
            data_source_subgraph.get('boolean_st'))
        benchmark_controller_subgraph.attach_data_source_async_scheduling_term(
            data_source_subgraph.get('async_st'))

        for i in range(args.num_parallel_codelets):

            # Benchmark buffer and publisher
            # Add benchmark buffer/publisher and connect to the data source
            benchmark_buffer_publisher = benchmark_utils.add_benchmark_buffer_publisher(
                self,
                controller=benchmark_controller_subgraph,
                subgraph_name=f'benchmark_buffer_publisher_{i}')
            std.connect(src=data_source_subgraph.get('tx'),
                        target=benchmark_buffer_publisher.get('rx'),
                        graph=self)

        # Benchmark sink
            benchmark_sink = benchmark_utils.add_benchmark_sink(
                self,
                controller=benchmark_controller_subgraph,
                benchmark_publisher=benchmark_buffer_publisher.get('publisher'),
                subgraph_name=f'benchmark_sink_{i}',
                report_namespace=f'Pipeline {i}')

            std.connect(src=benchmark_buffer_publisher.get('tx'),
                target=benchmark_sink.get('rx'),
                graph=self)

        self.add(benchmark_controller_subgraph)

        # Benchmark controller



class DummyDataSource(Graph):
    """
    Dummy data source subgraph
    """

    def __init__(self, name, clock):
        super().__init__(name)
        data_source_entity = self.add(ComputeEntity('data_source', recess_period = '1000Hz'))
        data_source_entity.add_codelet(sample.PingTx(
            name = 'ping',
            clock=clock), tx_capacity = 15)
        data_source_entity.add(std.BooleanSchedulingTerm('boolean_st', enable_tick=True))
        data_source_entity.add(std.AsynchronousSchedulingTerm('async_st'))

        # Interface
        self.make_visible('tx', data_source_entity.signal)
        self.make_visible('boolean_st', data_source_entity.boolean_st)
        self.make_visible('async_st', data_source_entity.async_st)


if __name__ == '__main__':
    g = StressTestBm()
    g.set_severity(logging.WARNING)
    g.load_extensions()
    g.run()
    g.destroy()
