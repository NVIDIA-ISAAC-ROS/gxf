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

from gxf.core import Component
from gxf.core import add_to_manifest

add_to_manifest("gxf/serialization/libgxf_serialization.so")
add_to_manifest("gxf/benchmark/libgxf_benchmark.so")


class BasicMetricsCalculator(Component):
    '''Calculates performance outcomes for basic metrics
    '''
    gxf_native_type: str = "nvidia::gxf::benchmark::BasicMetricsCalculator"

    _validation_info_parameters = {
        'namespace': {
            'key': 'namespace',
            'headline': 'Performance calculator namespace',
            'description': 'Namespace used to return performance results',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': ''
        }
    }

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class BenchmarkController(Component):
    '''Controls benchmark flow
    '''
    gxf_native_type: str = "nvidia::gxf::benchmark::BenchmarkController"

    _validation_info_parameters = {
        'data_source_boolean_scheduling_terms': {
            'key': 'data_source_boolean_scheduling_terms',
            'headline': 'Data source boolean scheduling terms',
            'description': 'Boolean scheduling terms for perminately stopping data sources',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 1,
            'shape': [-1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::BooleanSchedulingTerm',
            'default': 'N/A'
        },
        'data_source_async_scheduling_terms': {
            'key': 'data_source_async_scheduling_terms',
            'headline': 'Data source async scheduling terms',
            'description': 'Scheduling terms to control execution of data sourced',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 1,
            'shape': [-1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::AsynchronousSchedulingTerm',
            'default': 'N/A'
        },
        'benchmark_controller_boolean_scheduling_term': {
            'key': 'benchmark_controller_boolean_scheduling_term',
            'headline': 'Benchmark controller boolean scheduling term',
            'description': 'A boolean scheduling term for perminately stopping this controller',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::BooleanSchedulingTerm',
            'default': 'N/A'
        },
        'benchmark_controller_target_time_scheduling_term': {
            'key': 'benchmark_controller_target_time_scheduling_term',
            'headline': 'Benchmark controller target time scheduling term',
            'description':
            'A target time scheduling term for enforcing timeout during benchmarking',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::TargetTimeSchedulingTerm',
            'default': 'N/A'
        },
        'graph_boolean_scheduling_terms': {
            'key': 'graph_boolean_scheduling_terms',
            'headline': "Graph's boolean scheduling terms",
            'description': 'Boolean scheduling terms that will be disabled when benchmark ends',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 1,
            'shape': [-1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::BooleanSchedulingTerm',
            'default': 'N/A'
        },
        'dependent_components': {
            'key': 'dependent_components',
            'headline': 'Dependent components',
            'description':
            'A list of components whose states must be ready before starting to benchmark',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 1,
            'shape': [-1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::Component',
            'default': 'N/A'
        },
        'data_replay_control_transmitter': {
            'key': 'data_replay_control_transmitter',
            'headline': 'Transmitter of data replay control',
            'description': 'Transmitter to send replay command to the connected data replayer',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL',
            'handle_type': 'nvidia::gxf::Transmitter',
            'default': 'N/A'
        },
        'benchmark_publishers': {
            'key': 'benchmark_publishers',
            'headline': 'Benchmark publishers',
            'description':
            'A list of benchmark publishers that buffer and publish benchmark messages',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 1,
            'shape': [-1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::benchmark::BenchmarkPublisher',
            'default': 'N/A'
        },
        'benchmark_sinks': {
            'key': 'benchmark_sinks',
            'headline': 'Benchmark sinks',
            'description': 'A list of benchmark sinks that record message arrival timestamps',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 1,
            'shape': [-1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::benchmark::BenchmarkSink',
            'default': 'N/A'
        },
        'resource_profilers': {
            'key': 'resource_profilers',
            'headline': 'Associated resource profilers',
            'description':
            'A list of associated resource profilers to generate resource profiling reports',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 1,
            'shape': [-1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::benchmark::ResourceProfilerBase',
            'default': 'N/A'
        },
        'trial_run': {
            'key': 'trial_run',
            'headline': 'Trial run switch',
            'description': 'Enable a trial run when set to true',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': True
        },
        'kill_at_the_end': {
            'key': 'kill_at_the_end',
            'headline': 'Kill the benchmark process at the end',
            'description': 'Kill this process when the benchmark ends',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': False
        },
        'title': {
            'key': 'title',
            'headline': 'Benchmark title',
            'description': 'Benchmark title to generate benchmark reports',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': 'Undefined Benchmark Title'
        },
        'exported_report': {
            'key': 'exported_report',
            'headline': 'Exported report',
            'description': 'File to store exported report',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::File',
            'default': 'N/A'
        },
        'entity_buffer_size': {
            'key': 'entity_buffer_size',
            'headline': 'Entity buffer size',
            'description': 'The number of messages to be buffered in each entity buffer',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': 5
        },
        'benchmark_duration_ms': {
            'key': 'benchmark_duration_ms',
            'headline': 'Benchmark duration',
            'description': 'The duration of each benchmark iteration in miniseconds',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': 5000
        },
        'benchmark_iterations': {
            'key': 'benchmark_iterations',
            'headline': 'The number of benchmark iterations',
            'description':
            'The number of benchmark iterations to be conducted for each benchmark test case',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': 5
        },
        'benchmark_buffering_timeout_s': {
            'key': 'benchmark_buffering_timeout_s',
            'headline': 'Benchmark buffering timeout',
            'description':
            'The max wait time in seconds before stopping the benchmark buffering stage',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': 5
        },
        'post_trial_benchmark_iteration_delay_s': {
            'key': 'post_trial_benchmark_iteration_delay_s',
            'headline': 'Post trial benchmark iteration delay',
            'description':
            'The wait time in seconds after a trial benchmark iteration before summarizing results',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': 2
        },
        'post_benchmark_iteration_delay_s': {
            'key': 'post_benchmark_iteration_delay_s',
            'headline': 'Post benchmark iteration delay',
            'description':
            'The wait time in seconds after each benchmark iteration before summarizing results',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': 2
        }
    }

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class BenchmarkPublisher(Component):
    '''Publishes buffered benchmark messages
    '''
    gxf_native_type: str = "nvidia::gxf::benchmark::BenchmarkPublisher"

    _validation_info_parameters = {
        'transmitter': {
            'key': 'transmitter',
            'headline': 'Benchmark publisher transmitter',
            'description': 'Transmitter to publish benchmark messages',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::Transmitter',
            'default': 'N/A'
        },
        'entity_buffer': {
            'key': 'entity_buffer',
            'headline': 'Benchmark message entity buffer',
            'description': 'Component that holds buffered benchmark message entities',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::benchmark::EntityBuffer',
            'default': 'N/A'
        },
        'benchmark_publisher_async_scheduling_term': {
            'key': 'benchmark_publisher_async_scheduling_term',
            'headline': 'Benchmark publisher execution control scheduling term',
            'description':
            'A boolean scheduling term to control execution of the benchmark publisher',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::AsynchronousSchedulingTerm',
            'default': 'N/A'
        }
    }

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class BenchmarkSink(Component):
    '''Records message arrival timestamps
    '''
    gxf_native_type: str = "nvidia::gxf::benchmark::BenchmarkSink"

    _validation_info_parameters = {
        'receiver': {
            'key': 'receiver',
            'headline': 'Message sink receiver',
            'description': 'A receiver for retrieving incoming messages',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::Receiver',
            'default': 'N/A'
        },
        'benchmark_publisher': {
            'key': 'benchmark_publisher',
            'headline': 'A data source benchmark publisher',
            'description': 'A benchmark publisher for retrieving published timestamps',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL',
            'handle_type': 'nvidia::gxf::benchmark::BenchmarkPublisher',
            'default': 'N/A'
        },
        'performance_calculators': {
            'key': 'performance_calculators',
            'headline': 'Associated performance calculators',
            'description':
            'A list of associated performance calculators for the incoming message flow',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 1,
            'shape': [-1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::benchmark::PerformanceCalculatorBase',
            'default': 'N/A'
        },
        'use_received_acqtime_as_published': {
            'key': 'use_received_acqtime_as_published',
            'headline': 'Use acqtime from incoming messages',
            'description': 'Use acqtime from incoming messages as published message timestamps',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': False
        }
    }

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class EntityBuffer(Component):
    '''Buffers incoming message entities
    '''
    gxf_native_type: str = "nvidia::gxf::benchmark::EntityBuffer"

    _validation_info_parameters = {
        'receiver': {
            'key': 'receiver',
            'headline': 'Entity buffer receiver',
            'description': 'Receiver to get message entities for buffering',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::Receiver',
            'default': 'N/A'
        }
    }

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class PerformanceCalculatorBase(Component):
    '''Base class of performance calculators
    '''
    gxf_native_type: str = "nvidia::gxf::benchmark::PerformanceCalculatorBase"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class ResourceProfiler(Component):
    '''Profiles resource utilizations
    '''
    gxf_native_type: str = "nvidia::gxf::benchmark::ResourceProfiler"

    _validation_info_parameters = {
        'namespace': {
            'key': 'namespace',
            'headline': 'Performance calculator namespace',
            'description': 'Namespace used to return performance results',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': ''
        },
        'profile_sampling_rate_hz': {
            'key': 'profile_sampling_rate_hz',
            'headline': 'Resource profile sampling rate',
            'description': 'The sampling rate for profiling resource usage in Hz',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_FLOAT64',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': 6.0
        }
    }

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class ResourceProfilerBase(Component):
    '''Base class of resource profilers
    '''
    gxf_native_type: str = "nvidia::gxf::benchmark::ResourceProfilerBase"

    _validation_info_parameters = {
        'namespace': {
            'key': 'namespace',
            'headline': 'Performance calculator namespace',
            'description': 'Namespace used to return performance results',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': ''
        },
        'profile_sampling_rate_hz': {
            'key': 'profile_sampling_rate_hz',
            'headline': 'Resource profile sampling rate',
            'description': 'The sampling rate for profiling resource usage in Hz',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_FLOAT64',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': 6.0
        }
    }

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)

class DummyMessageGenerator(Component):
    '''Generates dummy messages.
    '''
    gxf_native_type: str = "nvidia::gxf::benchmark::DummyMessageGenerator"

    _validation_info_parameters = {
        'dummy_component_size': {
            'key': 'dummy_component_size',
            'headline': 'Dummy Component Count',
            'description': 'Number of dummy components to generate.',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': 5
        },
        'publish_new_message': {
            'key': 'publish_new_message',
            'headline': 'Publish New Dummy Messages',
            'description':
            'If true, create new dummy messages to publish otherwise forward messages from receiver.',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': True
        },
        'receiver': {
            'key': 'receiver',
            'headline': 'Receiver',
            'description': 'Handle to the receiver for receiving messages.',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL',
            'handle_type': 'nvidia::gxf::Receiver',
            'default': 'N/A'
        },
        'transmitter': {
            'key': 'transmitter',
            'headline': 'Transmitter',
            'description': 'Handle to the transmitter for sending messages.',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::Transmitter',
            'default': 'N/A'
        }
    }

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)
