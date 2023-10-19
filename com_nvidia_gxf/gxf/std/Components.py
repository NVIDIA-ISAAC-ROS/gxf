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

from .clock_pybind import Clock as Clock_pybind    # pylint: disable=no-name-in-module
from .clock_pybind import RealtimeClock as RealtimeClock_pybind    # pylint: disable=no-name-in-module
from .clock_pybind import ManualClock as ManualClock_pybind    # pylint: disable=no-name-in-module
from .receiver_pybind import Receiver as Receiver_pybind    # pylint: disable=no-name-in-module
from .receiver_pybind import DoubleBufferReceiver as DoubleBufferReceiver_pybind    # pylint: disable=no-name-in-module
from .transmitter_pybind import Transmitter as Transmitter_pybind    # pylint: disable=no-name-in-module
from .transmitter_pybind import DoubleBufferTransmitter as DoubleBufferTransmitter_pybind    # pylint: disable=no-name-in-module
from .tensor_pybind import Tensor as Tensor_pybind     # pylint: disable=no-name-in-module
from .scheduling_terms_pybind import BooleanSchedulingTerm as BooleanSchedulingTerm_pybind     # pylint: disable=no-name-in-module
from .scheduling_terms_pybind import CountSchedulingTerm as CountSchedulingTerm_pybind     # pylint: disable=no-name-in-module
from .scheduling_terms_pybind import PeriodicSchedulingTerm as PeriodicSchedulingTerm_pybind     # pylint: disable=no-name-in-module
from .scheduling_terms_pybind import BTSchedulingTerm as BTSchedulingTerm_pybind     # pylint: disable=no-name-in-module
from .scheduling_terms_pybind import TargetTimeSchedulingTerm as TargetTimeSchedulingTerm_pybind     # pylint: disable=no-name-in-module
from .allocator_pybind import Allocator as Allocator_pybind     # pylint: disable=no-name-in-module

from gxf.core import Component
from gxf.core import parameter_set_path
from gxf.core import parameter_set_from_yaml_node
from gxf.core import add_to_manifest

import yaml

add_to_manifest("gxf/std/libgxf_std.so")


class Allocator(Allocator_pybind):
    '''Provides allocation and deallocation of memory
    '''
    gxf_native_type: str = "nvidia::gxf::Allocator"


class AsynchronousSchedulingTerm(Component):
    '''A component which is used to inform of that an entity is dependent upon an async event for its execution
    '''
    gxf_native_type: str = "nvidia::gxf::AsynchronousSchedulingTerm"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class BTSchedulingTerm(Component, BTSchedulingTerm_pybind):
    # class variables
    gxf_native_type: str = "nvidia::gxf::BTSchedulingTerm"

    _validation_info_parameters = {'is_root': {'key': 'is_root', 'headline': 'is_root', 'description': 'N/A', 'gxf_parameter_type':
                                               'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        # TODO: explicitly check the parameters
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class BlockMemoryPool(Component):
    '''A memory pools which provides a maximum number of equally sized blocks of memory
    '''
    gxf_native_type: str = "nvidia::gxf::BlockMemoryPool"

    _validation_info_parameters = {'storage_type': {'key': 'storage_type', 'headline': 'Storage type', 'description': 'The memory storage type used by this allocator. Can be kHost (0), kDevice (1) or kSystem (2)', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}, 'block_size': {'key': 'block_size', 'headline': 'Block size', 'description': 'The size of one block of memory in byte. Allocation requests can only be fullfilled if they fit into one block. If less memory is requested still a full block is issued.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'num_blocks': {'key': 'num_blocks', 'headline': 'Number of blocks', 'description': 'The total number of blocks which are allocated by the pool. If more blocks are requested allocation requests will fail.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class BooleanSchedulingTerm(Component, BooleanSchedulingTerm_pybind):
    # class variables
    gxf_native_type: str = "nvidia::gxf::BooleanSchedulingTerm"

    _validation_info_parameters = {'enable_tick': {'key': 'enable_tick', 'headline': 'Enable Tick', 'description': 'The default initial condition for enabling tick.',
                                                   'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_DYNAMIC', 'handle_type': 'N/A', 'default': True}}

    def __init__(self, name: str = "", **params):
        # TODO: explicitly check the parameters
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Broadcast(Component):
    '''Messages arrived on the input channel are distributed to all transmitters.
    '''
    gxf_native_type: str = "nvidia::gxf::Broadcast"

    _validation_info_parameters = {'source': {'key': 'source', 'headline': 'Source channel', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'mode': {
        'key': 'mode', 'headline': 'Broadcast Mode', 'description': 'The broadcast mode. Can be Broadcast or RoundRobin.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_CUSTOM', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class CPUThread(Component):
    '''A resource component used to pin jobs to a given thread.
    '''
    gxf_native_type: str = "nvidia::gxf::CPUThread"

    _validation_info_parameters = {'pin_entity': {'key': 'pin_entity', 'headline': 'Pin Entity', 'description': 'Set the cpu_core to be pinned to a worker thread or not.',
                                                  'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': False}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Clock(Clock_pybind):
    '''Interface for clock components which provide time
    '''
    gxf_native_type: str = "nvidia::gxf::Clock"


class Codelet(Component):
    '''Interface for a component which can be executed to run custom code
    '''
    gxf_native_type: str = "nvidia::gxf::Codelet"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Connection(Component):
    '''A component which establishes a connection between two other components
    '''
    gxf_native_type: str = "nvidia::gxf::Connection"

    _validation_info_parameters = {'source': {'key': 'source', 'headline': 'Source channel', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}, 'target': {
        'key': 'target', 'headline': 'Target channel', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Controller(Component):
    '''Controls entities' termination policy and tracks behavior status during execution.
    '''
    gxf_native_type: str = "nvidia::gxf::Controller"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class CountSchedulingTerm(Component, CountSchedulingTerm_pybind):
    # class variables
    gxf_native_type: str = "nvidia::gxf::CountSchedulingTerm"

    _validation_info_parameters = {'count': {'key': 'count', 'headline': 'Count', 'description': 'The total number of time this term will permit execution.',
                                             'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        # TODO: explicitly check the parameters
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class DoubleBufferReceiver(Component, DoubleBufferReceiver_pybind):
    # class variables
    gxf_native_type: str = "nvidia::gxf::DoubleBufferReceiver"

    _validation_info_parameters = {'capacity': {'key': 'capacity', 'headline': 'Capacity', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 1}, 'policy': {
        'key': 'policy', 'headline': 'Policy', 'description': '0: pop, 1: reject, 2: fault', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 2}}

    def __init__(self, name: str = "", **params):
        # TODO: explicitly check the parameters
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class DoubleBufferTransmitter(Component, DoubleBufferTransmitter_pybind):
    # class variables
    gxf_native_type: str = "nvidia::gxf::DoubleBufferTransmitter"

    _validation_info_parameters = {'capacity': {'key': 'capacity', 'headline': 'Capacity', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 1}, 'policy': {
        'key': 'policy', 'headline': 'Policy', 'description': '0: pop, 1: reject, 2: fault', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 2}}

    def __init__(self, name: str = "", **params):
        # TODO: explicitly check the parameters
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class DownstreamReceptiveSchedulingTerm(Component):
    # class variables
    gxf_native_type: str = "nvidia::gxf::DownstreamReceptiveSchedulingTerm"

    _validation_info_parameters = {'transmitter': {'key': 'transmitter', 'headline': 'Transmitter', 'description': 'The term permits execution if this transmitter can publish a message, i.e. if the receiver which is connected to this transmitter can receive messages.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}, 'min_size': {
        'key': 'min_size', 'headline': 'Minimum size', 'description': 'The term permits execution if the receiver connected to the transmitter has at least the specified number of free slots in its back buffer.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        # TODO: explicitly check the parameters
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class EpochScheduler(Component):
    '''A scheduler for running loads in externally managed threads
    '''
    gxf_native_type: str = "nvidia::gxf::EpochScheduler"

    _validation_info_parameters = {'clock': {'key': 'clock', 'headline': 'Clock', 'description': 'The clock used by the scheduler to check maximum time budget. Typical choice is a RealtimeClock.',
                                             'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Clock', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class ExpiringMessageAvailableSchedulingTerm(Component):
    '''A component which tries to wait for specified number of messages in queue for at most specified time.
    '''
    gxf_native_type: str = "nvidia::gxf::ExpiringMessageAvailableSchedulingTerm"

    _validation_info_parameters = {'max_batch_size': {'key': 'max_batch_size', 'headline': 'Maximum Batch Size', 'description': 'The maximum number of messages to be batched together. ', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'max_delay_ns': {'key': 'max_delay_ns', 'headline': 'Maximum delay in nano seconds.', 'description': 'The maximum delay from first message to wait before submitting workload anyway.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'receiver': {'key': 'receiver', 'headline': 'Receiver', 'description': 'Receiver to watch on.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'clock': {'key': 'clock', 'headline': 'Clock', 'description': 'Clock to get time from.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Clock', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Forward(Component):
    '''Forwards incoming messages at the receiver to the transmitter
    '''
    gxf_native_type: str = "nvidia::gxf::Forward"

    _validation_info_parameters = {'in': {'key': 'in', 'headline': 'input', 'description': 'The channel for incoming messages.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'out': {
        'key': 'out', 'headline': 'output', 'description': 'The channel for outgoing messages', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class GPUDevice(Component):
    '''A GPU Resource for codelet workloads
    '''
    gxf_native_type: str = "nvidia::gxf::GPUDevice"

    _validation_info_parameters = {'dev_id': {'key': 'dev_id', 'headline': 'Device Id', 'description': 'Create CUDA Stream on which device.',
                                              'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Gather(Component):
    '''All messages arriving on any input channel are published on the single output channel.
    '''
    gxf_native_type: str = "nvidia::gxf::Gather"

    _validation_info_parameters = {'sink': {'key': 'sink', 'headline': 'Sink', 'description': 'The output channel for gathered messages.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}, 'tick_source_limit': {
        'key': 'tick_source_limit', 'headline': 'Tick Source Limit', 'description': 'Maximum number of messages to take from each source in one tick. 0 means no limit.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class GreedyScheduler(Component):
    # class variables
    gxf_native_type: str = "nvidia::gxf::GreedyScheduler"

    _validation_info_parameters = {'clock': {'key': 'clock', 'headline': 'Clock', 'description': 'The clock used by the scheduler to define flow of time. Typical choices are a RealtimeClock or a ManualClock.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'nvidia::gxf::Clock', 'default': 'N/A'}, 'realtime': {'key': 'realtime', 'headline': 'Realtime (deprecated)', 'description': 'This parameter is deprecated. Assign a clock directly.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'max_duration_ms': {'key': 'max_duration_ms', 'headline': 'Max Duration [ms]', 'description': 'The maximum duration for which the scheduler will execute (in ms). If not specified the scheduler will run until all work is done. If periodic terms are present this means the application will run indefinitely.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'stop_on_deadlock': {'key': 'stop_on_deadlock', 'headline': 'Stop on dead end', 'description': 'If enabled the scheduler will stop when all entities are in a waiting state, but no periodic entity exists to break the dead end. Should be disabled when scheduling conditions can be changed by external actors, for example by clearing queues manually.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': True}}

    def __init__(self, name: str = "", **params):
        # TODO: explicitly check the parameters
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class IPCServer(Component):
    '''Interface for a component which works as a API server to respond on remote requests
    '''
    gxf_native_type: str = "nvidia::gxf::IPCServer"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class JobStatistics(Component):
    # class variables
    gxf_native_type: str = "nvidia::gxf::JobStatistics"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        # TODO: explicitly check the parameters
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class ManualClock(Component, ManualClock_pybind):
    gxf_native_type: str = "nvidia::gxf::ManualClock"

    _validation_info_parameters = {'initial_timestamp': {'key': 'initial_timestamp', 'headline': 'Initial Timestamp', 'description': 'The initial timestamp on the clock (in nanoseconds).', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class MemoryAvailableSchedulingTerm(Component):
    '''A component waiting until a minimum amount of memory is available
    '''
    gxf_native_type: str = "nvidia::gxf::MemoryAvailableSchedulingTerm"

    _validation_info_parameters = {'allocator': {'key': 'allocator', 'headline': 'Allocator', 'description': 'The allocator to wait on.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Allocator', 'default': 'N/A'}, 'min_bytes': {'key': 'min_bytes', 'headline': 'Minimum bytes available', 'description': 'The minimum number of bytes that must be available for the codelet to get scheduled. Exclusive with min_blocks.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'min_blocks': {'key': 'min_blocks', 'headline': 'Minimum blocks available', 'description': 'The minimum number of blocks that must be available for the codelet to get scheduled. On allocators that do not support block allocation, this behaves the same as min_bytes. Exclusive with min_bytes.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class MessageAvailableSchedulingTerm(Component):
    # class variables
    gxf_native_type: str = "nvidia::gxf::MessageAvailableSchedulingTerm"

    _validation_info_parameters = {'receiver': {'key': 'receiver', 'headline': 'Queue channel', 'description': 'The scheduling term permits execution if this channel has at least a given number of messages available.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'min_size': {'key': 'min_size', 'headline': 'Minimum message count', 'description': 'The scheduling term permits execution if the given receiver has at least the given number of messages available.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'front_stage_max_size': {'key': 'front_stage_max_size', 'headline': 'Maximum front stage message count', 'description': 'If set the scheduling term will only allow execution if the number of messages in the front stage does not exceed this count. It can for example be used in combination with codelets which do not clear the front stage in every tick.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        # TODO: explicitly check the parameters
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class MessageRouter(Component):
    '''A router which sends transmitted messages to receivers.
    '''
    gxf_native_type: str = "nvidia::gxf::MessageRouter"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Metric(Component):
    '''Collects, aggregates, and evaluates metric data.
    '''
    gxf_native_type: str = "nvidia::gxf::Metric"

    _validation_info_parameters = {'aggregation_policy': {'key': 'aggregation_policy', 'headline': 'Aggregation Policy', 'description': 'Aggregation policy used to aggregate individual metric samples. Choices:{mean, min, max}.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'lower_threshold': {
        'key': 'lower_threshold', 'headline': 'Lower threshold', 'description': "Lower threshold of the metric's expected range", 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_FLOAT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'upper_threshold': {'key': 'upper_threshold', 'headline': 'Upper threshold', 'description': "Upper threshold of the metric's expected range", 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_FLOAT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Monitor(Component):
    '''Monitors entities during execution.
    '''
    gxf_native_type: str = "nvidia::gxf::Monitor"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class MultiMessageAvailableSchedulingTerm(Component):
    '''A component which specifies that an entity shall be executed when a queue has at least a certain number of elements
    '''
    gxf_native_type: str = "nvidia::gxf::MultiMessageAvailableSchedulingTerm"

    _validation_info_parameters = {'receivers': {'key': 'receivers', 'headline': 'Receivers', 'description': 'The scheduling term permits execution if the given channels have at least a given number of messages available.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_CUSTOM', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'min_size': {'key': 'min_size', 'headline': 'Minimum message count', 'description': 'The scheduling term permits execution if all given receivers together have at least the given number of messages available', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 1}, 'sampling_mode': {'key': 'sampling_mode', 'headline': 'Sampling Mode', 'description': 'The sampling method to use when checking for messages in receiver queues. Option: SumOfAll,PerReceiver', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_CUSTOM', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'min_sizes': {'key': 'min_sizes', 'headline': 'Minimum message counts', 'description': 'The scheduling term permits execution if all given receivers have at least the given number of messages available in this list.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_CUSTOM', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'min_sum': {'key': 'min_sum', 'headline': 'Minimum sum of message counts', 'description': 'The scheduling term permits execution if the sum of message counts of all receivers have at least the given number of messages available.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class MultiThreadScheduler(Component):
    '''A multi thread scheduler that executes codelets for maximum throughput.
    '''
    gxf_native_type: str = "nvidia::gxf::MultiThreadScheduler"

    _validation_info_parameters = {'clock': {'key': 'clock', 'headline': 'Clock', 'description': 'The clock used by the scheduler to define flow of time. Typical choices are a RealtimeClock or a ManualClock.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Clock', 'default': 'N/A'}, 'max_duration_ms': {'key': 'max_duration_ms', 'headline': 'Max Duration [ms]', 'description': 'The maximum duration for which the scheduler will execute (in ms). If not specified the scheduler will run until all work is done. If periodic terms are present this means the application will run indefinitely.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'check_recession_period_ms': {'key': 'check_recession_period_ms', 'headline': 'Duration to sleep before checking the condition of an entity again [ms]', 'description': 'The maximum duration for which the scheduler would wait (in ms) when an entity is not ready to run yet.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_FLOAT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 5.0}, 'stop_on_deadlock': {'key': 'stop_on_deadlock', 'headline': 'Stop on dead end', 'description': 'If enabled the scheduler will stop when all entities are in a waiting state, but no periodic entity exists to break the dead end. Should be disabled when scheduling conditions can be changed by external actors, for example by clearing queues manually.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': True}, 'worker_thread_number': {'key': 'worker_thread_number', 'headline': 'Thread Number', 'description': 'Number of threads.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 1}, 'thread_pool_allocation_auto': {'key': 'thread_pool_allocation_auto', 'headline': 'Automatic Pool Allocation', 'description': 'If enabled, only one thread pool will be created. If disabled, user should enumerate pools and priorities', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': True}, 'strict_job_thread_pinning': {'key': 'strict_job_thread_pinning', 'headline': 'Strict Job-Thread Pinning', 'description': 'If enabled, for entity pinned thread, it cannot execute other entities. i.e. true entity-thread pinning.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': False}, 'stop_on_deadlock_timeout': {'key': 'stop_on_deadlock_timeout', 'headline': 'A refreshing version of max_duration_ms when stop_on_dealock kick-in [ms]', 'description': 'Scheduler will wait this amount of time when stop_on_dead_lock indicates should stop. It will reset if a job comes in during the wait. Negative value means not stop on deadlock.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class NetworkContext(Component):
    '''Interface for a component for network context like UCX
    '''
    gxf_native_type: str = "nvidia::gxf::NetworkContext"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class NetworkRouter(Component):
    '''A router which sends transmitted messages to remote receivers.
    '''
    gxf_native_type: str = "nvidia::gxf::NetworkRouter"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class PeriodicSchedulingTerm(Component, PeriodicSchedulingTerm_pybind):
    # class variables
    gxf_native_type: str = "nvidia::gxf::PeriodicSchedulingTerm"

    _validation_info_parameters = {'recess_period': {'key': 'recess_period', 'headline': 'Recess Period', 'description': 'The recess period indicates the minimum amount of time which has to pass before the entity is permitted to execute again. The period is specified as a string containing of a number and an (optional) unit. If no unit is given the value is assumed to be in nanoseconds. Supported units are: Hz, s, ms. Example: 10ms, 10000000, 0.2s, 50Hz', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        # TODO: explicitly check the parameters
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Queue(Component):
    '''Interface for storing entities in a queue
    '''
    gxf_native_type: str = "nvidia::gxf::Queue"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class RealtimeClock(Component, RealtimeClock_pybind):
    gxf_native_type: str = "nvidia::gxf::RealtimeClock"

    _validation_info_parameters = {'initial_time_offset': {'key': 'initial_time_offset', 'headline': 'Initial Time Offset', 'description': 'The initial time offset used until time scale is changed manually.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_FLOAT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0.0}, 'initial_time_scale': {'key': 'initial_time_scale', 'headline': 'Initial Time Scale', 'description': 'The initial time scale used until time scale is changed manually.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_FLOAT64', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 1.0}, 'use_time_since_epoch': {'key': 'use_time_since_epoch', 'headline': 'Use Time Since Epoch', 'description': 'If true, clock time is time since epoch + initial_time_offset at initialize().Otherwise clock time is initial_time_offset at initialize().', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': False}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Receiver(Receiver_pybind):
    # class variables
    gxf_native_type: str = "nvidia::gxf::Receiver"


class ResourceBase(Component):
    '''A Resource base type
    '''
    gxf_native_type: str = "nvidia::gxf::ResourceBase"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class RouterGroup(Component):
    '''A group of routers
    '''
    gxf_native_type: str = "nvidia::gxf::RouterGroup"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Scheduler(Component):
    '''A simple poll-based single-threaded scheduler which executes codelets
    '''
    gxf_native_type: str = "nvidia::gxf::Scheduler"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class SchedulingTerm(Component):
    '''Interface for terms used by a scheduler to determine if codelets in an entity are ready to step
    '''
    gxf_native_type: str = "nvidia::gxf::SchedulingTerm"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Subgraph(Component):
    '''Helper component to import a subgraph
    '''
    gxf_native_type: str = "nvidia::gxf::Subgraph"

    _validation_info_parameters = {'location': {'key': 'location', 'headline': 'Yaml source of the subgraph', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_FILE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'prerequisites': {
        'key': 'prerequisites', 'headline': 'list of prerequisite components of the subgraph', 'description': 'a prerequisite is a component required by the subgraph and must be satisfied before the graph is loaded', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_CUSTOM', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'},
        'override_params': {'key': 'override_params', 'headline': 'Override params of the subgraph', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_CUSTOM', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)

    def set_params(self):
        # This is needed to handle the override_params parameter in the Python
        # Subgraph Component Impl. The override params are not part of Subgraph
        # Component and hence the default set_params, if used, will throw an error
        if not self._params:
            return
        if 'location' in self._params.keys():
            parameter_set_path(self._entity.context, self._cid,
                               'location', self._params['location'])
        if 'prerequisites' in self._params.keys():
            parameter_set_from_yaml_node(
                self._entity.context, self._cid, 'prerequisites', yaml.dump(yaml.safe_load(str(self._params['prerequisites']))))


class Synchronization(Component):
    '''Component to synchronize messages from multiple receivers based on theacq_time
    '''
    gxf_native_type: str = "nvidia::gxf::Synchronization"

    _validation_info_parameters = {'inputs': {'key': 'inputs', 'headline': 'Inputs', 'description': 'All the inputs for synchronization, number of inputs must match that of the outputs.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 1, 'shape': [-1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'outputs': {'key': 'outputs', 'headline': 'Outputs', 'description': 'All the outputs for synchronization, number of outpus must match that of the inputs.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 1, 'shape': [-1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}, 'sync_threshold': {
        'key': 'sync_threshold', 'headline': 'Synchronization threshold (ns)', 'description': 'Synchronization threshold in nanoseconds. Messages will not be synchronized if timestamp difference is above the threshold. By default, timestamps should be identical for synchronization (default threshold = 0). Synchronization threshold will only work if maximum timestamp variation is much less than minimal delta between timestamps of subsequent messages in any input.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class SyntheticClock(Component):
    '''A synthetic clock used to inject simulated time
    '''
    gxf_native_type: str = "nvidia::gxf::SyntheticClock"

    _validation_info_parameters = {'initial_timestamp': {'key': 'initial_timestamp', 'headline': 'Initial Timestamp', 'description': 'The initial timestamp on the clock (in nanoseconds).', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class System(Component):
    '''Component interface for systems which are run as part of the application run cycle
    '''
    gxf_native_type: str = "nvidia::gxf::System"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class SystemGroup(Component):
    '''A group of systems
    '''
    gxf_native_type: str = "nvidia::gxf::SystemGroup"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class TargetTimeSchedulingTerm(Component, TargetTimeSchedulingTerm_pybind):
    # class variables
    gxf_native_type: str = "nvidia::gxf::TargetTimeSchedulingTerm"

    _validation_info_parameters = {'clock': {'key': 'clock', 'headline': 'Clock', 'description': 'The clock used to define target time.',
                                             'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Clock', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        # TODO: explicitly check the parameters
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Tensor(Tensor_pybind):
    # class variables
    gxf_native_type: str = "nvidia::gxf::Tensor"


class TensorCopier(Component):
    '''Copies tensor either from host to device or from device to host
    '''
    gxf_native_type: str = "nvidia::gxf::TensorCopier"

    _validation_info_parameters = {'receiver': {'key': 'receiver', 'headline': 'Receiver', 'description': 'Receiver for incoming entities', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'transmitter': {'key': 'transmitter', 'headline': 'Transmitter', 'description': 'Transmitter for outgoing entities ', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}, 'allocator': {
        'key': 'allocator', 'headline': 'Allocator', 'description': 'Memory allocator for tensor data', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Allocator', 'default': 'N/A'}, 'mode': {'key': 'mode', 'headline': 'Copy mode', 'description': 'Configuration to select what tensors to copy - kCopyToDevice (0) - copies to device memory, ignores device allocation; kCopyToHost (1) - copies to pinned host memory, ignores host allocation; kCopyToSystem (2) - copies to system memory, ignores system allocation', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class ThreadPool(Component):
    '''A threadpool component we can use to prioritize workloads
    '''
    gxf_native_type: str = "nvidia::gxf::ThreadPool"

    _validation_info_parameters = {'initial_size': {'key': 'initial_size', 'headline': 'Initial ThreadPool Size', 'description': 'Initial number of worker threads in the pool', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}, 'priority': {
        'key': 'priority', 'headline': 'Thread Priorities', 'description': 'Priority level for threads in the pool. Default is 0 (low)Can also be set to 1 (medium) or 2 (high)', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class TimedThrottler(Component):
    '''Publishes the received entity respecting the timestamp within the entity
    '''
    gxf_native_type: str = "nvidia::gxf::TimedThrottler"

    _validation_info_parameters = {'transmitter': {'key': 'transmitter', 'headline': 'Transmitter', 'description': 'Transmitter channel publishing messages at appropriate timesteps', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}, 'receiver': {'key': 'receiver', 'headline': 'Receiver', 'description': 'Channel to receive messages that need to be synchronized', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'execution_clock': {'key': 'execution_clock', 'headline': 'Execution Clock', 'description': 'Clock on which the codelet is executed by the scheduler', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Clock', 'default': 'N/A'}, 'throttling_clock': {'key': 'throttling_clock', 'headline': 'Throttling Clock', 'description': 'Clock on which the received entity timestamps are based', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Clock', 'default': 'N/A'}, 'scheduling_term': {'key': 'scheduling_term', 'headline': 'Scheduling Term', 'description': 'Scheduling term for executing the codelet', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::TargetTimeSchedulingTerm', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Transmitter(Transmitter_pybind):
    '''Interface for publishing entities
    '''
    gxf_native_type: str = "nvidia::gxf::Transmitter"


class UnboundedAllocator(Component):
    '''Allocator that uses dynamic memory allocation without an upper bound
    '''
    gxf_native_type: str = "nvidia::gxf::UnboundedAllocator"

    _validation_info_parameters = {}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


# class Vault(Component):
#     '''Safely stores received entities for further processing.
#     '''
#     gxf_native_type: str = "nvidia::gxf::Vault"

#     _validation_info_parameters = {'source': {'key': 'source', 'headline': 'Source', 'description': 'Receiver from which messages are taken and transferred to the vault.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'max_waiting_count': {'key': 'max_waiting_count', 'headline': 'Maximum waiting count', 'description': 'The maximum number of waiting messages. If exceeded the codelet will stop pulling messages out of the input queue.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [
#         1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'drop_waiting': {'key': 'drop_waiting', 'headline': 'Drop waiting', 'description': 'If too many messages are waiting the oldest ones are dropped.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}}

#     def __init__(self, name: str = "", **params):
#         Component.__init__(self, type=self.get_gxf_type(), name=name, **params)
