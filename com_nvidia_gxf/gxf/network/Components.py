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


from gxf.core import Component
from gxf.core import add_to_manifest

add_to_manifest("gxf/network/libgxf_network.so")

class ClockSyncPrimary(Component):
    '''Publishes application clock timestamp for use by other apps
    '''
    gxf_native_type: str = "nvidia::gxf::ClockSyncPrimary"

    _validation_info_parameters = {'tx_timestamp': {'key': 'tx_timestamp', 'headline': 'Outgoing timestamp', 'description': 'The outgoing timestamp channel', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}, 'clock': {'key': 'clock', 'headline': 'Application clock', 'description': "Handle to application's clock component", 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Clock', 'default': 'N/A'}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class ClockSyncSecondary(Component):
    '''Advances application SyntheticClock to received timestamp
    '''
    gxf_native_type: str = "nvidia::gxf::ClockSyncSecondary"

    _validation_info_parameters = {'rx_timestamp': {'key': 'rx_timestamp', 'headline': 'Incoming timestamp', 'description': 'The incoming timestamp channel', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'synthetic_clock': {'key': 'synthetic_clock', 'headline': "Application's synthetic clock", 'description': "Handle to application's synthetic clock component", 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::SyntheticClock', 'default': 'N/A'}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class TcpClient(Component):
    '''Codelet that functions as a client in a TCP connection
    '''
    gxf_native_type: str = "nvidia::gxf::TcpClient"

    _validation_info_parameters = {'receivers': {'key': 'receivers', 'headline': 'Entity receivers', 'description': 'List of receivers to receive entities from', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 1, 'shape': [-1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'transmitters': {'key': 'transmitters', 'headline': 'Entity transmitters', 'description': 'List of transmitters to publish entities to', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 1, 'shape': [-1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}, 'entity_serializer': {'key': 'entity_serializer', 'headline': 'Entity serializer', 'description': 'Serializer for serializing entities', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::EntitySerializer', 'default': 'N/A'}, 'address': {'key': 'address', 'headline': 'Address', 'description': 'Address for TCP connection', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'port': {'key': 'port', 'headline': 'Port', 'description': 'Port for TCP connection', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'timeout_ms': {'key': 'timeout_ms', 'headline': 'Connection timeout', 'description': 'Time in milliseconds to wait before retrying connection. Deprecated - use timeout_period instead.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'timeout_period': {'key': 'timeout_period', 'headline': 'Connection timeout', 'description': 'Time to wait before retrying connection. The period is specified as a string containing a number and an (optional) unit. If no unit is given the value is assumed to be in nanoseconds. Supported units are: Hz, s, ms. Example: 10ms, 10000000, 0.2s, 50Hz', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': '100ms'}, 'maximum_attempts': {'key': 'maximum_attempts', 'headline': 'Maximum attempts', 'description': 'Maximum number of attempts for I/O operations before failing', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 10}, 'async_scheduling_term': {'key': 'async_scheduling_term', 'headline': 'Asynchronous Scheduling Term', 'description': 'Schedules execution when TCP socket or receivers have a message.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::AsynchronousSchedulingTerm', 'default': 'N/A'}, 'max_msg_delay_ms': {'key': 'max_msg_delay_ms', 'headline': 'Max message delay [ms]', 'description': 'Time in milliseconds to wait between messages before ending connection. Helpful for debugging.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'max_duration_ms': {'key': 'max_duration_ms', 'headline': 'Max duration [ms]', 'description': 'The maximum duration for which the component will run (in ms). If not specified the component will run indefinitely, unless another termination condition is specified. Helpful for debugging.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'max_connection_attempts': {'key': 'max_connection_attempts', 'headline': 'Max connection attempts', 'description': 'The maximum number of times the component will attempt to reconnect. If not specified the component will attempt reconnection indefinitely, unless another termination condition is specified. Helpful for debugging.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class TcpCodelet(Component):
    '''Interface for a codelet for either end of a TCP connection
    '''
    gxf_native_type: str = "nvidia::gxf::TcpCodelet"

    _validation_info_parameters = {}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class TcpServer(Component):
    '''Codelet that functions as a server in a TCP connection
    '''
    gxf_native_type: str = "nvidia::gxf::TcpServer"

    _validation_info_parameters = {'receivers': {'key': 'receivers', 'headline': 'Entity receivers', 'description': 'List of receivers to receive entities from', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 1, 'shape': [-1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'transmitters': {'key': 'transmitters', 'headline': 'Entity transmitters', 'description': 'List of transmitters to publish entities to', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 1, 'shape': [-1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}, 'entity_serializer': {'key': 'entity_serializer', 'headline': 'Entity serializer', 'description': 'Serializer for serializing entities', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::EntitySerializer', 'default': 'N/A'}, 'address': {'key': 'address', 'headline': 'Address', 'description': 'Address for TCP connection', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'port': {'key': 'port', 'headline': 'Port', 'description': 'Port for TCP connection', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'timeout_ms': {'key': 'timeout_ms', 'headline': 'Connection timeout', 'description': 'Time in milliseconds to wait before retrying connection. Deprecated - use timeout_period instead.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'timeout_period': {'key': 'timeout_period', 'headline': 'Connection timeout', 'description': 'Time to wait before retrying connection. The period is specified as a string containing a number and an (optional) unit. If no unit is given the value is assumed to be in nanoseconds. Supported units are: Hz, s, ms. Example: 10ms, 10000000, 0.2s, 50Hz', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': '100ms'}, 'maximum_attempts': {'key': 'maximum_attempts', 'headline': 'Maximum attempts', 'description': 'Maximum number of attempts for I/O operations before failing', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 10}, 'async_scheduling_term': {'key': 'async_scheduling_term', 'headline': 'Asynchronous Scheduling Term', 'description': 'Schedules execution when TCP socket or receivers have a message.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::AsynchronousSchedulingTerm', 'default': 'N/A'}, 'max_msg_delay_ms': {'key': 'max_msg_delay_ms', 'headline': 'Max message delay [ms]', 'description': 'Time in milliseconds to wait between messages before ending connection. Helpful for debugging.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'max_duration_ms': {'key': 'max_duration_ms', 'headline': 'Max duration [ms]', 'description': 'The maximum duration for which the component will run (in ms). If not specified the component will run indefinitely, unless another termination condition is specified. Helpful for debugging.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'max_connection_attempts': {'key': 'max_connection_attempts', 'headline': 'Max connection attempts', 'description': 'The maximum number of times the component will attempt to reconnect. If not specified the component will attempt reconnection indefinitely, unless another termination condition is specified. Helpful for debugging.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)

