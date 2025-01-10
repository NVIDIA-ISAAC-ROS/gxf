'''
 SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_to_manifest("gxf/ucx/libgxf_ucx.so")

class UcxComponentSerializer(Component):
    '''Component Serializer for UCX.
    '''
    gxf_native_type: str = "nvidia::gxf::UcxComponentSerializer"

    _validation_info_parameters = {'allocator': {'key': 'allocator', 'headline': 'Memory allocator', 'description': 'Memory allocator for tensor components', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Allocator', 'default': 'N/A'}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class UcxContext(Component):
    '''Component to hold Ucx Context.
    '''
    gxf_native_type: str = "nvidia::gxf::UcxContext"

    _validation_info_parameters = {
        'serializer': {
            'key': 'serializer',
            'headline': 'Entity Serializer',
            'description': '', 'gxf_parameter_type':
            'GXF_PARAMETER_TYPE_HANDLE',
            'rank': 0, 'shape': [1],
            'flags':
            'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'nvidia::gxf::EntitySerializer',
            'default': 'N/A',
        },
        'reconnect': {
            'key': 'reconnect',
            'headline': 'Reconnect',
            'description': 'Try to reconnect if a connection is closed during run',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': True,
        },
        'cpu_data_only': {
            'key': 'cpu_data_only',
            'headline': 'CPU Data Only',
            'description': 'If True, the UCX context will only support transmission of CPU (host) data.',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': False,
        },
        'enable_async': {
            'key': 'enable_async',
            'headline': 'enable asynchronous transmit/receive',
            'description': 'If True, UCX transmit and receive will queue messages to be sent asynchronously.',
            'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL',
            'rank': 0,
            'shape': [1],
            'flags': 'GXF_PARAMETER_FLAGS_NONE',
            'handle_type': 'N/A',
            'default': True,
        }
    }

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class UcxEntitySerializer(Component):
    '''Entity Serializer for UCX.
    '''
    gxf_native_type: str = "nvidia::gxf::UcxEntitySerializer"

    _validation_info_parameters = {'component_serializers': {'key': 'component_serializers', 'headline': 'Component serializers', 'description': 'List of serializers for serializing and deserializing components', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_CUSTOM', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'verbose_warning': {'key': 'verbose_warning', 'headline': 'Verbose Warning', 'description': 'Whether or to print verbose warning', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': True}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class UcxReceiver(Component):
    '''Component to receive UCX message.
    '''
    gxf_native_type: str = "nvidia::gxf::UcxReceiver"

    _validation_info_parameters = {'capacity': {'key': 'capacity', 'headline': 'Capacity', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 10}, 'policy': {'key': 'policy', 'headline': 'Policy', 'description': '0: pop, 1: reject, 2: fault', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 2}, 'address': {'key': 'address', 'headline': 'Listener Address', 'description': 'Address to listen on', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': '0.0.0.0'}, 'port': {'key': 'port', 'headline': 'rx_port', 'description': 'RX port', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 13337}, 'buffer': {'key': 'buffer', 'headline': 'Serialization Buffer', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::UcxSerializationBuffer', 'default': 'N/A'}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class UcxSerializationBuffer(Component):
    '''Serializer Buffer for UCX.
    '''
    gxf_native_type: str = "nvidia::gxf::UcxSerializationBuffer"

    _validation_info_parameters = {'allocator': {'key': 'allocator', 'headline': 'Allocator', 'description': 'Memory allocator', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Allocator', 'default': 'N/A'}, 'buffer_size': {'key': 'buffer_size', 'headline': 'Buffer Size', 'description': 'Size of the buffer in bytes (4kB by default)', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 4096}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class UcxTransmitter(Component):
    '''Component to send UCX message.
    '''
    gxf_native_type: str = "nvidia::gxf::UcxTransmitter"

    _validation_info_parameters = {'capacity': {'key': 'capacity', 'headline': 'Capacity', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 1}, 'policy': {'key': 'policy', 'headline': 'Policy', 'description': '0: pop, 1: reject, 2: fault', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 2}, 'receiver_address': {'key': 'receiver_address', 'headline': 'Receiver address', 'description': 'Address to connect to', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': '0.0.0.0'}, 'port': {'key': 'port', 'headline': 'Receiver Port', 'description': 'Receiver Port', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 13337}, 'buffer': {'key': 'buffer', 'headline': 'Serialization Buffer', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::UcxSerializationBuffer', 'default': 'N/A'}, 'maximum_connection_retries': {'key': 'maximum_connection_retries', 'headline': 'Maximum Connection Retries', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 200}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)

