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

add_to_manifest("gxf/serialization/libgxf_serialization.so")

class ComponentSerializer(Component):
    '''Interface for serializing components
    '''
    gxf_native_type: str = "nvidia::gxf::ComponentSerializer"

    _validation_info_parameters = {}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class Endpoint(Component):
    '''Interface for exchanging data external to an application graph
    '''
    gxf_native_type: str = "nvidia::gxf::Endpoint"

    _validation_info_parameters = {}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class EntityRecorder(Component):
    '''Serializes incoming messages and writes them to a file
    '''
    gxf_native_type: str = "nvidia::gxf::EntityRecorder"

    _validation_info_parameters = {'receiver': {'key': 'receiver', 'headline': 'Entity receiver', 'description': 'Receiver channel to log', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'entity_serializer': {'key': 'entity_serializer', 'headline': 'Entity serializer', 'description': 'Serializer for serializing entities', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::EntitySerializer', 'default': 'N/A'}, 'directory': {'key': 'directory', 'headline': 'Directory path', 'description': 'Directory path for storing files', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'basename': {'key': 'basename', 'headline': 'Base file name', 'description': 'User specified file name without extension', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'flush_on_tick': {'key': 'flush_on_tick', 'headline': 'Flush on tick', 'description': 'Flushes output buffer on every tick when true', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': False}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class EntityReplayer(Component):
    '''Deserializes and publishes messages from a file
    '''
    gxf_native_type: str = "nvidia::gxf::EntityReplayer"

    _validation_info_parameters = {'transmitter': {'key': 'transmitter', 'headline': 'Entity transmitter', 'description': 'Transmitter channel for replaying entities', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}, 'entity_serializer': {'key': 'entity_serializer', 'headline': 'Entity serializer', 'description': 'Serializer for serializing entities', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::EntitySerializer', 'default': 'N/A'}, 'boolean_scheduling_term': {'key': 'boolean_scheduling_term', 'headline': 'BooleanSchedulingTerm', 'description': 'BooleanSchedulingTerm to stop the codelet from ticking after all messages are published.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::BooleanSchedulingTerm', 'default': 'N/A'}, 'directory': {'key': 'directory', 'headline': 'Directory path', 'description': 'Directory path for storing files', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'basename': {'key': 'basename', 'headline': 'Base file name', 'description': 'User specified file name without extension', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}, 'batch_size': {'key': 'batch_size', 'headline': 'Batch Size', 'description': 'Number of entities to read and publish for one tick', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 1}, 'ignore_corrupted_entities': {'key': 'ignore_corrupted_entities', 'headline': 'Ignore Corrupted Entities', 'description': 'If an entity could not be deserialized, it is ignored by default; otherwise a failure is generated.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': True}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class EntitySerializer(Component):
    '''Interface for serializing entities
    '''
    gxf_native_type: str = "nvidia::gxf::EntitySerializer"

    _validation_info_parameters = {}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class File(Component):
    '''Wrapper around C file I/O API
    '''
    gxf_native_type: str = "nvidia::gxf::File"

    _validation_info_parameters = {'allocator': {'key': 'allocator', 'headline': 'Allocator', 'description': 'Memory allocator for stream buffer', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Allocator', 'default': 'N/A'}, 'file_path': {'key': 'file_path', 'headline': 'File Path', 'description': 'Path to file', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': ''}, 'file_mode': {'key': 'file_mode', 'headline': 'File Mode', 'description': 'Access mode for file ("wb+" by default)  "r(b)" Opens a (binary) file for reading  "r(b)+" Opens a (binary) file to update both reading and writing  "w(b)" Creates an empty (binary) file for writing  "w(b)+" Creates an empty (binary) file for both reading and writing  "a(b)" Appends to a (binary) file  "a(b)+" Opens a (binary) file for reading and appending', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'wb+'}, 'buffer_size': {'key': 'buffer_size', 'headline': 'Buffer Size', 'description': 'Size of the stream buffer in bytes (2MB by default)', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 2097152}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class SerializationBuffer(Component):
    '''Buffer to hold serialized data
    '''
    gxf_native_type: str = "nvidia::gxf::SerializationBuffer"

    _validation_info_parameters = {'allocator': {'key': 'allocator', 'headline': 'Allocator', 'description': 'Memory allocator', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Allocator', 'default': 'N/A'}, 'buffer_size': {'key': 'buffer_size', 'headline': 'Buffer Size', 'description': 'Initial size of the buffer in bytes (4kB by default)', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 4096}, 'storage_type': {'key': 'storage_type', 'headline': 'Storage type', 'description': 'The initial memory storage type used by this buffer (kSystem by default)', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 2}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class StdComponentSerializer(Component):
    '''Serializer for Timestamp and Tensor components
    '''
    gxf_native_type: str = "nvidia::gxf::StdComponentSerializer"

    _validation_info_parameters = {'allocator': {'key': 'allocator', 'headline': 'Memory allocator', 'description': 'Memory allocator for tensor components', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Allocator', 'default': 'N/A'}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class StdEntityIdSerializer(Component):
    '''Serializes entity ID for sharing between GXF applications
    '''
    gxf_native_type: str = "nvidia::gxf::StdEntityIdSerializer"

    _validation_info_parameters = {}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class StdEntitySerializer(Component):
    '''Serializes entities for sharing data between GXF applications
    '''
    gxf_native_type: str = "nvidia::gxf::StdEntitySerializer"

    _validation_info_parameters = {'component_serializers': {'key': 'component_serializers', 'headline': 'Component serializers', 'description': 'List of serializers for serializing and deserializing components', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_CUSTOM', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'verbose_warning': {'key': 'verbose_warning', 'headline': 'Verbose Warning', 'description': 'Whether or to print verbose warning', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': True}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)

