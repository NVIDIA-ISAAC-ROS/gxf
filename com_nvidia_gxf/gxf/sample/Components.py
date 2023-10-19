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

add_to_manifest("gxf/sample/libgxf_sample.so")

class HelloWorld(Component):
    '''Prints a 'Hello world' string on execution
    '''
    gxf_native_type: str = "nvidia::gxf::HelloWorld"

    _validation_info_parameters = {}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class MultiPingRx(Component):
    '''Receives message entities from multiple receivers
    '''
    gxf_native_type: str = "nvidia::gxf::MultiPingRx"

    _validation_info_parameters = {'receivers': {'key': 'receivers', 'headline': 'Receivers', 'description': 'A list of receivers which the entity can pop message entities', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_CUSTOM', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class PingBatchRx(Component):
    '''Receives a message entity from a Receiver of specified batch size
    '''
    gxf_native_type: str = "nvidia::gxf::PingBatchRx"

    _validation_info_parameters = {'signal': {'key': 'signal', 'headline': 'signal', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'batch_size': {'key': 'batch_size', 'headline': 'batch_size', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'assert_full_batch': {'key': 'assert_full_batch', 'headline': 'Assert Full Batch', 'description': 'Assert if the batch is not fully populated.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': True}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class PingRx(Component):
    '''Receives a message entity from a Receiver
    '''
    gxf_native_type: str = "nvidia::gxf::PingRx"

    _validation_info_parameters = {'signal': {'key': 'signal', 'headline': 'Signal', 'description': 'Channel to receive messages from another graph entity', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class PingTx(Component):
    '''Sends an empty message entity via a Transmitter
    '''
    gxf_native_type: str = "nvidia::gxf::PingTx"

    _validation_info_parameters = {'signal': {'key': 'signal', 'headline': 'Signal', 'description': 'Transmitter channel publishing messages to other graph entities', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}, 'clock': {'key': 'clock', 'headline': 'Clock', 'description': 'Clock component needed for timestamping messages', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'nvidia::gxf::Clock', 'default': 'N/A'}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)

