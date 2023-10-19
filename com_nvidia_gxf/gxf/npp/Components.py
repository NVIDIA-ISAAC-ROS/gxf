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

add_to_manifest("gxf/npp/libgxf_npp.so")

class NppiMulC(Component):
    '''Multiplies a tensor with a constant factor
    '''
    gxf_native_type: str = "nvidia::gxf::NppiMulC"

    _validation_info_parameters = {'in': {'key': 'in', 'headline': 'in', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'factor': {'key': 'factor', 'headline': 'factor', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_FLOAT64', 'rank': 1, 'shape': [-1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'pool': {'key': 'pool', 'headline': 'pool', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Allocator', 'default': 'N/A'}, 'out': {'key': 'out', 'headline': 'out', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class NppiSet(Component):
    '''Creates a tensor with constant values
    '''
    gxf_native_type: str = "nvidia::gxf::NppiSet"

    _validation_info_parameters = {'rows': {'key': 'rows', 'headline': 'rows', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'columns': {'key': 'columns', 'headline': 'columns', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'channels': {'key': 'channels', 'headline': 'channels', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'pool': {'key': 'pool', 'headline': 'pool', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Allocator', 'default': 'N/A'}, 'value': {'key': 'value', 'headline': 'value', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_FLOAT64', 'rank': 1, 'shape': [-1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'out': {'key': 'out', 'headline': 'out', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)

