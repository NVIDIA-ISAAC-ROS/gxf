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

add_to_manifest("gxf/rmm/libgxf_rmm.so")


class RMMAllocator(Component):
    '''Allocator based on RMM Memory Pools
    '''
    gxf_native_type: str = "nvidia::gxf::RMMAllocator"

    _validation_info_parameters = {'device_memory_initial_size': {'key': 'device_memory_initial_size', 'headline': 'Storage type', 'description': 'The initial memory pool size used by this device', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 33554432}, 'host_memory_initial_size': {
        'key': 'host_memory_initial_size', 'headline': 'Storage type', 'description': 'The initial memory pool size used by this host', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT64', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 33554432}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)
