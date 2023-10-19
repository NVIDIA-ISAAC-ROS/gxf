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

from .cuda_pybind import CudaStreamPool as CudaStreamPool_pybind

add_to_manifest("gxf/cuda/libgxf_cuda.so")


class CudaStreamPool(Component, CudaStreamPool_pybind):
    '''A Cuda stream pool provides stream allocation
    '''
    gxf_native_type: str = "nvidia::gxf::CudaStreamPool"

    _validation_info_parameters = {'dev_id': {'key': 'dev_id', 'headline': 'Device Id', 'description': 'Create CUDA Stream on which device.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}, 'stream_flags': {'key': 'stream_flags', 'headline': 'Stream Flags', 'description': 'Create CUDA streams with flags.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}, 'stream_priority': {'key': 'stream_priority', 'headline': 'Stream Priority', 'description': 'Create CUDA streams with priority.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [
        1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}, 'reserved_size': {'key': 'reserved_size', 'headline': 'Reserved Stream Size', 'description': 'Reserve serveral CUDA streams before 1st request coming', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 1}, 'max_size': {'key': 'max_size', 'headline': 'Maximum Stream Size', 'description': 'The maximum stream size for the pool to allocate, unlimited by default', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_UINT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 0}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class CudaStreamSync(Component):
    '''Synchronize all cuda streams which are carried by message entities
    '''
    gxf_native_type: str = "nvidia::gxf::CudaStreamSync"

    _validation_info_parameters = {'rx': {'key': 'rx', 'headline': 'Receiver', 'description': 'N/A', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'nvidia::gxf::Receiver', 'default': 'N/A'}, 'tx': {
        'key': 'tx', 'headline': 'Transmitter', 'description': '', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_HANDLE', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'nvidia::gxf::Transmitter', 'default': 'N/A'}}

    def __init__(self, name: str = "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)
